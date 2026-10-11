from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from bson.binary import Binary
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from xgboost import Booster, DMatrix, XGBRegressor

from src.config import settings
from src.db import get_db
from src.ml.features import FEATURES, HORIZONS, build
from src.api_clients.yfinance_client import AGRICULTURAL_COMMODITIES

MAX_HORIZON = max(HORIZONS)
FORECAST_CALENDAR_DAYS = 180


def sentiment_df(db, asset_id):
    pipeline = [
        {"$match": {"mentioned_asset_ids": asset_id}},
        {"$project": {
            "day": {"$dateTrunc": {"date": "$published_at", "unit": "day", "timezone": "UTC"}},
            "sentiment_compound": 1,
        }},
        {"$group": {"_id": "$day", "sentiment_compound": {"$avg": "$sentiment_compound"}}},
        {"$sort": {"_id": 1}},
    ]
    rows = list(db.news.aggregate(pipeline))
    return pd.DataFrame([
        {"date": row["_id"], "sentiment_compound": row.get("sentiment_compound", 0.0)}
        for row in rows
    ])


def price_df(db, asset_id):
    rows = list(db.ohlcv.find(
        {"asset_id": asset_id, "period": "1d"},
        {"_id": 0, "timestamp": 1, "close": 1, "volume": 1},
    ).sort("timestamp", 1))
    if not rows:
        return pd.DataFrame(columns=["date", "close", "volume"]), "none"
    df = pd.DataFrame(rows).rename(columns={"timestamp": "date"})
    if "volume" not in df.columns:
        df["volume"] = 0.0
    return df, "yfinance_ohlcv"


def _new_model():
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=350,
        max_depth=3,
        learning_rate=0.025,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.05,
        reg_lambda=5.0,
        random_state=42,
        n_jobs=2,
    )


def _evaluate_horizon(data, horizon):
    """Backtest cronológico con separación (purge) igual al horizonte objetivo."""
    target_col = f"target_return_{horizon}d"
    usable = data.dropna(subset=FEATURES + [target_col, "close", "date"]).copy()
    usable = usable.sort_values("date").reset_index(drop=True)

    min_rows = int(getattr(settings, "model_min_rows", 30))
    if len(usable) < max(min_rows, 50):
        return {"status": "insufficient_history", "rows": len(usable)}

    split = int(len(usable) * 0.80)
    test_start = split
    train_end = max(0, split - horizon)
    train = usable.iloc[:train_end]
    test = usable.iloc[test_start:]

    if len(train) < 30 or len(test) < 10:
        return {
            "status": "insufficient_history_for_horizon",
            "rows": len(usable),
            "train_rows": len(train),
            "test_rows": len(test),
        }

    model = _new_model()
    target = target_col
    model.fit(train[FEATURES], train[target])
    pred_return = model.predict(test[FEATURES])
    actual_return = test[target].to_numpy(dtype=float)
    base_price = test["close"].to_numpy(dtype=float)
    actual_price = base_price * np.exp(actual_return)
    predicted_price = base_price * np.exp(pred_return)
    naive_price = base_price.copy()

    mae_return = float(mean_absolute_error(actual_return, pred_return))
    rmse_return = float(np.sqrt(mean_squared_error(actual_return, pred_return)))
    r2_return = float(r2_score(actual_return, pred_return)) if len(test) > 1 else float("nan")
    mae_price = float(mean_absolute_error(actual_price, predicted_price))
    naive_mae_price = float(mean_absolute_error(actual_price, naive_price))
    mape_price = float(mean_absolute_percentage_error(actual_price, predicted_price) * 100.0)

    final_model = _new_model()
    final_model.fit(usable[FEATURES], usable[target])
    raw = bytes(final_model.get_booster().save_raw(raw_format="json"))

    return {
        "status": "trained",
        "horizon": int(horizon),
        "rows": int(len(usable)),
        "training_rows": int(len(train)),
        "test_rows": int(len(test)),
        "mae_return": mae_return,
        "rmse_return": rmse_return,
        "r2_return": r2_return,
        "mae_price": mae_price,
        "naive_mae_price": naive_mae_price,
        "mape_price_pct": mape_price,
        "beats_naive": bool(mae_price < naive_mae_price),
        "feature_importance": {
            name: float(value)
            for name, value in zip(FEATURES, final_model.feature_importances_)
        },
        "model_raw_json": Binary(raw),
        "backtest": [
            {
                "date": pd.to_datetime(row_date, utc=True).to_pydatetime(),
                "actual_price": float(actual),
                "predicted_price": float(predicted),
                "baseline_price": float(baseline),
            }
            for row_date, actual, predicted, baseline in zip(
                test["date"], actual_price, predicted_price, naive_price
            )
        ],
    }

def train_one(asset_id):
    db = get_db()
    prices, source = price_df(db, asset_id)
    if prices.empty:
        return {"asset_id": asset_id, "status": "no_data"}

    features = build(prices, sentiment_df(db, asset_id), target=True)
    if features.empty:
        return {"asset_id": asset_id, "status": "insufficient_history"}

    horizon_models = {}
    metrics_by_horizon = {}
    max_horizon_result = None

    for horizon in HORIZONS:
        result = _evaluate_horizon(features, horizon)
        status = result.get("status")
        if status != "trained":
            metrics_by_horizon[str(horizon)] = {
                key: value for key, value in result.items() if key != "model_raw_json"
            }
            continue

        horizon_models[str(horizon)] = {
            "model_raw_json": result["model_raw_json"],
            "feature_columns": FEATURES,
        }
        metrics_by_horizon[str(horizon)] = {
            key: value for key, value in result.items()
            if key not in ("model_raw_json", "backtest", "feature_importance")
        }
        if max_horizon_result is None or horizon > max_horizon_result["horizon"]:
            max_horizon_result = result

    if not horizon_models:
        return {
            "asset_id": asset_id,
            "status": "insufficient_history_for_horizons",
            "metrics_by_horizon": metrics_by_horizon,
        }

    now = datetime.now(timezone.utc)
    model_doc = {
        "asset_id": asset_id,
        "model_type": "XGBRegressor_multi_horizon",
        "feature_columns": FEATURES,
        "horizons": [int(key) for key in horizon_models],
        "horizon_models": horizon_models,
        "trained_at": now,
        "history_source": source,
    }
    db.models.update_one({"asset_id": asset_id}, {"$set": model_doc}, upsert=True)

    metric_doc = {
        "asset_id": asset_id,
        "trained_at": now,
        "history_source": source,
        "metrics_by_horizon": metrics_by_horizon,
        "max_trained_horizon": max(int(key) for key in horizon_models),
        "feature_columns": FEATURES,
    }
    db.model_metrics.insert_one(metric_doc)

    longest = max_horizon_result
    print(f"   Horizontes entrenados: {', '.join(horizon_models.keys())} sesiones")
    if longest:
        print(f"   Horizonte más largo validado: {longest['horizon']} sesiones")
        print(f"   MAE precio: {longest['mae_price']:.4f}")
        print(f"   MAE baseline naive: {longest['naive_mae_price']:.4f}")
        print(f"   MAPE precio: {longest['mape_price_pct']:.2f}%")
        print(f"   R² retorno: {longest['r2_return']:.4f}")
        print(f"   ¿Supera al baseline?: {'SÍ' if longest['beats_naive'] else 'NO'}")

    return {
        "asset_id": asset_id,
        "status": "trained",
        "training_rows": longest["training_rows"] if longest else 0,
        "test_rows": longest["test_rows"] if longest else 0,
        "metrics_by_horizon": metrics_by_horizon,
        "max_trained_horizon": max(int(key) for key in horizon_models),
        "beats_naive": bool(longest["beats_naive"]) if longest else False,
    }

def predict_one(asset_id):
    db = get_db()
    model_doc = db.models.find_one({"asset_id": asset_id})
    if not model_doc:
        return {"asset_id": asset_id, "status": "no_model"}
    if not model_doc.get("horizon_models"):
        return {"asset_id": asset_id, "status": "retrain_required", "message": "El modelo guardado usa el formato anterior. Ejecuta el entrenamiento actualizado."}

    prices, source = price_df(db, asset_id)
    if prices.empty:
        return {"asset_id": asset_id, "status": "no_prices"}

    sentiment = sentiment_df(db, asset_id)
    features = build(prices, sentiment, target=False)
    valid = features.dropna(subset=FEATURES)
    if valid.empty:
        return {"asset_id": asset_id, "status": "insufficient_features"}

    last_row = valid.iloc[-1]
    feature_frame = pd.DataFrame([{name: float(last_row[name]) for name in FEATURES}], columns=FEATURES)
    current_price = float(last_row["close"])
    if not np.isfinite(current_price) or current_price <= 0:
        return {"asset_id": asset_id, "status": "invalid_current_price"}

    anchor_points = [(0.0, current_price)]
    used_horizons = []
    for horizon_text, payload in model_doc["horizon_models"].items():
        horizon = int(horizon_text)
        booster = Booster()
        booster.load_model(bytearray(payload["model_raw_json"]))
        matrix = DMatrix(feature_frame, feature_names=FEATURES)
        predicted_log_return = float(booster.predict(matrix)[0])
        if not np.isfinite(predicted_log_return):
            continue
        predicted_log_return = float(np.clip(predicted_log_return, -1.5, 1.5))
        predicted_price = current_price * float(np.exp(predicted_log_return))
        calendar_day = min(FORECAST_CALENDAR_DAYS, horizon * FORECAST_CALENDAR_DAYS / MAX_HORIZON)
        anchor_points.append((calendar_day, predicted_price))
        used_horizons.append(horizon)

    if not used_horizons:
        return {"asset_id": asset_id, "status": "no_usable_horizon_models"}

    anchor_points = sorted(anchor_points, key=lambda item: item[0])
    x = np.array([point[0] for point in anchor_points], dtype=float)
    y = np.log(np.array([point[1] for point in anchor_points], dtype=float))
    daily_prices = np.exp(np.interp(np.arange(1, FORECAST_CALENDAR_DAYS + 1), x, y))

    now = datetime.now(timezone.utc)
    predictions_horizon = []
    for day, predicted_price in enumerate(daily_prices, start=1):
        target_date = now + timedelta(days=day)
        change_pct = (float(predicted_price) / current_price - 1.0) * 100.0
        predictions_horizon.append({
            "day": day,
            "target_time": target_date,
            "predicted_close": float(predicted_price),
            "change_pct": float(change_pct),
        })

    final_pred = predictions_horizon[-1]["predicted_close"]
    pct = (final_pred / current_price - 1.0) * 100.0
    threshold = float(getattr(settings, "prediction_neutral_threshold_pct", 2.5))
    trend = "ALCISTA" if pct > threshold else ("BAJISTA" if pct < -threshold else "ESTABLE")
    max_item = max(predictions_horizon, key=lambda item: item["predicted_close"])
    min_item = min(predictions_horizon, key=lambda item: item["predicted_close"])

    doc = {
        "asset_id": asset_id,
        "predicted_at": now,
        "current_price": current_price,
        "predictions_180d": predictions_horizon,
        "predicted_close_180d": float(final_pred),
        "predicted_change_pct": float(pct),
        "trend": trend,
        "forecast_method": "direct_multi_horizon_log_return_interpolation",
        "trained_horizons_sessions": sorted(used_horizons),
        "forecast_note": "Los puntos de anclaje son predicciones directas por horizonte; los valores diarios intermedios se interpolan para visualización.",
        "agent_summary": {
            "max_peak": {
                "day": int(max_item["day"]),
                "date": max_item["target_time"],
                "price": float(max_item["predicted_close"]),
                "change_pct": float(max_item["change_pct"]),
            },
            "min_valley": {
                "day": int(min_item["day"]),
                "date": min_item["target_time"],
                "price": float(min_item["predicted_close"]),
                "change_pct": float(min_item["change_pct"]),
            },
        },
        "history_source": source,
        "model_trained_at": model_doc.get("trained_at"),
        "sentiment_compound": float(last_row.get("sentiment_compound", 0.0)),
    }
    db.predictions.update_one({"asset_id": asset_id}, {"$set": doc}, upsert=True)
    return {"status": "predicted", **doc}


def train_all():
    results = []
    print("\n" + "=" * 60)
    print("ENTRENAMIENTO Y BACKTEST CRONOLÓGICO MULTI-HORIZONTE")
    print("=" * 60)
    for asset_id in AGRICULTURAL_COMMODITIES.keys():
        print(f"\nProcesando activo: [{asset_id}]...")
        result = train_one(asset_id)
        if result.get("status") == "trained":
            print("   Estado: ÉXITO")
            print(f"   Horizonte máximo entrenado: {result.get('max_trained_horizon')} sesiones")
        else:
            print(f"   Estado: {result.get('status')}")
        results.append(result)
    print("\nEntrenamiento finalizado.")
    return results


def predict_all():
    results = []
    print("\nGenerando proyecciones...")
    for asset_id in AGRICULTURAL_COMMODITIES.keys():
        print(f"Generando predicción para: [{asset_id}]...")
        result = predict_one(asset_id)
        print(f"   Estado: {result.get('status')}")
        results.append(result)
    print("Proyecciones finalizadas.\n")
    return results


if __name__ == "__main__":
    train_all()
    predict_all()
    print("Proceso completo finalizado.")
