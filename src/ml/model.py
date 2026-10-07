from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
from bson.binary import Binary
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from xgboost import XGBRegressor, Booster, DMatrix

from src.config import settings
from src.db import get_db
from src.ml.features import FEATURES, build
from src.api_clients.yfinance_client import AGRICULTURAL_COMMODITIES

def sentiment_df(db, asset_id):
    """Obtiene el promedio diario del sentimiento de noticias asociadas al activo agrícola."""
    pipeline = [
        {"$match": {"mentioned_asset_ids": asset_id}},
        {"$project": {
            "day": {"$dateTrunc": {"date": "$published_at", "unit": "day", "timezone": "UTC"}},
            "sentiment_compound": 1
        }},
        {"$group": {"_id": "$day", "sentiment_compound": {"$avg": "$sentiment_compound"}}},
        {"$sort": {"_id": 1}}
    ]
    rows = list(db.news.aggregate(pipeline))
    return pd.DataFrame([{"date": x["_id"], "sentiment_compound": x["sentiment_compound"]} for x in rows])

def price_df(db, asset_id):
    """Carga las velas diarias OHLCV desde la colección 'ohlcv' en MongoDB Atlas."""
    rows = list(db.ohlcv.find(
        {"asset_id": asset_id, "period": "1d"},
        {"_id": 0, "timestamp": 1, "close": 1, "volume": 1}
    ).sort("timestamp", 1))

    if len(rows) >= getattr(settings, "model_min_rows", 30):
        df = pd.DataFrame(rows).rename(columns={"timestamp": "date"})
        return df, "yfinance_ohlcv"

    return pd.DataFrame(columns=["date", "close", "volume"]), "none"

def train_one(asset_id):
    """Entrena el modelo XGBoost para un commodity específico."""
    db = get_db()
    prices, source = price_df(db, asset_id)

    if prices.empty:
        return {"asset_id": asset_id, "status": "no_data"}

    feat = build(prices, sentiment_df(db, asset_id), target=True)
    if len(feat) < getattr(settings, "model_min_rows", 30):
        return {"asset_id": asset_id, "status": "insufficient_history", "rows": len(feat)}

    split = int(len(feat) * 0.8)
    train, test = feat.iloc[:split], feat.iloc[split:]
    if len(test) < 10:
        return {"asset_id": asset_id, "status": "insufficient_test"}

    model = XGBRegressor(
        objective="reg:squarederror", n_estimators=500, max_depth=4,
        learning_rate=0.03, subsample=0.85, colsample_bytree=0.85,
        reg_lambda=1.0, random_state=42, n_jobs=2
    )
    model.fit(train[FEATURES], train["target_close_next"])
    pred = model.predict(test[FEATURES])
    y = test["target_close_next"]
    naive = test["close"].to_numpy()

    mae = float(mean_absolute_error(y, pred))
    test_dates = pd.to_datetime(test["date"], utc=True)

    backtest = [
        {
            "date": dt.to_pydatetime(),
            "actual": float(actual),
            "predicted": float(predicted),
            "baseline": float(base),
        }
        for dt, actual, predicted, base in zip(test_dates, y, pred, naive)
    ]

    feature_importance = {
        feature: float(value)
        for feature, value in zip(FEATURES, model.feature_importances_)
    }

    metrics = {
        "asset_id": asset_id,
        "trained_at": datetime.now(timezone.utc),
        "history_source": source,
        "training_rows": len(train),
        "test_rows": len(test),
        "mae": mae,
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mape_pct": float(mean_absolute_percentage_error(y, pred) * 100),
        "r2": float(r2_score(y, pred)),
        "naive_mae": float(mean_absolute_error(y, naive)),
        "feature_importance": feature_importance,
        "backtest": backtest,
    }
    metrics["beats_naive"] = metrics["mae"] < metrics["naive_mae"]

    raw = bytes(model.get_booster().save_raw(raw_format="json"))
    db.models.update_one(
        {"asset_id": asset_id},
        {"$set": {
            "asset_id": asset_id,
            "feature_columns": FEATURES,
            "model_type": "XGBRegressor",
            "model_raw_json": Binary(raw),
            "trained_at": metrics["trained_at"],
        }},
        upsert=True
    )
    db.model_metrics.insert_one(metrics)
    return {"status": "trained", **metrics}

def train_all():
    """Entrena los modelos para todos los activos configurados en AGRICULTURAL_COMMODITIES."""
    return [train_one(asset_id) for asset_id in AGRICULTURAL_COMMODITIES.keys()]

def predict_one(asset_id):
    """
    Genera la proyección autorregresiva paso a paso a 180 días con XGBoost
    y actualiza la colección 'predictions' en MongoDB Atlas.
    """
    db = get_db()
    model_doc = db.models.find_one({"asset_id": asset_id})
    if not model_doc:
        return {"asset_id": asset_id, "status": "no_model"}
    
    prices, source = price_df(db, asset_id)
    if prices.empty:
        return {"asset_id": asset_id, "status": "no_prices"}

    booster = Booster()
    booster.load_model(bytearray(model_doc["model_raw_json"]))
    
    now = datetime.now(timezone.utc)
 
    current_history = prices.copy()
    current_history["date"] = pd.to_datetime(current_history["date"], utc=True)
    current_price = float(current_history["close"].iloc[-1])
    
    sentiment = sentiment_df(db, asset_id)
    if not sentiment.empty and "date" in sentiment.columns:
        sentiment["date"] = pd.to_datetime(sentiment["date"], utc=True)

    predictions_horizon = []

    for day in range(1, 181):
        target_date = now + timedelta(days=day)
      
        updated_feat = build(current_history, sentiment, target=False)
        if updated_feat.empty:
            break
            
        last_row = updated_feat.iloc[-1]
        
        X_df = pd.DataFrame([last_row[FEATURES].to_dict()])[FEATURES]
        dmatrix = DMatrix(X_df, feature_names=FEATURES)
        
        pred_close = float(booster.predict(dmatrix)[0])
        pred_close = max(pred_close, 0.01)  
        
        change_pct = ((pred_close - current_price) / current_price) * 100 if current_price else 0
        
        predictions_horizon.append({
            "day": day,
            "target_time": target_date,
            "predicted_close": float(pred_close),
            "change_pct": float(change_pct)
        })

        new_row = pd.DataFrame([{
            "date": pd.to_datetime(target_date, utc=True),
            "close": pred_close,
            "volume": float(current_history["volume"].iloc[-1]) if "volume" in current_history else 0
        }])
        current_history = pd.concat([current_history, new_row], ignore_index=True)

    if not predictions_horizon:
        return {"asset_id": asset_id, "status": "insufficient_features"}

    final_pred = predictions_horizon[-1]["predicted_close"]
    pct = ((final_pred / current_price) - 1) * 100 if current_price else 0
    
    t = getattr(settings, "prediction_neutral_threshold_pct", 2.5)
    trend = "ALCISTA" if pct > t else ("BAJISTA" if pct < -t else "ESTABLE")

    max_item = max(predictions_horizon, key=lambda x: x["predicted_close"])
    min_item = min(predictions_horizon, key=lambda x: x["predicted_close"])

    doc = {
        "asset_id": asset_id,
        "predicted_at": now,
        "current_price": current_price,
        "predictions_180d": predictions_horizon,
        "predicted_close_180d": final_pred,
        "predicted_change_pct": float(pct),
        "trend": trend,
        "agent_summary": {
            "max_peak": {
                "day": int(max_item["day"]),
                "date": max_item["target_time"],
                "price": float(max_item["predicted_close"]),
                "change_pct": float(max_item["change_pct"])
            },
            "min_valley": {
                "day": int(min_item["day"]),
                "date": min_item["target_time"],
                "price": float(min_item["predicted_close"]),
                "change_pct": float(min_item["change_pct"])
            }
        },
        "history_source": source,
        "model_trained_at": model_doc.get("trained_at"),
        "sentiment_compound": float(sentiment["sentiment_compound"].iloc[-1]) if not sentiment.empty else 0.0,
    }

    db.predictions.update_one(
        {"asset_id": asset_id},
        {"$set": doc},
        upsert=True
    )
    return {"status": "predicted", **doc}

def predict_all():
    """Genera predicciones recorriendo todos los activos configurados."""
    return [predict_one(asset_id) for asset_id in AGRICULTURAL_COMMODITIES.keys()]
