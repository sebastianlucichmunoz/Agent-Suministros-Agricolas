import json
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from bson.binary import Binary
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import Booster, DMatrix, XGBRegressor
from scipy.interpolate import PchipInterpolator

from src.api_clients.yfinance_client import AGRICULTURAL_COMMODITIES
from src.config import settings
from src.db import get_db
from src.ml.features import FEATURES, HORIZONS, build, features_for_horizon

MAX_HORIZON = max(HORIZONS)
FORECAST_CALENDAR_DAYS = 180
RANDOM_STATE = 42
TEST_FRACTION = 0.15
MAX_VALIDATION_FOLDS = 3
VALIDATION_BLOCK_ROWS = 126
TRADING_SESSIONS_PER_YEAR = 252
TRAINING_WINDOWS_YEARS = (3, 5, 10)
MIN_TRAIN_ROWS = 252
MIN_TEST_ROWS = 60
MIN_RELATIVE_GAIN_OVER_NAIVE = 0.02
MAX_NAIVE_HORIZONS = 4
NAIVE_ALLOWED_HORIZONS = (1, 3, 5, 7)
MAX_PRICE_LOG_RETURN = 1.5
RIDGE_ALPHAS = (1.0, 10.0, 100.0)
PARAMETER_CANDIDATES = [
    {"n_estimators": 220, "max_depth": 2, "learning_rate": 0.035,
     "min_child_weight": 8, "subsample": 0.9, "colsample_bytree": 0.9,
     "reg_alpha": 0.1, "reg_lambda": 8.0},
    {"n_estimators": 350, "max_depth": 3, "learning_rate": 0.025,
     "min_child_weight": 10, "subsample": 0.85, "colsample_bytree": 0.85,
     "reg_alpha": 0.2, "reg_lambda": 10.0},
    {"n_estimators": 160, "max_depth": 2, "learning_rate": 0.04,
     "min_child_weight": 15, "subsample": 0.8, "colsample_bytree": 0.85,
     "reg_alpha": 0.5, "reg_lambda": 15.0},
]


def price_df(db, asset_id):
    rows = list(db.ohlcv.find(
        {"asset_id": asset_id, "period": "1d"},
        {"_id": 0, "timestamp": 1, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
    ).sort("timestamp", 1))
    if not rows:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"]), "none"
    frame = pd.DataFrame(rows).rename(columns={"timestamp": "date"})
    return frame, "yfinance_ohlcv"


def _new_model(params=None):
    return XGBRegressor(
        objective="reg:squarederror", random_state=RANDOM_STATE,
        n_jobs=2, tree_method="hist", **(params or PARAMETER_CANDIDATES[0]),
    )


def _new_ridge(alpha):
    return make_pipeline(StandardScaler(), Ridge(alpha=float(alpha)))


def _safe_float(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _metrics(actual_return, predicted_return, base_price, kind):
    actual_return = np.asarray(actual_return, dtype=float)
    predicted_return = np.asarray(predicted_return, dtype=float)
    base_price = np.asarray(base_price, dtype=float)
    actual_price = base_price * np.exp(actual_return)
    predicted_price = base_price * np.exp(np.clip(predicted_return, -MAX_PRICE_LOG_RETURN, MAX_PRICE_LOG_RETURN))
    error = np.abs(actual_price - predicted_price)
    denominator = np.abs(actual_price).sum()
    valid_direction = kind != "naive" and np.any(np.abs(predicted_return) > 1e-12)
    direction = 100.0 * np.mean(np.sign(actual_return) == np.sign(predicted_return)) if valid_direction else None
    naive_mae = float(np.mean(np.abs(actual_price - base_price)))
    mae = float(np.mean(error))
    return {
        "mae_price": mae,
        "wape_pct": _safe_float(100.0 * error.sum() / denominator) if denominator > 0 else None,
        "r2_price": _safe_float(r2_score(actual_price, predicted_price)) if len(actual_price) >= 2 else None,
        "r2_return": _safe_float(r2_score(actual_return, predicted_return)) if len(actual_return) >= 2 else None,
        "directional_accuracy_pct": _safe_float(direction) if direction is not None else None,
        "directional_evaluated": bool(valid_direction),
        "naive_mae_price": naive_mae,
        "beats_naive": bool(mae < naive_mae),
    }


def _walk_forward_splits(development_end, horizon):
    available = development_end - max(MIN_TRAIN_ROWS, 2 * horizon) - horizon
    n_folds = min(MAX_VALIDATION_FOLDS, available // VALIDATION_BLOCK_ROWS)
    if n_folds < 2:
        return []
    first = development_end - n_folds * VALIDATION_BLOCK_ROWS
    splits = []
    for i in range(n_folds):
        start = first + i * VALIDATION_BLOCK_ROWS
        end = start + VALIDATION_BLOCK_ROWS
        train_end = start - horizon
        if train_end >= MIN_TRAIN_ROWS:
            splits.append((train_end, start, end))
    return splits


def _window_start(train_end, years):
    return max(0, train_end - years * TRADING_SESSIONS_PER_YEAR)


def _fit_predict(kind, params, frame, target, train_start, train_end, predict_start, predict_end):
    if kind == "naive":
        return np.zeros(predict_end - predict_start, dtype=float)
    model = _new_model(params) if kind == "xgboost" else _new_ridge(params)
    model.fit(frame.iloc[train_start:train_end], target.iloc[train_start:train_end])
    return np.asarray(model.predict(frame.iloc[predict_start:predict_end]), dtype=float)


def _absolute_wape_components(actual_return, predicted_return, base_price):
    actual_price = base_price * np.exp(actual_return)
    predicted_price = base_price * np.exp(np.clip(predicted_return, -MAX_PRICE_LOG_RETURN, MAX_PRICE_LOG_RETURN))
    return float(np.abs(actual_price - predicted_price).sum()), float(np.abs(actual_price).sum())


def _validate_candidate(kind, years, params, frame, target, prices, folds):
    total_error = 0.0
    total_price = 0.0
    fold_wapes = []
    for train_end, val_start, val_end in folds:
        begin = _window_start(train_end, years) if years else 0
        predictions = _fit_predict(kind, params, frame, target, begin, train_end, val_start, val_end)
        err, denom = _absolute_wape_components(
            target.iloc[val_start:val_end].to_numpy(), predictions, prices[val_start:val_end]
        )
        total_error += err
        total_price += denom
        fold_wapes.append(100.0 * err / denom if denom > 0 else float("inf"))
    return {"kind": kind, "years": years, "params": params,
            "validation_wape_pct": 100.0 * total_error / total_price if total_price > 0 else float("inf"),
            "fold_wape_pct": fold_wapes}


def _select_candidate(frame, target, prices, folds, allow_naive=True):
    baseline = _validate_candidate("naive", None, None, frame, target, prices, folds)
    candidates = [baseline]
    window_scores = {"xgboost": [], "ridge": []}
    for years in TRAINING_WINDOWS_YEARS:
        for kind, params in (("xgboost", PARAMETER_CANDIDATES[0]), ("ridge", RIDGE_ALPHAS[1])):
            result = _validate_candidate(kind, years, params, frame, target, prices, folds)
            candidates.append(result)
            window_scores[kind].append(result)

    best_xgb_window = min(window_scores["xgboost"], key=lambda c: c["validation_wape_pct"])["years"]
    best_ridge_window = min(window_scores["ridge"], key=lambda c: c["validation_wape_pct"])["years"]

    for params in PARAMETER_CANDIDATES[1:]:
        candidates.append(_validate_candidate("xgboost", best_xgb_window, params, frame, target, prices, folds))
    for alpha in (RIDGE_ALPHAS[0], RIDGE_ALPHAS[2]):
        candidates.append(_validate_candidate("ridge", best_ridge_window, alpha, frame, target, prices, folds))

    if not np.isfinite(baseline["validation_wape_pct"]) or baseline["validation_wape_pct"] <= 0:
        selected = baseline if allow_naive else min(
            (candidate for candidate in candidates if candidate["kind"] != "naive"),
            key=lambda candidate: candidate["validation_wape_pct"]
        )
        comparisons = [
            {"model": c["kind"], "window_years": c["years"],
             "hyperparameters": c["params"], "validation_wape_pct": _safe_float(c["validation_wape_pct"]),
             "fold_wape_pct": [_safe_float(v) for v in c["fold_wape_pct"]]}
            for c in candidates
        ]
        return selected, baseline, comparisons

    required_wins = math.ceil(2 * len(folds) / 3)
    eligible = []
    for candidate in candidates:
        if candidate["kind"] == "naive":
            continue
        gain = (baseline["validation_wape_pct"] - candidate["validation_wape_pct"]) / baseline["validation_wape_pct"]
        fold_wins = sum(
            current < reference for current, reference in zip(candidate["fold_wape_pct"], baseline["fold_wape_pct"])
        )
        if gain >= MIN_RELATIVE_GAIN_OVER_NAIVE and fold_wins >= required_wins:
            eligible.append(candidate)

    non_naive = [candidate for candidate in candidates if candidate["kind"] != "naive"]
    best_non_naive = min(non_naive, key=lambda candidate: candidate["validation_wape_pct"])
    if allow_naive and baseline["validation_wape_pct"] <= best_non_naive["validation_wape_pct"]:
        selected = baseline
    else:
        selected = min(eligible, key=lambda candidate: candidate["validation_wape_pct"]) if eligible else best_non_naive
    comparisons = [
        {"model": c["kind"], "window_years": c["years"],
         "hyperparameters": c["params"], "validation_wape_pct": _safe_float(c["validation_wape_pct"]),
         "fold_wape_pct": [_safe_float(v) for v in c["fold_wape_pct"]]}
        for c in candidates
    ]
    return selected, baseline, comparisons


def _target_date_after(data, origin_date, horizon):
    dates = data["date"].dropna().sort_values().drop_duplicates().reset_index(drop=True)
    position = int(dates.searchsorted(pd.Timestamp(origin_date), side="right"))
    target_position = position + horizon - 1
    return dates.iloc[target_position].to_pydatetime() if target_position < len(dates) else None


def _serialize_model(kind, model):
    if kind == "xgboost":
        return Binary(bytes(model.get_booster().save_raw(raw_format="json")))
    if kind == "naive":
        return Binary(b'{"kind":"naive"}')
    scaler = model.named_steps["standardscaler"]
    ridge = model.named_steps["ridge"]
    payload = {"kind": "ridge", "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
               "coef": ridge.coef_.tolist(), "intercept": float(ridge.intercept_)}
    return Binary(json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def _evaluate_horizon(data, common, horizon, development_end, folds, allow_naive=True):
    features = features_for_horizon(horizon)
    target_name = f"target_return_{horizon}d"
    X = common[features]
    y = common[target_name]
    base = common["close"].to_numpy(dtype=float)
    selected, baseline, comparisons = _select_candidate(X, y, base, folds, allow_naive=allow_naive)

    train_end = development_end - horizon
    begin = _window_start(train_end, selected["years"]) if selected["years"] else 0
    test_pred = _fit_predict(
        selected["kind"], selected["params"], X, y, begin, train_end,
        development_end, len(common)
    )
    actual = y.iloc[development_end:].to_numpy(dtype=float)
    test_base = base[development_end:]
    metric = _metrics(actual, test_pred, test_base, selected["kind"])
    baseline_metrics = _metrics(actual, np.zeros_like(actual), test_base, "naive")

    fold_metrics = []
    for index, (fold_train_end, val_start, val_end) in enumerate(folds, start=1):
        start = _window_start(fold_train_end, selected["years"]) if selected["years"] else 0
        fold_pred = _fit_predict(
            selected["kind"], selected["params"], X, y, start, fold_train_end, val_start, val_end
        )
        fold_metrics.append({
            "fold": index,
            "train_rows": fold_train_end - start if selected["kind"] != "naive" else 0,
            "validation_rows": val_end - val_start,
            "train_start": common["date"].iloc[start].to_pydatetime() if selected["kind"] != "naive" else None,
            "train_end": common["date"].iloc[fold_train_end - 1].to_pydatetime() if selected["kind"] != "naive" else None,
            "validation_start": common["date"].iloc[val_start].to_pydatetime(),
            "validation_end": common["date"].iloc[val_end - 1].to_pydatetime(),
            **_metrics(y.iloc[val_start:val_end].to_numpy(), fold_pred,
                       base[val_start:val_end], selected["kind"]),
        })

    final_data = (data.dropna(subset=features + [target_name, "close", "date"])
                  .sort_values("date").reset_index(drop=True))
    final_start = _window_start(len(final_data), selected["years"]) if selected["years"] else 0
    if selected["kind"] == "naive":
        final_model = None
    else:
        final_model = (_new_model(selected["params"]) if selected["kind"] == "xgboost"
                       else _new_ridge(selected["params"]))
        final_model.fit(final_data[features].iloc[final_start:], final_data[target_name].iloc[final_start:])

    model_raw = _serialize_model(selected["kind"], final_model)
    result = {
        "status": "trained",
        "horizon": int(horizon),
        "rows": int(len(common)),
        "training_rows": int(train_end - begin) if selected["kind"] != "naive" else 0,
        "test_rows": int(len(common) - development_end),
        "validation_folds": len(folds),
        "selected_model": selected["kind"],
        "naive_allowed": bool(allow_naive),
        "naive_restricted_by_policy": not allow_naive,
        "validation_gap_vs_naive_pct_points": _safe_float(
            selected["validation_wape_pct"] - baseline["validation_wape_pct"]
        ),
        "selected_training_years": selected["years"],
        "selected_hyperparameters": selected["params"],
        "validation_wape_pct": _safe_float(selected["validation_wape_pct"]),
        "naive_validation_wape_pct": _safe_float(baseline["validation_wape_pct"]),
        "candidate_validation": comparisons,
        "feature_columns": features if selected["kind"] != "naive" else [],
        "train_start": common["date"].iloc[begin].to_pydatetime() if selected["kind"] != "naive" else None,
        "train_end": common["date"].iloc[train_end - 1].to_pydatetime() if selected["kind"] != "naive" else None,
        "test_start": common["date"].iloc[development_end].to_pydatetime(),
        "test_end": common["date"].iloc[-1].to_pydatetime(),
        "last_test_target_date": _target_date_after(data, common["date"].iloc[-1], horizon),
        "final_training_rows": len(final_data) - final_start if selected["kind"] != "naive" else 0,
        "final_training_end": final_data["date"].iloc[-1].to_pydatetime() if selected["kind"] != "naive" else None,
        "naive_test_metrics": baseline_metrics,
        "fold_metrics": fold_metrics,
        "model_raw_json": model_raw,
        **metric,
    }
    return result


def _period_label(horizon):
    return f"Mes {min(6, math.ceil(horizon / 21))} (aprox.)"


def _extreme(entries, field, choose_min):
    values = [(h, m) for h, m in entries if m.get(field) is not None]
    if not values:
        return None
    horizon, metric = (min(values, key=lambda item: item[1][field]) if choose_min
                       else max(values, key=lambda item: item[1][field]))
    return {"value": float(metric[field]), "horizon_sessions": int(horizon),
            "period": _period_label(horizon), "model": metric["selected_model"]}


def _mean_valid(values):
    array = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(array)) if array else None


def _metric_summary(metrics_by_horizon):
    entries = [(int(h), metric) for h, metric in metrics_by_horizon.items()
               if metric.get("status") == "trained"]
    return {
        "trained_horizons": len(entries),
        "best_mae": _extreme(entries, "mae_price", True),
        "worst_mae": _extreme(entries, "mae_price", False),
        "best_wape": _extreme(entries, "wape_pct", True),
        "worst_wape": _extreme(entries, "wape_pct", False),
        "mean_mae_price": _mean_valid([m["mae_price"] for _, m in entries]),
        "mean_wape_pct": _mean_valid([m["wape_pct"] for _, m in entries]),
        "r2_price": _mean_valid([m["r2_price"] for _, m in entries]),
        "r2_return": _mean_valid([m["r2_return"] for _, m in entries]),
        "directional_accuracy_pct": _mean_valid([m["directional_accuracy_pct"] for _, m in entries]),
        "directional_horizons_evaluated": sum(m["directional_evaluated"] for _, m in entries),
        "directional_coverage_pct": (100.0 * sum(m["directional_evaluated"] for _, m in entries) / len(entries)
                                     if entries else None),
        "horizons_beating_naive": sum(m["beats_naive"] for _, m in entries),
        "horizons_with_higher_validation_wape_than_naive": sum(
            (m.get("validation_gap_vs_naive_pct_points") or 0) > 0
            for _, m in entries
        ),
        "selected_model_counts": {
            kind: sum(m["selected_model"] == kind for _, m in entries)
            for kind in ("xgboost", "ridge", "naive")
        },
        "r2_aggregation": "arithmetic_mean_by_horizon_on_common_holdout",
    }


def train_one(asset_id):
    db = get_db()
    prices, source = price_df(db, asset_id)
    if prices.empty:
        return {"asset_id": asset_id, "status": "no_data"}

    data = build(prices, target=True)
    common = (data.dropna(subset=FEATURES + [f"target_return_{MAX_HORIZON}d", "close", "date"])
              .sort_values("date").reset_index(drop=True))
    min_rows = int(getattr(settings, "model_min_rows", 30))
    if len(common) < max(min_rows, 900):
        return {"asset_id": asset_id, "status": "insufficient_history", "rows": len(common)}

    test_rows = max(MIN_TEST_ROWS, int(round(len(common) * TEST_FRACTION)))
    development_end = len(common) - test_rows
    metrics_by_horizon = {}
    horizon_models = {}
    first_horizon_naive = False
    naive_selected_count = 0

    for horizon in HORIZONS:
        folds = _walk_forward_splits(development_end, horizon)
        if not folds:
            metrics_by_horizon[str(horizon)] = {"status": "insufficient_walk_forward_history"}
            continue
        allow_naive = (
            horizon == 1 or (
                horizon in NAIVE_ALLOWED_HORIZONS
                and first_horizon_naive
                and naive_selected_count < MAX_NAIVE_HORIZONS
            )
        )
        result = _evaluate_horizon(
            data, common, horizon, development_end, folds, allow_naive=allow_naive
        )
        if horizon == 1:
            first_horizon_naive = result["selected_model"] == "naive"
        if result["selected_model"] == "naive":
            naive_selected_count += 1
        horizon_models[str(horizon)] = {
            "model_raw_json": result["model_raw_json"],
            "feature_columns": result["feature_columns"],
        }
        metrics_by_horizon[str(horizon)] = {
            key: value for key, value in result.items() if key != "model_raw_json"
        }
        r2_text = f"{result['r2_price']:.3f}" if result["r2_price"] is not None else "N/A"
        direction_text = (f"{result['directional_accuracy_pct']:.1f}%"
                          if result["directional_accuracy_pct"] is not None else "N/A")
        print(f"   {horizon:>3} sesiones | {result['selected_model']:<7} | "
              f"MAE {result['mae_price']:.4f} | WAPE {result['wape_pct']:.2f}% | "
              f"R² precio {r2_text} | Dirección {direction_text} | "
              f"WF {result['validation_folds']} folds | "
              f"ventana {result['selected_training_years'] or 0} años")

    if len(horizon_models) != len(HORIZONS):
        return {"asset_id": asset_id, "status": "incomplete_horizons",
                "metrics_by_horizon": metrics_by_horizon}

    now = datetime.now(timezone.utc)
    model_doc = {
        "asset_id": asset_id,
        "model_type": "HybridRegressor_multi_horizon",
        "feature_columns": FEATURES,
        "horizons": [int(h) for h in horizon_models],
        "horizon_models": horizon_models,
        "trained_at": now,
        "history_source": source,
    }
    db.models.update_one({"asset_id": asset_id}, {"$set": model_doc}, upsert=True)

    metric_doc = {
        "asset_id": asset_id,
        "trained_at": now,
        "history_source": source,
        "evaluation_method": "common_holdout_purged_walk_forward_model_selection_naive_restricted",
        "naive_selection_policy": {
            "allowed_horizons": list(NAIVE_ALLOWED_HORIZONS),
            "maximum_selected": MAX_NAIVE_HORIZONS,
            "requires_selection_at_first_horizon": True,
        },
        "metrics_by_horizon": metrics_by_horizon,
        "max_trained_horizon": MAX_HORIZON,
        "feature_columns": FEATURES,
        "summary": _metric_summary(metrics_by_horizon),
    }
    db.model_metrics.insert_one(metric_doc)
    return {"asset_id": asset_id, "status": "trained", "trained_at": now,
            "metrics_by_horizon": metrics_by_horizon, "summary": metric_doc["summary"],
            "max_trained_horizon": MAX_HORIZON}


def _save_general_metrics(db, results):
    trained = [item for item in results if item.get("status") == "trained"]
    horizons = {}
    for horizon in HORIZONS:
        metrics = [item["metrics_by_horizon"][str(horizon)] for item in trained]
        if not metrics:
            continue
        horizons[str(horizon)] = {
            "commodities_evaluated": len(metrics),
            "mean_wape_pct": _mean_valid([m["wape_pct"] for m in metrics]),
            "mean_r2_price": _mean_valid([m["r2_price"] for m in metrics]),
            "mean_r2_return": _mean_valid([m["r2_return"] for m in metrics]),
            "mean_directional_accuracy_pct": _mean_valid([m["directional_accuracy_pct"] for m in metrics]),
            "directional_commodities_evaluated": sum(m["directional_evaluated"] for m in metrics),
            "selected_model_counts": {
                kind: sum(m["selected_model"] == kind for m in metrics)
                for kind in ("xgboost", "ridge", "naive")
            },
        }

    summaries = [item["summary"] for item in trained]
    overall = {
        "commodities_evaluated": len(trained),
        "commodities_expected": len(AGRICULTURAL_COMMODITIES),
        "mean_wape_pct": _mean_valid([s["mean_wape_pct"] for s in summaries]),
        "mean_r2_price": _mean_valid([s["r2_price"] for s in summaries]),
        "mean_r2_return": _mean_valid([s["r2_return"] for s in summaries]),
        "mean_directional_accuracy_pct": _mean_valid([s["directional_accuracy_pct"] for s in summaries]),
        "mean_directional_coverage_pct": _mean_valid([s["directional_coverage_pct"] for s in summaries]),
        "selected_model_counts": {
            kind: sum(s["selected_model_counts"][kind] for s in summaries)
            for kind in ("xgboost", "ridge", "naive")
        },
        "r2_aggregation": "macro_average_of_per_commodity_horizon_r2_price",
    }
    document = {
        "asset_id": "__general__", "trained_at": datetime.now(timezone.utc),
        "evaluation_method": "macro_average_of_selected_model_holdouts",
        "metrics_by_horizon": horizons, "summary": overall,
    }
    db.model_metrics.insert_one(document)
    return document


def _predict_from_binary(blob, feature_frame):
    raw = bytes(blob)
    if raw.startswith(b'{"kind":"naive"'):
        return 0.0
    if raw.startswith(b'{"kind":"ridge"'):
        params = json.loads(raw.decode("utf-8"))
        X = feature_frame.to_numpy(dtype=float).reshape(-1)
        value = np.dot((X - np.asarray(params["mean"])) / np.asarray(params["scale"]),
                       np.asarray(params["coef"])) + float(params["intercept"])
        return float(value)
    booster = Booster()
    booster.load_model(bytearray(raw))
    return float(booster.predict(DMatrix(feature_frame, feature_names=list(feature_frame.columns)))[0])


def predict_one(asset_id):
    db = get_db()
    model_doc = db.models.find_one({"asset_id": asset_id})
    if not model_doc:
        return {"asset_id": asset_id, "status": "no_model"}
    horizon_models = model_doc.get("horizon_models") or {}
    if not all(str(horizon) in horizon_models for horizon in HORIZONS):
        return {"asset_id": asset_id, "status": "incomplete_horizons_retrain_required"}

    prices, source = price_df(db, asset_id)
    if prices.empty:
        return {"asset_id": asset_id, "status": "no_prices"}
    data = build(prices, target=False)
    required = sorted({name for model in horizon_models.values() for name in model.get("feature_columns", [])})
    valid = data.dropna(subset=required + ["date", "close"])
    if valid.empty:
        return {"asset_id": asset_id, "status": "insufficient_features"}

    last = valid.iloc[-1]
    latest_data_date = pd.Timestamp(last["date"]).to_pydatetime()
    if (datetime.now(timezone.utc) - latest_data_date).total_seconds() > 10 * 86400:
        return {"asset_id": asset_id, "status": "stale_market_data"}
    current_price = float(last["close"])
    if not np.isfinite(current_price) or current_price <= 0:
        return {"asset_id": asset_id, "status": "invalid_current_price"}

    anchors = [(0.0, current_price)]
    used_horizons = []
    for horizon in HORIZONS:
        payload = horizon_models[str(horizon)]
        cols = payload.get("feature_columns", [])
        frame = pd.DataFrame([{col: float(last[col]) for col in cols}], columns=cols)
        predicted_return = _predict_from_binary(payload["model_raw_json"], frame)
        if not np.isfinite(predicted_return):
            return {"asset_id": asset_id, "status": "invalid_model_prediction", "horizon": horizon}
        predicted_price = current_price * float(np.exp(np.clip(predicted_return, -MAX_PRICE_LOG_RETURN, MAX_PRICE_LOG_RETURN)))
        calendar_day = horizon * FORECAST_CALENDAR_DAYS / MAX_HORIZON
        anchors.append((float(calendar_day), predicted_price))
        used_horizons.append(int(horizon))

    anchors.sort(key=lambda item: item[0])

    x = np.asarray([item[0] for item in anchors], dtype=float)
    log_y = np.log(np.asarray([item[1] for item in anchors], dtype=float))

    interpolator = PchipInterpolator(x, log_y)

    daily_prices = np.exp(
        interpolator(np.arange(1, FORECAST_CALENDAR_DAYS + 1))
    )

    now = datetime.now(timezone.utc)
    daily_forecasts = []
    for day, value in enumerate(daily_prices, start=1):
        target_date = latest_data_date + timedelta(days=day)
        change_pct = (float(value) / current_price - 1.0) * 100.0
        daily_forecasts.append({"day": day, "target_time": target_date,
                                "predicted_close": float(value), "change_pct": float(change_pct)})

    final_price = float(daily_forecasts[-1]["predicted_close"])
    change_pct = (final_price / current_price - 1.0) * 100.0
    threshold = float(getattr(settings, "prediction_neutral_threshold_pct", 2.5))
    trend = "ALCISTA" if change_pct > threshold else ("BAJISTA" if change_pct < -threshold else "ESTABLE")
    maximum = max(daily_forecasts, key=lambda item: item["predicted_close"])
    minimum = min(daily_forecasts, key=lambda item: item["predicted_close"])

    document = {
        "asset_id": asset_id,
        "predicted_at": now,
        "current_price": current_price,
        "predictions_180d": daily_forecasts,
        "predicted_close_180d": final_price,
        "predicted_change_pct": float(change_pct),
        "trend": trend,
        "forecast_method": "direct_multi_horizon_log_return_interpolation",
        "trained_horizons_sessions": sorted(used_horizons),
        "forecast_note": "Los puntos de anclaje son pronósticos directos de modelos seleccionados por horizonte; "
                         "los valores diarios intermedios se interpolan para visualización.",
        "agent_summary": {
            "max_peak": {"day": maximum["day"], "date": maximum["target_time"],
                         "price": maximum["predicted_close"], "change_pct": maximum["change_pct"]},
            "min_valley": {"day": minimum["day"], "date": minimum["target_time"],
                           "price": minimum["predicted_close"], "change_pct": minimum["change_pct"]},
        },
        "history_source": source,
        "model_trained_at": model_doc.get("trained_at"),
        "sentiment_compound": 0.0,
    }
    db.predictions.update_one({"asset_id": asset_id}, {"$set": document}, upsert=True)
    return {"status": "predicted", **document}


def train_all():
    results = []
    print("\n" + "=" * 60)
    print("ENTRENAMIENTO MULTIMODELO + WALK-FORWARD POR COMMODITY")
    print("=" * 60)
    for asset_id in AGRICULTURAL_COMMODITIES:
        print(f"\nEntrenando: [{asset_id}]")
        try:
            result = train_one(asset_id)
        except Exception as exc:
            print(f"   ERROR: {exc}")
            result = {"asset_id": asset_id, "status": "error", "message": str(exc)}
        print(f"   Estado: {result['status']}")
        results.append(result)
    general = _save_general_metrics(get_db(), results)
    print("\nResumen general:", general["summary"])
    return results


def predict_all():
    results = []
    print("\nGenerando proyecciones...")
    for asset_id in AGRICULTURAL_COMMODITIES:
        try:
            result = predict_one(asset_id)
        except Exception as exc:
            result = {"asset_id": asset_id, "status": "error", "message": str(exc)}
        print(f"   {asset_id}: {result.get('status')}")
        results.append(result)
    return results


if __name__ == "__main__":
    train_all()
    predict_all()











# -------Versión buenas
# import json
# import math
# from datetime import datetime, timedelta, timezone

# import numpy as np
# import pandas as pd
# from bson.binary import Binary
# from sklearn.linear_model import Ridge
# from sklearn.metrics import r2_score
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
# from xgboost import Booster, DMatrix, XGBRegressor

# from src.api_clients.yfinance_client import AGRICULTURAL_COMMODITIES
# from src.config import settings
# from src.db import get_db
# from src.ml.features import FEATURES, HORIZONS, build, features_for_horizon

# MAX_HORIZON = max(HORIZONS)
# FORECAST_CALENDAR_DAYS = 180
# RANDOM_STATE = 42
# TEST_FRACTION = 0.15
# MAX_VALIDATION_FOLDS = 3
# VALIDATION_BLOCK_ROWS = 126
# TRADING_SESSIONS_PER_YEAR = 252
# TRAINING_WINDOWS_YEARS = (3, 5, 10)
# MIN_TRAIN_ROWS = 252
# MIN_TEST_ROWS = 60
# MIN_RELATIVE_GAIN_OVER_NAIVE = 0.02
# MAX_PRICE_LOG_RETURN = 1.5
# RIDGE_ALPHAS = (1.0, 10.0, 100.0)
# PARAMETER_CANDIDATES = [
#     {"n_estimators": 220, "max_depth": 2, "learning_rate": 0.035,
#      "min_child_weight": 8, "subsample": 0.9, "colsample_bytree": 0.9,
#      "reg_alpha": 0.1, "reg_lambda": 8.0},
#     {"n_estimators": 350, "max_depth": 3, "learning_rate": 0.025,
#      "min_child_weight": 10, "subsample": 0.85, "colsample_bytree": 0.85,
#      "reg_alpha": 0.2, "reg_lambda": 10.0},
#     {"n_estimators": 160, "max_depth": 2, "learning_rate": 0.04,
#      "min_child_weight": 15, "subsample": 0.8, "colsample_bytree": 0.85,
#      "reg_alpha": 0.5, "reg_lambda": 15.0},
# ]


# def price_df(db, asset_id):
#     rows = list(db.ohlcv.find(
#         {"asset_id": asset_id, "period": "1d"},
#         {"_id": 0, "timestamp": 1, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
#     ).sort("timestamp", 1))
#     if not rows:
#         return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"]), "none"
#     frame = pd.DataFrame(rows).rename(columns={"timestamp": "date"})
#     return frame, "yfinance_ohlcv"


# def _new_model(params=None):
#     return XGBRegressor(
#         objective="reg:squarederror", random_state=RANDOM_STATE,
#         n_jobs=2, tree_method="hist", **(params or PARAMETER_CANDIDATES[0]),
#     )


# def _new_ridge(alpha):
#     return make_pipeline(StandardScaler(), Ridge(alpha=float(alpha)))


# def _safe_float(value):
#     value = float(value)
#     return value if np.isfinite(value) else None


# def _metrics(actual_return, predicted_return, base_price, kind):
#     actual_return = np.asarray(actual_return, dtype=float)
#     predicted_return = np.asarray(predicted_return, dtype=float)
#     base_price = np.asarray(base_price, dtype=float)
#     actual_price = base_price * np.exp(actual_return)
#     predicted_price = base_price * np.exp(np.clip(predicted_return, -MAX_PRICE_LOG_RETURN, MAX_PRICE_LOG_RETURN))
#     error = np.abs(actual_price - predicted_price)
#     denominator = np.abs(actual_price).sum()
#     valid_direction = kind != "naive" and np.any(np.abs(predicted_return) > 1e-12)
#     direction = 100.0 * np.mean(np.sign(actual_return) == np.sign(predicted_return)) if valid_direction else None
#     naive_mae = float(np.mean(np.abs(actual_price - base_price)))
#     mae = float(np.mean(error))
#     return {
#         "mae_price": mae,
#         "wape_pct": _safe_float(100.0 * error.sum() / denominator) if denominator > 0 else None,
#         "r2_price": _safe_float(r2_score(actual_price, predicted_price)) if len(actual_price) >= 2 else None,
#         "r2_return": _safe_float(r2_score(actual_return, predicted_return)) if len(actual_return) >= 2 else None,
#         "directional_accuracy_pct": _safe_float(direction) if direction is not None else None,
#         "directional_evaluated": bool(valid_direction),
#         "naive_mae_price": naive_mae,
#         "beats_naive": bool(mae < naive_mae),
#     }


# def _walk_forward_splits(development_end, horizon):
#     available = development_end - max(MIN_TRAIN_ROWS, 2 * horizon) - horizon
#     n_folds = min(MAX_VALIDATION_FOLDS, available // VALIDATION_BLOCK_ROWS)
#     if n_folds < 2:
#         return []
#     first = development_end - n_folds * VALIDATION_BLOCK_ROWS
#     splits = []
#     for i in range(n_folds):
#         start = first + i * VALIDATION_BLOCK_ROWS
#         end = start + VALIDATION_BLOCK_ROWS
#         train_end = start - horizon
#         if train_end >= MIN_TRAIN_ROWS:
#             splits.append((train_end, start, end))
#     return splits


# def _window_start(train_end, years):
#     return max(0, train_end - years * TRADING_SESSIONS_PER_YEAR)


# def _fit_predict(kind, params, frame, target, train_start, train_end, predict_start, predict_end):
#     if kind == "naive":
#         return np.zeros(predict_end - predict_start, dtype=float)
#     model = _new_model(params) if kind == "xgboost" else _new_ridge(params)
#     model.fit(frame.iloc[train_start:train_end], target.iloc[train_start:train_end])
#     return np.asarray(model.predict(frame.iloc[predict_start:predict_end]), dtype=float)


# def _absolute_wape_components(actual_return, predicted_return, base_price):
#     actual_price = base_price * np.exp(actual_return)
#     predicted_price = base_price * np.exp(np.clip(predicted_return, -MAX_PRICE_LOG_RETURN, MAX_PRICE_LOG_RETURN))
#     return float(np.abs(actual_price - predicted_price).sum()), float(np.abs(actual_price).sum())


# def _validate_candidate(kind, years, params, frame, target, prices, folds):
#     total_error = 0.0
#     total_price = 0.0
#     fold_wapes = []
#     for train_end, val_start, val_end in folds:
#         begin = _window_start(train_end, years) if years else 0
#         predictions = _fit_predict(kind, params, frame, target, begin, train_end, val_start, val_end)
#         err, denom = _absolute_wape_components(
#             target.iloc[val_start:val_end].to_numpy(), predictions, prices[val_start:val_end]
#         )
#         total_error += err
#         total_price += denom
#         fold_wapes.append(100.0 * err / denom if denom > 0 else float("inf"))
#     return {"kind": kind, "years": years, "params": params,
#             "validation_wape_pct": 100.0 * total_error / total_price if total_price > 0 else float("inf"),
#             "fold_wape_pct": fold_wapes}


# def _select_candidate(frame, target, prices, folds):
#     baseline = _validate_candidate("naive", None, None, frame, target, prices, folds)
#     candidates = [baseline]
#     window_scores = {"xgboost": [], "ridge": []}
#     for years in TRAINING_WINDOWS_YEARS:
#         for kind, params in (("xgboost", PARAMETER_CANDIDATES[0]), ("ridge", RIDGE_ALPHAS[1])):
#             result = _validate_candidate(kind, years, params, frame, target, prices, folds)
#             candidates.append(result)
#             window_scores[kind].append(result)

#     best_xgb_window = min(window_scores["xgboost"], key=lambda c: c["validation_wape_pct"])["years"]
#     best_ridge_window = min(window_scores["ridge"], key=lambda c: c["validation_wape_pct"])["years"]

#     for params in PARAMETER_CANDIDATES[1:]:
#         candidates.append(_validate_candidate("xgboost", best_xgb_window, params, frame, target, prices, folds))
#     for alpha in (RIDGE_ALPHAS[0], RIDGE_ALPHAS[2]):
#         candidates.append(_validate_candidate("ridge", best_ridge_window, alpha, frame, target, prices, folds))

#     if not np.isfinite(baseline["validation_wape_pct"]) or baseline["validation_wape_pct"] <= 0:
#         selected = baseline
#         comparisons = [
#             {"model": c["kind"], "window_years": c["years"],
#              "hyperparameters": c["params"], "validation_wape_pct": _safe_float(c["validation_wape_pct"]),
#              "fold_wape_pct": [_safe_float(v) for v in c["fold_wape_pct"]]}
#             for c in candidates
#         ]
#         return selected, baseline, comparisons

#     required_wins = math.ceil(2 * len(folds) / 3)
#     eligible = []
#     for candidate in candidates:
#         if candidate["kind"] == "naive":
#             continue
#         gain = (baseline["validation_wape_pct"] - candidate["validation_wape_pct"]) / baseline["validation_wape_pct"]
#         fold_wins = sum(
#             current < reference for current, reference in zip(candidate["fold_wape_pct"], baseline["fold_wape_pct"])
#         )
#         if gain >= MIN_RELATIVE_GAIN_OVER_NAIVE and fold_wins >= required_wins:
#             eligible.append(candidate)

#     selected = min(eligible, key=lambda c: c["validation_wape_pct"]) if eligible else baseline
#     comparisons = [
#         {"model": c["kind"], "window_years": c["years"],
#          "hyperparameters": c["params"], "validation_wape_pct": _safe_float(c["validation_wape_pct"]),
#          "fold_wape_pct": [_safe_float(v) for v in c["fold_wape_pct"]]}
#         for c in candidates
#     ]
#     return selected, baseline, comparisons


# def _target_date_after(data, origin_date, horizon):
#     dates = data["date"].dropna().sort_values().drop_duplicates().reset_index(drop=True)
#     position = int(dates.searchsorted(pd.Timestamp(origin_date), side="right"))
#     target_position = position + horizon - 1
#     return dates.iloc[target_position].to_pydatetime() if target_position < len(dates) else None


# def _serialize_model(kind, model):
#     if kind == "xgboost":
#         return Binary(bytes(model.get_booster().save_raw(raw_format="json")))
#     if kind == "naive":
#         return Binary(b'{"kind":"naive"}')
#     scaler = model.named_steps["standardscaler"]
#     ridge = model.named_steps["ridge"]
#     payload = {"kind": "ridge", "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
#                "coef": ridge.coef_.tolist(), "intercept": float(ridge.intercept_)}
#     return Binary(json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8"))


# def _evaluate_horizon(data, common, horizon, development_end, folds):
#     features = features_for_horizon(horizon)
#     target_name = f"target_return_{horizon}d"
#     X = common[features]
#     y = common[target_name]
#     base = common["close"].to_numpy(dtype=float)
#     selected, baseline, comparisons = _select_candidate(X, y, base, folds)

#     train_end = development_end - horizon
#     begin = _window_start(train_end, selected["years"]) if selected["years"] else 0
#     test_pred = _fit_predict(
#         selected["kind"], selected["params"], X, y, begin, train_end,
#         development_end, len(common)
#     )
#     actual = y.iloc[development_end:].to_numpy(dtype=float)
#     test_base = base[development_end:]
#     metric = _metrics(actual, test_pred, test_base, selected["kind"])
#     baseline_metrics = _metrics(actual, np.zeros_like(actual), test_base, "naive")

#     fold_metrics = []
#     for index, (fold_train_end, val_start, val_end) in enumerate(folds, start=1):
#         start = _window_start(fold_train_end, selected["years"]) if selected["years"] else 0
#         fold_pred = _fit_predict(
#             selected["kind"], selected["params"], X, y, start, fold_train_end, val_start, val_end
#         )
#         fold_metrics.append({
#             "fold": index,
#             "train_rows": fold_train_end - start if selected["kind"] != "naive" else 0,
#             "validation_rows": val_end - val_start,
#             "train_start": common["date"].iloc[start].to_pydatetime() if selected["kind"] != "naive" else None,
#             "train_end": common["date"].iloc[fold_train_end - 1].to_pydatetime() if selected["kind"] != "naive" else None,
#             "validation_start": common["date"].iloc[val_start].to_pydatetime(),
#             "validation_end": common["date"].iloc[val_end - 1].to_pydatetime(),
#             **_metrics(y.iloc[val_start:val_end].to_numpy(), fold_pred,
#                        base[val_start:val_end], selected["kind"]),
#         })

#     final_data = (data.dropna(subset=features + [target_name, "close", "date"])
#                   .sort_values("date").reset_index(drop=True))
#     final_start = _window_start(len(final_data), selected["years"]) if selected["years"] else 0
#     if selected["kind"] == "naive":
#         final_model = None
#     else:
#         final_model = (_new_model(selected["params"]) if selected["kind"] == "xgboost"
#                        else _new_ridge(selected["params"]))
#         final_model.fit(final_data[features].iloc[final_start:], final_data[target_name].iloc[final_start:])

#     model_raw = _serialize_model(selected["kind"], final_model)
#     result = {
#         "status": "trained",
#         "horizon": int(horizon),
#         "rows": int(len(common)),
#         "training_rows": int(train_end - begin) if selected["kind"] != "naive" else 0,
#         "test_rows": int(len(common) - development_end),
#         "validation_folds": len(folds),
#         "selected_model": selected["kind"],
#         "selected_training_years": selected["years"],
#         "selected_hyperparameters": selected["params"],
#         "validation_wape_pct": _safe_float(selected["validation_wape_pct"]),
#         "naive_validation_wape_pct": _safe_float(baseline["validation_wape_pct"]),
#         "candidate_validation": comparisons,
#         "feature_columns": features if selected["kind"] != "naive" else [],
#         "train_start": common["date"].iloc[begin].to_pydatetime() if selected["kind"] != "naive" else None,
#         "train_end": common["date"].iloc[train_end - 1].to_pydatetime() if selected["kind"] != "naive" else None,
#         "test_start": common["date"].iloc[development_end].to_pydatetime(),
#         "test_end": common["date"].iloc[-1].to_pydatetime(),
#         "last_test_target_date": _target_date_after(data, common["date"].iloc[-1], horizon),
#         "final_training_rows": len(final_data) - final_start if selected["kind"] != "naive" else 0,
#         "final_training_end": final_data["date"].iloc[-1].to_pydatetime() if selected["kind"] != "naive" else None,
#         "naive_test_metrics": baseline_metrics,
#         "fold_metrics": fold_metrics,
#         "model_raw_json": model_raw,
#         **metric,
#     }
#     return result


# def _period_label(horizon):
#     return f"Mes {min(6, math.ceil(horizon / 21))} (aprox.)"


# def _extreme(entries, field, choose_min):
#     values = [(h, m) for h, m in entries if m.get(field) is not None]
#     if not values:
#         return None
#     horizon, metric = (min(values, key=lambda item: item[1][field]) if choose_min
#                        else max(values, key=lambda item: item[1][field]))
#     return {"value": float(metric[field]), "horizon_sessions": int(horizon),
#             "period": _period_label(horizon), "model": metric["selected_model"]}


# def _mean_valid(values):
#     array = [float(value) for value in values if value is not None and np.isfinite(value)]
#     return float(np.mean(array)) if array else None


# def _metric_summary(metrics_by_horizon):
#     entries = [(int(h), metric) for h, metric in metrics_by_horizon.items()
#                if metric.get("status") == "trained"]
#     return {
#         "trained_horizons": len(entries),
#         "best_mae": _extreme(entries, "mae_price", True),
#         "worst_mae": _extreme(entries, "mae_price", False),
#         "best_wape": _extreme(entries, "wape_pct", True),
#         "worst_wape": _extreme(entries, "wape_pct", False),
#         "mean_mae_price": _mean_valid([m["mae_price"] for _, m in entries]),
#         "mean_wape_pct": _mean_valid([m["wape_pct"] for _, m in entries]),
#         "r2_price": _mean_valid([m["r2_price"] for _, m in entries]),
#         "r2_return": _mean_valid([m["r2_return"] for _, m in entries]),
#         "directional_accuracy_pct": _mean_valid([m["directional_accuracy_pct"] for _, m in entries]),
#         "directional_horizons_evaluated": sum(m["directional_evaluated"] for _, m in entries),
#         "directional_coverage_pct": (100.0 * sum(m["directional_evaluated"] for _, m in entries) / len(entries)
#                                      if entries else None),
#         "horizons_beating_naive": sum(m["beats_naive"] for _, m in entries),
#         "selected_model_counts": {
#             kind: sum(m["selected_model"] == kind for _, m in entries)
#             for kind in ("xgboost", "ridge", "naive")
#         },
#         "r2_aggregation": "arithmetic_mean_by_horizon_on_common_holdout",
#     }


# def train_one(asset_id):
#     db = get_db()
#     prices, source = price_df(db, asset_id)
#     if prices.empty:
#         return {"asset_id": asset_id, "status": "no_data"}

#     data = build(prices, target=True)
#     common = (data.dropna(subset=FEATURES + [f"target_return_{MAX_HORIZON}d", "close", "date"])
#               .sort_values("date").reset_index(drop=True))
#     min_rows = int(getattr(settings, "model_min_rows", 30))
#     if len(common) < max(min_rows, 900):
#         return {"asset_id": asset_id, "status": "insufficient_history", "rows": len(common)}

#     test_rows = max(MIN_TEST_ROWS, int(round(len(common) * TEST_FRACTION)))
#     development_end = len(common) - test_rows
#     metrics_by_horizon = {}
#     horizon_models = {}

#     for horizon in HORIZONS:
#         folds = _walk_forward_splits(development_end, horizon)
#         if not folds:
#             metrics_by_horizon[str(horizon)] = {"status": "insufficient_walk_forward_history"}
#             continue
#         result = _evaluate_horizon(data, common, horizon, development_end, folds)
#         horizon_models[str(horizon)] = {
#             "model_raw_json": result["model_raw_json"],
#             "feature_columns": result["feature_columns"],
#         }
#         metrics_by_horizon[str(horizon)] = {
#             key: value for key, value in result.items() if key != "model_raw_json"
#         }
#         r2_text = f"{result['r2_price']:.3f}" if result["r2_price"] is not None else "N/A"
#         direction_text = (f"{result['directional_accuracy_pct']:.1f}%"
#                           if result["directional_accuracy_pct"] is not None else "N/A")
#         print(f"   {horizon:>3} sesiones | {result['selected_model']:<7} | "
#               f"MAE {result['mae_price']:.4f} | WAPE {result['wape_pct']:.2f}% | "
#               f"R² precio {r2_text} | Dirección {direction_text} | "
#               f"WF {result['validation_folds']} folds | "
#               f"ventana {result['selected_training_years'] or 0} años")

#     if len(horizon_models) != len(HORIZONS):
#         return {"asset_id": asset_id, "status": "incomplete_horizons",
#                 "metrics_by_horizon": metrics_by_horizon}

#     now = datetime.now(timezone.utc)
#     model_doc = {
#         "asset_id": asset_id,
#         "model_type": "HybridRegressor_multi_horizon",
#         "feature_columns": FEATURES,
#         "horizons": [int(h) for h in horizon_models],
#         "horizon_models": horizon_models,
#         "trained_at": now,
#         "history_source": source,
#     }
#     db.models.update_one({"asset_id": asset_id}, {"$set": model_doc}, upsert=True)

#     metric_doc = {
#         "asset_id": asset_id,
#         "trained_at": now,
#         "history_source": source,
#         "evaluation_method": "common_holdout_purged_walk_forward_model_selection",
#         "metrics_by_horizon": metrics_by_horizon,
#         "max_trained_horizon": MAX_HORIZON,
#         "feature_columns": FEATURES,
#         "summary": _metric_summary(metrics_by_horizon),
#     }
#     db.model_metrics.insert_one(metric_doc)
#     return {"asset_id": asset_id, "status": "trained", "trained_at": now,
#             "metrics_by_horizon": metrics_by_horizon, "summary": metric_doc["summary"],
#             "max_trained_horizon": MAX_HORIZON}


# def _save_general_metrics(db, results):
#     trained = [item for item in results if item.get("status") == "trained"]
#     horizons = {}
#     for horizon in HORIZONS:
#         metrics = [item["metrics_by_horizon"][str(horizon)] for item in trained]
#         if not metrics:
#             continue
#         horizons[str(horizon)] = {
#             "commodities_evaluated": len(metrics),
#             "mean_wape_pct": _mean_valid([m["wape_pct"] for m in metrics]),
#             "mean_r2_price": _mean_valid([m["r2_price"] for m in metrics]),
#             "mean_r2_return": _mean_valid([m["r2_return"] for m in metrics]),
#             "mean_directional_accuracy_pct": _mean_valid([m["directional_accuracy_pct"] for m in metrics]),
#             "directional_commodities_evaluated": sum(m["directional_evaluated"] for m in metrics),
#             "selected_model_counts": {
#                 kind: sum(m["selected_model"] == kind for m in metrics)
#                 for kind in ("xgboost", "ridge", "naive")
#             },
#         }

#     summaries = [item["summary"] for item in trained]
#     overall = {
#         "commodities_evaluated": len(trained),
#         "commodities_expected": len(AGRICULTURAL_COMMODITIES),
#         "mean_wape_pct": _mean_valid([s["mean_wape_pct"] for s in summaries]),
#         "mean_r2_price": _mean_valid([s["r2_price"] for s in summaries]),
#         "mean_r2_return": _mean_valid([s["r2_return"] for s in summaries]),
#         "mean_directional_accuracy_pct": _mean_valid([s["directional_accuracy_pct"] for s in summaries]),
#         "mean_directional_coverage_pct": _mean_valid([s["directional_coverage_pct"] for s in summaries]),
#         "selected_model_counts": {
#             kind: sum(s["selected_model_counts"][kind] for s in summaries)
#             for kind in ("xgboost", "ridge", "naive")
#         },
#         "r2_aggregation": "macro_average_of_per_commodity_horizon_r2_price",
#     }
#     document = {
#         "asset_id": "__general__", "trained_at": datetime.now(timezone.utc),
#         "evaluation_method": "macro_average_of_selected_model_holdouts",
#         "metrics_by_horizon": horizons, "summary": overall,
#     }
#     db.model_metrics.insert_one(document)
#     return document


# def _predict_from_binary(blob, feature_frame):
#     raw = bytes(blob)
#     if raw.startswith(b'{"kind":"naive"'):
#         return 0.0
#     if raw.startswith(b'{"kind":"ridge"'):
#         params = json.loads(raw.decode("utf-8"))
#         X = feature_frame.to_numpy(dtype=float).reshape(-1)
#         value = np.dot((X - np.asarray(params["mean"])) / np.asarray(params["scale"]),
#                        np.asarray(params["coef"])) + float(params["intercept"])
#         return float(value)
#     booster = Booster()
#     booster.load_model(bytearray(raw))
#     return float(booster.predict(DMatrix(feature_frame, feature_names=list(feature_frame.columns)))[0])


# def predict_one(asset_id):
#     db = get_db()
#     model_doc = db.models.find_one({"asset_id": asset_id})
#     if not model_doc:
#         return {"asset_id": asset_id, "status": "no_model"}
#     horizon_models = model_doc.get("horizon_models") or {}
#     if not all(str(horizon) in horizon_models for horizon in HORIZONS):
#         return {"asset_id": asset_id, "status": "incomplete_horizons_retrain_required"}

#     prices, source = price_df(db, asset_id)
#     if prices.empty:
#         return {"asset_id": asset_id, "status": "no_prices"}
#     data = build(prices, target=False)
#     required = sorted({name for model in horizon_models.values() for name in model.get("feature_columns", [])})
#     valid = data.dropna(subset=required + ["date", "close"])
#     if valid.empty:
#         return {"asset_id": asset_id, "status": "insufficient_features"}

#     last = valid.iloc[-1]
#     latest_data_date = pd.Timestamp(last["date"]).to_pydatetime()
#     if (datetime.now(timezone.utc) - latest_data_date).total_seconds() > 10 * 86400:
#         return {"asset_id": asset_id, "status": "stale_market_data"}
#     current_price = float(last["close"])
#     if not np.isfinite(current_price) or current_price <= 0:
#         return {"asset_id": asset_id, "status": "invalid_current_price"}

#     anchors = [(0.0, current_price)]
#     used_horizons = []
#     for horizon in HORIZONS:
#         payload = horizon_models[str(horizon)]
#         cols = payload.get("feature_columns", [])
#         frame = pd.DataFrame([{col: float(last[col]) for col in cols}], columns=cols)
#         predicted_return = _predict_from_binary(payload["model_raw_json"], frame)
#         if not np.isfinite(predicted_return):
#             return {"asset_id": asset_id, "status": "invalid_model_prediction", "horizon": horizon}
#         predicted_price = current_price * float(np.exp(np.clip(predicted_return, -MAX_PRICE_LOG_RETURN, MAX_PRICE_LOG_RETURN)))
#         calendar_day = horizon * FORECAST_CALENDAR_DAYS / MAX_HORIZON
#         anchors.append((float(calendar_day), predicted_price))
#         used_horizons.append(int(horizon))

#     anchors.sort(key=lambda item: item[0])
#     x = np.asarray([item[0] for item in anchors], dtype=float)
#     log_y = np.log(np.asarray([item[1] for item in anchors], dtype=float))
#     daily_prices = np.exp(np.interp(np.arange(1, FORECAST_CALENDAR_DAYS + 1), x, log_y))

#     now = datetime.now(timezone.utc)
#     daily_forecasts = []
#     for day, value in enumerate(daily_prices, start=1):
#         target_date = latest_data_date + timedelta(days=day)
#         change_pct = (float(value) / current_price - 1.0) * 100.0
#         daily_forecasts.append({"day": day, "target_time": target_date,
#                                 "predicted_close": float(value), "change_pct": float(change_pct)})

#     final_price = float(daily_forecasts[-1]["predicted_close"])
#     change_pct = (final_price / current_price - 1.0) * 100.0
#     threshold = float(getattr(settings, "prediction_neutral_threshold_pct", 2.5))
#     trend = "ALCISTA" if change_pct > threshold else ("BAJISTA" if change_pct < -threshold else "ESTABLE")
#     maximum = max(daily_forecasts, key=lambda item: item["predicted_close"])
#     minimum = min(daily_forecasts, key=lambda item: item["predicted_close"])

#     document = {
#         "asset_id": asset_id,
#         "predicted_at": now,
#         "current_price": current_price,
#         "predictions_180d": daily_forecasts,
#         "predicted_close_180d": final_price,
#         "predicted_change_pct": float(change_pct),
#         "trend": trend,
#         "forecast_method": "direct_multi_horizon_log_return_interpolation",
#         "trained_horizons_sessions": sorted(used_horizons),
#         "forecast_note": "Los puntos de anclaje son pronósticos directos de modelos seleccionados por horizonte; "
#                          "los valores diarios intermedios se interpolan para visualización.",
#         "agent_summary": {
#             "max_peak": {"day": maximum["day"], "date": maximum["target_time"],
#                          "price": maximum["predicted_close"], "change_pct": maximum["change_pct"]},
#             "min_valley": {"day": minimum["day"], "date": minimum["target_time"],
#                            "price": minimum["predicted_close"], "change_pct": minimum["change_pct"]},
#         },
#         "history_source": source,
#         "model_trained_at": model_doc.get("trained_at"),
#         "sentiment_compound": 0.0,
#     }
#     db.predictions.update_one({"asset_id": asset_id}, {"$set": document}, upsert=True)
#     return {"status": "predicted", **document}


# def train_all():
#     results = []
#     print("\n" + "=" * 60)
#     print("ENTRENAMIENTO MULTIMODELO + WALK-FORWARD POR COMMODITY")
#     print("=" * 60)
#     for asset_id in AGRICULTURAL_COMMODITIES:
#         print(f"\nEntrenando: [{asset_id}]")
#         try:
#             result = train_one(asset_id)
#         except Exception as exc:
#             print(f"   ERROR: {exc}")
#             result = {"asset_id": asset_id, "status": "error", "message": str(exc)}
#         print(f"   Estado: {result['status']}")
#         results.append(result)
#     general = _save_general_metrics(get_db(), results)
#     print("\nResumen general:", general["summary"])
#     return results


# def predict_all():
#     results = []
#     print("\nGenerando proyecciones...")
#     for asset_id in AGRICULTURAL_COMMODITIES:
#         try:
#             result = predict_one(asset_id)
#         except Exception as exc:
#             result = {"asset_id": asset_id, "status": "error", "message": str(exc)}
#         print(f"   {asset_id}: {result.get('status')}")
#         results.append(result)
#     return results


# if __name__ == "__main__":
#     train_all()
#     predict_all()























# """Entrenamiento, evaluación y predicción de commodities con XGBoost.

# - Optimización por commodity y horizonte con validación Walk-Forward.
# - Prueba final cronológica fuera de muestra, aislada de la optimización.
# - Cuatro métricas principales: MAE, WAPE, R² (retornos) y dirección.
# - Sin noticias/sentimiento como características predictoras.
# - Conserva las estructuras de `models` y `predictions` de la versión previa.
# """

# from datetime import datetime, timedelta, timezone

# import numpy as np
# import pandas as pd
# from bson.binary import Binary
# from sklearn.metrics import mean_absolute_error, r2_score
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import Ridge
# from xgboost import Booster, DMatrix, XGBRegressor

# from src.config import settings
# from src.db import get_db
# from src.ml.features import FEATURES, HORIZONS, build, features_for_horizon
# from src.api_clients.yfinance_client import AGRICULTURAL_COMMODITIES

# MAX_HORIZON = max(HORIZONS)
# FORECAST_CALENDAR_DAYS = 180
# RANDOM_STATE = 42
# TEST_FRACTION = 0.15  # Prueba final sin contacto con búsqueda de parámetros.
# MAX_VALIDATION_FOLDS = 2
# TRADING_SESSIONS_PER_YEAR = 252
# TRAINING_WINDOWS_YEARS = (3, 5, 10)
# RIDGE_ALPHA = 10.0
# MIN_VALIDATION_ROWS = 25
# MIN_TEST_ROWS = 30
# MIN_TRAIN_ROWS = 120

# # Búsqueda controlada (sin dependencias nuevas). Incluir el modelo original
# # permite constatar si una configuración distinta realmente mejora al existente.
# PARAMETER_CANDIDATES = [
#     {"n_estimators": 350, "max_depth": 3, "learning_rate": 0.025,
#      "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8,
#      "reg_alpha": 0.05, "reg_lambda": 5.0},
#     {"n_estimators": 220, "max_depth": 2, "learning_rate": 0.035,
#      "min_child_weight": 8, "subsample": 0.9, "colsample_bytree": 0.9,
#      "reg_alpha": 0.1, "reg_lambda": 8.0},
#     {"n_estimators": 450, "max_depth": 3, "learning_rate": 0.018,
#      "min_child_weight": 10, "subsample": 0.85, "colsample_bytree": 0.85,
#      "reg_alpha": 0.2, "reg_lambda": 10.0},
#     {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.025,
#      "min_child_weight": 7, "subsample": 0.75, "colsample_bytree": 0.85,
#      "reg_alpha": 0.15, "reg_lambda": 8.0},
# ]


# def price_df(db, asset_id):
#     rows = list(
#         db.ohlcv.find(
#             {"asset_id": asset_id, "period": "1d"},
#             {
#                 "_id": 0,
#                 "timestamp": 1,
#                 "open": 1,
#                 "high": 1,
#                 "low": 1,
#                 "close": 1,
#                 "volume": 1
#             }
#         ).sort("timestamp", 1)
#     )
#     if not rows:
#         return pd.DataFrame(columns=["date", "close", "volume"]), "none"
#     df = pd.DataFrame(rows).rename(columns={"timestamp": "date"})
#     if "volume" not in df.columns:
#         df["volume"] = 0.0
#     return df, "yfinance_ohlcv"


# def _new_model(params=None):
#     return XGBRegressor(
#         objective="reg:squarederror", random_state=RANDOM_STATE,
#         n_jobs=2, tree_method="hist", **(params or PARAMETER_CANDIDATES[0]),
#     )


# def _metrics(actual_returns, predicted_returns, base_prices):
#     """Indicadores comparables; MAE conserva la unidad del activo.

#     R² se calcula sobre retornos logarítmicos para evitar resultados
#     artificialmente buenos causados por la persistencia del precio.
#     """
#     actual_returns = np.asarray(actual_returns, dtype=float)
#     predicted_returns = np.asarray(predicted_returns, dtype=float)
#     base_prices = np.asarray(base_prices, dtype=float)
#     actual_prices = base_prices * np.exp(actual_returns)
#     predicted_prices = base_prices * np.exp(np.clip(predicted_returns, -1.5, 1.5))

#     absolute_error = np.abs(actual_prices - predicted_prices)
#     denominator = np.abs(actual_prices).sum()
#     wape = 100.0 * absolute_error.sum() / denominator if denominator > 0 else np.nan
#     r2 = r2_score(actual_returns, predicted_returns) if len(actual_returns) >= 2 else np.nan
#     directional = 100.0 * np.mean(np.sign(actual_returns) == np.sign(predicted_returns))
#     naive_mae = mean_absolute_error(actual_prices, base_prices)
#     model_mae = float(np.mean(absolute_error))

#     return {
#         "mae_price": model_mae,
#         "wape_pct": float(wape),
#         "r2_return": float(r2),
#         "directional_accuracy_pct": float(directional),
#         # Solo controles internos; no son parte de las cuatro métricas oficiales.
#         "naive_mae_price": float(naive_mae),
#         "beats_naive": bool(model_mae < naive_mae),
#     }


# def _walk_forward_splits(n_rows, horizon, development_end):
#     """Validación expansiva con purga por horizonte y test aislado.

#     Los índices son sobre filas válidas de orígenes; cada horizonte termina
#     en una fecha diferente porque exige conocer su precio objetivo futuro.
#     """
#     min_train = max(MIN_TRAIN_ROWS, 2 * horizon)
#     available = development_end - min_train - horizon
#     if available < MIN_VALIDATION_ROWS:
#         return []
#     folds = min(MAX_VALIDATION_FOLDS, available // MIN_VALIDATION_ROWS)
#     block = max(MIN_VALIDATION_ROWS, available // folds)
#     first_validation_start = development_end - folds * block
#     splits = []
#     for fold in range(folds):
#         val_start = first_validation_start + fold * block
#         val_end = development_end if fold == folds - 1 else val_start + block
#         train_end = val_start - horizon
#         if train_end >= min_train and val_end > val_start:
#             splits.append((train_end, val_start, val_end))
#     return splits


# def _window_start(train_end, years):
#     return max(0, train_end - years * TRADING_SESSIONS_PER_YEAR)


# def _score_wape(actual_return, pred_return, base_price):
#     actual = base_price * np.exp(actual_return)
#     estimate = base_price * np.exp(np.clip(pred_return, -1.5, 1.5))
#     return float(np.abs(actual - estimate).sum()), float(np.abs(actual).sum())


# def _evaluate_horizon(data, horizon):
#     """Compara ventanas, XGBoost y Ridge en WF; reporta holdout intacto.

#     Ridge es un competidor diagnóstico, no se despliega: así se conserva
#     exactamente el formato Booster en `models.horizon_models`.
#     """
#     target_col = f"target_return_{horizon}d"
#     feature_cols = features_for_horizon(horizon)
#     usable = (data.dropna(subset=feature_cols + [target_col, "close", "date"])
#               .sort_values("date").reset_index(drop=True))
#     n = len(usable)
#     min_rows = int(getattr(settings, "model_min_rows", 30))
#     if n < max(min_rows, MIN_TRAIN_ROWS + horizon + MIN_TEST_ROWS):
#         return {"status": "insufficient_history", "horizon": horizon, "rows": n}

#     test_rows = max(MIN_TEST_ROWS, int(round(n * TEST_FRACTION)))
#     development_end = n - test_rows
#     final_train_end = development_end - horizon
#     folds = _walk_forward_splits(n, horizon, development_end)
#     if final_train_end < max(MIN_TRAIN_ROWS, 2 * horizon) or not folds:
#         return {"status": "insufficient_history_for_walk_forward", "horizon": horizon,
#                 "rows": n, "train_rows": final_train_end, "test_rows": test_rows}

#     X = usable[feature_cols]
#     y = usable[target_col]
#     prices = usable["close"].to_numpy(dtype=float)
#     y_values = y.to_numpy(dtype=float)

#     # Búsqueda en dos etapas para mantener el entrenamiento de 11 activos viable:
#     # 1) comparar ventanas con un XGBoost base;
#     # 2) elegir hiperparámetros sólo dentro de la ventana ganadora.
#     # Nunca usamos los resultados del test final en la selección.
#     ridge_candidates = []
#     window_candidates = []
#     for years in TRAINING_WINDOWS_YEARS:
#         total_error = total_price = 0.0
#         ridge_error = ridge_price = 0.0
#         for train_end, val_start, val_end in folds:
#             start = _window_start(train_end, years)
#             model = _new_model(PARAMETER_CANDIDATES[0])
#             model.fit(X.iloc[start:train_end], y.iloc[start:train_end])
#             err, denom = _score_wape(y_values[val_start:val_end],
#                                      model.predict(X.iloc[val_start:val_end]),
#                                      prices[val_start:val_end])
#             total_error += err
#             total_price += denom
#             ridge = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
#             ridge.fit(X.iloc[start:train_end], y.iloc[start:train_end])
#             ridge_err, ridge_denom = _score_wape(y_values[val_start:val_end],
#                 ridge.predict(X.iloc[val_start:val_end]), prices[val_start:val_end])
#             ridge_error += ridge_err
#             ridge_price += ridge_denom
#         window_candidates.append((100.0 * total_error / total_price if total_price else np.inf,
#                                   years))
#         ridge_candidates.append((100.0 * ridge_error / ridge_price if ridge_price else np.inf,
#                                  years))

#     _, best_window = min(window_candidates, key=lambda x: x[0])
#     candidates = []
#     for params_index, params in enumerate(PARAMETER_CANDIDATES):
#         total_error = total_price = 0.0
#         for train_end, val_start, val_end in folds:
#             start = _window_start(train_end, best_window)
#             model = _new_model(params)
#             model.fit(X.iloc[start:train_end], y.iloc[start:train_end])
#             err, denom = _score_wape(y_values[val_start:val_end],
#                 model.predict(X.iloc[val_start:val_end]), prices[val_start:val_end])
#             total_error += err
#             total_price += denom
#         candidates.append((100.0 * total_error / total_price if total_price else np.inf,
#                            params_index))
#     validation_wape, best_params_index = min(candidates, key=lambda x: x[0])
#     best_params = PARAMETER_CANDIDATES[best_params_index]
#     ridge_validation_wape, ridge_window = min(ridge_candidates, key=lambda x: x[0])

#     fold_metrics = []
#     for fold_index, (train_end, val_start, val_end) in enumerate(folds, start=1):
#         start = _window_start(train_end, best_window)
#         model = _new_model(best_params)
#         model.fit(X.iloc[start:train_end], y.iloc[start:train_end])
#         values = _metrics(y_values[val_start:val_end],
#                           model.predict(X.iloc[val_start:val_end]),
#                           prices[val_start:val_end])
#         fold_metrics.append({
#             "fold": fold_index,
#             "train_rows": train_end - start,
#             "validation_rows": val_end - val_start,
#             "train_start": usable["date"].iloc[start].to_pydatetime(),
#             "train_end": usable["date"].iloc[train_end - 1].to_pydatetime(),
#             "validation_start": usable["date"].iloc[val_start].to_pydatetime(),
#             "validation_end": usable["date"].iloc[val_end - 1].to_pydatetime(),
#             "target_end": data["date"].dropna().sort_values().iloc[
#                 data["date"].dropna().sort_values().searchsorted(usable["date"].iloc[val_end - 1]) + horizon
#             ].to_pydatetime() if (data["date"].dropna().sort_values().searchsorted(
#                 usable["date"].iloc[val_end - 1]) + horizon) < len(data["date"].dropna()) else None,
#             **values,
#         })

#     train_start = _window_start(final_train_end, best_window)
#     test_model = _new_model(best_params)
#     test_model.fit(X.iloc[train_start:final_train_end], y.iloc[train_start:final_train_end])
#     test_predictions = test_model.predict(X.iloc[development_end:])
#     actual = y_values[development_end:]
#     base = prices[development_end:]
#     metrics = _metrics(actual, test_predictions, base)

#     # Desafiante lineal, probado en el MISMO test, elegido sólo en validación.
#     ridge_train_start = _window_start(final_train_end, ridge_window)
#     challenger = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
#     challenger.fit(X.iloc[ridge_train_start:final_train_end],
#                    y.iloc[ridge_train_start:final_train_end])
#     ridge_metrics = _metrics(actual, challenger.predict(X.iloc[development_end:]), base)
#     naive_metrics = _metrics(actual, np.zeros_like(actual), base)

#     # Regla explícita: el despliegue sigue siendo XGBoost incluso si otro
#     # método gana. Se registra la comparación sin alterar `models`.
#     final_start = _window_start(n, best_window)
#     final_model = _new_model(best_params)
#     final_model.fit(X.iloc[final_start:], y.iloc[final_start:])
#     raw = bytes(final_model.get_booster().save_raw(raw_format="json"))

#     # Los targets del último origen de test existen en `data` aunque
#     # se hayan eliminado las últimas h filas de `usable`.
#     full_dates = data["date"].dropna().sort_values().reset_index(drop=True)
#     last_origin = pd.Timestamp(usable["date"].iloc[-1])
#     future_dates = full_dates[full_dates > last_origin]
#     target_end = (future_dates.iloc[horizon - 1].to_pydatetime()
#                   if len(future_dates) >= horizon else None)

#     return {
#         "status": "trained", "horizon": int(horizon), "rows": n,
#         "training_rows": int(final_train_end - train_start),
#         "test_rows": int(test_rows), "validation_folds": len(folds),
#         "validation_wape_pct": float(validation_wape),
#         "selected_training_years": int(best_window),
#         "feature_columns": feature_cols,
#         "best_params": best_params,
#         "ridge_validation_wape_pct": float(ridge_validation_wape),
#         "ridge_training_years": int(ridge_window),
#         "ridge_test_metrics": ridge_metrics,
#         "naive_test_metrics": naive_metrics,
#         "recommendation_by_validation": min(
#             [(validation_wape, "xgboost"), (ridge_validation_wape, "ridge"),
#              (float(np.mean([_metrics(
#                  y_values[vs:ve], np.zeros(ve-vs), prices[vs:ve])["wape_pct"]
#                  for _, vs, ve in folds])), "naive")], key=lambda x: x[0])[1],
#         "train_start": usable["date"].iloc[train_start].to_pydatetime(),
#         "train_end": usable["date"].iloc[final_train_end - 1].to_pydatetime(),
#         "test_start": usable["date"].iloc[development_end].to_pydatetime(),
#         "test_end": usable["date"].iloc[-1].to_pydatetime(),
#         "last_test_target_date": target_end,
#         "fold_metrics": fold_metrics,
#         **metrics,
#         "model_raw_json": Binary(raw),
#     }


# def _metric_summary(metrics_by_horizon):
#     """Promedio por horizontes válidos, sin mezclar los MAE de insumos distintos."""
#     values = [m for m in metrics_by_horizon.values() if m.get("status") == "trained"]
#     if not values:
#         return {"trained_horizons": 0}
#     result = {"trained_horizons": len(values)}
#     for field in ("wape_pct", "r2_return", "directional_accuracy_pct"):
#         arr = [float(v[field]) for v in values if np.isfinite(v.get(field, np.nan))]
#         result[f"mean_{field}"] = float(np.mean(arr)) if arr else None
#     result["horizons_beating_naive"] = sum(bool(v["beats_naive"]) for v in values)
#     return result


# def train_one(asset_id):
#     db = get_db()
#     prices, source = price_df(db, asset_id)
#     if prices.empty:
#         return {"asset_id": asset_id, "status": "no_data"}

#     data = build(prices, target=True)
#     if data.empty:
#         return {"asset_id": asset_id, "status": "insufficient_history"}

#     horizon_models = {}
#     metrics_by_horizon = {}
#     for horizon in HORIZONS:
#         result = _evaluate_horizon(data, horizon)
#         if result.get("status") != "trained":
#             metrics_by_horizon[str(horizon)] = result
#             print(f"   {horizon:>3} sesiones: {result['status']}")
#             continue
#         # No cambia la estructura interna de los modelos guardados.
#         horizon_models[str(horizon)] = {
#             "model_raw_json": result["model_raw_json"],
#             "feature_columns": result["feature_columns"],
#         }
#         metrics_by_horizon[str(horizon)] = {
#             k: v for k, v in result.items() if k != "model_raw_json"
#         }
#         print(f"   {horizon:>3} sesiones | MAE {result['mae_price']:.4f} | "
#               f"WAPE {result['wape_pct']:.2f}% | "
#               f"R² {result['r2_return']:.3f} | "
#               f"Dirección {result['directional_accuracy_pct']:.1f}% | "
#               f"WF {result['validation_folds']} folds | "
#               f"ventana {result['selected_training_years']} años | "
#               f"Ridge WAPE {result['ridge_test_metrics']['wape_pct']:.2f}% | "
#               f"Naive WAPE {result['naive_test_metrics']['wape_pct']:.2f}%")

#     # Preservar una predicción completa de 180 días: si falta el horizonte
#     # final, no reemplazar el modelo anterior por una versión incompleta.
#     if str(MAX_HORIZON) not in horizon_models:
#         return {"asset_id": asset_id, "status": "insufficient_history_for_full_forecast",
#                 "metrics_by_horizon": metrics_by_horizon}

#     now = datetime.now(timezone.utc)
#     model_doc = {
#         "asset_id": asset_id,
#         "model_type": "XGBRegressor_multi_horizon",
#         "feature_columns": FEATURES,
#         "horizons": [int(key) for key in horizon_models],
#         "horizon_models": horizon_models,
#         "trained_at": now,
#         "history_source": source,
#     }
#     db.models.update_one({"asset_id": asset_id}, {"$set": model_doc}, upsert=True)

#     # Única colección cuyo formato cambia. Se guarda historial por ejecución.
#     db.model_metrics.insert_one({
#         "asset_id": asset_id,
#         "trained_at": now,
#         "history_source": source,
#         "evaluation_method": "expanding_walk_forward_with_untouched_test",
#         "metrics_by_horizon": metrics_by_horizon,
#         "max_trained_horizon": max(int(key) for key in horizon_models),
#         "feature_columns": FEATURES,
#         "summary": _metric_summary(metrics_by_horizon),
#     })

#     longest = metrics_by_horizon[str(MAX_HORIZON)]
#     return {
#         "asset_id": asset_id,
#         "status": "trained",
#         "training_rows": longest["training_rows"],
#         "test_rows": longest["test_rows"],
#         "metrics_by_horizon": metrics_by_horizon,
#         "max_trained_horizon": MAX_HORIZON,
#         "beats_naive": bool(longest["beats_naive"]),
#     }


# def _save_general_metrics(db, results):
#     """Cada commodity aporta el mismo peso al resumen general."""
#     trained = [r for r in results if r.get("status") == "trained"]
#     by_horizon = {}
#     for horizon in HORIZONS:
#         records = [r["metrics_by_horizon"][str(horizon)] for r in trained
#                    if r["metrics_by_horizon"].get(str(horizon), {}).get("status") == "trained"]
#         if records:
#             by_horizon[str(horizon)] = {
#                 "commodities_evaluated": len(records),
#                 "mean_wape_pct": float(np.mean([r["wape_pct"] for r in records])),
#                 "mean_r2_return": float(np.mean([r["r2_return"] for r in records])),
#                 "mean_directional_accuracy_pct": float(np.mean(
#                     [r["directional_accuracy_pct"] for r in records])),
#             }

#     per_asset = [_metric_summary(r["metrics_by_horizon"]) for r in trained]
#     overall = {"commodities_evaluated": len(trained),
#                "commodities_expected": len(AGRICULTURAL_COMMODITIES)}
#     for field in ("wape_pct", "r2_return", "directional_accuracy_pct"):
#         key = f"mean_{field}"
#         valid = [v[key] for v in per_asset if v.get(key) is not None]
#         overall[key] = float(np.mean(valid)) if valid else None

#     doc = {"asset_id": "__general__", "trained_at": datetime.now(timezone.utc),
#            "evaluation_method": "macro_average_of_individual_holdout_metrics",
#            "metrics_by_horizon": by_horizon, "summary": overall}
#     db.model_metrics.insert_one(doc)
#     return doc


# def predict_one(asset_id):
#     db = get_db()
#     model_doc = db.models.find_one({"asset_id": asset_id})
#     if not model_doc:
#         return {"asset_id": asset_id, "status": "no_model"}
#     if not model_doc.get("horizon_models"):
#         return {"asset_id": asset_id, "status": "retrain_required",
#                 "message": "El modelo guardado usa un formato anterior. Reentrena."}
#     if str(MAX_HORIZON) not in model_doc["horizon_models"]:
#         return {"asset_id": asset_id, "status": "incomplete_horizons_retrain_required"}

#     prices, source = price_df(db, asset_id)
#     if prices.empty:
#         return {"asset_id": asset_id, "status": "no_prices"}
#     data = build(prices, target=False)
#     valid = data.dropna(subset=FEATURES)
#     if valid.empty:
#         return {"asset_id": asset_id, "status": "insufficient_features"}

#     last_row = valid.iloc[-1]
#     current_price = float(last_row["close"])
#     if not np.isfinite(current_price) or current_price <= 0:
#         return {"asset_id": asset_id, "status": "invalid_current_price"}
#     anchors = [(0.0, current_price)]
#     used_horizons = []

#     for horizon_text, payload in model_doc["horizon_models"].items():
#         horizon = int(horizon_text)
#         model_features = payload.get("feature_columns", model_doc.get("feature_columns", FEATURES))
#         if any(col not in data.columns for col in model_features):
#             return {"asset_id": asset_id, "status": "retrain_required_features_changed"}
#         frame = pd.DataFrame([{name: float(last_row[name]) for name in model_features}],
#                              columns=model_features)
#         booster = Booster()
#         booster.load_model(bytearray(payload["model_raw_json"]))
#         predicted_return = float(booster.predict(DMatrix(frame, feature_names=model_features))[0])
#         if not np.isfinite(predicted_return):
#             continue
#         predicted_return = float(np.clip(predicted_return, -1.5, 1.5))
#         predicted_price = current_price * float(np.exp(predicted_return))
#         # Mantiene la misma representación de 180 días del dashboard actual.
#         # Conversión aproximada: 126 sesiones -> 180 días calendario.
#         calendar_day = min(FORECAST_CALENDAR_DAYS,
#                            horizon * FORECAST_CALENDAR_DAYS / MAX_HORIZON)
#         anchors.append((calendar_day, predicted_price))
#         used_horizons.append(horizon)

#     if MAX_HORIZON not in used_horizons:
#         return {"asset_id": asset_id, "status": "missing_final_horizon_prediction"}

#     anchors.sort(key=lambda item: item[0])
#     x = np.asarray([point[0] for point in anchors], dtype=float)
#     y = np.log(np.asarray([point[1] for point in anchors], dtype=float))
#     daily_prices = np.exp(np.interp(np.arange(1, FORECAST_CALENDAR_DAYS + 1), x, y))

#     now = datetime.now(timezone.utc)
#     predictions_horizon = []
#     for day, predicted_price in enumerate(daily_prices, start=1):
#         target_date = now + timedelta(days=day)
#         change_pct = (float(predicted_price) / current_price - 1.0) * 100.0
#         predictions_horizon.append({
#             "day": day, "target_time": target_date,
#             "predicted_close": float(predicted_price), "change_pct": float(change_pct),
#         })

#     final_pred = float(predictions_horizon[-1]["predicted_close"])
#     pct = (final_pred / current_price - 1.0) * 100.0
#     threshold = float(getattr(settings, "prediction_neutral_threshold_pct", 2.5))
#     trend = "ALCISTA" if pct > threshold else ("BAJISTA" if pct < -threshold else "ESTABLE")
#     max_item = max(predictions_horizon, key=lambda item: item["predicted_close"])
#     min_item = min(predictions_horizon, key=lambda item: item["predicted_close"])

#     # Mantener todos los campos existentes, incluidos los consumidos por agentes.
#     doc = {
#         "asset_id": asset_id,
#         "predicted_at": now,
#         "current_price": current_price,
#         "predictions_180d": predictions_horizon,
#         "predicted_close_180d": final_pred,
#         "predicted_change_pct": float(pct),
#         "trend": trend,
#         "forecast_method": "direct_multi_horizon_log_return_interpolation",
#         "trained_horizons_sessions": sorted(used_horizons),
#         "forecast_note": "Los puntos de anclaje son predicciones directas por horizonte; "
#                          "los valores diarios intermedios se interpolan para visualización.",
#         "agent_summary": {
#             "max_peak": {"day": int(max_item["day"]), "date": max_item["target_time"],
#                          "price": float(max_item["predicted_close"]),
#                          "change_pct": float(max_item["change_pct"])},
#             "min_valley": {"day": int(min_item["day"]), "date": min_item["target_time"],
#                            "price": float(min_item["predicted_close"]),
#                            "change_pct": float(min_item["change_pct"])},
#         },
#         "history_source": source,
#         "model_trained_at": model_doc.get("trained_at"),
#         "sentiment_compound": 0.0,  # Campo conservado por compatibilidad; no interviene en XGBoost.
#     }
#     db.predictions.update_one({"asset_id": asset_id}, {"$set": doc}, upsert=True)
#     return {"status": "predicted", **doc}


# def train_all():
#     results = []
#     print("\n" + "=" * 60)
#     print("ENTRENAMIENTO XGBOOST + WALK-FORWARD POR COMMODITY")
#     print("=" * 60)
#     for asset_id in AGRICULTURAL_COMMODITIES.keys():
#         print(f"\nEntrenando: [{asset_id}]")
#         try:
#             result = train_one(asset_id)
#         except Exception as exc:
#             print(f"   ERROR: {exc}")
#             result = {"asset_id": asset_id, "status": "error", "message": str(exc)}
#         print(f"   Estado: {result['status']}")
#         results.append(result)
#     general = _save_general_metrics(get_db(), results)
#     print("\nResumen general:", general["summary"])
#     return results


# def predict_all():
#     results = []
#     print("\nGenerando proyecciones...")
#     for asset_id in AGRICULTURAL_COMMODITIES.keys():
#         print(f"Generando predicción para: [{asset_id}]...")
#         try:
#             result = predict_one(asset_id)
#         except Exception as exc:
#             print(f"   ERROR: {exc}")
#             result = {"asset_id": asset_id, "status": "error", "message": str(exc)}
#         print(f"   Estado: {result.get('status')}")
#         results.append(result)
#     print("Proyecciones finalizadas.\n")
#     return results


# if __name__ == "__main__":
#     train_all()
#     predict_all()
#     print("Proceso completo finalizado.")



















# """Entrenamiento, evaluación y predicción de commodities con XGBoost.

# - Optimización por commodity y horizonte con validación Walk-Forward.
# - Prueba final cronológica fuera de muestra, aislada de la optimización.
# - Cuatro métricas principales: MAE, WAPE, R² (retornos) y dirección.
# - Sin noticias/sentimiento como características predictoras.
# - Conserva las estructuras de `models` y `predictions` de la versión previa.
# """

# from datetime import datetime, timedelta, timezone

# import numpy as np
# import pandas as pd
# from bson.binary import Binary
# from sklearn.metrics import mean_absolute_error, r2_score
# from xgboost import Booster, DMatrix, XGBRegressor

# from src.config import settings
# from src.db import get_db
# from src.ml.features import FEATURES, HORIZONS, build
# from src.api_clients.yfinance_client import AGRICULTURAL_COMMODITIES

# MAX_HORIZON = max(HORIZONS)
# FORECAST_CALENDAR_DAYS = 180
# RANDOM_STATE = 42
# TEST_FRACTION = 0.15  # Prueba final sin contacto con búsqueda de parámetros.
# MAX_VALIDATION_FOLDS = 3
# MIN_VALIDATION_ROWS = 25
# MIN_TEST_ROWS = 30
# MIN_TRAIN_ROWS = 120

# # Búsqueda controlada (sin dependencias nuevas). Incluir el modelo original
# # permite constatar si una configuración distinta realmente mejora al existente.
# PARAMETER_CANDIDATES = [
#     {"n_estimators": 350, "max_depth": 3, "learning_rate": 0.025,
#      "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8,
#      "reg_alpha": 0.05, "reg_lambda": 5.0},
#     {"n_estimators": 220, "max_depth": 2, "learning_rate": 0.035,
#      "min_child_weight": 8, "subsample": 0.9, "colsample_bytree": 0.9,
#      "reg_alpha": 0.1, "reg_lambda": 8.0},
#     {"n_estimators": 450, "max_depth": 3, "learning_rate": 0.018,
#      "min_child_weight": 10, "subsample": 0.85, "colsample_bytree": 0.85,
#      "reg_alpha": 0.2, "reg_lambda": 10.0},
#     {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.025,
#      "min_child_weight": 7, "subsample": 0.75, "colsample_bytree": 0.85,
#      "reg_alpha": 0.15, "reg_lambda": 8.0},
# ]


# def price_df(db, asset_id):
#     rows = list(db.ohlcv.find(
#         {"asset_id": asset_id, "period": "1d"},
#         {"_id": 0, "timestamp": 1, "close": 1, "volume": 1},
#     ).sort("timestamp", 1))
#     if not rows:
#         return pd.DataFrame(columns=["date", "close", "volume"]), "none"
#     df = pd.DataFrame(rows).rename(columns={"timestamp": "date"})
#     if "volume" not in df.columns:
#         df["volume"] = 0.0
#     return df, "yfinance_ohlcv"


# def _new_model(params=None):
#     return XGBRegressor(
#         objective="reg:squarederror", random_state=RANDOM_STATE,
#         n_jobs=2, tree_method="hist", **(params or PARAMETER_CANDIDATES[0]),
#     )


# def _metrics(actual_returns, predicted_returns, base_prices):
#     """Indicadores comparables; MAE conserva la unidad del activo.

#     R² se calcula sobre retornos logarítmicos para evitar resultados
#     artificialmente buenos causados por la persistencia del precio.
#     """
#     actual_returns = np.asarray(actual_returns, dtype=float)
#     predicted_returns = np.asarray(predicted_returns, dtype=float)
#     base_prices = np.asarray(base_prices, dtype=float)
#     actual_prices = base_prices * np.exp(actual_returns)
#     predicted_prices = base_prices * np.exp(np.clip(predicted_returns, -1.5, 1.5))

#     absolute_error = np.abs(actual_prices - predicted_prices)
#     denominator = np.abs(actual_prices).sum()
#     wape = 100.0 * absolute_error.sum() / denominator if denominator > 0 else np.nan
#     r2 = r2_score(actual_returns, predicted_returns) if len(actual_returns) >= 2 else np.nan
#     directional = 100.0 * np.mean(np.sign(actual_returns) == np.sign(predicted_returns))
#     naive_mae = mean_absolute_error(actual_prices, base_prices)
#     model_mae = float(np.mean(absolute_error))

#     return {
#         "mae_price": model_mae,
#         "wape_pct": float(wape),
#         "r2_return": float(r2),
#         "directional_accuracy_pct": float(directional),
#         # Solo controles internos; no son parte de las cuatro métricas oficiales.
#         "naive_mae_price": float(naive_mae),
#         "beats_naive": bool(model_mae < naive_mae),
#     }


# def _walk_forward_splits(n_rows, horizon, development_end):
#     """Folds de ventana expansiva dentro del desarrollo (sin tocar test).

#     Si la fila t es la última fila entrenada, su objetivo se conoce en t+h.
#     Por ello el último origen del entrenamiento es anterior al inicio del
#     bloque validado al menos h posiciones.
#     """
#     min_train = max(MIN_TRAIN_ROWS, 2 * horizon)
#     minimum_block = MIN_VALIDATION_ROWS
#     available = development_end - min_train - horizon
#     if available < minimum_block:
#         return []

#     folds = min(MAX_VALIDATION_FOLDS, available // minimum_block)
#     block = max(minimum_block, available // folds)
#     first_validation_start = development_end - folds * block
#     splits = []
#     for fold in range(folds):
#         start = first_validation_start + fold * block
#         stop = development_end if fold == folds - 1 else start + block
#         train_end = start - horizon
#         if train_end >= min_train and stop > start:
#             splits.append((train_end, start, stop))
#     return splits


# def _evaluate_horizon(data, horizon):
#     """Selecciona parámetros con WF y los evalúa una vez en test final."""
#     target_col = f"target_return_{horizon}d"
#     usable = (data.dropna(subset=FEATURES + [target_col, "close", "date"])
#               .sort_values("date").reset_index(drop=True))
#     n = len(usable)
#     min_rows = int(getattr(settings, "model_min_rows", 30))
#     if n < max(min_rows, MIN_TRAIN_ROWS + horizon + MIN_TEST_ROWS):
#         return {"status": "insufficient_history", "horizon": horizon, "rows": n}

#     test_rows = max(MIN_TEST_ROWS, int(round(n * TEST_FRACTION)))
#     development_end = n - test_rows
#     # Los objetivos de entrenamiento nunca deben incluir cierres del test.
#     final_train_end = development_end - horizon
#     splits = _walk_forward_splits(n, horizon, development_end)
#     if final_train_end < max(MIN_TRAIN_ROWS, 2 * horizon) or not splits:
#         return {"status": "insufficient_history_for_walk_forward", "horizon": horizon,
#                 "rows": n, "train_rows": final_train_end, "test_rows": test_rows}

#     X = usable[FEATURES]
#     y = usable[target_col]
#     candidate_scores = []
#     for params in PARAMETER_CANDIDATES:
#         total_absolute_error = 0.0
#         total_actual_prices = 0.0
#         fold_count = 0
#         for train_end, val_start, val_end in splits:
#             model = _new_model(params)
#             model.fit(X.iloc[:train_end], y.iloc[:train_end])
#             pred = model.predict(X.iloc[val_start:val_end])
#             true_return = y.iloc[val_start:val_end].to_numpy(dtype=float)
#             base_price = usable["close"].iloc[val_start:val_end].to_numpy(dtype=float)
#             actual = base_price * np.exp(true_return)
#             estimate = base_price * np.exp(np.clip(pred, -1.5, 1.5))
#             total_absolute_error += float(np.abs(actual - estimate).sum())
#             total_actual_prices += float(np.abs(actual).sum())
#             fold_count += 1
#         score = (100.0 * total_absolute_error / total_actual_prices
#                  if total_actual_prices > 0 else np.inf)
#         candidate_scores.append((score, fold_count, params))

#     best_score, _, best_params = min(candidate_scores, key=lambda item: item[0])

#     # Repetimos solamente el candidato elegido para conservar las métricas WF.
#     fold_metrics = []
#     for fold_index, (train_end, val_start, val_end) in enumerate(splits, start=1):
#         model = _new_model(best_params)
#         model.fit(X.iloc[:train_end], y.iloc[:train_end])
#         values = _metrics(y.iloc[val_start:val_end].to_numpy(),
#                           model.predict(X.iloc[val_start:val_end]),
#                           usable["close"].iloc[val_start:val_end].to_numpy())
#         fold_metrics.append({
#             "fold": fold_index,
#             "train_rows": train_end,
#             "validation_rows": val_end - val_start,
#             "validation_start": usable["date"].iloc[val_start].to_pydatetime(),
#             "validation_end": usable["date"].iloc[val_end - 1].to_pydatetime(),
#             **values,
#         })

#     # Evaluación final de verdad fuera de muestra: el test no participó
#     # en la selección de parámetros ni en los folds de validación.
#     test_model = _new_model(best_params)
#     test_model.fit(X.iloc[:final_train_end], y.iloc[:final_train_end])
#     test_pred = test_model.predict(X.iloc[development_end:])
#     test_actual = y.iloc[development_end:].to_numpy(dtype=float)
#     test_base = usable["close"].iloc[development_end:].to_numpy(dtype=float)
#     metrics = _metrics(test_actual, test_pred, test_base)

#     # Entrenamiento final, tras medir el test, para generar nuevas predicciones.
#     final_model = _new_model(best_params)
#     final_model.fit(X, y)
#     raw = bytes(final_model.get_booster().save_raw(raw_format="json"))

#     return {
#         "status": "trained",
#         "horizon": int(horizon),
#         "rows": n,
#         "training_rows": int(final_train_end),
#         "test_rows": int(test_rows),
#         "validation_folds": len(splits),
#         "validation_wape_pct": float(best_score),
#         "test_start": usable["date"].iloc[development_end].to_pydatetime(),
#         "test_end": usable["date"].iloc[-1].to_pydatetime(),
#         "best_params": best_params,
#         "fold_metrics": fold_metrics,
#         **metrics,
#         "model_raw_json": Binary(raw),
#     }


# def _metric_summary(metrics_by_horizon):
#     """Promedio por horizontes válidos, sin mezclar los MAE de insumos distintos."""
#     values = [m for m in metrics_by_horizon.values() if m.get("status") == "trained"]
#     if not values:
#         return {"trained_horizons": 0}
#     result = {"trained_horizons": len(values)}
#     for field in ("wape_pct", "r2_return", "directional_accuracy_pct"):
#         arr = [float(v[field]) for v in values if np.isfinite(v.get(field, np.nan))]
#         result[f"mean_{field}"] = float(np.mean(arr)) if arr else None
#     result["horizons_beating_naive"] = sum(bool(v["beats_naive"]) for v in values)
#     return result


# def train_one(asset_id):
#     db = get_db()
#     prices, source = price_df(db, asset_id)
#     if prices.empty:
#         return {"asset_id": asset_id, "status": "no_data"}

#     data = build(prices, target=True)
#     if data.empty:
#         return {"asset_id": asset_id, "status": "insufficient_history"}

#     horizon_models = {}
#     metrics_by_horizon = {}
#     for horizon in HORIZONS:
#         result = _evaluate_horizon(data, horizon)
#         if result.get("status") != "trained":
#             metrics_by_horizon[str(horizon)] = result
#             print(f"   {horizon:>3} sesiones: {result['status']}")
#             continue
#         # No cambia la estructura interna de los modelos guardados.
#         horizon_models[str(horizon)] = {
#             "model_raw_json": result["model_raw_json"],
#             "feature_columns": FEATURES,
#         }
#         metrics_by_horizon[str(horizon)] = {
#             k: v for k, v in result.items() if k != "model_raw_json"
#         }
#         print(f"   {horizon:>3} sesiones | MAE {result['mae_price']:.4f} | "
#               f"WAPE {result['wape_pct']:.2f}% | "
#               f"R² {result['r2_return']:.3f} | "
#               f"Dirección {result['directional_accuracy_pct']:.1f}% | "
#               f"WF {result['validation_folds']} folds")

#     # Preservar una predicción completa de 180 días: si falta el horizonte
#     # final, no reemplazar el modelo anterior por una versión incompleta.
#     if str(MAX_HORIZON) not in horizon_models:
#         return {"asset_id": asset_id, "status": "insufficient_history_for_full_forecast",
#                 "metrics_by_horizon": metrics_by_horizon}

#     now = datetime.now(timezone.utc)
#     model_doc = {
#         "asset_id": asset_id,
#         "model_type": "XGBRegressor_multi_horizon",
#         "feature_columns": FEATURES,
#         "horizons": [int(key) for key in horizon_models],
#         "horizon_models": horizon_models,
#         "trained_at": now,
#         "history_source": source,
#     }
#     db.models.update_one({"asset_id": asset_id}, {"$set": model_doc}, upsert=True)

#     # Única colección cuyo formato cambia. Se guarda historial por ejecución.
#     db.model_metrics.insert_one({
#         "asset_id": asset_id,
#         "trained_at": now,
#         "history_source": source,
#         "evaluation_method": "expanding_walk_forward_with_untouched_test",
#         "metrics_by_horizon": metrics_by_horizon,
#         "max_trained_horizon": max(int(key) for key in horizon_models),
#         "feature_columns": FEATURES,
#         "summary": _metric_summary(metrics_by_horizon),
#     })

#     longest = metrics_by_horizon[str(MAX_HORIZON)]
#     return {
#         "asset_id": asset_id,
#         "status": "trained",
#         "training_rows": longest["training_rows"],
#         "test_rows": longest["test_rows"],
#         "metrics_by_horizon": metrics_by_horizon,
#         "max_trained_horizon": MAX_HORIZON,
#         "beats_naive": bool(longest["beats_naive"]),
#     }


# def _save_general_metrics(db, results):
#     """Cada commodity aporta el mismo peso al resumen general."""
#     trained = [r for r in results if r.get("status") == "trained"]
#     by_horizon = {}
#     for horizon in HORIZONS:
#         records = [r["metrics_by_horizon"][str(horizon)] for r in trained
#                    if r["metrics_by_horizon"].get(str(horizon), {}).get("status") == "trained"]
#         if records:
#             by_horizon[str(horizon)] = {
#                 "commodities_evaluated": len(records),
#                 "mean_wape_pct": float(np.mean([r["wape_pct"] for r in records])),
#                 "mean_r2_return": float(np.mean([r["r2_return"] for r in records])),
#                 "mean_directional_accuracy_pct": float(np.mean(
#                     [r["directional_accuracy_pct"] for r in records])),
#             }

#     per_asset = [_metric_summary(r["metrics_by_horizon"]) for r in trained]
#     overall = {"commodities_evaluated": len(trained),
#                "commodities_expected": len(AGRICULTURAL_COMMODITIES)}
#     for field in ("wape_pct", "r2_return", "directional_accuracy_pct"):
#         key = f"mean_{field}"
#         valid = [v[key] for v in per_asset if v.get(key) is not None]
#         overall[key] = float(np.mean(valid)) if valid else None

#     doc = {"asset_id": "__general__", "trained_at": datetime.now(timezone.utc),
#            "evaluation_method": "macro_average_of_individual_holdout_metrics",
#            "metrics_by_horizon": by_horizon, "summary": overall}
#     db.model_metrics.insert_one(doc)
#     return doc


# def predict_one(asset_id):
#     db = get_db()
#     model_doc = db.models.find_one({"asset_id": asset_id})
#     if not model_doc:
#         return {"asset_id": asset_id, "status": "no_model"}
#     if not model_doc.get("horizon_models"):
#         return {"asset_id": asset_id, "status": "retrain_required",
#                 "message": "El modelo guardado usa un formato anterior. Reentrena."}
#     if str(MAX_HORIZON) not in model_doc["horizon_models"]:
#         return {"asset_id": asset_id, "status": "incomplete_horizons_retrain_required"}

#     prices, source = price_df(db, asset_id)
#     if prices.empty:
#         return {"asset_id": asset_id, "status": "no_prices"}
#     data = build(prices, target=False)
#     valid = data.dropna(subset=FEATURES)
#     if valid.empty:
#         return {"asset_id": asset_id, "status": "insufficient_features"}

#     last_row = valid.iloc[-1]
#     current_price = float(last_row["close"])
#     if not np.isfinite(current_price) or current_price <= 0:
#         return {"asset_id": asset_id, "status": "invalid_current_price"}
#     anchors = [(0.0, current_price)]
#     used_horizons = []

#     for horizon_text, payload in model_doc["horizon_models"].items():
#         horizon = int(horizon_text)
#         model_features = payload.get("feature_columns", model_doc.get("feature_columns", FEATURES))
#         if any(col not in data.columns for col in model_features):
#             return {"asset_id": asset_id, "status": "retrain_required_features_changed"}
#         frame = pd.DataFrame([{name: float(last_row[name]) for name in model_features}],
#                              columns=model_features)
#         booster = Booster()
#         booster.load_model(bytearray(payload["model_raw_json"]))
#         predicted_return = float(booster.predict(DMatrix(frame, feature_names=model_features))[0])
#         if not np.isfinite(predicted_return):
#             continue
#         predicted_return = float(np.clip(predicted_return, -1.5, 1.5))
#         predicted_price = current_price * float(np.exp(predicted_return))
#         # Mantiene la misma representación de 180 días del dashboard actual.
#         # Conversión aproximada: 126 sesiones -> 180 días calendario.
#         calendar_day = min(FORECAST_CALENDAR_DAYS,
#                            horizon * FORECAST_CALENDAR_DAYS / MAX_HORIZON)
#         anchors.append((calendar_day, predicted_price))
#         used_horizons.append(horizon)

#     if MAX_HORIZON not in used_horizons:
#         return {"asset_id": asset_id, "status": "missing_final_horizon_prediction"}

#     anchors.sort(key=lambda item: item[0])
#     x = np.asarray([point[0] for point in anchors], dtype=float)
#     y = np.log(np.asarray([point[1] for point in anchors], dtype=float))
#     daily_prices = np.exp(np.interp(np.arange(1, FORECAST_CALENDAR_DAYS + 1), x, y))

#     now = datetime.now(timezone.utc)
#     predictions_horizon = []
#     for day, predicted_price in enumerate(daily_prices, start=1):
#         target_date = now + timedelta(days=day)
#         change_pct = (float(predicted_price) / current_price - 1.0) * 100.0
#         predictions_horizon.append({
#             "day": day, "target_time": target_date,
#             "predicted_close": float(predicted_price), "change_pct": float(change_pct),
#         })

#     final_pred = float(predictions_horizon[-1]["predicted_close"])
#     pct = (final_pred / current_price - 1.0) * 100.0
#     threshold = float(getattr(settings, "prediction_neutral_threshold_pct", 2.5))
#     trend = "ALCISTA" if pct > threshold else ("BAJISTA" if pct < -threshold else "ESTABLE")
#     max_item = max(predictions_horizon, key=lambda item: item["predicted_close"])
#     min_item = min(predictions_horizon, key=lambda item: item["predicted_close"])

#     # Mantener todos los campos existentes, incluidos los consumidos por agentes.
#     doc = {
#         "asset_id": asset_id,
#         "predicted_at": now,
#         "current_price": current_price,
#         "predictions_180d": predictions_horizon,
#         "predicted_close_180d": final_pred,
#         "predicted_change_pct": float(pct),
#         "trend": trend,
#         "forecast_method": "direct_multi_horizon_log_return_interpolation",
#         "trained_horizons_sessions": sorted(used_horizons),
#         "forecast_note": "Los puntos de anclaje son predicciones directas por horizonte; "
#                          "los valores diarios intermedios se interpolan para visualización.",
#         "agent_summary": {
#             "max_peak": {"day": int(max_item["day"]), "date": max_item["target_time"],
#                          "price": float(max_item["predicted_close"]),
#                          "change_pct": float(max_item["change_pct"])},
#             "min_valley": {"day": int(min_item["day"]), "date": min_item["target_time"],
#                            "price": float(min_item["predicted_close"]),
#                            "change_pct": float(min_item["change_pct"])},
#         },
#         "history_source": source,
#         "model_trained_at": model_doc.get("trained_at"),
#         "sentiment_compound": 0.0,  # Campo conservado por compatibilidad; no interviene en XGBoost.
#     }
#     db.predictions.update_one({"asset_id": asset_id}, {"$set": doc}, upsert=True)
#     return {"status": "predicted", **doc}


# def train_all():
#     results = []
#     print("\n" + "=" * 60)
#     print("ENTRENAMIENTO XGBOOST + WALK-FORWARD POR COMMODITY")
#     print("=" * 60)
#     for asset_id in AGRICULTURAL_COMMODITIES.keys():
#         print(f"\nEntrenando: [{asset_id}]")
#         try:
#             result = train_one(asset_id)
#         except Exception as exc:
#             print(f"   ERROR: {exc}")
#             result = {"asset_id": asset_id, "status": "error", "message": str(exc)}
#         print(f"   Estado: {result['status']}")
#         results.append(result)
#     general = _save_general_metrics(get_db(), results)
#     print("\nResumen general:", general["summary"])
#     return results


# def predict_all():
#     results = []
#     print("\nGenerando proyecciones...")
#     for asset_id in AGRICULTURAL_COMMODITIES.keys():
#         print(f"Generando predicción para: [{asset_id}]...")
#         try:
#             result = predict_one(asset_id)
#         except Exception as exc:
#             print(f"   ERROR: {exc}")
#             result = {"asset_id": asset_id, "status": "error", "message": str(exc)}
#         print(f"   Estado: {result.get('status')}")
#         results.append(result)
#     print("Proyecciones finalizadas.\n")
#     return results


# if __name__ == "__main__":
#     train_all()
#     predict_all()
#     print("Proceso completo finalizado.")


















# from datetime import datetime, timedelta, timezone

# import numpy as np
# import pandas as pd
# from bson.binary import Binary
# from sklearn.metrics import (
#     mean_absolute_error,
#     mean_absolute_percentage_error,
#     mean_squared_error,
#     r2_score,
# )
# from xgboost import Booster, DMatrix, XGBRegressor

# from src.config import settings
# from src.db import get_db
# from src.ml.features import FEATURES, HORIZONS, build
# from src.api_clients.yfinance_client import AGRICULTURAL_COMMODITIES

# MAX_HORIZON = max(HORIZONS)
# FORECAST_CALENDAR_DAYS = 180


# def sentiment_df(db, asset_id):
#     pipeline = [
#         {"$match": {"mentioned_asset_ids": asset_id}},
#         {"$project": {
#             "day": {"$dateTrunc": {"date": "$published_at", "unit": "day", "timezone": "UTC"}},
#             "sentiment_compound": 1,
#         }},
#         {"$group": {"_id": "$day", "sentiment_compound": {"$avg": "$sentiment_compound"}}},
#         {"$sort": {"_id": 1}},
#     ]
#     rows = list(db.news.aggregate(pipeline))
#     return pd.DataFrame([
#         {"date": row["_id"], "sentiment_compound": row.get("sentiment_compound", 0.0)}
#         for row in rows
#     ])


# def price_df(db, asset_id):
#     rows = list(db.ohlcv.find(
#         {"asset_id": asset_id, "period": "1d"},
#         {"_id": 0, "timestamp": 1, "close": 1, "volume": 1},
#     ).sort("timestamp", 1))
#     if not rows:
#         return pd.DataFrame(columns=["date", "close", "volume"]), "none"
#     df = pd.DataFrame(rows).rename(columns={"timestamp": "date"})
#     if "volume" not in df.columns:
#         df["volume"] = 0.0
#     return df, "yfinance_ohlcv"


# def _new_model():
#     return XGBRegressor(
#         objective="reg:squarederror",
#         n_estimators=350,
#         max_depth=3,
#         learning_rate=0.025,
#         min_child_weight=5,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         reg_alpha=0.05,
#         reg_lambda=5.0,
#         random_state=42,
#         n_jobs=2,
#     )


# def _evaluate_horizon(data, horizon):
#     """Backtest cronológico con separación (purge) igual al horizonte objetivo."""
#     target_col = f"target_return_{horizon}d"
#     usable = data.dropna(subset=FEATURES + [target_col, "close", "date"]).copy()
#     usable = usable.sort_values("date").reset_index(drop=True)

#     min_rows = int(getattr(settings, "model_min_rows", 30))
#     if len(usable) < max(min_rows, 50):
#         return {"status": "insufficient_history", "rows": len(usable)}

#     split = int(len(usable) * 0.80)
#     test_start = split
#     train_end = max(0, split - horizon)
#     train = usable.iloc[:train_end]
#     test = usable.iloc[test_start:]

#     if len(train) < 30 or len(test) < 10:
#         return {
#             "status": "insufficient_history_for_horizon",
#             "rows": len(usable),
#             "train_rows": len(train),
#             "test_rows": len(test),
#         }

#     model = _new_model()
#     target = target_col
#     model.fit(train[FEATURES], train[target])
#     pred_return = model.predict(test[FEATURES])
#     actual_return = test[target].to_numpy(dtype=float)
#     base_price = test["close"].to_numpy(dtype=float)
#     actual_price = base_price * np.exp(actual_return)
#     predicted_price = base_price * np.exp(pred_return)
#     naive_price = base_price.copy()

#     mae_return = float(mean_absolute_error(actual_return, pred_return))
#     rmse_return = float(np.sqrt(mean_squared_error(actual_return, pred_return)))
#     r2_return = float(r2_score(actual_return, pred_return)) if len(test) > 1 else float("nan")
#     mae_price = float(mean_absolute_error(actual_price, predicted_price))
#     naive_mae_price = float(mean_absolute_error(actual_price, naive_price))
#     mape_price = float(mean_absolute_percentage_error(actual_price, predicted_price) * 100.0)

#     final_model = _new_model()
#     final_model.fit(usable[FEATURES], usable[target])
#     raw = bytes(final_model.get_booster().save_raw(raw_format="json"))

#     return {
#         "status": "trained",
#         "horizon": int(horizon),
#         "rows": int(len(usable)),
#         "training_rows": int(len(train)),
#         "test_rows": int(len(test)),
#         "mae_return": mae_return,
#         "rmse_return": rmse_return,
#         "r2_return": r2_return,
#         "mae_price": mae_price,
#         "naive_mae_price": naive_mae_price,
#         "mape_price_pct": mape_price,
#         "beats_naive": bool(mae_price < naive_mae_price),
#         "feature_importance": {
#             name: float(value)
#             for name, value in zip(FEATURES, final_model.feature_importances_)
#         },
#         "model_raw_json": Binary(raw),
#         "backtest": [
#             {
#                 "date": pd.to_datetime(row_date, utc=True).to_pydatetime(),
#                 "actual_price": float(actual),
#                 "predicted_price": float(predicted),
#                 "baseline_price": float(baseline),
#             }
#             for row_date, actual, predicted, baseline in zip(
#                 test["date"], actual_price, predicted_price, naive_price
#             )
#         ],
#     }

# def train_one(asset_id):
#     db = get_db()
#     prices, source = price_df(db, asset_id)
#     if prices.empty:
#         return {"asset_id": asset_id, "status": "no_data"}

#     features = build(prices, sentiment_df(db, asset_id), target=True)
#     if features.empty:
#         return {"asset_id": asset_id, "status": "insufficient_history"}

#     horizon_models = {}
#     metrics_by_horizon = {}
#     max_horizon_result = None

#     for horizon in HORIZONS:
#         result = _evaluate_horizon(features, horizon)
#         status = result.get("status")
#         if status != "trained":
#             metrics_by_horizon[str(horizon)] = {
#                 key: value for key, value in result.items() if key != "model_raw_json"
#             }
#             continue

#         horizon_models[str(horizon)] = {
#             "model_raw_json": result["model_raw_json"],
#             "feature_columns": FEATURES,
#         }
#         metrics_by_horizon[str(horizon)] = {
#             key: value for key, value in result.items()
#             if key not in ("model_raw_json", "backtest", "feature_importance")
#         }
#         if max_horizon_result is None or horizon > max_horizon_result["horizon"]:
#             max_horizon_result = result

#     if not horizon_models:
#         return {
#             "asset_id": asset_id,
#             "status": "insufficient_history_for_horizons",
#             "metrics_by_horizon": metrics_by_horizon,
#         }

#     now = datetime.now(timezone.utc)
#     model_doc = {
#         "asset_id": asset_id,
#         "model_type": "XGBRegressor_multi_horizon",
#         "feature_columns": FEATURES,
#         "horizons": [int(key) for key in horizon_models],
#         "horizon_models": horizon_models,
#         "trained_at": now,
#         "history_source": source,
#     }
#     db.models.update_one({"asset_id": asset_id}, {"$set": model_doc}, upsert=True)

#     metric_doc = {
#         "asset_id": asset_id,
#         "trained_at": now,
#         "history_source": source,
#         "metrics_by_horizon": metrics_by_horizon,
#         "max_trained_horizon": max(int(key) for key in horizon_models),
#         "feature_columns": FEATURES,
#     }
#     db.model_metrics.insert_one(metric_doc)

#     longest = max_horizon_result
#     print(f"   Horizontes entrenados: {', '.join(horizon_models.keys())} sesiones")
#     if longest:
#         print(f"   Horizonte más largo validado: {longest['horizon']} sesiones")
#         print(f"   MAE precio: {longest['mae_price']:.4f}")
#         print(f"   MAE baseline naive: {longest['naive_mae_price']:.4f}")
#         print(f"   MAPE precio: {longest['mape_price_pct']:.2f}%")
#         print(f"   R² retorno: {longest['r2_return']:.4f}")
#         print(f"   ¿Supera al baseline?: {'SÍ' if longest['beats_naive'] else 'NO'}")

#     return {
#         "asset_id": asset_id,
#         "status": "trained",
#         "training_rows": longest["training_rows"] if longest else 0,
#         "test_rows": longest["test_rows"] if longest else 0,
#         "metrics_by_horizon": metrics_by_horizon,
#         "max_trained_horizon": max(int(key) for key in horizon_models),
#         "beats_naive": bool(longest["beats_naive"]) if longest else False,
#     }

# def predict_one(asset_id):
#     db = get_db()
#     model_doc = db.models.find_one({"asset_id": asset_id})
#     if not model_doc:
#         return {"asset_id": asset_id, "status": "no_model"}
#     if not model_doc.get("horizon_models"):
#         return {"asset_id": asset_id, "status": "retrain_required", "message": "El modelo guardado usa el formato anterior. Ejecuta el entrenamiento actualizado."}

#     prices, source = price_df(db, asset_id)
#     if prices.empty:
#         return {"asset_id": asset_id, "status": "no_prices"}

#     sentiment = sentiment_df(db, asset_id)
#     features = build(prices, sentiment, target=False)
#     valid = features.dropna(subset=FEATURES)
#     if valid.empty:
#         return {"asset_id": asset_id, "status": "insufficient_features"}

#     last_row = valid.iloc[-1]
#     feature_frame = pd.DataFrame([{name: float(last_row[name]) for name in FEATURES}], columns=FEATURES)
#     current_price = float(last_row["close"])
#     if not np.isfinite(current_price) or current_price <= 0:
#         return {"asset_id": asset_id, "status": "invalid_current_price"}

#     anchor_points = [(0.0, current_price)]
#     used_horizons = []
#     for horizon_text, payload in model_doc["horizon_models"].items():
#         horizon = int(horizon_text)
#         booster = Booster()
#         booster.load_model(bytearray(payload["model_raw_json"]))
#         matrix = DMatrix(feature_frame, feature_names=FEATURES)
#         predicted_log_return = float(booster.predict(matrix)[0])
#         if not np.isfinite(predicted_log_return):
#             continue
#         predicted_log_return = float(np.clip(predicted_log_return, -1.5, 1.5))
#         predicted_price = current_price * float(np.exp(predicted_log_return))
#         calendar_day = min(FORECAST_CALENDAR_DAYS, horizon * FORECAST_CALENDAR_DAYS / MAX_HORIZON)
#         anchor_points.append((calendar_day, predicted_price))
#         used_horizons.append(horizon)

#     if not used_horizons:
#         return {"asset_id": asset_id, "status": "no_usable_horizon_models"}

#     anchor_points = sorted(anchor_points, key=lambda item: item[0])
#     x = np.array([point[0] for point in anchor_points], dtype=float)
#     y = np.log(np.array([point[1] for point in anchor_points], dtype=float))
#     daily_prices = np.exp(np.interp(np.arange(1, FORECAST_CALENDAR_DAYS + 1), x, y))

#     now = datetime.now(timezone.utc)
#     predictions_horizon = []
#     for day, predicted_price in enumerate(daily_prices, start=1):
#         target_date = now + timedelta(days=day)
#         change_pct = (float(predicted_price) / current_price - 1.0) * 100.0
#         predictions_horizon.append({
#             "day": day,
#             "target_time": target_date,
#             "predicted_close": float(predicted_price),
#             "change_pct": float(change_pct),
#         })

#     final_pred = predictions_horizon[-1]["predicted_close"]
#     pct = (final_pred / current_price - 1.0) * 100.0
#     threshold = float(getattr(settings, "prediction_neutral_threshold_pct", 2.5))
#     trend = "ALCISTA" if pct > threshold else ("BAJISTA" if pct < -threshold else "ESTABLE")
#     max_item = max(predictions_horizon, key=lambda item: item["predicted_close"])
#     min_item = min(predictions_horizon, key=lambda item: item["predicted_close"])

#     doc = {
#         "asset_id": asset_id,
#         "predicted_at": now,
#         "current_price": current_price,
#         "predictions_180d": predictions_horizon,
#         "predicted_close_180d": float(final_pred),
#         "predicted_change_pct": float(pct),
#         "trend": trend,
#         "forecast_method": "direct_multi_horizon_log_return_interpolation",
#         "trained_horizons_sessions": sorted(used_horizons),
#         "forecast_note": "Los puntos de anclaje son predicciones directas por horizonte; los valores diarios intermedios se interpolan para visualización.",
#         "agent_summary": {
#             "max_peak": {
#                 "day": int(max_item["day"]),
#                 "date": max_item["target_time"],
#                 "price": float(max_item["predicted_close"]),
#                 "change_pct": float(max_item["change_pct"]),
#             },
#             "min_valley": {
#                 "day": int(min_item["day"]),
#                 "date": min_item["target_time"],
#                 "price": float(min_item["predicted_close"]),
#                 "change_pct": float(min_item["change_pct"]),
#             },
#         },
#         "history_source": source,
#         "model_trained_at": model_doc.get("trained_at"),
#         "sentiment_compound": float(last_row.get("sentiment_compound", 0.0)),
#     }
#     db.predictions.update_one({"asset_id": asset_id}, {"$set": doc}, upsert=True)
#     return {"status": "predicted", **doc}


# def train_all():
#     results = []
#     print("\n" + "=" * 60)
#     print("ENTRENAMIENTO Y BACKTEST CRONOLÓGICO MULTI-HORIZONTE")
#     print("=" * 60)
#     for asset_id in AGRICULTURAL_COMMODITIES.keys():
#         print(f"\nProcesando activo: [{asset_id}]...")
#         result = train_one(asset_id)
#         if result.get("status") == "trained":
#             print("   Estado: ÉXITO")
#             print(f"   Horizonte máximo entrenado: {result.get('max_trained_horizon')} sesiones")
#         else:
#             print(f"   Estado: {result.get('status')}")
#         results.append(result)
#     print("\nEntrenamiento finalizado.")
#     return results


# def predict_all():
#     results = []
#     print("\nGenerando proyecciones...")
#     for asset_id in AGRICULTURAL_COMMODITIES.keys():
#         print(f"Generando predicción para: [{asset_id}]...")
#         result = predict_one(asset_id)
#         print(f"   Estado: {result.get('status')}")
#         results.append(result)
#     print("Proyecciones finalizadas.\n")
#     return results


# if __name__ == "__main__":
#     train_all()
#     predict_all()
#     print("Proceso completo finalizado.")
