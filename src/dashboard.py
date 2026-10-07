from datetime import datetime, timedelta, timezone
import pandas as pd
from src.db import get_db
from src.nlp.sentiment import mood

def latest_market():
    """
    Obtiene la cotización más reciente de cada commodity agrícola desde la colección `ohlcv`.
    """
    db = get_db()
    pipe = [
        {"$sort": {"timestamp": -1}},
        {"$group": {"_id": "$asset_id", "doc": {"$first": "$$ROOT"}}},
        {"$replaceRoot": {"newRoot": "$doc"}},
        {"$sort": {"asset_id": 1}}
    ]
    rows = list(db.ohlcv.aggregate(pipe))
    for x in rows:
        x.pop("_id", None)
    return pd.DataFrame(rows)

def ohlcv(asset_id: str):
    """
    Obtiene la serie temporal histórica de velas para un commodity específico.
    """
    rows = list(get_db().ohlcv.find(
        {"asset_id": asset_id, "period": "1d"},
        {"_id": 0, "timestamp": 1, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}
    ).sort("timestamp", 1))
    return pd.DataFrame(rows)

def news_for_asset(asset_id: str, limit: int = 50):
    """
    Obtiene noticias vinculadas al commodity o noticias generales del sector agropecuario.
    """
    db = get_db()
    asset_clean = asset_id.replace("_", " ")
    
    rows = list(db.news.find(
        {
            "$or": [
                {"mentioned_asset_ids": asset_id},
                {"mentioned_asset_ids": {"$in": [asset_id]}},
                {"title": {"$regex": asset_id, "$options": "i"}},
                {"title": {"$regex": asset_clean, "$options": "i"}}
            ]
        },
        {
            "_id": 0, "source": 1, "title": 1, "url": 1, 
            "published_at": 1, "sentiment_compound": 1, 
            "sentiment_label": 1, "mentioned_asset_ids": 1
        }
    ).sort("published_at", -1).limit(limit))
    
    if not rows:
        rows = list(db.news.find(
            {},
            {
                "_id": 0, "source": 1, "title": 1, "url": 1, 
                "published_at": 1, "sentiment_compound": 1, 
                "sentiment_label": 1, "mentioned_asset_ids": 1
            }
        ).sort("published_at", -1).limit(limit))
        
    return pd.DataFrame(rows)

def latest_prediction(asset_id: str):
    """
    Obtiene la predicción más reciente generada por XGBoost para el commodity.
    """
    return get_db().predictions.find_one({"asset_id": asset_id}, sort=[("predicted_at", -1)])

def latest_metrics(asset_id: str):
    """
    Obtiene las métricas del último entrenamiento del modelo para el commodity.
    """
    return get_db().model_metrics.find_one({"asset_id": asset_id}, sort=[("trained_at", -1)])

def market_mood(hours: int = 24):
    """
    Calcula el sentimiento general del mercado agro en base a las noticias de las últimas N horas.
    """
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    rows = list(get_db().news.aggregate([
        {"$match": {"published_at": {"$gte": since}}},
        {"$group": {"_id": None, "avg": {"$avg": "$sentiment_compound"}, "count": {"$sum": 1}}}
    ]))
    if not rows:
        return "SIN DATOS", None, 0
    return mood(rows[0]["avg"]), rows[0]["avg"], rows[0]["count"]

def load_gemini_recommendations():
    """Obtiene el documento de recomendaciones de IA MÁS RECIENTE de MongoDB."""
    try:
        db = get_db()
        ultimo_reporte = db.agent_recommendations.find_one(
            sort=[("fecha_actualizacion", -1)]
        )
        return ultimo_reporte
    except Exception as e:
        print(f"Error al cargar recomendaciones de MongoDB: {e}")
        return None