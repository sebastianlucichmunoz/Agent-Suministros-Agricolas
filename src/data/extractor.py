import json
import numpy as np
from datetime import datetime, timedelta, timezone
from src.db import get_db

def extraer_picos_y_muestras(predictions_180d, step=30):
    
    if not predictions_180d:
        return []

    max_peak = max(predictions_180d, key=lambda x: x["predicted_close"])
    min_valley = min(predictions_180d, key=lambda x: x["predicted_close"])

    puntos_clave = {
        max_peak["day"]: {**max_peak, "tipo": "Pico Máximo"},
        min_valley["day"]: {**min_valley, "tipo": "Valle Mínimo"}
    }

    for p in predictions_180d:
        if p["day"] % step == 0 and p["day"] not in puntos_clave:
            puntos_clave[p["day"]] = {**p, "tipo": f"Hito de Proyección"}

    puntos_ordenados = sorted(puntos_clave.values(), key=lambda x: x["day"])

    return [
        {
            "tipo": item["tipo"],
            "fecha": str(item["target_time"])[:10],  # Formato YYYY-MM-DD
            "precio_proyectado": round(float(item["predicted_close"]), 2),
            "variacion_pct": round(float(item["change_pct"]), 2)
        }
        for item in puntos_ordenados
    ]

def extraer_datos_consolidados_agricolas():
    """
    Extrae las predicciones autorregresivas desde db.predictions,
    comprime los puntos clave por fecha y guarda el lote en db.market_summary.
    """
    db = get_db()
    preds_raw = list(db.predictions.find({}, {"_id": 0}))
    
    mercado_consolidado = {}
    
    for p in preds_raw:
        asset_id = p.get("asset_id")
        current_price = float(p.get("current_price", 0.0))
        predicted_close_180d = float(p.get("predicted_close_180d", current_price))
        predicted_change_pct = float(p.get("predicted_change_pct", 0.0))
        puntos_clave = extraer_picos_y_muestras(p.get("predictions_180d", []))
        
        mercado_consolidado[asset_id] = {
            "asset_id": asset_id,
            "current_price": round(current_price, 2),
            "predicted_close_180d": round(predicted_close_180d, 2),
            "predicted_change_pct": round(predicted_change_pct, 2),
            "trend": p.get("trend", "ESTABLE"),
            "puntos_clave_fechas": puntos_clave,
            "sentiment_compound": 0.0,
            "sentiment_label": "NEUTRAL",
            "news_count": 0
        }

    hace_7d = datetime.now(timezone.utc) - timedelta(days=7)
    pipeline_news = [
        {"$match": {"published_at": {"$gte": hace_7d}}},
        {"$unwind": "$mentioned_asset_ids"},
        {
            "$group": {
                "_id": "$mentioned_asset_ids",
                "avg_compound": {"$avg": "$sentiment_compound"},
                "news_count": {"$sum": 1}
            }
        }
    ]
    news_raw = list(db.news.aggregate(pipeline_news))
    
    for n in news_raw:
        asset_id = n["_id"]
        if asset_id in mercado_consolidado:
            compound = float(n.get("avg_compound", 0.0))
            mercado_consolidado[asset_id]["sentiment_compound"] = round(compound, 4)
            mercado_consolidado[asset_id]["news_count"] = n.get("news_count", 0)
            
            if compound >= 0.05:
                label = "POSITIVO"
            elif compound <= -0.05:
                label = "NEGATIVO"
            else:
                label = "NEUTRAL"
            mercado_consolidado[asset_id]["sentiment_label"] = label

    datos_finales = list(mercado_consolidado.values())

    try:
        for item in datos_finales:
            db.market_summary.update_one(
                {"asset_id": item["asset_id"]},
                {
                    "$set": {
                        **item,
                        "updated_at": datetime.now(timezone.utc)
                    }
                },
                upsert=True
            )
        print(f"¡Éxito! {len(datos_finales)} activos procesados y almacenados en 'market_summary'.")
    except Exception as e:
        print(f"Error al guardar en 'market_summary': {e}")

    return datos_finales

if __name__ == "__main__":
    extraer_datos_consolidados_agricolas()
