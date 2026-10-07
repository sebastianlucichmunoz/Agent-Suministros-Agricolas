from pymongo import UpdateOne
from src.api_clients.yfinance_client import YFinanceAgriClient
from src.db import get_db

def load_ohlcv(period="1y", interval="1d"):
    """"""
    db = get_db()
    client = YFinanceAgriClient()
    
    if isinstance(period, int):
        period = f"{period}d" if period <= 365 else "1y"

    registros = client.fetch_ohlcv_history(period=period, interval=interval)
    if not registros:
        return {"assets": 0, "rows": 0}

    collection = db.ohlcv
    collection.create_index([("asset_id", 1), ("timestamp", 1), ("period", 1)], unique=True)

    ops = [
        UpdateOne(
            {
                "asset_id": doc["asset_id"],
                "timestamp": doc["timestamp"],
                "period": doc["period"]
            },
            {"$set": doc},
            upsert=True
        )
        for doc in registros
    ]

    if ops:
        res = collection.bulk_write(ops, ordered=False)
        processed = res.upserted_count + res.modified_count
        assets_covered = len(set(doc["asset_id"] for doc in registros))
        return {"assets": assets_covered, "rows": processed}

    return {"assets": 0, "rows": 0}

def load_price_history(days=45):
    """
    Alias compatible para llamadas de actualización rápida.
    """
    return load_ohlcv(period=days)