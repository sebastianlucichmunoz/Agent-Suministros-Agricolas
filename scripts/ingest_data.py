import sys
import os
from pymongo import UpdateOne

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.db import get_db
from src.api_clients.yfinance_client import YFinanceAgriClient

def run_ingestion():
    db = get_db()
    client = YFinanceAgriClient()
    registros = client.fetch_ohlcv_history(period="3y", interval="1d")
    
    if not registros:
        return
        
    print(f"Obtenidos {len(registros)} registros.")
    collection = db.ohlcv

    collection.create_index([("asset_id", 1), ("timestamp", 1), ("period", 1)], unique=True)

    operations = [
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

    if operations:
        res = collection.bulk_write(operations, ordered=False)
        processed = res.upserted_count + res.modified_count
        print(f"Sincronización completada: {processed} registros procesados en MongoDB Atlas (`agri_analytics_db`).")

if __name__ == "__main__":
    run_ingestion()
