from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from datetime import datetime, timezone
from src.db import ensure_indexes, log_run
from src.etl.news import refresh_news
from src.etl.history import load_ohlcv
from src.ml.model import train_all, predict_all

def main(daily=False):
    started = datetime.now(timezone.utc)
    details = {"daily": daily}
    try:
        ensure_indexes()

        periodo_ingesta = "45d" if daily else "5d"
        details["ohlcv"] = load_ohlcv(period=periodo_ingesta)
        details["news"] = refresh_news()
        if daily:
            trained = train_all()
            details["models_trained"] = sum(x.get("status") == "trained" for x in trained)
            
        preds = predict_all()
        details["predictions"] = sum(x.get("status") == "predicted" for x in preds)
        
        log_run("refresh_daily" if daily else "refresh", "success", started, details)
        print("Sincronización agrocomercial completada exitosamente:", details)
        
    except Exception as exc:
        log_run("refresh_daily" if daily else "refresh", "error", started, details, str(exc))
        raise

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--daily", action="store_true")
    args = p.parse_args()
    main(args.daily)
