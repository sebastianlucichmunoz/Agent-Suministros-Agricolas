from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datetime import datetime, timezone
from src.db import ensure_indexes, log_run
from src.etl.history import load_ohlcv
from src.etl.news import refresh_news
from src.ml.model import train_all, predict_all

if __name__ == "__main__":
    started = datetime.now(timezone.utc)
    details = {}
    try:
        ensure_indexes()
        print("1/4 Cargando OHLCV Histórico de Commodities (Yahoo Finance)...")
        details["ohlcv"] = load_ohlcv(period="3y")
        print(details["ohlcv"])

        print("2/4 Noticias Agrícolas...")
        details["news"] = refresh_news()
        print(details["news"])

        print("3/4 Entrenando Modelos XGBoost...")
        trained = train_all()
        details["models_trained"] = sum(x.get("status") == "trained" for x in trained)
        print("Modelos entrenados:", details["models_trained"])

        print("4/4 Generando Predicciones...")
        preds = predict_all()
        details["predictions"] = sum(x.get("status") == "predicted" for x in preds)
        print("Predicciones:", details["predictions"])

        log_run("bootstrap", "success", started, details)
        print("Bootstrap Agro finalizado con éxito.")
    except Exception as exc:
        log_run("bootstrap", "error", started, details, str(exc))
        raise
