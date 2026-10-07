from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default

def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default

@dataclass(frozen=True)
class Settings:
    mongodb_uri: str = os.getenv("MONGODB_URI", "")
    mongodb_db_name: str = os.getenv("MONGODB_DB_NAME", "agri_analytics")
    google_api_key: str = os.getenv("GEMINI_API_KEY", "")
    smtp_host: str = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port: int = env_int("SMTP_PORT", 587)
    smtp_user: str = os.getenv("SMTP_USER", "")
    smtp_pass: str = os.getenv("SMTP_PASS", "")
    email_from: str = os.getenv("EMAIL_FROM", "")
    history_days: int = env_int("HISTORY_DAYS", 365)
    model_min_rows: int = env_int("MODEL_MIN_ROWS", 90)
    prediction_neutral_threshold_pct: float = env_float("PREDICTION_NEUTRAL_THRESHOLD_PCT", 0.50)
    rss_reuters_commodity: str = os.getenv(
        "RSS_REUTERS_COMMODITY", 
        "https://www.google.com/alerts/feeds/18260197779770542387/15865203798993856184"
    )

settings = Settings()

def require(*attrs: str) -> None:
    names = {
        "mongodb_uri": "MONGODB_URI",
        "google_api_key": "GEMINI_API_KEY",
        "smtp_user": "SMTP_USER",
        "smtp_pass": "SMTP_PASS",
        "email_from": "EMAIL_FROM",
    }
    missing = [names.get(a, a) for a in attrs if not getattr(settings, a)]
    if missing:
        raise RuntimeError("Faltan variables de entorno requeridas: " + ", ".join(missing))