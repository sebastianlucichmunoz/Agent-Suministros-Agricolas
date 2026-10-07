import numpy as np
import pandas as pd

FEATURES = [
    "close", "volume", "return_1d", "return_3d", "return_7d", "return_14d",
    "ma_7", "ma_14", "ma_30", "ema_7", "ema_21",
    "volatility_7", "volatility_14", "volatility_30",
    "momentum_7", "momentum_14", "rsi_14",
    "lag_1", "lag_2", "lag_3", "lag_7", "sentiment_compound"
]

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)

def build(prices, sentiment=None, target=True):
    if prices.empty:
        return prices.copy()
    
    df = prices.copy().sort_values("date").drop_duplicates("date", keep="last")
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0)

    # Retornos y Medias Móviles
    for n in (1, 3, 7, 14):
        df[f"return_{n}d"] = df["close"].pct_change(n)
    for n in (7, 14, 30):
        df[f"ma_{n}"] = df["close"].rolling(n).mean()
        
    df["ema_7"] = df["close"].ewm(span=7, adjust=False).mean()
    df["ema_21"] = df["close"].ewm(span=21, adjust=False).mean()
    
    ret = df["close"].pct_change()
    for n in (7, 14, 30):
        df[f"volatility_{n}"] = ret.rolling(n).std()
        
    df["momentum_7"] = df["close"] / df["close"].shift(7) - 1
    df["momentum_14"] = df["close"] / df["close"].shift(14) - 1
    df["rsi_14"] = rsi(df["close"])
    
    for n in (1, 2, 3, 7):
        df[f"lag_{n}"] = df["close"].shift(n)

    # Sentimiento de Noticias Agrícolas
    df["sentiment_compound"] = 0.0
    if sentiment is not None and not sentiment.empty:
        sent = sentiment.copy()
        sent["date"] = pd.to_datetime(sent["date"], utc=True).dt.floor("D")
        sent = sent.groupby("date", as_index=False)["sentiment_compound"].mean()
        
        df["day"] = df["date"].dt.floor("D")
        df = df.merge(sent, left_on="day", right_on="date", how="left", suffixes=("", "_sent"))
        df["sentiment_compound"] = df["sentiment_compound_sent"].fillna(0.0)
        df = df.drop(columns=["day", "date_sent"], errors="ignore")

    if target:
        df["target_close_next"] = df["close"].shift(-1)

    needed = FEATURES + (["target_close_next"] if target else [])
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.dropna(subset=needed).reset_index(drop=True)