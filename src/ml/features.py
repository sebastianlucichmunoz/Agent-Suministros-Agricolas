import numpy as np
import pandas as pd

FEATURES = [
    "close", "volume", "return_7d", "return_14d", "return_30d", "return_60d",
    "ma_7", "ma_14", "ma_30", "ma_60",
    "ema_7", "ema_21", "volatility_7", "volatility_30", "volatility_60",
    "momentum_7", "momentum_30", "rsi_14", "lag_1", "lag_7", "lag_30",
    "sentiment_compound",
]

HORIZONS = (1, 5, 10, 21, 42, 63, 84, 105, 126)


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    result = result.mask((loss == 0) & (gain > 0), 100)
    result = result.mask((loss == 0) & (gain == 0), 50)
    return result


def build(prices, sentiment=None, target=False):
    """Construye features históricas y, opcionalmente, objetivos de retorno futuro.

    Los objetivos son log-retornos desde el cierre de la fecha t hasta el cierre
    de t+h. No se rellenan objetivos futuros desconocidos; el entrenamiento debe
    eliminar sus NaN de forma independiente para cada horizonte.
    """
    if prices is None or prices.empty:
        return pd.DataFrame()

    df = prices.copy()
    if "date" not in df.columns or "close" not in df.columns:
        raise ValueError("prices debe contener las columnas 'date' y 'close'.")

    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    else:
        df["volume"] = 0.0
    df["volume"] = df["volume"].fillna(0.0)
    df = (df.dropna(subset=["date", "close"])
            .loc[lambda x: x["close"] > 0]
            .sort_values("date")
            .drop_duplicates("date", keep="last")
            .reset_index(drop=True))

    close = df["close"]
    for n in (7, 14, 30, 60):
        df[f"return_{n}d"] = close.pct_change(n)
        df[f"ma_{n}"] = close.rolling(n, min_periods=n).mean()
    df["ema_7"] = close.ewm(span=7, adjust=False).mean()
    df["ema_21"] = close.ewm(span=21, adjust=False).mean()

    daily_returns = close.pct_change()
    for n in (7, 30, 60):
        df[f"volatility_{n}"] = daily_returns.rolling(n, min_periods=n).std()
    df["momentum_7"] = close / close.shift(7) - 1.0
    df["momentum_30"] = close / close.shift(30) - 1.0
    df["rsi_14"] = rsi(close, period=14)
    for n in (1, 7, 30):
        df[f"lag_{n}"] = close.shift(n)

    df["sentiment_compound"] = 0.0
    if sentiment is not None and not sentiment.empty and {"date", "sentiment_compound"}.issubset(sentiment.columns):
        sent = sentiment[["date", "sentiment_compound"]].copy()
        sent["date"] = pd.to_datetime(sent["date"], utc=True, errors="coerce").dt.floor("D")
        sent["sentiment_compound"] = pd.to_numeric(sent["sentiment_compound"], errors="coerce")
        sent = (sent.dropna(subset=["date"])
                    .groupby("date", as_index=False)["sentiment_compound"].mean())
        df["_day"] = df["date"].dt.floor("D")
        df = df.merge(sent, left_on="_day", right_on="date", how="left", suffixes=("", "_news"))
        df["sentiment_compound"] = df["sentiment_compound_news"].fillna(0.0)
        df = df.drop(columns=["_day", "date_news", "sentiment_compound_news"], errors="ignore")

    if target:
        for horizon in HORIZONS:
            future_close = close.shift(-horizon)
            df[f"target_return_{horizon}d"] = np.log(future_close / close)

    df = df.replace([np.inf, -np.inf], np.nan)
    return df
