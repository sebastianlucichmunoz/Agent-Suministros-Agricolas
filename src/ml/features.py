import numpy as np
import pandas as pd

HORIZONS = (
    1, 3, 5, 7, 10, 14, 17, 21,
    26, 31, 36, 42, 47, 52, 57, 63,
    70, 77, 84, 91, 98, 105, 112, 126,
)

FEATURES = [
    "return_1d", "return_3d", "return_7d", "return_14d", "return_30d", "return_60d",
    "dist_ma_7", "dist_ma_14", "dist_ma_30", "dist_ma_60",
    "dist_ema_7", "dist_ema_21",
    "volatility_7", "volatility_30", "volatility_60", "rsi_14",
    "log_volume", "volume_rel_20", "month_sin", "month_cos",
    "return_90d", "return_126d", "return_180d",
    "dist_ma_90", "dist_ma_126", "dist_ma_180",
    "volatility_90", "volatility_126",
    "daily_range_pct", "open_close_pct", "high_close_pct", "low_close_pct",
    "range_ma_7", "range_ma_30", "volume_change",
]


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    result = 100.0 - 100.0 / (1.0 + rs)
    result = result.mask((loss == 0) & (gain > 0), 100.0)
    result = result.mask((loss == 0) & (gain == 0), 50.0)
    return result


def build(prices, sentiment=None, target=False):
    if prices is None or prices.empty:
        return pd.DataFrame()
    if "date" not in prices.columns or "close" not in prices.columns:
        raise ValueError("prices debe contener 'date' y 'close'.")

    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    for name in ("open", "high", "low"):
        if name not in df.columns:
            df[name] = np.nan
        df[name] = pd.to_numeric(df[name], errors="coerce")

    if "volume" not in df.columns:
        df["volume"] = 0.0
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0).clip(lower=0)

    df = (df.dropna(subset=["date", "close"])
          .loc[lambda x: x["close"] > 0]
          .sort_values("date")
          .drop_duplicates("date", keep="last")
          .reset_index(drop=True))
    if df.empty:
        return df

    close = df["close"]
    open_price = df["open"].where(df["open"] > 0, close).fillna(close)
    high_price = df["high"].where(df["high"] > 0, close).fillna(close)
    low_price = df["low"].where(df["low"] > 0, close).fillna(close)
    high_price = pd.concat([high_price, open_price, close], axis=1).max(axis=1)
    low_price = pd.concat([low_price, open_price, close], axis=1).min(axis=1)

    for n in (1, 3, 7, 14, 30, 60, 90, 126, 180):
        df[f"return_{n}d"] = close.pct_change(n, fill_method=None)

    for n in (7, 14, 30, 60, 90, 126, 180):
        ma = close.rolling(n, min_periods=n).mean()
        df[f"dist_ma_{n}"] = close / ma - 1.0

    for n in (7, 21):
        ema = close.ewm(span=n, adjust=False).mean()
        df[f"dist_ema_{n}"] = close / ema - 1.0

    for n in (7, 30, 60, 90, 126):
        df[f"volatility_{n}"] = df["return_1d"].rolling(n, min_periods=n).std()

    df["rsi_14"] = rsi(close)
    df["log_volume"] = np.log1p(df["volume"])
    rolling_volume = df["volume"].rolling(20, min_periods=1).mean()
    df["volume_rel_20"] = (df["volume"] / rolling_volume.where(rolling_volume > 0) - 1.0).fillna(0.0)

    df["daily_range_pct"] = (high_price - low_price) / close
    df["open_close_pct"] = (close - open_price) / open_price
    df["high_close_pct"] = (high_price - close) / close
    df["low_close_pct"] = (close - low_price) / close
    df["range_ma_7"] = df["daily_range_pct"].rolling(7, min_periods=7).mean()
    df["range_ma_30"] = df["daily_range_pct"].rolling(30, min_periods=30).mean()
    prior_volume = df["volume"].shift(1)
    df["volume_change"] = (df["volume"] / prior_volume.where(prior_volume > 0) - 1.0).fillna(0.0).clip(-5, 5)

    month = df["date"].dt.month
    df["month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
    df["month_cos"] = np.cos(2.0 * np.pi * month / 12.0)

    if target:
        for horizon in HORIZONS:
            df[f"target_return_{horizon}d"] = np.log(close.shift(-horizon) / close)

    return df.replace([np.inf, -np.inf], np.nan)


def features_for_horizon(horizon):
    long_features = {
        "return_90d", "return_126d", "return_180d",
        "dist_ma_90", "dist_ma_126", "dist_ma_180",
        "volatility_90", "volatility_126",
    }
    short = [name for name in FEATURES if name not in long_features]
    if horizon <= 10:
        return short
    if horizon <= 63:
        return short + ["return_90d", "dist_ma_90", "volatility_90"]
    return list(FEATURES)








#--Mejor version



# import numpy as np
# import pandas as pd

# HORIZONS = (1, 5, 10, 21, 42, 63, 84, 105, 126)

# FEATURES = [
#     "return_1d", "return_3d", "return_7d", "return_14d", "return_30d", "return_60d",
#     "dist_ma_7", "dist_ma_14", "dist_ma_30", "dist_ma_60",
#     "dist_ema_7", "dist_ema_21",
#     "volatility_7", "volatility_30", "volatility_60", "rsi_14",
#     "log_volume", "volume_rel_20", "month_sin", "month_cos",
#     "return_90d", "return_126d", "return_180d",
#     "dist_ma_90", "dist_ma_126", "dist_ma_180",
#     "volatility_90", "volatility_126",
#     "daily_range_pct", "open_close_pct", "high_close_pct", "low_close_pct",
#     "range_ma_7", "range_ma_30", "volume_change",
# ]

# def rsi(series: pd.Series, period: int = 14) -> pd.Series:
#     delta = series.diff()
#     gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
#     loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
#     rs = gain / loss.replace(0, np.nan)
#     result = 100.0 - 100.0 / (1.0 + rs)
#     result = result.mask((loss == 0) & (gain > 0), 100.0)
#     result = result.mask((loss == 0) & (gain == 0), 50.0)
#     return result


# def build(prices, sentiment=None, target=False):
#     if prices is None or prices.empty:
#         return pd.DataFrame()
#     if "date" not in prices.columns or "close" not in prices.columns:
#         raise ValueError("prices debe contener 'date' y 'close'.")

#     df = prices.copy()
#     df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
#     df["close"] = pd.to_numeric(df["close"], errors="coerce")

#     for name in ("open", "high", "low"):
#         if name not in df.columns:
#             df[name] = np.nan
#         df[name] = pd.to_numeric(df[name], errors="coerce")

#     if "volume" not in df.columns:
#         df["volume"] = 0.0
#     df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0).clip(lower=0)

#     df = (df.dropna(subset=["date", "close"])
#           .loc[lambda x: x["close"] > 0]
#           .sort_values("date")
#           .drop_duplicates("date", keep="last")
#           .reset_index(drop=True))
#     if df.empty:
#         return df

#     close = df["close"]
#     open_price = df["open"].where(df["open"] > 0, close).fillna(close)
#     high_price = df["high"].where(df["high"] > 0, close).fillna(close)
#     low_price = df["low"].where(df["low"] > 0, close).fillna(close)
#     high_price = pd.concat([high_price, open_price, close], axis=1).max(axis=1)
#     low_price = pd.concat([low_price, open_price, close], axis=1).min(axis=1)

#     for n in (1, 3, 7, 14, 30, 60, 90, 126, 180):
#         df[f"return_{n}d"] = close.pct_change(n, fill_method=None)

#     for n in (7, 14, 30, 60, 90, 126, 180):
#         ma = close.rolling(n, min_periods=n).mean()
#         df[f"dist_ma_{n}"] = close / ma - 1.0

#     for n in (7, 21):
#         ema = close.ewm(span=n, adjust=False).mean()
#         df[f"dist_ema_{n}"] = close / ema - 1.0

#     for n in (7, 30, 60, 90, 126):
#         df[f"volatility_{n}"] = df["return_1d"].rolling(n, min_periods=n).std()

#     df["rsi_14"] = rsi(close)
#     df["log_volume"] = np.log1p(df["volume"])
#     rolling_volume = df["volume"].rolling(20, min_periods=1).mean()
#     df["volume_rel_20"] = (df["volume"] / rolling_volume.where(rolling_volume > 0) - 1.0).fillna(0.0)

#     df["daily_range_pct"] = (high_price - low_price) / close
#     df["open_close_pct"] = (close - open_price) / open_price
#     df["high_close_pct"] = (high_price - close) / close
#     df["low_close_pct"] = (close - low_price) / close
#     df["range_ma_7"] = df["daily_range_pct"].rolling(7, min_periods=7).mean()
#     df["range_ma_30"] = df["daily_range_pct"].rolling(30, min_periods=30).mean()
#     prior_volume = df["volume"].shift(1)
#     df["volume_change"] = (df["volume"] / prior_volume.where(prior_volume > 0) - 1.0).fillna(0.0).clip(-5, 5)

#     month = df["date"].dt.month
#     df["month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
#     df["month_cos"] = np.cos(2.0 * np.pi * month / 12.0)

#     if target:
#         for horizon in HORIZONS:
#             df[f"target_return_{horizon}d"] = np.log(close.shift(-horizon) / close)

#     return df.replace([np.inf, -np.inf], np.nan)


# def features_for_horizon(horizon):
#     long_features = {
#         "return_90d", "return_126d", "return_180d",
#         "dist_ma_90", "dist_ma_126", "dist_ma_180",
#         "volatility_90", "volatility_126",
#     }
#     short = [name for name in FEATURES if name not in long_features]
#     if horizon <= 10:
#         return short
#     if horizon <= 63:
#         return short + ["return_90d", "dist_ma_90", "volatility_90"]
#     return list(FEATURES)









# """Variables de mercado para predicción agrícola multi-horizonte.

# No utiliza noticias ni sentimiento en los modelos. Cada fila representa la
# información conocida en el cierre de la sesión correspondiente.

# Incluye indicadores técnicos, estacionalidad y variables OHLCV.
# """

# import numpy as np
# import pandas as pd

# HORIZONS = (1, 5, 10, 21, 42, 63, 84, 105, 126)

# FEATURES = [
#     # Retornos históricos
#     "return_1d", "return_3d", "return_7d", "return_14d",
#     "return_30d", "return_60d",

#     # Distancias a medias móviles
#     "dist_ma_7", "dist_ma_14", "dist_ma_30", "dist_ma_60",

#     # Medias exponenciales
#     "dist_ema_7", "dist_ema_21",

#     # Volatilidad y RSI
#     "volatility_7", "volatility_30", "volatility_60",
#     "rsi_14",

#     # Volumen
#     "log_volume", "volume_rel_20",

#     # Estacionalidad
#     "month_sin", "month_cos",

#     # Indicadores de largo plazo
#     "return_90d", "return_126d", "return_180d",
#     "dist_ma_90", "dist_ma_126", "dist_ma_180",
#     "volatility_90", "volatility_126",

#     # Nuevos indicadores OHLCV
#     "daily_range_pct",
#     "open_close_pct",
#     "high_close_pct",
#     "low_close_pct",
#     "range_ma_7",
#     "range_ma_30",
#     "volume_change",
# ]


# def rsi(series: pd.Series, period: int = 14) -> pd.Series:
#     """Calcula el índice de fuerza relativa (RSI)."""

#     delta = series.diff()

#     gain = delta.clip(lower=0).rolling(
#         period, min_periods=period
#     ).mean()

#     loss = (-delta.clip(upper=0)).rolling(
#         period, min_periods=period
#     ).mean()

#     rs = gain / loss.replace(0, np.nan)

#     result = 100.0 - 100.0 / (1.0 + rs)

#     result = result.mask(
#         (loss == 0) & (gain > 0), 100.0
#     )

#     result = result.mask(
#         (loss == 0) & (gain == 0), 50.0
#     )

#     return result


# def build(prices, sentiment=None, target=False):
#     """Construye variables históricas y objetivos de retorno futuro.

#     Los horizontes se expresan en sesiones de mercado.

#     El sentimiento se mantiene como argumento opcional por
#     compatibilidad, pero no participa en las predicciones.

#     Los objetivos futuros desconocidos permanecen como NaN.
#     """

#     if prices is None or prices.empty:
#         return pd.DataFrame()

#     if "date" not in prices.columns or "close" not in prices.columns:
#         raise ValueError(
#             "prices debe contener las columnas 'date' y 'close'."
#         )

#     df = prices.copy()

#     # ==========================================
#     # 1. PREPARACIÓN DE DATOS
#     # ==========================================

#     df["date"] = pd.to_datetime(
#         df["date"], utc=True, errors="coerce"
#     )

#     df["close"] = pd.to_numeric(
#         df["close"], errors="coerce"
#     )

#     # Campos OHLC
#     for col in ("open", "high", "low"):
#         if col not in df.columns:
#             df[col] = np.nan

#         df[col] = pd.to_numeric(
#             df[col], errors="coerce"
#         )

#     # Respaldo para registros OHLC incompletos.
#     # Estos reemplazos no representan datos intradiarios reales.
#     df["open"] = df["open"].fillna(df["close"])
#     df["high"] = df["high"].fillna(df["close"])
#     df["low"] = df["low"].fillna(df["close"])

#     if "volume" not in df.columns:
#         df["volume"] = 0.0

#     df["volume"] = (
#         pd.to_numeric(df["volume"], errors="coerce")
#         .fillna(0.0)
#         .clip(lower=0)
#     )

#     # Orden cronológico
#     df = (
#         df.dropna(subset=["date", "close"])
#         .loc[lambda x: x["close"] > 0]
#         .sort_values("date")
#         .drop_duplicates("date", keep="last")
#         .reset_index(drop=True)
#     )

#     if df.empty:
#         return df

#     close = df["close"]

#     # ==========================================
#     # 2. RETORNOS HISTÓRICOS
#     # ==========================================

#     for n in (1, 3, 7, 14, 30, 60, 90, 126, 180):
#         df[f"return_{n}d"] = close.pct_change(n)

#     # ==========================================
#     # 3. MEDIAS MÓVILES
#     # ==========================================

#     for n in (7, 14, 30, 60, 90, 126, 180):
#         moving_average = close.rolling(
#             n, min_periods=n
#         ).mean()

#         df[f"dist_ma_{n}"] = (
#             close / moving_average - 1.0
#         )

#     # ==========================================
#     # 4. MEDIAS MÓVILES EXPONENCIALES
#     # ==========================================

#     for n in (7, 21):
#         exponential_average = close.ewm(
#             span=n, adjust=False
#         ).mean()

#         df[f"dist_ema_{n}"] = (
#             close / exponential_average - 1.0
#         )

#     # ==========================================
#     # 5. VOLATILIDAD
#     # ==========================================

#     for n in (7, 30, 60, 90, 126):
#         df[f"volatility_{n}"] = (
#             df["return_1d"]
#             .rolling(n, min_periods=n)
#             .std()
#         )

#     # ==========================================
#     # 6. RSI
#     # ==========================================

#     df["rsi_14"] = rsi(close)

#     # ==========================================
#     # 7. VARIABLES DE VOLUMEN
#     # ==========================================

#     # Algunos contratos de futuros tienen volumen
#     # igual a cero o no disponible.

#     df["log_volume"] = np.log1p(df["volume"])

#     rolling_volume = (
#         df["volume"]
#         .rolling(20, min_periods=1)
#         .mean()
#     )

#     df["volume_rel_20"] = (
#         df["volume"]
#         / rolling_volume.where(rolling_volume > 0, np.nan)
#         - 1.0
#     ).fillna(0.0)

#     # ==========================================
#     # 8. NUEVOS INDICADORES OHLCV
#     # ==========================================

#     open_price = df["open"]
#     high = df["high"]
#     low = df["low"]

#     # Rango intradiario relativo
#     df["daily_range_pct"] = (
#         (high - low) / close
#     )

#     # Variación entre apertura y cierre
#     df["open_close_pct"] = (
#         (close - open_price)
#         / open_price.where(open_price > 0, np.nan)
#     )

#     # Distancia entre precio máximo y cierre
#     df["high_close_pct"] = (
#         (high - close) / close
#     )

#     # Distancia entre precio mínimo y cierre
#     df["low_close_pct"] = (
#         (close - low) / close
#     )

#     # Promedio del rango de las últimas 7 sesiones
#     df["range_ma_7"] = (
#         df["daily_range_pct"]
#         .rolling(7, min_periods=7)
#         .mean()
#     )

#     # Promedio del rango de las últimas 30 sesiones
#     df["range_ma_30"] = (
#         df["daily_range_pct"]
#         .rolling(30, min_periods=30)
#         .mean()
#     )

#     # Variación relativa del volumen
#     previous_volume = df["volume"].shift(1)

#     df["volume_change"] = (
#         df["volume"]
#         / previous_volume.where(previous_volume > 0, np.nan)
#         - 1.0
#     ).fillna(0.0).clip(-5, 5)

#     # ==========================================
#     # 9. ESTACIONALIDAD
#     # ==========================================

#     month = df["date"].dt.month

#     df["month_sin"] = np.sin(
#         2 * np.pi * month / 12
#     )

#     df["month_cos"] = np.cos(
#         2 * np.pi * month / 12
#     )

#     # ==========================================
#     # 10. OBJETIVOS DE PREDICCIÓN
#     # ==========================================

#     if target:
#         for horizon in HORIZONS:
#             future_close = close.shift(-horizon)

#             df[f"target_return_{horizon}d"] = (
#                 np.log(future_close / close)
#             )

#     # Limpieza de valores infinitos
#     df = df.replace(
#         [np.inf, -np.inf], np.nan
#     )

#     return df


# def features_for_horizon(horizon):
#     """Selecciona variables según el horizonte predictivo.

#     Los horizontes cortos evitan depender de indicadores
#     que requieren 180 sesiones históricas.

#     Las variables OHLCV se utilizan en todos los horizontes.
#     """

#     long_features = {
#         "return_90d",
#         "return_126d",
#         "return_180d",
#         "dist_ma_90",
#         "dist_ma_126",
#         "dist_ma_180",
#         "volatility_90",
#         "volatility_126",
#     }

#     short = [
#         name for name in FEATURES
#         if name not in long_features
#     ]

#     # Corto plazo: 1 a 10 sesiones
#     if horizon <= 10:
#         return short

#     # Mediano plazo: 21 a 63 sesiones
#     if horizon <= 63:
#         return short + [
#             "return_90d",
#             "dist_ma_90",
#             "volatility_90",
#         ]

#     # Largo plazo: 84 a 126 sesiones
#     return list(FEATURES)























# """Variables de mercado para predicción agrícola multi-horizonte.

# No utiliza noticias ni sentimiento en los modelos. Cada fila representa la
# información conocida en el cierre de la sesión correspondiente.
# """

# import numpy as np
# import pandas as pd

# HORIZONS = (1, 5, 10, 21, 42, 63, 84, 105, 126)

# FEATURES = [
#     "return_1d", "return_3d", "return_7d", "return_14d", "return_30d", "return_60d",
#     "dist_ma_7", "dist_ma_14", "dist_ma_30", "dist_ma_60",
#     "dist_ema_7", "dist_ema_21",
#     "volatility_7", "volatility_30", "volatility_60",
#     "rsi_14", "log_volume", "volume_rel_20",
#     "month_sin", "month_cos",
# ]


# def rsi(series: pd.Series, period: int = 14) -> pd.Series:
#     delta = series.diff()
#     gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
#     loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
#     rs = gain / loss.replace(0, np.nan)
#     result = 100.0 - 100.0 / (1.0 + rs)
#     result = result.mask((loss == 0) & (gain > 0), 100.0)
#     result = result.mask((loss == 0) & (gain == 0), 50.0)
#     return result


# def build(prices, sentiment=None, target=False):
#     """Construye variables y log-retornos futuros por sesiones de mercado.

#     `sentiment` se mantiene como argumento opcional por compatibilidad con
#     llamadas anteriores, pero se ignora por diseño.
#     No se rellenan valores futuros desconocidos de los objetivos.
#     """
#     if prices is None or prices.empty:
#         return pd.DataFrame()
#     if "date" not in prices.columns or "close" not in prices.columns:
#         raise ValueError("prices debe contener las columnas 'date' y 'close'.")

#     df = prices.copy()
#     df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
#     df["close"] = pd.to_numeric(df["close"], errors="coerce")
#     if "volume" not in df.columns:
#         df["volume"] = 0.0
#     df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0).clip(lower=0)
#     df = (df.dropna(subset=["date", "close"])
#             .loc[lambda x: x["close"] > 0]
#             .sort_values("date")
#             .drop_duplicates("date", keep="last")
#             .reset_index(drop=True))
#     if df.empty:
#         return df

#     close = df["close"]
#     for n in (1, 3, 7, 14, 30, 60):
#         df[f"return_{n}d"] = close.pct_change(n)
#     for n in (7, 14, 30, 60):
#         moving_average = close.rolling(n, min_periods=n).mean()
#         df[f"dist_ma_{n}"] = close / moving_average - 1.0
#     for n in (7, 21):
#         exponential_average = close.ewm(span=n, adjust=False).mean()
#         df[f"dist_ema_{n}"] = close / exponential_average - 1.0

#     for n in (7, 30, 60):
#         df[f"volatility_{n}"] = df["return_1d"].rolling(n, min_periods=n).std()
#     df["rsi_14"] = rsi(close)

#     # Algunos futuros de Yahoo Finance tienen volumen igual a cero o ausente.
#     # En esos casos las variables de volumen quedan en 0 sin perder filas.
#     df["log_volume"] = np.log1p(df["volume"])
#     rolling_volume = df["volume"].rolling(20, min_periods=1).mean()
#     df["volume_rel_20"] = (df["volume"] / rolling_volume.where(rolling_volume > 0, np.nan) - 1.0).fillna(0.0)

#     month = df["date"].dt.month
#     df["month_sin"] = np.sin(2 * np.pi * month / 12)
#     df["month_cos"] = np.cos(2 * np.pi * month / 12)

#     if target:
#         for horizon in HORIZONS:
#             df[f"target_return_{horizon}d"] = np.log(close.shift(-horizon) / close)

#     df = df.replace([np.inf, -np.inf], np.nan)
#     return df


# import numpy as np
# import pandas as pd

# FEATURES = [
#     "close", "volume", "return_7d", "return_14d", "return_30d", "return_60d",
#     "ma_7", "ma_14", "ma_30", "ma_60",
#     "ema_7", "ema_21", "volatility_7", "volatility_30", "volatility_60",
#     "momentum_7", "momentum_30", "rsi_14", "lag_1", "lag_7", "lag_30",
#     "sentiment_compound",
# ]

# HORIZONS = (1, 5, 10, 21, 42, 63, 84, 105, 126)


# def rsi(series: pd.Series, period: int = 14) -> pd.Series:
#     delta = series.diff()
#     gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
#     loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
#     rs = gain / loss.replace(0, np.nan)
#     result = 100 - (100 / (1 + rs))
#     result = result.mask((loss == 0) & (gain > 0), 100)
#     result = result.mask((loss == 0) & (gain == 0), 50)
#     return result


# def build(prices, sentiment=None, target=False):
#     """Construye features históricas y, opcionalmente, objetivos de retorno futuro.

#     Los objetivos son log-retornos desde el cierre de la fecha t hasta el cierre
#     de t+h. No se rellenan objetivos futuros desconocidos; el entrenamiento debe
#     eliminar sus NaN de forma independiente para cada horizonte.
#     """
#     if prices is None or prices.empty:
#         return pd.DataFrame()

#     df = prices.copy()
#     if "date" not in df.columns or "close" not in df.columns:
#         raise ValueError("prices debe contener las columnas 'date' y 'close'.")

#     df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
#     df["close"] = pd.to_numeric(df["close"], errors="coerce")
#     if "volume" in df.columns:
#         df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
#     else:
#         df["volume"] = 0.0
#     df["volume"] = df["volume"].fillna(0.0)
#     df = (df.dropna(subset=["date", "close"])
#             .loc[lambda x: x["close"] > 0]
#             .sort_values("date")
#             .drop_duplicates("date", keep="last")
#             .reset_index(drop=True))

#     close = df["close"]
#     for n in (7, 14, 30, 60):
#         df[f"return_{n}d"] = close.pct_change(n)
#         df[f"ma_{n}"] = close.rolling(n, min_periods=n).mean()
#     df["ema_7"] = close.ewm(span=7, adjust=False).mean()
#     df["ema_21"] = close.ewm(span=21, adjust=False).mean()

#     daily_returns = close.pct_change()
#     for n in (7, 30, 60):
#         df[f"volatility_{n}"] = daily_returns.rolling(n, min_periods=n).std()
#     df["momentum_7"] = close / close.shift(7) - 1.0
#     df["momentum_30"] = close / close.shift(30) - 1.0
#     df["rsi_14"] = rsi(close, period=14)
#     for n in (1, 7, 30):
#         df[f"lag_{n}"] = close.shift(n)

#     df["sentiment_compound"] = 0.0
#     if sentiment is not None and not sentiment.empty and {"date", "sentiment_compound"}.issubset(sentiment.columns):
#         sent = sentiment[["date", "sentiment_compound"]].copy()
#         sent["date"] = pd.to_datetime(sent["date"], utc=True, errors="coerce").dt.floor("D")
#         sent["sentiment_compound"] = pd.to_numeric(sent["sentiment_compound"], errors="coerce")
#         sent = (sent.dropna(subset=["date"])
#                     .groupby("date", as_index=False)["sentiment_compound"].mean())
#         df["_day"] = df["date"].dt.floor("D")
#         df = df.merge(sent, left_on="_day", right_on="date", how="left", suffixes=("", "_news"))
#         df["sentiment_compound"] = df["sentiment_compound_news"].fillna(0.0)
#         df = df.drop(columns=["_day", "date_news", "sentiment_compound_news"], errors="ignore")

#     if target:
#         for horizon in HORIZONS:
#             future_close = close.shift(-horizon)
#             df[f"target_return_{horizon}d"] = np.log(future_close / close)

#     df = df.replace([np.inf, -np.inf], np.nan)
#     return df
