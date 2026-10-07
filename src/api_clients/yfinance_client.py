import yfinance as yf
import pandas as pd
from datetime import datetime, timezone

# Mapeo de tickers a IDs unificados para nuestra BD
AGRICULTURAL_COMMODITIES = {
    "corn": "ZC=F",       # Maíz
    "wheat": "ZW=F",      # Trigo 
    "soybeans": "ZS=F",   # Soya 
    "rice": "ZR=F",       # Arroz 
    "oats": "ZO=F",       # Avena 
    "coffee": "KC=F",     # Café Arábica 
    "sugar": "SB=F",      # Azúcar N°11 
    "cocoa": "CC=F",      # Cacao 
    "cotton": "CT=F",     # Algodón 
    "crude_oil_transport": "CL=F", # Petroleo 
    "natural_gas_fertilizer": "NG=F", # Gas Natural 
    "crude_oil_transport": "CL=F",
}

class YFinanceAgriClient:
    def __init__(self, assets_map=None):
        self.assets_map = assets_map or AGRICULTURAL_COMMODITIES

    # Cambia el valor por defecto de period a "1y"--------------------------Este es importante porque determina el preiodo de antiguedad
    def fetch_ohlcv_history(self, period="1y", interval="1d"):
        """
        Descarga el historial de precios OHLCV para los commodities
        y los devuelve formateados para la colección 'ohlcv' en MongoDB.
        """
        all_records = []
        
        for asset_id, ticker in self.assets_map.items():
            print(f"Descargando datos para {asset_id} ({ticker})...")
            obj = yf.Ticker(ticker)
            df = obj.history(period=period, interval=interval)
            
            if df.empty:
                print(f"No se obtuvieron datos para {asset_id}")
                continue
                
            for timestamp, row in df.iterrows():
                date_utc = timestamp.to_pydatetime()
                if date_utc.tzinfo is None:
                    date_utc = date_utc.replace(tzinfo=timezone.utc)

                record = {
                    "asset_id": asset_id,       # Identificador único del insumo/alimento
                    "symbol": ticker,
                    "timestamp": date_utc,
                    "period": interval,
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": float(row["Volume"]),
                    "source": "yfinance"
                }
                all_records.append(record)
                
        return all_records