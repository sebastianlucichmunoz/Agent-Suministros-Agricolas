# src/etl/news.py
import feedparser
from datetime import datetime, timezone
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from src.db import get_db

# Asegurar recursos de NLTK
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# Mapeo explícito de palabras clave por commodity
ASSET_KEYWORDS = {
    "corn": ["corn", "maize", "maíz", "grain futures"],
    "wheat": ["wheat", "trigo", "milling wheat"],
    "soybeans": ["soybean", "soybeans", "soya", "soja", "oilseeds"],
    "rice": ["rice", "arroz", "paddy rice"],
    "oats": ["oats", "avena"],
    "coffee": ["coffee", "café", "arabica", "robusta"],
    "sugar": ["sugar", "azúcar", "cane sugar"],
    "cocoa": ["cocoa", "cacao", "chocolate market"],
    "cotton": ["cotton", "algodón"],
    "crude_oil_transport": ["crude oil", "wti", "brent", "petroleum", "oil prices", "cattle", "livestock"],
    "natural_gas_fertilizer": ["natural gas", "gas natural", "fertilizer", "urea", "ammonia"]
}

# 2. Feeds RSS de Google News específicos para CADA insumo
RSS_FEEDS = {
    "corn": "https://news.google.com/rss/search?q=corn+maize+prices+futures&hl=en-US&gl=US&ceid=US:en",
    "wheat": "https://news.google.com/rss/search?q=wheat+prices+futures+grain&hl=en-US&gl=US&ceid=US:en",
    "soybeans": "https://news.google.com/rss/search?q=soybean+prices+futures&hl=en-US&gl=US&ceid=US:en",
    "rice": "https://news.google.com/rss/search?q=rice+paddy+prices+futures&hl=en-US&gl=US&ceid=US:en",
    "oats": "https://news.google.com/rss/search?q=oats+grain+prices+futures&hl=en-US&gl=US&ceid=US:en",
    "coffee": "https://news.google.com/rss/search?q=coffee+prices+futures&hl=en-US&gl=US&ceid=US:en",
    "sugar": "https://news.google.com/rss/search?q=sugar+prices+futures&hl=en-US&gl=US&ceid=US:en",
    "cocoa": "https://news.google.com/rss/search?q=cocoa+prices+futures&hl=en-US&gl=US&ceid=US:en",
    "cotton": "https://news.google.com/rss/search?q=cotton+prices+futures&hl=en-US&gl=US&ceid=US:en",
    "crude_oil_transport": "https://news.google.com/rss/search?q=crude+oil+prices+futures&hl=en-US&gl=US&ceid=US:en",
    "natural_gas_fertilizer": "https://news.google.com/rss/search?q=natural+gas+fertilizer+urea+prices&hl=en-US&gl=US&ceid=US:en"
}

def tag_assets_in_title(title: str) -> list[str]:
    """Detecta a qué commodities pertenece la noticia basándose en el título."""
    title_lower = title.lower()
    mentioned = []
    
    for asset_id, keywords in ASSET_KEYWORDS.items():
        if any(kw in title_lower for kw in keywords):
            mentioned.append(asset_id)
            
    # Fallback si menciona agricultura/granos generales
    if not mentioned and any(g in title_lower for g in ["grain", "crop", "harvest", "agriculture"]):
        mentioned.extend(["corn", "wheat"])
        
    return mentioned


def refresh_news(limit_per_asset=5):
    """
    Extrae noticias desde feeds RSS, calcula sentimiento con VADER e inserta en MongoDB Atlas.
    Limita la cantidad de noticias a procesar por cada commodity (por defecto max 5).
    """
    db = get_db()
    sia = SentimentIntensityAnalyzer()
    news_collection = db.news
    
    total_ingested = 0
    print(f"🚀 Iniciando extracción de noticias agrícolas (Límite: {limit_per_asset} por activo)...")
    
    for asset_key, feed_url in RSS_FEEDS.items():
        parsed = feedparser.parse(feed_url)
        
        # Límite por activo para no sobrecargar MongoDB ni el Dashboard
        entries_to_process = parsed.entries[:limit_per_asset]
        
        for entry in entries_to_process:
            title = entry.get("title", "")
            url = entry.get("link", "#")
            
            # Etiquetar commodities mencionados
            mentioned_assets = tag_assets_in_title(title)
            if asset_key not in mentioned_assets:
                mentioned_assets.append(asset_key)
                
            # Calcular Sentimiento VADER
            scores = sia.polarity_scores(title)
            compound = scores["compound"]
            
            if compound > 0.05:
                label = "POSITIVO"
            elif compound < -0.05:
                label = "NEGATIVO"
            else:
                label = "NEUTRAL"
                
            doc = {
                "news_id": url,
                "title": title,
                "url": url,
                "source": entry.get("source", {}).get("title", "Google News"),
                "published_at": datetime.now(timezone.utc),
                "ingested_at": datetime.now(timezone.utc),
                "mentioned_asset_ids": mentioned_assets,
                "sentiment_compound": compound,
                "sentiment_positive": scores["pos"],
                "sentiment_negative": scores["neg"],
                "sentiment_neutral": scores["neu"],
                "sentiment_label": label
            }
            
            # Upsert en MongoDB para no duplicar noticias
            result = news_collection.update_one(
                {"url": url},
                {"$set": doc},
                upsert=True
            )
            if result.upserted_id or result.modified_count:
                total_ingested += 1
                
    print(f"✅ Proceso finalizado. Se procesaron/actualizaron {total_ingested} noticias de commodities.")


# ¡Atención! El bloque principal debe ir alineado al margen izquierdo (sin espacios)
if __name__ == "__main__":
    refresh_news()














# import calendar, hashlib, re
# from datetime import datetime, timezone
# import feedparser
# from pymongo import UpdateOne
# from src.config import settings
# from src.db import get_db
# from src.nlp.sentiment import clean_html, score
# from src.api_clients.yfinance_client import AGRICULTURAL_COMMODITIES

# def published(entry):
#     for attr in ("published_parsed", "updated_parsed", "created_parsed"):
#         p = getattr(entry, attr, None)
#         if p:
#             return datetime.fromtimestamp(calendar.timegm(p), tz=timezone.utc)
#     return datetime.now(timezone.utc)

# # Definición de palabras clave por commodity
# ASSET_KEYWORDS = {
#     "corn": ["corn", "maize", "maíz"],
#     "wheat": ["wheat", "trigo"],
#     "soybeans": ["soybean", "soybeans", "soya", "soja"],
#     "rice": ["rice", "arroz"],
#     "oats": ["oats", "avena"],
#     "coffee": ["coffee", "café"],
#     "sugar": ["sugar", "azúcar"],
#     "cocoa": ["cocoa", "cacao"],
#     "cotton": ["cotton", "algodón"],
#     "orange_juice": ["orange juice", "jugo de naranja", "citrus"],
#     "live_cattle": ["cattle", "beef", "livestock", "ganado"],
#     "crude_oil_transport": ["crude oil", "oil", "petroleum", "petróleo", "brent", "wti"],
#     "natural_gas_fertilizer": ["natural gas", "gas natural", "fertilizer", "fertilizante", "ammonia"],
# }

# def mentions(text, commodities_dict):
#     """Busca menciones usando alias comunes y símbolos."""
#     found = []
#     text_lower = text.lower()
    
#     for asset_id, keywords in ASSET_KEYWORDS.items():
#         # 1. Verificar alias
#         for kw in keywords:
#             if re.search(rf"\b{re.escape(kw)}\b", text_lower):
#                 found.append(asset_id)
#                 break
                
#         # 2. Verificar ticker si no se encontró por alias (ej: ZC, ZW)
#         if asset_id not in found and asset_id in commodities_dict:
#             ticker_clean = str(commodities_dict[asset_id]).replace("=F", "")
#             if len(ticker_clean) >= 2 and re.search(rf"(?<![A-Za-z0-9]){re.escape(ticker_clean)}(?![A-Za-z0-9])", text, re.I):
#                 found.append(asset_id)
                
#     return found

# def refresh_news():
#     """Descarga noticias RSS sobre agricultura/commodities, analiza sentimiento e inserta en la BD."""
#     db = get_db()
    
#     # Feeds RSS de Commodities con parámetros para asegurar respuesta
#     feeds = {
#         "Yahoo Finance Commodities": getattr(settings, "rss_yahoo_commodities", "https://finance.yahoo.com/rss/headlines?s=ZC=F,ZW=F,KC=F,CC=F,CL=F,NG=F"),
#         "Investing Commodities": getattr(settings, "rss_investing_commodities", "https://www.investing.com/rss/news_11.rss"),
#     }
    
#     now = datetime.now(timezone.utc)
#     ops, counts = [], {}
    
#     # Simular User-Agent para evitar bloqueos HTTP en feedparser
#     agent_headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

#     for source, url in feeds.items():
#         try:
#             parsed = feedparser.parse(url, request_headers=agent_headers)
#             entries = parsed.entries[:50]
#             counts[source] = len(entries)
            
#             for entry in entries:
#                 title = clean_html(getattr(entry, "title", ""))
#                 summary = clean_html(getattr(entry, "summary", "") or getattr(entry, "description", ""))
#                 link = str(getattr(entry, "link", "")).strip()
#                 pub = published(entry)
                
#                 key = link or f"{source}|{title}|{pub.isoformat()}"
#                 news_id = hashlib.sha256(key.encode()).hexdigest()
#                 text = f"{title}. {summary}"
                
#                 doc = {
#                     "news_id": news_id,
#                     "source": source,
#                     "title": title,
#                     "summary": summary,
#                     "url": link,
#                     "published_at": pub,
#                     "mentioned_asset_ids": mentions(text, AGRICULTURAL_COMMODITIES),
#                     "ingested_at": now,
#                     **score(text),
#                 }
#                 ops.append(UpdateOne({"news_id": news_id}, {"$set": doc}, upsert=True))
#         except Exception as e:
#             print(f"⚠️ Error procesando feed '{source}': {e}")
#             counts[source] = 0

#     if ops:
#         db.news.bulk_write(ops, ordered=False)
        
#     return {"rows": len(ops), "per_source": counts}

# if __name__ == "__main__":
#     print("📰 Ejecutando actualización de noticias para commodities...")
#     res = refresh_news()
#     print(f"✅ Resultado: {res}")