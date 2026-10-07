import feedparser
from datetime import datetime, timezone
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from src.db import get_db

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

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
       
        entries_to_process = parsed.entries[:limit_per_asset]
        
        for entry in entries_to_process:
            title = entry.get("title", "")
            url = entry.get("link", "#")
            
            mentioned_assets = tag_assets_in_title(title)
            if asset_key not in mentioned_assets:
                mentioned_assets.append(asset_key)
                
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


if __name__ == "__main__":
    refresh_news()

