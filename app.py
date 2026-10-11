from __future__ import annotations
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

for key in ("MONGODB_URI", "MONGODB_DB_NAME"):
    try:
        if key in st.secrets and not os.getenv(key):
            os.environ[key] = str(st.secrets[key])
    except Exception:
        pass

from src.db import ping, get_db
from src.dashboard import (
    latest_prediction, load_gemini_recommendations
)
from src.api_clients.yfinance_client import AGRICULTURAL_COMMODITIES

st.set_page_config(
    page_title="Agri-Supply Chain Analytics",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
        /* Ajusta el espacio superior del contenedor principal de Streamlit */
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
        }
        /* Asegura que la cabecera o banner respire correctamente */
        header[data-testid="stHeader"] {
            background: transparent !important;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.stApp {
    background-color: #F8FAFC;
}
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
}
/* Sidebar estilo Agro/Verde y Dorado */
[data-testid="stSidebar"] {
    background-color: #0F172A !important;
    border-right: 3px solid #16A34A !important;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, 
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stMarkdown {
    color: #F8FAFC !important;
    font-weight: 600;
}
[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background-color: #16A34A !important;
    border-color: #16A34A !important;
    border-radius: 6px;
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] div[data-baseweb="select"] span {
    color: #FFFFFF !important;
    font-weight: 700;
}

/* Tarjetas de Métricas */
[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-left: 4px solid #15803D;
    border-radius: 8px;
    padding: 12px 16px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.04);
}
[data-testid="stMetricLabel"] {
    color: #475569 !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
}
[data-testid="stMetricValue"] {
    color: #0F172A !important;
    font-size: 1.6rem !important;
    font-weight: 800 !important;
}

.section-title {
    color: #15803D;
    font-size: 1.4rem;
    font-weight: 700;
    margin-top: 1.5rem;
    margin-bottom: 1rem;
    border-bottom: 2px solid #22C55E;
    padding-bottom: 5px;
    display: inline-block;
}
</style>
""", unsafe_allow_html=True)

try:
    ping()
except Exception:
    st.error("No se pudo conectar a MongoDB Atlas.")
    st.stop()

# CABECERA
AGRO_BANNER_URL = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ7xMp_XASpg1M5WOOXfMh-Jc2JUaFFYtU4cojK8O2jJw&s=10"

st.markdown(f"""
<div style="display: flex; align-items: center; gap: 20px; margin-bottom: 20px;">
    <img src="{AGRO_BANNER_URL}" style="width: 250px; height: 100px; object-fit: cover; border-radius: 12px; box-shadow: 2px 4px 6px rgba(0,0,0,0.1);">
    <div>
        <h1 style="margin: 0; color: #0F172A; font-size: 2.3rem; font-weight: 800; line-height: 1.1;">Panel de Suministros Agrícolas</h1>
        <p style="margin: 0; color: #64748B; font-size: 1.05rem; font-weight: 500;">Monitoreo inteligente de insumos - Apoyo a decisiones logísticas</p>
    </div>
</div>
""", unsafe_allow_html=True)

# FILTROS Y BARRA LATERAL 
st.sidebar.markdown("### 🌾 Menú de Insumo")

db = get_db()

@st.cache_data(ttl=3600)
def load_catalog_map():
    docs = list(db.assets_catalog.find({}, {"_id": 0}))
    if not docs:
        return {
            "corn": {"name_es": "Maíz", "display_label": "Precio por 1 Bushel (25.4 kg)", "ticker": "ZC=F"},
            "wheat": {"name_es": "Trigo", "display_label": "Precio por 1 Bushel (27.2 kg)", "ticker": "ZW=F"},
            "soybeans": {"name_es": "Soya", "display_label": "Precio por 1 Bushel (27.2 kg)", "ticker": "ZS=F"},
            "rice": {"name_es": "Arroz", "display_label": "Precio por 1 Quintal (45.36 kg)", "ticker": "ZR=F"},
            "oats": {"name_es": "Avena", "display_label": "Precio por 1 Bushel (14.5 kg)", "ticker": "ZO=F"},
            "coffee": {"name_es": "Café", "display_label": "Precio por 1 Quintal (45.36 kg)", "ticker": "KC=F"},
            "sugar": {"name_es": "Azúcar", "display_label": "Precio por 1 Libra (0.45 kg)", "ticker": "SB=F"},
            "cocoa": {"name_es": "Cacao", "display_label": "Precio por 1 Tonelada (1,000 kg)", "ticker": "CC=F"},
            "cotton": {"name_es": "Algodón", "display_label": "Precio por 1 Libra (0.45 kg)", "ticker": "CT=F"},
            "crude_oil_transport": {"name_es": "Petróleo WTI (Transporte/Fletes)", "display_label": "Precio por 1 Barril (159 Litros)", "ticker": "CL=F"},
            "natural_gas_fertilizer": {"name_es": "Gas Natural (Fertilizantes)", "display_label": "Precio por 1 MMBtu (Energía)", "ticker": "NG=F"}
        }
    return {d["asset_id"]: d for d in docs}

catalog = load_catalog_map()

asset_options = {info.get("name_es", asset_key): asset_key for asset_key, info in catalog.items()}

selected_name = st.sidebar.selectbox(
    "Insumo Agrícola",
    options=list(asset_options.keys())
)

asset_id = asset_options[selected_name]
asset_info = catalog.get(asset_id, {})
ticker = asset_info.get("ticker", AGRICULTURAL_COMMODITIES.get(asset_id, ""))

range_days = st.sidebar.selectbox(
    "Histórico",
    [30, 90, 180, 365, 730, 1095],
    index=5, 
    format_func=lambda x: (
        f"{x} días" if x < 365 
        else ("1 año" if x == 365 else f"{x // 365} años")
    )
)

db = get_db()
ohlcv_docs = list(db.ohlcv.find({"asset_id": asset_id}).sort("date", 1))
hist_df = pd.DataFrame(ohlcv_docs)

if not hist_df.empty:
    date_col = next((col for col in ["date", "timestamp", "datetime"] if col in hist_df.columns), None)
    if date_col:
        hist_df[date_col] = pd.to_datetime(hist_df[date_col])
        hist_df = hist_df.sort_values(date_col)

pred_doc = db.predictions.find_one({"asset_id": asset_id})

news_docs = list(db.news.find({"mentioned_asset_ids": asset_id}).sort("published_at", -1).limit(50))
news_df = pd.DataFrame(news_docs)

# 1. MÉTRICAS PRINCIPALES 
asset_title = asset_info.get("name_es", asset_id.replace('_', ' ').title())
st.markdown(f"<div align='center'><h2 style='color: #0F172A; font-weight: 800;'>Métricas de Mercado: {asset_title}</h2></div>", unsafe_allow_html=True)

if not hist_df.empty and "close" in hist_df.columns:
    current_price = float(hist_df["close"].iloc[-1]) / 100.0
else:
    raw_pred_price = float(pred_doc.get("current_price", 0.0)) if pred_doc else 0.0
    current_price = raw_pred_price / 100.0

trend = pred_doc.get("trend", "N/A") if pred_doc else "N/A"
predicted_change = float(pred_doc.get("predicted_change_pct", 0.0)) if pred_doc else 0.0

unit_label = asset_info.get("display_label", "USD")

m1, m2, m3 = st.columns(3)

with m1:
    st.markdown(
        f"""
        <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-left: 4px solid #15803D; border-radius: 8px; padding: 12px 16px; box-shadow: 0 2px 4px rgba(0,0,0,0.04);">
            <div style="color: #475569; font-size: 0.9rem; font-weight: 600;">Precio Actual</div>
            <div style="display: flex; align-items: baseline; gap: 8px; margin-top: 2px;">
                <span style="color: #0F172A; font-size: 1.6rem; font-weight: 800;">${current_price:,.2f}</span>
                <span style="color: #334155; font-size: 1.1rem; font-weight: 600;">{unit_label}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with m2:
    m2.metric("Tendencia (180d)", trend)

with m3:
    m3.metric("Proyección Acumulada", f"{predicted_change:+.2f}%")

asset_id_actual = locals().get("asset_id", locals().get("selected_asset", "corn"))
nombre_actual = locals().get("name", locals().get("selected_name", asset_id_actual))

st.markdown(f"<h3 style='color: #0F172A;'>Dictamen Principal de IA para {nombre_actual}</h3>", unsafe_allow_html=True)

rec_data = load_gemini_recommendations()

if rec_data and "analisis_diario" in rec_data:
    dictamen_activo = next(
        (item for item in rec_data["analisis_diario"] if item.get("asset_id") == asset_id_actual), 
        None
    )
    
    if dictamen_activo:
        rec = dictamen_activo.get("recomendacion", "N/A")
        riesgo = dictamen_activo.get("nivel_riesgo", "N/A")
        tendencia = dictamen_activo.get("tendencia", "N/A")
       
        if rec in ["COMPRAR_ANTICIPADO", "COMPRAR"]:
            color_hex = "#10B981" 
        elif rec in ["ESPERAR_VALLE", "VENDER"]:
            color_hex = "#EF4444" 
        else:
            color_hex = "#F59E0B" 

        fecha_bruta = rec_data.get("fecha_actualizacion", "")
        fecha_formateada = fecha_bruta.replace("T", " ")[:19] if fecha_bruta else "Desconocida"

        st.markdown(f"""
        <div style="background-color: #FFFFFF; border: 2px solid {color_hex}; border-radius: 12px; padding: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 25px;">
            <div style="display:flex; justify-content: space-between; margin-bottom: 15px;">
                <div style="text-align: center; width: 33%;">
                    <p style="margin:0; color: #64748B; font-size: 0.9rem; font-weight:bold;">Recomendación</p>
                    <h3 style="margin:0; color: {color_hex}; font-size: 1.6rem;">{rec}</h3>
                </div>
                <div style="text-align: center; width: 33%; border-left: 1px solid #E2E8F0; border-right: 1px solid #E2E8F0;">
                    <p style="margin:0; color: #64748B; font-size: 0.9rem; font-weight:bold;">Nivel de Riesgo</p>
                    <h3 style="margin:0; color: #0F172A; font-size: 1.5rem;">{riesgo}</h3>
                </div>
                <div style="text-align: center; width: 33%;">
                    <p style="margin:0; color: #64748B; font-size: 0.9rem; font-weight:bold;">Tendencia</p>
                    <h3 style="margin:0; color: #0F172A; font-size: 1.5rem;">{tendencia}</h3>
                </div>
            </div>
            <div style="background-color: #F8FAFC; padding: 15px; border-radius: 8px; border-left: 4px solid {color_hex};">
                <p style="margin:0 0 10px 0; color: #334155; font-size: 0.95rem;"><strong>Resumen del Mercado:</strong> {dictamen_activo.get('resumen_ejecutivo', '')}</p>
                <p style="margin:0; color: #334155; font-size: 0.95rem;"><strong>Justificación de la recomendación:</strong> {dictamen_activo.get('justificacion', '')}</p>
            </div>
            <p style="margin:10px 0 0 0; text-align:right; font-size:0.75rem; color:#94A3B8;">Última corrida del Agente: {fecha_formateada} UTC</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info(f"No hay un dictamen generado para {nombre_actual} en la última corrida del agente.")
else:
    st.warning("No se encontraron recomendaciones recientes de IA en la base de datos.")


# 2. GRÁFICO HISTÓRICO
c_chart, c_news = st.columns([1.8, 1], gap="large")

with c_chart:
    st.markdown("<div class='section-title'>Comportamiento Histórico</div>", unsafe_allow_html=True)
    
    if not hist_df.empty:
        date_col = next((col for col in ["timestamp", "date", "time", "datetime"] if col in hist_df.columns), None)
        
        if date_col:
            hist_df["parsed_date"] = pd.to_datetime(hist_df[date_col], utc=True)
            plot_df = hist_df.sort_values("parsed_date").tail(range_days)
            
            price_col = "close" if "close" in plot_df.columns else ("price" if "price" in plot_df.columns else None)
            
            if price_col:
                fig = go.Figure()
                real_prices_usd = plot_df[price_col] / 100.0

                fig.add_trace(go.Scatter(
                    x=plot_df["parsed_date"], y=real_prices_usd, 
                    mode="lines", name="Precio Real (USD)", 
                    line=dict(color="#15803D", width=2.5)
                ))
                
                if pred_doc and "predictions_180d" in pred_doc:
                    preds_array = pred_doc["predictions_180d"]
                    if preds_array:
                        pred_dates = [pd.to_datetime(p["target_time"], utc=True) for p in preds_array]
                        pred_prices = [float(p["predicted_close"]) / 100.0 for p in preds_array]
                        
                        last_real_date = plot_df["parsed_date"].max()
                        last_real_price = float(real_prices_usd.iloc[-1])
                        
                        pred_dates.insert(0, last_real_date)
                        pred_prices.insert(0, last_real_price)

                        fig.add_trace(go.Scatter(
                            x=pred_dates,
                            y=pred_prices,
                            mode="lines", name="Pronóstico XGBoost (180d)",
                            line=dict(dash="dash", color="#EAB308", width=3)
                        ))
                
                fig.update_layout(
                    template="plotly_white",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#0F172A", size=12),  
                    height=420, 
                    margin=dict(l=10, r=10, t=20, b=10),
                    hovermode="x unified",
                    showlegend=True,
                    legend=dict(
                        font=dict(color="#0F172A", size=11),
                        bgcolor="rgba(255,255,255,0.8)"
                    ),
                    xaxis=dict(
                        showgrid=False, 
                        title="", 
                        tickfont=dict(color="#334155", size=11),  
                        linecolor="#CBD5E1"
                    ), 
                    yaxis=dict(
                        showgrid=True, 
                        gridcolor="#E2E8F0", 
                        title="USD / Bushel", 
                        tickfont=dict(color="#334155", size=11),  
                        zerolinecolor="#E2E8F0"
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No se encontró la columna de precios ('close' o 'price') en el histórico.")
        else:
            st.info("No se encontró una columna de fecha en la colección `ohlcv`.")
    else:
        st.info("Sin datos históricos disponibles en `ohlcv` para este commodity.")

with c_news:
    st.markdown("<div class='section-title'>Últimas Noticias Detectadas</div>", unsafe_allow_html=True)
    
    if not news_df.empty:
        recent_news = news_df.head(5)
        
        for _, n in recent_news.iterrows():
            title = n.get("title", "Sin título")
            url = n.get("url", "#")
            source = n.get("source", "Fuente Desconocida")
            sentiment = n.get("sentiment_label", "NEUTRAL")
         
            pub_date = n.get("published_at")
            pub_date_str = pd.to_datetime(pub_date).strftime("%d/%m/%Y %H:%M") if pub_date else "Reciente"
            
            if sentiment == "POSITIVO":
                badge = "<span style='background-color:#DCFCE7; color:#15803D; padding:2px 8px; border-radius:12px; font-size:0.75rem; font-weight:600;'>POSITIVO</span>"
            elif sentiment == "NEGATIVO":
                badge = "<span style='background-color:#FEE2E2; color:#B91C1C; padding:2px 8px; border-radius:12px; font-size:0.75rem; font-weight:600;'>NEGATIVO</span>"
            else:
                badge = "<span style='background-color:#F1F5F9; color:#475569; padding:2px 8px; border-radius:12px; font-size:0.75rem; font-weight:600;'>NEUTRAL</span>"
            
            st.markdown(
                f"""
                <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px; margin-bottom: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-size: 0.8rem; color: #64748B; font-weight: 500;">{source}</span>
                        {badge}
                    </div>
                    <a href="{url}" target="_blank" style="text-decoration: none; color: #0F172A; font-weight: 600; font-size: 0.88rem; line-height: 1.3; display: block; margin-bottom: 8px;">
                        {title}
                    </a>
                    <div style="font-size: 0.72rem; color: #94A3B8; border-top: 1px solid #F1F5F9; padding-top: 4px; text-align: right;">
                        Publicado: {pub_date_str}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.info("No hay noticias registradas recientemente para este activo.")