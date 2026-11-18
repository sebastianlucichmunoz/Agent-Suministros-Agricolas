from __future__ import annotations
import os
import base64
import re
import uuid
import hmac
import hashlib
import secrets
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from urllib.parse import urlparse
from pathlib import Path
from html import escape
from pymongo import ASCENDING
from pymongo.errors import DuplicateKeyError, PyMongoError
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

for key in ("MONGODB_URI", "MONGODB_DB_NAME", "GEMINI_API_KEY", "SMTP_SERVER", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD"):
    try:
        if key in st.secrets and not os.getenv(key):
            os.environ[key] = str(st.secrets[key])
    except Exception:
        pass

from src.db import ping, get_db
from src.agents.consultation_agent import ejecutar_consulta, cupo_disponible, programar_consulta, resumen_r2_insumos
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
        
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
        }
        
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


# ===== ESTILOS DE ACCESO Y PERFIL =====
st.markdown("""
<style>
:root { color-scheme: light; }
div[data-testid="stDialog"] div[role="dialog"],
div[data-testid="stDialog"] div[role="dialog"] > div,
div[data-testid="stDialog"] div[data-testid="stForm"] {
    background: #FFFFFF !important;
    color: #0F172A !important;
    border-color: #DDE7E5 !important;
}
div[data-testid="stDialog"] div[role="dialog"] {
    max-width: 570px !important;
    border: 1px solid #D4E5D8 !important;
    box-shadow: 0 16px 48px rgba(15,23,42,.18) !important;
    border-radius: 16px !important;
}
div[data-testid="stDialog"] label,
div[data-testid="stDialog"] p,
div[data-testid="stDialog"] h3,
div[data-testid="stDialog"] span,
div[data-testid="stDialog"] [data-testid="stMarkdownContainer"] {
    color: #12304A !important;
}
div[data-testid="stDialog"] input,
div[data-testid="stDialog"] textarea,
div[data-testid="stDialog"] div[data-baseweb="select"] > div {
    background: #F8FAFC !important;
    color: #0F172A !important;
    border-color: #CBD5E1 !important;
    -webkit-text-fill-color: #0F172A !important;
}
div[data-testid="stDialog"] input:focus,
div[data-testid="stDialog"] textarea:focus,
div[data-testid="stDialog"] div[data-baseweb="select"] > div:focus-within {
    border-color: #15803D !important;
    box-shadow: 0 0 0 2px rgba(21,128,61,.15) !important;
}
div[data-testid="stDialog"] button[kind="primary"],
div[data-testid="stDialog"] button[data-testid="stBaseButton-primary"] {
    background: #16803B !important;
    border-color: #16803B !important;
    color: #FFFFFF !important;
}
div[data-testid="stDialog"] button[kind="primary"]:hover,
div[data-testid="stDialog"] button[data-testid="stBaseButton-primary"]:hover {
    background: #116B31 !important;
    color: #FFFFFF !important;
}
div[data-testid="stDialog"] button[kind="secondary"],
div[data-testid="stDialog"] button[data-testid="stBaseButton-secondary"] {
    background: #EAF4EE !important;
    color: #12422E !important;
    border-color: #BCD8C5 !important;
}
div[data-testid="stDialog"] button[kind="secondary"]:hover,
div[data-testid="stDialog"] button[data-testid="stBaseButton-secondary"]:hover {
    background: #D9EDDF !important;
    color: #0F3D2E !important;
}
.profile-title { color: #0F2747; font-size: 2rem; font-weight: 800; text-align: center; margin: 1.1rem 0 .35rem; }
.profile-line { width: 74px; height: 4px; margin: 0 auto 1rem; background: #F5BD13; border-radius: 5px; }
.profile-card-head { border-radius: 12px 12px 0 0; padding: 14px 15px; color: white; font-weight: 750; font-size: 1.05rem; }
.profile-blue { background: linear-gradient(115deg,#0C3559,#145678 84%,#F2C12E 85%); }
.profile-green { background: linear-gradient(115deg,#167435,#259746 84%,#F2C12E 85%); }
.profile-panel { background: #FFFFFF; border:1px solid #DEE7EC; border-radius: 0 0 12px 12px; padding: 17px; box-shadow: 0 5px 15px rgba(15,23,42,.06); min-height: 390px; }
.profile-row { display:grid; grid-template-columns: 42% 58%; gap:7px; border-bottom: 1px solid #E2E8F0; padding: 11px 0; font-size: .82rem; color:#0F172A; overflow-wrap:anywhere; }
.profile-row strong { color: #17436A; }
.profile-placeholder { background: #F2F8F5; border: 1px solid #D7E9DF; border-radius: 10px; padding: 14px; text-align:center; color:#617282; font-size:.9rem; }
.profile-avatar-box { min-height: 170px; display:flex; align-items:center; justify-content:center; }
.profile-history-empty { border: 1px dashed #A9C8BA; background: #F4F9F6; border-radius:10px; padding: 25px 12px; color: #557064; text-align:center; margin-top:14px; }
.profile-muted { color:#64748B; text-align:center; font-size:.84rem; margin-top:16px; }

/* Formularios: etiquetas visibles */
[data-testid="stMain"] .stForm label,
[data-testid="stMain"] .stTextArea label,
[data-testid="stMain"] .stMultiSelect label,
[data-testid="stMain"] .stCheckbox label,
[data-testid="stMain"] .stSelectbox label {
    color: #173454 !important;
    font-weight: 600 !important;
}

[data-testid="stMain"] label p,
[data-testid="stMain"] .stForm p {
    color: #173454 !important;
}

/* Campos de entrada */
[data-testid="stMain"] textarea,
[data-testid="stMain"] [data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    color: #0F2747 !important;
    border-color: #CBD5E1 !important;
}

[data-testid="stMain"] textarea {
    -webkit-text-fill-color: #0F2747 !important;
}

[data-testid="stMain"] textarea:focus,
[data-testid="stMain"] [data-baseweb="select"] > div:focus-within {
    border-color: #16803B !important;
    box-shadow: 0 0 0 2px rgba(22,128,59,.13) !important;
}

/* Insumos seleccionados */
[data-testid="stMain"] [data-baseweb="tag"] {
    background: #DCFCE7 !important;
    color: #166534 !important;
}

[data-testid="stMain"] [data-baseweb="tag"] span {
    color: #166534 !important;
}

/* Botón principal */
[data-testid="stMain"] button[kind="primary"],
[data-testid="stMain"] button[data-testid="stBaseButton-primary"] {
    background: #16803B !important;
    border: 1px solid #16803B !important;
    color: #FFFFFF !important;
}

[data-testid="stMain"] button[kind="primary"]:hover,
[data-testid="stMain"] button[data-testid="stBaseButton-primary"]:hover {
    background: #11652F !important;
    border-color: #11652F !important;
    color: #FFFFFF !important;
}

/* Mensajes de Streamlit */
[data-testid="stMain"] [data-testid="stAlert"] p {
    color: #173454 !important;
}

/* Respuesta del agente */
.consulta-respuesta {
    background: #FFFFFF;
    border: 1px solid #D7E5DC;
    border-left: 4px solid #16803B;
    border-radius: 12px;
    padding: 18px;
    margin-top: 12px;
    color: #173454;
    line-height: 1.7;
    overflow-wrap: anywhere;
}

.consulta-respuesta h4 {
    color: #126334;
    margin: 0 0 12px 0;
}

.consulta-respuesta p {
    color: #173454;
    margin: 0;
    white-space: pre-line;
}

/* Acción prioritaria */
.consulta-accion {
    background: #FFF9E6;
    border: 1px solid #F2D36D;
    border-left: 4px solid #EAB308;
    border-radius: 12px;
    padding: 16px;
    margin-top: 12px;
    color: #173454;
    overflow-wrap: anywhere;
}

.consulta-accion h4 {
    color: #755000;
    margin: 0 0 10px 0;
}

.consulta-accion p {
    color: #173454;
    line-height: 1.6;
    margin: 0;
    white-space: pre-line;
}
</style>
""", unsafe_allow_html=True)

# ===== ESTILOS DEL PANEL DE MERCADO =====
st.markdown("""
<style>
.agro-header {display:grid;grid-template-columns:180px minmax(0,1fr) 34px;align-items:center;gap:16px;background:linear-gradient(165deg,#FFFFFF 87%,#EAF6ED 88%,#F7CD38 100%);border:1px solid #E2E8F0;border-radius:16px;padding:16px 20px;box-shadow:0 5px 15px rgba(15,23,42,.07);margin-bottom:10px}
.agro-header-image {width:170px;height:82px;object-fit:cover;border-radius:11px}
.agro-header-text {text-align:center;min-width:0}
.agro-header-text h1 {font-size:clamp(1.35rem,2.5vw,2.05rem);font-weight:800;color:#0F2747;line-height:1.18;margin:0}
.agro-header-text p {margin:7px 0 0;color:#64748B;font-size:.95rem}
.agro-header-leaf {color:#15803D;font-size:1.65rem;text-align:center}
.market-title {color:#0F2747;text-align:center;font-size:1.7rem;font-weight:800;margin:8px 0 2px}
.market-subtitle {color:#64748B;text-align:center;font-size:.9rem;margin-bottom:16px}
.market-card {background:#FFFFFF;border:1px solid #E2E8F0;border-radius:12px;padding:16px 18px;box-shadow:0 3px 9px rgba(15,23,42,.045);min-height:104px}
.market-card .caption {font-size:.87rem;font-weight:650;color:#52637B}
.market-card .value {font-size:1.65rem;font-weight:800;color:#102441;margin:8px 0 2px}
.market-card .foot {color:#64748B;font-size:.76rem}
.dashboard-panel-title {color:#0F2747;font-size:1.15rem;font-weight:800;margin:6px 0 4px}
.dashboard-panel-subtitle {color:#64748B;font-size:.8rem;margin-bottom:10px}
.metric-side {background:#FFFFFF;border:1px solid #DDE6EE;border-radius:13px;padding:15px 14px;box-shadow:0 3px 10px rgba(15,23,42,.045)}
.metric-side h3 {color:#123453;font-size:1.13rem;font-weight:800;margin:0 0 5px}
.metric-side .context {color:#64748B;font-size:.76rem;margin-bottom:12px}
.metric-item {background:#F8FBFA;border:1px solid #E3ECE6;border-radius:10px;padding:11px 12px;margin:9px 0}
.metric-item .name {color:#53677A;font-size:.76rem;font-weight:700}
.metric-item .num {font-size:1.52rem;font-weight:800;color:#0F2747;line-height:1.2;margin:4px 0}
.metric-item .detail {color:#60748A;font-size:.73rem;line-height:1.35}
.metric-foot {font-size:.72rem;color:#64748B;line-height:1.4;margin-top:12px}
.news-card {background:#FFFFFF;border:1px solid #E2E8F0;border-radius:11px;padding:14px;min-height:152px;box-shadow:0 3px 10px rgba(15,23,42,.045)}
.news-card a {font-weight:700;font-size:.9rem;color:#142B46;text-decoration:none;line-height:1.4}
.news-card a:hover {color:#15803D;text-decoration:underline}
.news-meta {font-size:.76rem;color:#64748B;margin-bottom:10px}
.news-date {font-size:.73rem;color:#8191A4;margin-top:12px}
[data-testid="stSidebar"] div[data-baseweb="select"] > div {min-height:42px!important;background:#17304C!important;border:1px solid #436C66!important}
[data-testid="stSidebar"] div[data-baseweb="select"] > div:hover {background:#1E4058!important;border-color:#55C38A!important}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {font-size:.95rem!important}
@media(max-width:800px){.agro-header {grid-template-columns:75px 1fr;gap:10px;padding:12px}.agro-header-image{width:75px;height:65px}.agro-header-leaf{display:none}.agro-header-text h1{font-size:1.18rem}.agro-header-text p{font-size:.72rem}.market-title{font-size:1.35rem}}


/* ===== TÍTULOS BLANCOS DE ACCESO Y REGISTRO ===== */
div[data-testid="stDialog"] [role="dialog"] h2:first-of-type,
div[data-testid="stDialog"] [role="dialog"] h2:first-of-type *,
div[data-testid="stDialog"] [role="dialog"] [data-testid="stDialogHeader"] *,
div[data-testid="stDialog"] [role="dialog"] [data-testid="stDialogTitle"] {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

</style>
""", unsafe_allow_html=True)



# ===== AJUSTES VISUALES PARA EL GRÁFICO Y EL PERFIL =====
st.markdown("""
<style>
.agro-header {grid-template-columns:195px minmax(0,1fr) 36px !important;}
.agro-header-image {width:190px !important;height:90px !important;object-fit:cover !important;object-position:left center !important;}
.metric-side {padding:12px !important;border-radius:12px !important;}
.metric-side h3 {font-size:1rem !important;}
.metric-item {padding:10px 11px !important;margin:8px 0 !important;}
.metric-item .num {font-size:1.35rem !important;}
.metric-item .detail {color:#435B70 !important;font-size:.72rem !important;}
.consulta-guia {background:#F1F6FB;border-left:3px solid #1E6D91;border-radius:7px;color:#20415B !important;padding:9px 12px;margin:10px 0 13px;font-size:.84rem;line-height:1.55;}
.consulta-evaluacion {background:#F2F7FB;border:1px solid #DBE7F1;border-radius:9px;padding:10px 13px;margin-top:12px;color:#38536A;font-size:.8rem;line-height:1.65;}
.consulta-evaluacion strong {color:#17436A !important;}
.programacion-info {background:#FFF9E8;border-left:3px solid #D5A300;color:#715500;padding:10px 12px;border-radius:7px;margin:10px 0;font-size:.83rem;}
[data-testid="stMain"] [data-testid="stExpander"] summary p,
[data-testid="stMain"] [data-testid="stExpander"] summary span {color:#173454 !important;}
[data-testid="stMain"] [data-testid="stPlotlyChart"] {background:#FFFFFF;border:1px solid #DFE8F0;border-radius:13px;padding:8px 4px;}
[data-testid="stMain"] [data-testid="stPlotlyChart"] .legendtext {fill:#173454 !important;}
/* ===== LEGIBILIDAD DE ACCESO Y REGISTRO ===== */

/* ===== TEXTOS INTERNOS DE ACCESO Y REGISTRO ===== */
div[data-testid="stDialog"] label,
div[data-testid="stDialog"] [data-testid="stMarkdownContainer"] p,
div[data-testid="stDialog"] [data-testid="stMarkdownContainer"] h3 {
    color: #12304A !important;
}
/* ===== TEXTOS INTERNOS DE ACCESO Y REGISTRO ===== */
div[data-testid="stDialog"] [role="dialog"] label,
div[data-testid="stDialog"] [role="dialog"] label p,
div[data-testid="stDialog"] [role="dialog"] [data-testid="stCaptionContainer"] p {
    color: #123453 !important;
    -webkit-text-fill-color: #123453 !important;
}
div[data-testid="stDialog"] [role="dialog"] > div,
div[data-testid="stDialog"] [role="dialog"] [data-testid="stForm"] {background:#FFFFFF !important;}
div[data-testid="stDialog"] [role="dialog"] button[kind="primary"],
div[data-testid="stDialog"] [role="dialog"] button[kind="primary"] p,
div[data-testid="stDialog"] [role="dialog"] button[data-testid="stBaseButton-primary"] p {
    color:#FFFFFF !important;
    -webkit-text-fill-color:#FFFFFF !important;
}
div[data-testid="stDialog"] [role="dialog"] button[kind="secondary"] p,
div[data-testid="stDialog"] [role="dialog"] button[data-testid="stBaseButton-secondary"] p {
    color:#123453 !important;
    -webkit-text-fill-color:#123453 !important;
}
/* ===== CONSULTAS Y PANEL DE EVALUACIÓN ===== */
.consulta-success {background:#16783B;border:1px solid #126332;color:#FFFFFF !important;
    border-radius:10px;padding:12px 15px;margin:12px 0;font-size:.88rem;font-weight:650;}
.consulta-success * {color:#FFFFFF !important;-webkit-text-fill-color:#FFFFFF !important;}
.metric-side {padding:11px 12px !important;}
.metric-side h3 {font-size:1rem !important;line-height:1.2;margin-bottom:4px;}
.metric-side .context {font-size:.72rem !important;margin-bottom:8px !important;}
.metric-item {padding:8px 10px !important;margin:7px 0 !important;border-radius:9px !important;}
.metric-item .name {font-size:.74rem !important;line-height:1.2;}
.metric-item .num {font-size:1.2rem !important;margin:3px 0 !important;}
.metric-item .detail {font-size:.7rem !important;line-height:1.32 !important;}
.metric-foot {font-size:.67rem !important;line-height:1.35 !important;margin-top:7px !important;}
.consulta-evaluacion {background:#EFF8F0 !important;border-color:#BFDCC9 !important;
    border-left:3px solid #16803B !important;color:#123453 !important;}
.consulta-evaluacion strong,.consulta-evaluacion div {color:#123453 !important;}
[data-testid="stMain"] [data-testid="stExpander"] details,
[data-testid="stMain"] [data-testid="stExpander"] summary,
[data-testid="stMain"] [data-testid="stExpander"] summary:hover {background:#FFFFFF !important;}
[data-testid="stMain"] [data-testid="stExpander"] summary p {color:#17436A !important;-webkit-text-fill-color:#17436A !important;}
@media(max-width:850px){.agro-header {grid-template-columns:90px 1fr !important;}.agro-header-image{width:85px !important;height:65px !important;}}

/* ===== DISTRIBUCIÓN VERTICAL DE LAS MÉTRICAS ===== */
.metric-side {
    display: flex !important;
    flex-direction: column !important;
    min-height: 470px !important;
}
.metric-side .metric-item {
    flex: 1 !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
    min-height: 105px !important;
    margin: 8px 0 !important;
}

/* ===== TÍTULO BLANCO DEL DIÁLOGO ===== */
div[data-testid="stDialog"] h2[slot="title"] p {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

</style>
""", unsafe_allow_html=True)

# ===== CONEXIÓN A MONGODB =====
try:
    ping()
except Exception:
    st.error("No se pudo conectar a MongoDB Atlas.")
    st.stop()

db = get_db()

if "auth_view" not in st.session_state:
    st.session_state.auth_view = None
if "auth_user" not in st.session_state:
    st.session_state.auth_user = None
if "current_page" not in st.session_state:
    st.session_state.current_page = "dashboard"

# ===== UTILIDADES DE AUTENTICACIÓN Y USUARIOS =====
EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
INTEREST_NAMES = {
    "corn": "Maíz",
    "wheat": "Trigo",
    "soybeans": "Soya",
    "rice": "Arroz",
    "oats": "Avena",
    "coffee": "Café",
    "cocoa": "Cacao",
    "sugar": "Azúcar",
    "cotton": "Algodón",
    "crude_oil_transport": "Petróleo WTI",
    "natural_gas_fertilizer": "Gas natural",
}


def password_hash(password):
    salt = secrets.token_bytes(16)
    iterations = 600000
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}"


def verify_password(password, stored_hash):
    try:
        algorithm, rounds, salt_hex, digest_hex = stored_hash.split("$")
        if algorithm != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(rounds)
        )
        return hmac.compare_digest(digest, bytes.fromhex(digest_hex))
    except (ValueError, TypeError, AttributeError):
        return False


def switch_auth(view):
    st.session_state.auth_view = view
    st.rerun()


# ===== POPUPS DE INICIO DE SESIÓN Y REGISTRO =====
@st.dialog("Acceso al panel empresarial", width="small")
def auth_dialog():
    if st.session_state.auth_view == "login":
        st.markdown("### Bienvenido nuevamente")
        st.caption("Accede a tus consultas y recomendaciones personalizadas.")
        with st.form("login_form", clear_on_submit=False):
            email = st.text_input("Correo electrónico", placeholder="correo@empresa.com")
            password = st.text_input("Contraseña", type="password")
            submit = st.form_submit_button("Ingresar", use_container_width=True, type="primary")
        if submit:
            email = email.strip().lower()
            if not EMAIL_PATTERN.fullmatch(email) or not password:
                st.error("Ingresa un correo y una contraseña válidos.")
            else:
                try:
                    user = db["clientes_"].find_one({"email": email})
                    if not user or not verify_password(password, user.get("password_hash", "")):
                        st.error("Correo o contraseña incorrectos.")
                    elif not user.get("active", False):
                        st.error("La cuenta está inactiva.")
                    else:
                        st.session_state.auth_user = {
                            "client_id": user["client_id"],
                            "name": user.get("name", ""),
                            "email": user["email"],
                        }
                        st.session_state.auth_view = None
                        st.session_state.current_page = "profile"
                        st.rerun()
                except PyMongoError:
                    st.error("No se pudo completar el ingreso. Inténtalo nuevamente.")
        if st.button("¿No tienes una cuenta? Regístrate", use_container_width=True):
            switch_auth("register")
    else:
        st.markdown("### Regístrate como usuario")
        st.caption("Crea tu cuenta para consultar precios y recibir recomendaciones agrícolas.")
        with st.form("register_form", clear_on_submit=False):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Nombres")
            with col2:
                last_name = st.text_input("Apellidos")
            email = st.text_input("Correo electrónico", placeholder="correo@empresa.com")
            col1, col2 = st.columns(2)
            with col1:
                password = st.text_input("Contraseña (mínimo 8 caracteres)", type="password")
            with col2:
                confirmation = st.text_input("Confirmar contraseña", type="password")
            col1, col2 = st.columns(2)
            with col1:
                ocupacion = st.text_input("Ocupación", placeholder="Ej. Empresario agrícola")
            with col2:
                sexo = st.selectbox("Sexo", ["Masculino", "Femenino"])
            watchlist = st.multiselect(
                "Insumos y productos de interés",
                options=list(INTEREST_NAMES.keys()),
                format_func=lambda key: INTEREST_NAMES[key],
            )
            preferencia = st.text_area(
                "¿Qué información buscas obtener? (máximo 40 palabras)",
                height=85,
                placeholder="Ej. Conocer los mejores periodos para comprar maíz.",
            )
            submit = st.form_submit_button("Crear cuenta", use_container_width=True, type="primary")
        if submit:
            normalized_email = email.strip().lower()
            words = preferencia.strip().split()
            if not all([name.strip(), last_name.strip(), ocupacion.strip()]):
                st.error("Completa tus nombres, apellidos y ocupación.")
            elif not EMAIL_PATTERN.fullmatch(normalized_email) or len(normalized_email) > 254:
                st.error("Ingresa un correo electrónico válido.")
            elif not 8 <= len(password) <= 128:
                st.error("La contraseña debe tener entre 8 y 128 caracteres.")
            elif password != confirmation:
                st.error("Las contraseñas no coinciden.")
            elif not watchlist:
                st.error("Selecciona al menos un insumo de interés.")
            elif len(words) > 40:
                st.error("La preferencia puede contener un máximo de 40 palabras.")
            elif max(len(name.strip()), len(last_name.strip()), len(ocupacion.strip())) > 120:
                st.error("Los datos personales deben tener 120 caracteres como máximo por campo.")
            else:
                try:
                    collection = db["clientes_"]
                    collection.create_index([("email", ASCENDING)], unique=True)
                    if collection.find_one({"email": normalized_email}, {"_id": 1}):
                        st.warning("Este correo electrónico ya está registrado. Inicia sesión para continuar.")
                    else:
                        now = datetime.now(timezone.utc)
                        collection.insert_one({
                            "client_id": str(uuid.uuid4()),
                            "name": name.strip(),
                            "last_name": last_name.strip(),
                            "email": normalized_email,
                            "password_hash": password_hash(password),
                            "sexo": sexo,
                            "ocupacion": ocupacion.strip(),
                            "watchlist": watchlist,
                            "preferencia": " ".join(words),
                            "active": True,
                            "created_at": now,
                        })
                        st.success("Cuenta registrada correctamente. Ya puedes iniciar sesión.")
                except DuplicateKeyError:
                    st.warning("Este correo electrónico ya está registrado. Inicia sesión para continuar.")
                except PyMongoError:
                    st.error("No se pudo completar el registro. Inténtalo nuevamente.")
        if st.button("¿Ya tienes una cuenta? Inicia sesión", use_container_width=True):
            switch_auth("login")


# ===== CABECERA COMPARTIDA DEL DASHBOARD =====
# ===== IMAGEN LOCAL DE CABECERA =====
@st.cache_data(show_spinner=False)
def cargar_imagen_cabecera():
    candidatos = [
        Path(__file__).resolve().parent / "data" / "pictures" / "header_agro_banner.png",
        Path(__file__).resolve().parent / "data" / "pictures" / "agricultura_inteligente_y_cadena_de_suministro.png",
    ]
    for ruta in candidatos:
        if ruta.is_file():
            encoded = base64.b64encode(ruta.read_bytes()).decode("ascii")
            return f"data:image/png;base64,{encoded}"
    return "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ7xMp_XASpg1M5WOOXfMh-Jc2JUaFFYtU4cojK8O2jJw&s=10"


AGRO_BANNER_URL = cargar_imagen_cabecera()
st.markdown(f"""
<div class="agro-header">
    <img class="agro-header-image" src="{AGRO_BANNER_URL}" alt="Cultivos y cadena agroindustrial">
    <div class="agro-header-text">
        <h1>Panel de Suministros Agrícolas</h1>
        <p>Monitoreo inteligente · Apoyo a decisiones logísticas</p>
    </div>
    <span class="agro-header-leaf">✦</span>
</div>
""", unsafe_allow_html=True)

# ===== RUTAS DE IMÁGENES DEL PERFIL =====
PICTURES_DIR = Path(__file__).resolve().parent / "data" / "pictures"


def profile_avatar(sexo):
    if not PICTURES_DIR.is_dir():
        return None
    term = "mujer" if str(sexo).strip().lower() == "femenino" else "hombre"
    pictures = sorted(x for x in PICTURES_DIR.iterdir() if x.suffix.lower() == ".png")
    return next((p for p in pictures if term in p.stem.lower()), None)


# ===== MÉTRICAS EXPLICADAS AL USUARIO DESPUÉS DE LA CONSULTA =====
@st.cache_data(ttl=900, show_spinner=False)
def cargar_evaluacion_insumos(ids):
    return resumen_r2_insumos(get_db(), list(ids))



# ===== EVALUACIÓN SIMPLIFICADA DE LAS PREDICCIONES =====
def mostrar_evaluacion_consulta(registro):
    insumos = registro.get("insumos", []) or []
    resultados = registro.get("evaluacion_modelo") or cargar_evaluacion_insumos(tuple(insumos))

    filas = []

    for asset in insumos:
        item = resultados.get(asset) or {}
        nombre = escape(INTEREST_NAMES.get(asset, asset))

        try:
            r2 = float(item["r2"])
            if not np.isfinite(r2):
                continue
        except (KeyError, TypeError, ValueError):
            continue

        if 0 <= r2 <= 1:
            porcentaje = f"{r2 * 100:.1f}%"

            filas.append(
                f'<div style="padding:6px 0;color:#173454;">'
                f'📊 <strong>{nombre}:</strong> '
                f'<strong style="color:#15803D;">{porcentaje}</strong> '
                f'de confiabilidad de información'
                f'</div>'
            )

    if filas:
        st.markdown(
            '<div class="consulta-evaluacion">'
            '<div style="font-weight:750;color:#126334;margin-bottom:8px;">'
            '📈 Evaluación de las recomendaciones'
            '</div>'
            + "".join(filas)
            + '</div>',
            unsafe_allow_html=True,
        )


# ===== VISTA DEL PERFIL EMPRESARIAL =====
def render_profile():
    user_session = st.session_state.auth_user
    user = db["clientes_"].find_one({"client_id": user_session["client_id"]}, {"password_hash": 0})
    if not user or not user.get("active", False):
        st.session_state.auth_user = None
        st.session_state.current_page = "dashboard"
        st.warning("Tu sesión ya no está disponible.")
        st.rerun()

    title, go_back, logout = st.columns([2.1, 1.05, .8], vertical_alignment="center")
    with title:
        st.markdown('<div class="profile-title" style="text-align:left;margin-bottom:0;">🌱 Perfil empresarial</div><div style="height:4px;width:75px;background:#EAB308;border-radius:4px;"></div>', unsafe_allow_html=True)
    with go_back:
        if st.button("⌂ Volver al panel principal", use_container_width=True, key="back_dashboard"):
            st.session_state.current_page = "dashboard"
            st.rerun()
    with logout:
        if st.button("Cerrar sesión", type="primary", use_container_width=True, key="profile_logout"):
            st.session_state.auth_user = None
            st.session_state.auth_view = None
            for estado_key in ("ultima_consulta_ia", "consultation_assets", "consultation_question", "consultation_email", "scheduled_assets", "scheduled_question", "programacion_abierta"):
                st.session_state.pop(estado_key, None)
            st.session_state.current_page = "dashboard"
            st.rerun()

    st.write("")
    c_data, c_ai, c_history = st.columns([1.05, 1.0, 1.08], gap="medium")
    with c_data:
        st.markdown('<div class="profile-card-head profile-blue">♟ &nbsp; Datos del perfil</div>', unsafe_allow_html=True)
        st.markdown('<div class="profile-panel" style="min-height:0;padding-bottom:6px;">', unsafe_allow_html=True)
        img = profile_avatar(user.get("sexo", ""))
        if img:
            left, center, right = st.columns([1, 1.5, 1])
            with center:
                st.image(str(img), use_container_width=True)
        else:
            st.markdown('<div class="profile-avatar-box"><div style="font-size:5rem;">👤</div></div>', unsafe_allow_html=True)
        full_name = escape(" ".join(x for x in (user.get("name", ""), user.get("last_name", "")) if x))
        fields = [
            ("Nombres y apellidos", full_name),
            ("Correo electrónico", escape(str(user.get("email", "")))),
            ("Ocupación", escape(str(user.get("ocupacion", "")))),
            ("Sexo", escape(str(user.get("sexo", "")))),
            ("Insumos de interés", escape(", ".join(INTEREST_NAMES.get(k,k) for k in user.get("watchlist", []))))
        ]
        for label, value in fields:
            st.markdown(f'<div class="profile-row"><strong>{label}</strong><span>{value or "—"}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


    # ===== PERFIL: CONSULTAS INSTANTÁNEAS Y PROGRAMADAS =====
    with c_ai:
        st.markdown(
            '<div class="profile-card-head profile-green">'
            '✦ &nbsp; Consulta a la IA</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="consulta-guia">'
            'Consulta cotizaciones, compara insumos y conoce posibles '
            'cambios de precio durante los próximos <strong>6 meses</strong>. '
            'El asesor se centra en precios y decisiones de compra o venta.'
            '</div>',
            unsafe_allow_html=True,
        )

        favoritos = [
            k for k in user.get("watchlist", [])
            if k in INTEREST_NAMES
        ]

        opciones = favoritos + [
            k for k in INTEREST_NAMES if k not in favoritos
        ]

        # ===== SELECTOR DEL TIPO DE CONSULTA =====
        modo_consulta = st.segmented_control(
            "Tipo de consulta",
            options=["Instantáneas", "Programadas"],
            default="Instantáneas",
            key="tipo_consulta_ia",
            selection_mode="single",
            label_visibility="collapsed",
            width="stretch",
        )

        if modo_consulta == "Instantáneas":

            # ===== FORMULARIO DE CONSULTAS INSTANTÁNEAS =====
            try:
                cupo = cupo_disponible(user_session["client_id"])
                restantes = cupo["restantes"]
                limite = cupo["limite"]
            except Exception:
                restantes = 0
                limite = 10
                st.warning("No se pudo verificar el cupo disponible.")

            with st.container(border=True):
                st.markdown(
                    f"""
                    <div style="
                        background:#ECF8EF;
                        border:1px solid #C8E5D0;
                        border-radius:9px;
                        padding:11px 14px;
                        margin-bottom:14px;
                        color:#126334;
                        font-size:.85rem;
                        font-weight:650;">
                        🌱 Consultas disponibles hoy:
                        {restantes} de {limite}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                with st.form(
                    "consultation_form",
                    clear_on_submit=False,
                ):
                    insumos = st.multiselect(
                        "Insumos o productos (máximo 5)",
                        options=opciones,
                        default=favoritos[:1],
                        format_func=lambda k: INTEREST_NAMES.get(k, k),
                        key="consultation_assets",
                    )

                    pregunta = st.text_area(
                        "¿Qué deseas conocer sobre sus precios?",
                        height=148,
                        max_chars=600,
                        placeholder=(
                            "Ej.: ¿Me conviene comprar maíz ahora "
                            "o esperar al próximo mes?"
                        ),
                        key="consultation_question",
                    )

                    enviar_correo = st.checkbox(
                        "Enviar una copia de la respuesta a mi correo",
                        value=False,
                        key="consultation_email",
                    )


                    if restantes > 0:
                        enviado = st.form_submit_button(
                            "✦ Consultar a la IA",
                            use_container_width=True,
                            type="primary",
                        )
                    else:
                        enviado = False
                        st.markdown(
                            """
                            <div style="
                                background:#FFF8E1;
                                border:1px solid #E9D28C;
                                border-left:4px solid #E0A800;
                                border-radius:9px;
                                padding:12px 14px;
                                color:#1F2937;
                                font-size:.86rem;
                                font-weight:650;
                                text-align:center;">
                                Has alcanzado tu límite de consultas por hoy.
                                Podrás realizar nuevas consultas mañana.
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            # ===== PROCESAMIENTO DE LA CONSULTA =====
            if enviado:
                if not 1 <= len(insumos) <= 5:
                    st.error("Selecciona de 1 a 5 insumos.")

                elif not 12 <= len(pregunta.strip()) <= 600:
                    st.error(
                        "Escribe una pregunta de entre "
                        "12 y 600 caracteres."
                    )

                else:
                    with st.spinner(
                        "Analizando los precios y preparando "
                        "tu recomendación..."
                    ):
                        try:
                            resultado = ejecutar_consulta(
                                cliente_id=user_session["client_id"],
                                insumos=insumos,
                                pregunta=pregunta.strip(),
                                enviar_correo=enviar_correo,
                            )

                            st.session_state.ultima_consulta_ia = resultado

                            if resultado.get("estado") == "Completado":
                                st.success(
                                    "Consulta completada y registrada "
                                    "en el historial."
                                )
                            else:
                                st.error(
                                    "No fue posible completar la consulta."
                                )

                        except Exception as error:
                            st.session_state.pop(
                                "ultima_consulta_ia", None
                            )
                            st.error(
                                f"No se pudo procesar la consulta: {error}"
                            )

            # ===== RESPUESTA DE LA IA =====
            respuesta_actual = st.session_state.get(
                "ultima_consulta_ia"
            )

            if (
                respuesta_actual
                and respuesta_actual.get("cliente_id")
                == user_session["client_id"]
                and respuesta_actual.get("estado") == "Completado"
            ):
                texto = escape(
                    str(respuesta_actual.get("respuesta_ia") or "")
                )

                accion = escape(
                    str(respuesta_actual.get("accion_prioritaria") or "")
                )

                st.markdown(
                    f"""
                    <div class="consulta-respuesta">
                        <h4>🌱 Respuesta del asesor agrícola</h4>
                        <p>{texto}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""
                    <div class="consulta-accion">
                        <h4>🎯 Acción prioritaria</h4>
                        <p>{accion}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                mostrar_evaluacion_consulta(respuesta_actual)

                if respuesta_actual.get("enviar_correo"):
                    if respuesta_actual.get("correo_enviado"):
                        st.success(
                            "Se envió una copia a tu correo registrado."
                        )
                    else:
                        st.warning(
                            "La consulta se guardó, pero no se pudo "
                            "enviar la copia al correo."
                        )

        elif modo_consulta == "Programadas":

            # ===== FORMULARIO DE CONSULTAS PROGRAMADAS =====
            with st.container(border=True):
                st.markdown(
                    """
                    <div class="programacion-info">
                        <strong>
                            🕗 Consultas por correo a las 8:00 a. m.
                            (hora de Lima).
                        </strong>
                        Puedes programar hasta 10 consultas por entrega.
                        No consumen tu cupo de consultas instantáneas.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                with st.form(
                    "scheduled_consultation_form",
                    clear_on_submit=False,
                ):
                    insumos_prog = st.multiselect(
                        "Insumos para la consulta programada",
                        options=opciones,
                        default=favoritos[:1],
                        format_func=lambda k: INTEREST_NAMES.get(k, k),
                        key="scheduled_assets",
                    )

                    pregunta_prog = st.text_area(
                        "¿Qué deseas consultar en la próxima entrega?",
                        height=115,
                        max_chars=600,
                        placeholder=(
                            "Ej.: Compara la evolución prevista "
                            "del trigo y del maíz para los "
                            "próximos tres meses."
                        ),
                        key="scheduled_question",
                    )

                    st.caption(
                        "La respuesta se enviará al correo registrado. "
                        "Si el procesamiento falla, no se enviará "
                        "una respuesta incompleta."
                    )

                    programar = st.form_submit_button(
                        "🕗 Programar consulta",
                        use_container_width=True,
                        type="primary",
                    )

            # ===== REGISTRO DE LA PROGRAMACIÓN =====
            if programar:
                try:
                    registro = programar_consulta(
                        cliente_id=user_session["client_id"],
                        insumos=insumos_prog,
                        pregunta=pregunta_prog.strip(),
                    )

                    fecha_local = registro[
                        "programada_para"
                    ].astimezone(ZoneInfo("America/Lima"))

                    st.success(
                        f"Consulta programada para el "
                        f"{fecha_local:%d/%m/%Y} a las 08:00 "
                        f"(hora de Lima)."
                    )

                except Exception as error:
                    st.error(
                        f"No se pudo programar la consulta: {error}"
                    )


    # ===== PERFIL: HISTORIAL DE CONSULTAS ALMACENADAS =====
    with c_history:
        st.markdown('<div class="profile-card-head profile-blue">▤ &nbsp; Historial de consultas</div>', unsafe_allow_html=True)
        st.markdown('<p style="color:#526780;font-size:.86rem;margin:12px 0;">Consulta tus preguntas anteriores y las próximas entregas.</p>', unsafe_allow_html=True)
        try:
            historial = list(
                db.consultas_ia.find(
                    {"cliente_id": user_session["client_id"]},
                    {
                        "_id": 0, "consulta_id": 1, "fecha_consulta": 1,
                        "insumos": 1, "pregunta": 1, "respuesta_ia": 1,
                        "accion_prioritaria": 1, "estado": 1, "tipo": 1,
                        "programada_para": 1, "enviar_correo": 1,
                        "correo_enviado": 1, "evaluacion_modelo": 1,
                    },
                ).sort("fecha_consulta", -1).limit(120)
            )
            zona = ZoneInfo("America/Lima")
            for item in historial:
                fecha = item.get("fecha_consulta")
                if isinstance(fecha, datetime):
                    if fecha.tzinfo is None:
                        fecha = fecha.replace(tzinfo=timezone.utc)
                    item["fecha_local"] = fecha.astimezone(zona)
            fechas = [r["fecha_local"] for r in historial if "fecha_local" in r]
            anios = sorted({d.year for d in fechas}, reverse=True)
            meses = sorted({d.month for d in fechas})
            col_m, col_y = st.columns(2, gap="small")
            with col_m:
                mes = st.selectbox("Mes", ["Todos"] + meses, key="hist_month")
            with col_y:
                anio = st.selectbox("Año", ["Todos"] + anios, key="hist_year")
            filtrados = []
            for item in historial:
                fecha = item.get("fecha_local")
                if fecha is None:
                    continue
                if mes != "Todos" and fecha.month != mes:
                    continue
                if anio != "Todos" and fecha.year != anio:
                    continue
                filtrados.append(item)
            st.markdown(
                f'<div style="color:#526780;font-size:.82rem;margin:12px 0;">📋 {len(filtrados)} consulta(s) encontrada(s)</div>',
                unsafe_allow_html=True,
            )
            if not filtrados:
                st.markdown('<div class="profile-history-empty">📋<div style="font-weight:700;margin:8px 0 4px;">No hay consultas para este filtro</div>Tus consultas aparecerán aquí.</div>', unsafe_allow_html=True)
            for item in filtrados[:30]:
                fecha = item["fecha_local"].strftime("%d/%m/%Y %H:%M")
                nombres = ", ".join(INTEREST_NAMES.get(k, k) for k in item.get("insumos", [])) or "Insumo sin identificar"
                estado = item.get("estado", "Error")
                simbolo = {"Completado": "✅", "Error": "⚠️", "Programada": "🕗", "Procesando": "⏳"}.get(estado, "•")
                tipo = " · Programada" if item.get("tipo") == "programada" else ""
                with st.expander(f"{simbolo} {fecha} · {nombres}{tipo}", expanded=False):
                    st.markdown('<div style="font-weight:750;color:#126334;margin:7px 0 6px;">Pregunta realizada</div>', unsafe_allow_html=True)
                    st.markdown(
                        f'<div style="background:#F1F5F9;color:#173454;padding:12px;border-radius:9px;line-height:1.6;overflow-wrap:anywhere;white-space:pre-line;">{escape(str(item.get("pregunta") or ""))}</div>',
                        unsafe_allow_html=True,
                    )
                    if estado == "Completado":
                        texto = escape(str(item.get("respuesta_ia") or ""))
                        accion = escape(str(item.get("accion_prioritaria") or ""))
                        st.markdown(f'<div class="consulta-respuesta"><h4>🌱 Respuesta del asesor agrícola</h4><p>{texto}</p></div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="consulta-accion"><h4>🎯 Acción prioritaria</h4><p>{accion}</p></div>', unsafe_allow_html=True)
                        mostrar_evaluacion_consulta(item)
                        if item.get("enviar_correo"):
                            st.caption("✉️ Copia enviada al correo registrado." if item.get("correo_enviado") else "No se confirmó el envío por correo.")
                        else:
                            st.caption("No se solicitó copia por correo.")
                    elif estado in ("Programada", "Procesando"):
                        cuando = item.get("programada_para")
                        if isinstance(cuando, datetime):
                            when_local = (cuando.replace(tzinfo=timezone.utc) if cuando.tzinfo is None else cuando).astimezone(zona)
                            st.info(f"Consulta {estado.lower()}. Entrega prevista: {when_local:%d/%m/%Y a las %H:%M} (Lima).")
                        else:
                            st.info(f"Consulta {estado.lower()}.")
                    else:
                        st.warning("No fue posible completar esta consulta.")
            if len(filtrados) > 30:
                st.caption("Se muestran las 30 consultas más recientes que cumplen los filtros.")
        except PyMongoError:
            st.error("No se pudo cargar el historial de consultas.")

# ===== NAVEGACIÓN ENTRE EL PANEL Y EL PERFIL =====
if st.session_state.auth_user and st.session_state.current_page == "profile":
    render_profile()
    st.stop()

# ===== ACCESO ALINEADO A LA DERECHA SIN ABRIR SESIÓN AUTOMÁTICAMENTE =====
def cerrar_sesion_dashboard():
    st.session_state.auth_user = None
    st.session_state.auth_view = None
    for estado_key in ("ultima_consulta_ia", "consultation_assets", "consultation_question", "consultation_email", "scheduled_assets", "scheduled_question", "programacion_abierta"):
        st.session_state.pop(estado_key, None)
    st.session_state.current_page = "dashboard"
    st.rerun()

if st.session_state.auth_user is None:
    espacio_cabecera, acceso_cabecera = st.columns([5, 1.25], gap="small")
    with acceso_cabecera:
        if st.button("Ingresar como usuario", use_container_width=True, key="open_login"):
            st.session_state.auth_view = "login"
            st.rerun()
else:
    espacio_cabecera, perfil_cabecera, salida_cabecera = st.columns([4.7, 1.15, 1.15], gap="small")
    with perfil_cabecera:
        if st.button("Mi perfil", use_container_width=True, key="open_profile"):
            st.session_state.current_page = "profile"
            st.rerun()
    with salida_cabecera:
        if st.button("Cerrar sesión", use_container_width=True, key="logout_dashboard"):
            cerrar_sesion_dashboard()

if st.session_state.auth_view is not None and st.session_state.auth_user is None:
    auth_dialog()

# ===== DASHBOARD PÚBLICO: FILTROS Y DATOS =====
st.sidebar.markdown("### 🌾 Menú")

@st.cache_data(ttl=3600)
def load_catalog_map():
    database = get_db()
    docs = list(database.assets_catalog.find({}, {"_id": 0}))
    fallback = {
        "corn": {"name_es": "Maíz", "display_label": "Por bushel (25.4 kg)", "ticker": "ZC=F"},
        "wheat": {"name_es": "Trigo", "display_label": "Por bushel (27.2 kg)", "ticker": "ZW=F"},
        "soybeans": {"name_es": "Soya", "display_label": "Por bushel (27.2 kg)", "ticker": "ZS=F"},
        "rice": {"name_es": "Arroz", "display_label": "Por quintal (45.36 kg)", "ticker": "ZR=F"},
        "oats": {"name_es": "Avena", "display_label": "Por bushel (14.5 kg)", "ticker": "ZO=F"},
        "coffee": {"name_es": "Café", "display_label": "Por libra (0.45 kg)", "ticker": "KC=F"},
        "sugar": {"name_es": "Azúcar", "display_label": "Por libra (0.45 kg)", "ticker": "SB=F"},
        "cocoa": {"name_es": "Cacao", "display_label": "Por tonelada (1,000 kg)", "ticker": "CC=F"},
        "cotton": {"name_es": "Algodón", "display_label": "Por libra (0.45 kg)", "ticker": "CT=F"},
        "crude_oil_transport": {"name_es": "Petróleo WTI", "display_label": "Por barril (159 L)", "ticker": "CL=F"},
        "natural_gas_fertilizer": {"name_es": "Gas natural", "display_label": "Por MMBtu", "ticker": "NG=F"},
    }
    for doc in docs:
        key = doc.get("asset_id")
        if key:
            fallback[key] = {**fallback.get(key, {}), **doc}
    return fallback


# ===== UNIDADES Y FECHAS DE COTIZACIÓN =====
CENT_DENOMINATED_ASSETS = {"corn", "wheat", "soybeans", "oats", "coffee", "sugar", "cotton"}


def price_factor(asset_key):
    return 0.01 if asset_key in CENT_DENOMINATED_ASSETS else 1.0


def display_money(raw_value, factor):
    try:
        value = float(raw_value) * factor
        if not np.isfinite(value):
            return "N/D"
        return f"${value:,.2f}" if abs(value) >= 1 else f"${value:,.4f}"
    except (TypeError, ValueError):
        return "N/D"


def format_day(value):
    if value is None:
        return "N/D"
    try:
        return pd.to_datetime(value, utc=True).strftime("%d/%m/%Y")
    except (ValueError, TypeError):
        return "N/D"


def safe_number(value):
    try:
        result = float(value)
        return result if np.isfinite(result) else None
    except (TypeError, ValueError):
        return None


catalog = load_catalog_map()
asset_options = {info.get("name_es", asset_key): asset_key for asset_key, info in catalog.items()}

def cerrar_dialogo_en_filtros():
    st.session_state.auth_view = None

selected_name = st.sidebar.selectbox(
    "Insumo o producto", options=list(asset_options.keys()),
    on_change=cerrar_dialogo_en_filtros,
)
asset_id = asset_options[selected_name]
asset_info = catalog.get(asset_id, {})
asset_title = asset_info.get("name_es", selected_name)
unit_label = asset_info.get("display_label", "Unidad de cotización")
conversion = price_factor(asset_id)

period_choices = {
    "30d": ("Últimos 30 días", 30),
    "90d": ("Últimos 90 días", 90),
    "180d": ("Últimos 6 meses", 180),
    "1y": ("Último año", 365),
    "3y": ("Últimos 3 años", 1095),
    "5y": ("Últimos 5 años", 1826),
    "10y": ("Últimos 10 años", 3653),
    "all": ("Todo el historial", None),
}
selected_period = st.sidebar.selectbox(
    "Periodo de análisis", options=list(period_choices), index=4,
    format_func=lambda key: period_choices[key][0],
    on_change=cerrar_dialogo_en_filtros,
)

@st.cache_data(ttl=900)
def load_market_data(key):
    database = get_db()
    rows = list(database.ohlcv.find(
        {"asset_id": key, "period": "1d"}, {"_id": 0, "timestamp": 1, "close": 1}
    ).sort("timestamp", 1))
    return pd.DataFrame(rows)


hist_df = load_market_data(asset_id)
if not hist_df.empty and {"timestamp", "close"}.issubset(hist_df.columns):
    hist_df["parsed_date"] = pd.to_datetime(hist_df["timestamp"], utc=True, errors="coerce")
    hist_df["close"] = pd.to_numeric(hist_df["close"], errors="coerce")
    hist_df = (hist_df.dropna(subset=["parsed_date", "close"])
              .loc[lambda d: d["close"] > 0]
              .sort_values("parsed_date")
              .drop_duplicates("parsed_date", keep="last").reset_index(drop=True))
else:
    hist_df = pd.DataFrame(columns=["parsed_date", "close"])

last_trade = hist_df["parsed_date"].iloc[-1] if not hist_df.empty else None
period_length = period_choices[selected_period][1]
if period_length is not None and last_trade is not None:
    period_start = last_trade - pd.Timedelta(days=period_length)
    plot_df = hist_df.loc[hist_df["parsed_date"] >= period_start].copy()
else:
    plot_df = hist_df.copy()

if not plot_df.empty:
    st.sidebar.caption(f"Datos: {format_day(plot_df['parsed_date'].iloc[0])} – {format_day(last_trade)}")
else:
    st.sidebar.caption("No hay cotizaciones disponibles para este periodo")

pred_doc = db.predictions.find_one({"asset_id": asset_id}, sort=[("predicted_at", -1)]) or {}
metrics_doc = db.model_metrics.find_one({"asset_id": asset_id}, sort=[("trained_at", -1)]) or {}
news_docs = list(db.news.find({"mentioned_asset_ids": asset_id}).sort("published_at", -1).limit(12))

# ===== PANORAMA Y TARJETAS PRINCIPALES =====
st.markdown(
    f'<div class="market-title">Panorama del mercado: <span style="color:#15803D">{escape(str(asset_title))}</span></div>'
    '<div class="market-subtitle">Cotización actual, comportamiento histórico y proyección a seis meses</div>',
    unsafe_allow_html=True,
)
raw_current = float(hist_df["close"].iloc[-1]) if not hist_df.empty else safe_number(pred_doc.get("current_price"))
trend = str(pred_doc.get("trend", "N/D"))
predicted_change = safe_number(pred_doc.get("predicted_change_pct"))
last_quote_label = format_day(last_trade)
model_base_date = pd.to_datetime(pred_doc.get("base_date") or pred_doc.get("model_trained_at"), utc=True, errors="coerce")
quote_ahead_of_model = bool(pd.notna(model_base_date) and last_trade is not None and last_trade.normalize() > model_base_date.normalize())

m1, m2, m3 = st.columns(3, gap="medium")
with m1:
    st.markdown(f'''<div class="market-card" style="border-left:4px solid #16A34A">
    <div class="caption">Precio actual</div><div class="value">{display_money(raw_current, conversion)}</div>
    <div class="foot">{escape(str(unit_label))} · Última cotización: {last_quote_label}</div></div>''', unsafe_allow_html=True)
with m2:
    trend_color = "#B91C1C" if trend.upper() == "BAJISTA" else "#15803D" if trend.upper() == "ALCISTA" else "#0F2747"
    st.markdown(f'''<div class="market-card" style="border-left:4px solid #EAB308">
    <div class="caption">Tendencia proyectada</div><div class="value" style="color:{trend_color}">{escape(trend)}</div>
    <div class="foot">Según la última predicción disponible</div></div>''', unsafe_allow_html=True)
with m3:
    pct_text = f"{predicted_change:+.2f}%" if predicted_change is not None else "N/D"
    pct_color = "#B91C1C" if predicted_change is not None and predicted_change < 0 else "#15803D"
    st.markdown(f'''<div class="market-card" style="border-left:4px solid #0E7490">
    <div class="caption">Proyección acumulada · 6 meses</div><div class="value" style="color:{pct_color}">{pct_text}</div>
    <div class="foot">Variación estimada respecto al precio base</div></div>''', unsafe_allow_html=True)

# ===== DICTAMEN GENERAL DE INTELIGENCIA ARTIFICIAL =====
st.markdown(f'<div class="dashboard-panel-title">🤖 Dictamen de IA para {escape(str(asset_title))}</div>', unsafe_allow_html=True)
rec_data = db.agent_recommendations.find_one({}, sort=[("fecha_actualizacion", -1)]) or load_gemini_recommendations()
if rec_data and isinstance(rec_data.get("analisis_diario"), list):
    dictamen_activo = next(
        (item for item in rec_data["analisis_diario"] if item.get("asset_id") == asset_id), None
    )
    if dictamen_activo:
        recommendation = str(dictamen_activo.get("recomendacion", "N/D")).replace("_", " ")
        risk = str(dictamen_activo.get("nivel_riesgo", "N/D"))
        agent_trend = str(dictamen_activo.get("tendencia", "N/D"))
        summary = escape(str(dictamen_activo.get("resumen_ejecutivo", "")))
        justification = escape(str(dictamen_activo.get("justificacion", "")))
        agent_updated = rec_data.get("fecha_actualizacion")
        agent_date_label = format_day(agent_updated)
        agent_color = ("#15803D" if "COMPRAR" in recommendation.upper() else
                       "#B45309" if "MANTENER" in recommendation.upper() else "#B91C1C")
        st.markdown(f'''<div style="background:#FFFFFF;border:1px solid #E5D28A;border-left:4px solid #EAB308;border-radius:13px;padding:17px 21px;margin:8px 0 13px;box-shadow:0 3px 10px rgba(15,23,42,.045)">
            <div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;text-align:center;border-bottom:1px solid #E2E8F0;padding-bottom:14px;margin-bottom:13px">
              <div><div style="font-size:.78rem;color:#64748B;font-weight:700">Recomendación</div><div style="font-size:1.18rem;font-weight:800;color:{agent_color};margin-top:5px">{escape(recommendation)}</div></div>
              <div style="border-left:1px solid #E2E8F0"><div style="font-size:.78rem;color:#64748B;font-weight:700">Nivel de riesgo</div><div style="font-size:1.08rem;color:#0F2747;font-weight:700;margin-top:5px">{escape(risk)}</div></div>
              <div style="border-left:1px solid #E2E8F0"><div style="font-size:.78rem;color:#64748B;font-weight:700">Tendencia esperada</div><div style="font-size:1.08rem;color:#0F2747;font-weight:700;margin-top:5px">{escape(agent_trend)}</div></div>
            </div>
            <div style="font-size:.88rem;color:#334155;line-height:1.65"><strong>Resumen del mercado:</strong> {summary}</div>
            <div style="font-size:.85rem;color:#475569;line-height:1.65;margin-top:8px"><strong>Justificación:</strong> {justification}</div>
            <div style="font-size:.72rem;color:#94A3B8;text-align:right;margin-top:11px">Dictamen actualizado: {agent_date_label}</div>
        </div>''', unsafe_allow_html=True)
        prediction_date = pd.to_datetime(pred_doc.get("model_trained_at"), utc=True, errors="coerce")
        agent_date = pd.to_datetime(agent_updated, utc=True, errors="coerce")
        if pd.notna(prediction_date) and pd.notna(agent_date) and agent_date < prediction_date:
            st.info("El dictamen de IA es anterior al modelo que generó la predicción actual. Sus cifras pueden diferir de las tarjetas de mercado.")
    else:
        st.info(f"No se encontró un dictamen de IA para {asset_title}.")
else:
    st.info("No hay un dictamen de IA disponible para este insumo.")

# ===== GRÁFICO INTEGRADO: HISTÓRICO Y PROYECCIÓN =====
st.markdown('<div class="dashboard-panel-title">Evolución del precio y proyección a 6 meses</div>', unsafe_allow_html=True)
st.markdown('<div class="dashboard-panel-subtitle">Precios reales en verde · Trayectoria estimada en amarillo · El marcador señala la última cotización disponible</div>', unsafe_allow_html=True)
chart_col, metric_col = st.columns([4.55, 1.0], gap="small")

with chart_col:
    if not plot_df.empty:
        fig = go.Figure()
        real_dates = plot_df["parsed_date"]
        real_prices = plot_df["close"].astype(float) * conversion
        fig.add_trace(go.Scatter(
            x=real_dates, y=real_prices, mode="lines", name="Precio histórico",
            line=dict(color="#15803D", width=2.2),
            hovertemplate="<b>Precio histórico</b><br>%{x|%d/%m/%Y}<br>Precio: $%{y:,.4f}<extra></extra>",
        ))
        last_price = float(real_prices.iloc[-1])
        fig.add_trace(go.Scatter(
            x=[last_trade], y=[last_price], mode="markers", name="Última cotización",
            marker=dict(size=12, color="#16A34A", line=dict(color="#FFFFFF", width=2)),
            hovertemplate="<b>Última cotización</b><br>%{x|%d/%m/%Y}<br>Precio: $%{y:,.4f}<extra></extra>",
        ))
        fig.add_vline(x=last_trade.timestamp() * 1000, line_color="#64748B", line_dash="dot", line_width=1)
        fig.add_annotation(
            x=last_trade,
            xref="x",
            y=0.98,
            yref="paper",
            text=(
                f"<b>Última cotización</b><br>"
                f"{format_day(last_trade)} · "
                f"{display_money(raw_current, conversion)}"
            ),
            showarrow=False,
            xanchor="right",
            xshift=-8,
            yanchor="top",
            align="center",
            font=dict(size=9, color="#17436A"),
            bgcolor="rgba(255,255,255,0.96)",
            bordercolor="#BDD9C8",
            borderwidth=1,
            borderpad=4,
        )
        forecasts = pred_doc.get("predictions_180d") or pred_doc.get("forecast") or []
        future_records = []
        for forecast in forecasts:
            when = forecast.get("target_time", forecast.get("date"))
            value = safe_number(forecast.get("predicted_close"))
            when = pd.to_datetime(when, utc=True, errors="coerce")
            if pd.notna(when) and value is not None and when > last_trade:
                future_records.append((when, value * conversion))
        future_records.sort(key=lambda item: item[0])
        if future_records:
            forward_dates = [last_trade] + [point[0] for point in future_records]
            forward_prices = [last_price] + [point[1] for point in future_records]
            fig.add_trace(go.Scatter(
                x=forward_dates, y=forward_prices, mode="lines", name="Proyección multimodelo",
                line=dict(color="#E5A800", width=3, dash="dash"),
                hovertemplate="<b>Precio estimado</b><br>%{x|%d/%m/%Y}<br>Precio: $%{y:,.4f}<extra></extra>",
            ))
            fig.add_vrect(
                x0=last_trade.timestamp() * 1000,
                x1=forward_dates[-1].timestamp() * 1000,
                fillcolor="#F4C746", opacity=.07, line_width=0, layer="below",
            )
        else:
            st.caption("Todavía no se dispone de una proyección posterior a la última cotización.")
        fig.update_layout(
            template="plotly_white", height=520, margin=dict(l=22, r=16, t=52, b=26),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#17304B", size=13), hovermode="closest",
            hoverlabel=dict(bgcolor="#173454", bordercolor="#15803D", font=dict(color="#FFFFFF", size=13)),
            legend=dict(orientation="h", yanchor="bottom", y=1.025, xanchor="left", x=0,
                        font=dict(size=12, color="#173454"), bgcolor="#F3F7FB", bordercolor="#D8E4ED", borderwidth=1),
            xaxis=dict(title=dict(text="Fecha", font=dict(color="#173454", size=13)),
                       showgrid=False, showline=True, linewidth=1, linecolor="#8CA3B5",
                       tickfont=dict(color="#35546B", size=12), tickformat="%b %Y",
                       rangeslider=dict(visible=False)),
            yaxis=dict(title=dict(text=f"USD · {unit_label}", font=dict(color="#173454", size=13)),
                       tickfont=dict(color="#35546B", size=12),
                       showline=True, linecolor="#8CA3B5", gridcolor="#E3EBF2", zeroline=False),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
        st.caption(f"Último precio real: {format_day(last_trade)} · {display_money(raw_current, conversion)}. Los valores entre horizontes predictivos son interpolados, no estimaciones independientes por día.")
    else:
        st.info("No se encontraron datos históricos de mercado para el insumo seleccionado.")

# ===== MÉTRICAS DE CONFIABILIDAD POR INSUMO =====
def metric_card(title, value, detail):
    return f'''<div class="metric-item"><div class="name">{escape(title)}</div>
      <div class="num">{escape(value)}</div><div class="detail">{escape(detail)}</div></div>'''


def extrema_description(entry):
    if not entry:
        return "Sin evaluación disponible"
    return (f"{entry.get('horizon_sessions', 'N/D')} sesiones · "
            f"{str(entry.get('model', 'N/D')).upper()} · {entry.get('period', '')}")


with metric_col:
    resumen = metrics_doc.get("summary") or {}
    horizontes = metrics_doc.get("metrics_by_horizon") or {}
    entrenadas = [(int(h), m) for h, m in horizontes.items()
                  if str(h).isdigit() and isinstance(m, dict) and m.get("status") == "trained"]
    mejores_r2 = max(
        ((h, safe_number(m.get("r2_price")), m) for h, m in entrenadas
         if safe_number(m.get("r2_price")) is not None),
        key=lambda v: v[1], default=None,
    )
    mejores_direccion = max(
        ((h, safe_number(m.get("directional_accuracy_pct")), m) for h, m in entrenadas
         if safe_number(m.get("directional_accuracy_pct")) is not None),
        key=lambda v: v[1], default=None,
    )
    dato_wape = resumen.get("best_wape") or {}
    dato_mae = resumen.get("best_mae") or {}
    r2 = mejores_r2[1] if mejores_r2 else None
    if r2 is not None and 0 <= r2 <= 1:
        r2_display = f"{r2*100:.1f}%"
        r2_texto = f"Variación explicada en pruebas · {mejores_r2[0]} sesión(es)"
    elif r2 is not None:
        r2_display = f"{r2:.3f}"
        r2_texto = "El ajuste no superó la referencia en este horizonte"
    else:
        r2_display, r2_texto = "N/D", "Evaluación no disponible"
    wape = safe_number(dato_wape.get("value"))
    mae = safe_number(dato_mae.get("value"))
    dir_result = mejores_direccion[1] if mejores_direccion else None



    tarjetas = "".join([
        metric_card(
            "Precisión de la predicción",
            r2_display,
            "Coeficiente de determinación (R²)"
        ),
        metric_card(
            "¿Cuánto puede variar el precio estimado?",
            f"{wape:.2f}%" if wape is not None else "N/D",
            "Error porcentual absoluto ponderado (WAPE)"
        ),
        metric_card(
            "¿Qué monto representa la diferencia?",
            display_money(mae, conversion) if mae is not None else "N/D",
            "Error absoluto medio (MAE)"
        ),
    ])


    st.markdown(
        f'<div class="metric-side"><h3>📊 ¿Cómo rindió el modelo?</h3>'
        f'<div class="context">Resultados de las pruebas históricas</div>{tarjetas}'
        f'<div class="metric-foot">Cada indicador corresponde al horizonte señalado.</div>',
        unsafe_allow_html=True,
    )
    if not metrics_doc:
        st.info("Aún no hay métricas disponibles para este insumo.")

# ===== NOTICIAS RECIENTES EN SECCIÓN INDEPENDIENTE =====
st.markdown('<div class="dashboard-panel-title" style="margin-top:25px">📰 Noticias recientes del mercado</div>', unsafe_allow_html=True)
st.markdown('<div class="dashboard-panel-subtitle">Titulares informativos relacionados con el insumo seleccionado</div>', unsafe_allow_html=True)

if news_docs:
    visible_news = news_docs[:6]
    for offset in range(0, len(visible_news), 3):
        cols = st.columns(3, gap="medium")
        for col, news in zip(cols, visible_news[offset:offset + 3]):
            with col:
                title = escape(str(news.get("title") or "Sin título"))
                source = escape(str(news.get("source") or "Fuente no identificada"))
                sentiment = str(news.get("sentiment_label") or "NEUTRAL").upper()
                sentiment_color = ("#15803D" if sentiment == "POSITIVO" else
                                   "#B91C1C" if sentiment == "NEGATIVO" else "#64748B")
                news_link = str(news.get("url") or "")
                parsed_link = urlparse(news_link)
                link_ok = parsed_link.scheme in ("http", "https") and bool(parsed_link.netloc)
                headline = (f'<a href="{escape(news_link, quote=True)}" target="_blank" rel="noopener noreferrer">{title}</a>'
                            if link_ok else f'<div style="font-weight:700;color:#142B46">{title}</div>')
                published = format_day(news.get("published_at"))
                st.markdown(f'''<div class="news-card">
                    <div class="news-meta">{source}<span style="float:right;color:{sentiment_color};font-weight:800">{escape(sentiment)}</span></div>
                    {headline}<div class="news-date">Publicado: {published}</div>
                </div>''', unsafe_allow_html=True)
else:
    st.info("No se encontraron noticias recientes para este insumo.")
