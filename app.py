from __future__ import annotations
import os
import re
import uuid
import hmac
import hashlib
import secrets
from datetime import datetime, timezone
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
@st.dialog("Acceso empresarial", width="small")
def auth_dialog():
    if st.session_state.auth_view == "login":
        st.markdown("### Iniciar sesión")
        st.caption("Ingresa con el correo y la contraseña de tu cuenta.")
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
        st.markdown("### Crear cuenta")
        st.caption("Registra tus datos para acceder a futuras consultas personalizadas.")
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
AGRO_BANNER_URL = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ7xMp_XASpg1M5WOOXfMh-Jc2JUaFFYtU4cojK8O2jJw&s=10"
st.markdown(f"""
<div class="agro-header">
    <img class="agro-header-image" src="{AGRO_BANNER_URL}" alt="Paisaje agrícola">
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
    with c_ai:
        st.markdown('<div class="profile-card-head profile-green">✦ &nbsp; Consulta a la IA</div>', unsafe_allow_html=True)
        st.markdown('<div class="profile-panel" style="min-height:0;">', unsafe_allow_html=True)
        options = user.get("watchlist", [])
        if not options:
            options = list(INTEREST_NAMES)
        st.selectbox("Selecciona el insumo o producto", options=options,
                     format_func=lambda k: INTEREST_NAMES.get(k,k), key="profile_asset")
        st.text_area("¿Qué deseas conocer sobre tus insumos de interés?", height=155,
                     placeholder="Ej. ¿Qué tendencia podría tener el maíz en los próximos meses?", key="profile_question")
        st.button("✦ Consultar", use_container_width=True, type="primary", disabled=True, key="profile_consult")
        st.markdown('<div class="profile-muted">ⓘ Próximamente disponible</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c_history:
        st.markdown('<div class="profile-card-head profile-blue">▤ &nbsp; Historial de consultas</div>', unsafe_allow_html=True)
        st.markdown('<div class="profile-panel" style="min-height:0;">', unsafe_allow_html=True)
        col_date, col_month, col_year = st.columns(3, gap="small")
        with col_date:
            st.selectbox("Fecha", ["Todas"], key="history_date", disabled=True)
        with col_month:
            st.selectbox("Mes", ["Todos"], key="history_month", disabled=True)
        with col_year:
            st.selectbox("Año", ["Todos"], key="history_year", disabled=True)
        st.markdown('<div class="profile-history-empty">📋<div style="font-weight:650;margin:8px 0 3px;">Aún no hay consultas registradas</div>Próximamente podrás revisar tus consultas anteriores con IA.</div>', unsafe_allow_html=True)
        st.markdown('<div class="profile-muted">ⓘ Próximamente disponible</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ===== NAVEGACIÓN ENTRE EL PANEL Y EL PERFIL =====
if st.session_state.auth_user and st.session_state.current_page == "profile":
    render_profile()
    st.stop()

header_spacer, header_access = st.columns([4.0, 1.2])
with header_access:
    if st.session_state.auth_user is None:
        if st.button("Ingresar como usuario", use_container_width=True, key="open_login"):
            st.session_state.auth_view = "login"
    else:
        if st.button("Mi perfil", use_container_width=True, key="open_profile"):
            st.session_state.current_page = "profile"
            st.rerun()

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
selected_name = st.sidebar.selectbox("Insumo o producto", options=list(asset_options.keys()))
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

pred_doc = db.predictions.find_one({"asset_id": asset_id}) or {}
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
rec_data = load_gemini_recommendations()
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
chart_col, metric_col = st.columns([3.25, 1.15], gap="medium")

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
            template="plotly_white", height=500, margin=dict(l=6, r=8, t=18, b=14),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#17304B", size=12), hovermode="closest",
            legend=dict(orientation="h", yanchor="bottom", y=1.025, xanchor="left", x=0,
                        font=dict(size=11), bgcolor="rgba(255,255,255,.85)"),
            xaxis=dict(title=None, showgrid=False, linecolor="#CBD5E1", tickformat="%b %Y",
                       rangeslider=dict(visible=False)),
            yaxis=dict(title=f"USD · {unit_label}", gridcolor="#E6EBF1", zeroline=False),
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
    metric_summary = metrics_doc.get("summary") or {}
    per_horizon = metrics_doc.get("metrics_by_horizon") or {}
    eligible = [(int(h), metric) for h, metric in per_horizon.items()
                if str(h).isdigit() and isinstance(metric, dict) and metric.get("status") == "trained"]
    best_r2 = max(
        ((h, safe_number(metric.get("r2_price")), metric) for h, metric in eligible
         if safe_number(metric.get("r2_price")) is not None),
        key=lambda item: item[1], default=None,
    )
    best_direction = max(
        ((h, safe_number(metric.get("directional_accuracy_pct")), metric) for h, metric in eligible
         if safe_number(metric.get("directional_accuracy_pct")) is not None),
        key=lambda item: item[1], default=None,
    )
    best_wape = metric_summary.get("best_wape")
    best_mae = metric_summary.get("best_mae")
    r2_value = f"{best_r2[1]:.3f}" if best_r2 else "N/D"
    r2_desc = (f"{best_r2[0]} sesiones · {best_r2[2].get('selected_model','').upper()}"
               if best_r2 else "Sin resultados disponibles")
    wape_value = (f"{safe_number(best_wape.get('value')):.2f}%"
                  if best_wape and safe_number(best_wape.get("value")) is not None else "N/D")
    mae_value = (display_money(best_mae.get("value"), conversion)
                 if best_mae else "N/D")
    direction_value = f"{best_direction[1]:.1f}%" if best_direction else "N/A"
    direction_desc = (f"{best_direction[0]} sesiones · {best_direction[2].get('selected_model','').upper()}"
                      if best_direction else "No aplicable a los horizontes Naive")
    card_html = "".join([
        metric_card("R² de precios · máximo", r2_value, r2_desc),
        metric_card("WAPE · error mínimo", wape_value, extrema_description(best_wape)),
        metric_card("MAE · error mínimo", mae_value, extrema_description(best_mae)),
        metric_card("Precisión direccional · máxima", direction_value, direction_desc),
    ])
    st.markdown(f'''<div class="metric-side"><h3>📊 Confiabilidad del modelo</h3>
      <div class="context">Mejores resultados históricos del insumo seleccionado</div>
      {card_html}
      <div class="metric-foot">Las métricas corresponden a diferentes horizontes del modelo seleccionado en cada caso. No describen por sí solas la precisión de toda la trayectoria a seis meses.</div>
      <div class="metric-foot">Último entrenamiento: {format_day(metrics_doc.get('trained_at'))}</div>
    </div>''', unsafe_allow_html=True)
    if not metrics_doc:
        st.info("Aún no hay métricas de entrenamiento para este insumo.")

# ===== NOTICIAS RECIENTES EN SECCIÓN INDEPENDIENTE =====
st.markdown('<div class="dashboard-panel-title" style="margin-top:25px">📰 Noticias recientes del mercado</div>', unsafe_allow_html=True)
st.markdown('<div class="dashboard-panel-subtitle">Titulares informativos relacionados con el insumo seleccionado; no intervienen en el modelo predictivo</div>', unsafe_allow_html=True)

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










# from __future__ import annotations
# import os
# import re
# import uuid
# import hmac
# import hashlib
# import secrets
# from datetime import datetime, timezone
# from pymongo import ASCENDING
# from pymongo.errors import DuplicateKeyError, PyMongoError
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# import streamlit as st

# for key in ("MONGODB_URI", "MONGODB_DB_NAME"):
#     try:
#         if key in st.secrets and not os.getenv(key):
#             os.environ[key] = str(st.secrets[key])
#     except Exception:
#         pass

# from src.db import ping, get_db
# from src.dashboard import (
#     latest_prediction, load_gemini_recommendations
# )
# from src.api_clients.yfinance_client import AGRICULTURAL_COMMODITIES

# st.set_page_config(
#     page_title="Agri-Supply Chain Analytics",
#     page_icon="🌾",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# st.markdown("""
#     <style>
        
#         .block-container {
#             padding-top: 1rem !important;
#             padding-bottom: 1rem !important;
#         }
        
#         header[data-testid="stHeader"] {
#             background: transparent !important;
#         }
#     </style>
# """, unsafe_allow_html=True)

# st.markdown("""
# <style>
# .stApp {
#     background-color: #F8FAFC;
# }
# .block-container {
#     padding-top: 1.5rem !important;
#     padding-bottom: 3rem !important;
# }

# [data-testid="stSidebar"] {
#     background-color: #0F172A !important;
#     border-right: 3px solid #16A34A !important;
# }
# [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, 
# [data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stMarkdown {
#     color: #F8FAFC !important;
#     font-weight: 600;
# }
# [data-testid="stSidebar"] div[data-baseweb="select"] > div {
#     background-color: #16A34A !important;
#     border-color: #16A34A !important;
#     border-radius: 6px;
#     color: #FFFFFF !important;
# }
# [data-testid="stSidebar"] div[data-baseweb="select"] span {
#     color: #FFFFFF !important;
#     font-weight: 700;
# }


# [data-testid="stMetric"] {
#     background-color: #FFFFFF;
#     border: 1px solid #E2E8F0;
#     border-left: 4px solid #15803D;
#     border-radius: 8px;
#     padding: 12px 16px;
#     box-shadow: 0 2px 4px rgba(0,0,0,0.04);
# }
# [data-testid="stMetricLabel"] {
#     color: #475569 !important;
#     font-size: 0.9rem !important;
#     font-weight: 600 !important;
# }
# [data-testid="stMetricValue"] {
#     color: #0F172A !important;
#     font-size: 1.6rem !important;
#     font-weight: 800 !important;
# }

# .section-title {
#     color: #15803D;
#     font-size: 1.4rem;
#     font-weight: 700;
#     margin-top: 1.5rem;
#     margin-bottom: 1rem;
#     border-bottom: 2px solid #22C55E;
#     padding-bottom: 5px;
#     display: inline-block;
# }
# </style>
# """, unsafe_allow_html=True)

# try:
#     ping()
# except Exception:
#     st.error("No se pudo conectar a MongoDB Atlas.")
#     st.stop()

# db = get_db()

# if "auth_view" not in st.session_state:
#     st.session_state.auth_view = None
# if "auth_user" not in st.session_state:
#     st.session_state.auth_user = None

# EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
# INTEREST_NAMES = {
#     "corn": "Maíz",
#     "wheat": "Trigo",
#     "soybeans": "Soya",
#     "rice": "Arroz",
#     "oats": "Avena",
#     "coffee": "Café",
#     "cocoa": "Cacao",
#     "sugar": "Azúcar",
#     "cotton": "Algodón",
#     "crude_oil_transport": "Petróleo WTI",
#     "natural_gas_fertilizer": "Gas natural",
# }


# def password_hash(password):
#     salt = secrets.token_bytes(16)
#     iterations = 600000
#     digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
#     return f"pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}"


# def verify_password(password, stored_hash):
#     try:
#         algorithm, rounds, salt_hex, digest_hex = stored_hash.split("$")
#         if algorithm != "pbkdf2_sha256":
#             return False
#         digest = hashlib.pbkdf2_hmac(
#             "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(rounds)
#         )
#         return hmac.compare_digest(digest, bytes.fromhex(digest_hex))
#     except (ValueError, TypeError, AttributeError):
#         return False


# def switch_auth(view):
#     st.session_state.auth_view = view
#     st.rerun()


# @st.dialog("Acceso empresarial", width="large")
# def auth_dialog():
#     if st.session_state.auth_view == "login":
#         st.markdown("### Iniciar sesión")
#         st.caption("Ingresa con el correo y la contraseña de tu cuenta.")
#         with st.form("login_form", clear_on_submit=False):
#             email = st.text_input("Correo electrónico", placeholder="correo@empresa.com")
#             password = st.text_input("Contraseña", type="password")
#             submit = st.form_submit_button("Ingresar", use_container_width=True, type="primary")
#         if submit:
#             email = email.strip().lower()
#             if not EMAIL_PATTERN.fullmatch(email) or not password:
#                 st.error("Ingresa un correo y una contraseña válidos.")
#             else:
#                 try:
#                     user = db["clientes_"].find_one({"email": email})
#                     if not user or not verify_password(password, user.get("password_hash", "")):
#                         st.error("Correo o contraseña incorrectos.")
#                     elif not user.get("active", False):
#                         st.error("La cuenta está inactiva.")
#                     else:
#                         st.session_state.auth_user = {
#                             "client_id": user["client_id"],
#                             "name": user.get("name", ""),
#                             "email": user["email"],
#                         }
#                         st.session_state.auth_view = None
#                         st.rerun()
#                 except PyMongoError:
#                     st.error("No se pudo completar el ingreso. Inténtalo nuevamente.")
#         if st.button("¿No tienes una cuenta? Regístrate", use_container_width=True):
#             switch_auth("register")
#     else:
#         st.markdown("### Crear cuenta")
#         st.caption("Registra tus datos para acceder a futuras consultas personalizadas.")
#         with st.form("register_form", clear_on_submit=False):
#             col1, col2 = st.columns(2)
#             with col1:
#                 name = st.text_input("Nombres")
#             with col2:
#                 last_name = st.text_input("Apellidos")
#             email = st.text_input("Correo electrónico", placeholder="correo@empresa.com")
#             col1, col2 = st.columns(2)
#             with col1:
#                 password = st.text_input("Contraseña (mínimo 8 caracteres)", type="password")
#             with col2:
#                 confirmation = st.text_input("Confirmar contraseña", type="password")
#             col1, col2 = st.columns(2)
#             with col1:
#                 ocupacion = st.text_input("Ocupación", placeholder="Ej. Empresario agrícola")
#             with col2:
#                 sexo = st.selectbox("Sexo", ["Masculino", "Femenino"])
#             watchlist = st.multiselect(
#                 "Insumos y productos de interés",
#                 options=list(INTEREST_NAMES.keys()),
#                 format_func=lambda key: INTEREST_NAMES[key],
#             )
#             preferencia = st.text_area(
#                 "¿Qué información buscas obtener? (máximo 40 palabras)",
#                 height=85,
#                 placeholder="Ej. Conocer los mejores periodos para comprar maíz.",
#             )
#             submit = st.form_submit_button("Crear cuenta", use_container_width=True, type="primary")
#         if submit:
#             normalized_email = email.strip().lower()
#             words = preferencia.strip().split()
#             if not all([name.strip(), last_name.strip(), ocupacion.strip()]):
#                 st.error("Completa tus nombres, apellidos y ocupación.")
#             elif not EMAIL_PATTERN.fullmatch(normalized_email) or len(normalized_email) > 254:
#                 st.error("Ingresa un correo electrónico válido.")
#             elif not 8 <= len(password) <= 128:
#                 st.error("La contraseña debe tener entre 8 y 128 caracteres.")
#             elif password != confirmation:
#                 st.error("Las contraseñas no coinciden.")
#             elif not watchlist:
#                 st.error("Selecciona al menos un insumo de interés.")
#             elif len(words) > 40:
#                 st.error("La preferencia puede contener un máximo de 40 palabras.")
#             elif max(len(name.strip()), len(last_name.strip()), len(ocupacion.strip())) > 120:
#                 st.error("Los datos personales deben tener 120 caracteres como máximo por campo.")
#             else:
#                 try:
#                     collection = db["clientes_"]
#                     collection.create_index([("email", ASCENDING)], unique=True)
#                     if collection.find_one({"email": normalized_email}, {"_id": 1}):
#                         st.warning("Este correo electrónico ya está registrado. Inicia sesión para continuar.")
#                     else:
#                         now = datetime.now(timezone.utc)
#                         collection.insert_one({
#                             "client_id": str(uuid.uuid4()),
#                             "name": name.strip(),
#                             "last_name": last_name.strip(),
#                             "email": normalized_email,
#                             "password_hash": password_hash(password),
#                             "sexo": sexo,
#                             "ocupacion": ocupacion.strip(),
#                             "watchlist": watchlist,
#                             "preferencia": " ".join(words),
#                             "active": True,
#                             "created_at": now,
#                         })
#                         st.success("Cuenta registrada correctamente. Ya puedes iniciar sesión.")
#                 except DuplicateKeyError:
#                     st.warning("Este correo electrónico ya está registrado. Inicia sesión para continuar.")
#                 except PyMongoError:
#                     st.error("No se pudo completar el registro. Inténtalo nuevamente.")
#         if st.button("¿Ya tienes una cuenta? Inicia sesión", use_container_width=True):
#             switch_auth("login")


# AGRO_BANNER_URL = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ7xMp_XASpg1M5WOOXfMh-Jc2JUaFFYtU4cojK8O2jJw&s=10"

# st.markdown(f"""
# <div style="display: flex; align-items: center; gap: 20px; margin-bottom: 20px;">
#     <img src="{AGRO_BANNER_URL}" style="width: 250px; height: 100px; object-fit: cover; border-radius: 12px; box-shadow: 2px 4px 6px rgba(0,0,0,0.1);">
#     <div>
#         <h1 style="margin: 0; color: #0F172A; font-size: 2.3rem; font-weight: 800; line-height: 1.1;">Panel de Suministros Agrícolas</h1>
#         <p style="margin: 0; color: #64748B; font-size: 1.05rem; font-weight: 500;">Monitoreo inteligente - Apoyo a decisiones logísticas</p>
#     </div>
# </div>
# """, unsafe_allow_html=True)

# button_col1, button_col2 = st.columns([4, 1])
# with button_col2:
#     if st.session_state.auth_user is None:
#         if st.button("Ingresar como usuario", use_container_width=True):
#             st.session_state.auth_view = "login"
#     else:
#         st.caption("Sesión iniciada")
#         if st.button("Cerrar sesión", use_container_width=True):
#             st.session_state.auth_user = None
#             st.session_state.auth_view = None
#             st.rerun()

# if st.session_state.auth_view is not None and st.session_state.auth_user is None:
#     auth_dialog()

# st.sidebar.markdown("### 🌾 Menú")

# db = get_db()

# @st.cache_data(ttl=3600)
# def load_catalog_map():
#     docs = list(db.assets_catalog.find({}, {"_id": 0}))
#     if not docs:
#         return {
#             "corn": {"name_es": "Maíz", "display_label": "Precio por 1 Bushel (25.4 kg)", "ticker": "ZC=F"},
#             "wheat": {"name_es": "Trigo", "display_label": "Precio por 1 Bushel (27.2 kg)", "ticker": "ZW=F"},
#             "soybeans": {"name_es": "Soya", "display_label": "Precio por 1 Bushel (27.2 kg)", "ticker": "ZS=F"},
#             "rice": {"name_es": "Arroz", "display_label": "Precio por 1 Quintal (45.36 kg)", "ticker": "ZR=F"},
#             "oats": {"name_es": "Avena", "display_label": "Precio por 1 Bushel (14.5 kg)", "ticker": "ZO=F"},
#             "coffee": {"name_es": "Café", "display_label": "Precio por 1 Quintal (45.36 kg)", "ticker": "KC=F"},
#             "sugar": {"name_es": "Azúcar", "display_label": "Precio por 1 Libra (0.45 kg)", "ticker": "SB=F"},
#             "cocoa": {"name_es": "Cacao", "display_label": "Precio por 1 Tonelada (1,000 kg)", "ticker": "CC=F"},
#             "cotton": {"name_es": "Algodón", "display_label": "Precio por 1 Libra (0.45 kg)", "ticker": "CT=F"},
#             "crude_oil_transport": {"name_es": "Petróleo WTI (Transporte/Fletes)", "display_label": "Precio por 1 Barril (159 Litros)", "ticker": "CL=F"},
#             "natural_gas_fertilizer": {"name_es": "Gas Natural (Fertilizantes)", "display_label": "Precio por 1 MMBtu (Energía)", "ticker": "NG=F"}
#         }
#     return {d["asset_id"]: d for d in docs}

# catalog = load_catalog_map()

# asset_options = {info.get("name_es", asset_key): asset_key for asset_key, info in catalog.items()}

# selected_name = st.sidebar.selectbox(
#     "Insumo o producto",
#     options=list(asset_options.keys())
# )

# asset_id = asset_options[selected_name]
# asset_info = catalog.get(asset_id, {})
# ticker = asset_info.get("ticker", AGRICULTURAL_COMMODITIES.get(asset_id, ""))

# range_days = st.sidebar.selectbox(
#     "Histórico",
#     [30, 90, 180, 365, 730, 1095],
#     index=5, 
#     format_func=lambda x: (
#         f"{x} días" if x < 365 
#         else ("1 año" if x == 365 else f"{x // 365} años")
#     )
# )

# db = get_db()
# ohlcv_docs = list(db.ohlcv.find({"asset_id": asset_id}).sort("date", 1))
# hist_df = pd.DataFrame(ohlcv_docs)

# if not hist_df.empty:
#     date_col = next((col for col in ["date", "timestamp", "datetime"] if col in hist_df.columns), None)
#     if date_col:
#         hist_df[date_col] = pd.to_datetime(hist_df[date_col])
#         hist_df = hist_df.sort_values(date_col)

# pred_doc = db.predictions.find_one({"asset_id": asset_id})

# news_docs = list(db.news.find({"mentioned_asset_ids": asset_id}).sort("published_at", -1).limit(50))
# news_df = pd.DataFrame(news_docs)

# asset_title = asset_info.get("name_es", asset_id.replace('_', ' ').title())
# st.markdown(f"<div align='center'><h2 style='color: #0F172A; font-weight: 800;'>Métricas de Mercado: {asset_title}</h2></div>", unsafe_allow_html=True)

# if not hist_df.empty and "close" in hist_df.columns:
#     current_price = float(hist_df["close"].iloc[-1]) / 100.0
# else:
#     raw_pred_price = float(pred_doc.get("current_price", 0.0)) if pred_doc else 0.0
#     current_price = raw_pred_price / 100.0

# trend = pred_doc.get("trend", "N/A") if pred_doc else "N/A"
# predicted_change = float(pred_doc.get("predicted_change_pct", 0.0)) if pred_doc else 0.0

# unit_label = asset_info.get("display_label", "USD")

# m1, m2, m3 = st.columns(3)

# with m1:
#     st.markdown(
#         f"""
#         <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-left: 4px solid #15803D; border-radius: 8px; padding: 12px 16px; box-shadow: 0 2px 4px rgba(0,0,0,0.04);">
#             <div style="color: #475569; font-size: 0.9rem; font-weight: 600;">Precio Actual</div>
#             <div style="display: flex; align-items: baseline; gap: 8px; margin-top: 2px;">
#                 <span style="color: #0F172A; font-size: 1.6rem; font-weight: 800;">${current_price:,.2f}</span>
#                 <span style="color: #334155; font-size: 1.1rem; font-weight: 600;">{unit_label}</span>
#             </div>
#         </div>
#         """,
#         unsafe_allow_html=True
#     )

# with m2:
#     m2.metric("Tendencia", trend)

# with m3:
#     m3.metric("Proyección Acumulada", f"{predicted_change:+.2f}%")

# asset_id_actual = locals().get("asset_id", locals().get("selected_asset", "corn"))
# nombre_actual = locals().get("name", locals().get("selected_name", asset_id_actual))

# st.markdown(f"<h3 style='color: #0F172A;'>Dictamen Principal de IA para {nombre_actual}</h3>", unsafe_allow_html=True)

# rec_data = load_gemini_recommendations()

# if rec_data and "analisis_diario" in rec_data:
#     dictamen_activo = next(
#         (item for item in rec_data["analisis_diario"] if item.get("asset_id") == asset_id_actual), 
#         None
#     )
    
#     if dictamen_activo:
#         rec = dictamen_activo.get("recomendacion", "N/A")
#         riesgo = dictamen_activo.get("nivel_riesgo", "N/A")
#         tendencia = dictamen_activo.get("tendencia", "N/A")
       
#         if rec in ["COMPRAR_ANTICIPADO", "COMPRAR"]:
#             color_hex = "#10B981" 
#         elif rec in ["ESPERAR_VALLE", "VENDER"]:
#             color_hex = "#EF4444" 
#         else:
#             color_hex = "#F59E0B" 

#         fecha_bruta = rec_data.get("fecha_actualizacion", "")
#         fecha_formateada = fecha_bruta.replace("T", " ")[:19] if fecha_bruta else "Desconocida"

#         st.markdown(f"""
#         <div style="background-color: #FFFFFF; border: 2px solid {color_hex}; border-radius: 12px; padding: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 25px;">
#             <div style="display:flex; justify-content: space-between; margin-bottom: 15px;">
#                 <div style="text-align: center; width: 33%;">
#                     <p style="margin:0; color: #64748B; font-size: 0.9rem; font-weight:bold;">Recomendación</p>
#                     <h3 style="margin:0; color: {color_hex}; font-size: 1.6rem;">{rec}</h3>
#                 </div>
#                 <div style="text-align: center; width: 33%; border-left: 1px solid #E2E8F0; border-right: 1px solid #E2E8F0;">
#                     <p style="margin:0; color: #64748B; font-size: 0.9rem; font-weight:bold;">Nivel de Riesgo</p>
#                     <h3 style="margin:0; color: #0F172A; font-size: 1.5rem;">{riesgo}</h3>
#                 </div>
#                 <div style="text-align: center; width: 33%;">
#                     <p style="margin:0; color: #64748B; font-size: 0.9rem; font-weight:bold;">Tendencia</p>
#                     <h3 style="margin:0; color: #0F172A; font-size: 1.5rem;">{tendencia}</h3>
#                 </div>
#             </div>
#             <div style="background-color: #F8FAFC; padding: 15px; border-radius: 8px; border-left: 4px solid {color_hex};">
#                 <p style="margin:0 0 10px 0; color: #334155; font-size: 0.95rem;"><strong>Resumen del Mercado:</strong> {dictamen_activo.get('resumen_ejecutivo', '')}</p>
#                 <p style="margin:0; color: #334155; font-size: 0.95rem;"><strong>Justificación de la recomendación:</strong> {dictamen_activo.get('justificacion', '')}</p>
#             </div>
#             <p style="margin:10px 0 0 0; text-align:right; font-size:0.75rem; color:#94A3B8;">Última corrida del Agente: {fecha_formateada} UTC</p>
#         </div>
#         """, unsafe_allow_html=True)
#     else:
#         st.info(f"No hay un dictamen generado para {nombre_actual} en la última corrida del agente.")
# else:
#     st.warning("No se encontraron recomendaciones recientes de IA en la base de datos.")


# c_chart, c_news = st.columns([1.8, 1], gap="large")

# with c_chart:
#     st.markdown("<div class='section-title'>Comportamiento Histórico</div>", unsafe_allow_html=True)
    
#     if not hist_df.empty:
#         date_col = next((col for col in ["timestamp", "date", "time", "datetime"] if col in hist_df.columns), None)
        
#         if date_col:
#             hist_df["parsed_date"] = pd.to_datetime(hist_df[date_col], utc=True)
#             plot_df = hist_df.sort_values("parsed_date").tail(range_days)
            
#             price_col = "close" if "close" in plot_df.columns else ("price" if "price" in plot_df.columns else None)
            
#             if price_col:
#                 fig = go.Figure()
#                 real_prices_usd = plot_df[price_col] / 100.0

#                 fig.add_trace(go.Scatter(
#                     x=plot_df["parsed_date"], y=real_prices_usd, 
#                     mode="lines", name="Precio Real (USD)", 
#                     line=dict(color="#15803D", width=2.5)
#                 ))
                
#                 if pred_doc and "predictions_180d" in pred_doc:
#                     preds_array = pred_doc["predictions_180d"]
#                     if preds_array:
#                         pred_dates = [pd.to_datetime(p["target_time"], utc=True) for p in preds_array]
#                         pred_prices = [float(p["predicted_close"]) / 100.0 for p in preds_array]
                        
#                         last_real_date = plot_df["parsed_date"].max()
#                         last_real_price = float(real_prices_usd.iloc[-1])
                        
#                         pred_dates.insert(0, last_real_date)
#                         pred_prices.insert(0, last_real_price)

#                         fig.add_trace(go.Scatter(
#                             x=pred_dates,
#                             y=pred_prices,
#                             mode="lines", name="Pronóstico XGBoost (180d)",
#                             line=dict(dash="dash", color="#EAB308", width=3)
#                         ))
                
#                 fig.update_layout(
#                     template="plotly_white",
#                     paper_bgcolor="rgba(0,0,0,0)",
#                     plot_bgcolor="rgba(0,0,0,0)",
#                     font=dict(color="#0F172A", size=12),  
#                     height=420, 
#                     margin=dict(l=10, r=10, t=20, b=10),
#                     hovermode="x unified",
#                     showlegend=True,
#                     legend=dict(
#                         font=dict(color="#0F172A", size=11),
#                         bgcolor="rgba(255,255,255,0.8)"
#                     ),
#                     xaxis=dict(
#                         showgrid=False, 
#                         title="", 
#                         tickfont=dict(color="#334155", size=11),  
#                         linecolor="#CBD5E1"
#                     ), 
#                     yaxis=dict(
#                         showgrid=True, 
#                         gridcolor="#E2E8F0", 
#                         title="USD / Bushel", 
#                         tickfont=dict(color="#334155", size=11),  
#                         zerolinecolor="#E2E8F0"
#                     )
#                 )
#                 st.plotly_chart(fig, use_container_width=True)
#             else:
#                 st.info("No se encontró la columna de precios ('close' o 'price') en el histórico.")
#         else:
#             st.info("No se encontró una columna de fecha en la colección `ohlcv`.")
#     else:
#         st.info("Sin datos históricos disponibles en `ohlcv` para este commodity.")

# with c_news:
#     st.markdown("<div class='section-title'>Últimas Noticias Detectadas</div>", unsafe_allow_html=True)
    
#     if not news_df.empty:
#         recent_news = news_df.head(5)
        
#         for _, n in recent_news.iterrows():
#             title = n.get("title", "Sin título")
#             url = n.get("url", "#")
#             source = n.get("source", "Fuente Desconocida")
#             sentiment = n.get("sentiment_label", "NEUTRAL")
         
#             pub_date = n.get("published_at")
#             pub_date_str = pd.to_datetime(pub_date).strftime("%d/%m/%Y %H:%M") if pub_date else "Reciente"
            
#             if sentiment == "POSITIVO":
#                 badge = "<span style='background-color:#DCFCE7; color:#15803D; padding:2px 8px; border-radius:12px; font-size:0.75rem; font-weight:600;'>POSITIVO</span>"
#             elif sentiment == "NEGATIVO":
#                 badge = "<span style='background-color:#FEE2E2; color:#B91C1C; padding:2px 8px; border-radius:12px; font-size:0.75rem; font-weight:600;'>NEGATIVO</span>"
#             else:
#                 badge = "<span style='background-color:#F1F5F9; color:#475569; padding:2px 8px; border-radius:12px; font-size:0.75rem; font-weight:600;'>NEUTRAL</span>"
            
#             st.markdown(
#                 f"""
#                 <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px; margin-bottom: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
#                     <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
#                         <span style="font-size: 0.8rem; color: #64748B; font-weight: 500;">{source}</span>
#                         {badge}
#                     </div>
#                     <a href="{url}" target="_blank" style="text-decoration: none; color: #0F172A; font-weight: 600; font-size: 0.88rem; line-height: 1.3; display: block; margin-bottom: 8px;">
#                         {title}
#                     </a>
#                     <div style="font-size: 0.72rem; color: #94A3B8; border-top: 1px solid #F1F5F9; padding-top: 4px; text-align: right;">
#                         Publicado: {pub_date_str}
#                     </div>
#                 </div>
#                 """,
#                 unsafe_allow_html=True
#             )
#     else:
#         st.info("No hay noticias registradas recientemente para este activo.")










# from __future__ import annotations
# import os
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# import streamlit as st

# for key in ("MONGODB_URI", "MONGODB_DB_NAME"):
#     try:
#         if key in st.secrets and not os.getenv(key):
#             os.environ[key] = str(st.secrets[key])
#     except Exception:
#         pass

# from src.db import ping, get_db
# from src.dashboard import (
#     latest_prediction, load_gemini_recommendations
# )
# from src.api_clients.yfinance_client import AGRICULTURAL_COMMODITIES

# st.set_page_config(
#     page_title="Agri-Supply Chain Analytics",
#     page_icon="🌾",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# st.markdown("""
#     <style>
#         /* Ajusta el espacio superior del contenedor principal de Streamlit */
#         .block-container {
#             padding-top: 1rem !important;
#             padding-bottom: 1rem !important;
#         }
#         /* Asegura que la cabecera o banner respire correctamente */
#         header[data-testid="stHeader"] {
#             background: transparent !important;
#         }
#     </style>
# """, unsafe_allow_html=True)

# st.markdown("""
# <style>
# .stApp {
#     background-color: #F8FAFC;
# }
# .block-container {
#     padding-top: 1.5rem !important;
#     padding-bottom: 3rem !important;
# }
# /* Sidebar estilo Agro/Verde y Dorado */
# [data-testid="stSidebar"] {
#     background-color: #0F172A !important;
#     border-right: 3px solid #16A34A !important;
# }
# [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, 
# [data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stMarkdown {
#     color: #F8FAFC !important;
#     font-weight: 600;
# }
# [data-testid="stSidebar"] div[data-baseweb="select"] > div {
#     background-color: #16A34A !important;
#     border-color: #16A34A !important;
#     border-radius: 6px;
#     color: #FFFFFF !important;
# }
# [data-testid="stSidebar"] div[data-baseweb="select"] span {
#     color: #FFFFFF !important;
#     font-weight: 700;
# }

# /* Tarjetas de Métricas */
# [data-testid="stMetric"] {
#     background-color: #FFFFFF;
#     border: 1px solid #E2E8F0;
#     border-left: 4px solid #15803D;
#     border-radius: 8px;
#     padding: 12px 16px;
#     box-shadow: 0 2px 4px rgba(0,0,0,0.04);
# }
# [data-testid="stMetricLabel"] {
#     color: #475569 !important;
#     font-size: 0.9rem !important;
#     font-weight: 600 !important;
# }
# [data-testid="stMetricValue"] {
#     color: #0F172A !important;
#     font-size: 1.6rem !important;
#     font-weight: 800 !important;
# }

# .section-title {
#     color: #15803D;
#     font-size: 1.4rem;
#     font-weight: 700;
#     margin-top: 1.5rem;
#     margin-bottom: 1rem;
#     border-bottom: 2px solid #22C55E;
#     padding-bottom: 5px;
#     display: inline-block;
# }
# </style>
# """, unsafe_allow_html=True)

# try:
#     ping()
# except Exception:
#     st.error("No se pudo conectar a MongoDB Atlas.")
#     st.stop()

# # CABECERA
# AGRO_BANNER_URL = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ7xMp_XASpg1M5WOOXfMh-Jc2JUaFFYtU4cojK8O2jJw&s=10"

# st.markdown(f"""
# <div style="display: flex; align-items: center; gap: 20px; margin-bottom: 20px;">
#     <img src="{AGRO_BANNER_URL}" style="width: 250px; height: 100px; object-fit: cover; border-radius: 12px; box-shadow: 2px 4px 6px rgba(0,0,0,0.1);">
#     <div>
#         <h1 style="margin: 0; color: #0F172A; font-size: 2.3rem; font-weight: 800; line-height: 1.1;">Panel de Suministros Agrícolas</h1>
#         <p style="margin: 0; color: #64748B; font-size: 1.05rem; font-weight: 500;">Monitoreo inteligente - Apoyo a decisiones logísticas</p>
#     </div>
# </div>
# """, unsafe_allow_html=True)

# # FILTROS Y BARRA LATERAL 
# st.sidebar.markdown("### 🌾 Menú")

# db = get_db()

# @st.cache_data(ttl=3600)
# def load_catalog_map():
#     docs = list(db.assets_catalog.find({}, {"_id": 0}))
#     if not docs:
#         return {
#             "corn": {"name_es": "Maíz", "display_label": "Precio por 1 Bushel (25.4 kg)", "ticker": "ZC=F"},
#             "wheat": {"name_es": "Trigo", "display_label": "Precio por 1 Bushel (27.2 kg)", "ticker": "ZW=F"},
#             "soybeans": {"name_es": "Soya", "display_label": "Precio por 1 Bushel (27.2 kg)", "ticker": "ZS=F"},
#             "rice": {"name_es": "Arroz", "display_label": "Precio por 1 Quintal (45.36 kg)", "ticker": "ZR=F"},
#             "oats": {"name_es": "Avena", "display_label": "Precio por 1 Bushel (14.5 kg)", "ticker": "ZO=F"},
#             "coffee": {"name_es": "Café", "display_label": "Precio por 1 Quintal (45.36 kg)", "ticker": "KC=F"},
#             "sugar": {"name_es": "Azúcar", "display_label": "Precio por 1 Libra (0.45 kg)", "ticker": "SB=F"},
#             "cocoa": {"name_es": "Cacao", "display_label": "Precio por 1 Tonelada (1,000 kg)", "ticker": "CC=F"},
#             "cotton": {"name_es": "Algodón", "display_label": "Precio por 1 Libra (0.45 kg)", "ticker": "CT=F"},
#             "crude_oil_transport": {"name_es": "Petróleo WTI (Transporte/Fletes)", "display_label": "Precio por 1 Barril (159 Litros)", "ticker": "CL=F"},
#             "natural_gas_fertilizer": {"name_es": "Gas Natural (Fertilizantes)", "display_label": "Precio por 1 MMBtu (Energía)", "ticker": "NG=F"}
#         }
#     return {d["asset_id"]: d for d in docs}

# catalog = load_catalog_map()

# asset_options = {info.get("name_es", asset_key): asset_key for asset_key, info in catalog.items()}

# selected_name = st.sidebar.selectbox(
#     "Materia prima agrícola",
#     options=list(asset_options.keys())
# )

# asset_id = asset_options[selected_name]
# asset_info = catalog.get(asset_id, {})
# ticker = asset_info.get("ticker", AGRICULTURAL_COMMODITIES.get(asset_id, ""))

# range_days = st.sidebar.selectbox(
#     "Histórico",
#     [30, 90, 180, 365, 730, 1095],
#     index=5, 
#     format_func=lambda x: (
#         f"{x} días" if x < 365 
#         else ("1 año" if x == 365 else f"{x // 365} años")
#     )
# )

# db = get_db()
# ohlcv_docs = list(db.ohlcv.find({"asset_id": asset_id}).sort("date", 1))
# hist_df = pd.DataFrame(ohlcv_docs)

# if not hist_df.empty:
#     date_col = next((col for col in ["date", "timestamp", "datetime"] if col in hist_df.columns), None)
#     if date_col:
#         hist_df[date_col] = pd.to_datetime(hist_df[date_col])
#         hist_df = hist_df.sort_values(date_col)

# pred_doc = db.predictions.find_one({"asset_id": asset_id})

# news_docs = list(db.news.find({"mentioned_asset_ids": asset_id}).sort("published_at", -1).limit(50))
# news_df = pd.DataFrame(news_docs)

# # 1. MÉTRICAS PRINCIPALES 
# asset_title = asset_info.get("name_es", asset_id.replace('_', ' ').title())
# st.markdown(f"<div align='center'><h2 style='color: #0F172A; font-weight: 800;'>Métricas de Mercado: {asset_title}</h2></div>", unsafe_allow_html=True)

# if not hist_df.empty and "close" in hist_df.columns:
#     current_price = float(hist_df["close"].iloc[-1]) / 100.0
# else:
#     raw_pred_price = float(pred_doc.get("current_price", 0.0)) if pred_doc else 0.0
#     current_price = raw_pred_price / 100.0

# trend = pred_doc.get("trend", "N/A") if pred_doc else "N/A"
# predicted_change = float(pred_doc.get("predicted_change_pct", 0.0)) if pred_doc else 0.0

# unit_label = asset_info.get("display_label", "USD")

# m1, m2, m3 = st.columns(3)

# with m1:
#     st.markdown(
#         f"""
#         <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-left: 4px solid #15803D; border-radius: 8px; padding: 12px 16px; box-shadow: 0 2px 4px rgba(0,0,0,0.04);">
#             <div style="color: #475569; font-size: 0.9rem; font-weight: 600;">Precio Actual</div>
#             <div style="display: flex; align-items: baseline; gap: 8px; margin-top: 2px;">
#                 <span style="color: #0F172A; font-size: 1.6rem; font-weight: 800;">${current_price:,.2f}</span>
#                 <span style="color: #334155; font-size: 1.1rem; font-weight: 600;">{unit_label}</span>
#             </div>
#         </div>
#         """,
#         unsafe_allow_html=True
#     )

# with m2:
#     m2.metric("Tendencia", trend)

# with m3:
#     m3.metric("Proyección Acumulada", f"{predicted_change:+.2f}%")

# asset_id_actual = locals().get("asset_id", locals().get("selected_asset", "corn"))
# nombre_actual = locals().get("name", locals().get("selected_name", asset_id_actual))

# st.markdown(f"<h3 style='color: #0F172A;'>Dictamen Principal de IA para {nombre_actual}</h3>", unsafe_allow_html=True)

# rec_data = load_gemini_recommendations()

# if rec_data and "analisis_diario" in rec_data:
#     dictamen_activo = next(
#         (item for item in rec_data["analisis_diario"] if item.get("asset_id") == asset_id_actual), 
#         None
#     )
    
#     if dictamen_activo:
#         rec = dictamen_activo.get("recomendacion", "N/A")
#         riesgo = dictamen_activo.get("nivel_riesgo", "N/A")
#         tendencia = dictamen_activo.get("tendencia", "N/A")
       
#         if rec in ["COMPRAR_ANTICIPADO", "COMPRAR"]:
#             color_hex = "#10B981" 
#         elif rec in ["ESPERAR_VALLE", "VENDER"]:
#             color_hex = "#EF4444" 
#         else:
#             color_hex = "#F59E0B" 

#         fecha_bruta = rec_data.get("fecha_actualizacion", "")
#         fecha_formateada = fecha_bruta.replace("T", " ")[:19] if fecha_bruta else "Desconocida"

#         st.markdown(f"""
#         <div style="background-color: #FFFFFF; border: 2px solid {color_hex}; border-radius: 12px; padding: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 25px;">
#             <div style="display:flex; justify-content: space-between; margin-bottom: 15px;">
#                 <div style="text-align: center; width: 33%;">
#                     <p style="margin:0; color: #64748B; font-size: 0.9rem; font-weight:bold;">Recomendación</p>
#                     <h3 style="margin:0; color: {color_hex}; font-size: 1.6rem;">{rec}</h3>
#                 </div>
#                 <div style="text-align: center; width: 33%; border-left: 1px solid #E2E8F0; border-right: 1px solid #E2E8F0;">
#                     <p style="margin:0; color: #64748B; font-size: 0.9rem; font-weight:bold;">Nivel de Riesgo</p>
#                     <h3 style="margin:0; color: #0F172A; font-size: 1.5rem;">{riesgo}</h3>
#                 </div>
#                 <div style="text-align: center; width: 33%;">
#                     <p style="margin:0; color: #64748B; font-size: 0.9rem; font-weight:bold;">Tendencia</p>
#                     <h3 style="margin:0; color: #0F172A; font-size: 1.5rem;">{tendencia}</h3>
#                 </div>
#             </div>
#             <div style="background-color: #F8FAFC; padding: 15px; border-radius: 8px; border-left: 4px solid {color_hex};">
#                 <p style="margin:0 0 10px 0; color: #334155; font-size: 0.95rem;"><strong>Resumen del Mercado:</strong> {dictamen_activo.get('resumen_ejecutivo', '')}</p>
#                 <p style="margin:0; color: #334155; font-size: 0.95rem;"><strong>Justificación de la recomendación:</strong> {dictamen_activo.get('justificacion', '')}</p>
#             </div>
#             <p style="margin:10px 0 0 0; text-align:right; font-size:0.75rem; color:#94A3B8;">Última corrida del Agente: {fecha_formateada} UTC</p>
#         </div>
#         """, unsafe_allow_html=True)
#     else:
#         st.info(f"No hay un dictamen generado para {nombre_actual} en la última corrida del agente.")
# else:
#     st.warning("No se encontraron recomendaciones recientes de IA en la base de datos.")


# # 2. GRÁFICO HISTÓRICO
# c_chart, c_news = st.columns([1.8, 1], gap="large")

# with c_chart:
#     st.markdown("<div class='section-title'>Comportamiento Histórico</div>", unsafe_allow_html=True)
    
#     if not hist_df.empty:
#         date_col = next((col for col in ["timestamp", "date", "time", "datetime"] if col in hist_df.columns), None)
        
#         if date_col:
#             hist_df["parsed_date"] = pd.to_datetime(hist_df[date_col], utc=True)
#             plot_df = hist_df.sort_values("parsed_date").tail(range_days)
            
#             price_col = "close" if "close" in plot_df.columns else ("price" if "price" in plot_df.columns else None)
            
#             if price_col:
#                 fig = go.Figure()
#                 real_prices_usd = plot_df[price_col] / 100.0

#                 fig.add_trace(go.Scatter(
#                     x=plot_df["parsed_date"], y=real_prices_usd, 
#                     mode="lines", name="Precio Real (USD)", 
#                     line=dict(color="#15803D", width=2.5)
#                 ))
                
#                 if pred_doc and "predictions_180d" in pred_doc:
#                     preds_array = pred_doc["predictions_180d"]
#                     if preds_array:
#                         pred_dates = [pd.to_datetime(p["target_time"], utc=True) for p in preds_array]
#                         pred_prices = [float(p["predicted_close"]) / 100.0 for p in preds_array]
                        
#                         last_real_date = plot_df["parsed_date"].max()
#                         last_real_price = float(real_prices_usd.iloc[-1])
                        
#                         pred_dates.insert(0, last_real_date)
#                         pred_prices.insert(0, last_real_price)

#                         fig.add_trace(go.Scatter(
#                             x=pred_dates,
#                             y=pred_prices,
#                             mode="lines", name="Pronóstico XGBoost (180d)",
#                             line=dict(dash="dash", color="#EAB308", width=3)
#                         ))
                
#                 fig.update_layout(
#                     template="plotly_white",
#                     paper_bgcolor="rgba(0,0,0,0)",
#                     plot_bgcolor="rgba(0,0,0,0)",
#                     font=dict(color="#0F172A", size=12),  
#                     height=420, 
#                     margin=dict(l=10, r=10, t=20, b=10),
#                     hovermode="x unified",
#                     showlegend=True,
#                     legend=dict(
#                         font=dict(color="#0F172A", size=11),
#                         bgcolor="rgba(255,255,255,0.8)"
#                     ),
#                     xaxis=dict(
#                         showgrid=False, 
#                         title="", 
#                         tickfont=dict(color="#334155", size=11),  
#                         linecolor="#CBD5E1"
#                     ), 
#                     yaxis=dict(
#                         showgrid=True, 
#                         gridcolor="#E2E8F0", 
#                         title="USD / Bushel", 
#                         tickfont=dict(color="#334155", size=11),  
#                         zerolinecolor="#E2E8F0"
#                     )
#                 )
#                 st.plotly_chart(fig, use_container_width=True)
#             else:
#                 st.info("No se encontró la columna de precios ('close' o 'price') en el histórico.")
#         else:
#             st.info("No se encontró una columna de fecha en la colección `ohlcv`.")
#     else:
#         st.info("Sin datos históricos disponibles en `ohlcv` para este commodity.")

# with c_news:
#     st.markdown("<div class='section-title'>Últimas Noticias Detectadas</div>", unsafe_allow_html=True)
    
#     if not news_df.empty:
#         recent_news = news_df.head(5)
        
#         for _, n in recent_news.iterrows():
#             title = n.get("title", "Sin título")
#             url = n.get("url", "#")
#             source = n.get("source", "Fuente Desconocida")
#             sentiment = n.get("sentiment_label", "NEUTRAL")
         
#             pub_date = n.get("published_at")
#             pub_date_str = pd.to_datetime(pub_date).strftime("%d/%m/%Y %H:%M") if pub_date else "Reciente"
            
#             if sentiment == "POSITIVO":
#                 badge = "<span style='background-color:#DCFCE7; color:#15803D; padding:2px 8px; border-radius:12px; font-size:0.75rem; font-weight:600;'>POSITIVO</span>"
#             elif sentiment == "NEGATIVO":
#                 badge = "<span style='background-color:#FEE2E2; color:#B91C1C; padding:2px 8px; border-radius:12px; font-size:0.75rem; font-weight:600;'>NEGATIVO</span>"
#             else:
#                 badge = "<span style='background-color:#F1F5F9; color:#475569; padding:2px 8px; border-radius:12px; font-size:0.75rem; font-weight:600;'>NEUTRAL</span>"
            
#             st.markdown(
#                 f"""
#                 <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px; margin-bottom: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
#                     <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
#                         <span style="font-size: 0.8rem; color: #64748B; font-weight: 500;">{source}</span>
#                         {badge}
#                     </div>
#                     <a href="{url}" target="_blank" style="text-decoration: none; color: #0F172A; font-weight: 600; font-size: 0.88rem; line-height: 1.3; display: block; margin-bottom: 8px;">
#                         {title}
#                     </a>
#                     <div style="font-size: 0.72rem; color: #94A3B8; border-top: 1px solid #F1F5F9; padding-top: 4px; text-align: right;">
#                         Publicado: {pub_date_str}
#                     </div>
#                 </div>
#                 """,
#                 unsafe_allow_html=True
#             )
#     else:
#         st.info("No hay noticias registradas recientemente para este activo.")