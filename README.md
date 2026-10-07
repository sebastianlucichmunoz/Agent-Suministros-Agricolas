# Agent-Suministros-Agricolas

Sistema multiagente con Big Data de apoyo a la toma de decisiones para el monitoreo y predicción de la dinámica en la cadena de suministro agrícola.

> **Proyecto Académico / PMV:** Plataforma integral basada en aprendizaje automático y agentes de Inteligencia Artificial para el análisis continuo de commodities agrícolas e insumos clave en la cadena de suministro.

---

## Arquitectura del Sistema

```text
Yahoo Finance (yfinance) ──┐
                           ├── Python ETL ── MongoDB Atlas ── Streamlit (Dashboard Web)
RSS & Portales Agro ───────┤       │               │
                           │       ├── XGBoost     └── GitHub Actions (Automatización Daily)
                           └── NLTK / VADER
                                  │
                                  ├── Agente Recomendador (Gemini AI) ── Dictámenes Macro
                                  └── Agente de Alertas (SMTP) ─────── Notifications & Email
Insumos y Commodities Monitoreados

El sistema monitorea en tiempo real e históricamente 11 commodities estratégicos para la cadena de agro-suministros.

Granos y Cereales
Maíz (corn)
Trigo (wheat)
Soya (soybeans)
Arroz (rice)
Avena (oats)
Insumos y Energía
Gas Natural / Fertilizantes (natural_gas_fertilizer)
Petróleo Crudo / Transporte (crude_oil_transport)
Cultivos y Agroindustria
Café (coffee)
Cacao (cocoa)
Azúcar (sugar)
Algodón (cotton)
Colecciones en MongoDB Atlas

El esquema estandarizado utiliza asset_id como clave primaria en las siguientes colecciones:

Colección	Descripción
assets	Metadata de insumos y commodities.
market_snapshots	Estado actual de cotizaciones.
ohlcv	Serie de tiempo de precios: Apertura, Máximo, Mínimo, Cierre y Volumen.
news	Noticias del sector con análisis de sentimiento NLP.
models	Artefactos de los modelos de Machine Learning.
model_metrics	Métricas de rendimiento de los modelos: RMSE, MAE y R².
predictions	Proyecciones de precio a corto plazo generadas por XGBoost.
pipeline_runs	Logs de ejecución del orquestador.
Sistema Multiagente y Big Data

El sistema integra agentes especializados para la toma de decisiones.

Agente de Análisis Macro

Tecnología: Google Gemini AI

Evalúa:

Métricas de mercado.
Sentimiento de noticias.
Predicciones cuantitativas.

Con esta información genera dictámenes ejecutivos sobre la tendencia de la cadena de suministro.

Agente de Alertas y Notificaciones

Tecnología: SMTP

Evalúa variaciones atípicas de precios y dispara boletines por correo electrónico en caso de fluctuaciones severas en el mercado.

Frecuencia y Automatización

El workflow:

.github/workflows/refresh.yml

automatiza la ingesta, el reentrenamiento y la ejecución de los agentes en la nube mediante una programación cron ajustada a la hora de Lima (PET).

Hora Lima (PET)	Hora UTC	Evento
07:00 AM	12:00 UTC	Ingesta + Predicciones + Agentes
04:00 PM	21:00 UTC	Actualización de Mercado
06:30 PM	23:30 UTC	Cierre de Mercado y Dictamen Diario
Instalación y Ejecución Local
1. Clonar el repositorio
git clone https://github.com/sebastianlucichmunoz/Agent-Suministros-Agricolas.git
cd Agent-Suministros-Agricolas
2. Crear el entorno virtual
python -m venv .venv
.venv\Scripts\activate
3. Instalar dependencias
python -m pip install -r requirements.txt
Variables de Entorno

Crea un archivo .env en la raíz del proyecto con la siguiente estructura:

MONGODB_URI=mongodb+srv://<user>:<password>@cluster.mongodb.net/
MONGODB_DB_NAME=agri_analytics_db
GEMINI_API_KEY=tu_gemini_api_key

SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=tu_correo@gmail.com
SMTP_PASS=tu_app_password
EMAIL_FROM=tu_correo@gmail.com

No publiques el archivo .env ni las credenciales reales en el repositorio.

Comandos de Ejecución
Ejecutar ingesta y ETL local
python scripts/refresh.py
Ejecutar corrida diaria completa con reentrenamiento ML
python scripts/refresh.py --daily
Ejecutar el Agente de Recomendaciones Macro
python -m src.agents.recommendation_agent
Desplegar el Dashboard localmente
streamlit run app.py
Modelo de Machine Learning y Umbrales
Target

Predicción del precio de cierre para el siguiente periodo, aproximadamente 24 horas.

Variables utilizadas

Las principales variables o features utilizadas por el modelo son:

Retornos acumulados de 1d, 3d, 7d y 14d.
Medias móviles MA 7/14/30.
Medias móviles exponenciales EMA 7/21.
RSI 14.
Volatilidad acumulada.
Score de sentimiento diario derivado del NLP.
Clasificación de Tendencia
Tendencia	Condición
ALCISTA	Variación > +0.50%
ESTABLE	Variación entre -0.50% y +0.50%
BAJISTA	Variación < -0.50%
GitHub Actions Secrets Obligatorios

En el repositorio de GitHub, accede a:

Settings > Secrets and variables > Actions

y configura los siguientes Secrets:

MONGODB_URI
MONGODB_DB_NAME
GEMINI_API_KEY
SMTP_SERVER
SMTP_PORT
SMTP_USER
SMTP_PASS
EMAIL_FROM

Estos valores permiten que GitHub Actions ejecute los procesos automatizados sin exponer las credenciales directamente en el código fuente.

Despliegue Web con Streamlit Cloud

Para publicar el dashboard interactivo en la web mediante Streamlit Community Cloud:

Conecta el repositorio Agent-Suministros-Agricolas en Streamlit Community Cloud.
Configura los Secrets correspondientes a la conexión con MongoDB Atlas.
Utiliza la siguiente configuración:
MONGODB_URI = "tu_uri_de_mongodb_atlas"
MONGODB_DB_NAME = "agri_analytics_db"

El dashboard utilizará estas variables para conectarse a MongoDB Atlas.

Flujo General del Sistema
Yahoo Finance + RSS + Portales Agro
                │
                ▼
           Python ETL
                │
                ├── NLTK / VADER
                │
                ▼
          MongoDB Atlas
                │
        ┌───────┴────────┐
        │                │
        ▼                ▼
      XGBoost       Datos de Mercado
        │                │
        ▼                │
    Predicciones         │
        │                │
        └───────┬────────┘
                ▼
         Agentes IA
        ┌───────┴────────┐
        ▼                ▼
   Gemini AI          SMTP
        │                │
        ▼                ▼
 Dictamen Macro    Alertas Email
        │                │
        └───────┬────────┘
                ▼
          Streamlit
          Dashboard

Esta versión **no utiliza emojis ni elementos que den una apariencia artificial**, y conserva un estilo bastante normal para un README académico/técnico de GitHub.