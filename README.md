# Agent-Suministros-Agricolas

**Plataforma de análisis y apoyo a decisiones para el abastecimiento agrícola**, desarrollada como proyecto académico / producto mínimo viable (PMV).

Integra cotizaciones históricas, modelos predictivos de Machine Learning, un asesor basado en Google Gemini y un dashboard web en Streamlit. Su finalidad es ayudar a comprender cómo podrían variar los precios de insumos agrícolas y evaluar decisiones de compra, venta y abastecimiento.

> **Alcance:** Los precios y pronósticos son información de apoyo, no garantías de rentabilidad ni asesoramiento financiero personalizado. Los resultados dependen de la disponibilidad y actualización de los datos.

## 1. Funcionalidades principales

### Panel de mercado

- Seguimiento de **11 productos**: maíz, trigo, soya, arroz, avena, café, cacao, azúcar, algodón, petróleo WTI y gas natural.
- Consulta de la **última cotización disponible**, con fecha, precio y unidad comercial del producto.
- Gráfico interactivo que une **histórico de precios y proyección de hasta seis meses**, con una marca de la última cotización.
- Tarjetas de evaluación del modelo: **R²**, **WAPE**, **MAE** y **precisión direccional**, explicadas de forma sencilla.
- Dictámenes del agente de análisis de mercado y recomendaciones orientadas a decisiones de abastecimiento.
- Navegación y visualización adaptadas al perfil del usuario.

### Usuarios y consultas personalizadas

- Registro e inicio de sesión, perfil empresarial e insumos de interés.
- Dos modalidades independientes de consulta a la IA:
  - **Instantáneas:** generan una respuesta en el momento y permiten solicitar una copia por correo.
  - **Programadas:** se registran para su procesamiento y posterior entrega por correo, con referencia operativa a las **8:00 a. m. de Lima**.
- Selección de hasta **cinco insumos** por consulta.
- Cupo diario de consultas instantáneas y límite independiente de consultas programadas; el contador utiliza la zona horaria **America/Lima**.
- Aviso claro cuando se agota el cupo diario.
- Historial de consultas con pregunta, respuesta, acción prioritaria y estado; filtros por mes y año.
- Correo en formato HTML con **pregunta realizada, respuesta del asesor y acción prioritaria**.

### Asesor agrícola con Gemini

El agente responde consultas de precios y evolución futura, comparaciones entre productos, oportunidades de compra o venta, presupuestos y escenarios económicos aproximados. Cuando el usuario aporta un capital, una cantidad o un plazo, puede analizar su impacto usando los datos disponibles.

Las respuestas priorizan la pregunta del usuario, explican las fechas de manera natural y utilizan lenguaje comprensible. El agente distingue las **variaciones de valor estimadas** de las **ganancias netas**, que requieren información adicional sobre costos y condiciones comerciales. No inventa cotizaciones, conversiones ni datos de producción, clima o calidad.

## 2. Arquitectura general

```text
Yahoo Finance / yfinance
          │
          ▼
   Extracción y ETL ───────────────┐
          │                        │
          ▼                        ▼
     MongoDB Atlas          Datos y cotizaciones
          │
          ├── Modelo predictivo (XGBoost / alternativas evaluadas)
          │       └── Proyecciones y métricas de evaluación
          │
          ├── Agente de análisis general (Gemini)
          │       └── Dictámenes de mercado
          │
          └── Agente de consultas (Gemini)
                  ├── Consultas instantáneas
                  └── Consultas programadas ── SMTP / correo
          │
          ▼
     Streamlit (dashboard y perfil de usuario)

GitHub Actions: automatización de extracción, predicciones y agentes
```

El proyecto contempla además el procesamiento de noticias y sentimiento como información complementaria del ecosistema; las recomendaciones basadas en precios deben sustentarse en los datos cuantitativos disponibles.

## 3. Tecnologías

| Componente | Tecnología |
|---|---|
| Aplicación web y visualizaciones | Python, Streamlit, Plotly |
| Extracción de cotizaciones | yfinance |
| Almacenamiento | MongoDB Atlas (PyMongo) |
| Modelos predictivos | XGBoost y modelos de comparación, según configuración |
| Asesor y dictámenes | Google Gemini API |
| Envío de respuestas | SMTP |
| Ejecuciones automáticas | GitHub Actions |
| Procesamiento complementario de noticias | RSS y análisis de sentimiento, cuando esté habilitado |

## 4. Modelo predictivo

El sistema utiliza series históricas de precios e indicadores derivados, como retornos, medias móviles, volatilidad y medidas de comportamiento del mercado. Genera estimaciones con un horizonte de hasta **180 días** y realiza evaluaciones temporales por producto y horizonte.

Los resultados se presentan mediante cuatro indicadores:

| Indicador | Interpretación |
|---|---|
| **R²** | Qué tan bien se ajustan los valores estimados a los observados; no es un porcentaje de aciertos. |
| **WAPE** | Magnitud del error porcentual de las estimaciones. |
| **MAE** | Diferencia promedio entre precios estimados y reales, en la unidad monetaria correspondiente. |
| **Precisión direccional** | Frecuencia con la que se anticipó correctamente una subida o bajada. |

Las métricas históricas no garantizan que las proyecciones futuras tengan el mismo desempeño. Las unidades comerciales se respetan en cada producto para evitar comparaciones engañosas.

## 5. Agentes y módulos del proyecto

```text
app.py
src/
  agents/
    batch_agent.py
    consultation_agent.py
    scheduled_consultations.py
    email_sender.py
  data/
    extractor.py
  ml/
    model.py
.github/
  workflows/
    refresh.yml
```

- **`app.py`:** dashboard, visualizaciones, autenticación, perfil, consultas e historial.
- **`batch_agent.py`:** análisis general de los insumos y generación de dictámenes para el panel.
- **`consultation_agent.py`:** consultas instantáneas, contexto de precios y pronósticos, límites y registro de consultas programadas.
- **`scheduled_consultations.py`:** recuperación y procesamiento de consultas pendientes y envío de respuestas.
- **`email_sender.py`:** correo HTML y conexión SMTP; incluye la pregunta original cuando el módulo que lo llama la proporciona.
- **`extractor.py`:** preparación de datos consolidados para análisis.
- **`model.py`:** entrenamiento, evaluación y generación de proyecciones.

## 6. Colecciones principales de MongoDB

| Colección | Propósito |
|---|---|
| `assets_catalog` | Información y unidades de los productos. |
| `ohlcv` | Precios históricos y cotizaciones. |
| `predictions` | Predicciones de precios por producto. |
| `model_metrics` | Métricas de evaluación de los modelos. |
| `agent_recommendations` | Dictámenes generales del agente de mercado. |
| `agent_logs` | Registros técnicos de las ejecuciones de los agentes. |
| `clientes_` | Usuarios y perfiles empresariales. |
| `consultas_ia` | Preguntas, respuestas, consultas programadas y estados de procesamiento. |
| `consulta_cupos` | Control de consultas instantáneas y programadas. |

## 7. Instalación local

**Requisitos:** Python, acceso a MongoDB Atlas y claves para los servicios que se utilizarán.

```bash
git clone https://github.com/sebastianlucichmunoz/Agent-Suministros-Agricolas.git
cd Agent-Suministros-Agricolas
python -m venv .venv
```

Activar el entorno virtual:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
source .venv/bin/activate
```

Instalar las dependencias:

```bash
python -m pip install -r requirements.txt
```

Crear el archivo `.env` en la raíz del proyecto:

```dotenv
MONGODB_URI=mongodb+srv://USUARIO:CONTRASENA@CLUSTER.mongodb.net/
MONGODB_DB_NAME=agri_analytics_db
GEMINI_API_KEY=TU_CLAVE_GEMINI

SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=tu_correo@gmail.com
SMTP_PASSWORD=tu_contrasena_de_aplicacion
```

> **Seguridad:** Nunca subir `.env`, contraseñas o claves de API al repositorio. El nombre de variable utilizado por el código del correo es `SMTP_PASSWORD`.

Iniciar la aplicación:

```bash
streamlit run app.py
```

## 8. Ejecución de agentes y automatización

Ejecutar el análisis general:

```bash
python -m src.agents.batch_agent
```

Ejecutar únicamente el procesamiento de consultas programadas:

```bash
python -m src.agents.scheduled_consultations
```

El flujo automatizado se administra mediante **`.github/workflows/refresh.yml`**. Para mantener la información consistente conviene que el procesamiento de dictámenes ocurra después de actualizar las cotizaciones y predicciones. Las consultas programadas requieren una ejecución a las **13:00 UTC (8:00 a. m. de Lima)** o posteriormente, de acuerdo con la configuración del workflow.

> GitHub Actions puede comenzar una ejecución programada con retraso. La hora indicada es una referencia de programación, no una garantía de recepción exacta del correo.

### Límites de consultas

Los límites se administran en `src/agents/consultation_agent.py`:

```python
MAX_CONSULTAS_DIARIAS = 10
MAX_PROGRAMADAS_POR_DIA = 10
```

La configuración anterior representa **10 consultas instantáneas al día por usuario y 10 programadas por fecha de entrega**. Verifica que estos valores estén efectivamente aplicados en la versión desplegada: modificar el README no modifica el código. Los formularios también deben reflejar los mismos límites.

Para que el correo muestre la pregunta original, tanto las consultas instantáneas como las programadas deben llamar a `send_notification_email(...)` incluyendo el argumento `pregunta=`.

## 9. Despliegue en Streamlit Community Cloud

1. Publicar el repositorio en GitHub.
2. Crear la aplicación en Streamlit Community Cloud y seleccionar `app.py`.
3. Configurar los secretos de MongoDB, Gemini y SMTP en un formato compatible con las variables de entorno del proyecto.
4. Autorizar la conexión a MongoDB Atlas desde el entorno de despliegue.
5. Verificar autenticación, lectura de cotizaciones, consultas, historial y correos.

En **GitHub Actions → Settings → Secrets and variables → Actions**, configurar las credenciales necesarias para los trabajos automatizados. No añadirlas directamente a los archivos YAML.

## 10. Comprobaciones recomendadas

- Confirmar que la fecha de última cotización y la fecha de generación de la predicción sean coherentes.
- Verificar las cuatro métricas por producto, sus unidades y el horizonte evaluado.
- Probar consultas simples, comparaciones de insumos y preguntas con presupuestos.
- Confirmar el cupo diario con la hora de Lima y el aviso de límite alcanzado.
- Programar una consulta y revisar su transición entre estados (`Programada`, `Procesando`, `Completado` o `Error`).
- Comprobar la recepción del correo con **pregunta, respuesta y acción prioritaria**.
- Revisar la ejecución real de los workflows y los registros de agentes.

---

**Estado del proyecto:** PMV académico en evolución. La calidad de las decisiones sugeridas depende de la actualización, disponibilidad y precisión de los datos y de los modelos empleados.
