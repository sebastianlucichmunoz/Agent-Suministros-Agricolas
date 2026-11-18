import json
import os
import time
from datetime import datetime, timezone
from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.data.extractor import extraer_datos_consolidados_agricolas
from src.db import get_db
from src.agents.consultation_agent import _datos_insumo

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# PARÁMETROS DEL AGENTE
MODELOS_A_PROBAR = ["gemini-3.5-flash-lite", "gemini-3.5-flash"]
TEMPERATURE = 0.2
MAX_OUTPUT_TOKENS = 4000


def registrar_log_agente(
    cantidad_insumos: int,
    modelo_usado: str,
    tiempo_ejecucion_seg: float,
    estado: str,
    tokens_meta: dict = None,
    detalle_error: str = None,
):
    try:
        db = get_db()
        log_entry = {
            "fecha_hora": datetime.now(timezone.utc),
            "nombre_agente": "agri_batch_agent",
            "modelo_utilizado": modelo_usado,
            "estado": estado,
            "tiempo_ejecucion_segundos": round(tiempo_ejecucion_seg, 3),
            "cantidad_insumos_procesados": cantidad_insumos,
            "configuracion_parametros": {
                "creatividad_temperatura": TEMPERATURE,
                "limite_tokens_salida": MAX_OUTPUT_TOKENS,
            },
            "consumo_tokens": tokens_meta
            or {
                "tokens_entrada_prompt": 0,
                "tokens_salida_respuesta": 0,
                "tokens_totales_consumidos": 0,
            },
            "detalle_error": detalle_error,
        }

        db.agent_logs.insert_one(log_entry)
        print("Registro técnico guardado exitosamente en 'agent_logs'.")
    except Exception as e:
        print(f"Error al guardar el log de auditoría en MongoDB: {e}")


def analizar_mercado_agricola_en_lote(lista_activos: list):
    prompt = f"""
    Eres un asesor de compras especializado en productos agrícolas. Tu función es explicar cómo podrían cambiar sus precios y orientar al comprador para tomar mejores decisiones y reducir costos.

    DATOS DE MERCADO:
    Analiza los {len(lista_activos)} activos. Los precios están en USD y su unidad oficial figura en 'display_label'.

    {json.dumps(lista_activos, ensure_ascii=False, separators=(",", ":"), default=str)}

    INSTRUCCIONES:

    1. ANÁLISIS: Prioriza el bloque cotizacion_verificada para precios, fechas, unidades y estimaciones. Si difiere de otros campos, utiliza exclusivamente el dato verificado. Utiliza exclusivamente los datos cuantitativos proporcionados. Evalúa internamente el precio actual como referencia, pero no lo menciones en el resumen ejecutivo ni en la justificación. Centra las explicaciones en la evolución prevista, las variaciones y las oportunidades de compra. No utilices noticias, opiniones externas ni sentimiento del mercado.
    2. PRECIOS Y FECHAS: No menciones ni reproduzcas la cotización actual, ya que se presenta por separado en el dashboard. Explica los precios futuros estimados, cuánto podrían subir o bajar en dólares y porcentaje cuando sea posible, e identifica las fechas relevantes. Respeta siempre la unidad comercial indicada y evita conversiones no proporcionadas. Presenta las proyecciones como estimaciones, nunca como certezas.
    3. RECOMENDACIÓN DE COMPRA:
    - COMPRAR ANTICIPADO: si se prevé un encarecimiento relevante que justifique comprar antes.
    - MANTENER: si las variaciones son pequeñas, inciertas o no justifican cambiar el momento de compra.
    - POSTERGAR COMPRAS: si se espera una reducción relevante y es razonable esperar sin comprometer el abastecimiento.

    Considera la magnitud del cambio, su duración, las posibles fluctuaciones intermedias y la incertidumbre. No recomiendes esperar únicamente porque el precio podría bajar si el ahorro sería mínimo o el riesgo de encarecimiento es importante. Si las señales son contradictorias, prioriza la prudencia.

    4. REDACCIÓN CLARA Y AMIGABLE:
    - En 'resumen_ejecutivo', comienza siempre mencionando explícitamente el nombre del producto, utilizando una expresión natural como "El precio del azúcar...", "El precio del maíz..." o "El precio del trigo...". Después explica cómo podría evolucionar durante los próximos meses, destacando los aumentos, reducciones, fechas importantes y oportunidades de compra. No comiences mencionando unidades comerciales como "por libra", "por bushel" o "por barril", ni incluyas la cotización actual. Las unidades comerciales deben aparecer únicamente cuando acompañen valores de precios futuros estimados.
    - En 'justificacion', explica por qué recomiendas comprar anticipadamente, mantener o postergar las compras. Considera los cambios esperados, su importancia económica y los posibles beneficios o riesgos de cada alternativa, sin repetir el precio actual.
    - Evita palabras técnicas como "commodity", "pico", "valle", "tendencia alcista/bajista" dentro de las explicaciones. Utiliza expresiones como "el precio podría subir", "se espera una reducción" o "la variación sería pequeña". Es importante que todo se explique de forma amigable y práctica.
    - Personaliza cada recomendación según los datos del producto. No asumas que el comprador tiene capacidad de almacenamiento, presupuesto disponible o libertad para aplazar sus compras.
    - Si los cambios previstos son mínimos, indícalo claramente y explica si realmente vale la pena modificar el plan de compra.

    5. CLASIFICACIÓN:
    - 'nivel_riesgo': exclusivamente "Bajo", "Medio" o "Alto", según la variabilidad y la incertidumbre, no por el precio en sí.
    - 'tendencia': exclusivamente "ALCISTA", "BAJISTA" o "ESTABLE", según los datos cuantitativos. Si la evolución es contradictoria, no fuerces una conclusión.

    FORMATO DE SALIDA:
    Devuelve únicamente JSON válido, sin Markdown ni texto adicional, con la clave "dictamenes" y exactamente un objeto por activo:

    {{
    "asset_id": "identificador exacto",
    "recomendacion": "MANTENER",
    "nivel_riesgo": "Medio",
    "tendencia": "ESTABLE",
    "resumen_ejecutivo": "Comienza con 'El precio del [nombre del producto]...' y explica su evolución futura, los principales cambios, las fechas relevantes y el momento de compra conveniente. No incluyas la cotización actual ni comiences con unidades comerciales.",
    "justificacion": "Motivo de la recomendación, comparación de alternativas y ahorro o riesgo económico estimado."
    }}

    REGLAS FINALES:
    - Conserva exactamente cada 'asset_id' y no omitas activos.
    - Respeta las categorías permitidas y las propiedades indicadas.
    - No inventes datos, fechas ni ahorros garantizados.
    - Garantiza un JSON válido, sin propiedades adicionales.
    - Sé claro, natural, específico y conciso.
    """
    for modelo in MODELOS_A_PROBAR:
        start_time = time.time()
        try:
            config = types.GenerateContentConfig(
                temperature=TEMPERATURE,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                response_mime_type="application/json",
            )

            chat = client.chats.create(model=modelo, config=config)
            response = chat.send_message(prompt)

            elapsed_time = time.time() - start_time

            usage = getattr(response, "usage_metadata", None)
            tokens_meta = {
                "tokens_entrada_prompt": getattr(usage, "prompt_token_count", 0)
                if usage
                else 0,
                "tokens_salida_respuesta": getattr(usage, "candidates_token_count", 0)
                if usage
                else 0,
                "tokens_totales_consumidos": getattr(usage, "total_token_count", 0)
                if usage
                else 0,
            }

            registrar_log_agente(
                cantidad_insumos=len(lista_activos),
                modelo_usado=modelo,
                tiempo_ejecucion_seg=elapsed_time,
                estado="EXITOSO",
                tokens_meta=tokens_meta,
            )

            return json.loads(response.text)

        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"Error con el modelo '{modelo}': {e}")
            registrar_log_agente(
                cantidad_insumos=len(lista_activos),
                modelo_usado=modelo,
                tiempo_ejecucion_seg=elapsed_time,
                estado="ERROR",
                detalle_error=str(e),
            )

            if "503" in str(e):
                print(
                    "El modelo está saturado (503). Probando con el modelo de respaldo..."
                )
                time.sleep(5)
            else:
                raise e

    raise RuntimeError("Todos los modelos de Gemini fallaron o están saturados.")


def _ejecutar_dictamen_diario_agricola():
    print("[Paso 1] Extrayendo y comprimiendo datos a 180 días para MongoDB...")
    lista_activos = extraer_datos_consolidados_agricolas()
    db = get_db()
    for activo in lista_activos:
        asset_id = activo.get("asset_id")
        if not asset_id:
            continue
        referencia = _datos_insumo(db, asset_id)
        if referencia:
            puntos = referencia["proyeccion"]
            activo["cotizacion_verificada"] = {
                "insumo": referencia["insumo"],
                "unidad": referencia["unidad"],
                "precio_actual_usd": referencia["precio_actual_usd"],
                "fecha_ultima_cotizacion": referencia["fecha_ultima_cotizacion"],
                "variacion_estimada_pct": referencia["variacion_estimada_pct"],
                "fecha_prediccion": referencia["fecha_prediccion"],
                "precio_final_usd": puntos[-1]["precio_usd"],
                "fecha_final": puntos[-1]["fecha"],
                "minimo_futuro": min(puntos, key=lambda x: x["precio_usd"]),
                "maximo_futuro": max(puntos, key=lambda x: x["precio_usd"]),
            }


    if not lista_activos:
        print("No se encontraron datos agrícolas para procesar.")
        return

    print(f"[Paso 2] Consolidados {len(lista_activos)} activos. Procesando en LOTE con IA...")

    for intento in range(1, 4):
        try:
            resultado_agente = analizar_mercado_agricola_en_lote(lista_activos)

            reporte_final = {
                "fecha_actualizacion": datetime.now(timezone.utc).isoformat(),
                "fecha_dia": datetime.now(timezone.utc).date().isoformat(),
                "total_activos": len(lista_activos),
                "analisis_diario": resultado_agente.get("dictamenes", []),
            }

            os.makedirs("data", exist_ok=True)
            ruta_salida = os.path.join("data", "daily_agri_recommendations.json")

            with open(ruta_salida, "w", encoding="utf-8") as f:
                json.dump(reporte_final, f, ensure_ascii=False, indent=2)

            db = get_db()
            db.agent_recommendations.update_one(
                {"fecha_dia": reporte_final["fecha_dia"]},
                {"$set": reporte_final},
                upsert=True,
            )
            print("¡Recomendaciones guardadas exitosamente en 'agent_recommendations' de MongoDB!")
            break

        except Exception as e:
            print(
                f"Error en procesamiento en lote (Intento {intento}/3): {e}. Esperando 30s..."
            )
            time.sleep(30)


# ===== ORQUESTACIÓN DE DICTÁMENES Y CONSULTAS PROGRAMADAS =====
# ===== EJECUCIÓN DEL AGENTE DE DICTÁMENES =====
def ejecutar_proceso_diario_agricola():
    _ejecutar_dictamen_diario_agricola()


if __name__ == "__main__":
    ejecutar_proceso_diario_agricola()

