import json
import os
import time
from datetime import datetime, timezone
from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.data.extractor import extraer_datos_consolidados_agricolas
from src.db import get_db

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

    {json.dumps(lista_activos, ensure_ascii=False, indent=2)}

    INSTRUCCIONES:

    1. ANÁLISIS: Utiliza exclusivamente los datos cuantitativos proporcionados. Evalúa el precio actual, su evolución histórica, las variaciones y las proyecciones disponibles. No utilices noticias, opiniones externas ni sentimiento del mercado.
    2. PRECIOS Y FECHAS: Menciona el precio actual y su unidad comercial exacta. Explica cuánto podría subir o bajar, en dólares y porcentaje cuando sea posible, e identifica las fechas relevantes. No inventes datos ni conversiones. Presenta las proyecciones como estimaciones, nunca como certezas.
    3. RECOMENDACIÓN DE COMPRA:
    - COMPRAR_ANTICIPADO: si se prevé un encarecimiento relevante que justifique comprar antes.
    - MANTENER: si las variaciones son pequeñas, inciertas o no justifican cambiar el momento de compra.
    - POSTERGAR_COMPRAS: si se espera una reducción relevante y es razonable esperar sin comprometer el abastecimiento.

    Considera la magnitud del cambio, su duración, las posibles fluctuaciones intermedias y la incertidumbre. No recomiendes esperar únicamente porque el precio podría bajar si el ahorro sería mínimo o el riesgo de encarecimiento es importante. Si las señales son contradictorias, prioriza la prudencia.

    4. REDACCIÓN CLARA Y AMIGABLE:
    - En 'resumen_ejecutivo', comienza con el precio actual y explica, con palabras sencillas, qué podría ocurrir durante los próximos meses y cuándo convendría comprar.
    - En 'justificacion', explica por qué recomiendas comprar ahora, esperar o dividir las compras. Traduce las variaciones en su impacto económico por unidad comercial y por volumen cuando se conozca.
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
    "resumen_ejecutivo": "Explicación sencilla del precio actual, su unidad, los cambios esperados y el momento de compra más conveniente.",
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


def ejecutar_proceso_diario_agricola():
    print("[Paso 1] Extrayendo y comprimiendo datos a 180 días para MongoDB...")
    lista_activos = extraer_datos_consolidados_agricolas()

    if not lista_activos:
        print("No se encontraron datos agrícolas para procesar.")
        return

    print(f"[Paso 2] Consolidados {len(lista_activos)} activos. Procesando en LOTE con IA...")

    for intento in range(1, 4):
        try:
            resultado_agente = analizar_mercado_agricola_en_lote(lista_activos)

            reporte_final = {
                "fecha_actualizacion": datetime.now(timezone.utc).isoformat(),
                "total_activos": len(lista_activos),
                "analisis_diario": resultado_agente.get("dictamenes", []),
            }

            os.makedirs("data", exist_ok=True)
            ruta_salida = os.path.join("data", "daily_agri_recommendations.json")

            with open(ruta_salida, "w", encoding="utf-8") as f:
                json.dump(reporte_final, f, ensure_ascii=False, indent=2)

            db = get_db()
            db.agent_recommendations.update_one(
                {"fecha_actualizacion": reporte_final["fecha_actualizacion"][:10]},
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


if __name__ == "__main__":
    ejecutar_proceso_diario_agricola()
