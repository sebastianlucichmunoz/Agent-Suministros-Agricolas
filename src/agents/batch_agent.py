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


# PARÁMETROS CONFIGURABLES DEL AGENTE
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
    """Guarda un registro de auditoría técnica en español dentro de 'agent_logs'."""
    try:
        db = get_db()
        log_entry = {
            "fecha_hora": datetime.now(timezone.utc),
            "nombre_agente": "agri_batch_agent",
            "modelo_utilizado": modelo_usado,
            "estado": estado,  # "EXITOSO" o "ERROR"
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
        print("📜 [Auditoría] Registro técnico guardado exitosamente en 'agent_logs'.")
    except Exception as e:
        print(f"⚠️ Error al guardar el log de auditoría en MongoDB: {e}")


def analizar_mercado_agricola_en_lote(lista_activos: list):
    """Envía la lista completa de commodities/insumos agrícolas registrando métricas técnicas."""
    prompt = f"""
    Eres un analista experto en commodities agrícolas y agroindustria que explica las tendencias de insumos y alimentos de forma clara y directa para productores, compradores y empresarios agrícolas.
    
    A continuación se proporciona la información compacta de mercado, picos/valles clave a 180 días y el análisis de sentimiento de noticias para {len(lista_activos)} materias primas:

    {json.dumps(lista_activos, ensure_ascii=False, indent=2)}

    Instrucciones obligatorias de redacción:
    1. REGLA ESTRICTA DE FECHAS: Está PROHIBIDO mencionar números de días técnicos como "día 30", "día 45" o "en 60 días". Refiérete a los momentos del mercado SIEMPRE usando la fecha exacta (ej. "el 12 de noviembre de 2026") o periodos naturales de tiempo (ej. "a mediados de noviembre", "durante la tercera semana de diciembre").
    2. RESUMEN EJECUTIVO: Indica el precio actual, el pico más alto/bajo proyectado con su fecha exacta y el porcentaje de cambio esperado.
    3. RECOMENDACIÓN TÁCTICA: Explica claramente qué acción tomar según la fecha proyectada.
    
    Devuelve OBLIGATORIAMENTE un arreglo JSON bajo la clave "dictamenes", con la siguiente estructura exacta:
    [
      {{
        "asset_id": "identificador-del-activo",
        "recomendacion": "COMPRAR_ANTICIPADO" | "MANTENER" | "ESPERAR_VALLE",
        "nivel_riesgo": "Bajo" | "Medio" | "Alto",
        "tendencia": "ALCISTA" | "BAJISTA" | "ESTABLE",
        "resumen_ejecutivo": "Explicación clara del precio actual, picos proyectados a 180 días y cambio % acumulado.",
        "justificacion": "Análisis táctico claro para la toma de decisiones comerciales y agrícolas."
      }}
    ]
    """

    for modelo in MODELOS_A_PROBAR:
        start_time = time.time()
        try:
            print(f"📡 Enviando lote agrícola a Gemini usando modelo: '{modelo}'...")

            config = types.GenerateContentConfig(
                temperature=TEMPERATURE,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                response_mime_type="application/json",
            )

            chat = client.chats.create(model=modelo, config=config)
            response = chat.send_message(prompt)

            elapsed_time = time.time() - start_time

            # Extraer consumo de tokens
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

            # Guardar Log de éxito
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
            print(f"❌ Error con el modelo '{modelo}': {e}")

            # Guardar Log de Error
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
