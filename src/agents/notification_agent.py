import json
import os
import time
from datetime import datetime, timezone
from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.db import get_db

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# PARÁMETROS CONFIGURABLES DEL AGENTE
MODELOS_A_PROBAR = ["gemini-3.5-flash", "gemini-3.5-flash-lite"]
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
            "nombre_agente": "agri_notification_agent",
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


def ejecutar_agente_notificaciones():
    print("-> El script del agente de notificaciones agrícolas se ha iniciado...")
    db = get_db()

    clientes = list(db.clients.find({"active": True}))
    if not clientes:
        print("No hay clientes activos para evaluar.")
        return

    insumos_requeridos = set()
    for cliente in clientes:
        for insumo in cliente.get("watchlist", []):
            insumos_requeridos.add(insumo)

    resumen_mercado = {}
    for asset_id in insumos_requeridos:
        summary = db.market_summary.find_one({"asset_id": asset_id})
        if summary:
            resumen_mercado[asset_id] = {
                "precio_actual": summary.get("current_price"),
                "tendencia_180d": summary.get("trend"),
                "variacion_pct_180d": summary.get("predicted_change_pct"),
                "puntos_clave": summary.get("puntos_clave_180d", summary.get("puntos_clave_fechas", [])),
                "sentimiento_noticias": summary.get("sentiment_label")
            }
        else:
            resumen_mercado[asset_id] = "Sin datos de predicción disponibles actualmente."

    payload_clientes = []
    for cliente in clientes:
        payload_clientes.append({
            "cliente_id": str(cliente["_id"]),
            "nombre": cliente.get("name", "Cliente Agro"),
            "email": cliente.get("email"),
            "watchlist": cliente.get("watchlist", []),
            "preferencia": cliente.get("preferencia", "MANTENER")
        })

    prompt_sistema = f"""
    Eres un consultor experto en cadenas de suministro agrícolas y agroindustria. Tu objetivo es redactar un análisis estratégico y directo para productores, compradores e inversionistas agrícolas.

    Analiza el siguiente resumen de mercado a 180 días para los insumos de los clientes y evalúa estrictamente sus preferencias individuales.
    
    Resumen de Mercado por Insumo Agrícola:
    {json.dumps(resumen_mercado, ensure_ascii=False)}

    Lista de Clientes y Preferencias:
    {json.dumps(payload_clientes, ensure_ascii=False)}

    INSTRUCCIONES ESTRICTAS DE RESPUESTA:
    Para cada cliente, determina si se debe enviar una alerta comparando los movimientos de sus insumos con su campo "preferencia".
    REGLA DE FECHAS: Esta prohibido decir "día 30" o "en 60 días". Utiliza siempre las fechas exactas o periodos naturales de tiempo presentes en los puntos clave.
    
    Devuelve OBLIGATORIAMENTE un objeto JSON con la clave "resultados" que contenga exactamente estos campos por cliente:
    - cliente_id (string)
    - enviar_correo (booleano: true o false)
    - resumen_personalizado (Un texto claro explicando los picos más altos/bajos proyectados con fechas exactas del calendario y la tendencia esperada para sus insumos de interés).
    - orden_inversion (Una instrucción concisa que incluya obligatoriamente el insumo, ej. "Comprar Maíz antes de Noviembre", "Esperar Valle en Trigo" o "Mantener inventario de Café").
    """

    for modelo in MODELOS_A_PROBAR:
        start_time = time.time()
        try:
            print(f"📡 Enviando consulta de notificaciones a Gemini usando: '{modelo}'...")

            config = types.GenerateContentConfig(
                temperature=TEMPERATURE,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                response_mime_type="application/json",
            )

            chat = client.chats.create(model=modelo, config=config)
            response = chat.send_message(prompt_sistema)

            elapsed_time = time.time() - start_time

            usage = getattr(response, "usage_metadata", None)
            tokens_meta = {
                "tokens_entrada_prompt": getattr(usage, "prompt_token_count", 0) if usage else 0,
                "tokens_salida_respuesta": getattr(usage, "candidates_token_count", 0) if usage else 0,
                "tokens_totales_consumidos": getattr(usage, "total_token_count", 0) if usage else 0,
            }

            registrar_log_agente(
                cantidad_insumos=len(insumos_requeridos),
                modelo_usado=modelo,
                tiempo_ejecucion_seg=elapsed_time,
                estado="EXITOSO",
                tokens_meta=tokens_meta,
            )

            data_resultado = json.loads(response.text)
            resultados_finales = []
            clientes_dict = {str(c["_id"]): c for c in clientes}

            for res in data_resultado.get("resultados", []):
                c_id = res.get("cliente_id")
                cliente_info = clientes_dict.get(c_id, {})

                registro_procesado = {
                    "cliente_id": c_id,
                    "nombre": cliente_info.get("name", "Desconocido"),
                    "email": cliente_info.get("email", ""),
                    "watchlist": cliente_info.get("watchlist", []),
                    "enviar_correo": res.get("enviar_correo", False),
                    "resumen_personalizado": res.get("resumen_personalizado", ""),
                    "orden_inversion": res.get("orden_inversion", ""),
                    "modelo_usado": modelo,
                    "fecha_generacion": datetime.now(timezone.utc)
                }
                resultados_finales.append(registro_procesado)

            log_documento = {
                "fecha_ejecucion": datetime.now(timezone.utc),
                "modelo": modelo,
                "resultados": resultados_finales
            }
            db.notification_logs.insert_one(log_documento)
            print("¡Resultados de notificaciones guardados en 'notification_logs'!")
            return

        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"❌ Error con el modelo '{modelo}': {e}")

            registrar_log_agente(
                cantidad_insumos=len(insumos_requeridos),
                modelo_usado=modelo,
                tiempo_ejecucion_seg=elapsed_time,
                estado="ERROR",
                detalle_error=str(e),
            )

            if "503" in str(e):
                time.sleep(5)
            else:
                raise e


if __name__ == "__main__":
    ejecutar_agente_notificaciones()

