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
    try:
        db = get_db()
        log_entry = {
            "fecha_hora": datetime.now(timezone.utc),
            "nombre_agente": "agri_notification_agent",
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


def ejecutar_agente_notificaciones():
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
                "tendencia": summary.get("trend"),
                "variacion_pct": summary.get("predicted_change_pct"),
                "display_label": summary.get("display_label", "USD / Unidad"),
                "puntos_clave": summary.get("puntos_clave_fechas", [])
            }
        else:
            resumen_mercado[asset_id] = "Sin datos disponibles."

    payload_clientes = []
    for cliente in clientes:
        payload_clientes.append({
            "cliente_id": str(cliente["_id"]),
            "nombre": cliente.get("name", "Cliente Agro"),
            "email": cliente.get("email"),
            "watchlist": cliente.get("watchlist", []),
            "perfil_o_consulta": cliente.get("preferencia", "Empresario buscando optimizar compras y ventas de insumos agrícolas.")
        })

    prompt_sistema = f"""
    Eres un asesor experto en mercados agrícolas y agroindustria. Analiza los datos disponibles y genera recomendaciones comerciales personalizadas según el perfil, los productos y los objetivos de cada cliente.

    Datos de mercado:
    {json.dumps(resumen_mercado, ensure_ascii=False)}

    Clientes, watchlists y consultas personalizadas:
    {json.dumps(payload_clientes, ensure_ascii=False)}

    Instrucciones:
    -Personalización: Adapta cada análisis al negocio y objetivo del cliente (café, trigo u otro producto), considerando si busca comprar, producir, vender o exportar para maximizar sus ganancias. No generes recomendaciones genéricas.
    -Proyección cercana: Prioriza los próximos meses con datos disponibles y relevancia comercial. Identifica los periodos más convenientes y aquellos que requieren precaución. No extiendas innecesariamente el horizonte de predicción.
    -Resumen comercial: Indica el precio actual en dólares y su unidad exacta según display_label, explica su equivalencia en kilogramos cuando corresponda y describe la evolución prevista con fechas concretas. Traduce las proyecciones en recomendaciones prácticas. Si los cambios esperados son mínimos, indícalo claramente y evita generar alarmas.
    -Envío estricto (enviar_correo): Usa true únicamente cuando exista una oportunidad o un riesgo futuro suficientemente relevante, respaldado por los datos y directamente relacionado con los objetivos del cliente, que justifique contactarlo ahora. Usa false si los cambios son mínimos, no hay una decisión comercial que requiera atención, la información es insuficiente o la recomendación no aporta valor concreto. Ante la duda, usa true.
    -Comunicación: Emplea lenguaje claro, dinámico y orientado al negocio. No menciones explícitamente el sentimiento del mercado, noticias ni tecnicismos innecesarios. No inventes precios, conversiones, fechas ni proyecciones; si falta información, omite el dato no verificable y aclara la limitación cuando sea relevante.
    
    Formato de salida obligatorio: Devuelve únicamente un JSON válido con la clave resultados, que contenga un objeto por cliente y exactamente estos campos:
    cliente_id: string.
    enviar_correo: booleano real (true o false).
    resumen_personalizado: recomendación comercial clara, específica y sustentada en los datos disponibles.
    orden_inversion (Instrucción concisa y práctica orientada a su objetivo).
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
            print(f"Error con el modelo '{modelo}': {e}")

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

