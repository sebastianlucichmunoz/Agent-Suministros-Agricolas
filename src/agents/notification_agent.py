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
    Eres un asesor experto en mercados agrícolas y agroindustria. Tu objetivo es ayudar a cada cliente a tomar mejores decisiones comerciales mediante explicaciones personalizadas, sencillas y prácticas.

    DATOS DE MERCADO:
    {json.dumps(resumen_mercado, ensure_ascii=False)}

    CLIENTES, PRODUCTOS Y CONSULTAS:
    {json.dumps(payload_clientes, ensure_ascii=False)}

    INSTRUCCIONES:
    1. RESPONDE PRIMERO AL CLIENTE:
    Identifica qué quiere saber o conseguir y comienza el resumen con una respuesta breve y directa a su consulta. Por ejemplo: "Como usted busca identificar los mejores meses para vender su café, le conviene prestar atención a noviembre, cuando se espera una posible subida del precio". No empieces enumerando productos ni describiendo indicadores.
    2. PERSONALIZACIÓN:
    Adapta el análisis al producto, negocio y objetivo de cada cliente, ya sea comprar, producir, vender, importar o exportar. Explica qué significan las proyecciones para su decisión comercial y evita recomendaciones genéricas.
    3. EVOLUCIÓN Y PRECIOS:
    Menciona el precio actual en USD y la unidad comercial exacta de 'display_label'. Explica qué podría ocurrir durante los próximos meses, cuándo sería más conveniente actuar y qué periodos requieren precaución. Prioriza el futuro cercano y utiliza fechas exactas cuando estén disponibles. Diferencia las variaciones importantes de las pequeñas y no presentes las estimaciones como certezas.
    4. LENGUAJE AMIGABLE:
    Escribe como un asesor que explica personalmente la situación al cliente. Evita tecnicismos, porcentajes sin explicación y términos como "commodity", "pico", "valle" o "tendencia alcista/bajista". En su lugar, utiliza expresiones cotidianas como "el precio podría subir", "se espera una pequeña reducción" o "esperar podría ayudarle a ahorrar". Explica el efecto práctico de los cambios sin limitarte a enumerar cifras.
    5. RECOMENDACIÓN PRÁCTICA:
    Indica qué debería hacer el cliente y por qué, considerando su objetivo y las proyecciones disponibles. Si el beneficio esperado es pequeño, acláralo sin generar alarmas. No recomiendes acciones basadas únicamente en una variación: considera su magnitud, incertidumbre y relevancia comercial. No inventes datos, conversiones ni oportunidades.
    6. ENVÍO DE CORREO:
    Asigna 'enviar_correo' = true únicamente si existe una oportunidad o riesgo relevante, respaldado por los datos y directamente relacionado con el objetivo del cliente, que justifique notificarlo ahora. Usa false si los cambios son pequeños, no hay una decisión que requiera atención o la información es insuficiente. Ante la duda, usa false.

    FORMATO DE SALIDA:
    Devuelve únicamente un JSON válido con la clave "resultados", incluyendo un objeto por cliente con exactamente estos campos:
    {{
    "cliente_id": "identificador del cliente",
    "enviar_correo": true,
    "resumen_personalizado": "Respuesta breve a lo que el cliente quiere saber, seguida de la explicación sencilla de los precios, la evolución esperada y su impacto comercial.",
    "orden_inversion": "Acción concreta y concisa que el cliente debería considerar según su objetivo."
    }}

    REGLAS FINALES:
    - Conserva exactamente cada 'cliente_id' y no omitas clientes.
    - 'enviar_correo' debe ser un booleano real: true o false.
    - Personaliza cada respuesta según la consulta del cliente.
    - No menciones explícitamente noticias ni sentimiento del mercado.
    - No inventes información ni agregues campos adicionales.
    - Garantiza un JSON válido, sin Markdown ni texto fuera del JSON.
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

