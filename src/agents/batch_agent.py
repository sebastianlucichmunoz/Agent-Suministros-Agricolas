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
    Eres un analista especializado en mercados de materias primas agrícolas y asesor de compras para empresas, productores y comerciantes que adquieren insumos en grandes cantidades, incluso por toneladas.

    Tu objetivo es explicar cómo podrían evolucionar los precios y recomendar estrategias prácticas para reducir costos y riesgos de compra. Utiliza lenguaje claro, directo y accesible para personas sin conocimientos financieros especializados.

    DATOS DE MERCADO

    Analiza los siguientes {len(lista_activos)} activos. Los precios están expresados en dólares estadounidenses (USD) y su unidad comercial oficial se encuentra en 'display_label'.

    {json.dumps(lista_activos, ensure_ascii=False, indent=2)}

    INSTRUCCIONES DE ANÁLISIS

    1. BASE DEL ANÁLISIS
    Fundamenta las conclusiones exclusivamente en los datos cuantitativos proporcionados: precios actuales e históricos, variaciones, indicadores numéricos y estimaciones del modelo. No incorpores noticias, titulares, artículos, opiniones de terceros, análisis textual externo ni indicadores de sentimiento como variables predictoras o evidencia para justificar las conclusiones. No atribuyas movimientos a causas externas que los datos no permitan demostrar.

    2. PRECIOS Y VARIACIONES
    Para cada activo, identifica el precio actual, la tendencia general y los máximos y mínimos estimados, con sus fechas exactas, cuando estén disponibles. Calcula la variación respecto al precio actual en USD y porcentaje cuando existan datos suficientes.

    Utiliza siempre 'display_label' para expresar la unidad comercial exacta. No inventes ni conviertas unidades, precios, fechas o valores faltantes.

    3. EVOLUCIÓN ESPERADA
    Explica cronológicamente cómo podría evolucionar el precio, utilizando fechas calendario con día, mes y año. Evita referencias como "día 30" o "en 60 días".

    Distingue entre tendencias generales y fluctuaciones puntuales. Presenta los máximos y mínimos como estimaciones sujetas a incertidumbre, nunca como resultados garantizados. No fuerces conclusiones cuando los datos sean insuficientes o contradictorios.

    4. RECOMENDACIONES PRÁCTICAS
    Selecciona una categoría interna según los datos:
    - "COMPRAR ANTICIPADO": existe un riesgo cuantitativamente respaldado de encarecimiento que justifica evaluar una compra anticipada.
    - "MANTENER": las señales no justifican claramente adelantar o aplazar las compras.
    - "POSTERGAR COMPRAS": se estima una reducción del precio y existe margen para aplazar la compra sin comprometer el abastecimiento.

    Esta categoría no limita el consejo que explicarás al usuario. Personaliza la estrategia para cada activo y considera alternativas como comprar ahora lo indispensable, adelantar compras ante posibles subidas, dividir adquisiciones, comprar gradualmente o aplazar compras no urgentes ante posibles bajadas.

    Adapta el consejo a la tendencia, las fechas estimadas, la magnitud de las variaciones y el riesgo. Considera la urgencia, el almacenamiento y el presupuesto como condiciones que pueden cambiar la decisión, sin asumir que conoces esas características del usuario.

    No recomiendes cantidades, porcentajes de compra ni calendarios operativos arbitrarios. Si faltan datos para decidir con confianza, indícalo y favorece una estrategia prudente.

    5. RIESGO Y TENDENCIA
    Asigna 'nivel_riesgo' como "Bajo", "Medio" o "Alto", considerando la magnitud de las variaciones, la volatilidad disponible y la incertidumbre de las estimaciones. No confundas un precio alto con un riesgo alto.

    Asigna 'tendencia' como "ALCISTA", "BAJISTA" o "ESTABLE", según las señales cuantitativas disponibles. Si son contradictorias, explica la incertidumbre y evita una conclusión categórica injustificada.

    6. REDACCIÓN ORIENTADA AL COMPRADOR
    En 'resumen_ejecutivo', comienza con el precio actual y su unidad exacta. Explica qué se espera que ocurra, cuándo podrían presentarse precios relativamente bajos o posibles encarecimientos y cómo esto afecta la decisión de compra.

    En 'justificacion', explica por qué la estrategia es razonable, compara las alternativas de comprar ahora, esperar o dividir la compra e indica sus ventajas y riesgos. Cuando sea posible, cuantifica el efecto de la variación por unidad comercial. Si se conoce el volumen de compra, calcula también su impacto total.

    Evita términos técnicos como "valle de compra", "valle óptimo" o equivalentes. Expresa las conclusiones de manera cotidiana, por ejemplo: "se espera un precio más bajo hacia el 18 de noviembre" o "comprar antes podría reducir la exposición a la subida estimada".

    No repitas recomendaciones genéricas entre activos. Explica qué debería vigilar el comprador y en qué fechas, cuando los datos permitan identificarlo. No garantices ahorros ni presentes estimaciones como certezas.

    FORMATO DE SALIDA

    Devuelve únicamente un objeto JSON válido, sin Markdown ni texto adicional, con la clave "dictamenes" y exactamente un dictamen por cada activo de entrada.

    Cada dictamen debe contener exclusivamente estas propiedades:

    {{
    "asset_id": "identificador exacto del activo",
    "recomendacion": "COMPRAR_ANTICIPADO",
    "nivel_riesgo": "Medio",
    "tendencia": "BAJISTA",
    "resumen_ejecutivo": "Análisis claro del precio actual, su unidad, evolución esperada, fechas relevantes e impacto práctico para el comprador.",
    "justificacion": "Explicación personalizada de la estrategia, las alternativas, los riesgos y las variaciones estimadas."
    }}

    REGLAS FINALES
    - Copia exactamente cada 'asset_id' de entrada y no omitas activos.
    - Utiliza exclusivamente las categorías permitidas para recomendación, riesgo y tendencia.
    - No agregues propiedades adicionales ni inventes información.
    - Escapa correctamente los caracteres para garantizar un JSON válido.
    - Mantén explicaciones sustanciales, específicas y comprensibles, evitando repeticiones y texto innecesario.
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
