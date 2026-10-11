# run_agro_alerts.py
import sys
from src.agents.notification_agent import ejecutar_agente_notificaciones
from src.agents.email_sender import enviar_alertas_pendientes

def ejecutar_flujo_alertas():
    print("INICIANDO FLUIDO DE NOTIFICACIONES AGRÍCOLAS")
    
    print("PASO 1: Evaluando lista de clientes y generando dictámenes...")
    ejecutar_agente_notificaciones()
    
    print("\nPASO 2: Procesando cola de correos y enviando alertas activas...")
    enviar_alertas_pendientes()
    
    print("\nCOMPLETADO: Flujo de notificaciones finalizado con éxito.")

if __name__ == "__main__":
    ejecutar_flujo_alertas()