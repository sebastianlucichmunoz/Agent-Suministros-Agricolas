from datetime import datetime, timezone
from src.db import get_db

def seed_test_clients():
    db = get_db()
    db.clients.delete_many({})
    
    clientes_iniciales = [
        {
            "name": "Sebastian Lucich",
            "email": "sebastianmartinlucich@gmail.com", 
            "watchlist": ["rice", "cotton"],
            "preferencia": (
                "Somos una empresa agroindustrial enfocada en la exportación de arroz y trigo al mercado internacional. Necesitamos saber con precisión cuál de los dos productos presenta un mejor horizonte de rentabilidad y proyección futura basada en el precio actual, ya que requerimos enfocar nuestros recursos y capital de inversión en uno de ellos. Queremos un análisis detallado que nos indique precios actuales, unidades de cotización, tendencias cercanas y una recomendación clara sobre cuál commodity priorizar y en qué fechas o meses críticos debemos tener cuidado ante posibles bajas para asegurar una decisión comercial rentable."
            ),
            "active": True,
            "created_at": datetime.now(timezone.utc)
        }
    ]
    
    db.clients.insert_many(clientes_iniciales)
    print(f"¡Se han insertado {len(clientes_iniciales)} clientes agrícolas de prueba en la colección 'clients'!")

if __name__ == "__main__":
    seed_test_clients()