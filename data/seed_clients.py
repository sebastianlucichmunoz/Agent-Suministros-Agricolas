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
                "Empresario agroindustrial interesado en importar Arroz y Algodón. "
                "Necesito identificar la ventana oportuna de importación por caída de precios "
                "y la duración estimada de dicha bajada para planificar mis compras estratégicas."
            ),
            "active": True,
            "created_at": datetime.now(timezone.utc)
        }
        # {
        #     "name": "María Gómez",
        #     "email": "sebastianmarlucichmunoz@gmail.com",
        #     "watchlist": ["corn", "wheat"],
        #     "preferencia": (
        #         "Compradora de insumos de alimentos balanceados. "
        #         "Busco alertas de alzas de precio en Maíz y Trigo para realizar compras anticipadas "
        #         "y proteger el margen operativo de producción."
        #     ),
        #     "active": True,
        #     "created_at": datetime.now(timezone.utc)
        # }
    ]
    
    db.clients.insert_many(clientes_iniciales)
    print(f"¡Se han insertado {len(clientes_iniciales)} clientes agrícolas de prueba en la colección 'clients'!")

if __name__ == "__main__":
    seed_test_clients()