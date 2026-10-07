import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from src.db import get_db

def enviar_alertas_pendientes():
    db = get_db()
    
    ultimo_log = db.notification_logs.find_one(sort=[("fecha_ejecucion", -1)])
    
    if not ultimo_log or "resultados" not in ultimo_log:
        print("No se encontraron registros de ejecución recientes en 'notification_logs'.")
        return

    resultados = ultimo_log["resultados"]

    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    port_env = os.getenv("SMTP_PORT")
    smtp_port = int(port_env) if port_env and port_env.strip() else 587
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")

    if not smtp_user or not smtp_password:
        print("Error: Las credenciales SMTP (SMTP_USER y SMTP_PASSWORD) no están configuradas.")
        return

    print("Conectando al servidor SMTP...")
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_user, smtp_password)
        print("¡Conexión SMTP establecida con éxito!")

        enviados_count = 0

        for reg in resultados:
            if reg.get("enviar_correo") is True:
                destinatario = reg.get("email")
                nombre = reg.get("nombre", "Estimado Productor/Comprador")
                resumen = reg.get("resumen_personalizado")
                orden = reg.get("orden_inversion")
                watchlist = ", ".join(reg.get("watchlist", []))

                msg = MIMEMultipart("alternative")
                msg["Subject"] = f"🌾 Alerta Agrícola de Mercado: {orden}"
                msg["From"] = smtp_user
                msg["To"] = destinatario

                html_content = f"""
                <html>
                  <body style="font-family: Arial, sans-serif; color: #333333; line-height: 1.6;">
                    <div style="max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #E2E8F0; border-radius: 8px; background-color: #F8FAFC;">
                      <h2 style="color: #15803D; margin-top: 0;">Hola, {nombre}</h2>
                      <p>Has recibido un informe táctico de mercado para tus insumos agrícolas de interés (<strong>{watchlist}</strong>).</p>
                      
                      <div style="background-color: #FFFFFF; padding: 15px; border-left: 4px solid #16A34A; border-radius: 4px; margin: 20px 0;">
                        <p style="margin: 0; font-size: 1.05rem;">{resumen}</p>
                      </div>
                      
                      <p style="margin-bottom: 5px;"><strong>Recomendación / Acción Sugerida:</strong></p>
                      <p style="font-size: 1.2rem; font-weight: bold; color: #0F172A; margin-top: 0;">{orden}</p>
                      
                      <hr style="border: none; border-top: 1px solid #E2E8F0; margin: 20px 0;">
                      <p style="font-size: 0.8rem; color: #64748B; text-align: center;">Este es un mensaje automático de la Plataforma Analytics de Commodities Agrícolas.</p>
                    </div>
                  </body>
                </html>
                """

                msg.attach(MIMEText(html_content, "html"))

                try:
                    server.sendmail(smtp_user, destinatario, msg.as_string())
                    print(f"[CORREO ENVIADO] Alerta agrícola enviada a: {destinatario} ({nombre})")
                    enviados_count += 1
                except Exception as send_err:
                    print(f"[ERROR] No se pudo enviar el correo a {destinatario}: {send_err}")
            else:
                print(f"[OMITIDO] El cliente {reg.get('nombre')} no requiere envío de correo.")

        server.quit()
        print(f"\nProceso finalizado. Total de correos agrícolas enviados: {enviados_count}")

    except Exception as e:
        print(f"Error crítico en la conexión SMTP: {e}")

if __name__ == "__main__":
    enviar_alertas_pendientes()


# import os
# import smtplib
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart
# from src.db import get_db

# def enviar_alertas_pendientes():
#     db = get_db()
    
#     ultimo_log = db.notification_logs.find_one(sort=[("fecha_ejecucion", -1)])
    
#     if not ultimo_log or "resultados" not in ultimo_log:
#         print("No se encontraron registros de ejecución recientes en 'notification_logs'.")
#         return

#     resultados = ultimo_log["resultados"]

#     smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
#     port_env = os.getenv("SMTP_PORT")
#     smtp_port = int(port_env) if port_env and port_env.strip() else 587
#     smtp_user = os.getenv("SMTP_USER")
#     smtp_password = os.getenv("SMTP_PASSWORD")

#     if not smtp_user or not smtp_password:
#         print("Error: Las credenciales SMTP (SMTP_USER y SMTP_PASSWORD) no están configuradas en las variables de entorno.")
#         return

#     print("Conectando al servidor SMTP...")
#     try:
#         server = smtplib.SMTP(smtp_server, smtp_port)
#         server.starttls()
#         server.login(smtp_user, smtp_password)
#         print("¡Conexión SMTP establecida con éxito!")

#         enviados_count = 0

#         for reg in resultados:
#             if reg.get("enviar_correo") is True:
#                 destinatario = reg.get("email")
#                 nombre = reg.get("nombre", "Estimado cliente")
#                 resumen = reg.get("resumen_personalizado")
#                 orden = reg.get("orden_inversion")
#                 watchlist = ", ".join(reg.get("watchlist", []))

#                 msg = MIMEMultipart("alternative")
#                 msg["Subject"] = f"🔔 Alerta Personalizada de Criptomonedas: {orden}"
#                 msg["From"] = smtp_user
#                 msg["To"] = destinatario

#                 html_content = f"""
#                 <html>
#                   <body style="font-family: Arial, sans-serif; color: #333333; line-height: 1.6;">
#                     <div style="max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #E2E8F0; border-radius: 8px; background-color: #F8FAFC;">
#                       <h2 style="color: #1E3A8A; margin-top: 0;">Hola, {nombre}</h2>
#                       <p>Has recibido una actualización automática basada en tu perfil de inversión y seguimiento de activos (<strong>{watchlist}</strong>).</p>
                      
#                       <div style="background-color: #FFFFFF; padding: 15px; border-left: 4px solid #EAB308; border-radius: 4px; margin: 20px 0;">
#                         <p style="margin: 0; font-size: 1.05rem;">{resumen}</p>
#                       </div>
                      
#                       <p style="margin-bottom: 5px;"><strong>Acción Sugerida / Orden:</strong></p>
#                       <p style="font-size: 1.2rem; font-weight: bold; color: #0F172A; margin-top: 0;">{orden}</p>
                      
#                       <hr style="border: none; border-top: 1px solid #E2E8F0; margin: 20px 0;">
#                       <p style="font-size: 0.8rem; color: #64748B; text-align: center;">Este es un mensaje automatizado de tu Asistente Algorítmico de Criptomonedas. Por favor no respondas a este correo.</p>
#                     </div>
#                   </body>
#                 </html>
#                 """

#                 msg.attach(MIMEText(html_content, "html"))

#                 try:
#                     server.sendmail(smtp_user, destinatario, msg.as_string())
#                     print(f"[CORREO ENVIADO] Alerta enviada exitosamente a: {destinatario} ({nombre})")
#                     enviados_count += 1
#                 except Exception as send_err:
#                     print(f"[ERROR] No se pudo enviar el correo a {destinatario}: {send_err}")
#             else:
#                 print(f"[OMITIDO] El cliente {reg.get('nombre')} no requiere envío de correo (enviar_correo: false).")

#         server.quit()
#         print(f"\nProceso finalizado. Total de correos enviados en esta ejecución: {enviados_count}")

#     except Exception as e:
#         print(f"Error crítico en la conexión o autenticación SMTP: {e}")

# if __name__ == "__main__":
#     enviar_alertas_pendientes()