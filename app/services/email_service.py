from __future__ import annotations

import smtplib
from email.message import EmailMessage

from app.config import (
    FRONTEND_LOGIN_URL,
    SMTP_FROM_EMAIL,
    SMTP_FROM_NAME,
    SMTP_HOST,
    SMTP_PASSWORD,
    SMTP_PORT,
    SMTP_USERNAME,
)


class EmailService:
    def is_configured(self) -> bool:
        return bool(
            SMTP_HOST
            and SMTP_PORT
            and SMTP_USERNAME
            and SMTP_PASSWORD
            and SMTP_FROM_EMAIL
        )

    def send_trainer_credentials(
        self,
        to_email: str,
        full_name: str,
        password: str,
    ) -> None:
        if not self.is_configured():
            raise RuntimeError("SMTP is not configured.")

        subject = "Votre compte formateur CourseAI"

        body = f"""
Bonjour {full_name},

Votre compte formateur a été créé sur la plateforme CourseAI.

Voici vos informations de connexion :

Email : {to_email}
Mot de passe temporaire : {password}

Lien de connexion :
{FRONTEND_LOGIN_URL}

Pour des raisons de sécurité, veuillez changer votre mot de passe après votre première connexion.

Cordialement,
{SMTP_FROM_NAME}
""".strip()

        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = f"{SMTP_FROM_NAME} <{SMTP_FROM_EMAIL}>"
        message["To"] = to_email
        message.set_content(body)

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(message)