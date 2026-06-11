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

    def _send_email(
        self,
        to_email: str,
        subject: str,
        body: str,
    ) -> None:
        if not self.is_configured():
            raise RuntimeError("SMTP is not configured.")

        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = f"{SMTP_FROM_NAME} <{SMTP_FROM_EMAIL}>"
        message["To"] = to_email
        message.set_content(body)

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(message)

    def send_trainer_credentials(
        self,
        to_email: str,
        full_name: str,
        password: str,
    ) -> None:
        subject = "Votre compte formateur Evalea"

        body = f"""
Bonjour {full_name},

Votre compte formateur a été créé sur la plateforme Evalea.

Voici vos informations de connexion :

Email : {to_email}
Mot de passe temporaire : {password}

Lien de connexion :
{FRONTEND_LOGIN_URL}

Pour des raisons de sécurité, veuillez changer votre mot de passe après votre première connexion.

Cordialement,
{SMTP_FROM_NAME}
""".strip()

        self._send_email(
            to_email=to_email,
            subject=subject,
            body=body,
        )

    def send_password_reset(
        self,
        to_email: str,
        full_name: str,
        temporary_password: str,
    ) -> None:
        subject = "Réinitialisation de votre mot de passe Evalea"

        body = f"""
Bonjour {full_name},

Vous avez demandé la récupération de votre mot de passe sur la plateforme Evalea.

Un nouveau mot de passe temporaire a été généré pour votre compte.

Email : {to_email}
Nouveau mot de passe temporaire : {temporary_password}

Lien de connexion :
{FRONTEND_LOGIN_URL}

Pour des raisons de sécurité, veuillez vous connecter puis modifier votre mot de passe depuis votre profil.

Si vous n’êtes pas à l’origine de cette demande, veuillez contacter l’administrateur de la plateforme.

Cordialement,
{SMTP_FROM_NAME}
""".strip()

        self._send_email(
            to_email=to_email,
            subject=subject,
            body=body,
        )