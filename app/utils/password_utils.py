import secrets
import string


def generate_temporary_password(length: int = 10) -> str:
    alphabet = string.ascii_letters + string.digits

    while True:
        password = "".join(secrets.choice(alphabet) for _ in range(length))

        has_lower = any(c.islower() for c in password)
        has_upper = any(c.isupper() for c in password)
        has_digit = any(c.isdigit() for c in password)

        if has_lower and has_upper and has_digit:
            return password