from datetime import datetime, timedelta
import base64
import hashlib
import hmac
import os

from jose import JWTError, jwt


SECRET_KEY = os.getenv(
    "JWT_SECRET_KEY",
    "change-this-secret-key-in-production"
)

ALGORITHM = "HS256"

ACCESS_TOKEN_EXPIRE_MINUTES = int(
    os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440")
)

PBKDF2_ITERATIONS = int(
    os.getenv("PBKDF2_ITERATIONS", "600000")
)

PBKDF2_ALGORITHM = "sha256"
SALT_BYTES = 16


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("utf-8")


def _b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value.encode("utf-8"))


def hash_password(password: str) -> str:
    """
    Hash password using PBKDF2-HMAC-SHA256.

    Stored format:
    pbkdf2_sha256$iterations$salt_b64$hash_b64
    """

    if not password:
        raise ValueError("Password cannot be empty")

    salt = os.urandom(SALT_BYTES)

    password_hash = hashlib.pbkdf2_hmac(
        PBKDF2_ALGORITHM,
        password.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
    )

    return (
        f"pbkdf2_sha256$"
        f"{PBKDF2_ITERATIONS}$"
        f"{_b64encode(salt)}$"
        f"{_b64encode(password_hash)}"
    )


def verify_password(plain_password: str, stored_hash: str) -> bool:
    """
    Verify PBKDF2 password hash.
    """

    if not plain_password or not stored_hash:
        return False

    try:
        scheme, iterations_str, salt_b64, hash_b64 = stored_hash.split("$", 3)

        if scheme != "pbkdf2_sha256":
            return False

        iterations = int(iterations_str)
        salt = _b64decode(salt_b64)
        expected_hash = _b64decode(hash_b64)

        actual_hash = hashlib.pbkdf2_hmac(
            PBKDF2_ALGORITHM,
            plain_password.encode("utf-8"),
            salt,
            iterations,
        )

        return hmac.compare_digest(actual_hash, expected_hash)

    except Exception:
        return False


def create_access_token(data: dict) -> str:
    to_encode = data.copy()

    expire = datetime.utcnow() + timedelta(
        minutes=ACCESS_TOKEN_EXPIRE_MINUTES
    )

    to_encode.update({
        "exp": expire
    })

    return jwt.encode(
        to_encode,
        SECRET_KEY,
        algorithm=ALGORITHM,
    )


def decode_access_token(token: str) -> dict | None:
    try:
        return jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM],
        )

    except JWTError:
        return None