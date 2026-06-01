"""
JWT auth for logic_service.

Uses PBKDF2-HMAC-SHA256 password hashing and python-jose JWTs.
JWT_SECRET_KEY is shared with phone_gateway so tokens are mutually verifiable.
"""
from datetime import datetime, timedelta
import base64
import hashlib
import hmac
import os
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from db import get_db
from models import DashboardUser


# ── Config ────────────────────────────────────────────────────────────────────
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-this-secret-key-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))

PBKDF2_ITERATIONS = int(os.getenv("PBKDF2_ITERATIONS", "600000"))
PBKDF2_ALGORITHM = "sha256"
SALT_BYTES = 16

bearer_scheme = HTTPBearer(auto_error=False)


# ── Password hashing ──────────────────────────────────────────────────────────
def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("utf-8")


def _b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value.encode("utf-8"))


def hash_password(password: str) -> str:
    """
    Hash password using PBKDF2-HMAC-SHA256.
    Stored format: pbkdf2_sha256$iterations$salt_b64$hash_b64
    """
    if not password:
        raise ValueError("Password cannot be empty")

    salt = os.urandom(SALT_BYTES)
    pw_hash = hashlib.pbkdf2_hmac(
        PBKDF2_ALGORITHM,
        password.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
    )
    return (
        f"pbkdf2_sha256$"
        f"{PBKDF2_ITERATIONS}$"
        f"{_b64encode(salt)}$"
        f"{_b64encode(pw_hash)}"
    )


def verify_password(plain_password: str, stored_hash: str) -> bool:
    if not plain_password or not stored_hash:
        return False

    # Modern PBKDF2 format
    if stored_hash.startswith("pbkdf2_sha256$"):
        try:
            scheme, iterations_str, salt_b64, hash_b64 = stored_hash.split("$", 3)
            iterations = int(iterations_str)
            salt = _b64decode(salt_b64)
            expected = _b64decode(hash_b64)
            actual = hashlib.pbkdf2_hmac(
                PBKDF2_ALGORITHM,
                plain_password.encode("utf-8"),
                salt,
                iterations,
            )
            return hmac.compare_digest(actual, expected)
        except Exception:
            return False

    # Legacy bcrypt fallback (rows imported from old SQLite admin_users)
    try:
        import bcrypt as _bcrypt

        return _bcrypt.checkpw(plain_password.encode("utf-8"), stored_hash.encode("utf-8"))
    except Exception:
        return False


# ── JWT helpers ───────────────────────────────────────────────────────────────
def create_access_token(data: dict, expires_minutes: Optional[int] = None) -> str:
    to_encode = dict(data)
    expire = datetime.utcnow() + timedelta(
        minutes=expires_minutes or ACCESS_TOKEN_EXPIRE_MINUTES,
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None


# ── FastAPI dependencies ──────────────────────────────────────────────────────
def get_current_user(
    token: Optional[str] = None,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> DashboardUser:
    actual_token = None
    if credentials and credentials.scheme.lower() == "bearer":
        actual_token = credentials.credentials
    elif token:
        actual_token = token

    if not actual_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    payload = decode_access_token(actual_token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
        )

    user = db.query(DashboardUser).filter(DashboardUser.user_id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled",
        )

    return user


def require_roles(*allowed_roles: str):
    """Dependency factory that asserts the current user's role is in allowed_roles."""

    def role_checker(user: DashboardUser = Depends(get_current_user)) -> DashboardUser:
        if user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to access this resource",
            )
        return user

    return role_checker
