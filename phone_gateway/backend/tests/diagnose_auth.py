import sys
from pathlib import Path
from types import SimpleNamespace
from pydantic import EmailStr

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[3]))

# Mocking modules to avoid side effects
import sys
from unittest.mock import MagicMock
sys.modules['backend.database'] = MagicMock()
sys.modules['backend.bootstrap'] = MagicMock()
sys.modules['starlette.staticfiles'] = MagicMock()

from phone_gateway.backend.auth.schemas import LoginResponse, UserResponse

mock_user = SimpleNamespace(
    user_id="u1",
    email="admin@example.com",
    full_name="Admin User",
    password_hash="hashed_p",
    role="admin",
    is_active=True
)

response_data = {
    "access_token": "fake-token",
    "token_type": "bearer",
    "user": mock_user
}

try:
    obj = LoginResponse(**response_data)
    print("Successfully created LoginResponse!")
    print(obj.model_dump_json(indent=2))
except Exception as e:
    print(f"Validation failed: {e}")
