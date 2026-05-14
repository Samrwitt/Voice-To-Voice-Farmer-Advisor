import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import os

# Add phone_gateway to path so 'backend' package is importable
sys.path.append(str(Path(__file__).resolve().parents[2]))

# Create dummy directory for StaticFiles if it doesn't exist
os.makedirs("utterances_dummy", exist_ok=True)

# Mock sqlalchemy and other startup side-effects
with patch("sqlalchemy.create_engine"), \
     patch("backend.database.engine"), \
     patch("backend.database.Base.metadata.create_all"), \
     patch("backend.database.SessionLocal", return_value=MagicMock()), \
     patch("backend.bootstrap.seed_default_admin"), \
     patch("os.getenv", side_effect=lambda k, d=None: "utterances_dummy" if k == "UTTERANCES_DIR" else d), \
     patch("fastapi.staticfiles.StaticFiles", return_value=MagicMock()):
    from backend.main import app

@pytest.fixture
def client():
    return TestClient(app)
