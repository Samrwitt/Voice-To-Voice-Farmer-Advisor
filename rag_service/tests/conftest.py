import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add rag_service to path
service_path = str(Path(__file__).resolve().parents[1])
if service_path not in sys.path:
    sys.path.append(service_path)

# Mock heavy/database modules before importing app
mock_db = MagicMock()
sys.modules["database"] = mock_db
mock_rag_pg = MagicMock()
sys.modules["rag_pg"] = mock_rag_pg
mock_dynamic = MagicMock()
sys.modules["dynamic_layer_runtime"] = mock_dynamic
sys.modules["psycopg"] = MagicMock()

from main import app

@pytest.fixture
def client():
    return TestClient(app)
