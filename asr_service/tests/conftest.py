import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add the service path to sys.path
service_path = str(Path(__file__).resolve().parents[1])
if service_path not in sys.path:
    sys.path.append(service_path)

from main import app

@pytest.fixture
def client():
    return TestClient(app)
