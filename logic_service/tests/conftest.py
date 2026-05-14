import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add logic_service to path so its modules are importable directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from main import app

@pytest.fixture
def client():
    return TestClient(app)