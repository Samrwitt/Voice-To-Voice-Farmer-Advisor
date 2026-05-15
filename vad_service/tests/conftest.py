import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add vad_service to path
service_path = str(Path(__file__).resolve().parents[1])
if service_path not in sys.path:
    sys.path.append(service_path)

# Mock torch and silero_vad BEFORE importing app/engine
mock_torch = MagicMock()
sys.modules["torch"] = mock_torch
mock_silero = MagicMock()
sys.modules["silero_vad"] = mock_silero

from main import app

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def mock_vad_model():
    """Mock the Silero VAD model's forward pass."""
    with patch("vad_engine.load_silero_vad") as mock_load:
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        yield mock_model
