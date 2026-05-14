import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add vad_service to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Mock torch and silero_vad BEFORE importing app/engine
mock_torch = MagicMock()
mock_silero = MagicMock()

sys.modules["torch"] = mock_torch
sys.modules["silero_vad"] = mock_silero

# Setup default mocks for Silero model
mock_model = MagicMock()
mock_silero.load_silero_vad.return_value = mock_model
mock_model.return_value.item.return_value = 0.0 # Default silence

from main import app

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def mock_vad_model():
    return mock_model
