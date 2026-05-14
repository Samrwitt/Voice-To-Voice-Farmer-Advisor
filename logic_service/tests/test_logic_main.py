import sys
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

# IMPORTANT: go to logic_service root
sys.path.append(str(Path(__file__).resolve().parents[1]))

from main import app

client = TestClient(app)


# -----------------------------------
# /repeat endpoint
# -----------------------------------
@patch("main.get_conversation_history")
def test_repeat_last_response(mock_history):
    mock_history.return_value = [
        ("user", "hello"),
        ("assistant", "sample response"),
    ]

    response = client.get("/repeat/test_session")

    assert response.status_code == 200
    assert response.json()["response"] == "sample response"


# -----------------------------------
# /profile endpoint
# -----------------------------------
@patch("main.get_farmer_profile")
def test_get_profile(mock_profile):
    mock_profile.return_value = {
        "phone_number": "0911111111",
        "name": "Hanna",
        "location": "Addis Ababa",
        "preferred_language": "am",
    }

    response = client.get("/profile/0911111111")

    assert response.status_code == 200
    assert response.json()["name"] == "Hanna"


# -----------------------------------
# /system_check endpoint
# -----------------------------------
@patch("main.collection")
def test_system_check(mock_collection):
    mock_collection.count.return_value = 1

    response = client.get("/system_check")

    assert response.status_code == 200
    assert "chroma_db" in response.json()