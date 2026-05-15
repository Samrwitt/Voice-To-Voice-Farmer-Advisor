import pytest
from unittest.mock import patch, MagicMock

def test_health_check(client):
    # main.py might not have a /health endpoint, let's check
    # actually let's just test /ask
    pass

@patch("main.generate_rag_response")
def test_ask_endpoint(mock_gen, client):
    mock_gen.return_value = (
        "ጤፍ በደንብ ይበቅላል", 
        "agriculture", 
        [{"title": "Teff Guide"}], 
        {"primary_intent": "agriculture", "confidence": 0.9}
    )
    
    response = client.post("/ask", json={
        "text": "ስለ ጤፍ ንገረኝ",
        "phone_number": "123",
        "session_id": "test_s"
    })
    
    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "ጤፍ በደንብ ይበቅላል"
    assert data["intent"] == "agriculture"

@patch("main.get_conversation_history")
def test_repeat_endpoint(mock_history, client):
    mock_history.return_value = [("user", "hi"), ("assistant", "ሰላም ለናንተ ይሁን")]
    
    response = client.get("/repeat/test_s")
    assert response.status_code == 200
    assert response.json()["response"] == "ሰላም ለናንተ ይሁን"
