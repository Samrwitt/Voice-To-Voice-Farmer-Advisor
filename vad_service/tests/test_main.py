import pytest
from fastapi.testclient import TestClient

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["service"] == "silero-vad-service"

def test_websocket_connection(client):
    with client.websocket_connect("/ws/vad?session_id=test_ws&sample_rate=16000") as websocket:
        # Should receive vad_ready
        data = websocket.receive_json()
        assert data["event"] == "vad_ready"
        assert data["session_id"] == "test_ws"
        
        # Send binary audio (silence)
        websocket.send_bytes(b"\x00" * 1024)
        
        # Since it's silence, no further events should be sent immediately
        # (We can't easily wait for non-events, but we can verify it doesn't crash)
        
        # Send END message
        websocket.send_json({"event": "end_session"})
        # Should disconnect or finish
