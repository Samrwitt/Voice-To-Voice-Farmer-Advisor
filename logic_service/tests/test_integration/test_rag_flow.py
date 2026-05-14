from unittest.mock import patch

@patch("main.log_conversation")
@patch("main.get_session_state")
@patch("main.get_farmer_profile")
@patch("main.get_alerts_for_region")
@patch("main.collection.query")
def test_rag_flow(mock_query, mock_alerts, mock_profile, mock_state, mock_log, client):
    # mock farmer profile
    mock_profile.return_value = {
        "location": "Addis Ababa"
    }

    # mock alerts
    mock_alerts.return_value = [("Flood warning", "high")]

    # mock session state
    mock_state.return_value = None

    # mock vector DB response
    mock_query.return_value = {
        "documents": [["Use organic fertilizer for teff"]],
        "distances": [[0.5]],
        "metadatas": [[{"intent": "agriculture"}]],
    }

    response = client.post("/ask", json={
        "text": "teff farming advice",
        "phone_number": "123",
        "session_id": "test"
    })

    assert response.status_code == 200
    data = response.json()

    assert "response" in data
    assert "intent" in data