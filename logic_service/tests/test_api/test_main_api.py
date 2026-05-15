from unittest.mock import patch

def test_repeat_last_response(client):
    with patch("main.get_conversation_history") as mock_history:
        mock_history.return_value = [("user", "hi"), ("assistant", "hello")]
        response = client.get("/repeat/test_session")
        assert response.status_code == 200
        assert response.json()["response"] == "hello"


def test_get_profile_not_found(client):
    with patch("main.get_farmer_profile") as mock_profile:
        mock_profile.return_value = None
        response = client.get("/profile/000000")
        assert response.status_code == 404


def test_system_check(client):
    response = client.get("/system_check")
    assert response.status_code == 200
    data = response.json()

    assert "database" in data
    assert "chroma_db" in data
    assert "llm_loaded" in data
