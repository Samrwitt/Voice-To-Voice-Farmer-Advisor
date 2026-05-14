import pytest
from unittest.mock import patch, MagicMock

def test_login_invalid_credentials(client):
    with patch("backend.auth.routes.SessionLocal") as mock_session:
        # Mock user not found
        mock_db = mock_session.return_value
        mock_db.query.return_value.filter.return_value.first.return_value = None
        
        response = client.post("/api/auth/login", json={
            "email": "wrong@example.com",
            "password": "wrongpassword"
        })
        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid email or password"

@patch("backend.auth.routes.verify_password")
@patch("backend.auth.routes.create_access_token")
def test_login_success(mock_token, mock_verify, client):
    from types import SimpleNamespace
    with patch("backend.auth.routes.SessionLocal") as mock_session:
        mock_db = mock_session.return_value
        mock_user = SimpleNamespace(
            user_id="u1",
            email="admin@example.com",
            full_name="Admin User",
            password_hash="hashed_p",
            role="admin",
            is_active=True
        )
        
        # Correctly chain the mocks
        mock_db.query.return_value.filter.return_value.first.return_value = mock_user
        mock_verify.return_value = True
        mock_token.return_value = "fake-jwt-token"
        
        response = client.post("/api/auth/login", json={
            "email": "admin@example.com",
            "password": "admin123"
        })
        
        assert response.status_code == 200
        assert response.json()["access_token"] == "fake-jwt-token"

def test_get_me_unauthorized(client):
    response = client.get("/api/auth/me")
    assert response.status_code == 401
