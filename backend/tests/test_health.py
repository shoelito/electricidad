def test_health_reports_ok_without_model(client):
    response = client.get("/api/salud")

    assert response.status_code == 200
    assert response.json() == {"estado": "ok", "modelo_cargado": False}


def test_responses_include_security_headers(client):
    response = client.get("/api/salud")

    assert "default-src 'self'" in response.headers["Content-Security-Policy"]
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
