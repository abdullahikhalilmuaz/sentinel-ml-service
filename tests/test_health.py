from app import app


def test_health_endpoint():
    client = app.test_client()
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert payload["service"] == "sentinel-ml-service"


def test_model_status_is_truthful():
    client = app.test_client()
    response = client.get("/internal/model/status")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] in {"loaded", "not_loaded"}
    assert payload["modelAvailable"] is (payload["status"] == "loaded")
