from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info_endpoint():
    client = TestClient(app)
    response = client.get("/model-info")
    assert response.status_code == 200
    assert response.json()["model_name"] == "FraudDetectionModel"
