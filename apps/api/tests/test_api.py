from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health() -> None:
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info_reports_configuration() -> None:
    response = client.get("/api/v1/model/info")

    assert response.status_code == 200
    body = response.json()
    assert body["labels"]
    assert body["sequence_length"] == 60
    assert body["window_size"] == 40
    assert body["feature_length"] == 612


def test_translate_reset_message() -> None:
    with client.websocket_connect("/api/v1/translate/stream") as websocket:
        websocket.send_json({"type": "reset"})
        message = websocket.receive_json()

    assert message["type"] == "reset"
    assert message["sentence"] == []
