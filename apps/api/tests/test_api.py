from fastapi.testclient import TestClient

from app.main import app
from app.model.preprocessing import ALPHABET_FACE_INDICES, WORD_FACE_INDICES


client = TestClient(app)


def test_health() -> None:
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info_reports_hybrid_configuration() -> None:
    response = client.get("/api/v1/model/info")

    assert response.status_code == 200
    body = response.json()
    assert body["word"]["labels"]
    assert body["alphabet"]["labels"] == list("ABCDEFGHIKLMNOPQRSTUVWXY")
    assert body["sequence_length"] == 60
    assert body["window_size"] == 25
    assert body["feature_length"] == 918
    assert body["temporal_delta_order"] == 2
    assert body["word_available"] in {True, False}
    assert body["alphabet_available"] in {True, False}
    assert body["settle_seconds"] == 3.0
    assert body["inference_interval_seconds"] == 0.75
    assert body["hybrid_motion_threshold"] == 0.010
    assert body["hold_last_reading_seconds"] == 2.0


def test_preprocessing_uses_distinct_face_index_orders() -> None:
    assert len(WORD_FACE_INDICES) == 16
    assert len(ALPHABET_FACE_INDICES) == 16
    assert WORD_FACE_INDICES != ALPHABET_FACE_INDICES


def test_translate_reset_message() -> None:
    with client.websocket_connect("/api/v1/translate/stream") as websocket:
        websocket.send_json({"type": "reset"})
        message = websocket.receive_json()

    assert message["type"] == "reset"
    assert message["sentence"] == []


def test_translate_rejects_invalid_mode() -> None:
    with client.websocket_connect("/api/v1/translate/stream") as websocket:
        websocket.send_json({"type": "frame", "frame": "data:image/jpeg;base64,abc", "mode": "letters"})
        message = websocket.receive_json()

    assert message["type"] == "error"
    assert message["status"] == "modo inválido"
