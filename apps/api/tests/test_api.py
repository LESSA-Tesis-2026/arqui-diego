import numpy as np
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.lessa.inference import HybridLessaModelService
from app.lessa.preprocessing import (
    ALPHABET_FACE_INDICES,
    WORD_FACE_INDICES,
    build_sequence_features,
    pad_sequence,
)
from app.main import app


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


def test_temporal_features_match_current_model_shape() -> None:
    base_length = get_settings().base_feature_length
    sequence = [np.full(base_length, value, dtype=np.float32) for value in range(3)]

    features = build_sequence_features(sequence, use_temporal_features=True, temporal_delta_order=2)
    padded = pad_sequence(features, sequence_length=60, feature_length=base_length * 3)

    assert padded.shape == (1, 60, 918)
    assert np.allclose(padded[0, 0, base_length : base_length * 2], 0)
    assert np.allclose(padded[0, 1, base_length : base_length * 2], 1)


def test_stable_vote_requires_minimum_votes() -> None:
    service = HybridLessaModelService(settings=get_settings())
    buffer = service.create_session().word_buffer
    buffer.extend(["hola", "hola", "nada"])

    assert service._stable_vote(buffer, minimum_votes=3, fallback="nada") == "nada"
    assert service._stable_vote(buffer, minimum_votes=2, fallback="nada") == "hola"


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
