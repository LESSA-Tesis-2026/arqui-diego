from __future__ import annotations

import collections
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model

from app.core.config import get_settings
from app.core.config import Settings
from app.model.labels import LABELS
from app.model.preprocessing import (
    build_frame_features,
    decode_frame,
    extract_keypoints,
    holistic_context,
    mediapipe_detection,
    pad_sequence,
)
from app.model.schemas import ModelInfoResponse, TopPrediction, TranslationResponse


@dataclass
class TranslationSession:
    settings: Settings
    sequence: collections.deque[np.ndarray] = field(init=False)
    predictions_buffer: collections.deque[str] = field(init=False)
    previous_position: np.ndarray | None = None
    sentence: list[str] = field(default_factory=list)
    last_emitted_word: str = "nada"
    nada_counter: int = 0

    def __post_init__(self) -> None:
        self.sequence = collections.deque(maxlen=self.settings.window_size)
        self.predictions_buffer = collections.deque(maxlen=self.settings.voting_buffer_size)

    def reset(self) -> None:
        self.sequence.clear()
        self.predictions_buffer.clear()
        self.previous_position = None
        self.sentence.clear()
        self.last_emitted_word = "nada"
        self.nada_counter = 0


class LessaModelService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None
        self._load_error: str | None = None

    @property
    def model_available(self) -> bool:
        self._ensure_model_loaded()
        return self._model is not None

    def model_info(self) -> ModelInfoResponse:
        self._ensure_model_loaded()
        input_shape = None
        output_shape = None
        if self._model is not None:
            input_shape = [dim if dim is None else int(dim) for dim in self._model.input_shape]
            output_shape = [dim if dim is None else int(dim) for dim in self._model.output_shape]

        return ModelInfoResponse(
            available=self._model is not None,
            error=self._load_error,
            labels=LABELS,
            sequence_length=self.settings.sequence_length,
            window_size=self.settings.window_size,
            base_feature_length=self.settings.base_feature_length,
            feature_length=self.settings.feature_length,
            use_temporal_features=self.settings.use_temporal_features,
            confidence_threshold=self.settings.confidence_threshold,
            input_shape=input_shape,
            output_shape=output_shape,
        )

    def create_session(self) -> TranslationSession:
        return TranslationSession(settings=self.settings)

    def predict_frame(
        self,
        frame_data: str,
        session: TranslationSession,
        holistic,
    ) -> TranslationResponse:
        self._ensure_model_loaded()
        if self._model is None:
            return self._empty_response(
                session=session,
                status="modelo no disponible",
                error=self._load_error,
            )

        frame = decode_frame(frame_data)
        results = mediapipe_detection(frame, holistic)
        extraction = extract_keypoints(results, self.settings.base_feature_length)
        position_keypoints = extraction.position_keypoints

        if not extraction.has_hands:
            position_keypoints = np.zeros(self.settings.base_feature_length, dtype=np.float32)

        frame_features, session.previous_position = build_frame_features(
            position_keypoints,
            session.previous_position,
            self.settings.use_temporal_features,
        )
        session.sequence.append(frame_features)

        if len(session.sequence) < self.settings.window_size:
            return self._empty_response(
                session=session,
                status="preparando traducción" if extraction.has_hands else "sin señas detectadas",
                has_hands=extraction.has_hands,
            )

        model_input = pad_sequence(
            list(session.sequence),
            self.settings.sequence_length,
            self.settings.feature_length,
        )
        probabilities = self._model.predict(model_input, verbose=0)[0]
        return self._stabilize_prediction(probabilities, session, extraction.has_hands)

    def _ensure_model_loaded(self) -> None:
        if self._model is not None or self._load_error is not None:
            return

        model_path = self.settings.resolved_model_path
        if not Path(model_path).exists():
            self._load_error = f"Model artifact not found at {model_path}"
            return

        try:
            self._model = load_model(model_path, compile=False)
        except Exception as exc:  # pragma: no cover - depends on local artifact compatibility
            self._load_error = str(exc)

    def _stabilize_prediction(
        self,
        probabilities: np.ndarray,
        session: TranslationSession,
        has_hands: bool,
    ) -> TranslationResponse:
        best_index = int(np.argmax(probabilities))
        confidence = float(probabilities[best_index])
        prediction = LABELS[best_index]
        current_word = prediction if confidence > self.settings.confidence_threshold else "nada"

        session.predictions_buffer.append(current_word)
        word_counts = collections.Counter(session.predictions_buffer)
        most_common_word, count = word_counts.most_common(1)[0]
        stable_word = most_common_word if count >= self.settings.min_votes else "nada"

        emitted_word = None
        if stable_word != "nada":
            session.nada_counter = 0
            if stable_word != session.last_emitted_word:
                emitted_word = stable_word
                session.sentence.append(stable_word)
                session.last_emitted_word = stable_word
                if len(session.sentence) > self.settings.max_sentence_words:
                    session.sentence = session.sentence[-self.settings.max_sentence_words :]
        else:
            session.nada_counter += 1
            if session.nada_counter > self.settings.pause_reset_frames:
                session.last_emitted_word = "nada"

        return TranslationResponse(
            model_available=True,
            prediction=prediction,
            confidence=confidence,
            stable_word=None if stable_word == "nada" else stable_word,
            emitted_word=emitted_word,
            sentence=session.sentence,
            text=" ".join(session.sentence),
            top=self._top_predictions(probabilities),
            has_hands=has_hands,
            status="traduciendo" if has_hands else "sin señas detectadas",
        )

    def _empty_response(
        self,
        session: TranslationSession,
        status: str,
        error: str | None = None,
        has_hands: bool = False,
    ) -> TranslationResponse:
        return TranslationResponse(
            model_available=self._model is not None,
            prediction="nada",
            confidence=0.0,
            stable_word=None,
            emitted_word=None,
            sentence=session.sentence,
            text=" ".join(session.sentence),
            top=[],
            has_hands=has_hands,
            status=status,
            error=error,
        )

    @staticmethod
    def _top_predictions(probabilities: np.ndarray, limit: int = 3) -> list[TopPrediction]:
        indexes = np.argsort(probabilities)[::-1][:limit]
        return [
            TopPrediction(label=LABELS[int(index)], confidence=float(probabilities[int(index)]))
            for index in indexes
        ]


model_service = LessaModelService(settings=get_settings())
