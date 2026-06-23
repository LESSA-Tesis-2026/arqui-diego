from __future__ import annotations

import collections
import math
import time

import numpy as np

from app.core.config import Settings, get_settings
from app.lessa.labels import ALPHABET_LABELS, WORD_LABELS
from app.lessa.preprocessing import (
    build_sequence_features,
    decode_frame,
    extract_hybrid_keypoints,
    mediapipe_detection,
    motion_score as calculate_motion_score,
    pad_sequence,
)
from app.lessa.runtimes import AlphabetModelRuntime, WordModelRuntime
from app.lessa.schemas import (
    InferenceMode,
    ModelInfoResponse,
    ResolvedMode,
    TopPrediction,
    TranslationResponse,
)
from app.lessa.session import HybridTranslationSession


class HybridLessaModelService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.word_runtime = WordModelRuntime(settings)
        self.alphabet_runtime = AlphabetModelRuntime(settings)

    def model_info(self) -> ModelInfoResponse:
        word = self.word_runtime.info()
        alphabet = self.alphabet_runtime.info()
        return ModelInfoResponse(
            available=word.available or alphabet.available,
            word_available=word.available,
            alphabet_available=alphabet.available,
            word=word,
            alphabet=alphabet,
            sequence_length=self.settings.sequence_length,
            window_size=self.settings.window_size,
            base_feature_length=self.settings.base_feature_length,
            feature_length=self.settings.feature_length,
            use_temporal_features=self.settings.use_temporal_features,
            temporal_delta_order=self.settings.temporal_delta_order,
            word_confidence_threshold=self.settings.word_confidence_threshold,
            alphabet_confidence_threshold=self.settings.alphabet_confidence_threshold,
            hybrid_motion_threshold=self.settings.hybrid_motion_threshold,
            settle_seconds=self.settings.settle_seconds,
            inference_interval_seconds=self.settings.inference_interval_seconds,
            hold_last_reading_seconds=self.settings.hold_last_reading_seconds,
        )

    def create_session(self) -> HybridTranslationSession:
        return HybridTranslationSession(settings=self.settings)

    def predict_frame(
        self,
        frame_data: str,
        session: HybridTranslationSession,
        holistic,
        requested_mode: InferenceMode = "auto",
    ) -> TranslationResponse:
        now = time.monotonic()
        frame = decode_frame(frame_data)
        results = mediapipe_detection(frame, holistic)
        extraction = extract_hybrid_keypoints(results, self.settings.base_feature_length)

        word_keypoints = extraction.word_keypoints
        alphabet_keypoints = extraction.alphabet_keypoints
        if not extraction.has_hands:
            word_keypoints = np.zeros(self.settings.base_feature_length, dtype=np.float32)
            alphabet_keypoints = np.zeros(self.settings.base_feature_length, dtype=np.float32)

        session.word_sequence.append(word_keypoints)
        score = calculate_motion_score(list(session.word_sequence))

        if not extraction.has_hands:
            return self._handle_no_hands(session, requested_mode, now, score)

        session.nada_counter = 0
        session.last_hands_seen_at = now
        if session.hands_detected_since is None:
            session.hands_detected_since = now

        resolved_mode = self._resolve_mode(requested_mode, session, score)
        if resolved_mode != "none":
            session.last_resolved_mode = resolved_mode

        if resolved_mode != session.mode_candidate:
            if session.mode_candidate == "none" or resolved_mode == "none":
                session.mode_candidate_since = session.hands_detected_since
            else:
                session.mode_candidate_since = now
            session.mode_candidate = resolved_mode

        # The settling window trades latency for legibility: it gives the signer time
        # to hold a shape/motion before expensive inference and UI emission begin.
        settle_started_at = session.mode_candidate_since or session.hands_detected_since or now
        elapsed_settle = now - settle_started_at
        if elapsed_settle < self.settings.settle_seconds:
            remaining = max(0.0, self.settings.settle_seconds - elapsed_settle)
            remaining = math.ceil(remaining * 2) / 2
            return self._empty_response(
                session=session,
                requested_mode=requested_mode,
                resolved_mode=resolved_mode,
                status=f"mantén la seña {remaining:.1f}s",
                has_hands=extraction.has_hands,
                motion_score=score,
            )

        if resolved_mode == "none":
            return self._empty_response(
                session=session,
                requested_mode=requested_mode,
                resolved_mode="none",
                status="preparando ventana de análisis",
                has_hands=extraction.has_hands,
                motion_score=score,
            )

        if (
            session.last_inference_at
            and now - session.last_inference_at < self.settings.inference_interval_seconds
        ):
            return self._empty_response(
                session=session,
                requested_mode=requested_mode,
                resolved_mode=resolved_mode,
                status="mantén la postura; analizando por ventanas",
                has_hands=extraction.has_hands,
                motion_score=score,
            )

        session.last_inference_at = now

        if resolved_mode == "words":
            session.dynamic_frames += 1
            session.static_frames = 0
            if session.dynamic_frames > 4:
                session.letter_buffer.clear()
                session.last_emitted_letter = "-"
            return self._predict_word(session, requested_mode, extraction.has_hands, score)

        if resolved_mode == "alphabet":
            session.static_frames += 1
            session.dynamic_frames = 0
            if session.static_frames > 4:
                session.word_buffer.clear()
            return self._predict_alphabet(
                alphabet_keypoints,
                session,
                requested_mode,
                extraction.has_hands,
                score,
            )

        return self._empty_response(
            session=session,
            requested_mode=requested_mode,
            resolved_mode="none",
            status="preparando traducción",
            has_hands=extraction.has_hands,
            motion_score=score,
        )

    def _handle_no_hands(
        self,
        session: HybridTranslationSession,
        requested_mode: InferenceMode,
        now: float,
        score: float,
    ) -> TranslationResponse:
        session.nada_counter += 1
        hold_elapsed = (
            now - session.last_hands_seen_at
            if session.last_hands_seen_at is not None
            else self.settings.hold_last_reading_seconds + 1
        )
        if hold_elapsed <= self.settings.hold_last_reading_seconds and session.last_resolved_mode != "none":
            return self._empty_response(
                session=session,
                requested_mode=requested_mode,
                resolved_mode=session.last_resolved_mode,
                status="manteniendo lectura anterior",
                has_hands=False,
                motion_score=score,
            )

        session.word_buffer.clear()
        session.letter_buffer.clear()
        session.dynamic_frames = 0
        session.static_frames = 0
        session.hands_detected_since = None
        session.mode_candidate = "none"
        session.mode_candidate_since = None
        session.last_inference_at = 0.0
        session.last_resolved_mode = "none"
        if session.nada_counter > self.settings.pause_reset_frames:
            session.last_emitted_word = "nada"
            session.last_emitted_letter = "-"
        return self._empty_response(
            session=session,
            requested_mode=requested_mode,
            resolved_mode="none",
            status="sin señas detectadas",
            has_hands=False,
            motion_score=score,
        )

    def _resolve_mode(
        self,
        requested_mode: InferenceMode,
        session: HybridTranslationSession,
        score: float,
    ) -> ResolvedMode:
        if requested_mode == "words":
            return "words"
        if requested_mode == "alphabet":
            return "alphabet"

        if len(session.word_sequence) < self.settings.window_size:
            return "none"

        # Auto mode uses recent hand motion to choose dynamic word recognition or
        # static alphabet recognition, then falls back if the preferred model is absent.
        preferred: ResolvedMode = (
            "words" if score > self.settings.hybrid_motion_threshold else "alphabet"
        )
        fallback: ResolvedMode = "alphabet" if preferred == "words" else "words"

        if preferred == "words" and self.word_runtime.available:
            return preferred
        if preferred == "alphabet" and self.alphabet_runtime.available:
            return preferred
        if fallback == "words" and self.word_runtime.available:
            return fallback
        if fallback == "alphabet" and self.alphabet_runtime.available:
            return fallback
        return preferred

    def _predict_word(
        self,
        session: HybridTranslationSession,
        requested_mode: InferenceMode,
        has_hands: bool,
        score: float,
    ) -> TranslationResponse:
        model = self.word_runtime.model
        if model is None:
            return self._empty_response(
                session=session,
                requested_mode=requested_mode,
                resolved_mode="words",
                status="modelo de palabras no disponible",
                has_hands=has_hands,
                motion_score=score,
                error=self.word_runtime.error,
            )

        if len(session.word_sequence) < self.settings.window_size:
            return self._empty_response(
                session=session,
                requested_mode=requested_mode,
                resolved_mode="words",
                status="preparando palabras",
                has_hands=has_hands,
                motion_score=score,
            )

        sequence_features = build_sequence_features(
            list(session.word_sequence),
            self.settings.use_temporal_features,
            self.settings.temporal_delta_order,
        )
        model_input = pad_sequence(
            sequence_features,
            self.settings.sequence_length,
            self.settings.feature_length,
        )
        probabilities = model.predict(model_input, verbose=0)[0]
        best_index = int(np.argmax(probabilities))
        confidence = float(probabilities[best_index])
        prediction = WORD_LABELS[best_index]
        current_word = prediction if confidence > self.settings.word_confidence_threshold else "nada"

        session.word_buffer.append(current_word)
        stable_word = self._stable_vote(session.word_buffer, self.settings.min_votes, "nada")

        emitted_token = None
        if stable_word != "nada":
            session.nada_counter = 0
            if stable_word != session.last_emitted_word:
                emitted_token = stable_word
                session.append_word(stable_word)
                session.last_emitted_word = stable_word
        else:
            session.nada_counter += 1
            if session.nada_counter > self.settings.pause_reset_frames:
                session.last_emitted_word = "nada"

        top = self._top_predictions(probabilities, WORD_LABELS)
        return TranslationResponse(
            requested_mode=requested_mode,
            mode="words",
            prediction_type="word",
            word_available=self.word_runtime.available,
            alphabet_available=self.alphabet_runtime.available,
            prediction=prediction,
            confidence=confidence,
            stable_word=None if stable_word == "nada" else stable_word,
            emitted_word=emitted_token,
            emitted_token=emitted_token,
            sentence=session.sentence,
            text=session.display_text,
            top=top,
            word_top=top,
            alphabet_top=[],
            has_hands=has_hands,
            motion_score=score,
            status="modo palabras",
        )

    def _predict_alphabet(
        self,
        keypoints: np.ndarray,
        session: HybridTranslationSession,
        requested_mode: InferenceMode,
        has_hands: bool,
        score: float,
    ) -> TranslationResponse:
        model = self.alphabet_runtime.model
        if model is None:
            return self._empty_response(
                session=session,
                requested_mode=requested_mode,
                resolved_mode="alphabet",
                status="modelo de alfabeto no disponible",
                has_hands=has_hands,
                motion_score=score,
                error=self.alphabet_runtime.error,
            )

        probabilities = model.predict(np.expand_dims(keypoints, axis=0), verbose=0)[0]
        best_index = int(np.argmax(probabilities))
        confidence = float(probabilities[best_index])
        prediction = ALPHABET_LABELS[best_index]
        current_letter = prediction if confidence > self.settings.alphabet_confidence_threshold else "-"

        session.letter_buffer.append(current_letter)
        stable_letter = self._stable_vote(
            session.letter_buffer,
            self.settings.alphabet_min_votes,
            "-",
        )

        emitted_token = None
        if stable_letter != "-" and stable_letter != session.last_emitted_letter:
            emitted_token = stable_letter
            session.append_letter(stable_letter)
            session.last_emitted_letter = stable_letter
        elif stable_letter == "-":
            session.last_emitted_letter = "-"

        top = self._top_predictions(probabilities, ALPHABET_LABELS)
        return TranslationResponse(
            requested_mode=requested_mode,
            mode="alphabet",
            prediction_type="letter",
            word_available=self.word_runtime.available,
            alphabet_available=self.alphabet_runtime.available,
            prediction=prediction,
            confidence=confidence,
            stable_word=None if stable_letter == "-" else stable_letter,
            emitted_word=emitted_token,
            emitted_token=emitted_token,
            sentence=session.sentence,
            text=session.display_text,
            top=top,
            word_top=[],
            alphabet_top=top,
            has_hands=has_hands,
            motion_score=score,
            status="modo alfabeto",
        )

    def _empty_response(
        self,
        session: HybridTranslationSession,
        requested_mode: InferenceMode,
        resolved_mode: ResolvedMode,
        status: str,
        has_hands: bool,
        motion_score: float,
        error: str | None = None,
    ) -> TranslationResponse:
        return TranslationResponse(
            requested_mode=requested_mode,
            mode=resolved_mode,
            prediction_type="none",
            word_available=self.word_runtime.available,
            alphabet_available=self.alphabet_runtime.available,
            prediction=None,
            confidence=0.0,
            stable_word=None,
            emitted_word=None,
            emitted_token=None,
            sentence=session.sentence,
            text=session.display_text,
            top=[],
            word_top=[],
            alphabet_top=[],
            has_hands=has_hands,
            motion_score=motion_score,
            status=status,
            error=error,
        )

    @staticmethod
    def _stable_vote(buffer: collections.deque[str], minimum_votes: int, fallback: str) -> str:
        if not buffer:
            return fallback
        most_common, count = collections.Counter(buffer).most_common(1)[0]
        return most_common if count >= minimum_votes else fallback

    @staticmethod
    def _top_predictions(
        probabilities: np.ndarray,
        labels: list[str],
        limit: int = 3,
    ) -> list[TopPrediction]:
        indexes = np.argsort(probabilities)[::-1][:limit]
        return [
            TopPrediction(label=labels[int(index)], confidence=float(probabilities[int(index)]))
            for index in indexes
        ]


model_service = HybridLessaModelService(settings=get_settings())
