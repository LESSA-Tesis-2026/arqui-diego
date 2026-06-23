from typing import Literal

from pydantic import BaseModel


InferenceMode = Literal["auto", "words", "alphabet"]
ResolvedMode = Literal["words", "alphabet", "none"]
PredictionType = Literal["word", "letter", "none"]


class HealthResponse(BaseModel):
    status: str


class RuntimeModelInfo(BaseModel):
    available: bool
    error: str | None
    labels: list[str]
    model_path: str
    input_shape: list[int | None] | None
    output_shape: list[int | None] | None


class ModelInfoResponse(BaseModel):
    available: bool
    word_available: bool
    alphabet_available: bool
    word: RuntimeModelInfo
    alphabet: RuntimeModelInfo
    sequence_length: int
    window_size: int
    base_feature_length: int
    feature_length: int
    use_temporal_features: bool
    temporal_delta_order: int
    word_confidence_threshold: float
    alphabet_confidence_threshold: float
    hybrid_motion_threshold: float
    settle_seconds: float
    inference_interval_seconds: float
    hold_last_reading_seconds: float


class TopPrediction(BaseModel):
    label: str
    confidence: float


class TranslationResponse(BaseModel):
    type: str = "translation"
    requested_mode: InferenceMode
    mode: ResolvedMode
    prediction_type: PredictionType
    word_available: bool
    alphabet_available: bool
    prediction: str | None
    confidence: float
    stable_word: str | None
    emitted_word: str | None
    emitted_token: str | None
    sentence: list[str]
    text: str
    top: list[TopPrediction]
    word_top: list[TopPrediction]
    alphabet_top: list[TopPrediction]
    has_hands: bool
    motion_score: float
    status: str
    error: str | None = None
