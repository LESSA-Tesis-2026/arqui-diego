from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class ModelInfoResponse(BaseModel):
    available: bool
    error: str | None
    labels: list[str]
    sequence_length: int
    window_size: int
    base_feature_length: int
    feature_length: int
    use_temporal_features: bool
    confidence_threshold: float
    input_shape: list[int | None] | None
    output_shape: list[int | None] | None


class TopPrediction(BaseModel):
    label: str
    confidence: float


class TranslationResponse(BaseModel):
    type: str = "translation"
    model_available: bool
    prediction: str
    confidence: float
    stable_word: str | None
    emitted_word: str | None
    sentence: list[str]
    text: str
    top: list[TopPrediction]
    has_hands: bool
    status: str
    error: str | None = None
