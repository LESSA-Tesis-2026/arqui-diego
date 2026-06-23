from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "LESSA Translation API"
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = ["http://localhost:3000"]

    word_model_path: Path = Path("../../models/modelo_señas_lstm.keras")
    alphabet_model_path: Path = Path("../../models/modelo_letras.h5")
    sequence_length: int = 60
    window_size: int = 25
    base_feature_length: int = 306
    use_temporal_features: bool = True
    temporal_delta_order: int = 2
    word_confidence_threshold: float = 0.65
    alphabet_confidence_threshold: float = 0.80
    hybrid_motion_threshold: float = 0.010
    settle_seconds: float = 3.0
    inference_interval_seconds: float = 0.75
    hold_last_reading_seconds: float = 2.0
    voting_buffer_size: int = 10
    min_votes: int = 7
    alphabet_voting_buffer_size: int = 5
    alphabet_min_votes: int = 3
    max_sentence_words: int = 5
    pause_reset_frames: int = 15

    model_config = SettingsConfigDict(env_prefix="LESSA_", env_file=".env", extra="ignore")

    @property
    def feature_length(self) -> int:
        if not self.use_temporal_features:
            return self.base_feature_length
        return self.base_feature_length * (1 + self.temporal_delta_order)

    @property
    def resolved_word_model_path(self) -> Path:
        return self._resolve_app_path(self.word_model_path)

    @property
    def resolved_alphabet_model_path(self) -> Path:
        return self._resolve_app_path(self.alphabet_model_path)

    def _resolve_app_path(self, path: Path) -> Path:
        if path.is_absolute():
            return path
        return Path(__file__).resolve().parents[2] / path


@lru_cache
def get_settings() -> Settings:
    return Settings()
