from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "LESSA Translation API"
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = ["http://localhost:3000"]

    model_path: Path = Path("artifacts/modelo_senas_lstm.keras")
    sequence_length: int = 60
    window_size: int = 40
    base_feature_length: int = 306
    use_temporal_features: bool = True
    confidence_threshold: float = 0.75
    voting_buffer_size: int = 15
    min_votes: int = 10
    max_sentence_words: int = 12
    pause_reset_frames: int = 15

    model_config = SettingsConfigDict(env_prefix="LESSA_", env_file=".env")

    @property
    def feature_length(self) -> int:
        return self.base_feature_length * 2 if self.use_temporal_features else self.base_feature_length

    @property
    def resolved_model_path(self) -> Path:
        if self.model_path.is_absolute():
            return self.model_path
        return Path(__file__).resolve().parents[2] / self.model_path


@lru_cache
def get_settings() -> Settings:
    return Settings()
