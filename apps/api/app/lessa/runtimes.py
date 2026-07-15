from pathlib import Path

from tensorflow.keras.models import load_model

from app.core.config import Settings
from app.lessa.labels import ALPHABET_LABELS, WORD_LABELS
from app.lessa.schemas import RuntimeModelInfo


class ModelRuntime:
    """Cargador diferido de modelos Keras con reporte seguro de disponibilidad."""

    def __init__(self, model_path: Path, labels: list[str]) -> None:
        self.model_path = model_path
        self.labels = labels
        self._model = None
        self._load_error: str | None = None

    @property
    def available(self) -> bool:
        self._ensure_loaded()
        return self._model is not None

    @property
    def error(self) -> str | None:
        self._ensure_loaded()
        return self._load_error

    @property
    def model(self):
        self._ensure_loaded()
        return self._model

    def info(self) -> RuntimeModelInfo:
        self._ensure_loaded()
        input_shape = None
        output_shape = None
        if self._model is not None:
            input_shape = [dim if dim is None else int(dim) for dim in self._model.input_shape]
            output_shape = [dim if dim is None else int(dim) for dim in self._model.output_shape]

        return RuntimeModelInfo(
            available=self._model is not None,
            error=self._load_error,
            labels=self.labels,
            model_path=str(self.model_path),
            input_shape=input_shape,
            output_shape=output_shape,
        )

    def _ensure_loaded(self) -> None:
        if self._model is not None or self._load_error is not None:
            return

        if not self.model_path.is_file():
            self._load_error = f"Model artifact not found at {self.model_path}"
            return

        try:
            self._model = load_model(self.model_path, compile=False)
        except Exception as exc:  # pragma: no cover - depende de la compatibilidad del artefacto local
            self._load_error = str(exc)


class WordModelRuntime(ModelRuntime):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings.resolved_word_model_path, WORD_LABELS)


class AlphabetModelRuntime(ModelRuntime):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings.resolved_alphabet_model_path, ALPHABET_LABELS)
