"""Configuración central de la API de traducción, con todos los parámetros ajustables en un solo lugar.

Todos los campos pueden sobrescribirse por variable de entorno usando el prefijo `LESSA_`
(por ejemplo, `LESSA_SETTLE_SECONDS=2.0`) o desde un archivo `.env` local. Esto permite
afinar el comportamiento en tiempo de ejecución (por ejemplo, en Docker) sin tocar el código.

ADVERTENCIA sobre los contratos del modelo: `sequence_length`, `base_feature_length`,
`use_temporal_features` y `temporal_delta_order` describen la forma exacta de entrada con la
que se entrenó el artefacto `.keras`. Cambiarlos sin reentrenar el modelo hará que la
inferencia falle o produzca resultados incorrectos. Consulte `docs/ARCHITECTURE.md`
(sección "Contrato de características y del modelo") antes de modificarlos.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "LESSA Translation API"
    api_prefix: str = "/api/v1"
    # Orígenes permitidos por CORS. Debe incluir el origen del frontend (el navegador que abre el WebSocket).
    cors_origins: list[str] = ["http://localhost:3000"]

    # --- Rutas de artefactos de modelo ---
    # Relativas a apps/api (se resuelven en `_resolve_app_path`). Sobrescribir con rutas
    # absolutas vía LESSA_WORD_MODEL_PATH / LESSA_ALPHABET_MODEL_PATH según el entorno.
    word_model_path: Path = Path("../../models/modelo_señas_lstm.keras")
    alphabet_model_path: Path = Path("../../models/modelo_letras.h5")

    # --- Contrato de forma de entrada (debe coincidir con el modelo entrenado) ---
    # sequence_length: número de fotogramas por muestra que espera la LSTM tras el relleno (padding).
    sequence_length: int = 60
    # window_size: fotogramas del historial en vivo que se mantienen en memoria antes del relleno a
    # `sequence_length`. Es la ventana deslizante de la sesión, más corta que la secuencia entrenada.
    window_size: int = 25
    # base_feature_length: valores de posición por fotograma =
    #   33 pose x 4 (x,y,z,visibility) + 16 face x 3 + 21 mano izq x 3 + 21 mano der x 3 = 306.
    base_feature_length: int = 306
    # use_temporal_features: si es True, cada fotograma se enriquece con velocidad (y aceleración)
    # antes de la inferencia. temporal_delta_order=2 => posición + 1ª derivada + 2ª derivada.
    # La longitud de características resultante es 306 * (1 + 2) = 918 (ver `feature_length`).
    use_temporal_features: bool = True
    temporal_delta_order: int = 2

    # --- Umbrales de decisión ---
    # Confianza mínima (softmax) para aceptar una predicción; por debajo se trata como "nada"/"-".
    word_confidence_threshold: float = 0.65
    alphabet_confidence_threshold: float = 0.80
    # En modo Auto, si el movimiento reciente de las manos supera este umbral se enruta a palabras
    # (seña dinámica); si no, al alfabeto (seña estática). Ver `motion_score` en preprocessing.py.
    hybrid_motion_threshold: float = 0.010

    # --- Temporización (segundos) ---
    # settle_seconds: ventana de estabilización; la persona debe mantener la seña este tiempo antes
    # de que comience la inferencia. Sube la latencia pero reduce lecturas basura durante la transición.
    settle_seconds: float = 3.0
    # inference_interval_seconds: intervalo mínimo entre inferencias para no analizar cada fotograma.
    inference_interval_seconds: float = 0.75
    # hold_last_reading_seconds: cuánto se mantiene la última lectura tras perder las manos (evita
    # que el texto parpadee cuando MediaPipe pierde el tracking por un instante).
    hold_last_reading_seconds: float = 2.0

    # --- Votación / estabilización ---
    # Se acumulan las últimas `voting_buffer_size` predicciones y solo se emite una etiqueta cuando
    # alcanza `min_votes` apariciones. Filtra predicciones inestables fotograma a fotograma.
    voting_buffer_size: int = 10
    min_votes: int = 7
    alphabet_voting_buffer_size: int = 5
    alphabet_min_votes: int = 3

    # --- Construcción de la oración ---
    # max_sentence_words: cuántos tokens emitidos recientes se conservan en la oración devuelta.
    max_sentence_words: int = 5
    # pause_reset_frames: fotogramas consecutivos sin seña válida antes de reiniciar el estado emitido.
    pause_reset_frames: int = 15

    model_config = SettingsConfigDict(env_prefix="LESSA_", env_file=".env", extra="ignore")

    @property
    def feature_length(self) -> int:
        # Longitud real del vector de características por fotograma que se envía al modelo de palabras.
        # Sin características temporales es la posición base (306); con ellas se concatenan la posición
        # y `temporal_delta_order` derivadas: 306 * (1 + 2) = 918.
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
