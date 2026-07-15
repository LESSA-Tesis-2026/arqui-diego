# LESSA Translation API

Backend en FastAPI que sirve los modelos híbridos de traducción de LESSA a español. Esta aplicación es responsable del límite de ejecución alrededor del artefacto de palabras `.keras` entrenado y del artefacto de alfabeto `.h5` opcional, y luego expone los endpoints de salud, metadatos del modelo y streaming de traducción en vivo.

## Estructura

```text
app/
├── api/routes/            HTTP and WebSocket route handlers
├── core/config.py         Environment-driven settings
└── lessa/                 LESSA translation domain
    ├── labels.py          Stable model label ordering
    ├── preprocessing.py   Frame decoding, MediaPipe, feature vectors
    ├── runtimes.py        Keras model loading and metadata
    ├── session.py         Per-WebSocket translation state
    ├── inference.py       Mode routing, prediction, voting, responses
    └── schemas.py         Pydantic API/WebSocket schemas
```

Las rutas deben mantenerse ligeras. Agregue el comportamiento del modelo bajo `app.lessa`, no dentro de los manejadores de rutas.

## Tecnología

- FastAPI
- TensorFlow/Keras
- Preprocesamiento con MediaPipe Holistic
- Decodificación de fotogramas con OpenCV
- `uv` para el manejo del entorno de Python y de dependencias

## Contrato del modelo

Las etiquetas en `app/lessa/labels.py` son contratos del modelo entrenado. Conserve el orden y la ortografía a menos que se entrene un nuevo artefacto de modelo con un contrato diferente.

Configuración actual de entrada del modelo de palabras:

- longitud de secuencia: `60`
- longitud base de características: `306`
- orden de delta temporal: `2`
- longitud final de características: `918`

Los modelos de palabras y de alfabeto utilizan diferentes órdenes de índices de rostro seleccionados. Mantenga esos órdenes estables en `app/lessa/preprocessing.py`.

## Entorno

`.env.example` es la fuente de verdad versionada para la configuración del backend. Cree un `.env` local a partir de él cuando ejecute la API fuera de Docker:

```bash
cp .env.example .env
```

`.env` es solo local. Los valores específicos de Docker se declaran en el `docker-compose.yml` de la raíz.

Variables importantes:

- `LESSA_WORD_MODEL_PATH`: ruta al artefacto del modelo de palabras `.keras`. Las rutas relativas se resuelven desde `apps/api`. Por defecto apunta al `models/modelo_señas_lstm.keras` de la raíz del repositorio; en Docker, Compose monta el directorio `models/` de la raíz en `/models`.
- `LESSA_ALPHABET_MODEL_PATH`: ruta al artefacto del modelo de alfabeto `.h5` opcional. Si falta, la API igual se ejecuta y marca el modo Alfabeto como no disponible.
- `LESSA_CORS_ORIGINS`: lista JSON de orígenes de frontend permitidos.
- `LESSA_SEQUENCE_LENGTH`: longitud de secuencia del modelo de palabras. El modelo actual espera `60`.
- `LESSA_WINDOW_SIZE`: ventana deslizante activa antes del relleno. La ejecución actual usa `25` fotogramas.
- `LESSA_BASE_FEATURE_LENGTH`: longitud de características solo de posición. El preprocesamiento actual usa `306`.
- `LESSA_USE_TEMPORAL_FEATURES`: el modelo de palabras actual espera los deltas temporales habilitados.
- `LESSA_TEMPORAL_DELTA_ORDER`: número de grupos de derivadas temporales que se agregan a cada vector de posición.
- `LESSA_HYBRID_MOTION_THRESHOLD`: umbral del modo Auto que dirige las señas en movimiento a Palabras y las señas estáticas a Alfabeto.
- `LESSA_SETTLE_SECONDS`: segundos que la persona señante debe mantener/hacer la seña antes de que la API inicie la inferencia para el modo actual.
- `LESSA_INFERENCE_INTERVAL_SECONDS`: retraso mínimo entre llamadas de inferencia del modelo después de la estabilización.

Para el desarrollo local, el `.env.example` por defecto apunta a la carpeta `models/` de la raíz del repositorio.

## Instalación

```bash
uv sync
```

## Ejecución

```bash
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

La API estará disponible en:

```text
http://localhost:8000
```

## Endpoints

- `GET /api/v1/health`: accesibilidad de la API.
- `GET /api/v1/model/info`: disponibilidad de palabras/alfabeto, listas de etiquetas, formas de entrada y configuración de inferencia.
- `WS /api/v1/translate/stream`: streaming de fotogramas en vivo para la traducción.

## Mensajes de WebSocket

Mensaje de fotograma (frame):

```json
{
  "type": "frame",
  "frame": "data:image/jpeg;base64,...",
  "mode": "auto"
}
```

Mensaje de reinicio (reset):

```json
{
  "type": "reset"
}
```

Respuesta de traducción típica:

```json
{
  "type": "translation",
  "requested_mode": "auto",
  "mode": "words",
  "prediction_type": "word",
  "word_available": true,
  "alphabet_available": false,
  "prediction": "hola",
  "confidence": 0.91,
  "stable_word": "hola",
  "emitted_word": "hola",
  "emitted_token": "hola",
  "sentence": ["hola"],
  "text": "hola",
  "top": [{ "label": "hola", "confidence": 0.91 }],
  "word_top": [{ "label": "hola", "confidence": 0.91 }],
  "alphabet_top": [],
  "has_hands": true,
  "motion_score": 0.012,
  "status": "modo palabras",
  "error": null
}
```

## Verificaciones

```bash
uv run pytest
```
