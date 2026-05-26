# LESSA Translation API

FastAPI backend for serving the LESSA-to-Spanish translation model. This app owns the runtime boundary around the trained `.keras` artifact and exposes health, model metadata, and live translation streaming endpoints.

## Tech

- FastAPI
- TensorFlow/Keras
- MediaPipe Holistic preprocessing
- OpenCV frame decoding
- `uv` for Python environment and dependency management

## Environment

`.env.example` is the tracked source of truth for backend configuration. Create a local `.env` from it when running the API outside Docker:

```bash
cp .env.example .env
```

`.env` is local-only and should not be committed. Docker does not use a second env example file; Docker-specific values are declared in the root `docker-compose.yml` so they stay next to the container mount that requires them.

Important variables:

- `LESSA_MODEL_PATH`: path to the `.keras` model artifact. Relative paths are resolved from `apps/api`. In Docker, Compose sets this to `/models/modelo_senas_lstm.keras` and bind-mounts the research artifact there.
- `LESSA_CORS_ORIGINS`: JSON list of allowed frontend origins.
- `LESSA_SEQUENCE_LENGTH`: model sequence length. Current model expects `60`.
- `LESSA_WINDOW_SIZE`: active sliding window before padding. Current console inference uses `25` frames.
- `LESSA_BASE_FEATURE_LENGTH`: position-only feature length. Current preprocessing uses `306`.
- `LESSA_USE_TEMPORAL_FEATURES`: current model expects temporal deltas enabled.
- `LESSA_TEMPORAL_DELTA_ORDER`: number of temporal derivative groups appended to each position vector. Current model expects `2`, producing `918` features: positions, first-order deltas, and second-order deltas computed over the active window before padding.

For local development, `.env` points to the existing artifact in `modeloTutorialFtGemini/models/modelo_señas_lstm.keras`.

For a backend-owned runtime artifact, copy or mount the model at:

```text
apps/api/artifacts/modelo_senas_lstm.keras
```

## Install

```bash
uv sync
```

## Run

```bash
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:

```text
http://localhost:8000
```

## Endpoints

- `GET /api/v1/health`: API reachability.
- `GET /api/v1/model/info`: model availability, label list, input shape, and inference configuration.
- `WS /api/v1/translate/stream`: live frame streaming for translation.

## WebSocket Messages

Frame message:

```json
{
  "type": "frame",
  "frame": "data:image/jpeg;base64,..."
}
```

Reset message:

```json
{
  "type": "reset"
}
```

Typical translation response:

```json
{
  "type": "translation",
  "model_available": true,
  "prediction": "hola",
  "confidence": 0.91,
  "stable_word": "hola",
  "emitted_word": "hola",
  "sentence": ["hola"],
  "text": "hola",
  "top": [{ "label": "hola", "confidence": 0.91 }],
  "has_hands": true,
  "status": "traduciendo",
  "error": null
}
```

## Checks

```bash
uv run pytest
```
