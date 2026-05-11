# LESSA Translation API

FastAPI backend for serving the LESSA-to-Spanish translation model. This app owns the runtime boundary around the trained `.keras` artifact and exposes health, model metadata, and live translation streaming endpoints.

## Tech

- FastAPI
- TensorFlow/Keras
- MediaPipe Holistic preprocessing
- OpenCV frame decoding
- `uv` for Python environment and dependency management

## Environment

Create or update `.env` from `.env.example`:

```bash
cp .env.example .env
```

Important variables:

- `LESSA_MODEL_PATH`: path to the `.keras` model artifact. Relative paths are resolved from `apps/api`.
- `LESSA_CORS_ORIGINS`: JSON list of allowed frontend origins.
- `LESSA_SEQUENCE_LENGTH`: model sequence length. Current model expects `60`.
- `LESSA_BASE_FEATURE_LENGTH`: position-only feature length. Current preprocessing uses `306`.
- `LESSA_USE_TEMPORAL_FEATURES`: current model expects temporal deltas enabled, producing `612` features.

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
