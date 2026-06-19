# LESSA Translation API

FastAPI backend for serving the hybrid LESSA-to-Spanish translation models. This app owns the runtime boundary around the trained word `.keras` artifact and optional alphabet `.h5` artifact, then exposes health, model metadata, and live translation streaming endpoints.

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

- `LESSA_WORD_MODEL_PATH`: path to the word `.keras` model artifact. Relative paths are resolved from `apps/api`. By default this points to the repo-root `models/modelo_señas_lstm.keras`; in Docker, Compose mounts the root `models/` directory at `/models`.
- `LESSA_ALPHABET_MODEL_PATH`: path to the optional alphabet `.h5` model artifact. If it is missing, the API still runs and marks Alphabet mode unavailable.
- `LESSA_CORS_ORIGINS`: JSON list of allowed frontend origins.
- `LESSA_SEQUENCE_LENGTH`: word model sequence length. Current model expects `60`.
- `LESSA_WINDOW_SIZE`: active sliding window before padding. Current console inference uses `25` frames.
- `LESSA_BASE_FEATURE_LENGTH`: position-only feature length. Current preprocessing uses `306`.
- `LESSA_USE_TEMPORAL_FEATURES`: current word model expects temporal deltas enabled.
- `LESSA_TEMPORAL_DELTA_ORDER`: number of temporal derivative groups appended to each position vector. Current word model expects `2`, producing `918` features: positions, first-order deltas, and second-order deltas computed over the active window before padding.
- `LESSA_HYBRID_MOTION_THRESHOLD`: Auto mode threshold that routes moving signs to Words and static signs to Alphabet.
- `LESSA_SETTLE_SECONDS`: seconds the signer must hold/sign before the API starts inference for the current mode. Default is `3.0` to reduce flicker.
- `LESSA_INFERENCE_INTERVAL_SECONDS`: minimum delay between model inference calls after settling. Default is `0.75` to reduce processing load.

For local development, the default `.env.example` points to the repo-root `models/` folder where both runtime artifacts live.

For backend-owned runtime artifacts, copy or mount the models at:

```text
models/modelo_señas_lstm.keras
models/modelo_letras.h5
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
- `GET /api/v1/model/info`: word/alphabet availability, label lists, input shapes, and inference configuration.
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
  "status": "traduciendo",
  "error": null
}
```

## Checks

```bash
uv run pytest
```
