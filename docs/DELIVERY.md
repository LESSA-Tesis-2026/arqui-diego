# Thesis Delivery Guide

This guide is the handoff checklist for running and reviewing the LESSA-to-Spanish thesis prototype.

## Delivery Scope

Included in delivery:

- `apps/api`: FastAPI backend for model serving and live translation streaming.
- `apps/web`: Next.js frontend for the browser translation experience.
- `research/words`: active word/phrase research workspace.
- `research/alphabet`: active alphabet research workspace.
- `research/prototypes`: OpenCV prototypes for research checks.
- `docs`: setup, architecture, delivery, and future-work notes.
- `docker-compose.yml`: local full-stack runtime.

Not part of source delivery:

- Generated files such as `apps/web/next-env.d.ts`.
- Local dependencies, caches, `.env`, and virtual environments.
- Runtime model artifacts unless explicitly distributed through a separate release artifact.

## Model Artifacts

Place runtime model artifacts at the repository root:

```text
models/modelo_señas_lstm.keras
models/modelo_letras.h5
```

The word model is required for word recognition. The alphabet model is optional at startup; if missing, the API reports Alphabet mode unavailable.

## Local Backend

```bash
cd apps/api
cp .env.example .env
uv sync
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Health check:

```text
http://localhost:8000/api/v1/health
```

Model metadata:

```text
http://localhost:8000/api/v1/model/info
```

## Local Frontend

```bash
cd apps/web
cp .env.example .env
pnpm install
pnpm dev
```

Open:

```text
http://localhost:3000
```

The browser must be able to reach `NEXT_PUBLIC_API_URL`, which defaults to `http://localhost:8000`.

## Docker Runtime

From the repository root:

```bash
docker compose up --build
```

Then open:

```text
http://localhost:3000
```

Docker mounts `./models` into the API container at `/models`.

On Apple Silicon, if TensorFlow or MediaPipe wheels fail for native ARM builds, use:

```bash
DOCKER_DEFAULT_PLATFORM=linux/amd64 docker compose up --build
```

## Verification Checklist

Run before thesis delivery:

```bash
cd apps/api
uv run pytest
```

```bash
cd apps/web
pnpm lint
pnpm build
```

After checks, inspect Git status. Expected local-only ignored files may include `.env`, `.venv`, `.next`, `node_modules`, `models/`, and `apps/web/next-env.d.ts`. They should not appear as tracked modifications.

## Future Work

### Adding or changing signs

Additions require research/model work first. Update datasets, train a new model, and only then update runtime labels. Runtime label order must match the trained artifact exactly.

### Replacing model artifacts

When replacing models, verify:

- label order
- expected input sequence length
- feature length
- face landmark selection
- temporal delta order
- confidence and voting thresholds

Update docs and tests with the new contract.

### Threshold tuning

Relevant backend settings include:

- `LESSA_WORD_CONFIDENCE_THRESHOLD`
- `LESSA_ALPHABET_CONFIDENCE_THRESHOLD`
- `LESSA_HYBRID_MOTION_THRESHOLD`
- `LESSA_SETTLE_SECONDS`
- `LESSA_INFERENCE_INTERVAL_SECONDS`
- `LESSA_VOTING_BUFFER_SIZE`
- `LESSA_MIN_VOTES`
- `LESSA_ALPHABET_VOTING_BUFFER_SIZE`
- `LESSA_ALPHABET_MIN_VOTES`

Lower thresholds can make the demo feel faster but may emit incorrect signs. Higher thresholds can reduce errors but may feel less responsive.

### Deployment

This prototype is local/Docker-ready. A production deployment would still need explicit decisions for TLS, model artifact distribution, compute sizing, privacy, authentication if exposed beyond a demo network, observability, and camera/browser compatibility testing.
