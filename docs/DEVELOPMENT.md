# Local Development

## Backend

The backend has one tracked env template:

```text
apps/api/.env.example
```

Copy it to local `.env` when running outside Docker:

```bash
cp apps/api/.env.example apps/api/.env
```

The local `.env` file is intentionally untracked. Use it for machine-specific values such as absolute `LESSA_WORD_MODEL_PATH` and `LESSA_ALPHABET_MODEL_PATH` values.

Run the FastAPI backend from `apps/api` with `uv`:

```bash
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

By default the API looks for model artifacts at:

```text
models/modelo_señas_lstm.keras
models/modelo_letras.h5
```

The alphabet model is optional for startup; Alphabet mode is reported unavailable until the file exists and loads successfully.

For local development, you can point the API to the existing research artifact:

```bash
LESSA_WORD_MODEL_PATH="/absolute/path/to/models/modelo_señas_lstm.keras" \
LESSA_ALPHABET_MODEL_PATH="/absolute/path/to/models/modelo_letras.h5" \
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Useful backend checks:

```bash
uv run pytest
```

## Frontend

The frontend has one tracked env template:

```text
apps/web/.env.example
```

Copy it to local `.env` when running outside Docker:

```bash
cp apps/web/.env.example apps/web/.env
```

The local `.env` file is intentionally untracked. `NEXT_PUBLIC_API_URL` must be reachable by the browser, not just by the Next.js process.

Run the web app from `apps/web` with `pnpm`:

```bash
pnpm dev
```

The frontend expects the API at `http://localhost:8000` by default. Override it with:

```bash
NEXT_PUBLIC_API_URL="http://localhost:8000" pnpm dev
```

Useful frontend checks:

```bash
pnpm lint
pnpm build
```

## Docker

Docker does not have separate `.env.*` example files. The Docker-specific values live in `docker-compose.yml` because they are coupled to service ports and volume mounts.

Important Docker overrides:

- `LESSA_WORD_MODEL_PATH=/models/modelo_señas_lstm.keras`, matching the repo-root models bind mount in the API service.
- `LESSA_ALPHABET_MODEL_PATH=/models/modelo_letras.h5`, the expected container path for the optional alphabet artifact.
- `LESSA_TEMPORAL_DELTA_ORDER=2`, matching the current word model input shape of `(60, 918)`.
- `LESSA_WINDOW_SIZE=25`, matching the console translator's active frame window before padding to `60`.
- `LESSA_WORD_CONFIDENCE_THRESHOLD=0.65`, `LESSA_ALPHABET_CONFIDENCE_THRESHOLD=0.80`, `LESSA_HYBRID_MOTION_THRESHOLD=0.010`, `LESSA_SETTLE_SECONDS=3.0`, `LESSA_INFERENCE_INTERVAL_SECONDS=0.75`, `LESSA_VOTING_BUFFER_SIZE=10`, and `LESSA_MIN_VOTES=7`, matching the hybrid console translator while adding a delayed analysis window to reduce flicker.
- `LESSA_CORS_ORIGINS=["http://localhost:3000"]`, matching the published web origin.
- `NEXT_PUBLIC_API_URL=http://localhost:8000`, matching the API port reachable from the browser.

Run the full stack from the repository root:

```bash
docker compose up --build
```

The Compose setup mounts the repo-root model artifact folder into the API container:

```text
./models -> /models
```

Keep both runtime files there when available:

```text
models/modelo_señas_lstm.keras
models/modelo_letras.h5
```

Then open the web app at `http://localhost:3000`. The API is available at `http://localhost:8000/api/v1/health`.

On Apple Silicon, if TensorFlow or MediaPipe Linux wheels fail for native ARM builds, run:

```bash
DOCKER_DEFAULT_PLATFORM=linux/amd64 docker compose up --build
```
