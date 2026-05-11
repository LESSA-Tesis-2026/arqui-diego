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

The local `.env` file is intentionally untracked. Use it for machine-specific values such as an absolute `LESSA_MODEL_PATH`.

Run the FastAPI backend from `apps/api` with `uv`:

```bash
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

By default the API looks for the model at:

```text
apps/api/artifacts/modelo_senas_lstm.keras
```

For local development, you can point the API to the existing research artifact:

```bash
LESSA_MODEL_PATH="/absolute/path/to/modeloTutorialFtGemini/models/modelo_señas_lstm.keras" uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
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

- `LESSA_MODEL_PATH=/models/modelo_senas_lstm.keras`, matching the model bind mount in the API service.
- `LESSA_CORS_ORIGINS=["http://localhost:3000"]`, matching the published web origin.
- `NEXT_PUBLIC_API_URL=http://localhost:8000`, matching the API port reachable from the browser.

Run the full stack from the repository root:

```bash
docker compose up --build
```

The Compose setup mounts the existing model artifact into the API container:

```text
./modeloTutorialFtGemini/models/modelo_señas_lstm.keras -> /models/modelo_senas_lstm.keras
```

Then open the web app at `http://localhost:3000`. The API is available at `http://localhost:8000/api/v1/health`.

On Apple Silicon, if TensorFlow or MediaPipe Linux wheels fail for native ARM builds, run:

```bash
DOCKER_DEFAULT_PLATFORM=linux/amd64 docker compose up --build
```
