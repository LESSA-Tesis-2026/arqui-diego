# Local Development

## Backend

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
