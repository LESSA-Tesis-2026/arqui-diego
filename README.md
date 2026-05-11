# LESSA Translation

Web application for live LESSA-to-Spanish translation. The project is split into a browser experience, a FastAPI model-serving backend, shared development documentation, and Docker orchestration for running the stack locally.

## Parts

- `apps/web`: Next.js frontend. It owns camera access, user controls, live status, and Spanish translation output.
- `apps/api`: FastAPI backend. It owns frame preprocessing, model loading, inference, stabilization, and the public API/WebSocket contract.
- `docs`: project-level documentation, including local development and Docker instructions.
- `docker-compose.yml`: local full-stack runtime for the web app and API.
- `openspec`: change proposals, specs, and task tracking for product changes.
- `.agents`: reusable workflow guidance for agent-assisted development.

## How It Connects

The browser captures camera frames in `apps/web` and streams them to the API over a WebSocket:

```text
apps/web -> WS /api/v1/translate/stream -> apps/api
```

The API decodes each frame, extracts hand/body landmarks, builds the model input sequence, runs the trained `.keras` model, and returns prediction updates to the browser. The frontend then renders recognition status, confidence, recent predictions, and the progressive Spanish text.

The trained model is treated as a runtime artifact. It is not built by the web or API applications. For local Docker runs, Compose bind-mounts the artifact into the API container and sets `LESSA_MODEL_PATH` to the mounted path.

## Runtime Flow

1. User opens the web app at `http://localhost:3000`.
2. Browser requests camera permission.
3. Frontend connects to the API WebSocket at `http://localhost:8000/api/v1/translate/stream`.
4. Frontend sends encoded frames while translation is active.
5. API preprocesses frames and runs inference against the configured model artifact.
6. API sends translation events back to the browser.
7. Frontend displays the current translation and session history.

## Local Development

Use the detailed guide in [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md).

Backend checks:

```bash
cd apps/api
uv run pytest
```

Frontend checks:

```bash
cd apps/web
pnpm lint
pnpm build
```

## Docker

Run the full stack from the repository root:

```bash
docker compose up --build
```

Then open:

```text
http://localhost:3000
```

The API health endpoint is available at:

```text
http://localhost:8000/api/v1/health
```

## Environment

Each app has one tracked environment template:

- `apps/api/.env.example`
- `apps/web/.env.example`

Local `.env` files are intentionally untracked. Docker-specific values live in `docker-compose.yml` because they are tied to published ports and volume mounts.
