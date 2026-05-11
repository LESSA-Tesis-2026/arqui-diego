# LESSA Translation

Web application for live LESSA-to-Spanish translation. The project combines a research/modeling workspace, a FastAPI model-serving backend, a browser translation experience, shared documentation, and Docker orchestration for local runtime.

## Parts

- `apps/web`: Next.js frontend. It owns camera access, user controls, live status, and Spanish translation output.
- `apps/api`: FastAPI backend. It owns frame preprocessing, model loading, inference, stabilization, and the public API/WebSocket contract.
- `modeloTutorialFtGemini`: research and model-development workspace. It contains the scripts used to capture samples, process datasets, train the model, evaluate results, and run the original real-time translation prototype. The production apps consume the trained `.keras` artifact from this workflow instead of importing these scripts directly.
- `docs`: project-level documentation, including local development and Docker instructions.
- `docker-compose.yml`: local full-stack runtime for the web app and API.

## How It Connects

The browser captures camera frames in `apps/web` and streams them to the API over a WebSocket:

```text
apps/web -> WS /api/v1/translate/stream -> apps/api
```

The API decodes each frame, extracts hand/body landmarks, builds the model input sequence, runs the trained `.keras` model produced by the research workflow, and returns prediction updates to the browser. The frontend then renders recognition status, confidence, recent predictions, and the progressive Spanish text.

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
