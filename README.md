# LESSA Translation

Web application for live LESSA-to-Spanish translation. The project combines a research/modeling workspace, a FastAPI model-serving backend, a browser translation experience, shared documentation, and Docker orchestration for local runtime. The prototype now supports hybrid word and alphabet inference.

## Parts

- `apps/web`: Next.js frontend. It owns camera access, Auto/Words/Alphabet controls, live status, and Spanish translation output.
- `apps/api`: FastAPI backend. It owns frame preprocessing, model loading, hybrid inference, stabilization, and the public API/WebSocket contract.
- `modeloTutorialFtGemini`: research and model-development workspace for the word/phrase LSTM. It contains the scripts used to capture samples, process datasets, train the model, evaluate results, and run the original real-time translation prototype. The production apps consume the trained `.keras` artifact from this workflow instead of importing these scripts directly.
- `modeloAlfabeto`: research and model-development workspace for the static alphabet classifier. The app can load its trained `.h5` artifact when available.
- `docs`: project-level documentation, including local development and Docker instructions.
- `docker-compose.yml`: local full-stack runtime for the web app and API.

## How It Connects

The browser captures camera frames in `apps/web` and streams them to the API over a WebSocket:

```text
apps/web -> WS /api/v1/translate/stream -> apps/api
```

The API decodes each frame, extracts hand/body landmarks once, builds the word and alphabet feature vectors, and chooses the active recognizer. Auto mode routes moving signs to the word LSTM and static signs to the alphabet classifier; the UI can also force Words or Alphabet mode. The API waits for a short settling window before inference and throttles predictions to avoid flicker. The frontend then renders recognition status, confidence, recent predictions, and the progressive Spanish text.

The trained models are runtime artifacts. They are not built by the web or API applications. The word model remains required for word recognition; the alphabet model is optional and its mode is marked unavailable until the `.h5` artifact exists.

## Runtime Flow

1. User opens the web app at `http://localhost:3000`.
2. Browser requests camera permission.
3. Frontend connects to the API WebSocket at `http://localhost:8000/api/v1/translate/stream`.
4. Frontend sends encoded frames while translation is active.
5. API preprocesses frames and runs word or alphabet inference based on the selected mode and motion score.
6. API sends hybrid translation events back to the browser.
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

Local `.env` files are intentionally untracked. The API uses `LESSA_WORD_MODEL_PATH` and `LESSA_ALPHABET_MODEL_PATH` for the two runtime artifacts. Docker-specific values live in `docker-compose.yml` because they are tied to published ports and volume mounts.
