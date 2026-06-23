# LESSA Translation

Thesis prototype for live LESSA-to-Spanish translation. The project combines a FastAPI model-serving backend, a Next.js browser translation experience, active model-development workspaces, shared documentation, and Docker orchestration for local runtime. The prototype supports hybrid word/phrase and alphabet inference.

## Repository Map

- `apps/api`: FastAPI backend. It owns frame preprocessing, model loading, hybrid inference, stabilization, and the public REST/WebSocket contract.
- `apps/web`: Next.js frontend. It owns camera access, Auto/Words/Alphabet controls, live status, Spanish translation output, and optional browser speech feedback.
- `research/words`: active research/model-development workspace for the word/phrase LSTM. The production apps consume the trained `.keras` artifact from this workflow instead of importing these scripts directly.
- `research/alphabet`: active research/model-development workspace for the static alphabet classifier. The app can load its trained `.h5` artifact when available.
- `research/prototypes`: OpenCV hybrid-translation prototypes for research checks.
- `models`: local runtime artifact folder. It is ignored by Git and should contain model files for local/Docker execution.
- `docs`: project-level architecture, delivery, and development documentation.
- `docker-compose.yml`: local full-stack runtime for the web app and API.


## Language and Model Labels

The prototype is for LESSA and Spanish speakers:

- User-facing copy stays Spanish.
- LESSA classes and trained model labels stay stable, e.g. `hola`, `buenos_dias`, `mi_nombre_es`, and `nada`.
- Code identifiers, module names, API fields, and developer-facing documentation use English unless they represent user-facing Spanish or model labels.

## How It Connects

The browser captures camera frames in `apps/web` and streams them to the API over a WebSocket:

```text
apps/web -> WS /api/v1/translate/stream -> apps/api
```

The API decodes each frame, extracts hand/body landmarks once, builds the word and alphabet feature vectors, and chooses the active recognizer. Auto mode routes moving signs to the word LSTM and static signs to the alphabet classifier; the UI can also force Words or Alphabet mode. The API waits for a short settling window before inference and throttles predictions to avoid flicker. The frontend renders recognition status, confidence, recent predictions, and progressive Spanish text. When browser speech synthesis is available, the frontend can speak accepted emissions: full words/phrases in Words mode and individual letters in Alphabet mode.

The trained models are runtime artifacts. They are not built by the web or API applications. The word model remains required for word recognition; the alphabet model is optional and its mode is marked unavailable until the `.h5` artifact exists.

## Runtime Flow

1. User opens the web app at `http://localhost:3000`.
2. Browser requests camera permission.
3. Frontend connects to the API WebSocket at `http://localhost:8000/api/v1/translate/stream`.
4. Frontend sends encoded frames while translation is active.
5. API preprocesses frames and runs word or alphabet inference based on the selected mode and motion score.
6. API sends hybrid translation events back to the browser.
7. Frontend displays the current translation and session history.
8. If voice feedback is enabled and supported by the browser, accepted emitted tokens are spoken with the browser SpeechSynthesis API.

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): repository boundaries, runtime flow, backend/frontend structure, model contracts.
- [`docs/DELIVERY.md`](docs/DELIVERY.md): thesis-demo setup, verification checklist, generated-file policy, future-work notes.
- [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md): local development and Docker commands.
- [`research/README.md`](research/README.md): research workspace boundaries and generated-artifact policy.

## Local Development

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

See [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) for full setup.

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
