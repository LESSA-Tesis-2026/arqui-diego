# Architecture

This repository contains the thesis prototype for live LESSA-to-Spanish translation. The runnable product is a browser experience backed by a FastAPI inference service. The research workspaces explain how the model artifacts are produced, but they are not imported by the production apps at runtime.

## Repository Boundaries

```text
apps/api/                 FastAPI model-serving backend
apps/web/                 Next.js browser translation experience
research/words/           Active word/phrase research workspace
research/alphabet/        Active static alphabet research workspace
research/prototypes/      Earlier OpenCV hybrid-translation prototypes
models/                   Local runtime model artifacts, ignored by Git
docs/                     Architecture, delivery, and development docs
docker-compose.yml        Local full-stack runtime
```

The research workspaces use English folder and file names and are model-development assets, not production application packages. The production apps consume trained artifacts through configured model paths instead of importing research scripts.

## Language Convention

The prototype is for LESSA and Spanish speakers, so Spanish-facing domain output is intentional.

| Area | Convention |
| --- | --- |
| Code identifiers, modules, folders, API fields | English |
| Developer comments and docs | English |
| User-facing interface copy | Spanish |
| Model labels and dataset classes | Preserve trained label contract |

Examples such as `hola`, `buenos_dias`, `mi_nombre_es`, `nada`, and the alphabet classes are model labels. Do not rename or reorder them unless a future model artifact is trained with a new label contract.

## Runtime Flow

```text
Browser camera
  -> apps/web frame capture
  -> WS /api/v1/translate/stream
  -> apps/api frame decoding
  -> MediaPipe Holistic landmarks
  -> LESSA feature vectors
  -> word/alphabet model inference
  -> voting and stabilization
  -> Spanish text and UI feedback
```

1. The frontend requests camera permission and previews the stream locally.
2. While translation is active, frames are throttled and encoded as JPEG data URLs.
3. The backend receives frame messages over WebSocket.
4. MediaPipe extracts pose, selected face, and hand landmarks.
5. The backend builds the exact feature shape expected by the trained model artifacts.
6. Auto mode routes moving signs to word recognition and static signs to alphabet recognition.
7. Voting buffers and settling windows reduce flicker and repeated unstable emissions.
8. The frontend renders Spanish text, current status, prediction confidence, history, and optional browser speech feedback.

## Backend Structure

```text
apps/api/app/
├── api/routes/            FastAPI HTTP/WebSocket route handlers
├── core/config.py         Environment-driven settings
├── lessa/labels.py        Stable model label ordering
├── lessa/preprocessing.py Frame decoding, MediaPipe, feature vectors
├── lessa/runtimes.py      Lazy Keras model loading and metadata
├── lessa/session.py       Per-WebSocket translation state
├── lessa/inference.py     Mode routing, prediction, voting, responses
└── lessa/schemas.py       Pydantic API/WebSocket schemas
```

Routes should remain thin. Put request validation and socket lifecycle in route modules; put model behavior in `app.lessa` modules.

## Frontend Structure

```text
apps/web/src/
├── components/translation/  Translation UI pieces
├── hooks/                   Browser API and socket lifecycle hooks
├── lib/api.ts               REST API utilities
├── lib/websocket.ts         WebSocket message utilities/types
└── lib/labels.ts            Model-label to Spanish-display formatting
```

Browser API logic lives in hooks:

- `useCamera`: camera permission, stream ownership, cleanup.
- `useFrameStreaming`: frame throttling, canvas capture, socket backpressure.
- `useTranslationSocket`: model availability, socket lifecycle, translation state, reset handling, stable-emission speech.
- `useSpeechSynthesis`: hydration-safe Web Speech API integration.

Presentation components receive normalized props and should not own low-level browser resources.

## Model Artifact Contract

Runtime artifacts are loaded from `models/` by default:

```text
models/modelo_señas_lstm.keras
models/modelo_letras.h5
```

These files are intentionally ignored by Git. They can be copied locally or mounted into Docker.

Important contracts:

- Word model labels come from `apps/api/app/lessa/labels.py` in exact order.
- Alphabet model labels are `ABCDEFGHIKLMNOPQRSTUVWXY`; `J` and `Z` are absent because the current alphabet classifier is static-frame based.
- The current word model expects sequence length `60` and feature length `918`.
- The base position feature length is `306`.
- Temporal features append first-order and second-order deltas after position features.
- Selected face-index order differs between word and alphabet models and must remain stable.

## Generated and Local Files

Local/generated artifacts are runtime or machine-specific outputs:

- `.env` files
- `.venv`, `node_modules`, `.next`, caches
- `apps/web/next-env.d.ts`
- runtime files under `models/`
- generated datasets, videos, H5 files, and metrics unless explicitly accepted as thesis evidence
