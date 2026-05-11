# LESSA Translation Web

Immersive browser experience for real-time LESSA-to-Spanish translation. The frontend owns camera access, live translation controls, and progressive Spanish output while streaming frames to the FastAPI backend.

## Tech

- Next.js
- React
- TypeScript
- Tailwind CSS
- shadcn/ui
- `pnpm` for package management

## Environment

Create or update `.env` from `.env.example`:

```bash
cp .env.example .env
```

Variables:

- `NEXT_PUBLIC_API_URL`: FastAPI backend URL. Defaults to `http://localhost:8000` for local development.

## Install

```bash
pnpm install
```

## Run

Start the backend first, then run:

```bash
pnpm dev
```

Open:

```text
http://localhost:3000
```

## Main Flow

- Activate camera permission.
- Start live translation.
- Stream camera frames to `WS /api/v1/translate/stream`.
- Display recognition status and progressive Spanish text.
- Pause or clear the current translation session.

## Checks

```bash
pnpm lint
pnpm build
```

## Notes

- The browser requires camera permission for live translation.
- If the API is unavailable, the interface shows a recoverable unavailable state.
- The UI intentionally avoids a dashboard/navbar pattern and keeps the translation moment as the primary experience.
