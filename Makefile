.PHONY: dev api web docker down

dev:
	$(MAKE) -j2 api web

api:
	cd apps/api && uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

web:
	cd apps/web && pnpm dev

docker:
	docker compose up --build

down:
	docker compose down
