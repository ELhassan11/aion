.PHONY: setup up down logs lint format test

setup:
	cp -n .env.example .env || true
	@echo "Fill in .env, then run: make up"

up:
	docker compose up --build -d

down:
	docker compose down

logs:
	docker compose logs -f

lint:
	ruff check . && ruff format --check .

format:
	ruff format .

test:
	pytest tests/ -q
