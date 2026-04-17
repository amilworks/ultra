.PHONY: help install dev platform-up platform-down platform-logs platform-up-prod platform-down-prod platform-logs-prod platform-config-prod dev-stack run run-reload run-frontend restart-dev stop-dev status-dev test test-chat-stack verify-platform-smoke verify-integration seed-bisque-fixtures cleanup-bisque-fixtures verify-bisque-chat-api verify-bisque-chat-live smoke-pro-mode-opus postgres-up postgres-init postgres-down postgres-logs postgres-psql postgres-reset test-postgres-store migrate-run-store-postgres lint format clean codeexec-image frontend-lint frontend-type-check frontend-test-unit frontend-test-smoke frontend-quality

ENV_FILE := $(if $(wildcard .env),.env,.env.example)
PLATFORM_COMPOSE_FILES := -f platform/bisque/docker-compose.with-engine.yml -f platform/bisque/docker-compose.oidc.yml
PLATFORM_SERVICES := bisque postgres keycloak
PLATFORM_PROD_COMPOSE_FILES := -f platform/bisque/docker-compose.with-engine.yml -f platform/bisque/docker-compose.production.yml
PYTHON_QUALITY_SCOPE := src tests
PYTHON_TYPECHECK_SCOPE := --explicit-package-bases src/config.py src/auth src/api/client.py src/api/v3.py src/tooling/domains src/evals/golden_tasks.py
PYTHON_STRICT_SCOPE := src/auth src/config.py src/api/client.py src/api/v3.py src/tooling/domains src/tooling/engine.py src/evals/golden_tasks.py src/agno_backend/runtime.py src/agno_backend/pro_mode.py src/training/adapters.py src/api/main.py
PYTHON_STRICT_RULES := --select B,RUF,SIM,RET

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	uv sync

dev: ## Install all dependencies including dev tools
	uv sync --all-extras

platform-up: ## Start the absorbed BisQue platform from platform/bisque using the root env file
	docker compose --env-file $(ENV_FILE) $(PLATFORM_COMPOSE_FILES) up -d --build $(PLATFORM_SERVICES)

platform-down: ## Stop the absorbed BisQue platform containers
	docker compose --env-file $(ENV_FILE) $(PLATFORM_COMPOSE_FILES) down --remove-orphans

platform-logs: ## Tail logs from the absorbed BisQue platform container
	docker compose --env-file $(ENV_FILE) $(PLATFORM_COMPOSE_FILES) logs -f $(PLATFORM_SERVICES)

platform-up-prod: ## Start the production-shaped BisQue platform (localhost-bound ports, hardened Keycloak)
	docker compose --env-file $(ENV_FILE) $(PLATFORM_PROD_COMPOSE_FILES) up -d --build $(PLATFORM_SERVICES)

platform-down-prod: ## Stop the production-shaped BisQue platform containers
	docker compose --env-file $(ENV_FILE) $(PLATFORM_PROD_COMPOSE_FILES) down --remove-orphans

platform-logs-prod: ## Tail logs from the production-shaped BisQue platform container
	docker compose --env-file $(ENV_FILE) $(PLATFORM_PROD_COMPOSE_FILES) logs -f $(PLATFORM_SERVICES)

platform-config-prod: ## Print merged production docker compose config for BisQue + Keycloak + Postgres
	docker compose --env-file $(ENV_FILE) $(PLATFORM_PROD_COMPOSE_FILES) config

dev-stack: ## Start BisQue platform in Docker plus local API/frontend processes
	ENV_FILE=$(ENV_FILE) ./scripts/dev_stack.sh

run: ## Run the FastAPI backend
	uv run uvicorn src.api.main:app --host 127.0.0.1 --port 8000

run-reload: ## Run the FastAPI backend with scoped reload watchers
	@reload_dirs="--reload-dir src"; \
	if [ -d tests ]; then reload_dirs="$$reload_dirs --reload-dir tests"; fi; \
	uv run uvicorn src.api.main:app --reload $$reload_dirs --host 127.0.0.1 --port 8000

run-frontend: ## Run the React frontend
	pnpm --dir frontend dev

restart-dev: ## Restart local backend (with reload) and frontend dev servers
	./scripts/restart_dev.sh restart

stop-dev: ## Stop local backend and frontend dev servers
	./scripts/restart_dev.sh stop

status-dev: ## Check local backend and frontend dev server health
	./scripts/restart_dev.sh status

test: ## Run tests with pytest
	uv run pytest

test-chat-stack: ## Run chat streaming/tool composition integration checks
	uv run pytest tests/test_auth_bridge.py tests/test_bisque_auth_hardening.py tests/test_bisque_oidc_user_reconciliation.py -q

verify-platform-smoke: ## Validate platform/bisque compose config and health endpoint
	ENV_FILE=$(ENV_FILE) ./scripts/verify_platform_smoke.sh

verify-integration: ## Validate local API wiring against the absorbed BisQue platform
	ENV_FILE=$(ENV_FILE) ./scripts/verify_integration.sh

seed-bisque-fixtures: ## Seed deterministic BisQue fixtures for chat/live verification
	ENV_FILE=$(ENV_FILE) ./scripts/manage_bisque_chat_fixtures.sh seed

cleanup-bisque-fixtures: ## Remove deterministic BisQue fixtures created for chat/live verification
	ENV_FILE=$(ENV_FILE) ./scripts/manage_bisque_chat_fixtures.sh cleanup

verify-bisque-chat-api: ## Run deterministic BisQue chat selector/runtime/API verification
	./scripts/verify_bisque_chat_api.sh

verify-bisque-chat-live: ## Run seeded live Playwright BisQue chat verification against the local stack
	ENV_FILE=$(ENV_FILE) ./scripts/verify_bisque_chat_live.sh

smoke-pro-mode-opus: ## Probe the configured Pro Mode Opus gateway and optionally the local backend
	uv run python scripts/smoke_pro_mode_opus.py

postgres-up: ## Start local Postgres for production-like testing
	docker compose -f docker-compose.postgres.yml up -d

postgres-init: ## Ensure primary and test databases exist
	docker compose -f docker-compose.postgres.yml up -d
	@for i in $$(seq 1 40); do \
		STATUS=$$(docker inspect -f '{{.State.Health.Status}}' bisque-ultra-postgres 2>/dev/null || echo starting); \
		if [ "$$STATUS" = "healthy" ]; then break; fi; \
		sleep 1; \
	done
	@DB="$${POSTGRES_DB:-bisque_ultra}"; TEST_DB="$${DB}_test"; PGUSER="$${POSTGRES_USER:-postgres}"; \
	docker compose -f docker-compose.postgres.yml exec -T postgres sh -lc "psql -U \"$$PGUSER\" -d postgres -tAc \"SELECT 1 FROM pg_database WHERE datname='$$DB'\" | grep -q 1 || psql -U \"$$PGUSER\" -d postgres -c \"CREATE DATABASE \\\"$$DB\\\"\""; \
	docker compose -f docker-compose.postgres.yml exec -T postgres sh -lc "psql -U \"$$PGUSER\" -d postgres -tAc \"SELECT 1 FROM pg_database WHERE datname='$$TEST_DB'\" | grep -q 1 || psql -U \"$$PGUSER\" -d postgres -c \"CREATE DATABASE \\\"$$TEST_DB\\\"\""

postgres-down: ## Stop local Postgres test container
	docker compose -f docker-compose.postgres.yml down

postgres-logs: ## Tail local Postgres logs
	docker compose -f docker-compose.postgres.yml logs -f postgres

postgres-psql: ## Open psql shell in local Postgres container
	docker compose -f docker-compose.postgres.yml exec postgres psql -U $${POSTGRES_USER:-postgres} -d $${POSTGRES_DB:-bisque_ultra}

postgres-reset: ## Drop local Postgres data volume directory
	docker compose -f docker-compose.postgres.yml down -v
	rm -rf data/postgres

test-postgres-store: ## Run Postgres integration tests (requires RUN_STORE_POSTGRES_TEST_DSN)
	@if [ ! -f tests/test_run_store_postgres.py ]; then \
		echo "No dedicated Postgres store test is present in tests/."; \
		exit 0; \
	fi
	@DSN="$${RUN_STORE_POSTGRES_TEST_DSN:-postgresql://postgres:postgres@127.0.0.1:55432/bisque_ultra_test}"; \
	if command -v uv >/dev/null 2>&1; then \
		RUN_STORE_POSTGRES_TEST_DSN="$$DSN" uv run pytest tests/test_run_store_postgres.py; \
	else \
		RUN_STORE_POSTGRES_TEST_DSN="$$DSN" ./.venv/bin/pytest tests/test_run_store_postgres.py; \
	fi

migrate-run-store-postgres: ## Migrate SQLite run-store to Postgres (set SQLITE_RUN_STORE_PATH + POSTGRES_RUN_STORE_DSN)
	@SQLITE_PATH="$${SQLITE_RUN_STORE_PATH:-data/runs.db}"; \
	POSTGRES_DSN="$${POSTGRES_RUN_STORE_DSN:-postgresql://postgres:postgres@127.0.0.1:55432/bisque_ultra}"; \
	TRUNCATE_FLAG=""; if [ "$$MIGRATE_TRUNCATE" = "1" ]; then TRUNCATE_FLAG="--truncate-target"; fi; \
	if command -v uv >/dev/null 2>&1; then \
		uv run python scripts/migrate_run_store_to_postgres.py --sqlite-path "$$SQLITE_PATH" --postgres-dsn "$$POSTGRES_DSN" $$TRUNCATE_FLAG; \
	else \
		./.venv/bin/python scripts/migrate_run_store_to_postgres.py --sqlite-path "$$SQLITE_PATH" --postgres-dsn "$$POSTGRES_DSN" $$TRUNCATE_FLAG; \
	fi

test-cov: ## Run tests with coverage report
	uv run pytest --cov=src --cov-report=html --cov-report=term

lint: ## Run linting checks
	uv run ruff check $(PYTHON_QUALITY_SCOPE)

lint-strict: ## Run stricter backend lint checks on ratcheted backend scope
	uv run ruff check $(PYTHON_STRICT_SCOPE) $(PYTHON_STRICT_RULES)

format: ## Format backend code with Ruff
	uv run ruff format $(PYTHON_QUALITY_SCOPE)

format-check: ## Check backend formatting without making changes
	uv run ruff format --check $(PYTHON_QUALITY_SCOPE)

type-check: ## Run type checking with mypy
	uv run mypy $(PYTHON_TYPECHECK_SCOPE)

quality: lint format-check type-check lint-strict ## Run all quality checks

frontend-lint: ## Run frontend lint checks
	pnpm --dir frontend lint

frontend-type-check: ## Run frontend type checking
	pnpm --dir frontend typecheck

frontend-test-unit: ## Run frontend unit tests
	pnpm --dir frontend test:unit

frontend-test-smoke: ## Run frontend smoke tests
	pnpm --dir frontend test:smoke

frontend-quality: frontend-lint frontend-type-check frontend-test-unit ## Run core frontend quality checks

clean: ## Clean up generated files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage

shell: ## Open a shell in the virtual environment
	uv run bash

codeexec-image: ## Build Python sandbox image for execute_python_job
	docker build -f docker/codeexec/Dockerfile -t $${CODE_EXECUTION_DOCKER_IMAGE:-bisque-ultra-codeexec:py311} .
