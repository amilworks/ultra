.PHONY: help install dev dev-stack run run-reload run-frontend restart-dev stop-dev status-dev restart-control-stack stop-control-stack status-control-stack deploy-control-stack release-artifact test test-chat-stack verify-integration postgres-up postgres-init postgres-down postgres-logs postgres-psql postgres-reset test-postgres-store migrate-run-store-postgres control-migrate lint format clean codeexec-image frontend-lint frontend-type-check frontend-test-unit frontend-test-smoke frontend-quality frontend-autonomy-test control-test control-integration control-soak control-run control-tidy control-generate deepagents-test deepagents-worker-test deepagents-autonomy-test deepagents-smoke autonomy-live-smoke delegation-live-smoke async-delegation-live-smoke rigor-live-smoke episodic-live-smoke autonomy-gate up up-detached down down-clean logs ps scale-workers

ENV_FILE := $(if $(wildcard .env),.env,.env.example)
COMPOSE_ENV_FILE := $(if $(wildcard .env.docker),.env.docker,.env.docker.example)
PYTHON_QUALITY_SCOPE := backend/deepagents_runtime/src backend/deepagents_runtime/tests tests
PYTHON_TYPECHECK_SCOPE := backend/deepagents_runtime/src
PYTHON_STRICT_SCOPE := backend/deepagents_runtime/src
PYTHON_STRICT_RULES := --select B,RUF,SIM,RET

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# ─── Docker stack (canonical local "near-production" environment) ──────────────
# `make up` builds (if needed) and runs the WHOLE stack in containers. Prefer this
# over the native *-control-stack / dev-stack targets, which run Go/Python/Vite on
# the host. Uses .env.docker if present, else .env.docker.example.

up: ## Start the production-parity stack in Docker (mirrors the deployed topology)
	docker compose --env-file $(COMPOSE_ENV_FILE) up --build

up-detached: ## Start the production-parity stack in the background
	docker compose --env-file $(COMPOSE_ENV_FILE) up --build -d

down: ## Stop the Docker stack (keeps data volumes: Postgres, uploads, JetStream)
	docker compose down

down-clean: ## Stop the Docker stack AND delete its data volumes
	docker compose down -v

logs: ## Tail logs from the Docker stack (CTRL-C to stop tailing)
	docker compose logs -f

ps: ## Show status of the Docker stack
	docker compose ps

scale-workers: ## Run N agent workers as a NATS queue group, e.g. make scale-workers N=3 (default 2 = prod-like)
	docker compose --env-file $(COMPOSE_ENV_FILE) up --build -d --scale worker=$(or $(N),2)

install: ## Install production dependencies
	uv sync

dev: ## Install all dependencies including dev tools
	uv sync --all-extras

dev-stack: ## (native/no-Docker) Start the host Go stack + workers + frontend — prefer 'make up'
	./scripts/restart_control_stack.sh restart

run: ## Run the Go control plane API
	$(MAKE) -C backend/controlplane run

run-reload: ## Run the Go control plane API; Go hot reload is not configured
	$(MAKE) -C backend/controlplane run

run-frontend: ## Run the React frontend
	pnpm --dir frontend dev

restart-dev: ## Restart production-like Go control stack
	./scripts/restart_control_stack.sh restart

stop-dev: ## Stop production-like Go control stack
	./scripts/restart_control_stack.sh stop

status-dev: ## Inspect production-like Go control stack
	./scripts/restart_control_stack.sh status

restart-control-stack: ## (native, FAST INNER LOOP — not prod parity) host Go+NATS+PG+worker+Vite; use 'make up' to mirror prod
	./scripts/restart_control_stack.sh restart

stop-control-stack: ## Stop production-like Go control stack
	./scripts/restart_control_stack.sh stop

status-control-stack: ## Inspect production-like Go control stack durability and worker health
	./scripts/restart_control_stack.sh status

deploy-control-stack: ## Deploy Go control plane + Deep Agents worker stack (set RELEASE_SHA)
	@release="$${RELEASE_SHA:-$${SHA:-}}"; \
	if [ -z "$$release" ]; then \
		echo "Set RELEASE_SHA=<git-sha> or SHA=<git-sha>." >&2; \
		exit 1; \
	fi; \
	./scripts/deploy_ultra_control_stack.sh "$$release"

release-artifact: ## Build immutable Go control + frontend release tarball
	./scripts/build_ultra_release_artifact.sh

test: ## Run tests with pytest
	uv run pytest

test-chat-stack: ## Run production-like Go control and Deep Agents chat checks
	$(MAKE) control-test
	$(MAKE) deepagents-test

verify-integration: ## Validate Go control plane persistence and transport integration
	$(MAKE) control-integration

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

control-migrate: ## Apply the Go control-plane Postgres schema
	cd backend/controlplane && go run ./cmd/ultra-control migrate

test-cov: ## Run tests with coverage report
	uv run pytest --cov=backend/deepagents_runtime/src --cov-report=html --cov-report=term

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

codeexec-image: ## Build Python sandbox image for execute_python_job (bakes RareSpot weights)
	@test -f data/models/yolo/RareSpotWeights.pt || { \
		echo "ERROR: data/models/yolo/RareSpotWeights.pt is missing — the prairie-dog-detection"; \
		echo "Skill bakes it into the sandbox image. Stage it from the model store first, e.g.:"; \
		echo "  mkdir -p data/models/yolo && cp /path/to/RareSpotWeights.pt data/models/yolo/"; \
		exit 1; }
	@vcs_ref="$${GITHUB_SHA:-$$(git rev-parse HEAD)}"; \
	docker build --build-arg "VCS_REF=$$vcs_ref" -f deploy/docker/deepagents-sandbox.Dockerfile -t $${CODE_EXECUTION_DOCKER_IMAGE:-bisque-ultra-codeexec:py311} .

control-test: ## Run Go control plane tests
	$(MAKE) -C backend/controlplane test

control-integration: ## Run Go control plane Postgres + NATS integration gate
	$(MAKE) -C backend/controlplane integration

control-soak: ## Run deterministic Go control-plane autonomous-run soak gate
	$(MAKE) -C backend/controlplane soak

control-run: ## Run Go control plane API
	$(MAKE) -C backend/controlplane run

control-tidy: ## Tidy Go control plane module
	$(MAKE) -C backend/controlplane tidy

control-generate: ## Regenerate Go control plane OpenAPI and sqlc code
	$(MAKE) -C backend/controlplane generate

deepagents-test: ## Run Python Deep Agents runtime tests
	cd backend/deepagents_runtime && uv run --python 3.11 --extra dev pytest -q

deepagents-worker-test: ## Run Deep Agents worker transport, lease, and redelivery tests
	cd backend/deepagents_runtime && uv run --python 3.11 --extra dev pytest -q tests/test_worker_transport.py

deepagents-autonomy-test: ## Run deterministic Deep Agents autonomy quality and routing tests
	cd backend/deepagents_runtime && uv run --python 3.11 --extra dev pytest -q \
		tests/test_live_trace.py \
		tests/test_runner_paper_preload.py

deepagents-smoke: ## Probe the configured Python Deep Agents vLLM model endpoint
	cd backend/deepagents_runtime && OPENAI_BASE_URL=$${OPENAI_BASE_URL:-http://127.0.0.1:8003/v1} OPENAI_MODEL=$${OPENAI_MODEL:-deepseek_v4} uv run --python 3.11 python -m ultra_deepagents.smoke

frontend-autonomy-test: ## Run frontend autonomous-chat recovery and artifact hydration tests
	pnpm --dir frontend exec vitest run \
		src/features/chat/run-artifact-hydration.test.ts \
		src/features/chat/run-stream-recovery-app.test.ts \
		src/features/chat/chat-submit-terminal.test.ts \
		src/features/chat/run-recovery.test.ts \
		src/features/chat/stale-conversation.test.ts \
		src/lib/api.test.ts

autonomy-live-smoke: ## Run opt-in live two-turn tool-autonomy trace against a running stack
	cd backend/deepagents_runtime && uv run --python 3.11 --extra dev python -m ultra_deepagents.live_trace \
		--base-url $${ULTRA_LIVE_TRACE_BASE_URL:-http://127.0.0.1:8000} \
		--title "Autonomy live smoke" \
		--prompt "Start by calling tool_capability_manifest. Then use Python to create a small matplotlib figure showing y = x^2 for x = 0..5, save the code and plot as durable outputs, and briefly explain what the plot demonstrates." \
		--followup "Start by calling tool_capability_manifest. Using the prior code/artifacts, modify the analysis to plot y = x^3 for x = 0..5, save updated code and plot, and explain what changed." \
		--timeout $${ULTRA_LIVE_TRACE_TIMEOUT_SECONDS:-600} \
		--poll-interval $${ULTRA_LIVE_TRACE_POLL_INTERVAL_SECONDS:-1} \
		--http-timeout $${ULTRA_LIVE_TRACE_HTTP_TIMEOUT_SECONDS:-15} \
		--verify-downloads \
		--require-downloads \
		--min-artifacts 2 \
		--min-response-chars 50 \
		--require-tool-autonomy-quality \
		--require-tool-capability-quality \
		--capability-matrix \
		--require-thread-quality

# For WorkOS-gated stacks, export ULTRA_LIVE_TRACE_COOKIE='ultra_workos_session=...'
# or ULTRA_LIVE_TRACE_AUTHORIZATION before running this target.
delegation-live-smoke: ## Run opt-in live scoped subagent delegation trace against a running stack
	cd backend/deepagents_runtime && uv run --python 3.11 --extra dev python -m ultra_deepagents.live_trace \
		--base-url $${ULTRA_LIVE_TRACE_BASE_URL:-http://127.0.0.1:8000} \
		--title "Delegation live smoke" \
		--prompt "Start by calling tool_capability_manifest. Delegate a focused data/code inspection subtask to an available subagent: create a tiny CSV with x=0..5 and y=x^2, have the subagent inspect or compute summary statistics from it, then reconcile the subagent result into a concise final answer with durable code/report outputs." \
		--timeout $${ULTRA_LIVE_TRACE_TIMEOUT_SECONDS:-600} \
		--poll-interval $${ULTRA_LIVE_TRACE_POLL_INTERVAL_SECONDS:-1} \
		--http-timeout $${ULTRA_LIVE_TRACE_HTTP_TIMEOUT_SECONDS:-15} \
		--verify-downloads \
		--require-downloads \
		--min-artifacts 1 \
		--min-response-chars 50 \
		--require-delegation-quality \
		--require-tool-capability-quality \
		--capability-matrix \
		--require-thread-quality

rigor-live-smoke: ## Run opt-in live Intelligence-Pro rigor results-contract trace against a running stack
	cd backend/deepagents_runtime && uv run --python 3.11 --extra dev python -m ultra_deepagents.live_trace \
		--base-url $${ULTRA_LIVE_TRACE_BASE_URL:-http://127.0.0.1:8000} \
		--title "Rigor live smoke" \
		--workflow-hint-id pro_mode \
		--prompt "Run a small computational study: integrate the logistic map x_{n+1} = r x_n (1 - x_n) for r in {2.8, 3.2, 3.5, 3.9}, estimate the Lyapunov exponent for each r, classify each regime, save a metrics CSV, a bifurcation figure at 300 DPI, the analysis code, and a short markdown report, and finish with a concise summary." \
		--timeout $${ULTRA_LIVE_TRACE_TIMEOUT_SECONDS:-1500} \
		--poll-interval $${ULTRA_LIVE_TRACE_POLL_INTERVAL_SECONDS:-2} \
		--min-artifacts 4 \
		--min-response-chars 400

episodic-live-smoke: ## Run opt-in live episodic-memory recall trace (seed a conclusion, recall it in a new thread)
	cd backend/deepagents_runtime && uv run --python 3.11 --extra dev python -m ultra_deepagents.live_trace \
		--base-url $${ULTRA_LIVE_TRACE_BASE_URL:-http://127.0.0.1:8000} \
		--title "Episodic seed" \
		--prompt "Record for later: in my synthetic benchmark, method Foo reached 91.2% accuracy and method Bar reached 87.5%. Acknowledge in one sentence." \
		--followup "Search my past sessions: which method scored higher in my earlier synthetic benchmark, and by how much? Use your memory of past sessions, not a new computation." \
		--timeout $${ULTRA_LIVE_TRACE_TIMEOUT_SECONDS:-400} \
		--poll-interval $${ULTRA_LIVE_TRACE_POLL_INTERVAL_SECONDS:-2}

async-delegation-live-smoke: ## Run opt-in live async/background subagent trace (requires a configured external Agent Protocol server)
	@if [ -z "$$ULTRA_DEEPAGENTS_ENABLE_ASYNC_SUBAGENTS" ] || [ -z "$$ULTRA_DEEPAGENTS_ASYNC_SUBAGENTS_JSON" ]; then \
		echo "SKIP: async subagents not configured (set ULTRA_DEEPAGENTS_ENABLE_ASYNC_SUBAGENTS=1 + ULTRA_DEEPAGENTS_ASYNC_SUBAGENTS_JSON pointing at a remote Agent Protocol server). This feature requires an external LangGraph deployment Ultra does not run by default."; \
		exit 0; \
	fi
	cd backend/deepagents_runtime && uv run --python 3.11 --extra dev python -m ultra_deepagents.live_trace \
		--base-url $${ULTRA_LIVE_TRACE_BASE_URL:-http://127.0.0.1:8000} \
		--title "Async delegation live smoke" \
		--prompt "Start by calling tool_capability_manifest. If available_async_subagents is non-empty, launch exactly one background task with start_async_task using an available async subagent. Ask it to run a concise scientific/code analysis and return the full task_id to me. After launching, stop and do not check status in this turn." \
		--followup "Start by calling tool_capability_manifest. Use list_async_tasks to fetch current status for all async tasks. If a task is complete, call check_async_task once with the full task_id and summarize the status/result. If it is still running, report the full task_id and current status without polling." \
		--timeout $${ULTRA_LIVE_TRACE_TIMEOUT_SECONDS:-600} \
		--poll-interval $${ULTRA_LIVE_TRACE_POLL_INTERVAL_SECONDS:-1} \
		--http-timeout $${ULTRA_LIVE_TRACE_HTTP_TIMEOUT_SECONDS:-15} \
		--min-response-chars 30 \
		--require-async-delegation-quality \
		--require-tool-capability-quality \
		--capability-matrix \
		--require-thread-quality

autonomy-gate: control-soak control-integration deepagents-worker-test deepagents-autonomy-test frontend-autonomy-test ## Run production autonomy durability gates
