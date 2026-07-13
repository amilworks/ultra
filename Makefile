.PHONY: help install dev dev-stack run run-reload run-frontend restart-dev stop-dev status-dev restart-control-stack stop-control-stack status-control-stack deploy-control-stack release-artifact test test-chat-stack verify-integration postgres-up postgres-init postgres-down postgres-logs postgres-psql postgres-reset test-postgres-store migrate-run-store-postgres control-migrate lint format clean codeexec-image materials-kinetics-image materials-domain-gate materials-domain-test materials-production-parity materials-production-source-contract materials-production-parity-test calphad-ledger-qualification calphad-cross-language-qualification calphad-cross-language-test materials-production-readiness materials-production-readiness-test materials-promotion-envelope-test materials-promotion-envelope-create materials-promotion-envelope-verify-root materials-promotion-attestation-verify mattools-evaluator-build mattools-evaluator-verify mattools-promotion-test mattools-promotion-inspect mattools-promotion-diagnostic mattools-promotion-gate frontend-lint frontend-type-check frontend-test-unit frontend-test-smoke frontend-quality frontend-autonomy-test control-test control-integration control-soak control-run control-tidy control-generate deepagents-test deepagents-worker-test deepagents-autonomy-test deepagents-smoke autonomy-live-smoke delegation-live-smoke async-delegation-live-smoke rigor-live-smoke episodic-live-smoke autonomy-gate up up-detached down down-clean logs ps scale-workers

ENV_FILE := $(if $(wildcard .env),.env,.env.example)
COMPOSE_ENV_FILE := $(if $(wildcard .env.docker),.env.docker,.env.docker.example)
PYTHON_QUALITY_SCOPE := backend/deepagents_runtime/src backend/deepagents_runtime/tests tests
PYTHON_TYPECHECK_SCOPE := backend/deepagents_runtime/src
PYTHON_STRICT_SCOPE := backend/deepagents_runtime/src
PYTHON_STRICT_RULES := --select B,RUF,SIM,RET
MATTOOLS_EVALUATOR_ENV_LOCK ?= deploy/docker/mattools-evaluator-linux-arm64-lock.json

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# ─── Docker stack (canonical local "near-production" environment) ──────────────
# `make up` builds (if needed) and runs the WHOLE stack in containers. Prefer this
# over the native *-control-stack / dev-stack targets, which run Go/Python/Vite on
# the host. Uses .env.docker if present, else .env.docker.example.

up: materials-kinetics-image ## Start the full stack in Docker, building if needed (canonical local stack)
	@image="$$(docker compose --env-file $(COMPOSE_ENV_FILE) config --format json | python3 -c 'import json,sys; print(json.load(sys.stdin)["services"]["materials-kinetics-runtime"]["image"])')"; \
	image_id="$$(docker image inspect --format '{{.Id}}' "$$image")"; \
	ULTRA_MATERIALS_KINETICS_RUNTIME_IMAGE="$$image" ULTRA_MATERIALS_KINETICS_RUNTIME_IMAGE_ID="$$image_id" \
		docker compose --env-file $(COMPOSE_ENV_FILE) up --build

up-detached: materials-kinetics-image ## Start the full Docker stack in the background
	@image="$$(docker compose --env-file $(COMPOSE_ENV_FILE) config --format json | python3 -c 'import json,sys; print(json.load(sys.stdin)["services"]["materials-kinetics-runtime"]["image"])')"; \
	image_id="$$(docker image inspect --format '{{.Id}}' "$$image")"; \
	ULTRA_MATERIALS_KINETICS_RUNTIME_IMAGE="$$image" ULTRA_MATERIALS_KINETICS_RUNTIME_IMAGE_ID="$$image_id" \
		docker compose --env-file $(COMPOSE_ENV_FILE) up --build -d

down: ## Stop the Docker stack (keeps data volumes: Postgres, uploads, JetStream)
	docker compose down

down-clean: ## Stop the Docker stack AND delete its data volumes
	docker compose down -v

logs: ## Tail logs from the Docker stack (CTRL-C to stop tailing)
	docker compose logs -f

ps: ## Show status of the Docker stack
	docker compose ps

scale-workers: materials-kinetics-image ## Run N agent workers as a NATS queue group, e.g. make scale-workers N=3
	@image="$$(docker compose --env-file $(COMPOSE_ENV_FILE) config --format json | python3 -c 'import json,sys; print(json.load(sys.stdin)["services"]["materials-kinetics-runtime"]["image"])')"; \
	image_id="$$(docker image inspect --format '{{.Id}}' "$$image")"; \
	ULTRA_MATERIALS_KINETICS_RUNTIME_IMAGE="$$image" ULTRA_MATERIALS_KINETICS_RUNTIME_IMAGE_ID="$$image_id" \
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

restart-control-stack: ## (native/no-Docker) Restart the host Go+NATS+PG+worker+frontend stack — prefer 'make up'
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

materials-kinetics-image: ## Build and qualify the separate Kawin/NumPy-2 typed runtime image
	docker compose --env-file $(COMPOSE_ENV_FILE) build materials-kinetics-runtime
	@image="$$(docker compose --env-file $(COMPOSE_ENV_FILE) config --format json | python3 -c 'import json,sys; print(json.load(sys.stdin)["services"]["materials-kinetics-runtime"]["image"])')"; \
	image_id="$$(docker image inspect --format '{{.Id}}' "$$image")"; \
	title="$$(docker image inspect --format '{{ index .Config.Labels "org.opencontainers.image.title" }}' "$$image")"; \
	test "$$title" = "Ultra isolated materials kinetics runtime"; \
	echo "Qualified isolated materials kinetics image $$image_id"

materials-domain-gate: ## Run pinned, non-skipping deterministic materials invariants in Docker
	./scripts/run_materials_domain_gate.sh

materials-domain-test: materials-domain-gate ## Alias for deterministic materials evidence (not full readiness)

materials-production-parity: ## Run exact full production image parity through DockerSandboxBackend
	@test -n "$${MATERIALS_RELEASE_ROOT:-}" || { echo "Set MATERIALS_RELEASE_ROOT to the extracted immutable release tree (not a Git checkout)." >&2; exit 1; }
	@test -f "$${MATERIALS_RELEASE_ROOT}/release-manifest.json" || { echo "MATERIALS_RELEASE_ROOT lacks release-manifest.json." >&2; exit 1; }
	@test ! -e "$${MATERIALS_RELEASE_ROOT}/.git" || { echo "MATERIALS_RELEASE_ROOT must be an extracted release tree without .git." >&2; exit 1; }
	@release_root="$$(cd "$${MATERIALS_RELEASE_ROOT}" && pwd -P)"; \
	image="$${MATERIALS_PRODUCTION_PARITY_IMAGE:-$${ULTRA_DEEPAGENTS_SANDBOX_IMAGE:-bisque-ultra-codeexec:py311}}"; \
	sha="$${MATERIALS_EXPECTED_GIT_SHA:-$${GITHUB_SHA:-$$(git rev-parse HEAD)}}"; \
	uv run --frozen --project "$$release_root/backend/deepagents_runtime" --python 3.11 python \
		"$$release_root/scripts/verify_production_materials_sandbox.py" \
		--repo-root "$$release_root" --image "$$image" --expected-git-sha "$$sha" \
		--scope production-full \
		--output-dir "$${MATERIALS_PRODUCTION_PARITY_REPORT_DIR:-.tmp/materials-production-parity}"

materials-production-source-contract: ## Run pinned lean-image backend contract (not full-image parity)
	@image="$${MATERIALS_DOMAIN_GATE_IMAGE:-bisque-ultra-materials-domain-gate:py311}"; \
	sha="$${MATERIALS_EXPECTED_GIT_SHA:-$${GITHUB_SHA:-$$(git rev-parse HEAD)}}"; \
	uv run --project backend/deepagents_runtime --python 3.11 python \
		scripts/verify_production_materials_sandbox.py \
		--repo-root . --image "$$image" --expected-git-sha "$$sha" \
		--scope ci-pinned-materials --prepare-entrypoint-adapter \
		--output-dir "$${MATERIALS_PRODUCTION_PARITY_REPORT_DIR:-.tmp/materials-production-source-contract}"

materials-production-parity-test: ## Test production materials parity runner contracts
	uv run --project backend/deepagents_runtime --python 3.11 --extra dev \
		pytest -q tests/test_production_materials_sandbox.py \
			tests/test_materials_domain_gate_runner.py

calphad-ledger-qualification: ## Qualify append-only CALPHAD governance against a dedicated test Postgres
	@test -n "$${ULTRA_CONTROL_TEST_DATABASE_URL:-}" || { echo "Set ULTRA_CONTROL_TEST_DATABASE_URL to a dedicated test/CI/qualification database." >&2; exit 1; }
	@test -n "$${ULTRA_CONTROL_TEST_MIGRATION_DATABASE_URL:-}" || { echo "Set ULTRA_CONTROL_TEST_MIGRATION_DATABASE_URL to the distinct schema-owner URL for the same disposable database." >&2; exit 1; }
	@test "$${MATERIALS_CALPHAD_QUALIFICATION_CONFIRMED:-}" = "dedicated-test-database" || { echo "Set MATERIALS_CALPHAD_QUALIFICATION_CONFIRMED=dedicated-test-database after independently confirming the target is disposable." >&2; exit 1; }
	@sha="$${MATERIALS_EXPECTED_GIT_SHA:-$${GITHUB_SHA:-$$(git rev-parse HEAD)}}"; \
		uv run --isolated --no-project --python 3.11 python \
			scripts/calphad_ledger_gate.py \
			--repository-root . \
			--expected-git-sha "$$sha" \
			--qualification-database-confirmed \
			--output-dir "$${MATERIALS_CALPHAD_LEDGER_OUTPUT_DIR:-.tmp/calphad-ledger-qualification}"

calphad-cross-language-test: ## Test fail-closed typed-CLI -> Go HTTP -> PostgreSQL qualification contracts
	uv run --isolated --no-project --python 3.11 --with pytest==8.4.2 \
		pytest -q -p no:cacheprovider tests/test_calphad_cross_language_gate.py
	go -C backend/controlplane test ./integration -run '^$$' -count=1

calphad-cross-language-qualification: ## Qualify real pycalphad evidence through live Go HTTP and dedicated Postgres
	@test -n "$${ULTRA_CONTROL_TEST_DATABASE_URL:-}" || { echo "Set ULTRA_CONTROL_TEST_DATABASE_URL to the non-owner role on a dedicated qualification database." >&2; exit 1; }
	@test -n "$${ULTRA_CONTROL_TEST_MIGRATION_DATABASE_URL:-}" || { echo "Set ULTRA_CONTROL_TEST_MIGRATION_DATABASE_URL to the distinct schema-owner role on that database." >&2; exit 1; }
	@test "$${MATERIALS_CALPHAD_QUALIFICATION_CONFIRMED:-}" = "dedicated-test-database" || { echo "Set MATERIALS_CALPHAD_QUALIFICATION_CONFIRMED=dedicated-test-database after confirming the target is disposable." >&2; exit 1; }
	@test -n "$${MATERIALS_CALPHAD_CROSS_LANGUAGE_IMAGE:-}" || { echo "Set MATERIALS_CALPHAD_CROSS_LANGUAGE_IMAGE to the exact production scientific sandbox image reference." >&2; exit 1; }
	@test -n "$${MATERIALS_EXPECTED_CROSS_LANGUAGE_IMAGE:-}" || { echo "Set MATERIALS_EXPECTED_CROSS_LANGUAGE_IMAGE=sha256:... from the trusted image build output." >&2; exit 1; }
	@image="$${MATERIALS_CALPHAD_CROSS_LANGUAGE_IMAGE}"; \
		image_id="$${MATERIALS_EXPECTED_CROSS_LANGUAGE_IMAGE}"; \
		sha="$${MATERIALS_EXPECTED_GIT_SHA:-$${GITHUB_SHA:-$$(git rev-parse HEAD)}}"; \
		uv run --isolated --no-project --python 3.11 python \
			scripts/calphad_cross_language_gate.py \
			--repository-root . \
			--output-dir "$${MATERIALS_CALPHAD_CROSS_LANGUAGE_OUTPUT_DIR:-.tmp/calphad-cross-language-qualification}" \
			--expected-git-sha "$$sha" \
			--qualification-database-confirmed \
			--mode pinned-image \
			--image "$$image" \
			--expected-image-id "$$image_id" \
			--expected-image-title "$${MATERIALS_CALPHAD_CROSS_LANGUAGE_IMAGE_TITLE:-Ultra Deep Agents scientific sandbox}"

materials-production-readiness: ## Aggregate full-image parity, MatTools >=80%/60%, and live evidence
	@test -n "$${MATERIALS_DETERMINISTIC_REPORT:-}" || { echo "Set MATERIALS_DETERMINISTIC_REPORT." >&2; exit 1; }
	@test -n "$${MATERIALS_PRODUCTION_PARITY_REPORT:-}" || { echo "Set MATERIALS_PRODUCTION_PARITY_REPORT to a production-full content-addressed report." >&2; exit 1; }
	@test -n "$${MATERIALS_CALPHAD_LEDGER_REPORT:-}" || { echo "Set MATERIALS_CALPHAD_LEDGER_REPORT to a dedicated-Postgres CALPHAD ledger qualification report." >&2; exit 1; }
	@test -n "$${MATERIALS_CALPHAD_CROSS_LANGUAGE_REPORT:-}" || { echo "Set MATERIALS_CALPHAD_CROSS_LANGUAGE_REPORT to a production-runtime typed-CLI/HTTP/Postgres qualification report." >&2; exit 1; }
	@test -n "$${MATERIALS_CALPHAD_CROSS_LANGUAGE_REPORT_MANIFEST:-}" || { echo "Set MATERIALS_CALPHAD_CROSS_LANGUAGE_REPORT_MANIFEST to that qualification's report_manifest.json." >&2; exit 1; }
	@test -n "$${MATTOOLS_REPORT:-}" || { echo "Set MATTOOLS_REPORT to the complete three-trial promotion report." >&2; exit 1; }
	@test -n "$${MATTOOLS_REPORT_MANIFEST:-}" || { echo "Set MATTOOLS_REPORT_MANIFEST to the campaign report_manifest.json." >&2; exit 1; }
	@test -n "$${MATERIALS_LIVE_TRACE_REPORT:-}" || { echo "Set MATERIALS_LIVE_TRACE_REPORT to a designated live-trace report." >&2; exit 1; }
	@test -n "$${MATTOOLS_BENCHMARK_ROOT:-}" || { echo "Set MATTOOLS_BENCHMARK_ROOT to the pinned official checkout." >&2; exit 1; }
	@test -n "$${MATERIALS_EXPECTED_GIT_SHA:-}" || { echo "Set MATERIALS_EXPECTED_GIT_SHA." >&2; exit 1; }
	@test -n "$${MATERIALS_EXPECTED_DOMAIN_IMAGE:-}" || { echo "Set MATERIALS_EXPECTED_DOMAIN_IMAGE=sha256:..." >&2; exit 1; }
	@test -n "$${MATERIALS_EXPECTED_RUNTIME_IMAGE:-}" || { echo "Set MATERIALS_EXPECTED_RUNTIME_IMAGE=sha256:..." >&2; exit 1; }
	@test -n "$${MATERIALS_EXPECTED_EVALUATOR_IMAGE:-}" || { echo "Set MATERIALS_EXPECTED_EVALUATOR_IMAGE=sha256:..." >&2; exit 1; }
	@output="$${MATERIALS_READINESS_OUTPUT_DIR:-.tmp/materials-production-readiness}"; \
		mkdir -p "$$output"; \
		uv run --project backend/deepagents_runtime --python 3.11 python \
			scripts/materials_readiness_gate.py \
			--deterministic-report "$$MATERIALS_DETERMINISTIC_REPORT" \
			--production-parity-report "$$MATERIALS_PRODUCTION_PARITY_REPORT" \
			--calphad-ledger-report "$$MATERIALS_CALPHAD_LEDGER_REPORT" \
			--calphad-cross-language-report "$$MATERIALS_CALPHAD_CROSS_LANGUAGE_REPORT" \
			--calphad-cross-language-report-manifest "$$MATERIALS_CALPHAD_CROSS_LANGUAGE_REPORT_MANIFEST" \
			--mattools-report "$$MATTOOLS_REPORT" \
			--mattools-report-manifest "$$MATTOOLS_REPORT_MANIFEST" \
			--live-trace "$$MATERIALS_LIVE_TRACE_REPORT" \
			--repository-root . \
			--benchmark-root "$$MATTOOLS_BENCHMARK_ROOT" \
			--expected-git-sha "$$MATERIALS_EXPECTED_GIT_SHA" \
			--expected-domain-image "$$MATERIALS_EXPECTED_DOMAIN_IMAGE" \
			--expected-runtime-image "$$MATERIALS_EXPECTED_RUNTIME_IMAGE" \
			--expected-evaluator-image "$$MATERIALS_EXPECTED_EVALUATOR_IMAGE" \
			--output-json "$$output/materials-production-readiness.json" \
			--output-markdown "$$output/materials-production-readiness.md" \
			--output-manifest "$$output/materials-production-readiness-manifest.json"

materials-production-readiness-test: ## Test fail-closed full materials promotion aggregation
	uv run --project backend/deepagents_runtime --python 3.11 --extra dev \
		pytest -q tests/test_materials_readiness_gate.py

materials-promotion-envelope-test: ## Test restricted evidence closure, workflow, and GitHub attestation policy
	uv run --project backend/deepagents_runtime --python 3.11 --extra dev \
		pytest -q -p no:cacheprovider \
		tests/test_materials_promotion_envelope.py \
		tests/test_materials_production_workflow.py

materials-promotion-envelope-create: ## Create a restricted closure and sanitized candidate envelope from explicit role paths
	@: "$${MATERIALS_EVIDENCE_ROOT:?Set MATERIALS_EVIDENCE_ROOT.}"
	@: "$${MATERIALS_EVIDENCE_ROOT_MANIFEST:?Set MATERIALS_EVIDENCE_ROOT_MANIFEST outside the evidence root.}"
	@: "$${MATERIALS_RELEASE_ENVELOPE:?Set MATERIALS_RELEASE_ENVELOPE outside the evidence root.}"
	@: "$${MATERIALS_EXPECTED_GIT_SHA:?Set MATERIALS_EXPECTED_GIT_SHA.}"
	@: "$${MATERIALS_GITHUB_RUN_ID:?Set MATERIALS_GITHUB_RUN_ID.}"
	@: "$${MATERIALS_GITHUB_RUN_ATTEMPT:?Set MATERIALS_GITHUB_RUN_ATTEMPT.}"
	@: "$${MATERIALS_WORKFLOW_SIGNER_DIGEST:?Set MATERIALS_WORKFLOW_SIGNER_DIGEST.}"
	@: "$${MATERIALS_EXPECTED_WORKFLOW_FILE:?Set MATERIALS_EXPECTED_WORKFLOW_FILE.}"
	@: "$${MATERIALS_RUNTIME_OCI_DIGEST:?Set MATERIALS_RUNTIME_OCI_DIGEST.}"
	@: "$${MATERIALS_EXPECTED_RUNTIME_IMAGE:?Set MATERIALS_EXPECTED_RUNTIME_IMAGE.}"
	@: "$${MATERIALS_EXPECTED_DOMAIN_IMAGE:?Set MATERIALS_EXPECTED_DOMAIN_IMAGE.}"
	@: "$${MATERIALS_EXPECTED_EVALUATOR_IMAGE:?Set MATERIALS_EXPECTED_EVALUATOR_IMAGE.}"
	@: "$${MATTOOLS_LICENSE_BASIS:?Set MATTOOLS_LICENSE_BASIS.}"
	@: "$${MATERIALS_RESTRICTED_STORE_LOCATOR_SHA256:?Set the already-hashed restricted store locator.}"
	@: "$${MATERIALS_READINESS_REPORT_ROLE:?Set the readiness-report path relative to the evidence root.}"
	@: "$${MATERIALS_READINESS_MANIFEST_ROLE:?Set the readiness-manifest path relative to the evidence root.}"
	@: "$${MATERIALS_DETERMINISTIC_REPORT_ROLE:?Set the deterministic-report path relative to the evidence root.}"
	@: "$${MATERIALS_PRODUCTION_PARITY_REPORT_ROLE:?Set the production-parity path relative to the evidence root.}"
	@: "$${MATERIALS_CALPHAD_LEDGER_REPORT_ROLE:?Set the CALPHAD-ledger path relative to the evidence root.}"
	@: "$${MATERIALS_CALPHAD_CROSS_LANGUAGE_REPORT_ROLE:?Set the CALPHAD cross-language report path.}"
	@: "$${MATERIALS_CALPHAD_CROSS_LANGUAGE_MANIFEST_ROLE:?Set the CALPHAD cross-language manifest path.}"
	@: "$${MATTOOLS_REPORT_ROLE:?Set the MatTools report path relative to the evidence root.}"
	@: "$${MATTOOLS_MANIFEST_ROLE:?Set the MatTools manifest path relative to the evidence root.}"
	@: "$${MATERIALS_LIVE_TRACE_ROLE:?Set the designated live-trace path relative to the evidence root.}"
	@: "$${MATERIALS_RELEASE_TARBALL_ROLE:?Set the release-tarball path relative to the evidence root.}"
	@: "$${MATERIALS_RELEASE_MANIFEST_ROLE:?Set the release-manifest path relative to the evidence root.}"
	@set --; \
	if [ "$${MATTOOLS_LICENSE_BASIS}" = "separately_licensed" ]; then \
		: "$${MATTOOLS_LICENSE_EVIDENCE_SHA256:?Set MATTOOLS_LICENSE_EVIDENCE_SHA256.}"; \
		: "$${MATTOOLS_LICENSE_EVIDENCE_ROLE:?Set MATTOOLS_LICENSE_EVIDENCE_ROLE relative to the evidence root.}"; \
		set -- --role "license_evidence=$${MATTOOLS_LICENSE_EVIDENCE_ROLE}" --license-evidence-sha256 "$${MATTOOLS_LICENSE_EVIDENCE_SHA256}"; \
	else \
		test -z "$${MATTOOLS_LICENSE_EVIDENCE_SHA256:-}"; \
		test -z "$${MATTOOLS_LICENSE_EVIDENCE_ROLE:-}"; \
	fi; \
	uv run --isolated --no-project --python 3.11 python scripts/materials_promotion_envelope.py create \
		--evidence-root "$${MATERIALS_EVIDENCE_ROOT}" \
		--evidence-root-manifest "$${MATERIALS_EVIDENCE_ROOT_MANIFEST}" \
		--envelope "$${MATERIALS_RELEASE_ENVELOPE}" \
		--role "readiness_report=$${MATERIALS_READINESS_REPORT_ROLE}" \
		--role "readiness_manifest=$${MATERIALS_READINESS_MANIFEST_ROLE}" \
		--role "deterministic_report=$${MATERIALS_DETERMINISTIC_REPORT_ROLE}" \
		--role "production_parity_report=$${MATERIALS_PRODUCTION_PARITY_REPORT_ROLE}" \
		--role "calphad_ledger_report=$${MATERIALS_CALPHAD_LEDGER_REPORT_ROLE}" \
		--role "calphad_cross_language_report=$${MATERIALS_CALPHAD_CROSS_LANGUAGE_REPORT_ROLE}" \
		--role "calphad_cross_language_manifest=$${MATERIALS_CALPHAD_CROSS_LANGUAGE_MANIFEST_ROLE}" \
		--role "mattools_report=$${MATTOOLS_REPORT_ROLE}" \
		--role "mattools_manifest=$${MATTOOLS_MANIFEST_ROLE}" \
		--role "live_trace:1=$${MATERIALS_LIVE_TRACE_ROLE}" \
		--role "release_tarball=$${MATERIALS_RELEASE_TARBALL_ROLE}" \
		--role "release_manifest=$${MATERIALS_RELEASE_MANIFEST_ROLE}" \
		"$$@" \
		--repository amilworks/ultra --repository-id 1204778765 --owner-id 22850980 \
		--source-git-sha "$${MATERIALS_EXPECTED_GIT_SHA}" --source-ref refs/heads/main \
		--workflow-path .github/workflows/materials-production-qualification.yml \
		--workflow-file "$${MATERIALS_EXPECTED_WORKFLOW_FILE}" \
		--workflow-signer-digest "$${MATERIALS_WORKFLOW_SIGNER_DIGEST}" \
		--run-id "$${MATERIALS_GITHUB_RUN_ID}" --run-attempt "$${MATERIALS_GITHUB_RUN_ATTEMPT}" \
		--environment materials-production-qualification --event-name workflow_dispatch \
		--runtime-oci-digest "$${MATERIALS_RUNTIME_OCI_DIGEST}" \
		--runtime-config-id "$${MATERIALS_EXPECTED_RUNTIME_IMAGE}" \
		--domain-image-id "$${MATERIALS_EXPECTED_DOMAIN_IMAGE}" \
		--evaluator-image-id "$${MATERIALS_EXPECTED_EVALUATOR_IMAGE}" \
		--license-basis "$${MATTOOLS_LICENSE_BASIS}" \
		--restricted-store-locator-sha256 "$${MATERIALS_RESTRICTED_STORE_LOCATOR_SHA256}"

materials-promotion-envelope-verify-root: ## Rehash the exact restricted materials evidence closure
	@: "$${MATERIALS_EVIDENCE_ROOT:?Set MATERIALS_EVIDENCE_ROOT.}"
	@: "$${MATERIALS_EVIDENCE_ROOT_MANIFEST:?Set MATERIALS_EVIDENCE_ROOT_MANIFEST.}"
	uv run --isolated --no-project --python 3.11 python scripts/materials_promotion_envelope.py verify-root \
		--evidence-root "$${MATERIALS_EVIDENCE_ROOT}" \
		--evidence-root-manifest "$${MATERIALS_EVIDENCE_ROOT_MANIFEST}"

materials-promotion-attestation-verify: ## Emit full readiness only after closure and exact GitHub/Sigstore verification
	@: "$${MATERIALS_EVIDENCE_ROOT:?Set MATERIALS_EVIDENCE_ROOT.}"
	@: "$${MATERIALS_EVIDENCE_ROOT_MANIFEST:?Set MATERIALS_EVIDENCE_ROOT_MANIFEST.}"
	@: "$${MATERIALS_RELEASE_ENVELOPE:?Set MATERIALS_RELEASE_ENVELOPE.}"
	@: "$${MATERIALS_ATTESTATION_BUNDLE:?Set MATERIALS_ATTESTATION_BUNDLE.}"
	@: "$${MATERIALS_FINAL_VERIFICATION_REPORT:?Set MATERIALS_FINAL_VERIFICATION_REPORT.}"
	@: "$${MATERIALS_EXPECTED_GIT_SHA:?Set MATERIALS_EXPECTED_GIT_SHA.}"
	@: "$${MATERIALS_GITHUB_RUN_ID:?Set MATERIALS_GITHUB_RUN_ID.}"
	@: "$${MATERIALS_GITHUB_RUN_ATTEMPT:?Set MATERIALS_GITHUB_RUN_ATTEMPT.}"
	@: "$${MATERIALS_EXPECTED_WORKFLOW_SHA256:?Set MATERIALS_EXPECTED_WORKFLOW_SHA256.}"
	uv run --isolated --no-project --python 3.11 python scripts/materials_promotion_envelope.py verify-attestation \
		--evidence-root "$${MATERIALS_EVIDENCE_ROOT}" \
		--evidence-root-manifest "$${MATERIALS_EVIDENCE_ROOT_MANIFEST}" \
		--envelope "$${MATERIALS_RELEASE_ENVELOPE}" \
		--bundle "$${MATERIALS_ATTESTATION_BUNDLE}" \
		--output "$${MATERIALS_FINAL_VERIFICATION_REPORT}" \
		--repository amilworks/ultra --repository-id 1204778765 --owner-id 22850980 \
		--signer-repo amilworks/ultra \
		--signer-workflow amilworks/ultra/.github/workflows/materials-production-qualification.yml \
		--signer-digest "$${MATERIALS_EXPECTED_GIT_SHA}" \
		--source-digest "$${MATERIALS_EXPECTED_GIT_SHA}" --source-ref refs/heads/main \
		--expected-run-id "$${MATERIALS_GITHUB_RUN_ID}" \
		--expected-run-attempt "$${MATERIALS_GITHUB_RUN_ATTEMPT}" \
		--expected-environment materials-production-qualification \
		--expected-event-name workflow_dispatch \
		--expected-workflow-sha256 "$${MATERIALS_EXPECTED_WORKFLOW_SHA256}"

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

mattools-evaluator-build: ## Build and metadata-verify the reviewed MatTools evaluator reconstruction
	@test -n "$(MATTOOLS_BENCHMARK_ROOT)" || { echo "Set MATTOOLS_BENCHMARK_ROOT to the pinned checkout." >&2; exit 1; }
	PYTHONDONTWRITEBYTECODE=1 uv run python scripts/build_mattools_evaluator.py build \
		--benchmark-root "$(MATTOOLS_BENCHMARK_ROOT)"

mattools-evaluator-verify: ## Verify evaluator image, complete lock, source manifest, and no-task evidence
	@test -n "$(MATTOOLS_BENCHMARK_ROOT)" || { echo "Set MATTOOLS_BENCHMARK_ROOT to the pinned checkout." >&2; exit 1; }
	PYTHONDONTWRITEBYTECODE=1 uv run python scripts/build_mattools_evaluator.py verify \
		--benchmark-root "$(MATTOOLS_BENCHMARK_ROOT)" \
		--lock "$(MATTOOLS_EVALUATOR_ENV_LOCK)"

mattools-promotion-test: ## Run lean MatTools parsing, scoring, isolation, and resume tests
	PYTHONDONTWRITEBYTECODE=1 uv run python -m py_compile scripts/build_mattools_evaluator.py scripts/mattools_promotion_gate.py scripts/mattools_safe_parser.py scripts/mattools_runner_wrapper.py scripts/mattools_strict_shadow.py scripts/mattools_semantic_repairs.py
	PYTHONDONTWRITEBYTECODE=1 uv run --extra dev pytest -p no:cacheprovider tests/test_mattools_evaluator_image.py tests/test_mattools_promotion_gate.py tests/test_mattools_safe_parser.py tests/test_mattools_semantic_repairs.py -q

mattools-promotion-inspect: ## Verify an explicit pinned official MatTools snapshot
	@test -n "$(MATTOOLS_BENCHMARK_ROOT)" || { echo "Set MATTOOLS_BENCHMARK_ROOT to the official pinned checkout." >&2; exit 1; }
	@uv run python scripts/mattools_promotion_gate.py inspect \
		--benchmark-root "$(MATTOOLS_BENCHMARK_ROOT)"

mattools-promotion-diagnostic: ## Evaluate one complete non-promotable MatTools diagnostic trial
	@test -n "$(MATTOOLS_BENCHMARK_ROOT)" || { echo "Set MATTOOLS_BENCHMARK_ROOT." >&2; exit 1; }
	@test -n "$(MATTOOLS_OUTPUT_DIR)" || { echo "Set MATTOOLS_OUTPUT_DIR." >&2; exit 1; }
	@test -n "$(ULTRA_RUNTIME_IMAGE_DIGEST)" || { echo "Set ULTRA_RUNTIME_IMAGE_DIGEST=sha256:..." >&2; exit 1; }
	@test -n "$(ULTRA_RUNTIME_PYMATGEN_VERSION)" || { echo "Set ULTRA_RUNTIME_PYMATGEN_VERSION." >&2; exit 1; }
	@test -n "$(ULTRA_RUNTIME_DEFECTS_VERSION)" || { echo "Set ULTRA_RUNTIME_DEFECTS_VERSION." >&2; exit 1; }
	@test -n "$(ULTRA_MODEL_ID)" || { echo "Set ULTRA_MODEL_ID." >&2; exit 1; }
	@test -n "$(ULTRA_PROVIDER_ID)" || { echo "Set ULTRA_PROVIDER_ID." >&2; exit 1; }
	@test -n "$(MATTOOLS_USE_PURPOSE)" || { echo "Set MATTOOLS_USE_PURPOSE after reviewing benchmark licenses." >&2; exit 1; }
	@test -n "$(MATTOOLS_LICENSE_BASIS)" || { echo "Set MATTOOLS_LICENSE_BASIS=noncommercial or separately_licensed." >&2; exit 1; }
	@if [ "$(MATTOOLS_LICENSE_BASIS)" = "separately_licensed" ] && [ -z "$(MATTOOLS_LICENSE_EVIDENCE_SHA256)" ]; then echo "Set MATTOOLS_LICENSE_EVIDENCE_SHA256 for separately licensed use." >&2; exit 1; fi
	@test -n "$(MATTOOLS_SANDBOX_ATTESTATION)" || { echo "Set MATTOOLS_SANDBOX_ATTESTATION to the evaluator policy JSON." >&2; exit 1; }
	@test -n "$(MATTOOLS_SANDBOX_SIGNATURE)" || { echo "Set MATTOOLS_SANDBOX_SIGNATURE to its detached signature." >&2; exit 1; }
	@test -n "$(MATTOOLS_SANDBOX_PUBLIC_KEY)" || { echo "Set MATTOOLS_SANDBOX_PUBLIC_KEY to the operator public key." >&2; exit 1; }
	@test -n "$(MATTOOLS_EVALUATOR_IMAGE_ID)" || { echo "Set MATTOOLS_EVALUATOR_IMAGE_ID=sha256:..." >&2; exit 1; }
	@test -n "$(MATTOOLS_EVALUATOR_ENV_LOCK)" || { echo "Set MATTOOLS_EVALUATOR_ENV_LOCK to the reviewed tracked JSON lock." >&2; exit 1; }
	uv run python scripts/mattools_promotion_gate.py run \
		--benchmark-root "$(MATTOOLS_BENCHMARK_ROOT)" \
		--output-dir "$(MATTOOLS_OUTPUT_DIR)" \
		--base-url "$${ULTRA_LIVE_TRACE_BASE_URL:-http://127.0.0.1:8000}" \
		--model-id "$(ULTRA_MODEL_ID)" \
		--provider-id "$(ULTRA_PROVIDER_ID)" \
		--runtime-image-digest "$(ULTRA_RUNTIME_IMAGE_DIGEST)" \
		--runtime-pymatgen-version "$(ULTRA_RUNTIME_PYMATGEN_VERSION)" \
		--runtime-defects-version "$(ULTRA_RUNTIME_DEFECTS_VERSION)" \
		--trials 1 \
		--concurrency "$(or $(MATTOOLS_CONCURRENCY),1)" \
		--benchmark-license-basis "$(MATTOOLS_LICENSE_BASIS)" \
		--benchmark-license-evidence-sha256 "$(MATTOOLS_LICENSE_EVIDENCE_SHA256)" \
		--benchmark-use-purpose "$(MATTOOLS_USE_PURPOSE)" \
		--accept-benchmark-license \
		--sandbox-policy-attestation "$(MATTOOLS_SANDBOX_ATTESTATION)" \
		--sandbox-attestation-signature "$(MATTOOLS_SANDBOX_SIGNATURE)" \
		--sandbox-attestation-public-key "$(MATTOOLS_SANDBOX_PUBLIC_KEY)" \
		--expected-evaluator-image-id "$(MATTOOLS_EVALUATOR_IMAGE_ID)" \
		--evaluator-environment-lock "$(MATTOOLS_EVALUATOR_ENV_LOCK)" \
		--diagnostic-evaluate

mattools-promotion-gate: ## Run three complete MatTools trials and the pinned independent evaluator
	@test -n "$(MATTOOLS_BENCHMARK_ROOT)" || { echo "Set MATTOOLS_BENCHMARK_ROOT." >&2; exit 1; }
	@test -n "$(MATTOOLS_OUTPUT_DIR)" || { echo "Set MATTOOLS_OUTPUT_DIR." >&2; exit 1; }
	@test -n "$(ULTRA_RUNTIME_IMAGE_DIGEST)" || { echo "Set ULTRA_RUNTIME_IMAGE_DIGEST=sha256:..." >&2; exit 1; }
	@test -n "$(ULTRA_RUNTIME_PYMATGEN_VERSION)" || { echo "Set ULTRA_RUNTIME_PYMATGEN_VERSION." >&2; exit 1; }
	@test -n "$(ULTRA_RUNTIME_DEFECTS_VERSION)" || { echo "Set ULTRA_RUNTIME_DEFECTS_VERSION." >&2; exit 1; }
	@test -n "$(ULTRA_MODEL_ID)" || { echo "Set ULTRA_MODEL_ID." >&2; exit 1; }
	@test -n "$(ULTRA_PROVIDER_ID)" || { echo "Set ULTRA_PROVIDER_ID." >&2; exit 1; }
	@test -n "$(MATTOOLS_USE_PURPOSE)" || { echo "Set MATTOOLS_USE_PURPOSE after reviewing benchmark licenses." >&2; exit 1; }
	@test -n "$(MATTOOLS_LICENSE_BASIS)" || { echo "Set MATTOOLS_LICENSE_BASIS=noncommercial or separately_licensed." >&2; exit 1; }
	@if [ "$(MATTOOLS_LICENSE_BASIS)" = "separately_licensed" ] && [ -z "$(MATTOOLS_LICENSE_EVIDENCE_SHA256)" ]; then echo "Set MATTOOLS_LICENSE_EVIDENCE_SHA256 for separately licensed use." >&2; exit 1; fi
	@test -n "$(MATTOOLS_SANDBOX_ATTESTATION)" || { echo "Set MATTOOLS_SANDBOX_ATTESTATION to the evaluator policy JSON." >&2; exit 1; }
	@test -n "$(MATTOOLS_SANDBOX_SIGNATURE)" || { echo "Set MATTOOLS_SANDBOX_SIGNATURE to its detached signature." >&2; exit 1; }
	@test -n "$(MATTOOLS_SANDBOX_PUBLIC_KEY)" || { echo "Set MATTOOLS_SANDBOX_PUBLIC_KEY to the operator public key." >&2; exit 1; }
	@test -n "$(MATTOOLS_EVALUATOR_IMAGE_ID)" || { echo "Set MATTOOLS_EVALUATOR_IMAGE_ID=sha256:..." >&2; exit 1; }
	@test -n "$(MATTOOLS_EVALUATOR_ENV_LOCK)" || { echo "Set MATTOOLS_EVALUATOR_ENV_LOCK to the reviewed tracked JSON lock." >&2; exit 1; }
	uv run python scripts/mattools_promotion_gate.py run \
		--benchmark-root "$(MATTOOLS_BENCHMARK_ROOT)" \
		--output-dir "$(MATTOOLS_OUTPUT_DIR)" \
		--base-url "$${ULTRA_LIVE_TRACE_BASE_URL:-http://127.0.0.1:8000}" \
		--model-id "$(ULTRA_MODEL_ID)" \
		--provider-id "$(ULTRA_PROVIDER_ID)" \
		--runtime-image-digest "$(ULTRA_RUNTIME_IMAGE_DIGEST)" \
		--runtime-pymatgen-version "$(ULTRA_RUNTIME_PYMATGEN_VERSION)" \
		--runtime-defects-version "$(ULTRA_RUNTIME_DEFECTS_VERSION)" \
		--concurrency "$(or $(MATTOOLS_CONCURRENCY),1)" \
		--benchmark-license-basis "$(MATTOOLS_LICENSE_BASIS)" \
		--benchmark-license-evidence-sha256 "$(MATTOOLS_LICENSE_EVIDENCE_SHA256)" \
		--benchmark-use-purpose "$(MATTOOLS_USE_PURPOSE)" \
		--accept-benchmark-license \
		--sandbox-policy-attestation "$(MATTOOLS_SANDBOX_ATTESTATION)" \
		--sandbox-attestation-signature "$(MATTOOLS_SANDBOX_SIGNATURE)" \
		--sandbox-attestation-public-key "$(MATTOOLS_SANDBOX_PUBLIC_KEY)" \
		--expected-evaluator-image-id "$(MATTOOLS_EVALUATOR_IMAGE_ID)" \
		--evaluator-environment-lock "$(MATTOOLS_EVALUATOR_ENV_LOCK)"
