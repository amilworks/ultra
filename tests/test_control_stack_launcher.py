from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_control_stack_launcher_runs_durable_go_nats_postgres_path():
    script = ROOT / "scripts" / "restart_control_stack.sh"

    assert script.exists(), "production-like Go/NATS/Postgres launcher is missing"
    assert os.access(script, os.X_OK), "launcher must be executable"

    text = script.read_text(encoding="utf-8")
    for required in [
        "ULTRA_CONTROL_DATABASE_URL",
        "ULTRA_CONTROL_NATS_URL",
        "go run ./cmd/ultra-control migrate",
        "go run ./cmd/ultra-control",
        "python -m ultra_deepagents.nats_worker",
        "pnpm exec vite",
        "store_backend",
        "postgres",
    ]:
        assert required in text


def test_control_stack_launcher_tolerates_existing_named_postgres_container():
    script = ROOT / "scripts" / "restart_control_stack.sh"
    text = script.read_text(encoding="utf-8")

    assert "docker inspect bisque-ultra-postgres" in text
    assert "docker start bisque-ultra-postgres" in text
    assert "CREATE DATABASE" in text


def test_control_stack_launcher_keeps_background_services_alive_after_exit():
    script = ROOT / "scripts" / "restart_control_stack.sh"
    text = script.read_text(encoding="utf-8")

    assert "screen -dmS" in text
    assert "nohup" in text


def test_control_stack_launcher_does_not_serialize_secrets_into_executable_runner():
    script = ROOT / "scripts" / "restart_control_stack.sh"
    text = script.read_text(encoding="utf-8")

    assert 'env_file="$STATE_DIR/run-$session_name.env"' in text
    assert "umask 077" in text
    assert "printf 'export %s=%q\\n'" in text
    assert "printf 'source %q\\n' \"$env_file\"" in text
    assert "printf 'rm -f %q\\n' \"$env_file\"" in text
    assert 'chmod 0600 "$env_file"' in text
    assert 'chmod 0700 "$runner"' in text
    assert 'chmod +x "$runner"' not in text


def test_control_stack_launcher_uses_long_model_idle_watchdog():
    script = ROOT / "scripts" / "restart_control_stack.sh"
    text = script.read_text(encoding="utf-8")

    assert "ULTRA_DEEPAGENTS_MODEL_STREAM_IDLE_TIMEOUT_SECONDS:-3600" in text
    assert "ULTRA_DEEPAGENTS_MODEL_STREAM_IDLE_TIMEOUT_SECONDS:-240" not in text


def test_control_stack_launcher_defaults_deepagents_worker_to_vllm_capacity():
    script = ROOT / "scripts" / "restart_control_stack.sh"
    text = script.read_text(encoding="utf-8")

    assert "ULTRA_DEEPAGENTS_WORKER_MAX_CONCURRENCY:-64" in text
    assert "ULTRA_DEEPAGENTS_WORKER_MAX_CONCURRENCY:-2" not in text


def test_control_stack_launcher_has_finite_local_sandbox_defaults():
    script = ROOT / "scripts" / "restart_control_stack.sh"
    text = script.read_text(encoding="utf-8")

    assert 'ULTRA_DEEPAGENTS_SANDBOX_CPUS="${ULTRA_DEEPAGENTS_SANDBOX_CPUS:-2}"' in text
    assert (
        'ULTRA_DEEPAGENTS_SANDBOX_MEMORY="${ULTRA_DEEPAGENTS_SANDBOX_MEMORY:-4g}"'
        in text
    )
    assert (
        'ULTRA_DEEPAGENTS_SANDBOX_PIDS_LIMIT="${ULTRA_DEEPAGENTS_SANDBOX_PIDS_LIMIT:-512}"'
        in text
    )
    assert (
        'ULTRA_DEEPAGENTS_SANDBOX_OUTPUT_LIMIT_BYTES="${ULTRA_DEEPAGENTS_SANDBOX_OUTPUT_LIMIT_BYTES:-52428800}"'
        in text
    )


def test_control_stack_launcher_reports_model_endpoint_health():
    script = ROOT / "scripts" / "restart_control_stack.sh"
    text = script.read_text(encoding="utf-8")

    assert "check_model_endpoint" in text
    assert "Model endpoint responding" in text
    assert "WARNING: model endpoint not responding" in text


def test_control_stack_launcher_stops_legacy_screen_sessions_by_full_id():
    script = ROOT / "scripts" / "restart_control_stack.sh"
    text = script.read_text(encoding="utf-8")

    assert "awk '{print $1}'" in text
    assert 'screen -S "$screen_session" -X quit' in text


def test_control_stack_launcher_stops_stale_repo_worker_processes():
    script = ROOT / "scripts" / "restart_control_stack.sh"
    text = script.read_text(encoding="utf-8")

    assert "kill_repo_python_module" in text
    assert '"ultra_deepagents.nats_worker"' in text
    assert 'lsof -a -p "$pid" -d cwd' in text


def test_makefile_exposes_control_stack_operations():
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert "restart-control-stack:" in makefile
    assert "status-control-stack:" in makefile
    assert "stop-control-stack:" in makefile
