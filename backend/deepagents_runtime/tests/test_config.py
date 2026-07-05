import json

import pytest
from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.context import AgentRunContext
from ultra_deepagents.rarespot.config import RareSpotConfig


@pytest.fixture(autouse=True)
def _enable_async_subagents(monkeypatch):
    """Async subagents are off unless explicitly enabled. Config-parsing tests
    that exercise async spec validation need the kill switch on; tests asserting
    the off/kill-switch behavior delenv it themselves."""
    monkeypatch.setenv("ULTRA_DEEPAGENTS_ENABLE_ASYNC_SUBAGENTS", "1")


def test_runtime_settings_load_vllm_defaults(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8003/v1")
    monkeypatch.setenv("OPENAI_MODEL", "deepseek_v4")

    settings = RuntimeSettings.from_env()

    assert settings.openai_base_url == "http://127.0.0.1:8003/v1"
    assert settings.openai_model == "deepseek_v4"
    assert settings.openai_api_key == "EMPTY"
    assert settings.sandbox_network == "none"
    assert settings.model_supports_multimodal is False


def test_runtime_settings_can_enable_multimodal_model(monkeypatch):
    monkeypatch.setenv("ULTRA_DEEPAGENTS_MODEL_SUPPORTS_MULTIMODAL", "true")

    settings = RuntimeSettings.from_env()

    assert settings.model_supports_multimodal is True


def test_runtime_settings_default_to_no_hidden_long_run_caps(monkeypatch):
    for name in (
        "ULTRA_DEEPAGENTS_TIMEOUT_SECONDS",
        "ULTRA_DEEPAGENTS_SANDBOX_CPUS",
        "ULTRA_DEEPAGENTS_SANDBOX_MEMORY",
        "ULTRA_DEEPAGENTS_SANDBOX_PIDS_LIMIT",
        "ULTRA_DEEPAGENTS_SANDBOX_SHM_SIZE",
        "ULTRA_DEEPAGENTS_SANDBOX_GPUS",
        "ULTRA_DEEPAGENTS_SANDBOX_TIMEOUT_SECONDS",
        "ULTRA_DEEPAGENTS_SANDBOX_OUTPUT_LIMIT_BYTES",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = RuntimeSettings.from_env()

    assert settings.request_timeout_seconds == 0.0
    assert settings.sandbox_cpus == 0.0
    assert settings.sandbox_memory == ""
    assert settings.sandbox_pids_limit == 0
    # shm/gpu are opt-in: empty by default (no GPU in-sandbox; Docker-default /dev/shm).
    assert settings.sandbox_shm_size == ""
    assert settings.sandbox_gpus == ""
    # A generous finite ceiling (6h) is armed by default so a hung/zombie sandbox call
    # cannot await forever; long scientific batch work is allowed up to it. (Was 0 = unbounded.)
    assert settings.sandbox_timeout_seconds == 21600
    assert settings.sandbox_output_limit_bytes == 0


def test_runtime_settings_load_sandbox_shm_and_gpus(monkeypatch):
    monkeypatch.setenv("ULTRA_DEEPAGENTS_SANDBOX_SHM_SIZE", "8g")
    monkeypatch.setenv("ULTRA_DEEPAGENTS_SANDBOX_GPUS", "all")

    settings = RuntimeSettings.from_env()

    assert settings.sandbox_shm_size == "8g"
    assert settings.sandbox_gpus == "all"


def test_runtime_settings_arm_idle_watchdog_by_default(monkeypatch):
    # The idle/stall watchdog is now ARMED by default (1h) so a stalled tool call cannot
    # wedge a run/worker forever (it did, for 1h43m, when this defaulted to 0). An explicit
    # env 0 is still honored as the operator escape hatch.
    monkeypatch.delenv("ULTRA_DEEPAGENTS_MODEL_STREAM_IDLE_TIMEOUT_SECONDS", raising=False)
    assert RuntimeSettings.from_env().model_stream_idle_timeout_seconds == 3600.0

    monkeypatch.setenv("ULTRA_DEEPAGENTS_MODEL_STREAM_IDLE_TIMEOUT_SECONDS", "0")
    assert RuntimeSettings.from_env().model_stream_idle_timeout_seconds == 0.0


def test_runtime_settings_load_model_stream_liveness_guard(monkeypatch):
    monkeypatch.setenv("ULTRA_DEEPAGENTS_MODEL_STREAM_IDLE_TIMEOUT_SECONDS", "12.5")

    settings = RuntimeSettings.from_env()

    assert settings.model_stream_idle_timeout_seconds == 12.5


def test_runtime_settings_default_to_model_stream_idle_recoveries(monkeypatch):
    monkeypatch.delenv("ULTRA_DEEPAGENTS_MODEL_STREAM_IDLE_MAX_RECOVERIES", raising=False)

    settings = RuntimeSettings.from_env()

    assert settings.model_stream_idle_max_recoveries == 2


def test_runtime_settings_load_model_stream_idle_recoveries(monkeypatch):
    monkeypatch.setenv("ULTRA_DEEPAGENTS_MODEL_STREAM_IDLE_MAX_RECOVERIES", "4")

    settings = RuntimeSettings.from_env()

    assert settings.model_stream_idle_max_recoveries == 4


def test_runtime_settings_default_progress_stall_guard_armed(monkeypatch):
    monkeypatch.delenv("ULTRA_DEEPAGENTS_PROGRESS_STALL_THRESHOLD", raising=False)
    monkeypatch.delenv("ULTRA_DEEPAGENTS_PROGRESS_STALL_MAX_RECOVERIES", raising=False)

    settings = RuntimeSettings.from_env()

    assert settings.progress_stall_threshold == 12
    assert settings.progress_stall_max_recoveries == 2


def test_runtime_settings_progress_stall_guard_env_overrides(monkeypatch):
    monkeypatch.setenv("ULTRA_DEEPAGENTS_PROGRESS_STALL_THRESHOLD", "0")
    monkeypatch.setenv("ULTRA_DEEPAGENTS_PROGRESS_STALL_MAX_RECOVERIES", "5")

    settings = RuntimeSettings.from_env()

    assert settings.progress_stall_threshold == 0  # 0 disables the guard
    assert settings.progress_stall_max_recoveries == 5


def test_runtime_settings_builder_disabled_by_default_opt_in(monkeypatch):
    # Cleanup candidate: OFF by default after a live A/B; opt back in with env=1.
    monkeypatch.delenv("ULTRA_DEEPAGENTS_BUILDER_ENABLED", raising=False)
    assert RuntimeSettings.from_env().builder_enabled is False

    monkeypatch.setenv("ULTRA_DEEPAGENTS_BUILDER_ENABLED", "1")
    assert RuntimeSettings.from_env().builder_enabled is True


def test_runtime_settings_default_to_higher_autonomous_continuation_guard(monkeypatch):
    monkeypatch.delenv("ULTRA_DEEPAGENTS_COMPLETION_MAX_CONTINUATIONS", raising=False)

    settings = RuntimeSettings.from_env()

    assert settings.completion_max_continuations == 8


def test_runtime_settings_default_to_vllm_worker_slot_capacity(monkeypatch):
    monkeypatch.delenv("ULTRA_DEEPAGENTS_WORKER_MAX_CONCURRENCY", raising=False)

    settings = RuntimeSettings.from_env()

    assert settings.worker_max_concurrency == 64


def test_runtime_settings_load_worker_max_concurrency(monkeypatch):
    monkeypatch.setenv("ULTRA_DEEPAGENTS_WORKER_MAX_CONCURRENCY", "4")

    settings = RuntimeSettings.from_env()

    assert settings.worker_max_concurrency == 4


def test_runtime_settings_default_to_high_langgraph_recursion_limit(monkeypatch):
    monkeypatch.delenv("ULTRA_DEEPAGENTS_RECURSION_LIMIT", raising=False)

    settings = RuntimeSettings.from_env()

    assert settings.langgraph_recursion_limit == 1000


def test_runtime_settings_load_langgraph_recursion_limit(monkeypatch):
    monkeypatch.setenv("ULTRA_DEEPAGENTS_RECURSION_LIMIT", "2500")

    settings = RuntimeSettings.from_env()

    assert settings.langgraph_recursion_limit == 2500


def test_runtime_settings_load_completion_continuation_guard(monkeypatch):
    monkeypatch.setenv("ULTRA_DEEPAGENTS_COMPLETION_MAX_CONTINUATIONS", "6")

    settings = RuntimeSettings.from_env()

    assert settings.completion_max_continuations == 6


def test_runtime_settings_default_to_no_async_subagents(monkeypatch):
    monkeypatch.delenv("ULTRA_DEEPAGENTS_ASYNC_SUBAGENTS_JSON", raising=False)

    settings = RuntimeSettings.from_env()

    assert settings.async_subagents == ()


def test_runtime_settings_load_async_subagents_from_json(monkeypatch):
    monkeypatch.setenv(
        "ULTRA_DEEPAGENTS_ASYNC_SUBAGENTS_JSON",
        json.dumps(
            [
                {
                    "name": "remote-training-runner",
                    "description": "Runs long model training jobs on a remote LangGraph server.",
                    "graph_id": "ultra-training-agent",
                    "url": "https://langgraph.example.test",
                    "headers": {"X-Ultra-Async-Token": "secret-ref"},
                }
            ]
        ),
    )

    settings = RuntimeSettings.from_env()

    assert settings.async_subagents == (
        {
            "name": "remote-training-runner",
            "description": "Runs long model training jobs on a remote LangGraph server.",
            "graph_id": "ultra-training-agent",
            "url": "https://langgraph.example.test",
            "headers": {"X-Ultra-Async-Token": "secret-ref"},
        },
    )


def test_runtime_settings_reject_async_subagents_without_required_fields(monkeypatch):
    monkeypatch.setenv(
        "ULTRA_DEEPAGENTS_ASYNC_SUBAGENTS_JSON",
        json.dumps([{"name": "remote-training-runner", "graph_id": "ultra-training-agent"}]),
    )

    with pytest.raises(ValueError, match="description"):
        RuntimeSettings.from_env()


def test_runtime_settings_reject_duplicate_async_subagent_names(monkeypatch):
    monkeypatch.setenv(
        "ULTRA_DEEPAGENTS_ASYNC_SUBAGENTS_JSON",
        json.dumps(
            [
                {
                    "name": "remote-training-runner",
                    "description": "Runs long model training jobs.",
                    "graph_id": "ultra-training-agent",
                },
                {
                    "name": "remote-training-runner",
                    "description": "Runs another remote job.",
                    "graph_id": "ultra-other-agent",
                },
            ]
        ),
    )

    with pytest.raises(ValueError, match="Duplicate async subagent name"):
        RuntimeSettings.from_env()


def test_runtime_settings_reject_case_insensitive_duplicate_async_subagent_names(
    monkeypatch,
):
    monkeypatch.setenv(
        "ULTRA_DEEPAGENTS_ASYNC_SUBAGENTS_JSON",
        json.dumps(
            [
                {
                    "name": "remote-training-runner",
                    "description": "Runs long model training jobs.",
                    "graph_id": "ultra-training-agent",
                },
                {
                    "name": "Remote-Training-Runner",
                    "description": "Runs another remote job.",
                    "graph_id": "ultra-other-agent",
                },
            ]
        ),
    )

    with pytest.raises(ValueError, match="Duplicate async subagent name"):
        RuntimeSettings.from_env()


def test_runtime_settings_reject_case_insensitive_duplicate_async_subagent_headers(
    monkeypatch,
):
    monkeypatch.setenv(
        "ULTRA_DEEPAGENTS_ASYNC_SUBAGENTS_JSON",
        json.dumps(
            [
                {
                    "name": "remote-training-runner",
                    "description": "Runs long model training jobs.",
                    "graph_id": "ultra-training-agent",
                    "headers": {
                        "X-Auth-Scheme": "ultra-workos",
                        "x-auth-scheme": "langsmith",
                    },
                },
            ]
        ),
    )

    with pytest.raises(ValueError, match="Duplicate async subagent header name"):
        RuntimeSettings.from_env()


@pytest.mark.parametrize(
    "url",
    [
        "file:///tmp/langgraph.sock",
        "/tmp/langgraph.sock",
        "langgraph.example.test",
        "ftp://langgraph.example.test",
        "http:/langgraph.example.test",
    ],
)
def test_runtime_settings_reject_async_subagent_urls_that_are_not_http_endpoints(
    monkeypatch,
    url,
):
    monkeypatch.setenv(
        "ULTRA_DEEPAGENTS_ASYNC_SUBAGENTS_JSON",
        json.dumps(
            [
                {
                    "name": "remote-training-runner",
                    "description": "Runs long model training jobs.",
                    "graph_id": "ultra-training-agent",
                    "url": url,
                },
            ]
        ),
    )

    with pytest.raises(ValueError, match=r"url.*http:// or https://"):
        RuntimeSettings.from_env()


@pytest.mark.parametrize(
    "url",
    [
        "https://runner-token@langgraph.example.test",
        "https://ultra:secret@langgraph.example.test",
    ],
)
def test_runtime_settings_reject_async_subagent_urls_with_credentials(
    monkeypatch,
    url,
):
    monkeypatch.setenv(
        "ULTRA_DEEPAGENTS_ASYNC_SUBAGENTS_JSON",
        json.dumps(
            [
                {
                    "name": "remote-training-runner",
                    "description": "Runs long model training jobs.",
                    "graph_id": "ultra-training-agent",
                    "url": url,
                },
            ]
        ),
    )

    with pytest.raises(ValueError, match=r"url.*credentials"):
        RuntimeSettings.from_env()


@pytest.mark.parametrize("header_value", ["", 123, None])
def test_runtime_settings_reject_async_subagent_header_values_that_are_not_non_empty_strings(
    monkeypatch,
    header_value,
):
    monkeypatch.setenv(
        "ULTRA_DEEPAGENTS_ASYNC_SUBAGENTS_JSON",
        json.dumps(
            [
                {
                    "name": "remote-training-runner",
                    "description": "Runs long model training jobs.",
                    "graph_id": "ultra-training-agent",
                    "headers": {"Authorization": header_value},
                },
            ]
        ),
    )

    with pytest.raises(ValueError, match="headers.Authorization"):
        RuntimeSettings.from_env()


@pytest.mark.parametrize("field_name", ["name", "description", "graph_id", "url"])
def test_runtime_settings_reject_async_subagent_string_fields_with_non_string_values(
    monkeypatch,
    field_name,
):
    spec = {
        "name": "remote-training-runner",
        "description": "Runs long model training jobs.",
        "graph_id": "ultra-training-agent",
        "url": "https://langgraph.example.test",
    }
    spec[field_name] = 123
    monkeypatch.setenv(
        "ULTRA_DEEPAGENTS_ASYNC_SUBAGENTS_JSON",
        json.dumps([spec]),
    )

    with pytest.raises(ValueError, match=rf"{field_name}.*string"):
        RuntimeSettings.from_env()


def test_runtime_settings_load_control_status_poll_interval(monkeypatch):
    monkeypatch.setenv("ULTRA_DEEPAGENTS_CONTROL_STATUS_POLL_INTERVAL_SECONDS", "7.5")

    settings = RuntimeSettings.from_env()

    assert settings.control_status_poll_interval_seconds == 7.5


def test_runtime_settings_load_control_run_lease_options(monkeypatch):
    monkeypatch.setenv("ULTRA_DEEPAGENTS_WORKER_ID", "worker-prod-1")
    monkeypatch.setenv("ULTRA_DEEPAGENTS_WORKER_KIND", "deepagents-gpu")
    monkeypatch.setenv("ULTRA_DEEPAGENTS_WORKER_HEARTBEAT_INTERVAL_SECONDS", "12.5")
    monkeypatch.setenv("ULTRA_DEEPAGENTS_CONTROL_RUN_LEASE_TTL_SECONDS", "900")
    monkeypatch.setenv("ULTRA_DEEPAGENTS_REQUIRE_CONTROL_RUN_LEASE", "true")

    settings = RuntimeSettings.from_env()

    assert settings.worker_id == "worker-prod-1"
    assert settings.worker_kind == "deepagents-gpu"
    assert settings.worker_heartbeat_interval_seconds == 12.5
    assert settings.control_run_lease_ttl_seconds == 900.0
    assert settings.control_run_lease_required is True


def test_runtime_settings_load_data_agent_worker_and_queue_defaults(monkeypatch):
    monkeypatch.delenv("ULTRA_CONTROL_NATS_DATA_AGENT_JOBS_SUBJECT", raising=False)
    monkeypatch.delenv("ULTRA_CONTROL_NATS_DATA_AGENT_WORKER_DURABLE", raising=False)
    monkeypatch.setenv("ULTRA_DATA_AGENT_WORKER_ID", "data-agent-prod-1")
    monkeypatch.setenv("ULTRA_DATA_AGENT_WORKER_KIND", "data-agent-gpu")
    monkeypatch.setenv("ULTRA_DATA_AGENT_CONTROL_LEASE_TTL_SECONDS", "900")
    monkeypatch.setenv("ULTRA_DATA_AGENT_REQUIRE_CONTROL_LEASE", "true")

    settings = RuntimeSettings.from_env()

    assert settings.data_agent_nats_jobs_subject == "ultra.data_agent.jobs"
    assert settings.data_agent_nats_worker_durable == "ultra-data-agent-worker"
    assert settings.data_agent_nats_ack_wait_seconds == 300.0
    assert settings.data_agent_nats_ack_progress_interval_seconds == 60.0
    assert settings.data_agent_worker_id == "data-agent-prod-1"
    assert settings.data_agent_worker_kind == "data-agent-gpu"
    assert settings.data_agent_control_lease_ttl_seconds == 900.0
    assert settings.data_agent_control_lease_required is True


def test_runtime_settings_load_rarespot_defaults(monkeypatch):
    monkeypatch.delenv("YOLOV5_RARESPOT_WEIGHTS", raising=False)
    monkeypatch.delenv("PRAIRIE_FIXED_CONF_THRESHOLD", raising=False)
    monkeypatch.delenv("PRAIRIE_FIXED_IOU_THRESHOLD", raising=False)
    monkeypatch.delenv("PRAIRIE_FIXED_IMGSZ", raising=False)

    settings = RuntimeSettings.from_env()

    assert settings.rarespot_tile_overlap == 0.25
    assert settings.rarespot_conf_threshold == 0.25
    assert settings.rarespot_iou_threshold == 0.45
    assert settings.rarespot_imgsz == 512
    assert settings.rarespot_weights_path == "data/models/yolo/RareSpotWeights.pt"

    config = RareSpotConfig.from_settings(settings)
    assert config.tile_size == 512
    assert config.tile_overlap == 0.25
    assert config.stride == 384


def test_runtime_settings_allow_rarespot_tile_env_override(monkeypatch):
    monkeypatch.setenv("PRAIRIE_FIXED_IMGSZ", "1280")
    monkeypatch.setenv("ULTRA_RARESPOT_TILE_OVERLAP", "0.25")

    settings = RuntimeSettings.from_env()
    config = RareSpotConfig.from_settings(settings)

    assert settings.rarespot_imgsz == 1280
    assert config.tile_size == 1280
    assert config.stride == 960


def test_agent_run_context_payload_is_snake_case_and_scoped():
    context = AgentRunContext(
        assistant_id="ultra-research",
        org_id="allen",
        user_id="researcher-1",
        project_id="project-1",
        thread_id="thread-1",
        run_id="run-1",
        selected_file_ids=("file-1",),
        allowed_tool_packs=("workspace", "code"),
    )

    payload = context.to_payload()

    assert payload["run_id"] == "run-1"
    assert payload["selected_file_ids"] == ["file-1"]
    assert payload["allowed_tool_packs"] == ["workspace", "code"]


def test_async_subagents_require_explicit_enable_flag(monkeypatch):
    """Kill switch: a JSON value alone must NOT activate async subagents."""
    monkeypatch.setenv(
        "ULTRA_DEEPAGENTS_ASYNC_SUBAGENTS_JSON",
        json.dumps(
            [
                {
                    "name": "remote-runner",
                    "description": "Runs long jobs.",
                    "graph_id": "ultra-runner",
                    "url": "https://langgraph.example.test",
                }
            ]
        ),
    )
    # Enabled (autouse fixture sets the flag) -> the spec loads.
    assert len(RuntimeSettings.from_env().async_subagents) == 1
    # Disabled -> off regardless of JSON, with no parsing/validation.
    monkeypatch.delenv("ULTRA_DEEPAGENTS_ENABLE_ASYNC_SUBAGENTS", raising=False)
    assert RuntimeSettings.from_env().async_subagents == ()


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:2024",
        "http://127.0.0.1:2024",
        "https://10.0.0.5",
        "http://169.254.169.254/latest/meta-data/",
        "https://192.168.1.10:8123",
    ],
)
def test_runtime_settings_reject_private_async_subagent_urls(monkeypatch, url):
    """Server URL must not target a private/loopback/link-local host (the run
    context + task egress to it), unless the local-dev opt-out is set."""
    monkeypatch.setenv(
        "ULTRA_DEEPAGENTS_ASYNC_SUBAGENTS_JSON",
        json.dumps(
            [
                {
                    "name": "remote-runner",
                    "description": "Runs long jobs.",
                    "graph_id": "ultra-runner",
                    "url": url,
                }
            ]
        ),
    )
    with pytest.raises(ValueError, match=r"localhost/private/link-local"):
        RuntimeSettings.from_env()

    # Local-dev opt-out allows it.
    monkeypatch.setenv("ULTRA_DEEPAGENTS_ALLOW_PRIVATE_ASYNC_SUBAGENT_URL", "1")
    assert RuntimeSettings.from_env().async_subagents[0]["url"] == url
