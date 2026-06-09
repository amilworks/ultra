from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.context import AgentRunContext
from ultra_deepagents.rarespot.config import RareSpotConfig


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
        "ULTRA_DEEPAGENTS_SANDBOX_TIMEOUT_SECONDS",
        "ULTRA_DEEPAGENTS_SANDBOX_OUTPUT_LIMIT_BYTES",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = RuntimeSettings.from_env()

    assert settings.request_timeout_seconds == 0.0
    assert settings.sandbox_cpus == 0.0
    assert settings.sandbox_memory == ""
    assert settings.sandbox_pids_limit == 0
    assert settings.sandbox_timeout_seconds == 0
    assert settings.sandbox_output_limit_bytes == 0


def test_runtime_settings_default_to_no_model_stream_idle_timeout(monkeypatch):
    monkeypatch.delenv("ULTRA_DEEPAGENTS_MODEL_STREAM_IDLE_TIMEOUT_SECONDS", raising=False)

    settings = RuntimeSettings.from_env()

    assert settings.model_stream_idle_timeout_seconds == 0.0


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


def test_runtime_settings_default_to_higher_autonomous_continuation_guard(monkeypatch):
    monkeypatch.delenv("ULTRA_DEEPAGENTS_COMPLETION_MAX_CONTINUATIONS", raising=False)

    settings = RuntimeSettings.from_env()

    assert settings.completion_max_continuations == 8


def test_runtime_settings_default_to_two_worker_slots(monkeypatch):
    monkeypatch.delenv("ULTRA_DEEPAGENTS_WORKER_MAX_CONCURRENCY", raising=False)

    settings = RuntimeSettings.from_env()

    assert settings.worker_max_concurrency == 2


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


def test_runtime_settings_load_rarespot_worker_identity(monkeypatch):
    monkeypatch.setenv("ULTRA_RARESPOT_WORKER_ID", "rarespot-prod-1")
    monkeypatch.setenv("ULTRA_RARESPOT_WORKER_KIND", "rarespot-gpu")

    settings = RuntimeSettings.from_env()

    assert settings.rarespot_worker_id == "rarespot-prod-1"
    assert settings.rarespot_worker_kind == "rarespot-gpu"


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

    assert settings.rarespot_tool_enabled is True
    assert settings.rarespot_tile_overlap == 0.25
    assert settings.rarespot_conf_threshold == 0.25
    assert settings.rarespot_iou_threshold == 0.45
    assert settings.rarespot_imgsz == 512
    assert settings.rarespot_nats_jobs_subject == "ultra.runs.rarespot.jobs"
    assert settings.rarespot_nats_ack_wait_seconds == 120.0
    assert settings.rarespot_nats_ack_progress_interval_seconds == 30.0
    assert settings.rarespot_weights_path == "data/models/yolo/RareSpotWeights.pt"
    assert (
        settings.rarespot_nats_ack_progress_interval_seconds
        < settings.rarespot_nats_ack_wait_seconds
    )

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
