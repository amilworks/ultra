from __future__ import annotations

import os
from dataclasses import dataclass
from socket import gethostname


@dataclass(frozen=True)
class RuntimeSettings:
    openai_base_url: str
    openai_model: str
    openai_api_key: str = "EMPTY"
    model_supports_multimodal: bool = False
    request_timeout_seconds: float = 0.0
    model_stream_idle_timeout_seconds: float = 0.0
    model_stream_idle_max_recoveries: int = 2
    max_retries: int = 1
    langgraph_recursion_limit: int = 1000
    sandbox_image: str = "bisque-ultra-codeexec:py311"
    sandbox_network: str = "none"
    sandbox_cpus: float = 0.0
    sandbox_memory: str = ""
    sandbox_pids_limit: int = 0
    sandbox_timeout_seconds: int = 0
    sandbox_output_limit_bytes: int = 0
    completion_max_continuations: int = 8
    nats_url: str = "nats://127.0.0.1:4222"
    nats_stream: str = "ULTRA_RUNS"
    nats_jobs_subject: str = "ultra.runs.jobs"
    nats_events_subject: str = "ultra.runs.events"
    nats_cancel_subject: str = "ultra.runs.cancel"
    nats_worker_durable: str = "ultra-deepagents-worker"
    worker_max_concurrency: int = 1
    worker_ack_wait_seconds: float = 300.0
    worker_ack_progress_interval_seconds: float = 60.0
    worker_max_deliver: int = 5
    worker_id: str = "ultra-deepagents-worker"
    worker_kind: str = "deepagents"
    worker_heartbeat_interval_seconds: float = 30.0
    control_base_url: str = "http://127.0.0.1:8088"
    control_status_timeout_seconds: float = 2.0
    control_status_poll_interval_seconds: float = 30.0
    control_run_lease_ttl_seconds: float = 600.0
    control_run_lease_required: bool = False
    workspace_root: str = "data/deepagents/workspaces"
    memory_root: str = "data/deepagents/memory"
    artifact_root: str = "data/artifacts"
    rarespot_tool_enabled: bool = True
    rarespot_control_base_url: str = "http://127.0.0.1:8088"
    rarespot_nats_url: str = "nats://127.0.0.1:4222"
    rarespot_nats_stream: str = "ULTRA_RUNS"
    rarespot_nats_jobs_subject: str = "ultra.runs.rarespot.jobs"
    rarespot_nats_events_subject: str = "ultra.runs.events"
    rarespot_nats_ack_wait_seconds: float = 120.0
    rarespot_nats_ack_progress_interval_seconds: float = 30.0
    rarespot_worker_id: str = "ultra-rarespot-worker"
    rarespot_worker_kind: str = "rarespot"
    rarespot_database_url: str = ""
    rarespot_artifact_root: str = "data/artifacts"
    rarespot_weights_path: str = "data/models/yolo/RareSpotWeights.pt"
    rarespot_yolov5_path: str = "third_party/yolov5"
    rarespot_allowed_input_roots: tuple[str, ...] = ()
    rarespot_upload_roots: tuple[str, ...] = ("data/uploads",)
    rarespot_upload_database_url: str = ""
    rarespot_tile_overlap: float = 0.25
    rarespot_conf_threshold: float = 0.25
    rarespot_iou_threshold: float = 0.45
    rarespot_imgsz: int = 512

    @classmethod
    def from_env(cls) -> "RuntimeSettings":
        return cls(
            openai_base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:8001/v1"),
            openai_model=os.getenv("OPENAI_MODEL", "deepseek_v4"),
            openai_api_key=os.getenv("OPENAI_API_KEY") or "EMPTY",
            model_supports_multimodal=_env_bool(
                "ULTRA_DEEPAGENTS_MODEL_SUPPORTS_MULTIMODAL",
                False,
            ),
            request_timeout_seconds=float(os.getenv("ULTRA_DEEPAGENTS_TIMEOUT_SECONDS", "0")),
            model_stream_idle_timeout_seconds=max(
                0.0,
                float(os.getenv("ULTRA_DEEPAGENTS_MODEL_STREAM_IDLE_TIMEOUT_SECONDS", "0")),
            ),
            model_stream_idle_max_recoveries=max(
                0,
                int(os.getenv("ULTRA_DEEPAGENTS_MODEL_STREAM_IDLE_MAX_RECOVERIES", "2")),
            ),
            max_retries=int(os.getenv("ULTRA_DEEPAGENTS_MAX_RETRIES", "1")),
            langgraph_recursion_limit=max(
                1,
                int(os.getenv("ULTRA_DEEPAGENTS_RECURSION_LIMIT", "1000")),
            ),
            sandbox_image=os.getenv(
                "ULTRA_DEEPAGENTS_SANDBOX_IMAGE",
                "bisque-ultra-codeexec:py311",
            ),
            sandbox_network=os.getenv("ULTRA_DEEPAGENTS_SANDBOX_NETWORK", "none"),
            sandbox_cpus=float(os.getenv("ULTRA_DEEPAGENTS_SANDBOX_CPUS", "0")),
            sandbox_memory=os.getenv("ULTRA_DEEPAGENTS_SANDBOX_MEMORY", ""),
            sandbox_pids_limit=int(os.getenv("ULTRA_DEEPAGENTS_SANDBOX_PIDS_LIMIT", "0")),
            sandbox_timeout_seconds=int(
                os.getenv("ULTRA_DEEPAGENTS_SANDBOX_TIMEOUT_SECONDS", "0")
            ),
            sandbox_output_limit_bytes=int(
                os.getenv("ULTRA_DEEPAGENTS_SANDBOX_OUTPUT_LIMIT_BYTES", "0")
            ),
            completion_max_continuations=max(
                0,
                int(os.getenv("ULTRA_DEEPAGENTS_COMPLETION_MAX_CONTINUATIONS", "8")),
            ),
            nats_url=os.getenv(
                "ULTRA_NATS_URL",
                os.getenv("ULTRA_CONTROL_NATS_URL", "nats://127.0.0.1:4222"),
            ),
            nats_stream=os.getenv(
                "ULTRA_NATS_STREAM",
                os.getenv("ULTRA_CONTROL_NATS_STREAM", "ULTRA_RUNS"),
            ),
            nats_jobs_subject=os.getenv(
                "ULTRA_NATS_JOBS_SUBJECT",
                os.getenv("ULTRA_CONTROL_NATS_JOBS_SUBJECT", "ultra.runs.jobs"),
            ),
            nats_events_subject=os.getenv(
                "ULTRA_NATS_EVENTS_SUBJECT",
                os.getenv("ULTRA_CONTROL_NATS_EVENTS_SUBJECT", "ultra.runs.events"),
            ),
            nats_cancel_subject=os.getenv(
                "ULTRA_NATS_CANCEL_SUBJECT",
                os.getenv("ULTRA_CONTROL_NATS_CANCEL_SUBJECT", "ultra.runs.cancel"),
            ),
            nats_worker_durable=os.getenv(
                "ULTRA_DEEPAGENTS_WORKER_DURABLE",
                "ultra-deepagents-worker",
            ),
            worker_max_concurrency=max(
                1,
                int(os.getenv("ULTRA_DEEPAGENTS_WORKER_MAX_CONCURRENCY", "1")),
            ),
            worker_ack_wait_seconds=float(
                os.getenv("ULTRA_DEEPAGENTS_WORKER_ACK_WAIT_SECONDS", "300")
            ),
            worker_ack_progress_interval_seconds=float(
                os.getenv("ULTRA_DEEPAGENTS_WORKER_ACK_PROGRESS_INTERVAL_SECONDS", "60")
            ),
            worker_max_deliver=max(
                1,
                int(os.getenv("ULTRA_DEEPAGENTS_WORKER_MAX_DELIVER", "5")),
            ),
            worker_id=os.getenv(
                "ULTRA_DEEPAGENTS_WORKER_ID",
                f"ultra-deepagents-worker@{gethostname()}:{os.getpid()}",
            ),
            worker_kind=os.getenv("ULTRA_DEEPAGENTS_WORKER_KIND", "deepagents"),
            worker_heartbeat_interval_seconds=max(
                0.0,
                float(
                    os.getenv(
                        "ULTRA_DEEPAGENTS_WORKER_HEARTBEAT_INTERVAL_SECONDS",
                        "30",
                    )
                ),
            ),
            control_base_url=os.getenv(
                "ULTRA_CONTROL_BASE_URL",
                "http://127.0.0.1:8088",
            ).rstrip("/"),
            control_status_timeout_seconds=float(
                os.getenv("ULTRA_DEEPAGENTS_CONTROL_STATUS_TIMEOUT_SECONDS", "2")
            ),
            control_status_poll_interval_seconds=float(
                os.getenv("ULTRA_DEEPAGENTS_CONTROL_STATUS_POLL_INTERVAL_SECONDS", "30")
            ),
            control_run_lease_ttl_seconds=max(
                1.0,
                float(os.getenv("ULTRA_DEEPAGENTS_CONTROL_RUN_LEASE_TTL_SECONDS", "600")),
            ),
            control_run_lease_required=_env_bool(
                "ULTRA_DEEPAGENTS_REQUIRE_CONTROL_RUN_LEASE",
                False,
            ),
            workspace_root=os.getenv(
                "ULTRA_DEEPAGENTS_WORKSPACE_ROOT",
                "data/deepagents/workspaces",
            ),
            memory_root=os.getenv(
                "ULTRA_DEEPAGENTS_MEMORY_ROOT",
                "data/deepagents/memory",
            ),
            artifact_root=os.getenv(
                "ULTRA_ARTIFACT_ROOT",
                os.getenv("ULTRA_CONTROL_ARTIFACT_ROOT", os.getenv("ARTIFACT_ROOT", "data/artifacts")),
            ),
            rarespot_tool_enabled=_env_bool("ULTRA_RARESPOT_TOOL_ENABLED", True),
            rarespot_control_base_url=os.getenv(
                "ULTRA_CONTROL_BASE_URL",
                "http://127.0.0.1:8088",
            ).rstrip("/"),
            rarespot_nats_url=os.getenv("ULTRA_CONTROL_NATS_URL", "nats://127.0.0.1:4222"),
            rarespot_nats_stream=os.getenv("ULTRA_CONTROL_NATS_STREAM", "ULTRA_RUNS"),
            rarespot_nats_jobs_subject=os.getenv(
                "ULTRA_CONTROL_NATS_RARESPOT_JOBS_SUBJECT",
                "ultra.runs.rarespot.jobs",
            ),
            rarespot_nats_events_subject=os.getenv(
                "ULTRA_CONTROL_NATS_EVENTS_SUBJECT",
                "ultra.runs.events",
            ),
            rarespot_nats_ack_wait_seconds=float(
                os.getenv("ULTRA_RARESPOT_NATS_ACK_WAIT_SECONDS", "120")
            ),
            rarespot_nats_ack_progress_interval_seconds=float(
                os.getenv("ULTRA_RARESPOT_NATS_ACK_PROGRESS_INTERVAL_SECONDS", "30")
            ),
            rarespot_worker_id=os.getenv(
                "ULTRA_RARESPOT_WORKER_ID",
                f"ultra-rarespot-worker@{gethostname()}:{os.getpid()}",
            ),
            rarespot_worker_kind=os.getenv("ULTRA_RARESPOT_WORKER_KIND", "rarespot"),
            rarespot_database_url=os.getenv("ULTRA_CONTROL_DATABASE_URL", ""),
            rarespot_artifact_root=os.getenv(
                "ULTRA_RARESPOT_ARTIFACT_ROOT",
                os.getenv("ULTRA_CONTROL_ARTIFACT_ROOT", os.getenv("ARTIFACT_ROOT", "data/artifacts")),
            ),
            rarespot_weights_path=os.getenv(
                "YOLOV5_RARESPOT_WEIGHTS",
                "data/models/yolo/RareSpotWeights.pt",
            ),
            rarespot_yolov5_path=os.getenv("YOLOV5_RUNTIME_PATH", "third_party/yolov5"),
            rarespot_allowed_input_roots=_env_tuple("ULTRA_RARESPOT_ALLOWED_INPUT_ROOTS"),
            rarespot_upload_roots=_rarespot_upload_roots(),
            rarespot_upload_database_url=os.getenv(
                "ULTRA_RARESPOT_UPLOAD_DATABASE_URL",
                os.getenv("RUN_STORE_PATH", ""),
            ),
            rarespot_tile_overlap=float(os.getenv("ULTRA_RARESPOT_TILE_OVERLAP", "0.25")),
            rarespot_conf_threshold=float(os.getenv("PRAIRIE_FIXED_CONF_THRESHOLD", "0.25")),
            rarespot_iou_threshold=float(os.getenv("PRAIRIE_FIXED_IOU_THRESHOLD", "0.45")),
            rarespot_imgsz=int(os.getenv("PRAIRIE_FIXED_IMGSZ", "512")),
        )


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _env_tuple(name: str) -> tuple[str, ...]:
    raw = os.getenv(name, "")
    return tuple(token.strip() for token in raw.split(os.pathsep) if token.strip())


def _rarespot_upload_roots() -> tuple[str, ...]:
    explicit = _env_tuple("ULTRA_RARESPOT_UPLOAD_ROOTS")
    if explicit:
        return explicit
    shared = _env_tuple("ULTRA_UPLOAD_ROOTS")
    if shared:
        return shared
    roots = [
        os.getenv("UPLOAD_STORE_ROOT", ""),
        os.getenv("SESSION_UPLOAD_ROOT", ""),
        "data/uploads",
    ]
    return tuple(root for root in roots if root)
