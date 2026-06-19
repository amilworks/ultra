from __future__ import annotations

import ipaddress
import json
import os
from dataclasses import dataclass
from socket import gethostname
from typing import Any
from urllib.parse import urlparse


def is_local_http_host(hostname: str | None) -> bool:
    """True for localhost / loopback / private / link-local / unspecified hosts.

    Shared by every async-subagent URL guard so the server-URL check and the
    context-URI check can never drift apart. Note: this is a literal-host check
    only — it does not resolve DNS, so a public name pointing at a private IP is
    not caught here (defense-in-depth, not a complete SSRF control).
    """
    host = str(hostname or "").strip().lower().rstrip(".")
    if not host or host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_unspecified


def allow_private_async_subagent_url() -> bool:
    """Local-dev opt-out for the async-subagent private-host URL guard."""
    return _env_bool("ULTRA_DEEPAGENTS_ALLOW_PRIVATE_ASYNC_SUBAGENT_URL", False)

DEFAULT_WORKER_MAX_CONCURRENCY = 64


@dataclass(frozen=True)
class RuntimeSettings:
    openai_base_url: str
    openai_model: str
    openai_api_key: str = "EMPTY"
    model_supports_multimodal: bool = False
    # Served model's input context window. When >0 it is published on the chat
    # model's profile (`max_input_tokens`), which flips deepagents summarization
    # from the conservative no-profile fallback (trigger 170k tokens / keep 6
    # messages) to adaptive fraction-based compaction (trigger 85% / keep 10% of
    # the window). MUST match the served model: too high and the agent never
    # summarizes before the model overflows. 0 keeps the safe fallback. Setting
    # only max_input_tokens does NOT enable native structured output, so the
    # subagent ToolStrategy auto-retry handoff is preserved.
    model_max_input_tokens: int = 0
    request_timeout_seconds: float = 0.0
    model_stream_idle_timeout_seconds: float = 0.0
    model_stream_idle_max_recoveries: int = 2
    max_retries: int = 1
    title_generation_enabled: bool = False
    title_generation_timeout_seconds: float = 8.0
    # Sidebar titles need ~12 tokens; on hybrid-reasoning models the thinking
    # phase alone can exceed the title timeout, so it is disabled per call.
    title_thinking_disabled: bool = True
    title_max_tokens: int = 96
    langgraph_recursion_limit: int = 1000
    sandbox_image: str = "bisque-ultra-codeexec:py311"
    sandbox_network: str = "none"
    sandbox_cpus: float = 0.0
    sandbox_memory: str = ""
    sandbox_pids_limit: int = 0
    sandbox_timeout_seconds: int = 0
    sandbox_output_limit_bytes: int = 0
    completion_max_continuations: int = 8
    async_subagents: tuple[dict[str, Any], ...] = ()
    nats_url: str = "nats://127.0.0.1:4222"
    nats_stream: str = "ULTRA_RUNS"
    nats_jobs_subject: str = "ultra.runs.jobs"
    nats_events_subject: str = "ultra.runs.events"
    nats_cancel_subject: str = "ultra.runs.cancel"
    nats_worker_durable: str = "ultra-deepagents-worker"
    data_agent_nats_url: str = "nats://127.0.0.1:4222"
    data_agent_nats_stream: str = "ULTRA_RUNS"
    data_agent_nats_jobs_subject: str = "ultra.data_agent.jobs"
    data_agent_nats_worker_durable: str = "ultra-data-agent-worker"
    data_agent_nats_ack_wait_seconds: float = 300.0
    data_agent_nats_ack_progress_interval_seconds: float = 60.0
    data_agent_worker_id: str = "ultra-data-agent-worker"
    data_agent_worker_kind: str = "data_agent"
    data_agent_control_lease_ttl_seconds: float = 600.0
    data_agent_control_lease_required: bool = False
    worker_max_concurrency: int = DEFAULT_WORKER_MAX_CONCURRENCY
    worker_ack_wait_seconds: float = 300.0
    worker_ack_progress_interval_seconds: float = 60.0
    worker_max_deliver: int = 5
    worker_id: str = "ultra-deepagents-worker"
    worker_kind: str = "deepagents"
    worker_heartbeat_interval_seconds: float = 30.0
    control_base_url: str = "http://127.0.0.1:8088"
    control_worker_token: str = ""
    control_status_timeout_seconds: float = 2.0
    control_status_poll_interval_seconds: float = 30.0
    control_run_lease_ttl_seconds: float = 600.0
    control_run_lease_required: bool = False
    # Durable LangGraph checkpointing so long runs resume mid-trajectory after a
    # worker/replica restart. Uses the control-plane Postgres when configured.
    control_database_url: str = ""
    checkpointer_enabled: bool = True
    # Reap abandoned durable checkpoint rows (runs that crashed before terminal
    # ack and were never resumed) older than this. Completed runs delete their row
    # on terminal ack, so this is only a backstop for stragglers. 0 disables.
    checkpoint_retention_seconds: int = 72 * 3600
    workspace_root: str = "data/deepagents/workspaces"
    memory_root: str = "data/deepagents/memory"
    artifact_root: str = "data/artifacts"
    # Agent skills (progressive-disclosure protocols, e.g. experiment rigor and
    # report contracts). Empty means the repo-shipped skills directory next to
    # the package; set to an absolute path to override per deployment.
    skills_root: str = ""
    # Organization-level read-only policy memory (/policies/*.md), scoped per org.
    # Empty resolves to ``<memory_root>/policies`` so it rides the same shared
    # barrel as per-user memory in production.
    policies_root: str = ""
    rarespot_tool_enabled: bool = True
    rarespot_control_base_url: str = "http://127.0.0.1:8088"
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
    # Host-side Git repo staging (clone a public repo into /workspace so the
    # network-none sandbox can run it on staged data). Trust boundary lives in
    # code_execution/git_staging.py; the sandbox is never given network egress.
    git_staging_enabled: bool = True
    git_staging_allowed_hosts: tuple[str, ...] = ("github.com",)
    git_staging_max_bytes: int = 2 * 1024**3
    git_staging_timeout_seconds: int = 600
    git_staging_depth: int = 1
    # Retention sweep for per-run scratch workspaces (<workspace_root>/<run_id>).
    # 0 disables. Durable artifacts live separately under artifact_root, so an
    # expired scratch workspace never holds deliverables.
    workspace_retention_seconds: int = 7 * 24 * 3600

    @classmethod
    def from_env(cls) -> RuntimeSettings:
        return cls(
            openai_base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:8001/v1"),
            openai_model=os.getenv("OPENAI_MODEL", "deepseek_v4"),
            openai_api_key=os.getenv("OPENAI_API_KEY") or "EMPTY",
            model_supports_multimodal=_env_bool(
                "ULTRA_DEEPAGENTS_MODEL_SUPPORTS_MULTIMODAL",
                False,
            ),
            model_max_input_tokens=max(
                0,
                int(os.getenv("ULTRA_DEEPAGENTS_MODEL_MAX_INPUT_TOKENS", "0")),
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
            title_generation_enabled=_env_bool(
                "ULTRA_DEEPAGENTS_TITLE_GENERATION_ENABLED",
                True,
            ),
            title_generation_timeout_seconds=max(
                0.0,
                float(os.getenv("ULTRA_DEEPAGENTS_TITLE_GENERATION_TIMEOUT_SECONDS", "8")),
            ),
            title_thinking_disabled=_env_bool(
                "ULTRA_DEEPAGENTS_TITLE_THINKING_DISABLED",
                True,
            ),
            title_max_tokens=max(
                0,
                int(os.getenv("ULTRA_DEEPAGENTS_TITLE_MAX_TOKENS", "96")),
            ),
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
            async_subagents=_async_subagents_from_env(),
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
            data_agent_nats_url=os.getenv(
                "ULTRA_DATA_AGENT_NATS_URL",
                os.getenv(
                    "ULTRA_CONTROL_NATS_URL",
                    os.getenv("ULTRA_NATS_URL", "nats://127.0.0.1:4222"),
                ),
            ),
            data_agent_nats_stream=os.getenv(
                "ULTRA_DATA_AGENT_NATS_STREAM",
                os.getenv(
                    "ULTRA_CONTROL_NATS_STREAM",
                    os.getenv("ULTRA_NATS_STREAM", "ULTRA_RUNS"),
                ),
            ),
            data_agent_nats_jobs_subject=os.getenv(
                "ULTRA_DATA_AGENT_NATS_JOBS_SUBJECT",
                os.getenv("ULTRA_CONTROL_NATS_DATA_AGENT_JOBS_SUBJECT", "ultra.data_agent.jobs"),
            ),
            data_agent_nats_worker_durable=os.getenv(
                "ULTRA_DATA_AGENT_NATS_WORKER_DURABLE",
                os.getenv("ULTRA_CONTROL_NATS_DATA_AGENT_WORKER_DURABLE", "ultra-data-agent-worker"),
            ),
            data_agent_nats_ack_wait_seconds=float(
                os.getenv("ULTRA_DATA_AGENT_NATS_ACK_WAIT_SECONDS", "300")
            ),
            data_agent_nats_ack_progress_interval_seconds=float(
                os.getenv("ULTRA_DATA_AGENT_NATS_ACK_PROGRESS_INTERVAL_SECONDS", "60")
            ),
            data_agent_worker_id=os.getenv(
                "ULTRA_DATA_AGENT_WORKER_ID",
                f"ultra-data-agent-worker@{gethostname()}:{os.getpid()}",
            ),
            data_agent_worker_kind=os.getenv("ULTRA_DATA_AGENT_WORKER_KIND", "data_agent"),
            data_agent_control_lease_ttl_seconds=max(
                1.0,
                float(os.getenv("ULTRA_DATA_AGENT_CONTROL_LEASE_TTL_SECONDS", "600")),
            ),
            data_agent_control_lease_required=_env_bool(
                "ULTRA_DATA_AGENT_REQUIRE_CONTROL_LEASE",
                False,
            ),
            worker_max_concurrency=max(
                1,
                int(
                    os.getenv(
                        "ULTRA_DEEPAGENTS_WORKER_MAX_CONCURRENCY",
                        str(DEFAULT_WORKER_MAX_CONCURRENCY),
                    )
                ),
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
            control_worker_token=os.getenv("ULTRA_CONTROL_WORKER_TOKEN", "").strip(),
            control_database_url=os.getenv("ULTRA_CONTROL_DATABASE_URL", "").strip(),
            checkpointer_enabled=_env_bool("ULTRA_DEEPAGENTS_CHECKPOINTER_ENABLED", True),
            checkpoint_retention_seconds=max(
                0,
                int(os.getenv("ULTRA_DEEPAGENTS_CHECKPOINT_RETENTION_SECONDS", str(72 * 3600))),
            ),
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
            skills_root=os.getenv("ULTRA_DEEPAGENTS_SKILLS_ROOT", "").strip(),
            policies_root=os.getenv("ULTRA_DEEPAGENTS_POLICIES_ROOT", "").strip(),
            rarespot_tool_enabled=_env_bool("ULTRA_RARESPOT_TOOL_ENABLED", True),
            rarespot_control_base_url=os.getenv(
                "ULTRA_CONTROL_BASE_URL",
                "http://127.0.0.1:8088",
            ).rstrip("/"),
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
            git_staging_enabled=_env_bool("ULTRA_DEEPAGENTS_ENABLE_GIT_STAGING", True),
            git_staging_allowed_hosts=_env_host_tuple(
                "ULTRA_DEEPAGENTS_GIT_STAGING_ALLOWED_HOSTS",
                ("github.com",),
            ),
            git_staging_max_bytes=max(
                0,
                int(os.getenv("ULTRA_DEEPAGENTS_GIT_STAGING_MAX_BYTES", str(2 * 1024**3))),
            ),
            git_staging_timeout_seconds=max(
                0,
                int(os.getenv("ULTRA_DEEPAGENTS_GIT_STAGING_TIMEOUT_SECONDS", "600")),
            ),
            git_staging_depth=max(
                1,
                int(os.getenv("ULTRA_DEEPAGENTS_GIT_STAGING_DEPTH", "1")),
            ),
            workspace_retention_seconds=max(
                0,
                int(os.getenv("ULTRA_DEEPAGENTS_WORKSPACE_RETENTION_SECONDS", str(7 * 24 * 3600))),
            ),
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


def _env_host_tuple(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    """Comma/space-separated lowercase host allowlist; falls back to ``default``."""
    raw = os.getenv(name)
    if raw is None:
        return default
    hosts = tuple(
        host.strip().lower()
        for host in raw.replace(",", " ").split()
        if host.strip()
    )
    return hosts or default


def _async_subagents_from_env() -> tuple[dict[str, Any], ...]:
    # Experimental, off by default: requires an EXTERNAL Agent Protocol /
    # LangGraph deployment that Ultra does not run in-process. Demand an explicit
    # enable flag IN ADDITION to non-empty JSON so the feature can never activate
    # by accident (a JSON value alone is not enough).
    if not _env_bool("ULTRA_DEEPAGENTS_ENABLE_ASYNC_SUBAGENTS", False):
        return ()
    raw = os.getenv("ULTRA_DEEPAGENTS_ASYNC_SUBAGENTS_JSON", "").strip()
    if not raw:
        return ()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "ULTRA_DEEPAGENTS_ASYNC_SUBAGENTS_JSON must be valid JSON"
        ) from exc
    if not isinstance(parsed, list):
        raise ValueError("ULTRA_DEEPAGENTS_ASYNC_SUBAGENTS_JSON must be a JSON array")

    specs: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for index, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise ValueError(
                "ULTRA_DEEPAGENTS_ASYNC_SUBAGENTS_JSON entries must be objects"
            )
        prefix = f"ULTRA_DEEPAGENTS_ASYNC_SUBAGENTS_JSON[{index}]"
        name = _required_async_subagent_string(item, "name", prefix=prefix)
        description = _required_async_subagent_string(item, "description", prefix=prefix)
        graph_id = _required_async_subagent_string(item, "graph_id", prefix=prefix)
        normalized_name = name.lower()
        if normalized_name in seen_names:
            raise ValueError(f"Duplicate async subagent name: {name}")
        seen_names.add(normalized_name)
        spec: dict[str, Any] = {
            "name": name,
            "description": description,
            "graph_id": graph_id,
        }
        url = _optional_async_subagent_url(item, "url", prefix=prefix)
        if url:
            spec["url"] = url
        headers = _async_subagent_headers(item.get("headers"), prefix=prefix)
        if headers:
            spec["headers"] = headers
        specs.append(spec)
    return tuple(specs)


def _required_async_subagent_string(item: dict[str, Any], field: str, *, prefix: str) -> str:
    if field not in item or item[field] is None:
        raise ValueError(f"{prefix}.{field} is required")
    value = item[field]
    if not isinstance(value, str):
        raise ValueError(f"{prefix}.{field} must be a non-empty string")
    value = value.strip()
    if not value:
        raise ValueError(f"{prefix}.{field} is required")
    return value


def _optional_async_subagent_string(
    item: dict[str, Any],
    field: str,
    *,
    prefix: str,
) -> str:
    if field not in item:
        return ""
    value = item[field]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{prefix}.{field} must be a non-empty string when provided")
    return value.strip()


def _optional_async_subagent_url(
    item: dict[str, Any],
    field: str,
    *,
    prefix: str,
) -> str:
    value = _optional_async_subagent_string(item, field, prefix=prefix)
    if not value:
        return ""
    parsed = urlparse(value)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            f"{prefix}.{field} must be an http:// or https:// endpoint when provided"
        )
    if parsed.username or parsed.password:
        raise ValueError(f"{prefix}.{field} must not include credentials")
    if is_local_http_host(parsed.hostname) and not allow_private_async_subagent_url():
        raise ValueError(
            f"{prefix}.{field} must not target a localhost/private/link-local host "
            "(the run context + task egress to it); set "
            "ULTRA_DEEPAGENTS_ALLOW_PRIVATE_ASYNC_SUBAGENT_URL=1 for local dev"
        )
    return value


def _async_subagent_headers(value: Any, *, prefix: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{prefix}.headers must be an object when provided")
    headers: dict[str, str] = {}
    seen_header_names: set[str] = set()
    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str):
            raise ValueError(f"{prefix}.headers contains a non-string header name")
        key = raw_key.strip()
        if not key:
            raise ValueError(f"{prefix}.headers contains an empty header name")
        normalized_key = key.lower()
        if normalized_key in seen_header_names:
            raise ValueError(f"Duplicate async subagent header name: {key}")
        seen_header_names.add(normalized_key)
        if not isinstance(raw_value, str) or not raw_value.strip():
            raise ValueError(f"{prefix}.headers.{key} must be a non-empty string")
        headers[key] = raw_value.strip()
    return headers


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
