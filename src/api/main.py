from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import mimetypes
import os
import re
import shutil
import tempfile
import time
from base64 import b64encode, urlsafe_b64decode
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, suppress
from copy import deepcopy
from datetime import datetime, timedelta
from functools import lru_cache
from http.cookies import SimpleCookie
from io import BytesIO
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, cast
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlencode, urlparse
from urllib.request import Request, urlopen
from uuid import uuid4
from xml.etree import ElementTree

import httpx
import numpy as np
from fastapi import (
    APIRouter,
    Body,
    Cookie,
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi import (
    Request as FastAPIRequest,
)
from fastapi import (
    Response as FastAPIResponse,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from PIL import Image
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)

from src.agno_backend import AgnoChatRuntime, build_agno_v3_services
from src.api.schemas import (
    AdminConversationActionResponse,
    AdminIssueListResponse,
    AdminIssueRecord,
    AdminOverviewResponse,
    AdminPlatformKpis,
    AdminRunActionResponse,
    AdminRunListResponse,
    AdminRunRecord,
    AdminToolUsageRecord,
    AdminUsageBucket,
    AdminUserListResponse,
    AdminUserSummary,
    AnalysisHistoryResponse,
    AnalysisRunSummary,
    ArtifactListResponse,
    ArtifactRecord,
    BisqueAuthGuestRequest,
    BisqueAuthLoginRequest,
    BisqueAuthSessionResponse,
    BisqueGuestProfile,
    BisqueImportItem,
    BisqueImportRequest,
    BisqueImportResponse,
    ChatRequest,
    ChatResponse,
    ChatTitleRequest,
    ChatTitleResponse,
    ContractAuditRecord,
    ContractAuditRequest,
    ContractAuditResponse,
    ConversationListResponse,
    ConversationRecord,
    ConversationSearchResponse,
    ConversationUpsertRequest,
    CreateRunRequest,
    CreateRunResponse,
    ImageLoadRequest,
    ImageLoadResponse,
    InferenceJobCreateRequest,
    ModelHealthRecord,
    ModelHealthResponse,
    PrairieBenchmarkRunRequest,
    PrairieBenchmarkRunResponse,
    PrairieRetrainListResponse,
    PrairieRetrainRecord,
    PrairieRetrainRequest,
    PrairieStatusResponse,
    PrairieSyncResponse,
    ReproReportRequest,
    ReproReportResponse,
    ResourceComputationLookupRequest,
    ResourceComputationLookupResponse,
    ResourceComputationSuggestion,
    ResourceListResponse,
    ResourceRecord,
    ResumableUploadChunkResponse,
    ResumableUploadCompleteResponse,
    ResumableUploadInitRequest,
    ResumableUploadSessionResponse,
    RunEventsResponse,
    RunResponse,
    RunResultResponse,
    Sam3InteractiveRequest,
    Sam3InteractiveResponse,
    SantaBarbaraWeatherResponse,
    StatsRunRequest,
    StatsRunResponse,
    StatsToolRecord,
    StatsToolsResponse,
    TrainingDatasetCreateRequest,
    TrainingDatasetItemsRequest,
    TrainingDatasetListResponse,
    TrainingDatasetRecord,
    TrainingDatasetResponse,
    TrainingDomainCreateRequest,
    TrainingDomainListResponse,
    TrainingDomainRecord,
    TrainingForkLineageRequest,
    TrainingJobControlRequest,
    TrainingJobCreateRequest,
    TrainingJobRecord,
    TrainingJobResponse,
    TrainingLineageListResponse,
    TrainingLineageRecord,
    TrainingMergeRequestCreateRequest,
    TrainingMergeRequestDecisionRequest,
    TrainingMergeRequestListResponse,
    TrainingMergeRequestRecord,
    TrainingMergeRequestResponse,
    TrainingModelRecord,
    TrainingModelsResponse,
    TrainingModelVersionListResponse,
    TrainingModelVersionRecord,
    TrainingModelVersionResponse,
    TrainingPreflightRequest,
    TrainingPreflightResponse,
    TrainingUpdateProposalDecisionRequest,
    TrainingUpdateProposalListResponse,
    TrainingUpdateProposalPreviewRequest,
    TrainingUpdateProposalPreviewResponse,
    TrainingUpdateProposalRecord,
    TrainingUpdateProposalResponse,
    TrainingVersionPromoteRequest,
    TrainingVersionRollbackRequest,
    UploadedFileRecord,
    UploadFilesResponse,
    V2ArtifactListResponse,
    V2ArtifactRecord,
    V2ArtifactResponse,
    V2GraphEventRecord,
    V2RunCancelRequest,
    V2RunCreateRequest,
    V2RunEventsResponse,
    V2RunListResponse,
    V2RunRecord,
    V2RunResumeRequest,
    V2ThreadCreateRequest,
    V2ThreadListResponse,
    V2ThreadMessage,
    V2ThreadMessageListResponse,
    V2ThreadRecord,
)
from src.api.v3 import build_v3_router
from src.auth import (
    BisqueAuthContext,
    get_request_bisque_auth,
    reset_request_bisque_auth,
    set_request_bisque_auth,
)
from src.auth import (
    cleanup_expired_bisque_sessions as shared_cleanup_expired_bisque_sessions,
)
from src.auth import (
    create_bisque_session as shared_create_bisque_session,
)
from src.auth import (
    delete_bisque_session as shared_delete_bisque_session,
)
from src.auth import (
    get_bisque_session as shared_get_bisque_session,
)
from src.auth import (
    touch_bisque_session as shared_touch_bisque_session,
)
from src.chat_titles import generate_chat_title
from src.config import get_settings
from src.evals.research_review import audit_contract_payload, score_research_value
from src.llm_client import get_openai_client
from src.orchestration.executor import PlanExecutor
from src.orchestration.models import RunStatus, WorkflowPlan, WorkflowRun
from src.orchestration.store import RunStore
from src.science.derivatives import invalidate_file_derivatives
from src.science.imaging import (
    _render_preview_image,
    extract_actionable_image_metadata,
    load_scientific_image,
    probe_scientific_image,
)
from src.science.reporting import generate_repro_report
from src.science.stats import list_curated_stat_tools, run_stat_tool
from src.science.viewer import (
    DEFAULT_HISTOGRAM_BINS,
    VIEWER_TILE_SIZE,
    build_hdf5_dataset_histogram,
    build_hdf5_dataset_summary,
    build_hdf5_dataset_table_preview,
    build_hdf5_materials_dashboard,
    build_hdf5_viewer_manifest,
    build_viewer_manifest,
    is_hdf5_viewer_path,
    is_ordinary_display_image_path,
    load_scalar_volume_texture,
    load_view_volume_source,
    normalize_view_axis,
    plan_volume_source_atlas,
    purge_scalar_volume_persistent_cache,
    render_hdf5_dataset_slice,
    render_view_atlas_png,
    render_view_histogram,
    render_view_plane_image,
    render_view_tile_png,
    render_volume_source_atlas_png,
)
from src.tooling.domains import BISQUE_TOOL_SCHEMAS
from src.tooling.progress import decode_progress_chunk
from src.tooling.tool_selection import _select_tool_subset
from src.tools import _render_yolo_detection_figure, sam2_prompt_image, segment_image_sam3
from src.training import (
    ContinuousLearningPolicy,
    DatasetValidationError,
    TrainingCancelledError,
    TrainingRunner,
    build_dataset_manifest,
    build_model_version,
    build_preflight_report,
    build_replay_mix_plan,
    compute_model_health_entries,
    evaluate_promotion_guardrails,
    evaluate_trigger_policy,
    get_model_definition,
    list_model_definitions,
    merge_transition_allowed,
    normalize_merge_status,
    normalize_proposal_status,
    normalize_spatial_dims,
    normalize_version_status,
    proposal_transition_allowed,
    version_transition_allowed,
)

# Optional test hook for injecting legacy chunk stream behavior in unit tests.
stream_chat_completion: Any | None = None
logger = logging.getLogger(__name__)


def _first_csv_header_value(value: str | None) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    token = raw.split(",", 1)[0].strip()
    return token or None


def _normalized_public_origin(value: str | None) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    parsed = urlparse(raw)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    return None


def _public_request_origin(
    request: FastAPIRequest | None,
    *,
    fallback_url: str | None = None,
) -> str | None:
    if request is not None:
        forwarded_proto = _first_csv_header_value(
            request.headers.get("x-forwarded-proto") or request.headers.get("x-forwarded-scheme")
        )
        forwarded_host = _first_csv_header_value(request.headers.get("x-forwarded-host"))
        forwarded_port = _first_csv_header_value(request.headers.get("x-forwarded-port"))
        host = forwarded_host or _first_csv_header_value(request.headers.get("host"))
        if host and forwarded_port and ":" not in host and forwarded_port not in {"80", "443"}:
            host = f"{host}:{forwarded_port}"
        scheme = forwarded_proto or str(getattr(request.url, "scheme", "") or "").strip()
        if scheme and host:
            return f"{scheme}://{host}"
        base_url = str(getattr(request, "base_url", "") or "").strip().rstrip("/")
        if base_url:
            return base_url
    return _normalized_public_origin(fallback_url)


def _normalize_bisque_resource_uri_with_root(resource: str, bisque_root: str) -> str:
    if not resource:
        raise ValueError("BisQue resource reference is empty.")

    root = bisque_root.rstrip("/")
    value = str(resource).strip()

    if "resource=" in value:
        candidate = value.split("resource=", 1)[-1].split("&", 1)[0]
        candidate = unquote(candidate)
        if candidate:
            value = candidate

    if value.startswith(root):
        value = value[len(root) :]

    if value.startswith("/resource/"):
        value = value.replace("/resource/", "/data_service/", 1)
    if value.startswith("/image_service/"):
        value = value.replace("/image_service/", "/data_service/", 1)

    if value.startswith("http://") or value.startswith("https://"):
        parsed = urlparse(value)
        path = str(parsed.path or "").strip()
        if path.startswith("/resource/"):
            path = path.replace("/resource/", "/data_service/", 1)
        elif path.startswith("/image_service/"):
            path = path.replace("/image_service/", "/data_service/", 1)
        if path.startswith("/data_service/"):
            return f"{root}{path}"
        if "/data_service/" in path:
            return f"{root}{path[path.index('/data_service/') :]}"
        return (
            value.replace("/image_service/", "/data_service/", 1)
            if "/image_service/" in value
            else value
        )

    if value.startswith("/data_service/"):
        return f"{root}{value}"
    if value.startswith("/image_service/"):
        return f"{root}{value.replace('/image_service/', '/data_service/', 1)}"

    return f"{root}/data_service/{value}"


def _build_bisque_links_for_root(resource: str, bisque_root: str) -> dict[str, str | None]:
    resource_uri = _normalize_bisque_resource_uri_with_root(resource, bisque_root)
    resource_uniq = resource_uri.rstrip("/").split("/")[-1] or None
    root = bisque_root.rstrip("/")
    image_service_url = f"{root}/image_service/{resource_uniq}" if resource_uniq else None
    client_view_url = (
        f"{root}/client_service/view?resource={resource_uri}" if resource_uri else None
    )
    return {
        "resource_uri": resource_uri,
        "resource_uniq": resource_uniq,
        "client_view_url": client_view_url,
        "image_service_url": image_service_url,
    }


def create_app() -> FastAPI:
    """Build and configure the FastAPI application and all ``/v1`` routes.

    Returns
    -------
    FastAPI
        Configured application instance with routers, middleware, storage, and
        runtime integrations initialized.

    Notes
    -----
    Keep API contracts synchronized with frontend types and preserve stream
    semantics (`token`, `done`, `error`) when modifying
    chat execution paths.
    """
    settings = get_settings()
    llm_model_name = str(
        getattr(settings, "resolved_llm_model", None)
        or getattr(settings, "openai_model", "unknown")
    )

    db_path = getattr(settings, "run_store_path", None) or os.path.join("data", "runs.db")
    store = RunStore(db_path)
    artifact_root = Path(settings.artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    upload_store_root = Path(getattr(settings, "upload_store_root", "data/uploads"))
    upload_store_root.mkdir(parents=True, exist_ok=True)
    upload_staging_root = upload_store_root / ".staging"
    upload_staging_root.mkdir(parents=True, exist_ok=True)
    upload_cache_root = upload_store_root / ".cache"
    upload_cache_root.mkdir(parents=True, exist_ok=True)
    upload_resumable_root = upload_staging_root / ".resumable"
    upload_resumable_root.mkdir(parents=True, exist_ok=True)
    agno_runtime = AgnoChatRuntime(settings=settings)
    agentic_v3_services = build_agno_v3_services(
        settings=settings,
        artifact_root=artifact_root / "v3",
        runtime=agno_runtime,
    )
    upload_max_files_per_request = max(
        1, int(getattr(settings, "upload_max_files_per_request", 64))
    )
    upload_max_file_size_bytes = max(
        1,
        int(getattr(settings, "upload_max_file_size_mb", 2048)) * 1024 * 1024,
    )
    upload_viewer_max_dimension = max(
        256,
        int(getattr(settings, "upload_viewer_max_dimension", 2048)),
    )
    viewer_hdf5_enabled = bool(getattr(settings, "viewer_hdf5_enabled", True))
    viewer_hdf5_atlas_max_voxels = max(
        1024,
        int(getattr(settings, "viewer_hdf5_atlas_max_voxels", 16_777_216)),
    )
    viewer_hdf5_atlas_max_texture_bytes = max(
        4 * 1024 * 1024,
        int(getattr(settings, "viewer_hdf5_atlas_max_texture_mb", 16)) * 1024 * 1024,
    )
    context_compaction_enabled = bool(getattr(settings, "context_compaction_enabled", True))
    context_compaction_min_messages = max(
        4,
        int(getattr(settings, "context_compaction_min_messages", 18)),
    )
    context_compaction_trigger_tokens = max(
        256,
        int(getattr(settings, "context_compaction_trigger_tokens", 12000)),
    )
    context_compaction_target_tokens = max(
        128,
        int(getattr(settings, "context_compaction_target_tokens", 7000)),
    )
    if context_compaction_target_tokens >= context_compaction_trigger_tokens:
        context_compaction_target_tokens = max(
            128,
            context_compaction_trigger_tokens - 512,
        )
    upload_sync_status_local_complete = "local_complete"
    upload_sync_status_queued = "bisque_sync_queued"
    upload_sync_status_running = "bisque_sync_running"
    upload_sync_status_succeeded = "bisque_sync_succeeded"
    upload_sync_status_failed = "bisque_sync_failed"
    caption_cache: dict[str, str] = {}
    viewer_payload_cache_lock = Lock()
    viewer_hdf5_preview_cache_lock = Lock()
    viewer_hdf5_atlas_cache_lock = Lock()
    bisque_session_cookie_name = "bisque_ultra_session"
    bisque_session_ttl_seconds = max(
        900,
        int(getattr(settings, "bisque_auth_session_ttl_seconds", 43200)),
    )
    oidc_state_cookie_name = "bisque_ultra_oidc_state"
    oidc_next_cookie_name = "bisque_ultra_oidc_next"
    oidc_state_ttl_seconds = 600
    anonymous_session_cookie_name = "bisque_ultra_anon"
    anonymous_session_ttl_seconds = max(
        86400,
        int(getattr(settings, "anonymous_session_ttl_seconds", 30 * 24 * 60 * 60)),
    )
    configured_admin_usernames: set[str] = {
        token.strip().lower()
        for token in str(getattr(settings, "admin_usernames", "") or "").split(",")
        if token.strip()
    }
    configured_primary_bisque_user = str(getattr(settings, "bisque_user", "") or "").strip().lower()
    if configured_primary_bisque_user:
        configured_admin_usernames.add(configured_primary_bisque_user)
    training_runner = TrainingRunner()
    training_job_threads: dict[str, Thread] = {}
    training_job_threads_lock = Lock()
    continuous_scheduler_owner = f"api-{uuid4().hex[:8]}"
    continuous_scheduler_interval_seconds = max(
        300,
        int(getattr(settings, "training_trigger_interval_seconds", 6 * 60 * 60)),
    )
    continuous_scheduler_ttl_seconds = max(
        120,
        int(getattr(settings, "training_scheduler_lease_ttl_seconds", 7 * 60)),
    )
    continuous_scheduler_stop = Event()
    continuous_scheduler_thread: Thread | None = None
    prairie_sync_scheduler_owner = f"prairie-{uuid4().hex[:8]}"
    prairie_sync_interval_seconds = max(
        300,
        int(getattr(settings, "prairie_sync_interval_seconds", 6 * 60 * 60) or 6 * 60 * 60),
    )
    prairie_sync_scheduler_stop = Event()
    prairie_sync_scheduler_thread: Thread | None = None

    @asynccontextmanager
    async def _app_lifespan(_app: FastAPI) -> AsyncIterator[None]:
        del _app
        _start_continuous_scheduler()
        _start_prairie_sync_scheduler()
        try:
            yield
        finally:
            nonlocal continuous_scheduler_thread, prairie_sync_scheduler_thread
            continuous_scheduler_stop.set()
            prairie_sync_scheduler_stop.set()
            if continuous_scheduler_thread is not None and continuous_scheduler_thread.is_alive():
                continuous_scheduler_thread.join(timeout=2.0)
            if (
                prairie_sync_scheduler_thread is not None
                and prairie_sync_scheduler_thread.is_alive()
            ):
                prairie_sync_scheduler_thread.join(timeout=2.0)
            continuous_scheduler_thread = None
            prairie_sync_scheduler_thread = None

    app = FastAPI(title="Bisque Ultra API", version=settings.app_version, lifespan=_app_lifespan)
    enable_cors = bool(settings.enable_cors or settings.allowed_origins)
    if enable_cors:
        allow_origins = settings.allowed_origins or [
            "http://127.0.0.1:5173",
            "http://localhost:5173",
            "http://127.0.0.1:3000",
            "http://localhost:3000",
        ]
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allow_origins,
            allow_credentials="*" not in allow_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    v1 = APIRouter(prefix="/v1")
    v2 = APIRouter(prefix="/v2")
    legacy = APIRouter()
    metrics_registry = CollectorRegistry()
    http_request_counter = Counter(
        "ultra_http_requests_total",
        "Count of HTTP requests served by the Ultra API.",
        ["method", "route", "status_code"],
        registry=metrics_registry,
    )
    http_request_latency_seconds = Histogram(
        "ultra_http_request_duration_seconds",
        "Latency of HTTP requests served by the Ultra API.",
        ["method", "route"],
        registry=metrics_registry,
    )
    request_id_header_name = "X-Request-Id"
    query_api_key_deprecation_warning = '299 - "Query-string api_key authentication is deprecated; use X-API-Key or a browser session."'

    def _utc_now_epoch() -> float:
        return time.time()

    def _cleanup_expired_bisque_sessions(now: float | None = None) -> None:
        shared_cleanup_expired_bisque_sessions(now=_utc_now_epoch() if now is None else float(now))

    def _create_bisque_session(
        *,
        username: str,
        password: str,
        bisque_root: str,
        mode: str = "bisque",
        guest_profile: dict[str, Any] | None = None,
        auth_provider: str = "local",
        access_token: str | None = None,
        id_token: str | None = None,
        bisque_cookie_header: str | None = None,
    ) -> dict[str, Any]:
        return shared_create_bisque_session(
            username=username,
            password=password,
            bisque_root=bisque_root,
            ttl_seconds=bisque_session_ttl_seconds,
            mode=mode,
            guest_profile=guest_profile,
            auth_provider=auth_provider,
            access_token=access_token,
            id_token=id_token,
            bisque_cookie_header=bisque_cookie_header,
            now=_utc_now_epoch(),
        )

    def _delete_bisque_session(session_id: str | None) -> None:
        shared_delete_bisque_session(session_id)

    def _touch_bisque_session(session: dict[str, Any]) -> dict[str, Any]:
        return shared_touch_bisque_session(
            session,
            ttl_seconds=bisque_session_ttl_seconds,
            now=_utc_now_epoch(),
        )

    def _get_bisque_session(session_id: str | None) -> dict[str, Any] | None:
        return shared_get_bisque_session(
            session_id,
            ttl_seconds=bisque_session_ttl_seconds,
            touch=False,
            now=_utc_now_epoch(),
        )

    def _is_hex_session_token(value: str | None) -> bool:
        token = str(value or "").strip().lower()
        return bool(re.fullmatch(r"[0-9a-f]{32}", token))

    def _should_secure_cookie(request: FastAPIRequest | None = None) -> bool:
        environment = str(getattr(settings, "environment", "development")).strip().lower()
        if environment != "production":
            return False
        if request is None:
            return True
        hostname = str(getattr(request.url, "hostname", "") or "").strip().lower()
        return hostname not in {"localhost", "127.0.0.1", "::1"}

    def _issue_anonymous_auth_context(
        response: FastAPIResponse,
        *,
        request: FastAPIRequest,
        anonymous_session: str | None,
    ) -> dict[str, Any]:
        session_id = str(anonymous_session or "").strip().lower()
        should_set_cookie = not _is_hex_session_token(session_id)
        if should_set_cookie:
            session_id = uuid4().hex
            response.set_cookie(
                key=anonymous_session_cookie_name,
                value=session_id,
                max_age=anonymous_session_ttl_seconds,
                httponly=True,
                secure=_should_secure_cookie(request),
                samesite="lax",
                path="/",
            )
        now = _utc_now_epoch()
        return {
            "session_id": session_id,
            "username": "anonymous",
            "password": "",
            "bisque_root": str(getattr(settings, "bisque_root", "") or "").strip().rstrip("/"),
            "created_at": now,
            "expires_at": now + anonymous_session_ttl_seconds,
            "mode": "guest",
            "guest_profile": None,
            "anonymous_session": True,
        }

    def _extract_bisque_user_from_session_xml(payload: bytes) -> str | None:
        if not payload:
            return None
        try:
            node = ElementTree.fromstring(payload.decode("utf-8", errors="ignore"))
        except Exception:
            return None
        if str(node.tag or "").strip().lower() == "user":
            for child in node.findall(".//tag"):
                name = str(child.attrib.get("name") or "").strip().lower()
                if name == "name":
                    value = str(child.attrib.get("value") or "").strip()
                    if value:
                        return value
        for child in node.findall(".//tag"):
            name = str(child.attrib.get("name") or "").strip().lower()
            if name in {"user", "username"}:
                value = str(child.attrib.get("value") or "").strip()
                if value:
                    return value
        for key in ("user", "owner", "name"):
            value = node.attrib.get(key)
            if value:
                return str(value)
        text = (node.text or "").strip()
        return text or None

    def _extract_bisque_group_from_session_xml(payload: bytes) -> str | None:
        if not payload:
            return None
        try:
            node = ElementTree.fromstring(payload.decode("utf-8", errors="ignore"))
        except Exception:
            return None
        for child in node.findall(".//tag"):
            name = str(child.attrib.get("name") or "").strip().lower()
            if name == "group":
                value = str(child.attrib.get("value") or "").strip()
                if value:
                    return value
        value = node.attrib.get("group")
        if value:
            return str(value).strip() or None
        return None

    def _normalize_bisque_username(value: str | None) -> str:
        raw = str(value or "").strip().lower()
        if not raw:
            return ""
        if "?" in raw:
            raw = raw.split("?", 1)[0]
        raw = raw.rstrip("/")
        if "/" in raw:
            raw = raw.rsplit("/", 1)[-1]
        return raw.strip()

    def _is_admin_username(value: str | None) -> bool:
        normalized = _normalize_bisque_username(value)
        if not normalized:
            return False
        if normalized in configured_admin_usernames:
            return True
        for candidate in configured_admin_usernames:
            if _normalize_bisque_username(candidate) == normalized:
                return True
        return False

    def _is_guest_like_user(value: str | None) -> bool:
        normalized = _normalize_bisque_username(value)
        if not normalized:
            return True
        guest_tokens = {"guest", "anonymous", "anon", "public", "nobody", "none"}
        return normalized in guest_tokens

    def _oidc_via_bisque_login_enabled() -> bool:
        return bool(getattr(settings, "bisque_auth_oidc_via_bisque_login", False))

    def _bisque_auth_mode() -> str:
        mode = str(getattr(settings, "bisque_auth_mode", "") or "").strip().lower()
        if mode in {"local", "oidc", "dual"}:
            return mode
        return "local"

    def _is_oidc_enabled() -> bool:
        if not bool(getattr(settings, "bisque_auth_oidc_enabled", False)):
            return False
        issuer = str(getattr(settings, "bisque_auth_oidc_issuer_url", "") or "").strip()
        client_id = str(getattr(settings, "bisque_auth_oidc_client_id", "") or "").strip()
        return bool(issuer and client_id)

    def _is_browser_oidc_enabled() -> bool:
        if _oidc_via_bisque_login_enabled():
            root = str(getattr(settings, "bisque_root", "") or "").strip().rstrip("/")
            return bool(root)
        return _is_oidc_enabled()

    def _extract_passthrough_bisque_cookie_header(request: FastAPIRequest) -> str | None:
        raw_cookie = str(request.headers.get("cookie") or "").strip()
        if not raw_cookie:
            return None
        parsed = SimpleCookie()
        try:
            parsed.load(raw_cookie)
        except Exception:
            return None
        allowed_names = {
            "authtkt",
            "tg-visit",
            "tg-remember",
            "repoze.who.plugins.auth_tkt.userid",
        }
        selected: list[str] = []
        for key, morsel in parsed.items():
            normalized = str(key or "").strip().lower()
            if normalized in allowed_names or normalized.startswith("repoze.who"):
                selected.append(f"{key}={morsel.value}")
        if not selected:
            return None
        return "; ".join(selected)

    def _resolve_bisque_user_from_cookie_header(
        *,
        cookie_header: str,
        bisque_root: str,
    ) -> str | None:
        normalized_root = str(bisque_root or "").strip().rstrip("/")
        if not normalized_root:
            return None
        request = Request(
            f"{normalized_root}/auth_service/whoami",
            headers={
                "Accept": "text/xml,*/*",
                "Cookie": str(cookie_header or "").strip(),
            },
            method="GET",
        )
        try:
            with urlopen(request, timeout=12) as response:
                payload = response.read() or b""
        except HTTPError as exc:
            if int(exc.code) in {401, 403}:
                return None
            return None
        except Exception:
            return None
        resolved = _extract_bisque_user_from_session_xml(payload)
        if _is_guest_like_user(resolved):
            return None
        value = str(resolved or "").strip()
        return value or None

    def _bootstrap_bisque_session_from_browser_cookie(
        *,
        request: FastAPIRequest,
        response: FastAPIResponse,
    ) -> dict[str, Any] | None:
        cookie_header = _extract_passthrough_bisque_cookie_header(request)
        if not cookie_header:
            return None
        root = str(getattr(settings, "bisque_root", "") or "").strip().rstrip("/")
        if not root:
            return None
        username = _resolve_bisque_user_from_cookie_header(
            cookie_header=cookie_header,
            bisque_root=root,
        )
        if not username:
            return None
        session = _create_bisque_session(
            username=username,
            password="",
            bisque_root=root,
            mode="bisque",
            auth_provider="bisque_cookie",
            bisque_cookie_header=cookie_header,
        )
        response.set_cookie(
            key=bisque_session_cookie_name,
            value=str(session["session_id"]),
            max_age=bisque_session_ttl_seconds,
            httponly=True,
            secure=_should_secure_cookie(request),
            samesite="lax",
            path="/",
        )
        return session

    def _refresh_bisque_session_from_browser_cookie_if_needed(
        *,
        session: dict[str, Any],
        request: FastAPIRequest,
        response: FastAPIResponse,
    ) -> dict[str, Any]:
        if not _oidc_via_bisque_login_enabled():
            return session
        current_cookie_header = str(session.get("bisque_cookie_header") or "").strip()
        if current_cookie_header:
            return session
        refreshed = _bootstrap_bisque_session_from_browser_cookie(
            request=request,
            response=response,
        )
        if refreshed:
            return refreshed
        return session

    def _oidc_redirect_uri(request: FastAPIRequest | None = None) -> str:
        configured = str(getattr(settings, "bisque_auth_oidc_redirect_uri", "") or "").strip()
        if configured:
            return configured
        public_origin = _public_request_origin(
            request,
            fallback_url=str(getattr(settings, "orchestrator_api_url", "") or "").strip(),
        )
        if public_origin:
            return f"{public_origin}/v1/auth/oidc/callback"
        return "http://localhost:8000/v1/auth/oidc/callback"

    def _oidc_endpoint(kind: str) -> str:
        normalized = str(kind or "").strip().lower()
        endpoint_overrides = {
            "authorize": str(getattr(settings, "bisque_auth_oidc_authorize_url", "") or "").strip(),
            "token": str(getattr(settings, "bisque_auth_oidc_token_url", "") or "").strip(),
            "userinfo": str(getattr(settings, "bisque_auth_oidc_userinfo_url", "") or "").strip(),
            "logout": str(getattr(settings, "bisque_auth_oidc_logout_url", "") or "").strip(),
        }
        override = endpoint_overrides.get(normalized)
        if override:
            return override.rstrip("/")

        issuer = str(getattr(settings, "bisque_auth_oidc_issuer_url", "") or "").strip().rstrip("/")
        if not issuer:
            return ""
        path_suffix = {
            "authorize": "auth",
            "token": "token",
            "userinfo": "userinfo",
            "logout": "logout",
        }.get(normalized)
        if not path_suffix:
            return ""
        return f"{issuer}/protocol/openid-connect/{path_suffix}"

    def _frontend_oidc_redirect_url(request: FastAPIRequest | None = None) -> str:
        configured = str(
            getattr(settings, "bisque_auth_oidc_frontend_redirect_url", "") or ""
        ).strip()
        if configured:
            return configured
        if request is not None:
            for header_name in ("origin", "referer"):
                origin = _normalized_public_origin(request.headers.get(header_name))
                if origin:
                    return f"{origin}/"
        root = str(getattr(settings, "bisque_root", "") or "").strip()
        if root:
            parsed = urlparse(root)
            if parsed.scheme and parsed.netloc:
                return f"{parsed.scheme}://{parsed.netloc}/"
        return "http://localhost:5173/"

    def _is_loopback_host(hostname: str | None) -> bool:
        normalized = str(hostname or "").strip().lower()
        return normalized in {"localhost", "127.0.0.1", "::1"}

    def _is_safe_frontend_redirect_target(
        request: FastAPIRequest,
        target: str | None,
    ) -> bool:
        value = str(target or "").strip()
        if not value:
            return False
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return False

        target_host = str(parsed.hostname or "").strip().lower()
        request_host = str(getattr(request.url, "hostname", "") or "").strip().lower()
        configured_host = (
            str(urlparse(_frontend_oidc_redirect_url(request)).hostname or "").strip().lower()
        )
        allowed_hosts = {host for host in (request_host, configured_host) if host}
        if target_host in allowed_hosts:
            return True
        return bool(target_host) and any(
            _is_loopback_host(candidate) and _is_loopback_host(target_host)
            for candidate in allowed_hosts
        )

    def _resolve_frontend_redirect_target(
        request: FastAPIRequest,
        preferred: str | None = None,
    ) -> str:
        candidate = str(preferred or "").strip()
        if _is_safe_frontend_redirect_target(request, candidate):
            return candidate
        return _frontend_oidc_redirect_url(request)

    def _default_logout_redirect_url(
        request: FastAPIRequest | None = None,
        *,
        preferred: str | None = None,
    ) -> str:
        explicit = str(getattr(settings, "bisque_auth_logout_redirect_url", "") or "").strip()
        if explicit:
            return explicit
        candidate = str(preferred or "").strip()
        if request is not None and _is_safe_frontend_redirect_target(request, candidate):
            return candidate
        if candidate:
            return candidate
        return _frontend_oidc_redirect_url(request)

    def _append_query_params(url: str, params: dict[str, str]) -> str:
        normalized_url = str(url or "").strip()
        if not normalized_url:
            return normalized_url
        filtered = {key: value for key, value in params.items() if str(value or "").strip()}
        if not filtered:
            return normalized_url
        separator = "&" if "?" in normalized_url else "?"
        return f"{normalized_url}{separator}{urlencode(filtered)}"

    def _bisque_browser_bootstrap_url(destination: str) -> str:
        target = str(destination or "").strip()
        root = str(getattr(settings, "bisque_root", "") or "").strip().rstrip("/")
        if not target or not root:
            return target
        if not _oidc_via_bisque_login_enabled():
            return target
        return _append_query_params(
            f"{root}/auth_service/oidc_login",
            {"came_from": target},
        )

    def _jwt_claims(token: str | None) -> dict[str, Any]:
        raw = str(token or "").strip()
        if not raw:
            return {}
        segments = raw.split(".")
        if len(segments) < 2:
            return {}
        payload = segments[1]
        payload += "=" * (-len(payload) % 4)
        try:
            decoded = urlsafe_b64decode(payload.encode("utf-8")).decode("utf-8")
            parsed = json.loads(decoded)
        except Exception:
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}

    def _fetch_oidc_userinfo(access_token: str | None) -> dict[str, Any]:
        token = str(access_token or "").strip()
        if not token:
            return {}
        userinfo_url = _oidc_endpoint("userinfo")
        if not userinfo_url:
            return {}
        try:
            response = httpx.get(
                userinfo_url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/json",
                },
                timeout=15.0,
            )
            if response.status_code >= 400:
                return {}
            payload = response.json()
            return dict(payload) if isinstance(payload, dict) else {}
        except Exception:
            return {}

    def _resolve_oidc_username(
        *,
        userinfo_claims: dict[str, Any],
        id_token_claims: dict[str, Any],
    ) -> str | None:
        configured_claim = str(
            getattr(settings, "bisque_auth_oidc_username_claim", "") or ""
        ).strip()
        candidate_claims: list[str] = []
        for claim in (
            configured_claim,
            "preferred_username",
            "email",
            "upn",
            "name",
            "sub",
        ):
            if claim and claim not in candidate_claims:
                candidate_claims.append(claim)
        for claim_set in (userinfo_claims, id_token_claims):
            if not claim_set:
                continue
            for claim in candidate_claims:
                value = claim_set.get(claim)
                normalized = str(value or "").strip()
                if normalized:
                    return normalized
        return None

    def _build_oidc_logout_url(
        *,
        id_token_hint: str | None = None,
        request: FastAPIRequest | None = None,
        preferred_redirect: str | None = None,
    ) -> str:
        logout_endpoint = _oidc_endpoint("logout")
        if not logout_endpoint:
            return _default_logout_redirect_url(request, preferred=preferred_redirect)
        params: dict[str, str] = {
            "post_logout_redirect_uri": _default_logout_redirect_url(
                request,
                preferred=preferred_redirect,
            ),
        }
        client_id = str(getattr(settings, "bisque_auth_oidc_client_id", "") or "").strip()
        if client_id:
            params["client_id"] = client_id
        hint = str(id_token_hint or "").strip()
        if hint:
            params["id_token_hint"] = hint
        return _append_query_params(logout_endpoint, params)

    def _frontend_auth_error_redirect(
        request: FastAPIRequest,
        message: str,
        preferred_redirect: str | None = None,
    ) -> RedirectResponse:
        destination_base = _resolve_frontend_redirect_target(request, preferred_redirect)
        destination = _append_query_params(
            destination_base,
            {"auth_error": message},
        )
        return RedirectResponse(destination, status_code=302)

    def _session_to_auth_response(session: dict[str, Any]) -> BisqueAuthSessionResponse:
        mode = str(session.get("mode") or "bisque").strip().lower()
        guest_profile_raw = session.get("guest_profile")
        guest_profile: BisqueGuestProfile | None = None
        if mode == "guest" and isinstance(guest_profile_raw, dict):
            try:
                guest_profile = BisqueGuestProfile.model_validate(guest_profile_raw)
            except Exception:
                guest_profile = None
        expires_at_raw = float(session.get("expires_at", 0.0))
        return BisqueAuthSessionResponse(
            authenticated=True,
            username=str(session.get("username") or "").strip() or None,
            bisque_root=str(session.get("bisque_root") or "").strip() or None,
            expires_at=datetime.utcfromtimestamp(expires_at_raw) if expires_at_raw > 0 else None,
            mode="guest" if mode == "guest" else "bisque",
            guest_profile=guest_profile,
            is_admin=(mode != "guest" and _is_admin_username(session.get("username"))),
        )

    def _bisque_login_redirects_to_oidc(*, bisque_root: str) -> bool:
        normalized_root = str(bisque_root or "").strip().rstrip("/")
        if not normalized_root:
            return False
        try:
            response = httpx.get(
                f"{normalized_root}/auth_service/login",
                follow_redirects=False,
                timeout=8.0,
            )
        except Exception:
            return False
        if int(response.status_code) not in {301, 302, 303, 307, 308}:
            return False
        location = str(response.headers.get("location") or "").strip().lower()
        return "oidc_login" in location or "openid-connect" in location

    def _request_bisque_local_access_token(
        *,
        username: str,
        password: str,
        bisque_root: str,
    ) -> str:
        normalized_root = str(bisque_root or "").strip().rstrip("/")
        if not normalized_root:
            raise HTTPException(status_code=500, detail="BisQue root is not configured.")

        encoded_payload = urlencode(
            {
                "username": str(username or "").strip(),
                "password": str(password or "").strip(),
            }
        ).encode("utf-8")
        request = Request(
            f"{normalized_root}/auth_service/token",
            data=encoded_payload,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json,*/*",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=15) as response:
                status_code = getattr(response, "status", None) or response.getcode()
                if int(status_code) >= 400:
                    raise HTTPException(status_code=401, detail="Invalid BisQue credentials.")
                payload = response.read(16384)
        except HTTPError as exc:
            if exc.code in {401, 403}:
                raise HTTPException(status_code=401, detail="Invalid BisQue credentials.") from exc
            if int(exc.code) >= 500 and _bisque_login_redirects_to_oidc(
                bisque_root=normalized_root
            ):
                raise HTTPException(
                    status_code=401,
                    detail=(
                        "BisQue server requires OIDC/SSO login and rejected direct "
                        "username/password authentication. Configure BISQUE_AUTH_OIDC_* "
                        "for this app or use a BisQue deployment with basic auth enabled."
                    ),
                ) from exc
            raise HTTPException(
                status_code=502,
                detail=f"BisQue local token request failed (HTTP {exc.code}).",
            ) from exc
        except URLError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"BisQue local token request failed: {exc.reason}",
            ) from exc
        except TimeoutError as exc:
            raise HTTPException(
                status_code=504, detail="BisQue local token request timed out."
            ) from exc
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"BisQue local token request failed: {exc}",
            ) from exc

        try:
            parsed = json.loads(payload.decode("utf-8", errors="ignore") or "{}")
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail="BisQue local token response was not valid JSON.",
            ) from exc
        token = str(
            parsed.get("access_token")
            or parsed.get("token")
            or parsed.get("auth_token")
            or parsed.get("local_token")
            or ""
        ).strip()
        if not token:
            raise HTTPException(
                status_code=502,
                detail="BisQue local token response did not include an access token.",
            )
        return token

    def _validate_bisque_access_token(
        *,
        access_token: str,
        requested_username: str,
        bisque_root: str,
    ) -> str:
        normalized_root = str(bisque_root or "").strip().rstrip("/")
        if not normalized_root:
            raise HTTPException(status_code=500, detail="BisQue root is not configured.")
        request = Request(
            f"{normalized_root}/auth_service/whoami",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/xml,text/xml,*/*",
            },
            method="GET",
        )
        try:
            with urlopen(request, timeout=15) as response:
                status_code = getattr(response, "status", None) or response.getcode()
                if int(status_code) >= 400:
                    raise HTTPException(status_code=401, detail="Invalid BisQue credentials.")
                payload = response.read(8192)
        except HTTPError as exc:
            if exc.code in {401, 403}:
                raise HTTPException(status_code=401, detail="Invalid BisQue credentials.") from exc
            raise HTTPException(
                status_code=502,
                detail=f"BisQue token validation failed (HTTP {exc.code}).",
            ) from exc
        except URLError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"BisQue token validation failed: {exc.reason}",
            ) from exc
        except TimeoutError as exc:
            raise HTTPException(
                status_code=504, detail="BisQue token validation timed out."
            ) from exc
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"BisQue token validation failed: {exc}",
            ) from exc

        resolved_user = _extract_bisque_user_from_session_xml(payload)
        normalized_requested_username = _normalize_bisque_username(requested_username)
        normalized_resolved_username = _normalize_bisque_username(resolved_user)

        if _is_guest_like_user(normalized_resolved_username):
            raise HTTPException(
                status_code=401,
                detail=(
                    "BisQue returned a guest/anonymous session for these credentials. "
                    "Please verify username and password."
                ),
            )
        if (
            normalized_requested_username
            and normalized_resolved_username
            and normalized_requested_username != normalized_resolved_username
        ):
            raise HTTPException(
                status_code=401,
                detail=(
                    "BisQue session user did not match the supplied username. "
                    "Please verify your credentials."
                ),
            )
        return str(requested_username or "").strip() or str(resolved_user or "").strip()

    def _validate_bisque_credentials(*, username: str, password: str, bisque_root: str) -> str:
        normalized_root = str(bisque_root or "").strip().rstrip("/")
        if not normalized_root:
            raise HTTPException(status_code=500, detail="BisQue root is not configured.")

        normalized_requested_username = _normalize_bisque_username(username)
        creds = f"{username}:{password}".encode()
        request = Request(
            f"{normalized_root}/auth_service/session",
            headers={
                "Authorization": f"Basic {b64encode(creds).decode('ascii')}",
                "Accept": "application/xml,text/xml,*/*",
            },
            method="GET",
        )
        try:
            with urlopen(request, timeout=15) as response:
                status_code = getattr(response, "status", None) or response.getcode()
                if int(status_code) >= 400:
                    raise HTTPException(status_code=401, detail="Invalid BisQue credentials.")
                payload = response.read(8192)
        except HTTPError as exc:
            if exc.code in {401, 403}:
                raise HTTPException(status_code=401, detail="Invalid BisQue credentials.") from exc
            if int(exc.code) >= 500 and _bisque_login_redirects_to_oidc(
                bisque_root=normalized_root
            ):
                raise HTTPException(
                    status_code=401,
                    detail=(
                        "BisQue server requires OIDC/SSO login and rejected direct "
                        "username/password authentication. Configure BISQUE_AUTH_OIDC_* "
                        "for this app or use a BisQue deployment with basic auth enabled."
                    ),
                ) from exc
            raise HTTPException(
                status_code=502,
                detail=f"BisQue authentication request failed (HTTP {exc.code}).",
            ) from exc
        except URLError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"BisQue authentication request failed: {exc.reason}",
            ) from exc
        except TimeoutError as exc:
            raise HTTPException(status_code=504, detail="BisQue authentication timed out.") from exc
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"BisQue authentication request failed: {exc}",
            ) from exc

        resolved_user = _extract_bisque_user_from_session_xml(payload)
        resolved_group = _extract_bisque_group_from_session_xml(payload)
        normalized_resolved_username = _normalize_bisque_username(resolved_user)

        if _is_guest_like_user(normalized_resolved_username):
            raise HTTPException(
                status_code=401,
                detail=(
                    "BisQue returned a guest/anonymous session for these credentials. "
                    "Please verify username and password."
                ),
            )

        if not normalized_resolved_username:
            raise HTTPException(
                status_code=401,
                detail="BisQue did not return an authenticated user for these credentials.",
            )

        if resolved_group:
            lowered_groups = {
                token.strip().lower() for token in resolved_group.split(",") if token.strip()
            }
            if lowered_groups.intersection({"guest", "guests", "anonymous", "public"}):
                raise HTTPException(
                    status_code=401,
                    detail=(
                        "BisQue returned a guest/anonymous session for these credentials. "
                        "Please verify username and password."
                    ),
                )

        should_enforce_user_match = bool(
            resolved_user and "://" not in str(resolved_user) and "/" not in str(resolved_user)
        )
        if (
            should_enforce_user_match
            and normalized_requested_username
            and normalized_resolved_username
        ):
            matches = (
                normalized_requested_username == normalized_resolved_username
                or normalized_requested_username in normalized_resolved_username
                or normalized_resolved_username in normalized_requested_username
            )
            if not matches:
                raise HTTPException(
                    status_code=401,
                    detail=(
                        "BisQue session user did not match the supplied username. "
                        "Please verify your credentials."
                    ),
                )

        return str(username or "").strip() or str(resolved_user or "").strip()

    def _get_bisque_auth_optional(
        request: FastAPIRequest,
        response: FastAPIResponse,
        bisque_session: str | None = Cookie(default=None, alias=bisque_session_cookie_name),
        anonymous_session: str | None = Cookie(default=None, alias=anonymous_session_cookie_name),
    ) -> dict[str, Any] | None:
        session = _get_bisque_session(bisque_session)
        if session:
            session = _refresh_bisque_session_from_browser_cookie_if_needed(
                session=session,
                request=request,
                response=response,
            )
            return _touch_bisque_session(session)
        bootstrap_session = _bootstrap_bisque_session_from_browser_cookie(
            request=request,
            response=response,
        )
        if bootstrap_session:
            return bootstrap_session
        return _issue_anonymous_auth_context(
            response,
            request=request,
            anonymous_session=anonymous_session,
        )

    def _get_existing_bisque_auth_optional(
        request: FastAPIRequest,
        response: FastAPIResponse,
        bisque_session: str | None = Cookie(default=None, alias=bisque_session_cookie_name),
    ) -> dict[str, Any] | None:
        session = _get_bisque_session(bisque_session)
        if session:
            session = _refresh_bisque_session_from_browser_cookie_if_needed(
                session=session,
                request=request,
                response=response,
            )
            return _touch_bisque_session(session)
        return _bootstrap_bisque_session_from_browser_cookie(
            request=request,
            response=response,
        )

    def _has_authenticated_browser_session(bisque_auth: dict[str, Any] | None) -> bool:
        if not bisque_auth or bool(bisque_auth.get("anonymous_session")):
            return False
        mode = str(bisque_auth.get("mode") or "").strip().lower()
        if mode == "guest":
            return True
        username = str(bisque_auth.get("username") or "").strip()
        return mode == "bisque" and bool(username)

    def _record_request_auth_context(
        request: FastAPIRequest,
        *,
        auth_source: str,
        user_id: str | None = None,
    ) -> None:
        request.state.auth_source = auth_source
        if user_id:
            request.state.auth_user_id = user_id

    def _resolve_authenticated_session_user_id(
        bisque_auth: dict[str, Any] | None,
    ) -> str | None:
        if not _has_authenticated_browser_session(bisque_auth):
            return None
        try:
            return _current_user_id(bisque_auth, allow_anonymous=False)
        except HTTPException:
            return None

    def _extract_presented_api_key(
        *,
        x_api_key: str | None,
        authorization: str | None,
        api_key_query: str | None,
    ) -> tuple[str, str | None]:
        presented = (x_api_key or "").strip()
        if presented:
            return presented, "header"

        auth_header = (authorization or "").strip()
        if auth_header.lower().startswith("bearer "):
            token = auth_header[7:].strip()
            if token:
                return token, "bearer"

        query_token = (api_key_query or "").strip()
        if query_token:
            return query_token, "query"
        return "", None

    def _validate_presented_api_key(
        request: FastAPIRequest,
        *,
        x_api_key: str | None,
        authorization: str | None,
        api_key_query: str | None,
    ) -> str:
        required_api_key = (getattr(settings, "orchestrator_api_key", None) or "").strip()
        if not required_api_key:
            _record_request_auth_context(request, auth_source="unprotected")
            return "unprotected"

        presented, source = _extract_presented_api_key(
            x_api_key=x_api_key,
            authorization=authorization,
            api_key_query=api_key_query,
        )
        if not presented or not source:
            raise HTTPException(status_code=401, detail="Unauthorized")
        if source == "query":
            if not bool(settings.resolved_allow_query_api_key_compat):
                raise HTTPException(status_code=401, detail="Unauthorized")
            request.state.query_api_key_compat_used = True
            logger.warning(
                "Deprecated query-string api_key authentication used for %s %s",
                request.method,
                request.url.path,
            )
        if presented != required_api_key:
            raise HTTPException(status_code=401, detail="Unauthorized")
        _record_request_auth_context(request, auth_source=f"api_key:{source}")
        return source

    def _require_authenticated_session(
        request: FastAPIRequest,
        bisque_auth: dict[str, Any] | None = Depends(_get_existing_bisque_auth_optional),
    ) -> dict[str, Any]:
        if not _has_authenticated_browser_session(bisque_auth):
            raise HTTPException(status_code=401, detail="Authentication required.")
        _record_request_auth_context(
            request,
            auth_source="browser_session",
            user_id=_resolve_authenticated_session_user_id(bisque_auth),
        )
        return cast(dict[str, Any], bisque_auth)

    def _require_authenticated_bisque_auth(
        bisque_auth: dict[str, Any] | None,
        *,
        detail: str = "BisQue authentication is required.",
    ) -> dict[str, Any]:
        if not bisque_auth:
            raise HTTPException(status_code=401, detail=detail)
        mode = str(bisque_auth.get("mode") or "").strip().lower()
        username = str(bisque_auth.get("username") or "").strip()
        if mode != "bisque" or not username:
            raise HTTPException(status_code=401, detail=detail)
        resolved_username, resolved_password, access_token, cookie_header = (
            _resolve_chat_bisque_transfer_auth(
                bisque_auth,
                allow_settings_fallback=False,
            )
        )
        if access_token or cookie_header or (resolved_username and resolved_password):
            return bisque_auth
        raise HTTPException(
            status_code=401,
            detail="BisQue session credentials are unavailable or expired. Sign in again and retry.",
        )

    def _get_effective_bisque_credentials(
        bisque_auth: dict[str, Any] | None,
    ) -> tuple[str | None, str | None]:
        configured_user = str(getattr(settings, "bisque_user", "") or "").strip()
        configured_password = str(getattr(settings, "bisque_password", "") or "").strip()
        if bisque_auth:
            mode = str(bisque_auth.get("mode") or "bisque").strip().lower()
            if mode != "bisque":
                return (None, None)
            session_username = str(bisque_auth.get("username") or "").strip()
            session_password = str(bisque_auth.get("password") or "").strip()
            return (session_username or None, session_password or None)
        return (
            configured_user or None,
            configured_password or None,
        )

    def _get_effective_bisque_access_token(
        bisque_auth: dict[str, Any] | None,
    ) -> str | None:
        if not bisque_auth:
            return None
        token = str(bisque_auth.get("access_token") or "").strip()
        if token:
            return token
        return None

    def _get_effective_bisque_cookie_header(
        bisque_auth: dict[str, Any] | None,
    ) -> str | None:
        if not bisque_auth:
            return None
        cookie_header = str(bisque_auth.get("bisque_cookie_header") or "").strip()
        return cookie_header or None

    def _current_user_id(
        bisque_auth: dict[str, Any] | None,
        *,
        allow_anonymous: bool = False,
    ) -> str:
        if not bisque_auth:
            if allow_anonymous:
                return "anonymous"
            raise HTTPException(status_code=401, detail="Authentication required.")
        mode = str(bisque_auth.get("mode") or "bisque").strip().lower()
        username = str(bisque_auth.get("username") or "").strip().lower()
        if mode == "bisque":
            if not username:
                raise HTTPException(status_code=401, detail="Authenticated username missing.")
            return f"bisque:{username}"
        session_id = str(bisque_auth.get("session_id") or "").strip().lower()
        guest_key = session_id or username or "guest"
        return f"guest:{guest_key}"

    def _resolve_request_user_id(
        bisque_auth: dict[str, Any] | None,
    ) -> str | None:
        try:
            return _current_user_id(bisque_auth, allow_anonymous=True)
        except HTTPException:
            return None

    def _assert_run_owner_access(
        *,
        run_id: str,
        request_user_id: str | None,
    ) -> None:
        metadata = store.get_run_metadata(run_id)
        run_owner = str((metadata or {}).get("user_id") or "").strip() or None
        if not run_owner:
            return
        if not request_user_id or run_owner != request_user_id:
            raise HTTPException(status_code=404, detail="Run not found")

    def _iso_from_ms(value: int) -> str:
        ts_ms = int(value)
        ts_sec = float(ts_ms) / 1000.0
        return datetime.utcfromtimestamp(ts_sec).isoformat() + "Z"

    def _ms_from_iso(value: str | None) -> int:
        raw = str(value or "").strip()
        if not raw:
            return int(time.time() * 1000)
        parsed = raw.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(parsed)
            return int(dt.timestamp() * 1000)
        except Exception:
            return int(time.time() * 1000)

    def _latest_user_text(messages: list[dict[str, str]]) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "user" and msg.get("content"):
                return str(msg["content"])
        return "chat request"

    def _selected_bisque_tool_names(
        messages: list[dict[str, str]],
        *,
        uploaded_files: list[str] | None = None,
        file_ids: list[str] | None = None,
    ) -> list[str]:
        upload_hints = [str(path or "") for path in (uploaded_files or [])]
        upload_hints.extend(
            str(file_id or "").strip() for file_id in (file_ids or []) if str(file_id or "").strip()
        )
        selected_tools = _select_tool_subset(
            messages,
            uploaded_files=upload_hints,
            all_tools=list(BISQUE_TOOL_SCHEMAS),
        )
        selected_names = {
            str(tool.get("function", {}).get("name") or "").strip()
            for tool in selected_tools
            if isinstance(tool, dict)
        }
        return sorted(name for name in selected_names if name)

    def _requires_bisque_auth_preflight(
        messages: list[dict[str, str]],
        *,
        uploaded_files: list[str] | None = None,
        file_ids: list[str] | None = None,
    ) -> list[str]:
        selected_names = _selected_bisque_tool_names(
            messages,
            uploaded_files=uploaded_files,
            file_ids=file_ids,
        )
        if not selected_names:
            return []
        user_text = _latest_user_text(messages).strip().lower()
        explicit_tool_mentions = {name for name in selected_names if name.lower() in user_text}
        strong_signal_tokens = (
            "data_service",
            "image_service",
            "client_service",
            "resource uri",
            "dataset uri",
            "gobject",
            "gobjects",
            "metadata tag",
            "annotation tag",
            "bisque module",
        )
        if explicit_tool_mentions or any(token in user_text for token in strong_signal_tokens):
            return selected_names
        return []

    def _ensure_run_artifact_dir(run_id: str) -> Path:
        path = artifact_root / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _artifact_listing_priority(entry: dict[str, Any]) -> tuple[int, str]:
        path = str(entry.get("path") or "").strip()
        title = str(entry.get("title") or "").strip()
        source_path = str(entry.get("source_path") or "").strip()
        haystack = f"{path} {title} {source_path}".lower()
        if path.startswith("uploads/"):
            return (0, path)
        if "matplotlib_annotated" in haystack:
            return (1, path)
        if path.startswith("tool_outputs/raw/"):
            return (2, path)
        if path.endswith("predictions.json"):
            return (3, path)
        if re.search(r"-\d{4}-x\d+-y\d+(?=\.[^.]+$)", haystack):
            return (5, path)
        return (4, path)

    def _list_artifacts(run_id: str, limit: int) -> list[ArtifactRecord]:
        run_dir = _ensure_run_artifact_dir(run_id)
        manifest = _materialize_manifest_if_missing(run_id)
        artifacts = manifest.get("artifacts")
        if isinstance(artifacts, list) and artifacts:
            records: list[ArtifactRecord] = []
            manifest_updated = False
            for item in sorted(
                (
                    entry
                    for entry in artifacts
                    if isinstance(entry, dict) and str(entry.get("path") or "").strip()
                ),
                key=_artifact_listing_priority,
            )[:limit]:
                raw_modified_at = item.get("modified_at")
                modified_at: datetime
                if isinstance(raw_modified_at, str):
                    try:
                        modified_at = datetime.fromisoformat(raw_modified_at.replace("Z", "+00:00"))
                    except ValueError:
                        modified_at = datetime.utcnow()
                else:
                    modified_at = datetime.utcnow()
                raw_path = str(item.get("path") or "").strip()
                source_path = str(item.get("source_path") or "").strip() or None
                _, resolved_rel_path = _resolve_artifact_path_alias(
                    run_id,
                    raw_path,
                    source_path=source_path,
                )
                artifact_path = resolved_rel_path or raw_path
                if resolved_rel_path and resolved_rel_path != raw_path:
                    item["path"] = resolved_rel_path
                    manifest_updated = True
                records.append(
                    ArtifactRecord(
                        path=artifact_path,
                        size_bytes=int(item.get("size_bytes") or 0),
                        mime_type=(
                            str(item.get("mime_type"))
                            if item.get("mime_type") is not None
                            else mimetypes.guess_type(artifact_path)[0]
                        ),
                        modified_at=modified_at,
                        source_path=source_path,
                        title=str(item.get("title") or "") or None,
                        result_group_id=str(item.get("result_group_id") or "").strip() or None,
                    )
                )
            if manifest_updated:
                _write_manifest(run_id, manifest)
            if records:
                return records

        records = []
        for file_path in sorted(run_dir.rglob("*")):
            if not file_path.is_file():
                continue
            stat = file_path.stat()
            records.append(
                ArtifactRecord(
                    path=str(file_path.relative_to(run_dir)),
                    size_bytes=int(stat.st_size),
                    mime_type=mimetypes.guess_type(file_path.name)[0],
                    modified_at=datetime.utcfromtimestamp(stat.st_mtime),
                    result_group_id=None,
                )
            )
            if len(records) >= limit:
                break
        return records

    def _safe_artifact_name(name: str) -> str:
        sanitized = re.sub(r"[^a-zA-Z0-9._-]+", "_", (name or "").strip())
        sanitized = re.sub(r"_+", "_", sanitized).strip("._")
        return sanitized or "file"

    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _safe_user_storage_segment(user_id: str | None) -> str:
        raw = str(user_id or "anonymous").strip().lower()
        sanitized = re.sub(r"[^a-z0-9._-]+", "_", raw).strip("._")
        return sanitized or "anonymous"

    def _safe_upload_local_path(raw_path: str | None) -> Path | None:
        candidate_raw = str(raw_path or "").strip()
        if not candidate_raw:
            return None
        try:
            candidate = Path(candidate_raw).expanduser().resolve()
            root = upload_store_root.resolve()
            candidate.relative_to(root)
            return candidate
        except Exception:
            return None

    def _upload_staging_dir_for_user(user_id: str | None) -> Path:
        destination = upload_staging_root / _safe_user_storage_segment(user_id)
        destination.mkdir(parents=True, exist_ok=True)
        return destination

    def _upload_cache_dir_for_user(user_id: str | None) -> Path:
        destination = upload_cache_root / _safe_user_storage_segment(user_id)
        destination.mkdir(parents=True, exist_ok=True)
        return destination

    def _infer_resource_kind(
        *,
        original_name: str,
        content_type: str | None,
        source_uri: str | None = None,
    ) -> str:
        lowered_name = str(original_name or "").strip().lower()
        lowered_type = str(content_type or "").strip().lower()
        lowered_uri = str(source_uri or "").strip().lower()
        if "/data_service/table" in lowered_uri:
            return "table"
        if "/data_service/image" in lowered_uri:
            return "image"
        if lowered_type.startswith("image/"):
            return "image"
        if lowered_type.startswith("video/"):
            return "video"
        if any(
            lowered_name.endswith(ext)
            for ext in [
                ".png",
                ".jpg",
                ".jpeg",
                ".gif",
                ".bmp",
                ".webp",
                ".tif",
                ".tiff",
                ".ome.tif",
                ".ome.tiff",
                ".czi",
                ".nd2",
                ".nii",
                ".nii.gz",
                ".nrrd",
                ".mha",
                ".mhd",
                ".svs",
                ".lif",
                ".lsm",
                ".vsi",
                ".dv",
                ".r3d",
            ]
        ):
            return "image"
        if any(
            lowered_name.endswith(ext) for ext in [".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"]
        ):
            return "video"
        if any(
            lowered_name.endswith(ext)
            for ext in [".csv", ".tsv", ".parquet", ".xlsx", ".xls", ".feather", ".jsonl"]
        ):
            return "table"
        return "file"

    def _request_scoped_bisque_auth_material(
        bisque_auth: dict[str, Any] | None,
    ) -> tuple[str | None, str | None, str | None, str | None, str | None]:
        request_context = get_request_bisque_auth()
        username = str(getattr(request_context, "username", "") or "").strip() or None
        password = str(getattr(request_context, "password", "") or "").strip() or None
        access_token = str(getattr(request_context, "access_token", "") or "").strip() or None
        cookie_header = (
            str(getattr(request_context, "bisque_cookie_header", "") or "").strip() or None
        )
        bisque_root = (
            str(getattr(request_context, "bisque_root", "") or "").strip().rstrip("/") or None
        )
        if not any((username, password, access_token, cookie_header)) and bisque_auth:
            mode = str(bisque_auth.get("mode") or "").strip().lower()
            username = str(bisque_auth.get("username") or "").strip() or None
            access_token = str(bisque_auth.get("access_token") or "").strip() or None
            cookie_header = str(bisque_auth.get("bisque_cookie_header") or "").strip() or None
            bisque_root = (
                str(bisque_auth.get("bisque_root") or "").strip().rstrip("/") or bisque_root
            )
            if mode == "bisque":
                password = str(bisque_auth.get("password") or "").strip() or None
        if not (access_token or cookie_header):
            if username and not password:
                username = None
            if password and not username:
                password = None
        return username, password, access_token, cookie_header, bisque_root

    def _session_bisque_auth_material(
        session_id: str | None,
    ) -> tuple[str | None, str | None, str | None, str | None, str | None]:
        session = _get_bisque_session(session_id)
        if not session:
            return None, None, None, None, None
        session = _touch_bisque_session(session)
        return (
            str(session.get("username") or "").strip() or None,
            str(session.get("password") or "").strip() or None,
            str(session.get("access_token") or "").strip() or None,
            str(session.get("bisque_cookie_header") or "").strip() or None,
            str(session.get("bisque_root") or "").strip().rstrip("/") or None,
        )

    def _resolve_upload_bisque_auth_material(
        upload_row: dict[str, Any],
        bisque_auth: dict[str, Any] | None = None,
    ) -> tuple[str | None, str | None, str | None, str | None, str | None]:
        username, password, access_token, cookie_header, bisque_root = (
            _request_scoped_bisque_auth_material(bisque_auth)
        )
        if any((username, password, access_token, cookie_header)):
            return (
                username,
                password,
                access_token,
                cookie_header,
                bisque_root
                or str(getattr(settings, "bisque_root", "") or "").strip().rstrip("/")
                or None,
            )

        metadata = (
            upload_row.get("metadata") if isinstance(upload_row.get("metadata"), dict) else {}
        )
        session_username, session_password, session_token, session_cookie, session_root = (
            _session_bisque_auth_material(
                str(metadata.get("bisque_session_id") or "").strip() or None
            )
        )
        if any((session_username, session_password, session_token, session_cookie)):
            return (
                session_username,
                session_password,
                session_token,
                session_cookie,
                session_root
                or str(getattr(settings, "bisque_root", "") or "").strip().rstrip("/")
                or None,
            )

        configured_root = (
            str(getattr(settings, "bisque_root", "") or "").strip().rstrip("/") or None
        )
        configured_user = str(getattr(settings, "bisque_user", "") or "").strip() or None
        configured_password = str(getattr(settings, "bisque_password", "") or "").strip() or None
        return configured_user, configured_password, None, None, configured_root

    def _catalog_file_id_for_upload(upload_row: dict[str, Any]) -> str:
        sync_status = str(upload_row.get("sync_status") or "").strip().lower()
        canonical = str(upload_row.get("canonical_resource_uniq") or "").strip()
        if sync_status == upload_sync_status_succeeded and canonical:
            return canonical
        return str(upload_row.get("file_id") or "").strip()

    def _merge_upload_metadata(
        upload_row: dict[str, Any],
        updates: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        merged = (
            dict(upload_row.get("metadata") or {})
            if isinstance(upload_row.get("metadata"), dict)
            else {}
        )
        if isinstance(updates, dict):
            for key, value in updates.items():
                if value is None:
                    merged.pop(key, None)
                else:
                    merged[key] = value
        return merged

    def _persist_upload_row(upload_row: dict[str, Any], **updates: Any) -> None:
        merged = dict(upload_row)
        merged.update(updates)
        metadata_updates = updates.get("metadata_updates")
        metadata_value = updates.get("metadata", merged.get("metadata"))
        if metadata_updates is not None:
            metadata_value = _merge_upload_metadata(upload_row, dict(metadata_updates or {}))
        store.put_upload(
            file_id=str(merged.get("file_id") or ""),
            original_name=str(merged.get("original_name") or "upload.bin"),
            stored_path=str(merged.get("stored_path") or ""),
            content_type=str(merged.get("content_type") or "").strip() or None,
            size_bytes=int(merged.get("size_bytes") or 0),
            sha256=str(merged.get("sha256") or ""),
            created_at=str(merged.get("created_at") or datetime.utcnow().isoformat()),
            user_id=str(merged.get("user_id") or "").strip() or None,
            source_type=str(merged.get("source_type") or "upload"),
            source_uri=str(merged.get("source_uri") or "").strip() or None,
            client_view_url=str(merged.get("client_view_url") or "").strip() or None,
            image_service_url=str(merged.get("image_service_url") or "").strip() or None,
            resource_kind=str(merged.get("resource_kind") or "").strip() or None,
            thumbnail_path=str(merged.get("thumbnail_path") or "").strip() or None,
            metadata=metadata_value if isinstance(metadata_value, dict) else None,
            deleted_at=str(merged.get("deleted_at") or "").strip() or None,
            staging_path=str(merged.get("staging_path") or "").strip() or None,
            cache_path=str(merged.get("cache_path") or "").strip() or None,
            canonical_resource_uniq=str(merged.get("canonical_resource_uniq") or "").strip()
            or None,
            canonical_resource_uri=str(merged.get("canonical_resource_uri") or "").strip() or None,
            sync_status=str(merged.get("sync_status") or "").strip() or None,
            sync_error=str(merged.get("sync_error") or "").strip() or None,
            sync_retry_count=int(merged.get("sync_retry_count") or 0),
            sync_started_at=str(merged.get("sync_started_at") or "").strip() or None,
            sync_completed_at=str(merged.get("sync_completed_at") or "").strip() or None,
            sync_run_id=str(merged.get("sync_run_id") or "").strip() or None,
        )

    def _build_upload_analysis_context(
        *,
        stored_path: str | None,
        original_name: str,
        resource_kind: str | None,
    ) -> dict[str, Any]:
        source_path = Path(str(stored_path or "").strip()).expanduser()
        if not source_path.exists() or not source_path.is_file():
            return {}
        normalized_kind = str(resource_kind or "").strip().lower()
        if normalized_kind not in {"image", "video"}:
            suffix = source_path.suffix.lower()
            if suffix not in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".gif"}:
                return {}
        settings_obj = get_settings()
        metadata_root = Path(
            str(getattr(settings_obj, "science_data_root", "data/science") or "data/science")
        ).expanduser()
        summary = extract_actionable_image_metadata(
            file_path=str(source_path),
            output_root=str((metadata_root / "upload-metadata").resolve()),
        )
        if not isinstance(summary, dict) or not bool(summary.get("success")):
            return {}
        return {
            "original_name": str(original_name or source_path.name).strip() or source_path.name,
            "summary": {
                key: value
                for key, value in summary.items()
                if key != "success" and value not in (None, {}, [])
            },
            "version": "v1",
            "preprocessed_at": datetime.utcnow().isoformat() + "Z",
        }

    def _enrich_upload_metadata_context(upload_row: dict[str, Any]) -> None:
        metadata = (
            upload_row.get("metadata") if isinstance(upload_row.get("metadata"), dict) else {}
        )
        existing = (
            metadata.get("analysis_context")
            if isinstance(metadata.get("analysis_context"), dict)
            else {}
        )
        if existing.get("version") == "v1":
            return
        analysis_context = _build_upload_analysis_context(
            stored_path=str(
                upload_row.get("stored_path") or upload_row.get("staging_path") or ""
            ).strip()
            or None,
            original_name=str(upload_row.get("original_name") or "").strip() or "upload.bin",
            resource_kind=str(upload_row.get("resource_kind") or "").strip() or None,
        )
        if not analysis_context:
            return
        _persist_upload_row(upload_row, metadata_updates={"analysis_context": analysis_context})

    def _existing_upload_local_path(upload_row: dict[str, Any]) -> Path | None:
        for key in ("cache_path", "staging_path", "stored_path"):
            candidate = _safe_upload_local_path(str(upload_row.get(key) or ""))
            if candidate is not None and candidate.exists() and candidate.is_file():
                return candidate
        return None

    def _cache_path_for_upload(upload_row: dict[str, Any], user_id: str | None) -> Path:
        cache_dir = _upload_cache_dir_for_user(user_id)
        token = (
            str(upload_row.get("canonical_resource_uniq") or "").strip()
            or str(upload_row.get("file_id") or "").strip()
            or uuid4().hex
        )
        original_name = str(upload_row.get("original_name") or token)
        return cache_dir / f"{token}__{_safe_artifact_name(original_name)}"

    def _materialize_upload_local_path(
        upload_row: dict[str, Any],
        *,
        user_id: str | None,
        bisque_auth: dict[str, Any] | None = None,
    ) -> Path | None:
        existing = _existing_upload_local_path(upload_row)
        if existing is not None:
            return existing

        resource_uri = (
            str(upload_row.get("canonical_resource_uri") or "").strip()
            or str(upload_row.get("source_uri") or "").strip()
        )
        if not resource_uri:
            return None
        resource_uniq = str(
            upload_row.get("canonical_resource_uniq") or ""
        ).strip() or _extract_bisque_resource_uniq(resource_uri)
        destination = _cache_path_for_upload(upload_row, user_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp_path = destination.parent / f".materialize-{uuid4().hex}.part"
        username, password, access_token, cookie_header, bisque_root = (
            _resolve_upload_bisque_auth_material(upload_row, bisque_auth)
        )
        if not bisque_root:
            return None
        try:
            download = _download_bisque_resource_to_temp(
                resource_uri=resource_uri,
                resource_uniq=resource_uniq,
                temp_path=temp_path,
                bisque_username=username,
                bisque_password=password,
                bisque_access_token=access_token,
                bisque_cookie_header=cookie_header,
                allow_settings_fallback=False,
            )
            temp_path.replace(destination)
            _persist_upload_row(
                upload_row,
                stored_path=str(destination),
                cache_path=str(destination),
                content_type=str(download.get("content_type") or "").strip()
                or str(upload_row.get("content_type") or "").strip()
                or None,
                size_bytes=int(download.get("size_bytes") or upload_row.get("size_bytes") or 0),
                sha256=str(download.get("sha256") or upload_row.get("sha256") or ""),
            )
            refreshed = store.get_upload(str(upload_row.get("file_id") or ""), user_id=user_id)
            if refreshed is not None:
                upload_row.clear()
                upload_row.update(refreshed)
            return destination
        finally:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)

    def _ensure_upload_local_path(
        upload_row: dict[str, Any],
        *,
        user_id: str | None,
        bisque_auth: dict[str, Any] | None = None,
    ) -> Path | None:
        existing = _existing_upload_local_path(upload_row)
        if existing is not None:
            return existing
        return _materialize_upload_local_path(upload_row, user_id=user_id, bisque_auth=bisque_auth)

    def _resolved_local_path_for_upload(
        upload_row: dict[str, Any],
        *,
        user_id: str | None,
        bisque_auth: dict[str, Any] | None = None,
    ) -> str | None:
        cached = str(upload_row.get("_resolved_local_path") or "").strip()
        if cached:
            cached_path = _safe_upload_local_path(cached)
            if cached_path is not None and cached_path.exists() and cached_path.is_file():
                return str(cached_path)

        resolved = _ensure_upload_local_path(upload_row, user_id=user_id, bisque_auth=bisque_auth)
        if resolved is None or not resolved.exists() or not resolved.is_file():
            return None
        token = str(resolved)
        upload_row["_resolved_local_path"] = token
        return token

    def _thumbnail_root_for_user(user_id: str | None) -> Path:
        return upload_store_root / ".thumbnails" / _safe_user_storage_segment(user_id)

    thumbnail_schema_version = "v2"

    def _thumbnail_path_for_upload(file_id: str, user_id: str | None) -> Path:
        return _thumbnail_root_for_user(user_id) / f"{file_id}-{thumbnail_schema_version}.png"

    def _build_upload_thumbnail(
        *,
        upload_row: dict[str, Any],
        user_id: str,
        bisque_auth: dict[str, Any] | None = None,
    ) -> Path | None:
        file_id = str(upload_row.get("file_id") or "").strip()
        if not file_id:
            return None

        cached_path = _safe_upload_local_path(str(upload_row.get("thumbnail_path") or ""))
        if cached_path and cached_path.exists() and cached_path.is_file():
            return cached_path

        source_path = _ensure_upload_local_path(
            upload_row,
            user_id=user_id,
            bisque_auth=bisque_auth,
        )
        if source_path is None or not source_path.exists() or not source_path.is_file():
            return None

        destination = _thumbnail_path_for_upload(file_id, user_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.is_file():
            store.update_upload_thumbnail(
                file_id=file_id,
                user_id=user_id,
                thumbnail_path=str(destination),
            )
            return destination

        try:
            payload = load_scientific_image(
                file_path=str(source_path),
                array_mode="plane",
                generate_preview=False,
                save_array=False,
                include_array=False,
                max_inline_elements=1024,
                return_array=True,
            )
            if not payload.get("success"):
                return None
            image_array = payload.pop("_array", None)
            if image_array is None:
                return None
            image_u8 = _render_preview_image(image_array)
            image = Image.fromarray(image_u8)
            resampling = getattr(Image, "Resampling", Image)
            image.thumbnail((640, 640), getattr(resampling, "LANCZOS", Image.LANCZOS))
            image.save(destination, format="PNG", optimize=True)
            store.update_upload_thumbnail(
                file_id=file_id,
                user_id=user_id,
                thumbnail_path=str(destination),
            )
            return destination
        except Exception:
            pass

        try:
            with Image.open(source_path) as image:
                with suppress(Exception):
                    image.seek(0)
                if image.mode not in {"RGB", "RGBA"}:
                    image = image.convert("RGB")
                resampling = getattr(Image, "Resampling", Image)
                image.thumbnail((640, 640), getattr(resampling, "LANCZOS", Image.LANCZOS))
                image.save(destination, format="PNG", optimize=True)
            store.update_upload_thumbnail(
                file_id=file_id,
                user_id=user_id,
                thumbnail_path=str(destination),
            )
            return destination
        except Exception:
            return None

    def _resource_thumbnail_url(file_id: str) -> str:
        return f"/v1/resources/{file_id}/thumbnail"

    def _resource_preview_url(file_id: str) -> str:
        return f"/v1/uploads/{file_id}/preview"

    def _resource_record_from_upload(upload_row: dict[str, Any]) -> ResourceRecord:
        created_at_raw = str(upload_row.get("created_at") or "").strip()
        created_at = datetime.utcnow()
        if created_at_raw:
            try:
                created_at = datetime.fromisoformat(created_at_raw.replace("Z", "+00:00"))
            except Exception:
                created_at = datetime.utcnow()
        file_id = _catalog_file_id_for_upload(upload_row)
        original_name = str(upload_row.get("original_name") or "upload.bin")
        content_type = str(upload_row.get("content_type") or "").strip() or None
        source_uri = (
            str(upload_row.get("canonical_resource_uri") or "").strip()
            or str(upload_row.get("source_uri") or "").strip()
            or None
        )
        resource_kind = str(upload_row.get("resource_kind") or "").strip().lower()
        if not resource_kind:
            resource_kind = _infer_resource_kind(
                original_name=original_name,
                content_type=content_type,
                source_uri=source_uri,
            )
        thumbnail_path = _safe_upload_local_path(str(upload_row.get("thumbnail_path") or ""))
        has_thumbnail = bool(
            thumbnail_path and thumbnail_path.exists() and thumbnail_path.is_file()
        )
        cache_ready = _existing_upload_local_path(upload_row) is not None
        staged_locally = bool(
            _safe_upload_local_path(str(upload_row.get("staging_path") or ""))
            and _safe_upload_local_path(str(upload_row.get("staging_path") or "")).exists()
        )
        supports_thumbnail = resource_kind == "image"
        return ResourceRecord(
            file_id=file_id,
            original_name=original_name,
            content_type=content_type,
            size_bytes=int(upload_row.get("size_bytes") or 0),
            sha256=str(upload_row.get("sha256") or ""),
            created_at=created_at,
            source_type=str(upload_row.get("source_type") or "upload"),
            resource_kind=resource_kind,
            source_uri=source_uri,
            client_view_url=str(upload_row.get("client_view_url") or "").strip() or None,
            image_service_url=str(upload_row.get("image_service_url") or "").strip() or None,
            has_thumbnail=has_thumbnail,
            thumbnail_url=_resource_thumbnail_url(file_id) if supports_thumbnail else None,
            preview_url=_resource_preview_url(file_id) if supports_thumbnail else None,
            sync_status=str(upload_row.get("sync_status") or "").strip() or None,
            sync_error=str(upload_row.get("sync_error") or "").strip() or None,
            canonical_resource_uniq=(
                str(upload_row.get("canonical_resource_uniq") or "").strip() or None
            ),
            canonical_resource_uri=(
                str(upload_row.get("canonical_resource_uri") or "").strip() or None
            ),
            cache_ready=cache_ready,
            staged_locally=staged_locally,
            sync_run_id=str(upload_row.get("sync_run_id") or "").strip() or None,
        )

    def _extract_bisque_resource_uniq(resource_uri: str | None) -> str | None:
        if not resource_uri:
            return None
        value = str(resource_uri).strip().rstrip("/")
        if not value:
            return None
        return value.split("/")[-1] or None

    def _normalize_bisque_resource_uri(resource: str, bisque_root: str) -> str:
        return _normalize_bisque_resource_uri_with_root(resource, bisque_root)

    def _build_bisque_links(resource: str, bisque_root: str) -> dict[str, str | None]:
        return _build_bisque_links_for_root(resource, bisque_root)

    def _build_bisque_upload_xml(upload_row: dict[str, Any], fallback_name: str) -> str:
        resource_kind = str(upload_row.get("resource_kind") or "").strip().lower()
        tag_name = resource_kind if resource_kind in {"image", "table", "file"} else "file"
        resource_name = str(upload_row.get("original_name") or "").strip() or str(fallback_name)
        resource_node = ElementTree.Element(tag_name, name=resource_name)
        return ElementTree.tostring(resource_node, encoding="unicode")

    def _parse_content_disposition_filename(value: str | None) -> str | None:
        if not value:
            return None
        match_utf = re.search(r"filename\*=UTF-8''([^;]+)", value, flags=re.IGNORECASE)
        if match_utf:
            return unquote(match_utf.group(1)).strip().strip('"')
        match_std = re.search(r'filename="([^"]+)"', value, flags=re.IGNORECASE)
        if match_std:
            return match_std.group(1).strip()
        match_bare = re.search(r"filename=([^;]+)", value, flags=re.IGNORECASE)
        if match_bare:
            return match_bare.group(1).strip().strip('"')
        return None

    def _probe_is_gzip(probe: bytes) -> bool:
        return len(probe) >= 2 and probe[0] == 0x1F and probe[1] == 0x8B

    def _probe_is_nifti_header(probe: bytes) -> bool:
        if len(probe) < 4:
            return False
        little = int.from_bytes(probe[:4], byteorder="little", signed=False)
        big = int.from_bytes(probe[:4], byteorder="big", signed=False)
        return little in {348, 540} or big in {348, 540}

    def _normalize_download_filename(
        *,
        original_name: str | None,
        content_type: str,
        probe: bytes,
        resource_uniq: str | None,
    ) -> str:
        fallback_name = str(original_name or "").strip() or f"{resource_uniq or 'bisque-resource'}"
        lowered = fallback_name.lower()
        content_type_lower = str(content_type or "").lower()

        def _guess_ext_from_probe(payload: bytes) -> str:
            if payload.startswith(b"\x89PNG\r\n\x1a\n"):
                return ".png"
            if payload.startswith(b"\xff\xd8\xff"):
                return ".jpg"
            if payload.startswith(b"II*\x00") or payload.startswith(b"MM\x00*"):
                return ".tif"
            if payload.startswith(b"GIF87a") or payload.startswith(b"GIF89a"):
                return ".gif"
            if payload.startswith(b"RIFF") and b"WEBP" in payload[:32]:
                return ".webp"
            return ""

        looks_like_nifti = (
            lowered.endswith(".nii")
            or lowered.endswith(".nii.gz")
            or "nifti" in content_type_lower
            or _probe_is_nifti_header(probe)
        )
        if not looks_like_nifti:
            if Path(fallback_name).suffix:
                return fallback_name
            guessed_ext = (
                mimetypes.guess_extension(content_type_lower.split(";", 1)[0].strip()) or ""
            )
            if guessed_ext == ".jpe":
                guessed_ext = ".jpg"
            if not guessed_ext:
                guessed_ext = _guess_ext_from_probe(probe)
            if guessed_ext:
                return f"{fallback_name}{guessed_ext}"
            return fallback_name

        is_gzip_payload = _probe_is_gzip(probe)
        if is_gzip_payload:
            if lowered.endswith(".nii.gz"):
                return fallback_name
            if lowered.endswith(".nii"):
                return f"{fallback_name}.gz"
            trimmed = re.sub(r"(?i)\.gz$", "", fallback_name).rstrip(".")
            return f"{trimmed or (resource_uniq or 'bisque-resource')}.nii.gz"

        if lowered.endswith(".nii.gz"):
            return re.sub(r"(?i)\.gz$", "", fallback_name)
        if lowered.endswith(".nii"):
            return fallback_name
        trimmed = re.sub(r"(?i)\.gz$", "", fallback_name).rstrip(".")
        return f"{trimmed or (resource_uniq or 'bisque-resource')}.nii"

    def _download_bisque_resource_to_temp(
        *,
        resource_uri: str,
        resource_uniq: str | None,
        temp_path: Path,
        bisque_username: str | None = None,
        bisque_password: str | None = None,
        bisque_access_token: str | None = None,
        bisque_cookie_header: str | None = None,
        allow_settings_fallback: bool = True,
    ) -> dict[str, Any]:
        def _looks_like_html_or_xml(content_type: str, probe: bytes) -> bool:
            lowered_type = content_type.lower()
            if "text/html" in lowered_type or "application/xhtml" in lowered_type:
                return True
            if lowered_type in {"application/xml", "text/xml"}:
                return True
            head = probe.lstrip().lower()
            return (
                head.startswith(b"<!doctype html")
                or head.startswith(b"<html")
                or head.startswith(b"<?xml")
            )

        parsed_resource = urlparse(resource_uri)
        resource_root = (
            f"{parsed_resource.scheme}://{parsed_resource.netloc}"
            if parsed_resource.scheme and parsed_resource.netloc
            else str(getattr(settings, "bisque_root", "") or "").rstrip("/")
        )
        image_service_url = (
            f"{resource_root}/image_service/{resource_uniq}" if resource_uniq else None
        )
        blob_url: str | None = None
        resource_path = (parsed_resource.path or "").rstrip("/")
        if resource_path and not resource_path.endswith("/blob"):
            blob_url = f"{resource_root}{resource_path}/blob"

        headers = {"Accept": "*/*"}
        resolved_token = str(bisque_access_token or "").strip()
        resolved_cookie_header = str(bisque_cookie_header or "").strip()
        resolved_user = str(bisque_username or "").strip()
        resolved_password = str(bisque_password or "").strip()
        if allow_settings_fallback and not resolved_token:
            if not resolved_user:
                resolved_user = str(getattr(settings, "bisque_user", "") or "").strip()
            if not resolved_password:
                resolved_password = str(getattr(settings, "bisque_password", "") or "").strip()

        if resolved_token:
            headers["Authorization"] = f"Bearer {resolved_token}"
        elif resolved_cookie_header:
            headers["Cookie"] = resolved_cookie_header
        elif resolved_user and resolved_password:
            creds = f"{resolved_user}:{resolved_password}".encode()
            headers["Authorization"] = f"Basic {b64encode(creds).decode('ascii')}"
        attempts: list[str] = []
        candidate_urls = [url for url in [image_service_url, blob_url, resource_uri] if url]

        def _download_from_url(candidate_url: str, source: str) -> dict[str, Any]:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)

            request = Request(candidate_url, headers=headers)
            with urlopen(request, timeout=60) as response:
                content_type = (
                    str(response.headers.get("Content-Type") or "").split(";", 1)[0].strip()
                    or "application/octet-stream"
                )
                disposition_name = _parse_content_disposition_filename(
                    response.headers.get("Content-Disposition")
                )
                digest = hashlib.sha256()
                total_bytes = 0
                probe = b""
                with temp_path.open("wb") as out:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        if len(probe) < 1024:
                            probe += chunk[: 1024 - len(probe)]
                        total_bytes += len(chunk)
                        if total_bytes > upload_max_file_size_bytes:
                            raise HTTPException(
                                status_code=413,
                                detail=(
                                    f"BisQue resource exceeded maximum file size of "
                                    f"{upload_max_file_size_bytes // (1024 * 1024)} MB: {candidate_url}"
                                ),
                            )
                        digest.update(chunk)
                        out.write(chunk)

                if _looks_like_html_or_xml(content_type, probe):
                    temp_path.unlink(missing_ok=True)
                    raise ValueError(
                        f"{source} returned HTML/XML content instead of file bytes (content-type={content_type})."
                    )

                original_name = disposition_name
                if not original_name:
                    guessed_ext = (
                        mimetypes.guess_extension(content_type)
                        or Path(urlparse(candidate_url).path).suffix
                        or ""
                    )
                    fallback_name = resource_uniq or "bisque-resource"
                    original_name = f"{fallback_name}{guessed_ext}"
                original_name = _normalize_download_filename(
                    original_name=original_name,
                    content_type=content_type,
                    probe=probe,
                    resource_uniq=resource_uniq,
                )

                return {
                    "original_name": _safe_artifact_name(original_name),
                    "content_type": content_type,
                    "size_bytes": total_bytes,
                    "sha256": digest.hexdigest(),
                    "download_source": source,
                }

        for candidate_url in candidate_urls:
            source = "resource_uri"
            if "/image_service/" in candidate_url:
                source = "image_service"
            elif candidate_url.endswith("/blob"):
                source = "resource_blob"
            try:
                return _download_from_url(candidate_url, source)
            except HTTPError as exc:
                detail = getattr(exc, "reason", None) or str(exc)
                attempts.append(f"{source} HTTP {exc.code}: {detail}")
            except URLError as exc:
                attempts.append(f"{source} URL error: {exc.reason}")
            except Exception as exc:
                attempts.append(f"{source} error: {exc}")

        if temp_path.exists():
            temp_path.unlink(missing_ok=True)

        try:
            from bqapi.comm import BQSession  # type: ignore
        except Exception:
            attempts.append("bqapi fallback unavailable (bqapi not installed).")
        else:
            if not resolved_user or not resolved_password:
                attempts.append("bqapi fallback unavailable (BISQUE_USER/BISQUE_PASSWORD not set).")
            else:
                try:
                    bq = BQSession()
                    bq.init_local(
                        resolved_user,
                        resolved_password,
                        bisque_root=resource_root,
                    )
                    resource = bq.load(resource_uri, view="deep")
                    blob_uri = getattr(resource, "blob", None) if resource is not None else None
                    if not blob_uri:
                        blob_uri = f"{resource_uri}/blob"
                    bq.fetchblob(blob_uri, path=str(temp_path))
                    if not temp_path.exists() or not temp_path.is_file():
                        raise ValueError("bqapi fetchblob completed without writing a local file.")

                    with temp_path.open("rb") as handle:
                        probe = handle.read(1024)
                    if _looks_like_html_or_xml("", probe):
                        temp_path.unlink(missing_ok=True)
                        raise ValueError("bqapi fallback returned HTML/XML instead of file bytes.")

                    original_name = getattr(resource, "name", None) or None
                    if not original_name:
                        guessed_ext = Path(urlparse(resource_uri).path).suffix
                        original_name = f"{resource_uniq or 'bisque-resource'}{guessed_ext}"
                    original_name = _normalize_download_filename(
                        original_name=original_name,
                        content_type=mimetypes.guess_type(original_name)[0]
                        or "application/octet-stream",
                        probe=probe,
                        resource_uniq=resource_uniq,
                    )
                    original_name = _safe_artifact_name(original_name)
                    content_type = (
                        mimetypes.guess_type(original_name)[0] or "application/octet-stream"
                    )
                    size_bytes = int(temp_path.stat().st_size)
                    sha256 = _sha256_file(temp_path)
                    return {
                        "original_name": original_name,
                        "content_type": content_type,
                        "size_bytes": size_bytes,
                        "sha256": sha256,
                        "download_source": "bqapi_blob",
                    }
                except Exception as exc:
                    if temp_path.exists():
                        temp_path.unlink(missing_ok=True)
                    attempts.append(f"bqapi fallback error: {exc}")

        details = " | ".join(attempts[-4:]) if attempts else "Unknown download failure."
        raise ValueError(f"BisQue download failed after all strategies. {details}")

    def _artifact_entry(
        run_id: str, path: Path, *, source_path: str | None = None
    ) -> dict[str, Any]:
        run_dir = _ensure_run_artifact_dir(run_id).resolve()
        resolved = path.resolve()
        try:
            rel_path = str(resolved.relative_to(run_dir))
        except ValueError:
            rel_path = resolved.name
        stat = resolved.stat()
        return {
            "path": rel_path,
            "size_bytes": int(stat.st_size),
            "mime_type": mimetypes.guess_type(resolved.name)[0],
            "modified_at": datetime.utcfromtimestamp(stat.st_mtime).isoformat() + "Z",
            "sha256": _sha256_file(resolved),
            "source_path": source_path,
            "result_group_id": None,
        }

    def _manifest_path(run_id: str) -> Path:
        return _ensure_run_artifact_dir(run_id) / "artifact_manifest.json"

    def _load_manifest(run_id: str) -> dict[str, Any]:
        path = _manifest_path(run_id)
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    payload.setdefault("run_id", run_id)
                    payload.setdefault("artifacts", [])
                    return payload
            except Exception:
                pass
        return {
            "run_id": run_id,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "artifacts": [],
        }

    def _write_manifest(run_id: str, manifest: dict[str, Any]) -> Path:
        manifest["run_id"] = run_id
        manifest["generated_at"] = datetime.utcnow().isoformat() + "Z"
        manifest_path = _manifest_path(run_id)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest_path

    def _resolve_artifact_path_alias(
        run_id: str,
        raw_path: str,
        *,
        source_path: str | None = None,
    ) -> tuple[Path | None, str | None]:
        run_dir = _ensure_run_artifact_dir(run_id).resolve()
        normalized_raw = str(raw_path or "").replace("\\", "/").strip()
        normalized_source = str(source_path or "").replace("\\", "/").strip()
        candidate_specs: list[tuple[str, Path]] = []
        seen_candidates: set[str] = set()

        def _append_candidate(path: Path) -> None:
            try:
                resolved = path.resolve()
                resolved.relative_to(run_dir)
            except Exception:
                return
            token = str(resolved)
            if token in seen_candidates:
                return
            seen_candidates.add(token)
            candidate_specs.append((token, resolved))

        def _append_relative(raw_value: str) -> None:
            if not raw_value:
                return
            relative = Path(
                *[part for part in Path(raw_value).parts if part not in {"", ".", ".."}]
            )
            if not str(relative):
                return
            _append_candidate(run_dir / relative)
            if relative.parts[0] not in {"workspace", "tool_outputs", "uploads"}:
                _append_candidate(run_dir / "workspace" / relative)
                _append_candidate(run_dir / "tool_outputs" / relative)

        _append_relative(normalized_raw)
        if normalized_source and not Path(normalized_source).is_absolute():
            _append_relative(normalized_source)

        basename = Path(normalized_raw or normalized_source).name
        if basename:
            suffix_preferences = [
                normalized_raw,
                normalized_source,
                f"workspace/{normalized_raw}" if normalized_raw else "",
                f"workspace/{normalized_source}"
                if normalized_source and not Path(normalized_source).is_absolute()
                else "",
            ]
            matches = sorted(
                (path.resolve() for path in run_dir.rglob(basename) if path.is_file()),
                key=lambda path: (
                    next(
                        (
                            index
                            for index, suffix in enumerate(suffix_preferences)
                            if suffix
                            and str(path.relative_to(run_dir)).replace("\\", "/").endswith(suffix)
                        ),
                        len(suffix_preferences),
                    ),
                    len(path.parts),
                    str(path),
                ),
            )
            for match in matches[:12]:
                _append_candidate(match)

        for _, candidate in candidate_specs:
            if candidate.exists() and candidate.is_file():
                return candidate, str(candidate.relative_to(run_dir)).replace("\\", "/")
        return None, None

    def _resolve_run_artifact_source_path(run_dir: Path, raw_path: str) -> Path | None:
        candidate_raw = str(raw_path or "").strip()
        if not candidate_raw:
            return None
        candidate = Path(candidate_raw).expanduser()
        search_order: list[Path] = []
        if candidate.is_absolute():
            search_order.append(candidate)
        else:
            search_order.extend([run_dir / candidate, candidate])

        basename = candidate.name.strip()
        if basename:
            search_order.extend(
                [
                    run_dir / "uploads" / basename,
                    run_dir / "tool_outputs" / basename,
                ]
            )

        seen: set[str] = set()
        for option in search_order:
            try:
                resolved = option.resolve()
            except Exception:
                resolved = option
            key = str(resolved)
            if key in seen:
                continue
            seen.add(key)
            try:
                if option.exists() and option.is_file():
                    return option.resolve()
            except Exception:
                continue
        return None

    def _backfill_yolo_annotated_artifacts(run_id: str, manifest: dict[str, Any]) -> dict[str, Any]:
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            return manifest

        if any(
            isinstance(item, dict) and "matplotlib_annotated" in str(item.get("path") or "")
            for item in artifacts
        ):
            return manifest

        run_dir = _ensure_run_artifact_dir(run_id).resolve()
        tool_output_dir = run_dir / "tool_outputs"
        tool_output_dir.mkdir(parents=True, exist_ok=True)

        prediction_entries = [
            item
            for item in artifacts
            if isinstance(item, dict)
            and str(item.get("path") or "").startswith("tool_outputs/")
            and str(item.get("path") or "").endswith("predictions.json")
        ]
        if not prediction_entries:
            return manifest

        added_entries: list[dict[str, Any]] = []
        existing_paths = {
            str(item.get("path") or "")
            for item in artifacts
            if isinstance(item, dict) and str(item.get("path") or "").strip()
        }

        for entry in prediction_entries:
            predictions_path = _resolve_run_artifact_source_path(
                run_dir,
                str(entry.get("source_path") or "") or str(entry.get("path") or ""),
            )
            if predictions_path is None:
                continue
            try:
                payload = json.loads(predictions_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            predictions = payload.get("predictions")
            if not isinstance(predictions, list):
                continue

            for index, prediction in enumerate(predictions):
                if not isinstance(prediction, dict):
                    continue
                source_path = _resolve_run_artifact_source_path(
                    run_dir,
                    str(prediction.get("input_path") or prediction.get("path") or ""),
                )
                if source_path is None:
                    continue
                boxes = prediction.get("boxes") if isinstance(prediction.get("boxes"), list) else []
                logical_name = _safe_artifact_name(
                    f"{index:03d}-{source_path.stem}__matplotlib_annotated.png"
                )
                temp_root = (
                    Path(tempfile.gettempdir()) / "bisque_ultra_yolo_backfill" / str(run_id).strip()
                )
                temp_root.mkdir(parents=True, exist_ok=True)
                temp_output = temp_root / logical_name
                render_result = _render_yolo_detection_figure(
                    source_path=str(source_path),
                    boxes=boxes,
                    output_path=temp_output,
                )
                if not bool(render_result.get("success")) or not temp_output.exists():
                    continue

                file_sha = _sha256_file(temp_output)
                destination = tool_output_dir / f"{file_sha[:12]}__{logical_name}"
                if not destination.exists():
                    shutil.copy2(temp_output, destination)
                rel_path = str(destination.relative_to(run_dir))
                if rel_path in existing_paths:
                    continue

                artifact = _artifact_entry(run_id, destination, source_path=str(destination))
                artifact["category"] = "tool_output"
                artifact["tool"] = "yolo_detect"
                artifact["kind"] = "image"
                artifact["title"] = logical_name
                added_entries.append(artifact)
                existing_paths.add(rel_path)

        if not added_entries:
            return manifest
        return _update_manifest_with_entries(run_id, added_entries)

    def _backfill_progress_output_artifacts(
        run_id: str, manifest: dict[str, Any]
    ) -> dict[str, Any]:
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, list):
            artifacts = []

        progress_events: list[dict[str, Any]] = []
        for event in store.list_events(run_id, limit=300):
            if str(event.get("event_type") or "").strip() != "chat_done_payload":
                continue
            payload = event.get("payload")
            payload_dict = payload if isinstance(payload, dict) else {}
            response = payload_dict.get("response")
            response_dict = response if isinstance(response, dict) else {}
            progress_items = response_dict.get("progress_events")
            if not isinstance(progress_items, list):
                continue
            progress_events.extend(
                item for item in progress_items[:300] if isinstance(item, dict)
            )

        if not progress_events:
            return manifest

        existing_paths = {
            str(item.get("path") or "").strip()
            for item in artifacts
            if isinstance(item, dict) and str(item.get("path") or "").strip()
        }
        added_entries: list[dict[str, Any]] = []
        for entry in _snapshot_progress_artifacts(run_id, progress_events):
            rel_path = str(entry.get("path") or "").strip()
            if not rel_path or rel_path in existing_paths:
                continue
            existing_paths.add(rel_path)
            added_entries.append(entry)

        if not added_entries:
            return manifest
        return _update_manifest_with_entries(run_id, added_entries)

    def _update_manifest_with_entries(run_id: str, entries: list[dict[str, Any]]) -> dict[str, Any]:
        manifest = _load_manifest(run_id)
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, list):
            artifacts = []

        by_path: dict[str, dict[str, Any]] = {}
        for item in artifacts:
            if isinstance(item, dict) and item.get("path"):
                by_path[str(item["path"])] = item
        for entry in entries:
            if isinstance(entry, dict) and entry.get("path"):
                by_path[str(entry["path"])] = entry

        manifest["artifacts"] = [by_path[key] for key in sorted(by_path.keys())]
        manifest["artifact_count"] = len(manifest["artifacts"])
        manifest_path = _write_manifest(run_id, manifest)
        store.append_event(
            run_id,
            "artifact_manifest_updated",
            {
                "path": str(manifest_path),
                "artifact_count": manifest["artifact_count"],
            },
        )
        return manifest

    def _snapshot_uploaded_files(
        run_id: str, uploaded_files: list[str]
    ) -> tuple[list[str], list[dict[str, Any]], list[str]]:
        run_dir = _ensure_run_artifact_dir(run_id)
        upload_dir = run_dir / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)

        snap_paths: list[str] = []
        entries: list[dict[str, Any]] = []
        missing: list[str] = []

        for raw in uploaded_files or []:
            src = Path(str(raw)).expanduser()
            if not src.exists() or not src.is_file():
                missing.append(str(src))
                continue

            sha256 = _sha256_file(src)
            safe_name = _safe_artifact_name(src.name)
            candidate = upload_dir / f"{sha256[:12]}__{safe_name}"
            if candidate.exists() and candidate.is_file():
                try:
                    if _sha256_file(candidate) != sha256:
                        candidate = upload_dir / f"{sha256[:12]}__{int(time.time())}__{safe_name}"
                except Exception:
                    candidate = upload_dir / f"{sha256[:12]}__{int(time.time())}__{safe_name}"

            if not candidate.exists():
                shutil.copy2(src, candidate)

            snap_paths.append(str(candidate))
            entry = _artifact_entry(run_id, candidate, source_path=str(src))
            entry["category"] = "upload"
            entries.append(entry)

        return snap_paths, entries, missing

    def _snapshot_progress_artifacts(
        run_id: str, progress_events: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if not progress_events:
            return []

        image_exts = {
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".webp",
            ".bmp",
            ".svg",
            ".avif",
            ".tif",
            ".tiff",
        }
        web_image_exts = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg", ".avif"}
        artifact_file_exts = {
            ".json",
            ".jsonl",
            ".csv",
            ".tsv",
            ".txt",
            ".md",
            ".npy",
            ".npz",
            ".nii",
            ".h5",
            ".hdf5",
            ".parquet",
            ".xml",
        }
        run_dir = _ensure_run_artifact_dir(run_id)
        artifact_dir = run_dir / "tool_outputs"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        repo_root = Path(__file__).resolve().parents[2]

        def _resolve_existing_path(raw_path: str, *, allow_directory: bool = False) -> Path | None:
            candidate_raw = str(raw_path or "").strip()
            if not candidate_raw:
                return None
            candidate = Path(candidate_raw).expanduser()
            search_order = [candidate]
            if not candidate.is_absolute():
                search_order.append(run_dir / candidate)
                search_order.append(repo_root / candidate)
            for option in search_order:
                try:
                    if option.exists() and (
                        option.is_file() or (allow_directory and option.is_dir())
                    ):
                        return option.resolve()
                except Exception:
                    continue
            return None

        def _resolve_source_path(raw_path: str) -> Path | None:
            resolved = _resolve_existing_path(raw_path)
            if resolved is None or not resolved.is_file():
                return None
            return resolved

        def _is_supported_artifact(path: Path) -> bool:
            suffix = path.suffix.lower()
            name = path.name.lower()
            if suffix in image_exts or suffix in artifact_file_exts:
                return True
            return bool(name.endswith(".nii.gz"))

        def _is_image_artifact(path: Path) -> bool:
            return path.suffix.lower() in image_exts

        def _save_web_preview(source_path: Path, source_sha: str, safe_name: str) -> Path | None:
            if not _is_image_artifact(source_path):
                return None
            if source_path.suffix.lower() in web_image_exts:
                return None
            try:
                base_name = Path(safe_name).stem
                preview_name = _safe_artifact_name(f"{base_name}__preview.png")
                preview_path = artifact_dir / f"{source_sha[:12]}__{preview_name}"
                if preview_path.exists() and preview_path.is_file():
                    return preview_path
                with Image.open(source_path) as image:
                    with suppress(Exception):
                        image.seek(0)
                    if image.mode not in {"RGB", "RGBA"}:
                        image = image.convert("RGB")
                    resampling = getattr(Image, "Resampling", Image)
                    image.thumbnail((1600, 1600), getattr(resampling, "LANCZOS", Image.LANCZOS))
                    image.save(preview_path, format="PNG", optimize=True)
                return preview_path
            except Exception:
                return None

        entries: list[dict[str, Any]] = []
        copied_paths: set[str] = set()

        for event in progress_events:
            if not isinstance(event, dict):
                continue
            artifacts = event.get("artifacts")
            candidate_items: list[Any] = []
            summary = event.get("summary")
            summary_result_group_id = (
                str(summary.get("result_group_id") or "").strip()
                if isinstance(summary, dict)
                else ""
            )
            if isinstance(artifacts, list) and artifacts:
                candidate_items.extend(artifacts[:160])
            output_dir_raw = summary.get("output_directory") if isinstance(summary, dict) else None
            output_dir = _resolve_existing_path(str(output_dir_raw or ""), allow_directory=True)
            if output_dir is not None and output_dir.is_dir():
                discovered: list[dict[str, str]] = []
                for child in output_dir.rglob("*"):
                    if not child.is_file():
                        continue
                    if not _is_supported_artifact(child):
                        continue
                    discovered.append({"path": str(child), "title": child.name})
                    if len(discovered) >= 160:
                        break
                candidate_items.extend(discovered)
            if not candidate_items:
                continue
            tool_name = str(event.get("tool") or "tool")
            for item in candidate_items[:160]:
                if isinstance(item, dict):
                    source_raw = item.get("path")
                    title = str(item.get("title") or "").strip() or None
                    result_group_id = (
                        str(item.get("result_group_id") or "").strip()
                        or summary_result_group_id
                        or None
                    )
                else:
                    source_raw = item
                    title = None
                    result_group_id = summary_result_group_id or None

                source_path = _resolve_source_path(str(source_raw or ""))
                if source_path is None:
                    continue
                if not _is_supported_artifact(source_path):
                    continue

                source_key = str(source_path.resolve())
                if source_key in copied_paths:
                    continue
                source_sha = _sha256_file(source_path)
                safe_name = _safe_artifact_name(source_path.name)
                dest_path = artifact_dir / f"{source_sha[:12]}__{safe_name}"
                if not dest_path.exists():
                    shutil.copy2(source_path, dest_path)
                copied_paths.add(source_key)

                entry = _artifact_entry(run_id, dest_path, source_path=str(source_path))
                entry["category"] = "tool_output"
                entry["tool"] = tool_name
                entry["kind"] = "image" if _is_image_artifact(source_path) else "file"
                entry["result_group_id"] = result_group_id
                if title:
                    entry["title"] = title
                entries.append(entry)

                preview_path = _save_web_preview(source_path, source_sha, safe_name)
                if preview_path is not None:
                    preview_entry = _artifact_entry(
                        run_id, preview_path, source_path=str(source_path)
                    )
                    preview_entry["category"] = "tool_output_preview"
                    preview_entry["tool"] = tool_name
                    preview_entry["result_group_id"] = result_group_id
                    preview_entry["title"] = f"{title or source_path.name} (preview)"
                    entries.append(preview_entry)

        deduped_by_path: dict[str, dict[str, Any]] = {}
        for entry in entries:
            path_key = str(entry.get("path") or "").strip()
            if not path_key:
                continue
            deduped_by_path[path_key] = entry
        return list(deduped_by_path.values())

    def _resolve_file_ids(
        file_ids: list[str],
        *,
        user_id: str | None = None,
        bisque_auth: dict[str, Any] | None = None,
    ) -> tuple[list[str], list[dict[str, Any]], list[str]]:
        resolved_paths: list[str] = []
        resolved_records: list[dict[str, Any]] = []
        missing: list[str] = []

        for raw_id in file_ids or []:
            file_id = str(raw_id or "").strip()
            if not file_id:
                continue
            row = store.get_upload(file_id, user_id=user_id)
            if not row:
                missing.append(file_id)
                continue
            resolved_path = _resolved_local_path_for_upload(
                row,
                user_id=user_id,
                bisque_auth=bisque_auth,
            )
            if not resolved_path:
                missing.append(file_id)
                continue
            resolved_paths.append(resolved_path)
            resolved_records.append(row)
        return resolved_paths, resolved_records, missing

    def _format_size_mb(size_bytes: int | float | None) -> str:
        size = float(size_bytes or 0.0)
        return f"{size / (1024 * 1024):.2f} MB"

    def _build_file_context(rows: list[dict[str, Any]]) -> str:
        if not rows:
            return ""

        preview_lines: list[str] = []
        metadata_lines: list[str] = []
        pdf_names: list[str] = []
        prairie_hint_names: list[str] = []
        for row in rows[:100]:
            file_name = str(
                row.get("original_name") or Path(str(row.get("path") or "")).name or "file"
            )
            size_bytes = int(row.get("size_bytes") or 0)
            path = str(row.get("path") or "").strip()
            file_id = str(row.get("file_id") or "").strip()
            client_view_url = str(row.get("client_view_url") or "").strip()
            suffix = Path(file_name).suffix.lower()
            content_type = str(row.get("content_type") or "").lower()
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            analysis_context = (
                metadata.get("analysis_context")
                if isinstance(metadata.get("analysis_context"), dict)
                else {}
            )
            analysis_summary = (
                analysis_context.get("summary")
                if isinstance(analysis_context.get("summary"), dict)
                else {}
            )
            if suffix == ".pdf" or content_type == "application/pdf":
                pdf_names.append(file_name)
            lowered_hint = re.sub(r"[^a-z0-9]+", " ", f"{file_name} {path}".lower()).strip()
            if (
                lowered_hint
                and any(token in lowered_hint for token in ("prairie", "burrow", "rarespot"))
                and file_name not in prairie_hint_names
            ):
                prairie_hint_names.append(file_name)

            line = f"- {file_name} ({_format_size_mb(size_bytes)})"
            if file_id:
                line += f" [file_id={file_id}]"
            if path:
                line += f" at {path}"
            if client_view_url:
                line += f" [BisQue view={client_view_url}]"
            preview_lines.append(line)
            metadata_bits: list[str] = []
            dimensions = (
                analysis_summary.get("dimensions")
                if isinstance(analysis_summary.get("dimensions"), dict)
                else {}
            )
            x_dim = dimensions.get("X") if isinstance(dimensions, dict) else None
            y_dim = dimensions.get("Y") if isinstance(dimensions, dict) else None
            if isinstance(x_dim, (int, float)) and isinstance(y_dim, (int, float)):
                metadata_bits.append(f"size={int(x_dim)}x{int(y_dim)}")
            captured_at = str(analysis_summary.get("captured_at") or "").strip()
            if captured_at:
                metadata_bits.append(f"captured_at={captured_at}")
            geo = (
                analysis_summary.get("geo") if isinstance(analysis_summary.get("geo"), dict) else {}
            )
            latitude = geo.get("latitude") if isinstance(geo, dict) else None
            longitude = geo.get("longitude") if isinstance(geo, dict) else None
            if latitude is not None and longitude is not None:
                metadata_bits.append(f"gps={latitude},{longitude}")
            insights = (
                analysis_summary.get("actionable_insights")
                if isinstance(analysis_summary.get("actionable_insights"), list)
                else []
            )
            if metadata_bits or insights:
                rendered = metadata_bits + [
                    str(item).strip()
                    for item in insights[:4]
                    if str(item or "").strip() and str(item or "").strip() not in metadata_bits
                ]
                if rendered:
                    metadata_lines.append(f"- {file_name}: " + "; ".join(rendered))

        context = (
            f"The user provided {len(rows)} file(s) available on the server for this run:\n"
            + "\n".join(preview_lines)
            + "\n\nThese are server-managed paths available to tools in this turn. "
        )
        if metadata_lines:
            context += (
                "\n\nPreprocessed metadata context is already available for some uploaded images:\n"
                + "\n".join(metadata_lines[:40])
                + "\nUse this metadata to ground follow-up analysis, especially image size, acquisition time, and GPS context when present. "
            )

        if pdf_names:
            context += (
                f"\n\nThe user uploaded {len(pdf_names)} PDF file(s):\n"
                + "\n".join([f"- {name}" for name in pdf_names[:50]])
                + "\nPDF OCR/text extraction is not enabled in this deployment. "
                "Do not claim to have read document text from these files unless the user provides the text directly."
            )
        else:
            prairie_hint_block = ""
            if prairie_hint_names:
                prairie_hint_block = (
                    "\n\nPrairie-model hint: some uploaded filenames suggest the prairie dog workflow:\n"
                    + "\n".join([f"- {name}" for name in prairie_hint_names[:20]])
                    + "\nIf the user asks for detection on those files, prefer the prairie YOLO model."
                )
            context += (
                "If the user asks to upload them to BisQue, use the upload_to_bisque tool. "
                "If the user names a destination dataset in plain language, pass that dataset name directly; do not ask for a dataset URI first when the name is already present. "
                "If the user wants to group existing BisQue resources into a dataset, use bisque_add_to_dataset. "
                "If the user asks to describe/summarize image files, start with bioio_load_image to report metadata/header findings first. "
                "For scientific volumes such as NIfTI, use bioio_load_image first to summarize axes, dimensions, channels, and orientation/header cues before heavier inference. "
                "For microscopy stacks, OME-TIFF, confocal, light-sheet, multiphoton, time-series, or multichannel volumes, start with bioio_load_image before segmentation or quantification. "
                "Only run segmentation/detection/depth when the user explicitly asks for those analyses. "
                "Only use combined segmentation+evaluation workflows when label/ground-truth mask files are actually available. "
                "For segmentation follow-ups, prefer quantitative outputs such as connected components, morphology, overlap, distance, or tabular summaries via execute_python_job instead of prose-only summaries. "
                "If the user explicitly asks for Megaseg or DynUNet microscopy inference, call segment_image_megaseg. "
                "If the user asks for automatic segmentation, call segment_image_sam2 first. "
                "Only use segment_image_sam3 when the user explicitly asks for SAM3 or concept-prompt/region-guided segmentation. "
                "If the user asks for depth estimation, call estimate_depth_pro first; "
                "for follow-up segmentation on depth outputs, use returned depth_map_paths with segment_image_sam2. "
                "If segmentation tools return preferred_upload_path or preferred_upload_paths, use those paths for upload_to_bisque "
                "(do not upload overlay visualizations unless the user explicitly asks). "
                "If the user asks to run detection on uploaded images, call yolo_detect directly on these local paths. "
                "For generic detection, keep the pretrained baseline unless the user names a model or explicitly asks for the latest finetuned detector. "
                "If the user explicitly asks for prairie dog or burrow detection, use the prairie model "
                "(model_name='yolov5_rarespot', which resolves to the active shared prairie checkpoint or RareSpot baseline). "
                "If the user names a specific model, honor it. "
                "If the detection target is ambiguous and the user did not name the object or model, ask one short clarifying question "
                "before choosing between generic YOLO and a specialized detector."
            )
            context += prairie_hint_block
        return context

    def _estimate_text_tokens(value: str) -> int:
        text = str(value or "")
        if not text:
            return 0
        # Lightweight approximation suitable for trigger-based compaction.
        byte_based = max(1, (len(text.encode("utf-8")) + 3) // 4)
        word_based = len(re.findall(r"\w+|[^\w\s]", text))
        return max(byte_based, word_based // 2)

    def _estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
        total = 0
        for message in messages:
            role = str(message.get("role") or "")
            content = str(message.get("content") or "")
            total += 4 + _estimate_text_tokens(role) + _estimate_text_tokens(content)
        return max(0, total)

    def _compact_message_content(value: str, max_len: int) -> str:
        collapsed = re.sub(r"\s+", " ", str(value or "").strip())
        if len(collapsed) <= max_len:
            return collapsed
        return f"{collapsed[: max(1, max_len - 1)]}…"

    def _conversation_messages_hash(messages: list[dict[str, Any]]) -> str:
        hasher = hashlib.sha256()
        for message in messages:
            role = str(message.get("role") or "").strip().lower()
            content = re.sub(r"\s+", " ", str(message.get("content") or "").strip())
            hasher.update(role.encode("utf-8", errors="ignore"))
            hasher.update(b"\x00")
            hasher.update(content.encode("utf-8", errors="ignore"))
            hasher.update(b"\x1f")
        return hasher.hexdigest()

    def _recent_turn_start_index(messages: list[dict[str, Any]], keep_user_turns: int) -> int:
        if not messages:
            return 0
        user_turn_target = max(1, int(keep_user_turns))
        seen_user_turns = 0
        for index in range(len(messages) - 1, -1, -1):
            role = str(messages[index].get("role") or "").strip().lower()
            if role == "user":
                seen_user_turns += 1
                if seen_user_turns >= user_turn_target:
                    return index
        return 0

    def _extract_scientific_highlights(text: str, max_items: int = 6) -> list[str]:
        normalized = str(text or "")
        if not normalized:
            return []
        highlights: list[str] = []
        seen: set[str] = set()
        number_hits = re.findall(
            r"\b\d+(?:\.\d+)?(?:\s?(?:%|ms|s|m|h|px|kb|mb|gb|cells?|objects?|masks?|files?|images?|slices?))?\b",
            normalized,
            flags=re.IGNORECASE,
        )
        for hit in number_hits:
            candidate = hit.strip()
            lowered = candidate.lower()
            if not candidate or lowered in seen:
                continue
            seen.add(lowered)
            highlights.append(candidate)
            if len(highlights) >= max_items:
                return highlights

        tool_hits = re.findall(
            r"\b(segment_image_sam3|yolo_detect|estimate_depth_pro|sam3|yolo|depth|segmentation|detection|artifact|report)\b",
            normalized,
            flags=re.IGNORECASE,
        )
        for hit in tool_hits:
            candidate = hit.strip()
            lowered = candidate.lower()
            if not candidate or lowered in seen:
                continue
            seen.add(lowered)
            highlights.append(candidate)
            if len(highlights) >= max_items:
                break
        return highlights

    def _extract_referenced_files(text: str, max_items: int = 12) -> list[str]:
        normalized = str(text or "")
        if not normalized:
            return []
        file_hits = re.findall(
            r"\b[\w.\-]+\.(?:tif|tiff|png|jpg|jpeg|csv|json|npy|nii|nrrd|czi|nd2|ome\.tif|ome\.tiff)\b",
            normalized,
            flags=re.IGNORECASE,
        )
        output: list[str] = []
        seen: set[str] = set()
        for hit in file_hits:
            lowered = hit.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            output.append(hit)
            if len(output) >= max_items:
                break
        return output

    def _extract_run_refs(text: str, max_items: int = 8) -> list[str]:
        normalized = str(text or "")
        if not normalized:
            return []
        run_hits = re.findall(r"\brun[-_ ]?([0-9a-f]{6,32})\b", normalized, flags=re.IGNORECASE)
        output: list[str] = []
        seen: set[str] = set()
        for hit in run_hits:
            compact = hit.lower()[:12]
            if compact in seen:
                continue
            seen.add(compact)
            output.append(compact)
            if len(output) >= max_items:
                break
        return output

    def _summarize_compacted_prefix(
        messages: list[dict[str, Any]],
        *,
        max_chars: int,
        max_turns: int = 24,
    ) -> tuple[str, dict[str, Any]]:
        turns: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None

        def _flush_current() -> None:
            nonlocal current
            if current is None:
                return
            turns.append(current)
            current = None

        for message in messages:
            role = str(message.get("role") or "").strip().lower()
            content = str(message.get("content") or "").strip()
            if not content:
                continue
            if role == "user":
                _flush_current()
                current = {
                    "user_goal": _compact_message_content(content, 260),
                    "assistant_outcomes": [],
                    "highlights": _extract_scientific_highlights(content, max_items=4),
                }
                continue
            if role != "assistant":
                continue
            if current is None:
                current = {
                    "user_goal": "(implicit or prior request)",
                    "assistant_outcomes": [],
                    "highlights": [],
                }
            outcome = _compact_message_content(content, 320)
            outcomes = current.get("assistant_outcomes")
            if isinstance(outcomes, list) and len(outcomes) < 2:
                outcomes.append(outcome)
            highlights = current.get("highlights")
            if not isinstance(highlights, list):
                highlights = []
            for item in _extract_scientific_highlights(content, max_items=6):
                if item not in highlights:
                    highlights.append(item)
                if len(highlights) >= 6:
                    break
            current["highlights"] = highlights
        _flush_current()

        selected_turns = turns[-max(1, int(max_turns)) :]
        lines = [
            "Compacted memory of earlier conversation turns (research continuity context):",
        ]
        for index, turn in enumerate(selected_turns, start=1):
            user_goal = _compact_message_content(str(turn.get("user_goal") or ""), 220)
            lines.append(f"{index}. User objective: {user_goal or 'n/a'}")
            outcomes = turn.get("assistant_outcomes")
            outcome_list = outcomes if isinstance(outcomes, list) else []
            if outcome_list:
                lines.append(
                    f"   Assistant outcome: {_compact_message_content(str(outcome_list[0] or ''), 260)}"
                )
            highlights = turn.get("highlights")
            highlight_list = highlights if isinstance(highlights, list) else []
            if highlight_list:
                lines.append(
                    "   Key scientific details: "
                    + ", ".join(
                        _compact_message_content(str(item or ""), 36) for item in highlight_list[:5]
                    )
                )

        aggregate_text = "\n".join(str(message.get("content") or "") for message in messages)
        referenced_files = _extract_referenced_files(aggregate_text, max_items=12)
        if referenced_files:
            lines.append("Referenced files: " + ", ".join(referenced_files))
        run_refs = _extract_run_refs(aggregate_text, max_items=8)
        if run_refs:
            lines.append("Internal run refs seen earlier: " + ", ".join(run_refs))

        summary_text = "\n".join(lines).strip()
        if len(summary_text) > max_chars:
            summary_text = summary_text[: max(1, max_chars - 1)].rstrip() + "…"

        summary_payload = {
            "turn_count": len(selected_turns),
            "source_message_count": len(messages),
            "referenced_files": referenced_files,
            "run_refs": run_refs,
        }
        return summary_text, summary_payload

    def _build_conversation_memory_system_message(summary_text: str) -> dict[str, str]:
        return {
            "role": "system",
            "content": (
                "Persistent conversation memory summary (older turns compacted):\n"
                + str(summary_text or "").strip()
                + "\n\nUse this as continuity context. Prioritize exact numeric outputs, files, and run provenance when relevant."
            ),
        }

    def _apply_conversation_compaction(
        *,
        user_id: str | None,
        conversation_id: str | None,
        messages: list[dict[str, Any]],
        system_context_messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        del user_id
        del conversation_id
        normalized_messages = [
            {
                "role": str(message.get("role") or "").strip().lower(),
                "content": str(message.get("content") or "").strip(),
            }
            for message in messages
            if isinstance(message, dict)
            and str(message.get("role") or "").strip()
            and str(message.get("content") or "").strip()
        ]
        meta: dict[str, Any] = {
            "enabled": bool(context_compaction_enabled),
            "used": False,
            "original_message_count": len(normalized_messages),
            "compacted_message_count": len(normalized_messages),
            "token_estimate_before": (
                _estimate_messages_tokens(system_context_messages)
                + _estimate_messages_tokens(normalized_messages)
            ),
            "token_estimate_after": (
                _estimate_messages_tokens(system_context_messages)
                + _estimate_messages_tokens(normalized_messages)
            ),
            "reason": "not_needed",
            "summary_chars": 0,
            "summary_hash": "",
            "persisted_memory": False,
            "reused_summary": False,
        }
        if not context_compaction_enabled:
            meta["reason"] = "disabled"
            return normalized_messages, meta
        if len(normalized_messages) <= 1:
            meta["reason"] = "not_needed"
            return normalized_messages, meta
        if len(normalized_messages) < context_compaction_min_messages:
            meta["reason"] = "below_min_messages"
            return normalized_messages, meta
        if int(meta["token_estimate_before"]) < context_compaction_trigger_tokens:
            meta["reason"] = "below_trigger_tokens"
            return normalized_messages, meta

        compacted_messages = list(normalized_messages)
        available_target_tokens = max(
            96,
            int(context_compaction_target_tokens)
            - _estimate_messages_tokens(system_context_messages),
        )
        while (
            _estimate_messages_tokens(compacted_messages) > available_target_tokens
            and len(compacted_messages) > 1
        ):
            compacted_messages.pop(0)

        token_estimate_after = _estimate_messages_tokens(
            system_context_messages
        ) + _estimate_messages_tokens(compacted_messages)
        meta.update(
            {
                "used": True,
                "reason": "truncated",
                "compacted_message_count": len(compacted_messages),
                "token_estimate_after": token_estimate_after,
                "summary_chars": 0,
                "summary_hash": "",
                "persisted_memory": False,
                "reused_summary": False,
                "prefix_message_count": max(
                    0,
                    len(normalized_messages) - len(compacted_messages),
                ),
                "tail_message_count": len(compacted_messages),
            }
        )
        return compacted_messages, meta

    def _trim_history_text(value: str, max_len: int = 180) -> str:
        collapsed = re.sub(r"\s+", " ", str(value or "").strip())
        if len(collapsed) <= max_len:
            return collapsed
        return f"{collapsed[: max_len - 1]}…"

    def _format_history_timestamp(value: str) -> str:
        raw = str(value or "").strip()
        if not raw:
            return "unknown"
        normalized = raw.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except Exception:
            return _trim_history_text(raw.replace("T", " "), 36)
        prefix = parsed.strftime("%Y-%m-%d %H:%M")
        offset = parsed.strftime("%z")
        if not offset:
            return prefix
        if offset == "+0000":
            return f"{prefix} UTC"
        return f"{prefix} {offset}"

    def _extract_last_user_message(messages: list[dict[str, Any]]) -> str:
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            if str(message.get("role") or "").strip().lower() != "user":
                continue
            content = str(message.get("content") or "").strip()
            if content:
                return content
        return ""

    def _normalize_search_text(value: str) -> str:
        lowered = str(value or "").strip().lower()
        normalized = re.sub(r"[^a-z0-9_.:/\\-]+", " ", lowered)
        return re.sub(r"\s+", " ", normalized).strip()

    def _stem_search_token(token: str) -> str:
        if token.endswith("ies") and len(token) > 4:
            return token[:-3] + "y"
        if token.endswith("s") and len(token) > 3 and not token.endswith("ss"):
            return token[:-1]
        return token

    def _extract_history_search_terms(latest_user_text: str, max_terms: int = 10) -> list[str]:
        normalized_query = _normalize_search_text(latest_user_text)
        if not normalized_query:
            return []

        stopwords = {
            "what",
            "about",
            "from",
            "that",
            "this",
            "with",
            "have",
            "been",
            "were",
            "your",
            "there",
            "which",
            "previous",
            "history",
            "chat",
            "chats",
            "analysis",
            "session",
            "please",
            "find",
            "show",
            "tell",
            "list",
            "using",
            "file",
            "files",
            "run",
            "runs",
            "did",
            "can",
            "you",
            "for",
            "the",
            "and",
        }
        alias_map: dict[str, list[str]] = {
            "detect": ["yolo", "yolo_detect", "detection", "detections", "bbox", "bounding"],
            "segment": [
                "sam",
                "sam2",
                "sam3",
                "medsam",
                "medsam2",
                "segmentation",
                "mask",
                "masks",
            ],
            "depth": ["depth", "depthpro", "depth_map", "estimate_depth_pro"],
            "quant": ["quantification", "metrics", "measurements"],
            "csv": ["csv", "table", "dataframe", "analyze_csv"],
            "upload": ["upload", "uploaded", "import", "bisque"],
        }

        phrases = [
            _normalize_search_text(match)
            for match in re.findall(
                r'"([^"]{2,120})"|\'([^\']{2,120})\'', str(latest_user_text or "")
            )
            for match in match
            if match
        ]
        terms: list[str] = []
        seen: set[str] = set()

        def _push(term: str) -> None:
            value = _normalize_search_text(term)
            if not value:
                return
            if value in seen:
                return
            seen.add(value)
            terms.append(value)

        for phrase in phrases:
            _push(phrase)

        for token in normalized_query.split(" "):
            clean = token.strip("._:-")
            if not clean or clean in stopwords:
                continue
            if len(clean) < 2 and "." not in clean:
                continue
            _push(clean)
            _push(_stem_search_token(clean))
            if "." in clean:
                _push(Path(clean).stem)
            for key, aliases in alias_map.items():
                if clean.startswith(key) or key in clean:
                    for alias in aliases:
                        _push(alias)
            if len(terms) >= max_terms * 2:
                break

        return terms[:max_terms]

    def _score_query_match(haystack_text: str, query_text: str, terms: list[str]) -> float:
        normalized_haystack = _normalize_search_text(haystack_text)
        if not normalized_haystack:
            return 0.0
        score = 0.0
        normalized_query = _normalize_search_text(query_text)
        if normalized_query and normalized_query in normalized_haystack:
            score += 8.0

        haystack_tokens = normalized_haystack.split(" ")
        for term in terms:
            normalized_term = _normalize_search_text(term)
            if not normalized_term:
                continue
            if normalized_term in normalized_haystack:
                score += 2.8 if len(normalized_term) >= 5 else 1.8
                continue
            if any(
                token.startswith(normalized_term) or normalized_term.startswith(token)
                for token in haystack_tokens
                if len(token) >= 3
            ):
                score += 0.75
        return score

    def _should_attach_history_context(latest_user_text: str) -> bool:
        normalized = _normalize_search_text(latest_user_text)
        if not normalized:
            return False

        if re.search(
            r"\b(run[-_ ]?[0-9a-f]{6,}|[0-9a-f]{8,32}|[\w.-]+\.(tif|tiff|png|jpg|jpeg|csv|npy|nii|nrrd|czi|nd2|ome\.tif|ome\.tiff))\b",
            normalized,
        ):
            return True

        tokens = set(normalized.split(" "))
        intent_terms = {
            "what",
            "which",
            "find",
            "locate",
            "show",
            "list",
            "did",
            "have",
            "where",
            "when",
            "remember",
            "recall",
            "compare",
        }
        domain_terms = {
            "history",
            "previous",
            "prior",
            "earlier",
            "last",
            "yesterday",
            "analysis",
            "analyses",
            "run",
            "runs",
            "detection",
            "segment",
            "segmentation",
            "file",
            "files",
            "uploaded",
            "upload",
            "result",
            "results",
            "conversation",
            "conversations",
            "chat",
            "chats",
        }

        has_intent = any(token in intent_terms for token in tokens)
        has_domain = any(token in domain_terms for token in tokens)
        if has_intent and has_domain:
            return True
        if "?" in str(latest_user_text or "") and has_domain:
            return True

        keyword_patterns = [
            r"\b(previous|prior|earlier)\s+(chat|conversation|session|run|analysis)\b",
            r"\bchat history\b",
            r"\bhave i run\b",
            r"\bwhat analysis\b",
            r"\bwhat did i run\b",
            r"\bacross chats?\b",
        ]
        return any(re.search(pattern, normalized) for pattern in keyword_patterns)

    def _extract_run_tools_and_files(run_id: str) -> tuple[list[str], list[str], float | None]:
        events = store.list_events(run_id, limit=400)
        tools: list[str] = []
        files: list[str] = []
        seen_tools: set[str] = set()
        seen_files: set[str] = set()
        duration_seconds: float | None = None

        def _register_file_name(value: str) -> None:
            candidate = str(value or "").strip()
            if not candidate:
                return
            name = Path(candidate).name or candidate
            lowered = name.lower()
            if lowered in seen_files:
                return
            seen_files.add(lowered)
            files.append(name)

        for event in events:
            event_type = str(event.get("event_type") or "").strip()
            payload = event.get("payload")
            payload_dict = payload if isinstance(payload, dict) else {}
            if event_type == "chat_completed":
                raw_duration = payload_dict.get("duration_seconds")
                with suppress(TypeError, ValueError):
                    if raw_duration is not None:
                        duration_seconds = float(raw_duration)
                continue

            if event_type == "uploads_snapshot":
                uploads = payload_dict.get("uploads")
                if isinstance(uploads, list):
                    for item in uploads[:100]:
                        if not isinstance(item, dict):
                            continue
                        _register_file_name(str(item.get("source_path") or ""))
                        _register_file_name(str(item.get("path") or ""))
                continue

            if event_type != "chat_done_payload":
                continue
            response_payload = payload_dict.get("response")
            response_dict = response_payload if isinstance(response_payload, dict) else {}
            progress_events = response_dict.get("progress_events")
            if not isinstance(progress_events, list):
                continue
            for progress in progress_events:
                if not isinstance(progress, dict):
                    continue
                if str(progress.get("event") or "").strip().lower() == "started":
                    tool_name = str(progress.get("tool") or "").strip()
                    if tool_name and tool_name not in seen_tools:
                        seen_tools.add(tool_name)
                        tools.append(tool_name)
                artifacts = progress.get("artifacts")
                if isinstance(artifacts, list):
                    for artifact in artifacts[:50]:
                        if isinstance(artifact, dict):
                            _register_file_name(str(artifact.get("path") or ""))
                        else:
                            _register_file_name(str(artifact or ""))
        return tools, files, duration_seconds

    def _normalize_reuse_tool_name(value: str) -> str | None:
        normalized = _normalize_search_text(value)
        if not normalized:
            return None
        if any(token in normalized for token in ("segment_image_megaseg", "megaseg", "dynunet")):
            return "segment_image_megaseg"
        if any(token in normalized for token in ("segment_image_sam3", "sam3")):
            return "segment_image_sam3"
        if any(
            token in normalized
            for token in (
                "segment_image_sam2",
                "medsam2",
                "medsam",
                "segmentation",
                "segment",
                "mask",
                "sam2",
                "sam",
            )
        ):
            return "segment_image_sam2"
        if any(
            token in normalized for token in ("yolo_detect", "yolo", "detection", "detect", "bbox")
        ):
            return "yolo_detect"
        if any(
            token in normalized
            for token in ("estimate_depth_pro", "depthpro", "depth", "depth map")
        ):
            return "estimate_depth_pro"
        return None

    def _extract_reuse_tool_names(
        *, prompt: str | None = None, tool_names: list[str] | None = None
    ) -> list[str]:
        selected: list[str] = []
        seen: set[str] = set()
        for raw in tool_names or []:
            normalized = _normalize_reuse_tool_name(raw)
            if normalized and normalized not in seen:
                seen.add(normalized)
                selected.append(normalized)
        if selected:
            return selected

        normalized_prompt = _normalize_search_text(prompt or "")
        if not normalized_prompt:
            return []

        if re.search(r"\b(segment_image_megaseg|megaseg|dynunet)\b", normalized_prompt):
            selected.append("segment_image_megaseg")
        elif re.search(r"\b(segment_image_sam3|sam3)\b", normalized_prompt):
            selected.append("segment_image_sam3")
        elif re.search(
            r"\b(segment_image_sam2|medsam2|medsam|segment|segmentation|mask|sam2|sam)\b",
            normalized_prompt,
        ):
            selected.append("segment_image_sam2")

        inferred_candidates = [
            ("yolo_detect", r"\b(yolo|detect|detection|bbox|bounding)\b"),
            ("estimate_depth_pro", r"\b(depth|depthpro|depth_map|depth map)\b"),
        ]
        for tool_name, pattern in inferred_candidates:
            if re.search(pattern, normalized_prompt):
                selected.append(tool_name)
        return selected

    def _index_run_resource_computations(
        *,
        run_id: str,
        user_id: str | None,
        conversation_id: str | None,
        run_goal: str | None,
        run_status: str,
        run_created_at: str | None,
        run_updated_at: str | None,
        progress_events: list[dict[str, Any]],
        tool_invocations: list[dict[str, Any]] | None = None,
        input_files: list[dict[str, Any]],
    ) -> None:
        normalized_user = str(user_id or "").strip()
        normalized_run = str(run_id or "").strip()
        if not normalized_user or not normalized_run:
            return
        if not input_files:
            return

        tool_names: list[str] = []
        seen_tools: set[str] = set()
        source_labels: set[str] = set()

        def _record_tool_name(raw_tool_name: str, *, source: str) -> None:
            normalized_tool = _normalize_reuse_tool_name(raw_tool_name)
            if not normalized_tool or normalized_tool in seen_tools:
                return
            seen_tools.add(normalized_tool)
            tool_names.append(normalized_tool)
            source_labels.add(source)

        for event in progress_events:
            if not isinstance(event, dict):
                continue
            event_name = str(event.get("event") or "").strip().lower()
            if event_name not in {"started", "completed"}:
                continue
            _record_tool_name(str(event.get("tool") or ""), source="chat_progress_events")

        if isinstance(tool_invocations, list):
            for invocation in tool_invocations:
                if not isinstance(invocation, dict):
                    continue
                status = str(invocation.get("status") or "").strip().lower()
                if status in {"error", "failed", "timeout", "cancelled", "canceled"}:
                    continue
                raw_tool_name = (
                    str(invocation.get("tool") or "").strip()
                    or str(invocation.get("name") or "").strip()
                )
                if not raw_tool_name:
                    continue
                _record_tool_name(raw_tool_name, source="chat_tool_invocations")

        if not tool_names:
            return

        source_label = (
            "chat_progress_and_tool_invocations"
            if len(source_labels) > 1
            else next(iter(source_labels), "chat_progress_events")
        )

        now = datetime.utcnow().isoformat()
        indexed_rows: list[dict[str, Any]] = []
        seen_pairs: set[tuple[str, str]] = set()
        for file_row in input_files:
            file_sha256 = str(file_row.get("file_sha256") or "").strip().lower()
            file_name = str(file_row.get("file_name") or "").strip()
            if not file_sha256 or not file_name:
                continue
            for tool_name in tool_names:
                dedupe_key = (tool_name, file_sha256)
                if dedupe_key in seen_pairs:
                    continue
                seen_pairs.add(dedupe_key)
                indexed_rows.append(
                    {
                        "run_id": normalized_run,
                        "user_id": normalized_user,
                        "conversation_id": str(conversation_id or "").strip() or None,
                        "tool_name": tool_name,
                        "file_sha256": file_sha256,
                        "file_id": str(file_row.get("file_id") or "").strip() or None,
                        "file_name": file_name,
                        "source_path": str(file_row.get("source_path") or "").strip() or None,
                        "run_goal": str(run_goal or "").strip() or None,
                        "run_status": str(run_status or "").strip() or "unknown",
                        "run_created_at": str(run_created_at or "").strip() or None,
                        "run_updated_at": str(run_updated_at or "").strip() or None,
                        "metadata": {
                            "indexed_at": now,
                            "source": source_label,
                        },
                        "created_at": now,
                        "updated_at": now,
                    }
                )
        if not indexed_rows:
            return

        store.upsert_resource_computations(indexed_rows)
        store.append_event(
            normalized_run,
            "resource_computations_indexed",
            {
                "entries": len(indexed_rows),
                "tools": tool_names,
                "files": len({row["file_sha256"] for row in indexed_rows}),
            },
        )

    def _progress_events_from_tool_invocations(
        tool_invocations: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        if not isinstance(tool_invocations, list):
            return []

        progress_events: list[dict[str, Any]] = []
        for invocation in tool_invocations[:256]:
            if not isinstance(invocation, dict):
                continue
            tool_name = str(invocation.get("tool") or "").strip()
            if not tool_name:
                continue
            status = str(invocation.get("status") or "").strip().lower()
            if status not in {"success", "completed", "error", "failed"}:
                continue
            envelope = invocation.get("output_envelope")
            envelope_dict = envelope if isinstance(envelope, dict) else {}
            summary = invocation.get("output_summary")
            summary_dict = dict(summary) if isinstance(summary, dict) else {}
            result_group_id = str(
                summary_dict.get("result_group_id")
                or envelope_dict.get("result_group_id")
                or ""
            ).strip()
            if result_group_id and "result_group_id" not in summary_dict:
                summary_dict["result_group_id"] = result_group_id
            if tool_name == "yolo_detect" and "classes" not in summary_dict:
                top_classes = (
                    summary_dict.get("top_classes")
                    if isinstance(summary_dict.get("top_classes"), list)
                    else []
                )
                if top_classes:
                    summary_dict["classes"] = [
                        {
                            "class_name": str(item.get("class") or "").strip(),
                            "count": int(item.get("count") or 0),
                        }
                        for item in top_classes
                        if isinstance(item, dict) and str(item.get("class") or "").strip()
                    ]
            artifacts: list[dict[str, Any]] = []
            for artifact_group in ("ui_artifacts", "download_artifacts"):
                raw_group = envelope_dict.get(artifact_group)
                if not isinstance(raw_group, list):
                    continue
                for item in raw_group[:64]:
                    if not isinstance(item, dict):
                        continue
                    path_value = str(item.get("path") or "").strip()
                    if not path_value:
                        continue
                    artifacts.append(
                        {
                            "path": path_value,
                            "title": str(item.get("title") or "").strip() or Path(path_value).name,
                            "kind": ("image" if artifact_group == "ui_artifacts" else "artifact"),
                            "source_path": str(item.get("file") or "").strip() or None,
                            "result_group_id": (
                                str(item.get("result_group_id") or "").strip()
                                or result_group_id
                                or None
                            ),
                        }
                    )
            progress_events.append(
                {
                    "event": ("completed" if status in {"success", "completed"} else "error"),
                    "tool": tool_name,
                    "message": (
                        str(envelope_dict.get("summary") or "").strip()
                        or str(invocation.get("output_preview") or "").strip()
                        or None
                    ),
                    "summary": summary_dict,
                    "artifacts": artifacts,
                    "ts": datetime.utcnow().isoformat() + "Z",
                }
            )
        return progress_events

    def _build_analysis_summaries(
        *,
        user_id: str,
        limit: int,
        query_text: str | None = None,
        query_terms: list[str] | None = None,
    ) -> list[AnalysisRunSummary]:
        rows = store.list_runs_for_user(user_id=user_id, limit=max(20, limit * 8))
        run_ids = [
            str(row.get("run_id") or "").strip()
            for row in rows
            if str(row.get("run_id") or "").strip()
        ]
        resource_summary_by_run = store.summarize_resource_computations_for_runs(
            user_id=user_id,
            run_ids=run_ids,
            max_files_per_run=12,
            max_tools_per_run=8,
        )
        lowered_terms = [
            _normalize_search_text(term)
            for term in (query_terms or [])
            if _normalize_search_text(str(term))
        ]
        normalized_query_text = _normalize_search_text(str(query_text or ""))
        scored_summaries: list[tuple[float, AnalysisRunSummary]] = []
        for row in rows:
            run_id = str(row.get("run_id") or "").strip()
            if not run_id:
                continue
            run_summary = resource_summary_by_run.get(run_id) or {}
            tools = [
                str(item).strip() for item in (run_summary.get("tools") or []) if str(item).strip()
            ]
            file_names = [
                str(item).strip()
                for item in (run_summary.get("file_names") or [])
                if str(item).strip()
            ]
            duration_seconds: float | None = None
            if not tools and not file_names:
                tools, file_names, duration_seconds = _extract_run_tools_and_files(run_id)
            goal = str(row.get("goal") or "").strip()
            conversation = str(row.get("conversation_id") or "").strip() or None
            searchable = " ".join(
                [
                    run_id.lower(),
                    goal.lower(),
                    (conversation or "").lower(),
                    " ".join(tool.lower() for tool in tools),
                    " ".join(name.lower() for name in file_names),
                ]
            )
            if normalized_query_text or lowered_terms:
                match_score = _score_query_match(
                    searchable,
                    query_text=normalized_query_text,
                    terms=lowered_terms,
                )
                if match_score <= 0:
                    continue
            else:
                match_score = 0.0

            created_at_raw = str(row.get("created_at") or "").replace("Z", "+00:00")
            updated_at_raw = str(row.get("updated_at") or "").replace("Z", "+00:00")
            try:
                created_at = datetime.fromisoformat(created_at_raw)
            except Exception:
                created_at = datetime.utcnow()
            try:
                updated_at = datetime.fromisoformat(updated_at_raw)
            except Exception:
                updated_at = created_at
            if duration_seconds is None:
                duration_seconds = max(0.0, (updated_at - created_at).total_seconds())

            summary = AnalysisRunSummary(
                run_id=run_id,
                conversation_id=conversation,
                goal=goal or "chat request",
                status=str(row.get("status") or RunStatus.PENDING.value),  # type: ignore[arg-type]
                created_at=created_at,
                updated_at=updated_at,
                error=str(row.get("error") or "").strip() or None,
                tools=tools[:8],
                file_names=file_names[:8],
                duration_seconds=duration_seconds,
            )
            recency_bonus = updated_at.timestamp() / 1_000_000_000
            scored_summaries.append((match_score + recency_bonus, summary))
        scored_summaries.sort(
            key=lambda item: (
                item[0],
                item[1].updated_at.timestamp(),
            ),
            reverse=True,
        )
        return [summary for _, summary in scored_summaries[:limit]]

    def _build_history_context(
        *,
        user_id: str | None,
        conversation_id: str | None,
        latest_user_text: str,
    ) -> str:
        if not user_id:
            return ""
        if not _should_attach_history_context(latest_user_text):
            return ""

        recent_rows = store.list_conversations(user_id=user_id, limit=25)
        filtered_rows = [
            row
            for row in recent_rows
            if str(row.get("conversation_id") or "").strip() != str(conversation_id or "").strip()
        ]
        if not filtered_rows:
            return "Persisted history context: no prior saved conversations are available for this user."

        summary_lines: list[str] = []
        conversation_meta_by_id: dict[str, dict[str, str]] = {}
        recent_ids: set[str] = set()
        for row in filtered_rows[:5]:
            conv_id = str(row.get("conversation_id") or "").strip()
            if not conv_id:
                continue
            recent_ids.add(conv_id)
            title = _trim_history_text(str(row.get("title") or "Untitled"), 70)
            updated_at_raw = str(row.get("updated_at") or "").strip()
            updated_at = _format_history_timestamp(updated_at_raw)
            state = row.get("state")
            state_dict = state if isinstance(state, dict) else {}
            messages = state_dict.get("messages")
            messages_list = messages if isinstance(messages, list) else []

            typed_messages_list = [msg for msg in messages_list if isinstance(msg, dict)]
            last_user = _extract_last_user_message(typed_messages_list)
            run_ids: list[str] = []
            seen_run_ids: set[str] = set()
            for msg in typed_messages_list:
                run_id = str(msg.get("runId") or "").strip()
                if run_id and run_id not in seen_run_ids:
                    seen_run_ids.add(run_id)
                    run_ids.append(run_id)

            uploaded = state_dict.get("uploadedFiles")
            uploaded_names: list[str] = []
            if isinstance(uploaded, list):
                for item in uploaded:
                    if not isinstance(item, dict):
                        continue
                    name = str(item.get("original_name") or "").strip()
                    if name and name not in uploaded_names:
                        uploaded_names.append(name)
                    if len(uploaded_names) >= 4:
                        break

            line = f'- chat "{title}" (updated {updated_at}) [conversation_id={conv_id}]'
            if run_ids:
                line += f" | internal run refs: {', '.join(run_ids[:3])}"
            if uploaded_names:
                line += f" | files: {', '.join(uploaded_names[:3])}"
            if last_user:
                line += f"\n  last user: {_trim_history_text(last_user, 140)}"
            summary_lines.append(line)
            conversation_meta_by_id[conv_id] = {
                "title": title,
                "updated_at": updated_at,
                "last_user": _trim_history_text(last_user, 140) if last_user else "",
            }

        match_lines: list[str] = []
        ranked_hits = _search_conversation_messages_flexible(
            user_id=user_id,
            query_text=latest_user_text,
            limit=5,
            include_conversation_ids=recent_ids,
            exclude_conversation_id=str(conversation_id or "").strip() or None,
        )
        for hit in ranked_hits[:5]:
            conv_id = str(hit.get("conversation_id") or "").strip()
            role = str(hit.get("role") or "assistant").strip().lower()
            content = _trim_history_text(str(hit.get("content") or ""), 140)
            score = float(hit.get("relevance_score") or 0.0)
            meta = conversation_meta_by_id.get(conv_id) or {}
            title = str(meta.get("title") or f"Untitled {conv_id}").strip()
            updated_at = str(meta.get("updated_at") or "unknown").strip()
            match_lines.append(
                f'- chat "{title}" (updated {updated_at}) [conversation_id={conv_id}] '
                f"| {role} (score {score:.1f}): {content}"
            )

        analysis_terms = _extract_history_search_terms(latest_user_text, max_terms=8)
        recent_analyses = _build_analysis_summaries(
            user_id=user_id,
            limit=5,
            query_text=latest_user_text,
            query_terms=analysis_terms,
        )
        analysis_lines: list[str] = []
        for item in recent_analyses:
            conversation_ref = str(item.conversation_id or "").strip()
            meta = conversation_meta_by_id.get(conversation_ref) if conversation_ref else None
            if meta is None and conversation_ref:
                conversation_row = store.get_conversation(
                    conversation_id=conversation_ref,
                    user_id=user_id,
                )
                if conversation_row:
                    title = _trim_history_text(
                        str(conversation_row.get("title") or "Untitled"),
                        70,
                    )
                    updated_at = _format_history_timestamp(
                        str(conversation_row.get("updated_at") or "")
                    )
                    state = conversation_row.get("state")
                    state_dict = state if isinstance(state, dict) else {}
                    messages = state_dict.get("messages")
                    messages_list = (
                        [msg for msg in messages if isinstance(msg, dict)]
                        if isinstance(messages, list)
                        else []
                    )
                    last_user = _extract_last_user_message(messages_list)
                    meta = {
                        "title": title,
                        "updated_at": updated_at,
                        "last_user": _trim_history_text(last_user, 140) if last_user else "",
                    }
                    conversation_meta_by_id[conversation_ref] = meta

            title = str((meta or {}).get("title") or "Unknown chat").strip()
            updated_at = str((meta or {}).get("updated_at") or "unknown").strip()
            line = (
                f'- chat "{title}" (updated {updated_at})'
                + (f" [conversation_id={conversation_ref}]" if conversation_ref else "")
                + f" | analysis status: {item.status}"
            )
            if item.goal:
                line += f" | goal: {_trim_history_text(item.goal, 64)}"
            if item.tools:
                line += f" | tools: {', '.join(item.tools[:3])}"
            if item.file_names:
                line += f" | files: {', '.join(item.file_names[:3])}"
            if item.duration_seconds is not None:
                line += f" | duration: {item.duration_seconds:.1f}s"
            line += f" | internal run ref: {item.run_id[:8]}"
            if meta and meta.get("last_user"):
                line += f"\n  last user: {meta['last_user']}"
            analysis_lines.append(line)

        context = (
            "Persisted history context (same user, prior chats). "
            "If the user asks about previous analyses, use this context before replying:\n"
            "Recent conversations (latest 5):\n"
            + ("\n".join(summary_lines) if summary_lines else "- none")
        )
        if analysis_lines:
            context += "\n\nRecent analysis runs (latest 5):\n" + "\n".join(analysis_lines)
        if match_lines:
            context += "\n\nPotentially relevant message hits:\n" + "\n".join(match_lines)
        context += (
            "\n\nResponse rules for history lookups:\n"
            "- If the user asks where a prior result/run is, identify the chat by title and updated time first.\n"
            "- Run IDs are internal references; do not tell users to locate chats by run_id alone.\n"
            "- Mention internal run refs only as optional secondary details.\n"
            "- If multiple chats are plausible, list the top 2-3 candidate chats with distinguishing file/tool details.\n"
            "If details are missing, say the answer is based on saved chat history."
        )
        return context

    def _should_attach_artifact_followup_context(latest_user_text: str) -> bool:
        normalized = _normalize_search_text(latest_user_text)
        if not normalized:
            return False

        tokens = set(normalized.split(" "))
        action_terms = {
            "upload",
            "share",
            "send",
            "push",
            "publish",
            "export",
            "transfer",
            "save",
            "store",
        }
        followup_action_terms = {
            "run",
            "rerun",
            "reuse",
            "use",
            "analyze",
            "analyse",
            "perform",
            "compute",
            "quantify",
            "evaluate",
            "compare",
            "process",
            "continue",
            "extend",
        }
        referential_terms = {
            "previous",
            "prior",
            "earlier",
            "same",
            "last",
        }
        reference_terms = {
            "result",
            "results",
            "output",
            "outputs",
            "artifact",
            "artifacts",
            "file",
            "files",
            "run",
            "runs",
            "detection",
            "detections",
            "segment",
            "segmentation",
            "mask",
            "masks",
            "analysis",
            "analyses",
            "yolo",
            "sam3",
            "sam",
            "depth",
            "depthpro",
            "image",
            "images",
            "dataset",
            "datasets",
            "table",
            "tables",
            "metric",
            "metrics",
            "csv",
        }

        if any(token in action_terms for token in tokens) and any(
            token in reference_terms for token in tokens
        ):
            return True
        if (
            any(token in followup_action_terms for token in tokens)
            and any(token in referential_terms for token in tokens)
            and any(token in reference_terms for token in tokens)
        ):
            return True

        keyword_patterns = [
            r"\b(these|those|this|that)\s+(result|results|output|outputs|artifact|artifacts|file|files)\b",
            r"\b(upload|share|send|push|publish|export)\b.{0,40}\b(result|results|output|outputs|artifact|artifacts|run|runs)\b",
            r"\b(to|into)\s+bisque\b",
            r"\b(other|additional|further|next|follow[- ]?up)\s+(analysis|analyses)\b",
            r"\b(run|re[- ]?run|rerun|analy[sz]e|perform|apply|quantif(?:y|ication)|compute|use|reuse)\b.{0,70}\b(previous|prior|earlier|same|last)\b.{0,50}\b(image|images|mask|masks|output|outputs|result|results|artifact|artifacts|file|files|dataset|datasets|table|tables|csv|metrics?)\b",
            r"\b(using|use)\s+(our\s+)?(previous|prior|earlier|same|last)\s+(image|mask|output|result|artifact|file|dataset|table|csv)\b",
        ]
        return any(re.search(pattern, normalized) for pattern in keyword_patterns)

    def _collect_run_artifact_context_rows(
        run_id: str,
        *,
        limit: int = 24,
    ) -> list[dict[str, str]]:
        run_dir = _ensure_run_artifact_dir(run_id).resolve()
        repo_root = Path(__file__).resolve().parents[2]
        by_path: dict[str, dict[str, str]] = {}

        def _resolve_local_path(raw_path: str) -> Path | None:
            candidate_raw = str(raw_path or "").strip()
            if not candidate_raw:
                return None
            candidate = Path(candidate_raw).expanduser()
            search_order = [candidate]
            if not candidate.is_absolute():
                search_order.append(run_dir / candidate)
                search_order.append(repo_root / candidate)
            for option in search_order:
                try:
                    if option.exists() and option.is_file():
                        return option.resolve()
                except Exception:
                    continue
            return None

        def _register(
            path_value: str,
            *,
            title: str,
            category: str,
            tool: str | None = None,
        ) -> bool:
            resolved = _resolve_local_path(path_value)
            if resolved is None:
                return False
            key = str(resolved)
            by_path[key] = {
                "path": key,
                "title": title or resolved.name,
                "category": category or "artifact",
                "tool": str(tool or "").strip(),
            }
            return True

        manifest = _materialize_manifest_if_missing(run_id)
        manifest_artifacts = manifest.get("artifacts")
        if isinstance(manifest_artifacts, list):
            for item in manifest_artifacts[:400]:
                if not isinstance(item, dict):
                    continue
                rel_path = str(item.get("path") or "").strip()
                source_path = str(item.get("source_path") or "").strip()
                title = str(item.get("title") or "").strip() or Path(rel_path or source_path).name
                category = str(item.get("category") or "artifact").strip() or "artifact"
                tool_name = str(item.get("tool") or "").strip() or None
                registered_source = False
                if source_path:
                    registered_source = _register(
                        source_path,
                        title=title,
                        category=category,
                        tool=tool_name,
                    )
                if not registered_source:
                    _register(
                        rel_path,
                        title=title,
                        category=category,
                        tool=tool_name,
                    )

        events = store.list_events(run_id, limit=300)
        for event in events:
            event_type = str(event.get("event_type") or "").strip()
            if event_type != "chat_done_payload":
                continue
            payload = event.get("payload")
            payload_dict = payload if isinstance(payload, dict) else {}
            response = payload_dict.get("response")
            response_dict = response if isinstance(response, dict) else {}
            progress_events = response_dict.get("progress_events")
            if not isinstance(progress_events, list):
                continue
            for progress in progress_events[:300]:
                if not isinstance(progress, dict):
                    continue
                tool_name = str(progress.get("tool") or "").strip() or None
                artifacts = progress.get("artifacts")
                if not isinstance(artifacts, list):
                    continue
                for artifact in artifacts[:200]:
                    if isinstance(artifact, dict):
                        _register(
                            str(artifact.get("path") or ""),
                            title=str(artifact.get("title") or "").strip()
                            or Path(str(artifact.get("path") or "")).name,
                            category=str(artifact.get("kind") or "artifact").strip() or "artifact",
                            tool=tool_name,
                        )
                    else:
                        raw_path = str(artifact or "").strip()
                        _register(
                            raw_path, title=Path(raw_path).name, category="artifact", tool=tool_name
                        )

        def _score_row(row: dict[str, str]) -> tuple[int, float]:
            path = Path(str(row.get("path") or ""))
            name = path.name.lower()
            suffix = path.suffix.lower()
            category = str(row.get("category") or "").lower()
            score = 0
            if category in {"tool_output", "upload"}:
                score += 40
            if category in {"tool_output_preview"} or "__preview" in name:
                score -= 20
            if name.endswith(".nii.gz"):
                score += 36
            elif suffix in {".npy", ".npz", ".nii", ".h5", ".hdf5"}:
                score += 34
            elif suffix in {".json", ".jsonl", ".csv", ".tsv", ".xml"}:
                score += 30
            elif suffix in {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".webp"}:
                score += 24
            if any(token in name for token in ("mask", "overlay", "prediction", "label", "report")):
                score += 4
            try:
                mtime = float(path.stat().st_mtime)
            except Exception:
                mtime = 0.0
            return (score, mtime)

        ranked = sorted(
            by_path.values(),
            key=lambda row: _score_row(row),
            reverse=True,
        )
        return ranked[: max(1, int(limit))]

    def _build_followup_artifact_context(
        *,
        user_id: str | None,
        conversation_id: str | None,
        latest_user_text: str,
    ) -> tuple[str, list[str]]:
        if not user_id:
            return "", []
        if not _should_attach_artifact_followup_context(latest_user_text):
            return "", []

        rows = store.list_runs_for_user(user_id=user_id, limit=200)
        if not rows:
            return "", []

        current_conversation = str(conversation_id or "").strip()
        if current_conversation:
            ordered_rows = [
                row
                for row in rows
                if str(row.get("conversation_id") or "").strip() == current_conversation
            ]
        else:
            ordered_rows = list(rows)

        lines: list[str] = []
        reusable_paths: list[str] = []
        run_count = 0
        artifact_count = 0
        for row in ordered_rows:
            run_id = str(row.get("run_id") or "").strip()
            if not run_id:
                continue
            status = str(row.get("status") or "").strip().lower()
            if status and status not in {
                RunStatus.SUCCEEDED.value,
                RunStatus.RUNNING.value,
                RunStatus.PENDING.value,
            }:
                continue

            artifact_rows = _collect_run_artifact_context_rows(run_id, limit=8)
            if not artifact_rows:
                continue

            goal = _trim_history_text(str(row.get("goal") or "chat request"), 72)
            lines.append(f"- run {run_id} ({status or 'unknown'}) | goal: {goal}")
            run_count += 1
            for artifact in artifact_rows:
                path = str(artifact.get("path") or "").strip()
                if not path:
                    continue
                reusable_paths.append(path)
                title = _trim_history_text(
                    str(artifact.get("title") or Path(path).name),
                    72,
                )
                category = str(artifact.get("category") or "artifact").strip()
                tool = str(artifact.get("tool") or "").strip()
                line = f"  - {path} | {title} | {category}"
                if tool:
                    line += f" | tool={tool}"
                lines.append(line)
                artifact_count += 1
                if artifact_count >= 20:
                    break

            if run_count >= 4 or artifact_count >= 20:
                break

        if not lines:
            return "", []

        return (
            "Follow-up artifact context for this user. The user is likely referring to prior analysis outputs. "
            "For follow-up tool calls that can reuse local file paths (for example bioio_load_image, "
            "segment_image_sam2, quantify_segmentation_masks, yolo_detect, execute_python_job inputs, upload_to_bisque), "
            "use these exact paths before asking for regeneration. For quantify_segmentation_masks, prefer paths that are mask-only files "
            "or preferred_upload paths from segmentation tools:\n" + "\n".join(lines),
            reusable_paths,
        )

    def _search_conversation_messages_flexible(
        *,
        user_id: str,
        query_text: str,
        limit: int = 50,
        include_conversation_ids: set[str] | None = None,
        exclude_conversation_id: str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_query = _normalize_search_text(query_text)
        if not normalized_query:
            return []
        terms = _extract_history_search_terms(query_text, max_terms=10)
        term_candidates = [
            candidate
            for candidate in terms
            if candidate and candidate not in normalized_query and len(candidate) >= 3
        ]
        candidate_queries = [normalized_query]
        candidate_queries.extend(term_candidates[:2])
        seen: dict[tuple[str, str], dict[str, Any]] = {}
        include_ids_list = (
            sorted(include_conversation_ids) if include_conversation_ids is not None else None
        )
        for idx, candidate in enumerate(candidate_queries):
            if idx >= 1 and len(seen) >= max(limit, 20):
                break
            query_limit = max(limit * 2, 36) if idx == 0 else max(limit, 20)
            hits = store.search_conversation_messages(
                user_id=user_id,
                query=candidate,
                limit=min(250, query_limit),
                include_conversation_ids=include_ids_list,
                exclude_conversation_id=exclude_conversation_id,
            )
            for hit in hits:
                conv_id = str(hit.get("conversation_id") or "").strip()
                if not conv_id:
                    continue
                if exclude_conversation_id and conv_id == exclude_conversation_id:
                    continue
                if include_conversation_ids is not None and conv_id not in include_conversation_ids:
                    continue
                message_id = str(hit.get("message_id") or "").strip()
                key = (conv_id, message_id)
                content = str(hit.get("content") or "")
                score = _score_query_match(
                    f"{conv_id} {content}",
                    query_text=query_text,
                    terms=terms,
                )
                backend_score = float(hit.get("search_score") or 0.0)
                if backend_score:
                    score += max(0.0, backend_score) * 0.35
                existing = seen.get(key)
                if existing is None or float(existing.get("relevance_score") or 0.0) < score:
                    seen[key] = {
                        **hit,
                        "relevance_score": round(score, 3),
                    }
        ranked = sorted(
            seen.values(),
            key=lambda item: (
                float(item.get("relevance_score") or 0.0),
                int(item.get("created_at_ms") or 0),
            ),
            reverse=True,
        )
        return ranked[:limit]

    def _materialize_manifest_if_missing(run_id: str) -> dict[str, Any]:
        manifest_path = _manifest_path(run_id)
        if manifest_path.exists():
            manifest = _load_manifest(run_id)
            manifest = _backfill_progress_output_artifacts(run_id, manifest)
            return _backfill_yolo_annotated_artifacts(run_id, manifest)

        run_dir = _ensure_run_artifact_dir(run_id)
        entries: list[dict[str, Any]] = []
        for file_path in sorted(run_dir.rglob("*")):
            if file_path.is_file() and file_path.name != "artifact_manifest.json":
                entries.append(_artifact_entry(run_id, file_path))
        manifest = _update_manifest_with_entries(run_id, entries)
        manifest = _backfill_progress_output_artifacts(run_id, manifest)
        return _backfill_yolo_annotated_artifacts(run_id, manifest)

    def _request_hash(payload: Any) -> str:
        wire = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(wire.encode("utf-8")).hexdigest()

    def _bisque_nav_links() -> dict[str, str]:
        root = str(getattr(settings, "bisque_root", "") or "").strip().rstrip("/")
        home = f"{root}/client_service/"
        images = f"{root}/client_service/browser?resource=/data_service/image"
        datasets = f"{root}/client_service/browser?resource=/data_service/dataset"
        tables = f"{root}/client_service/browser?resource=/data_service/table"
        return {
            # These links are opened from an already-authenticated Ultra session.
            # Sending users through BisQue's explicit oidc_login bootstrap again can
            # create slow or error-prone extra redirects on the shared-host setup.
            "home": home,
            "images": images,
            "datasets": datasets,
            "tables": tables,
        }

    def _bisque_browser_url() -> str:
        return _bisque_nav_links()["images"]

    santa_barbara_weather_profile = {
        "location": "Santa Barbara, CA",
        "micro_location": "Campus Point",
        "forecast_latitude": "34.4139",
        "forecast_longitude": "-119.8489",
        "marine_latitude": "34.4080",
        "marine_longitude": "-119.8450",
    }

    def _weather_code_label(code: int | None) -> str:
        mapping = {
            0: "clear sky",
            1: "mainly clear",
            2: "partly cloudy",
            3: "overcast",
            45: "fog",
            48: "depositing rime fog",
            51: "light drizzle",
            53: "moderate drizzle",
            55: "dense drizzle",
            56: "light freezing drizzle",
            57: "dense freezing drizzle",
            61: "slight rain",
            63: "moderate rain",
            65: "heavy rain",
            66: "light freezing rain",
            67: "heavy freezing rain",
            71: "slight snowfall",
            73: "moderate snowfall",
            75: "heavy snowfall",
            77: "snow grains",
            80: "slight rain showers",
            81: "moderate rain showers",
            82: "violent rain showers",
            85: "slight snow showers",
            86: "heavy snow showers",
            95: "thunderstorm",
            96: "thunderstorm with slight hail",
            99: "thunderstorm with heavy hail",
        }
        if code is None:
            return "unknown conditions"
        return mapping.get(int(code), "variable conditions")

    def _to_float(value: Any) -> float | None:
        try:
            numeric = float(value)
        except Exception:
            return None
        if numeric != numeric:  # NaN guard
            return None
        return numeric

    def _meters_to_feet(value: float | None) -> float | None:
        if value is None:
            return None
        return value * 3.28084

    def _fetch_weather_json(url: str) -> dict[str, Any]:
        response = httpx.get(
            url,
            timeout=8.0,
            headers={"User-Agent": "bisque-ultra/0.1"},
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Weather payload was not a JSON object.")
        return payload

    def _build_santa_barbara_weather_blip(
        *,
        micro_location: str,
        observed_at: str | None,
        weather_label: str | None,
        wind_speed_mph: float | None,
        daily_high_f: float | None,
        precipitation_probability_percent: float | None,
        wave_height_ft: float | None,
    ) -> str:
        safe_label = str(weather_label or "").strip().lower()
        hour = None
        if observed_at:
            with suppress(ValueError):
                hour = datetime.fromisoformat(observed_at).hour

        is_foggy = "fog" in safe_label
        is_wet = bool(re.search(r"rain|drizzle|thunderstorm|snow", safe_label))
        dry_enough = (
            precipitation_probability_percent is None or precipitation_probability_percent <= 20
        )
        breezy = wind_speed_mph is not None and wind_speed_mph >= 15
        mellow_wind = wind_speed_mph is None or wind_speed_mph <= 12
        mild_day = daily_high_f is None or 60 <= daily_high_f <= 74
        walkable = dry_enough and not breezy and not is_wet and not is_foggy and mild_day
        surf_peek = (
            wave_height_ft is not None
            and 2.0 <= wave_height_ft <= 5.5
            and dry_enough
            and mellow_wind
            and not is_wet
        )

        if is_wet or (
            precipitation_probability_percent is not None
            and precipitation_probability_percent >= 55
        ):
            return (
                f"Looks more like a jacket-and-coffee day around {micro_location} "
                "than a beach detour, but getting to class should still be manageable."
            )
        if is_foggy and hour is not None and hour < 11:
            if surf_peek:
                return (
                    f"Marine layer start around {micro_location} this morning, but if it burns off "
                    "later Campus Point could still be worth a quick surf check."
                )
            return (
                f"Foggy start around {micro_location} today, so it feels more like a class-first "
                "morning with a possible Campus Point walk if the coast clears up later."
            )
        if surf_peek:
            if wave_height_ft >= 3.5:
                return (
                    f"Dry weather, manageable wind, and roughly {wave_height_ft:.1f} ft surf make "
                    "Campus Point worth a look after class."
                )
            return (
                "Small-but-possibly-fun surf setup near Campus Point today, with light wind and "
                "enough swell to justify a quick post-class check."
            )
        if walkable:
            return (
                "Classic coastal weather today: easy walk-to-class conditions and a nice window for "
                "a stroll out to Campus Point."
            )
        if breezy and dry_enough:
            return (
                "Dry but breezier along the coast today, so it feels better for class and a quick "
                "campus loop than a long beach hang."
            )
        return (
            "Steady Santa Barbara coastal weather today — probably best for class, a short walk, "
            "and checking Campus Point if you're already headed that way."
        )

    def _fetch_santa_barbara_weather() -> SantaBarbaraWeatherResponse:
        query = urlencode(
            {
                "latitude": santa_barbara_weather_profile["forecast_latitude"],
                "longitude": santa_barbara_weather_profile["forecast_longitude"],
                "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "forecast_days": "1",
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "timezone": "America/Los_Angeles",
            }
        )
        url = f"https://api.open-meteo.com/v1/forecast?{query}"
        payload = _fetch_weather_json(url)

        current = payload.get("current")
        daily = payload.get("daily")
        if not isinstance(current, dict) or not isinstance(daily, dict):
            raise ValueError("Weather payload missing current/daily sections.")

        temperature_f = _to_float(current.get("temperature_2m"))
        apparent_temperature_f = _to_float(current.get("apparent_temperature"))
        wind_speed_mph = _to_float(current.get("wind_speed_10m"))
        weather_code = (
            int(current.get("weather_code")) if current.get("weather_code") is not None else None
        )
        weather_label = _weather_code_label(weather_code)
        observed_at = str(current.get("time") or "").strip() or None

        max_values = daily.get("temperature_2m_max")
        min_values = daily.get("temperature_2m_min")
        precip_values = daily.get("precipitation_probability_max")

        daily_high_f = (
            _to_float(max_values[0]) if isinstance(max_values, list) and max_values else None
        )
        daily_low_f = (
            _to_float(min_values[0]) if isinstance(min_values, list) and min_values else None
        )
        precipitation_probability_percent = (
            _to_float(precip_values[0])
            if isinstance(precip_values, list) and precip_values
            else None
        )

        wave_height_ft = None
        swell_wave_height_ft = None
        wave_period_seconds = None
        marine_query = urlencode(
            {
                "latitude": santa_barbara_weather_profile["marine_latitude"],
                "longitude": santa_barbara_weather_profile["marine_longitude"],
                "daily": "wave_height_max,swell_wave_height_max,wave_period_max",
                "forecast_days": "1",
                "timezone": "America/Los_Angeles",
            }
        )
        marine_url = f"https://marine-api.open-meteo.com/v1/marine?{marine_query}"
        try:
            marine_payload = _fetch_weather_json(marine_url)
            marine_daily = marine_payload.get("daily")
            if isinstance(marine_daily, dict):
                wave_values = marine_daily.get("wave_height_max")
                swell_values = marine_daily.get("swell_wave_height_max")
                period_values = marine_daily.get("wave_period_max")
                wave_height_ft = _meters_to_feet(
                    _to_float(wave_values[0])
                    if isinstance(wave_values, list) and wave_values
                    else None
                )
                swell_wave_height_ft = _meters_to_feet(
                    _to_float(swell_values[0])
                    if isinstance(swell_values, list) and swell_values
                    else None
                )
                wave_period_seconds = (
                    _to_float(period_values[0])
                    if isinstance(period_values, list) and period_values
                    else None
                )
        except Exception:
            # Marine data is a quality-of-life enhancement; weather should still render without it.
            pass

        summary_bits: list[str] = []
        if temperature_f is not None:
            summary_bits.append(f"{round(temperature_f):.0f}°F")
        summary_bits.append(weather_label)
        if daily_high_f is not None and daily_low_f is not None:
            summary_bits.append(f"high/low {round(daily_high_f):.0f}°/{round(daily_low_f):.0f}°")
        if precipitation_probability_percent is not None:
            summary_bits.append(f"rain chance {round(precipitation_probability_percent):.0f}%")
        if wave_height_ft is not None:
            summary_bits.append(f"Campus Point waves ~{wave_height_ft:.1f} ft")
        summary = "Santa Barbara right now: " + ", ".join(summary_bits) + "."
        blip = _build_santa_barbara_weather_blip(
            micro_location=str(santa_barbara_weather_profile["micro_location"]),
            observed_at=observed_at,
            weather_label=weather_label,
            wind_speed_mph=wind_speed_mph,
            daily_high_f=daily_high_f,
            precipitation_probability_percent=precipitation_probability_percent,
            wave_height_ft=wave_height_ft,
        )

        return SantaBarbaraWeatherResponse(
            success=True,
            location=str(santa_barbara_weather_profile["location"]),
            micro_location=str(santa_barbara_weather_profile["micro_location"]),
            observed_at=observed_at,
            temperature_f=temperature_f,
            apparent_temperature_f=apparent_temperature_f,
            weather_code=weather_code,
            weather_label=weather_label,
            wind_speed_mph=wind_speed_mph,
            daily_high_f=daily_high_f,
            daily_low_f=daily_low_f,
            precipitation_probability_percent=precipitation_probability_percent,
            wave_height_ft=wave_height_ft,
            swell_wave_height_ft=swell_wave_height_ft,
            wave_period_seconds=wave_period_seconds,
            blip=blip,
            summary=summary,
            source="open-meteo forecast + marine",
        )

    def _require_api_key_only(
        request: FastAPIRequest,
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
        authorization: str | None = Header(default=None, alias="Authorization"),
        api_key_query: str | None = Query(default=None, alias="api_key"),
    ) -> None:
        _validate_presented_api_key(
            request,
            x_api_key=x_api_key,
            authorization=authorization,
            api_key_query=api_key_query,
        )

    def _require_api_key(
        request: FastAPIRequest,
        bisque_auth: dict[str, Any] | None = Depends(_get_existing_bisque_auth_optional),
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
        authorization: str | None = Header(default=None, alias="Authorization"),
        api_key_query: str | None = Query(default=None, alias="api_key"),
    ) -> None:
        if _has_authenticated_browser_session(bisque_auth):
            _record_request_auth_context(
                request,
                auth_source="browser_session",
                user_id=_resolve_authenticated_session_user_id(bisque_auth),
            )
            return
        _validate_presented_api_key(
            request,
            x_api_key=x_api_key,
            authorization=authorization,
            api_key_query=api_key_query,
        )

    def _require_admin_session(
        request: FastAPIRequest,
        bisque_auth: dict[str, Any] = Depends(_require_authenticated_session),
    ) -> dict[str, Any]:
        if not bisque_auth:
            raise HTTPException(status_code=401, detail="Authentication required.")
        mode = str(bisque_auth.get("mode") or "guest").strip().lower()
        username = str(bisque_auth.get("username") or "").strip()
        if mode != "bisque" or not username:
            raise HTTPException(
                status_code=403,
                detail="Admin access requires a signed-in BisQue account.",
            )
        if not _is_admin_username(username):
            raise HTTPException(status_code=403, detail="Admin access denied.")
        admin_context = {
            "username": username,
            "user_id": _current_user_id(bisque_auth, allow_anonymous=False),
        }
        _record_request_auth_context(
            request,
            auth_source="admin_session",
            user_id=admin_context["user_id"],
        )
        return admin_context

    @app.middleware("http")
    async def _request_observability_middleware(
        request: FastAPIRequest,
        call_next: Callable[[FastAPIRequest], Any],
    ) -> FastAPIResponse:
        request_id = str(request.headers.get(request_id_header_name) or "").strip() or uuid4().hex
        request.state.request_id = request_id
        started_at = time.perf_counter()
        route_path = str(request.url.path)
        try:
            response = await call_next(request)
        except Exception:
            elapsed_seconds = max(time.perf_counter() - started_at, 0.0)
            http_request_counter.labels(
                method=request.method,
                route=route_path,
                status_code="500",
            ).inc()
            http_request_latency_seconds.labels(
                method=request.method,
                route=route_path,
            ).observe(elapsed_seconds)
            logger.exception(
                json.dumps(
                    {
                        "event": "http_request_failed",
                        "request_id": request_id,
                        "method": request.method,
                        "route": route_path,
                        "status_code": 500,
                        "duration_ms": round(elapsed_seconds * 1000, 2),
                        "auth_source": getattr(request.state, "auth_source", None),
                        "user_id": getattr(request.state, "auth_user_id", None),
                    }
                )
            )
            raise

        route = request.scope.get("route")
        route_path = str(getattr(route, "path", route_path))
        elapsed_seconds = max(time.perf_counter() - started_at, 0.0)
        response.headers[request_id_header_name] = request_id
        if bool(getattr(request.state, "query_api_key_compat_used", False)):
            response.headers.setdefault("Warning", query_api_key_deprecation_warning)
        http_request_counter.labels(
            method=request.method,
            route=route_path,
            status_code=str(response.status_code),
        ).inc()
        http_request_latency_seconds.labels(
            method=request.method,
            route=route_path,
        ).observe(elapsed_seconds)
        logger.info(
            json.dumps(
                {
                    "event": "http_request_completed",
                    "request_id": request_id,
                    "method": request.method,
                    "route": route_path,
                    "status_code": int(response.status_code),
                    "duration_ms": round(elapsed_seconds * 1000, 2),
                    "auth_source": getattr(request.state, "auth_source", None),
                    "user_id": getattr(request.state, "auth_user_id", None),
                }
            )
        )
        return response

    @v1.get("/health")
    @legacy.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "ts": datetime.utcnow().isoformat()}

    @v1.get("/metrics", include_in_schema=False)
    @legacy.get("/metrics", include_in_schema=False)
    def metrics(
        _auth: None = Depends(_require_api_key_only),
    ) -> FastAPIResponse:
        del _auth
        return FastAPIResponse(
            content=generate_latest(metrics_registry),
            media_type=CONTENT_TYPE_LATEST,
        )

    @v1.get("/config/public")
    @legacy.get("/config/public")
    def public_config() -> dict[str, Any]:
        root = str(getattr(settings, "bisque_root", "") or "").strip().rstrip("/")
        bisque_urls = _bisque_nav_links()
        auth_mode = _bisque_auth_mode()
        return {
            "bisque_root": root,
            "bisque_browser_url": bisque_urls["images"],
            "bisque_urls": bisque_urls,
            "bisque_auth_enabled": True,
            "bisque_auth_mode": auth_mode,
            "bisque_oidc_enabled": _is_browser_oidc_enabled(),
            "bisque_guest_enabled": auth_mode != "oidc",
            "admin_enabled": bool(configured_admin_usernames),
        }

    @v1.get("/fun/weather/santa-barbara", response_model=SantaBarbaraWeatherResponse)
    @legacy.get("/fun/weather/santa-barbara", response_model=SantaBarbaraWeatherResponse)
    def santa_barbara_weather(
        _auth: None = Depends(_require_api_key),
    ) -> SantaBarbaraWeatherResponse:
        del _auth
        try:
            return _fetch_santa_barbara_weather()
        except Exception:
            return SantaBarbaraWeatherResponse(
                success=False,
                location=str(santa_barbara_weather_profile["location"]),
                micro_location=str(santa_barbara_weather_profile["micro_location"]),
                blip="Santa Barbara weather is temporarily unavailable. We can still dive into your analysis.",
                summary="Santa Barbara weather is temporarily unavailable. Want to continue with your analysis?",
                source="open-meteo",
            )

    @v1.get("/auth/oidc/start")
    @legacy.get("/auth/oidc/start")
    def auth_oidc_start(
        request: FastAPIRequest,
        next: str | None = Query(default=None),
    ) -> RedirectResponse:
        if _oidc_via_bisque_login_enabled():
            root = str(getattr(settings, "bisque_root", "") or "").strip().rstrip("/")
            if not root:
                raise HTTPException(status_code=503, detail="BisQue root is not configured.")
            redirect_target = _resolve_frontend_redirect_target(request, next)
            bisque_oidc_login_url = _append_query_params(
                f"{root}/auth_service/oidc_login",
                {"came_from": redirect_target},
            )
            return RedirectResponse(bisque_oidc_login_url, status_code=302)

        if not _is_oidc_enabled():
            raise HTTPException(status_code=503, detail="OIDC login is not configured.")

        authorize_url = _oidc_endpoint("authorize")
        client_id = str(getattr(settings, "bisque_auth_oidc_client_id", "") or "").strip()
        redirect_uri = _oidc_redirect_uri(request)
        scope = str(getattr(settings, "bisque_auth_oidc_scope", "") or "").strip() or "openid"
        if not authorize_url or not client_id or not redirect_uri:
            raise HTTPException(status_code=503, detail="OIDC endpoints are not configured.")

        state = uuid4().hex
        nonce = uuid4().hex
        authorization_url = _append_query_params(
            authorize_url,
            {
                "client_id": client_id,
                "response_type": "code",
                "scope": scope,
                "redirect_uri": redirect_uri,
                "state": state,
                "nonce": nonce,
            },
        )
        response = RedirectResponse(authorization_url, status_code=302)
        response.set_cookie(
            key=oidc_state_cookie_name,
            value=state,
            max_age=oidc_state_ttl_seconds,
            httponly=True,
            secure=_should_secure_cookie(request),
            samesite="lax",
            path="/",
        )
        next_target = str(next or "").strip()
        if _is_safe_frontend_redirect_target(request, next_target):
            response.set_cookie(
                key=oidc_next_cookie_name,
                value=next_target,
                max_age=oidc_state_ttl_seconds,
                httponly=True,
                secure=_should_secure_cookie(request),
                samesite="lax",
                path="/",
            )
        else:
            response.delete_cookie(
                key=oidc_next_cookie_name,
                path="/",
            )
        return response

    @v1.get("/auth/oidc/callback")
    @legacy.get("/auth/oidc/callback")
    def auth_oidc_callback(
        request: FastAPIRequest,
        code: str | None = Query(default=None),
        state: str | None = Query(default=None),
        error: str | None = Query(default=None),
        error_description: str | None = Query(default=None),
        oidc_state: str | None = Cookie(default=None, alias=oidc_state_cookie_name),
        oidc_next: str | None = Cookie(default=None, alias=oidc_next_cookie_name),
    ) -> RedirectResponse:
        redirect_target = _resolve_frontend_redirect_target(request, oidc_next)

        if not _is_oidc_enabled():
            return _frontend_auth_error_redirect(
                request,
                "OIDC login is not configured.",
                preferred_redirect=redirect_target,
            )

        if error:
            return _frontend_auth_error_redirect(
                request,
                str(error_description or error).strip() or "Authentication was cancelled.",
                preferred_redirect=redirect_target,
            )

        code_value = str(code or "").strip()
        state_value = str(state or "").strip()
        expected_state = str(oidc_state or "").strip()
        if not code_value:
            return _frontend_auth_error_redirect(
                request,
                "Missing authorization code.",
                preferred_redirect=redirect_target,
            )
        if not state_value or not expected_state or state_value != expected_state:
            return _frontend_auth_error_redirect(
                request,
                "Authentication state validation failed.",
                preferred_redirect=redirect_target,
            )

        token_url = _oidc_endpoint("token")
        client_id = str(getattr(settings, "bisque_auth_oidc_client_id", "") or "").strip()
        client_secret = str(getattr(settings, "bisque_auth_oidc_client_secret", "") or "").strip()
        redirect_uri = _oidc_redirect_uri(request)
        if not token_url or not client_id or not redirect_uri:
            return _frontend_auth_error_redirect(
                request,
                "OIDC token exchange is not configured.",
                preferred_redirect=redirect_target,
            )

        token_payload: dict[str, Any]
        try:
            form: dict[str, str] = {
                "grant_type": "authorization_code",
                "code": code_value,
                "redirect_uri": redirect_uri,
                "client_id": client_id,
            }
            if client_secret:
                form["client_secret"] = client_secret
            token_response = httpx.post(
                token_url,
                data=form,
                headers={"Accept": "application/json"},
                timeout=20.0,
            )
            if token_response.status_code >= 400:
                return _frontend_auth_error_redirect(
                    request,
                    "Could not complete login with identity provider.",
                    preferred_redirect=redirect_target,
                )
            raw_payload = token_response.json()
            token_payload = dict(raw_payload) if isinstance(raw_payload, dict) else {}
        except Exception:
            return _frontend_auth_error_redirect(
                request,
                "Authentication service is currently unavailable.",
                preferred_redirect=redirect_target,
            )

        access_token = str(token_payload.get("access_token") or "").strip()
        id_token = str(token_payload.get("id_token") or "").strip()
        if not access_token and not id_token:
            return _frontend_auth_error_redirect(
                request,
                "Identity provider did not return usable tokens.",
                preferred_redirect=redirect_target,
            )

        userinfo_claims = _fetch_oidc_userinfo(access_token)
        id_token_claims = _jwt_claims(id_token)
        username = _resolve_oidc_username(
            userinfo_claims=userinfo_claims,
            id_token_claims=id_token_claims,
        )
        if not username:
            return _frontend_auth_error_redirect(
                request,
                "Could not determine account identity from OIDC claims.",
                preferred_redirect=redirect_target,
            )

        root = str(getattr(settings, "bisque_root", "") or "").strip().rstrip("/")
        session = _create_bisque_session(
            username=username,
            password="",
            bisque_root=root,
            mode="bisque",
            auth_provider="oidc",
            access_token=access_token or None,
            id_token=id_token or None,
        )
        response = RedirectResponse(redirect_target, status_code=302)
        response.set_cookie(
            key=bisque_session_cookie_name,
            value=str(session["session_id"]),
            max_age=bisque_session_ttl_seconds,
            httponly=True,
            secure=_should_secure_cookie(request),
            samesite="lax",
            path="/",
        )
        response.delete_cookie(
            key=oidc_state_cookie_name,
            path="/",
        )
        response.delete_cookie(
            key=oidc_next_cookie_name,
            path="/",
        )
        return response

    @v1.post("/auth/login", response_model=BisqueAuthSessionResponse)
    @legacy.post("/auth/login", response_model=BisqueAuthSessionResponse)
    def auth_login(
        req: BisqueAuthLoginRequest,
        request: FastAPIRequest,
        response: FastAPIResponse,
    ) -> BisqueAuthSessionResponse:
        auth_mode = _bisque_auth_mode()
        if auth_mode == "oidc":
            if not _is_browser_oidc_enabled():
                raise HTTPException(
                    status_code=503,
                    detail="OIDC login is required, but OIDC is not configured.",
                )
            raise HTTPException(
                status_code=403,
                detail="Username/password login is disabled. Use BisQue SSO.",
            )
        username = str(req.username or "").strip()
        password = str(req.password or "").strip()
        if not username or not password:
            raise HTTPException(status_code=400, detail="Username and password are required.")

        root = str(getattr(settings, "bisque_root", "") or "").strip().rstrip("/")
        passthrough_cookie_header = _extract_passthrough_bisque_cookie_header(request)
        should_try_local_token = (
            bool(getattr(settings, "bisque_auth_local_token_enabled", False)) or auth_mode == "dual"
        )
        local_access_token: str | None = None
        token_username: str | None = None
        if should_try_local_token:
            try:
                local_access_token = _request_bisque_local_access_token(
                    username=username,
                    password=password,
                    bisque_root=root,
                )
                token_username = _validate_bisque_access_token(
                    access_token=local_access_token,
                    requested_username=username,
                    bisque_root=root,
                )
            except HTTPException:
                local_access_token = None
                token_username = None

        if local_access_token and token_username:
            session = _create_bisque_session(
                username=token_username,
                password=password,
                bisque_root=root,
                mode="bisque",
                auth_provider="local_token",
                access_token=local_access_token,
                bisque_cookie_header=passthrough_cookie_header,
            )
        else:
            validated_username = _validate_bisque_credentials(
                username=username,
                password=password,
                bisque_root=root,
            )
            session = _create_bisque_session(
                username=validated_username,
                password=password,
                bisque_root=root,
                mode="bisque",
                auth_provider="local",
                bisque_cookie_header=passthrough_cookie_header,
            )
        response.set_cookie(
            key=bisque_session_cookie_name,
            value=str(session["session_id"]),
            max_age=bisque_session_ttl_seconds,
            httponly=True,
            secure=_should_secure_cookie(request),
            samesite="lax",
            path="/",
        )
        return _session_to_auth_response(session)

    @v1.post("/auth/guest", response_model=BisqueAuthSessionResponse)
    @legacy.post("/auth/guest", response_model=BisqueAuthSessionResponse)
    def auth_guest(
        req: BisqueAuthGuestRequest,
        request: FastAPIRequest,
        response: FastAPIResponse,
    ) -> BisqueAuthSessionResponse:
        if _bisque_auth_mode() == "oidc":
            raise HTTPException(
                status_code=403,
                detail="Guest access is disabled. Sign in with BisQue SSO.",
            )
        root = str(getattr(settings, "bisque_root", "") or "").strip().rstrip("/")
        guest_profile = {
            "name": str(req.name or "").strip(),
            "email": str(req.email or "").strip(),
            "affiliation": str(req.affiliation or "").strip(),
        }
        session = _create_bisque_session(
            username=guest_profile["name"],
            password="",
            bisque_root=root,
            mode="guest",
            guest_profile=guest_profile,
            auth_provider="guest",
        )
        response.set_cookie(
            key=bisque_session_cookie_name,
            value=str(session["session_id"]),
            max_age=bisque_session_ttl_seconds,
            httponly=True,
            secure=_should_secure_cookie(request),
            samesite="lax",
            path="/",
        )
        return _session_to_auth_response(session)

    @v1.get("/auth/session", response_model=BisqueAuthSessionResponse)
    @legacy.get("/auth/session", response_model=BisqueAuthSessionResponse)
    def auth_session(
        request: FastAPIRequest,
        response: FastAPIResponse,
        bisque_session: str | None = Cookie(default=None, alias=bisque_session_cookie_name),
        anonymous_session: str | None = Cookie(default=None, alias=anonymous_session_cookie_name),
    ) -> BisqueAuthSessionResponse:
        del anonymous_session
        session = _get_bisque_session(bisque_session)
        if session:
            session = _refresh_bisque_session_from_browser_cookie_if_needed(
                session=session,
                request=request,
                response=response,
            )
            return _session_to_auth_response(_touch_bisque_session(session))
        bootstrap_session = _bootstrap_bisque_session_from_browser_cookie(
            request=request,
            response=response,
        )
        if not bootstrap_session:
            return BisqueAuthSessionResponse(authenticated=False)
        session = bootstrap_session
        return _session_to_auth_response(_touch_bisque_session(session))

    @v1.post("/auth/logout", response_model=BisqueAuthSessionResponse)
    @legacy.post("/auth/logout", response_model=BisqueAuthSessionResponse)
    def auth_logout(
        response: FastAPIResponse,
        bisque_session: str | None = Cookie(default=None, alias=bisque_session_cookie_name),
    ) -> BisqueAuthSessionResponse:
        _delete_bisque_session(bisque_session)
        response.delete_cookie(
            key=bisque_session_cookie_name,
            path="/",
        )
        response.delete_cookie(
            key=anonymous_session_cookie_name,
            path="/",
        )
        response.delete_cookie(
            key=oidc_state_cookie_name,
            path="/",
        )
        response.delete_cookie(
            key=oidc_next_cookie_name,
            path="/",
        )
        return BisqueAuthSessionResponse(authenticated=False)

    @v1.get("/auth/logout/browser")
    @legacy.get("/auth/logout/browser")
    def auth_logout_browser(
        request: FastAPIRequest,
        next: str | None = Query(default=None),
        bisque_session: str | None = Cookie(default=None, alias=bisque_session_cookie_name),
    ) -> RedirectResponse:
        session = _get_bisque_session(bisque_session)
        id_token_hint = str((session or {}).get("id_token") or "").strip() or None
        _delete_bisque_session(bisque_session)
        preferred_redirect = _resolve_frontend_redirect_target(request, next)
        if _oidc_via_bisque_login_enabled():
            root = str(getattr(settings, "bisque_root", "") or "").strip().rstrip("/")
            if root:
                redirect_target = _append_query_params(
                    f"{root}/auth_service/oidc_logout",
                    {
                        "came_from": _default_logout_redirect_url(
                            request, preferred=preferred_redirect
                        )
                    },
                )
            else:
                redirect_target = _default_logout_redirect_url(
                    request, preferred=preferred_redirect
                )
        else:
            redirect_target = (
                _build_oidc_logout_url(
                    id_token_hint=id_token_hint,
                    request=request,
                    preferred_redirect=preferred_redirect,
                )
                if _is_oidc_enabled()
                else _default_logout_redirect_url(request, preferred=preferred_redirect)
            )
        response = RedirectResponse(redirect_target, status_code=302)
        response.delete_cookie(
            key=bisque_session_cookie_name,
            path="/",
        )
        response.delete_cookie(
            key=anonymous_session_cookie_name,
            path="/",
        )
        response.delete_cookie(
            key=oidc_state_cookie_name,
            path="/",
        )
        response.delete_cookie(
            key=oidc_next_cookie_name,
            path="/",
        )
        response.delete_cookie(
            key="authtkt",
            path="/",
        )
        response.delete_cookie(
            key="tg-visit",
            path="/",
        )
        response.delete_cookie(
            key="tg-remember",
            path="/",
        )
        return response

    @v1.get("/admin/overview", response_model=AdminOverviewResponse)
    @legacy.get("/admin/overview", response_model=AdminOverviewResponse)
    def admin_overview(
        top_users: int = Query(default=8, ge=1, le=50),
        issue_limit: int = Query(default=12, ge=1, le=100),
        _admin: dict[str, Any] = Depends(_require_admin_session),
    ) -> AdminOverviewResponse:
        del _admin
        payload = store.admin_overview(
            top_user_limit=max(1, int(top_users)),
            issue_limit=max(1, int(issue_limit)),
        )
        return AdminOverviewResponse(
            generated_at=payload.get("generated_at") or datetime.utcnow().isoformat(),
            kpis=AdminPlatformKpis.model_validate(payload.get("kpis") or {}),
            usage_last_24h=[
                AdminUsageBucket.model_validate(item)
                for item in (payload.get("usage_last_24h") or [])
            ],
            tool_usage_7d=[
                AdminToolUsageRecord.model_validate(item)
                for item in (payload.get("tool_usage_7d") or [])
            ],
            top_users=[
                AdminUserSummary.model_validate(item) for item in (payload.get("top_users") or [])
            ],
            recent_issues=[
                AdminIssueRecord.model_validate(item)
                for item in (payload.get("recent_issues") or [])
            ],
        )

    @v1.get("/admin/users", response_model=AdminUserListResponse)
    @legacy.get("/admin/users", response_model=AdminUserListResponse)
    def admin_users(
        limit: int = Query(default=200, ge=1, le=1000),
        q: str | None = Query(default=None),
        _admin: dict[str, Any] = Depends(_require_admin_session),
    ) -> AdminUserListResponse:
        del _admin
        rows = store.list_admin_users_summary(limit=max(1, int(limit)), query=q)
        users = [AdminUserSummary.model_validate(row) for row in rows]
        return AdminUserListResponse(count=len(users), users=users)

    @v1.get("/admin/runs", response_model=AdminRunListResponse)
    @legacy.get("/admin/runs", response_model=AdminRunListResponse)
    def admin_runs(
        limit: int = Query(default=200, ge=1, le=1000),
        offset: int = Query(default=0, ge=0),
        status: str | None = Query(default=None),
        user_id: str | None = Query(default=None),
        q: str | None = Query(default=None),
        _admin: dict[str, Any] = Depends(_require_admin_session),
    ) -> AdminRunListResponse:
        del _admin
        rows = store.list_admin_runs(
            limit=max(1, int(limit)),
            offset=max(0, int(offset)),
            status=status,
            user_id=user_id,
            query=q,
        )
        records = [AdminRunRecord.model_validate(row) for row in rows]
        return AdminRunListResponse(count=len(records), runs=records)

    @v1.get("/admin/issues", response_model=AdminIssueListResponse)
    @legacy.get("/admin/issues", response_model=AdminIssueListResponse)
    def admin_issues(
        limit: int = Query(default=25, ge=1, le=200),
        _admin: dict[str, Any] = Depends(_require_admin_session),
    ) -> AdminIssueListResponse:
        del _admin
        rows = store.list_admin_issues(limit=max(1, int(limit)))
        issues = [AdminIssueRecord.model_validate(row) for row in rows]
        return AdminIssueListResponse(count=len(issues), issues=issues)

    @v1.post("/admin/runs/{run_id}/cancel", response_model=AdminRunActionResponse)
    @legacy.post("/admin/runs/{run_id}/cancel", response_model=AdminRunActionResponse)
    def admin_cancel_run(
        run_id: str,
        _admin: dict[str, Any] = Depends(_require_admin_session),
    ) -> AdminRunActionResponse:
        result = store.admin_cancel_run(
            run_id=run_id,
            reason=f"Canceled by admin {str(_admin.get('username') or '').strip() or 'user'}",
        )
        if result is None:
            raise HTTPException(status_code=404, detail="Run not found")
        if bool(result.get("updated")):
            store.append_event(
                run_id=run_id,
                event_type="admin_cancelled",
                payload={
                    "actor": str(_admin.get("username") or "").strip() or "admin",
                },
                level="warning",
            )
        return AdminRunActionResponse.model_validate(result)

    @v1.delete(
        "/admin/conversations/{conversation_id}",
        response_model=AdminConversationActionResponse,
    )
    @legacy.delete(
        "/admin/conversations/{conversation_id}",
        response_model=AdminConversationActionResponse,
    )
    def admin_delete_conversation(
        conversation_id: str,
        user_id: str = Query(..., min_length=1),
        _admin: dict[str, Any] = Depends(_require_admin_session),
    ) -> AdminConversationActionResponse:
        del _admin
        deleted = store.admin_delete_conversation_for_user(
            conversation_id=conversation_id,
            user_id=user_id,
        )
        if not deleted:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return AdminConversationActionResponse(
            conversation_id=conversation_id,
            user_id=user_id,
            deleted=True,
        )

    def _finalize_upload_from_path(
        *,
        file_id: str,
        original_name: str,
        content_type: str | None,
        source_path: Path,
        user_id: str | None = None,
        source_type: str | None = None,
        source_uri: str | None = None,
        client_view_url: str | None = None,
        image_service_url: str | None = None,
        resource_kind: str | None = None,
        metadata: dict[str, Any] | None = None,
        expected_size_bytes: int | None = None,
        expected_sha256: str | None = None,
        sync_status: str | None = None,
    ) -> UploadedFileRecord:
        if not source_path.exists() or not source_path.is_file():
            raise HTTPException(
                status_code=404, detail=f"Upload source file missing: {source_path}"
            )

        digest = hashlib.sha256()
        size_bytes = 0
        with source_path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                size_bytes += len(chunk)
                digest.update(chunk)

        if expected_size_bytes is not None and int(expected_size_bytes) != int(size_bytes):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Upload size mismatch for {original_name}: "
                    f"expected {expected_size_bytes} bytes, got {size_bytes} bytes."
                ),
            )

        sha256 = digest.hexdigest()
        if expected_sha256 and str(expected_sha256).strip().lower() != sha256.lower():
            raise HTTPException(
                status_code=400,
                detail=f"Upload checksum mismatch for {original_name}.",
            )

        final_name = f"{file_id}__{sha256[:12]}__{_safe_artifact_name(original_name)}"
        final_path = _upload_staging_dir_for_user(user_id) / final_name
        source_path.replace(final_path)

        created_at = datetime.utcnow()
        inferred_kind = str(resource_kind or "").strip().lower() or _infer_resource_kind(
            original_name=original_name,
            content_type=content_type,
            source_uri=source_uri,
        )
        store.put_upload(
            file_id=file_id,
            original_name=_safe_artifact_name(original_name),
            stored_path=str(final_path),
            content_type=content_type,
            size_bytes=size_bytes,
            sha256=sha256,
            created_at=created_at.isoformat() + "Z",
            user_id=user_id,
            source_type=str(source_type or "upload"),
            source_uri=source_uri,
            client_view_url=client_view_url,
            image_service_url=image_service_url,
            resource_kind=inferred_kind,
            metadata=metadata,
            staging_path=str(final_path),
            cache_path=None,
            canonical_resource_uniq=None,
            canonical_resource_uri=None,
            sync_status=str(sync_status or upload_sync_status_local_complete),
            sync_error=None,
            sync_retry_count=0,
            sync_started_at=None,
            sync_completed_at=None,
            sync_run_id=None,
        )
        upload_row = store.get_upload(file_id, user_id=user_id)
        if upload_row is not None:
            _enrich_upload_metadata_context(upload_row)
        return UploadedFileRecord(
            file_id=file_id,
            original_name=_safe_artifact_name(original_name),
            content_type=content_type,
            size_bytes=size_bytes,
            sha256=sha256,
            created_at=created_at,
            sync_status=upload_sync_status_local_complete,
            canonical_resource_uniq=None,
            canonical_resource_uri=None,
            client_view_url=client_view_url,
            image_service_url=image_service_url,
            sync_error=None,
            sync_run_id=None,
        )

    def _uploaded_record_from_store(upload_row: dict[str, Any]) -> UploadedFileRecord:
        created_at_raw = str(upload_row.get("created_at") or "").strip()
        created_at = datetime.utcnow()
        if created_at_raw:
            normalized = created_at_raw.replace("Z", "+00:00")
            try:
                created_at = datetime.fromisoformat(normalized)
            except Exception:
                created_at = datetime.utcnow()
        return UploadedFileRecord(
            file_id=_catalog_file_id_for_upload(upload_row),
            original_name=str(upload_row.get("original_name") or "upload.bin"),
            content_type=str(upload_row.get("content_type") or "").strip() or None,
            size_bytes=int(upload_row.get("size_bytes") or 0),
            sha256=str(upload_row.get("sha256") or ""),
            created_at=created_at,
            sync_status=str(upload_row.get("sync_status") or "").strip() or None,
            canonical_resource_uniq=(
                str(upload_row.get("canonical_resource_uniq") or "").strip() or None
            ),
            canonical_resource_uri=(
                str(upload_row.get("canonical_resource_uri") or "").strip() or None
            ),
            client_view_url=str(upload_row.get("client_view_url") or "").strip() or None,
            image_service_url=str(upload_row.get("image_service_url") or "").strip() or None,
            sync_error=str(upload_row.get("sync_error") or "").strip() or None,
            sync_run_id=str(upload_row.get("sync_run_id") or "").strip() or None,
        )

    def _serialize_upload_session(
        session_row: dict[str, Any],
    ) -> ResumableUploadSessionResponse:
        uploaded_record: UploadedFileRecord | None = None
        file_id = str(session_row.get("file_id") or "").strip()
        user_id = str(session_row.get("user_id") or "").strip() or None
        if file_id:
            upload_row = store.get_upload(file_id, user_id=user_id)
            if upload_row:
                uploaded_record = _uploaded_record_from_store(upload_row)
        return ResumableUploadSessionResponse(
            upload_id=str(session_row.get("session_id") or ""),
            file_name=str(session_row.get("original_name") or "upload.bin"),
            size_bytes=int(session_row.get("size_bytes") or 0),
            content_type=str(session_row.get("content_type") or "").strip() or None,
            chunk_size_bytes=int(session_row.get("chunk_size_bytes") or 0),
            bytes_received=int(session_row.get("bytes_received") or 0),
            status=str(session_row.get("status") or "active"),  # type: ignore[arg-type]
            uploaded=uploaded_record,
            error=str(session_row.get("error") or "").strip() or None,
        )

    def _repair_stale_completed_upload_session(session_row: dict[str, Any]) -> bool:
        if str(session_row.get("status") or "active") != "completed":
            return False

        session_id = str(session_row.get("session_id") or "").strip()
        user_id = str(session_row.get("user_id") or "").strip()
        file_id = str(session_row.get("file_id") or "").strip()
        if file_id and store.get_upload(file_id, user_id=user_id):
            return False

        detail = (
            "Stale completed upload session references missing uploaded file."
            if file_id
            else "Stale completed upload session has no uploaded file reference."
        )
        if session_id and user_id:
            store.fail_upload_session(session_id=session_id, user_id=user_id, error=detail)
        session_row["status"] = "failed"
        session_row["error"] = detail
        return True

    def _session_temp_path(session_row: dict[str, Any]) -> Path:
        raw = str(session_row.get("temp_path") or "").strip()
        if not raw:
            raise HTTPException(status_code=500, detail="Upload session temp path is missing.")
        candidate = Path(raw).expanduser().resolve()
        root = upload_resumable_root.resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail="Invalid resumable upload temp path."
            ) from exc
        return candidate

    def _upload_sync_session_id(bisque_auth: dict[str, Any] | None) -> str | None:
        return str((bisque_auth or {}).get("session_id") or "").strip() or None

    def _can_attempt_bisque_sync(
        *,
        upload_row: dict[str, Any],
        bisque_auth: dict[str, Any] | None,
    ) -> bool:
        username, password, access_token, cookie_header, bisque_root = (
            _resolve_upload_bisque_auth_material(upload_row, bisque_auth)
        )
        if not bisque_root:
            return False
        return bool(access_token or cookie_header or (username and password))

    def _build_upload_sync_run(
        *,
        upload_row: dict[str, Any],
        user_id: str | None,
    ) -> WorkflowRun:
        display_name = str(upload_row.get("original_name") or upload_row.get("file_id") or "upload")
        run = WorkflowRun.new(
            goal=f"Sync upload to BisQue: {display_name}",
            plan=None,
            workflow_kind="upload_bisque_sync",
            mode="durable",
            planner_version="upload_sync_v1",
            agent_role="bisque_upload_sync",
            checkpoint_state={
                "file_id": str(upload_row.get("file_id") or "").strip() or None,
                "sync_status": upload_sync_status_queued,
            },
        )
        store.create_run(run)
        store.set_run_metadata(
            run.run_id,
            user_id=user_id,
            conversation_id=None,
            workflow_kind="upload_bisque_sync",
            mode="durable",
            planner_version="upload_sync_v1",
            agent_role="bisque_upload_sync",
            checkpoint_state={
                "file_id": str(upload_row.get("file_id") or "").strip() or None,
                "sync_status": upload_sync_status_queued,
            },
        )
        store.append_event(
            run.run_id,
            "upload_local_completed",
            {
                "file_id": str(upload_row.get("file_id") or "").strip() or None,
                "original_name": str(upload_row.get("original_name") or "").strip() or None,
                "size_bytes": int(upload_row.get("size_bytes") or 0),
            },
        )
        store.append_event(
            run.run_id,
            "bisque_sync_queued",
            {
                "file_id": str(upload_row.get("file_id") or "").strip() or None,
            },
        )
        return run

    def _delete_bisque_resource_for_upload(
        *,
        upload_row: dict[str, Any],
        bisque_auth: dict[str, Any] | None,
    ) -> None:
        resource_uri = (
            str(upload_row.get("canonical_resource_uri") or "").strip()
            or str(upload_row.get("source_uri") or "").strip()
        )
        if not resource_uri:
            return
        try:
            from bqapi.comm import BQSession  # type: ignore

            from src.tools import _apply_bisque_auth_preference, _init_bq_session
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"BisQue delete support unavailable: {exc}"
            ) from exc

        username, password, access_token, cookie_header, bisque_root = (
            _resolve_upload_bisque_auth_material(upload_row, bisque_auth)
        )
        if not bisque_root:
            raise HTTPException(status_code=503, detail="BisQue root is not configured.")
        username, password, access_token, cookie_header = _apply_bisque_auth_preference(
            username=username,
            password=password,
            access_token=access_token,
            cookie_header=cookie_header,
            preferred_auth_mode="basic",
        )
        bq = BQSession()
        try:
            _init_bq_session(
                bq,
                username=username,
                password=password,
                access_token=access_token,
                bisque_root=bisque_root,
                cookie_header=cookie_header,
            )
            bq.deletexml(resource_uri)
        except Exception as exc:
            error_text = str(exc).lower()
            if "404" in error_text or "not found" in error_text:
                return
            raise HTTPException(
                status_code=502, detail=f"Failed to delete BisQue resource: {exc}"
            ) from exc

    def _execute_upload_sync(
        *,
        file_id: str,
        user_id: str | None,
        run_id: str | None = None,
    ) -> UploadedFileRecord:
        upload_row = store.get_upload(file_id, user_id=user_id)
        if upload_row is None:
            raise HTTPException(status_code=404, detail=f"Upload not found: {file_id}")
        if str(upload_row.get("sync_status") or "").strip().lower() == upload_sync_status_succeeded:
            return _uploaded_record_from_store(upload_row)

        staged_path = _safe_upload_local_path(
            str(upload_row.get("staging_path") or upload_row.get("stored_path") or "")
        )
        if staged_path is None or not staged_path.exists() or not staged_path.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"Local staged upload missing for file_id={upload_row.get('file_id') or file_id!s}",
            )
        try:
            from bqapi.comm import BQSession  # type: ignore

            from src.tools import (
                _build_bisque_resource_links,
                _init_bq_session,
            )
            from src.tools import (
                _extract_bisque_resource_uniq as _extract_postblob_resource_uniq,
            )
            from src.tools import (
                _extract_bisque_resource_uri as _extract_postblob_resource_uri,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"BisQue upload support unavailable: {exc}"
            ) from exc

        username, password, access_token, cookie_header, bisque_root = (
            _resolve_upload_bisque_auth_material(upload_row, None)
        )
        if not bisque_root:
            raise HTTPException(status_code=503, detail="BisQue root is not configured.")

        started_at = datetime.utcnow().isoformat() + "Z"
        _persist_upload_row(
            upload_row,
            sync_status=upload_sync_status_running,
            sync_started_at=started_at,
            sync_error=None,
            sync_run_id=run_id or str(upload_row.get("sync_run_id") or "").strip() or None,
            sync_retry_count=int(upload_row.get("sync_retry_count") or 0),
        )
        if run_id:
            store.append_event(
                run_id,
                "bisque_sync_started",
                {
                    "file_id": str(upload_row.get("file_id") or "").strip() or None,
                    "original_name": str(upload_row.get("original_name") or "").strip() or None,
                },
            )

        bq = BQSession()
        try:
            _init_bq_session(
                bq,
                username=username,
                password=password,
                access_token=access_token,
                bisque_root=bisque_root,
                cookie_header=cookie_header,
            )
            resource_xml = _build_bisque_upload_xml(upload_row, staged_path.name)
            response = bq.postblob(str(staged_path), xml=resource_xml)
            resource_uri = _extract_postblob_resource_uri(response)
            resource_uniq = _extract_postblob_resource_uniq(response)
            if not resource_uri and resource_uniq:
                resource_uri = f"{bisque_root.rstrip('/')}/data_service/{resource_uniq}"
            if not resource_uri:
                raise ValueError("BisQue upload did not return a resource URI.")
            links = _build_bisque_resource_links(resource_uri, bisque_root)
            resolved_uri = str(links.get("resource_uri") or resource_uri).strip()
            resolved_uniq = str(
                links.get("resource_uniq") or resource_uniq or ""
            ).strip() or _extract_bisque_resource_uniq(resolved_uri)
            cache_destination = _cache_path_for_upload(upload_row, user_id)
            cache_destination.parent.mkdir(parents=True, exist_ok=True)
            staged_path.replace(cache_destination)
            _persist_upload_row(
                upload_row,
                stored_path=str(cache_destination),
                staging_path=None,
                cache_path=str(cache_destination),
                source_uri=resolved_uri,
                client_view_url=str(links.get("client_view_url") or "").strip() or None,
                image_service_url=str(links.get("image_service_url") or "").strip() or None,
                canonical_resource_uniq=resolved_uniq,
                canonical_resource_uri=resolved_uri,
                sync_status=upload_sync_status_succeeded,
                sync_error=None,
                sync_run_id=run_id or str(upload_row.get("sync_run_id") or "").strip() or None,
                sync_completed_at=datetime.utcnow().isoformat() + "Z",
            )
            refreshed = store.get_upload(str(upload_row.get("file_id") or file_id), user_id=user_id)
            if refreshed is None:
                raise ValueError("Failed to reload upload record after BisQue sync.")
            if run_id:
                store.append_event(
                    run_id,
                    "bisque_sync_succeeded",
                    {
                        "file_id": str(refreshed.get("file_id") or "").strip() or None,
                        "canonical_resource_uniq": (
                            str(refreshed.get("canonical_resource_uniq") or "").strip() or None
                        ),
                        "canonical_resource_uri": (
                            str(refreshed.get("canonical_resource_uri") or "").strip() or None
                        ),
                    },
                )
                store.append_event(
                    run_id,
                    "resource_canonicalized",
                    {
                        "file_id": str(refreshed.get("file_id") or "").strip() or None,
                        "canonical_resource_uniq": (
                            str(refreshed.get("canonical_resource_uniq") or "").strip() or None
                        ),
                    },
                )
            return _uploaded_record_from_store(refreshed)
        except Exception as exc:
            retry_count = int(upload_row.get("sync_retry_count") or 0) + 1
            _persist_upload_row(
                upload_row,
                sync_status=upload_sync_status_failed,
                sync_error=str(exc),
                sync_retry_count=retry_count,
                sync_run_id=run_id or str(upload_row.get("sync_run_id") or "").strip() or None,
            )
            if run_id:
                store.append_event(
                    run_id,
                    "bisque_sync_failed",
                    {
                        "file_id": str(upload_row.get("file_id") or "").strip() or None,
                        "error": str(exc),
                        "retry_count": retry_count,
                    },
                    level="error",
                )
            raise

    def _run_upload_sync_inline(
        *,
        run_id: str,
        file_id: str,
        user_id: str | None,
    ) -> None:
        store.update_status(run_id, RunStatus.RUNNING)
        store.append_event(run_id, "run_started", {"goal": f"Sync upload {file_id} to BisQue"})
        try:
            _execute_upload_sync(file_id=file_id, user_id=user_id, run_id=run_id)
            store.update_status(run_id, RunStatus.SUCCEEDED)
            store.append_event(run_id, "run_succeeded", {})
        except Exception as exc:
            store.update_status(run_id, RunStatus.FAILED, error=str(exc))
            store.append_event(run_id, "run_failed", {"error": str(exc)}, level="error")

    def _enqueue_plan_inline_thread(*, run_id: str, plan: WorkflowPlan, thread_name: str) -> None:
        """Run a stored workflow plan in a background thread for this release."""

        def _run_plan() -> None:
            executor = PlanExecutor(store)
            executor.execute(run_id, plan)

        worker = Thread(
            target=_run_plan,
            daemon=True,
            name=thread_name,
        )
        worker.start()

    async def _enqueue_upload_sync(
        *,
        upload_row: dict[str, Any],
        user_id: str | None,
        bisque_auth: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not _can_attempt_bisque_sync(upload_row=upload_row, bisque_auth=bisque_auth):
            return upload_row
        run = _build_upload_sync_run(upload_row=upload_row, user_id=user_id)
        _persist_upload_row(
            upload_row,
            sync_status=upload_sync_status_queued,
            sync_error=None,
            sync_run_id=run.run_id,
        )
        store.append_event(run.run_id, "enqueued", {"queue": "inline-thread"})
        worker = Thread(
            target=_run_upload_sync_inline,
            kwargs={
                "run_id": run.run_id,
                "file_id": str(upload_row.get("file_id") or "").strip(),
                "user_id": user_id,
            },
            daemon=True,
            name=f"upload-sync-{run.run_id[:8]}",
        )
        worker.start()
        refreshed = store.get_upload(str(upload_row.get("file_id") or ""), user_id=user_id)
        return refreshed or upload_row

    @v1.post("/uploads", response_model=UploadFilesResponse)
    @legacy.post("/uploads", response_model=UploadFilesResponse)
    async def upload_files(
        files: list[UploadFile] = File(...),
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> UploadFilesResponse:
        del _auth
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        if len(files) > upload_max_files_per_request:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Too many files in one request ({len(files)}). "
                    f"Max allowed is {upload_max_files_per_request}."
                ),
            )

        uploaded_records: list[UploadedFileRecord] = []
        bisque_session_id = _upload_sync_session_id(bisque_auth)
        for upload in files:
            original_name = _safe_artifact_name(upload.filename or "upload.bin")
            content_type = (
                (upload.content_type or "").strip()
                or mimetypes.guess_type(original_name)[0]
                or "application/octet-stream"
            )
            file_id = uuid4().hex
            temp_path = upload_store_root / f".{file_id}.part"
            digest = hashlib.sha256()
            total_bytes = 0

            try:
                with temp_path.open("wb") as out:
                    while True:
                        chunk = await upload.read(1024 * 1024)
                        if not chunk:
                            break
                        total_bytes += len(chunk)
                        if total_bytes > upload_max_file_size_bytes:
                            raise HTTPException(
                                status_code=413,
                                detail=(
                                    f"Upload exceeded maximum file size of "
                                    f"{upload_max_file_size_bytes // (1024 * 1024)} MB: {original_name}"
                                ),
                            )
                        digest.update(chunk)
                        out.write(chunk)
            except HTTPException:
                if temp_path.exists():
                    temp_path.unlink(missing_ok=True)
                raise
            except Exception as exc:
                if temp_path.exists():
                    temp_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=500, detail=f"Failed to store upload: {exc}"
                ) from exc
            finally:
                with suppress(Exception):
                    await upload.close()

            sha256 = digest.hexdigest()
            final_name = f"{file_id}__{sha256[:12]}__{original_name}"
            final_path = _upload_staging_dir_for_user(user_id) / final_name
            temp_path.replace(final_path)

            created_at = datetime.utcnow()
            resource_kind = _infer_resource_kind(
                original_name=original_name,
                content_type=content_type,
            )
            store.put_upload(
                file_id=file_id,
                original_name=original_name,
                stored_path=str(final_path),
                content_type=content_type,
                size_bytes=total_bytes,
                sha256=sha256,
                created_at=created_at.isoformat() + "Z",
                user_id=user_id,
                source_type="upload",
                resource_kind=resource_kind,
                metadata={"bisque_session_id": bisque_session_id} if bisque_session_id else None,
                staging_path=str(final_path),
                cache_path=None,
                canonical_resource_uniq=None,
                canonical_resource_uri=None,
                sync_status=upload_sync_status_local_complete,
                sync_error=None,
                sync_retry_count=0,
                sync_started_at=None,
                sync_completed_at=None,
                sync_run_id=None,
            )
            upload_row = store.get_upload(file_id, user_id=user_id)
            if upload_row is None:
                raise HTTPException(status_code=500, detail="Failed to persist uploaded file.")
            _enrich_upload_metadata_context(upload_row)
            upload_row = store.get_upload(file_id, user_id=user_id) or upload_row
            refreshed_row = await _enqueue_upload_sync(
                upload_row=upload_row,
                user_id=user_id,
                bisque_auth=bisque_auth,
            )
            uploaded_records.append(_uploaded_record_from_store(refreshed_row))

        return UploadFilesResponse(file_count=len(uploaded_records), uploaded=uploaded_records)

    @v1.post("/uploads/resumable/init", response_model=ResumableUploadSessionResponse)
    @legacy.post("/uploads/resumable/init", response_model=ResumableUploadSessionResponse)
    async def upload_resumable_init(
        req: ResumableUploadInitRequest,
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> ResumableUploadSessionResponse:
        del _auth
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        file_name = _safe_artifact_name(req.file_name)
        size_bytes = int(req.size_bytes)
        if size_bytes > upload_max_file_size_bytes:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Upload exceeded maximum file size of "
                    f"{upload_max_file_size_bytes // (1024 * 1024)} MB: {file_name}"
                ),
            )

        session_args = {
            "session_id": uuid4().hex,
            "user_id": user_id,
            "fingerprint": str(req.fingerprint),
            "original_name": file_name,
            "content_type": (str(req.content_type or "").strip() or None),
            "size_bytes": size_bytes,
            "chunk_size_bytes": int(req.chunk_size_bytes),
        }
        session_args["temp_path"] = str(
            (upload_resumable_root / f"{session_args['session_id']}.part").resolve()
        )
        session: dict[str, Any] | None = None
        for _ in range(2):
            session = store.create_or_resume_upload_session(**session_args)
            if not _repair_stale_completed_upload_session(session):
                break
            session_args["session_id"] = uuid4().hex
            session_args["temp_path"] = str(
                (upload_resumable_root / f"{session_args['session_id']}.part").resolve()
            )
        if session is None:
            raise HTTPException(status_code=500, detail="Failed to initialize upload session.")

        if str(session.get("status") or "active") != "active":
            return _serialize_upload_session(session)

        session_temp = _session_temp_path(session)
        if not session_temp.exists():
            session_temp.parent.mkdir(parents=True, exist_ok=True)
            session_temp.touch()
            if int(session.get("bytes_received") or 0) > 0:
                store.update_upload_session_progress(
                    session_id=str(session.get("session_id") or ""),
                    user_id=user_id,
                    bytes_received=0,
                    status="active",
                )
                session["bytes_received"] = 0

        return _serialize_upload_session(session)

    @v1.get("/uploads/resumable/{upload_id}", response_model=ResumableUploadSessionResponse)
    @legacy.get("/uploads/resumable/{upload_id}", response_model=ResumableUploadSessionResponse)
    async def upload_resumable_status(
        upload_id: str,
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> ResumableUploadSessionResponse:
        del _auth
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        session = store.get_upload_session(session_id=upload_id, user_id=user_id)
        if not session:
            raise HTTPException(status_code=404, detail="Upload session not found.")
        return _serialize_upload_session(session)

    @v1.put("/uploads/resumable/{upload_id}/chunk", response_model=ResumableUploadChunkResponse)
    @legacy.put("/uploads/resumable/{upload_id}/chunk", response_model=ResumableUploadChunkResponse)
    async def upload_resumable_chunk(
        upload_id: str,
        request: FastAPIRequest,
        offset: int = Query(default=0, ge=0),
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> ResumableUploadChunkResponse:
        del _auth
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        session = store.get_upload_session(session_id=upload_id, user_id=user_id)
        if not session:
            raise HTTPException(status_code=404, detail="Upload session not found.")

        status = str(session.get("status") or "active")
        if status != "active":
            raise HTTPException(
                status_code=409,
                detail=f"Upload session is {status}. Chunks can only be appended to active sessions.",
            )

        expected_offset = int(session.get("bytes_received") or 0)
        if offset != expected_offset:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "Upload offset mismatch.",
                    "expected_offset": expected_offset,
                    "received_offset": int(offset),
                    "upload_id": upload_id,
                },
            )

        chunk = await request.body()
        if not chunk:
            raise HTTPException(status_code=400, detail="Chunk body is empty.")

        size_bytes = int(session.get("size_bytes") or 0)
        next_size = expected_offset + len(chunk)
        if next_size > size_bytes:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Chunk exceeds declared size for upload {upload_id}: "
                    f"{next_size} > {size_bytes}."
                ),
            )

        temp_path = _session_temp_path(session)
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        if not temp_path.exists():
            temp_path.touch()

        with temp_path.open("r+b") as handle:
            handle.seek(offset)
            handle.write(chunk)

        store.update_upload_session_progress(
            session_id=upload_id,
            user_id=user_id,
            bytes_received=next_size,
            status="active",
        )
        return ResumableUploadChunkResponse(
            upload_id=upload_id,
            bytes_received=next_size,
            size_bytes=size_bytes,
            complete=next_size >= size_bytes,
            status="active",
        )

    @v1.post(
        "/uploads/resumable/{upload_id}/complete", response_model=ResumableUploadCompleteResponse
    )
    @legacy.post(
        "/uploads/resumable/{upload_id}/complete", response_model=ResumableUploadCompleteResponse
    )
    async def upload_resumable_complete(
        upload_id: str,
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> ResumableUploadCompleteResponse:
        del _auth
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        session = store.get_upload_session(session_id=upload_id, user_id=user_id)
        if not session:
            raise HTTPException(status_code=404, detail="Upload session not found.")

        status = str(session.get("status") or "active")
        if status == "completed":
            if _repair_stale_completed_upload_session(session):
                raise HTTPException(
                    status_code=409,
                    detail="Upload session was stale and has been reset. Retry the upload.",
                )
            file_id = str(session.get("file_id") or "").strip()
            if not file_id:
                raise HTTPException(
                    status_code=500,
                    detail="Upload session is completed but has no uploaded file reference.",
                )
            upload_row = store.get_upload(file_id, user_id=user_id)
            if not upload_row:
                raise HTTPException(
                    status_code=500,
                    detail="Upload session references missing uploaded file.",
                )
            return ResumableUploadCompleteResponse(
                upload_id=upload_id,
                uploaded=_uploaded_record_from_store(upload_row),
            )

        if status != "active":
            raise HTTPException(
                status_code=409,
                detail=f"Upload session is {status}.",
            )

        bytes_received = int(session.get("bytes_received") or 0)
        size_bytes = int(session.get("size_bytes") or 0)
        if bytes_received != size_bytes:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Upload incomplete for {upload_id}: "
                    f"received {bytes_received} of {size_bytes} bytes."
                ),
            )

        temp_path = _session_temp_path(session)
        if not temp_path.exists() or not temp_path.is_file():
            raise HTTPException(status_code=404, detail="Upload data is missing on disk.")

        file_id = uuid4().hex
        bisque_session_id = _upload_sync_session_id(bisque_auth)
        try:
            uploaded_record = _finalize_upload_from_path(
                file_id=file_id,
                original_name=str(session.get("original_name") or "upload.bin"),
                content_type=str(session.get("content_type") or "").strip() or None,
                source_path=temp_path,
                user_id=user_id,
                source_type="upload",
                expected_size_bytes=size_bytes,
                metadata={"bisque_session_id": bisque_session_id} if bisque_session_id else None,
                sync_status=upload_sync_status_local_complete,
            )
        except HTTPException as exc:
            store.fail_upload_session(session_id=upload_id, user_id=user_id, error=str(exc.detail))
            raise
        except Exception as exc:
            store.fail_upload_session(session_id=upload_id, user_id=user_id, error=str(exc))
            raise HTTPException(
                status_code=500, detail=f"Failed to finalize upload: {exc}"
            ) from exc

        store.complete_upload_session(
            session_id=upload_id,
            user_id=user_id,
            file_id=uploaded_record.file_id,
            sha256=uploaded_record.sha256,
        )
        upload_row = store.get_upload(uploaded_record.file_id, user_id=user_id)
        if upload_row is not None:
            refreshed_row = await _enqueue_upload_sync(
                upload_row=upload_row,
                user_id=user_id,
                bisque_auth=bisque_auth,
            )
            uploaded_record = _uploaded_record_from_store(refreshed_row)
        return ResumableUploadCompleteResponse(upload_id=upload_id, uploaded=uploaded_record)

    @v1.post("/internal/uploads/{file_id}/sync")
    @legacy.post("/internal/uploads/{file_id}/sync")
    def sync_uploaded_file_to_bisque(
        file_id: str,
        payload: dict[str, Any] | None = Body(default=None),
        _auth: None = Depends(_require_api_key),
    ) -> dict[str, Any]:
        del _auth
        body = payload if isinstance(payload, dict) else {}
        user_id = str(body.get("user_id") or "").strip() or None
        run_id = str(body.get("run_id") or "").strip() or None
        uploaded = _execute_upload_sync(file_id=file_id, user_id=user_id, run_id=run_id)
        return {
            "success": True,
            "uploaded": uploaded.model_dump(mode="json"),
        }

    def _resolve_chat_bisque_transfer_auth(
        bisque_auth: dict[str, Any] | None,
        *,
        allow_settings_fallback: bool = False,
    ) -> tuple[str | None, str | None, str | None, str | None]:
        request_context = get_request_bisque_auth()
        username = str(getattr(request_context, "username", "") or "").strip() or None
        password = str(getattr(request_context, "password", "") or "").strip() or None
        access_token = str(getattr(request_context, "access_token", "") or "").strip() or None
        cookie_header = (
            str(getattr(request_context, "bisque_cookie_header", "") or "").strip() or None
        )
        if not any((username, password, access_token, cookie_header)) and bisque_auth:
            mode = str(bisque_auth.get("mode") or "").strip().lower()
            if mode == "bisque":
                username = str(bisque_auth.get("username") or "").strip() or None
                access_token = str(bisque_auth.get("access_token") or "").strip() or None
                cookie_header = str(bisque_auth.get("bisque_cookie_header") or "").strip() or None
                password = str(bisque_auth.get("password") or "").strip() or None
        if not (access_token or cookie_header) and username and not password:
            username = None
        if password and not username:
            password = None
        if allow_settings_fallback and not any((username, password, access_token, cookie_header)):
            configured_username = str(getattr(settings, "bisque_user", "") or "").strip() or None
            configured_password = (
                str(getattr(settings, "bisque_password", "") or "").strip() or None
            )
            if configured_username and configured_password:
                username = configured_username
                password = configured_password
        return username, password, access_token, cookie_header

    def _require_bisque_user_execution_access(
        bisque_auth: dict[str, Any] | None,
    ) -> tuple[str | None, str | None, str | None, str | None]:
        if not bisque_auth:
            raise HTTPException(
                status_code=401,
                detail="BisQue sign-in is required for user-scoped BisQue actions.",
            )
        mode = str(bisque_auth.get("mode") or "").strip().lower()
        if mode != "bisque":
            raise HTTPException(
                status_code=403,
                detail="BisQue sign-in is required. Guest sessions cannot access BisQue resources.",
            )
        username, password, access_token, cookie_header = _resolve_chat_bisque_transfer_auth(
            bisque_auth,
            allow_settings_fallback=False,
        )
        if access_token or cookie_header or (username and password):
            return username, password, access_token, cookie_header
        raise HTTPException(
            status_code=401,
            detail="BisQue session credentials are unavailable or expired. Sign in again and retry.",
        )

    def _build_request_bisque_auth_context(
        *,
        bisque_auth: dict[str, Any] | None,
        request: FastAPIRequest | None = None,
    ) -> BisqueAuthContext | None:
        bisque_user, bisque_password, bisque_access_token, bisque_cookie_header = (
            _resolve_chat_bisque_transfer_auth(
                bisque_auth,
                allow_settings_fallback=False,
            )
        )
        if not bisque_cookie_header and request is not None:
            bisque_cookie_header = _extract_passthrough_bisque_cookie_header(request)
        bisque_root = str((bisque_auth or {}).get("bisque_root") or "").strip().rstrip("/") or str(
            getattr(settings, "bisque_root", "") or ""
        ).strip().rstrip("/")
        if not any(
            (
                str(bisque_user or "").strip(),
                str(bisque_password or "").strip(),
                str(bisque_access_token or "").strip(),
                str(bisque_cookie_header or "").strip(),
            )
        ):
            return None
        return BisqueAuthContext(
            username=bisque_user,
            password=bisque_password,
            bisque_root=bisque_root or None,
            access_token=bisque_access_token,
            id_token=str((bisque_auth or {}).get("id_token") or "").strip() or None,
            bisque_cookie_header=bisque_cookie_header,
        )

    def _has_chat_bisque_execution_access(bisque_auth: dict[str, Any] | None) -> bool:
        username, password, access_token, cookie_header = _resolve_chat_bisque_transfer_auth(
            bisque_auth,
            allow_settings_fallback=False,
        )
        return bool(access_token or cookie_header or (username and password))

    def _fetch_bisque_dataset_members_by_uri(
        *,
        dataset_uri: str,
        bisque_root: str,
        headers: dict[str, str],
    ) -> tuple[str, list[str]]:
        normalized_dataset_uri = _normalize_bisque_resource_uri(dataset_uri, bisque_root)
        root_node = _fetch_xml_url(
            url=f"{normalized_dataset_uri}{'&' if '?' in normalized_dataset_uri else '?'}view=deep",
            headers=headers,
            timeout=45,
        )
        resolved_dataset_uri, members = _extract_dataset_members_from_xml(
            root_node=root_node,
            dataset_name="",
            bisque_root=bisque_root,
        )
        return str(resolved_dataset_uri or normalized_dataset_uri), members

    def _fetch_bisque_resource_metadata(
        *,
        resource_uri: str,
        bisque_root: str,
        headers: dict[str, str],
    ) -> dict[str, Any]:
        node = _fetch_xml_url(
            url=f"{resource_uri}{'&' if '?' in resource_uri else '?'}view=deep",
            headers=headers,
            timeout=45,
        )
        tag_name = str(node.tag or "").strip().lower()
        resource_type = (
            str(node.attrib.get("resource_type") or "").strip().lower()
            or str(node.attrib.get("type") or "").strip().lower()
            or tag_name
        )
        original_name = (
            str(node.attrib.get("name") or "").strip()
            or _extract_bisque_resource_uniq(resource_uri)
            or "bisque-resource"
        )
        if resource_type == "dataset":
            resource_kind = "file"
        elif resource_type in {"image", "video", "table", "file"}:
            resource_kind = resource_type
        else:
            resource_kind = _infer_resource_kind(
                original_name=original_name,
                content_type=None,
                source_uri=resource_uri,
            )
        return {
            "original_name": _safe_artifact_name(original_name),
            "resource_kind": resource_kind,
            "content_type": mimetypes.guess_type(original_name)[0] or None,
        }

    async def _import_bisque_resource_for_user(
        *,
        input_url: str,
        user_id: str | None,
        bisque_root: str,
        bisque_username: str | None,
        bisque_password: str | None,
        bisque_access_token: str | None,
        bisque_cookie_header: str | None,
        import_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        links: dict[str, str | None] = {}
        links = _build_bisque_links(input_url, bisque_root)
        resource_uri = links.get("resource_uri")
        resource_uniq = links.get("resource_uniq")
        if not resource_uri:
            raise ValueError("Could not normalize BisQue resource URL.")

        if user_id:
            existing = _find_upload_by_source_uri(user_id=user_id, source_uri=resource_uri)
        else:
            existing = None
        if existing is not None:
            uploaded_record = _uploaded_record_from_store(existing)
            return {
                "status": "reused",
                "input_url": input_url,
                "resource_uri": resource_uri,
                "resource_uniq": resource_uniq,
                "client_view_url": links.get("client_view_url"),
                "image_service_url": links.get("image_service_url"),
                "download_source": "bisque_reference",
                "uploaded_record": uploaded_record,
                "upload_row": existing,
                "metadata": dict(existing.get("metadata") or {}),
            }

        headers = _build_bisque_request_headers(
            bisque_username=bisque_username,
            bisque_password=bisque_password,
            bisque_access_token=bisque_access_token,
            bisque_cookie_header=bisque_cookie_header,
        )
        metadata_payload = await run_in_threadpool(
            _fetch_bisque_resource_metadata,
            resource_uri=resource_uri,
            bisque_root=bisque_root,
            headers=headers,
        )
        combined_metadata = {
            "input_url": input_url,
            "download_source": "bisque_reference",
        }
        if isinstance(import_metadata, dict):
            combined_metadata.update(import_metadata)
        file_id = uuid4().hex
        store.put_upload(
            file_id=file_id,
            original_name=str(
                metadata_payload.get("original_name") or resource_uniq or "bisque-resource"
            ),
            stored_path="",
            content_type=str(metadata_payload.get("content_type") or "").strip() or None,
            size_bytes=0,
            sha256="",
            created_at=datetime.utcnow().isoformat() + "Z",
            user_id=user_id,
            source_type="bisque_import",
            source_uri=resource_uri,
            client_view_url=links.get("client_view_url"),
            image_service_url=links.get("image_service_url"),
            resource_kind=str(metadata_payload.get("resource_kind") or "file"),
            metadata=combined_metadata,
            staging_path=None,
            cache_path=None,
            canonical_resource_uniq=str(resource_uniq or "").strip() or None,
            canonical_resource_uri=resource_uri,
            sync_status=upload_sync_status_succeeded,
            sync_error=None,
            sync_retry_count=0,
            sync_started_at=None,
            sync_completed_at=datetime.utcnow().isoformat() + "Z",
            sync_run_id=None,
        )
        upload_row = store.get_upload(file_id, user_id=user_id)
        if upload_row is None:
            raise ValueError("Imported upload record could not be reloaded from store.")
        return {
            "status": "imported",
            "input_url": input_url,
            "resource_uri": resource_uri,
            "resource_uniq": resource_uniq,
            "client_view_url": links.get("client_view_url"),
            "image_service_url": links.get("image_service_url"),
            "download_source": "bisque_reference",
            "uploaded_record": _uploaded_record_from_store(upload_row),
            "upload_row": upload_row,
            "metadata": combined_metadata,
        }

    def _create_bisque_child_run(
        *,
        parent_run: WorkflowRun,
        user_id: str | None,
        conversation_id: str | None,
        goal: str,
        source_uri: str,
        dataset_uri: str | None = None,
    ) -> WorkflowRun:
        child_run = WorkflowRun.new(
            goal=goal,
            plan=None,
            workflow_kind="interactive_chat_bisque_resource",
            mode="interactive",
            parent_run_id=parent_run.run_id,
            planner_version="workgraph_v1",
            agent_role="bisque_import",
            checkpoint_state={
                "phase": "queued",
                "source_uri": source_uri,
                "dataset_uri": str(dataset_uri or "").strip() or None,
            },
            trace_group_id=str(parent_run.trace_group_id or "").strip() or conversation_id,
        )
        store.create_run(child_run)
        store.set_run_metadata(
            child_run.run_id,
            user_id=user_id,
            conversation_id=conversation_id,
            workflow_kind="interactive_chat_bisque_resource",
            mode="interactive",
            parent_run_id=parent_run.run_id,
            planner_version="workgraph_v1",
            agent_role="bisque_import",
            checkpoint_state={
                "phase": "queued",
                "source_uri": source_uri,
                "dataset_uri": str(dataset_uri or "").strip() or None,
            },
            trace_group_id=str(parent_run.trace_group_id or "").strip() or conversation_id,
        )
        return child_run

    async def _prepare_bisque_chat_inputs(
        *,
        active_run: WorkflowRun,
        req: ChatRequest,
        user_id: str | None,
        conversation_id: str | None,
        bisque_auth: dict[str, Any] | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        requested_resource_uris = [
            str(item or "").strip() for item in req.resource_uris if str(item or "").strip()
        ]
        requested_dataset_uris = [
            str(item or "").strip() for item in req.dataset_uris if str(item or "").strip()
        ]
        if not requested_resource_uris and not requested_dataset_uris:
            return [], [], []

        _require_authenticated_bisque_auth(
            bisque_auth,
            detail="BisQue authentication is required to import BisQue resources into chat.",
        )

        bisque_root = (
            str(getattr(get_request_bisque_auth(), "bisque_root", "") or "").strip().rstrip("/")
            or str((bisque_auth or {}).get("bisque_root") or "").strip().rstrip("/")
            or str(getattr(settings, "bisque_root", "") or "").strip().rstrip("/")
        )
        bisque_username, bisque_password, bisque_access_token, bisque_cookie_header = (
            _require_bisque_user_execution_access(bisque_auth)
        )
        if not bisque_root:
            raise HTTPException(status_code=400, detail="BisQue root is not configured.")

        headers = _build_bisque_request_headers(
            bisque_username=bisque_username,
            bisque_password=bisque_password,
            bisque_access_token=bisque_access_token,
            bisque_cookie_header=bisque_cookie_header,
        )
        dataset_summaries: list[dict[str, Any]] = []
        fanout_inputs: list[dict[str, Any]] = []
        seen_resource_uris: set[str] = set()

        for dataset_input in list(dict.fromkeys(requested_dataset_uris)):
            store.append_event(
                active_run.run_id,
                "dataset_resolution_started",
                {"dataset_uri": dataset_input},
            )
            normalized_dataset_uri = _normalize_bisque_resource_uri(dataset_input, bisque_root)
            try:
                resolved_dataset_uri, members = await run_in_threadpool(
                    _fetch_bisque_dataset_members_by_uri,
                    dataset_uri=normalized_dataset_uri,
                    bisque_root=bisque_root,
                    headers=headers,
                )
            except Exception as exc:
                store.append_event(
                    active_run.run_id,
                    "dataset_resolution_failed",
                    {
                        "dataset_uri": normalized_dataset_uri,
                        "error": str(exc),
                    },
                    level="error",
                )
                dataset_summaries.append(
                    {
                        "dataset_uri": normalized_dataset_uri,
                        "status": "error",
                        "resource_count": 0,
                        "error": str(exc),
                    }
                )
                continue

            dataset_member_count = 0
            for member in members:
                normalized_member = _normalize_bisque_resource_uri(member, bisque_root)
                if normalized_member in seen_resource_uris:
                    continue
                seen_resource_uris.add(normalized_member)
                dataset_member_count += 1
                fanout_inputs.append(
                    {
                        "input_url": normalized_member,
                        "resource_uri": normalized_member,
                        "dataset_uri": resolved_dataset_uri,
                        "source": "dataset_uri",
                    }
                )
            store.append_event(
                active_run.run_id,
                "dataset_resolution_completed",
                {
                    "dataset_uri": resolved_dataset_uri,
                    "resource_count": dataset_member_count,
                },
            )
            dataset_summaries.append(
                {
                    "dataset_uri": resolved_dataset_uri,
                    "status": "resolved",
                    "resource_count": dataset_member_count,
                }
            )

        for resource_input in list(dict.fromkeys(requested_resource_uris)):
            normalized_resource_uri = _normalize_bisque_resource_uri(resource_input, bisque_root)
            if normalized_resource_uri in seen_resource_uris:
                continue
            seen_resource_uris.add(normalized_resource_uri)
            fanout_inputs.append(
                {
                    "input_url": resource_input,
                    "resource_uri": normalized_resource_uri,
                    "dataset_uri": None,
                    "source": "resource_uri",
                }
            )

        if len(fanout_inputs) > upload_max_files_per_request:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Resolved {len(fanout_inputs)} BisQue resources, which exceeds the per-request "
                    f"limit of {upload_max_files_per_request}."
                ),
            )

        imported_rows: list[dict[str, Any]] = []
        import_summaries: list[dict[str, Any]] = []

        for item in fanout_inputs:
            resource_uri = str(item.get("resource_uri") or "").strip()
            dataset_uri = str(item.get("dataset_uri") or "").strip() or None
            child_run = _create_bisque_child_run(
                parent_run=active_run,
                user_id=user_id,
                conversation_id=conversation_id,
                goal=f"Import BisQue resource {resource_uri}",
                source_uri=resource_uri,
                dataset_uri=dataset_uri,
            )
            store.update_status(child_run.run_id, RunStatus.RUNNING)
            store.append_event(
                active_run.run_id,
                "resource_download_started",
                {
                    "resource_uri": resource_uri,
                    "dataset_uri": dataset_uri,
                    "source": item.get("source"),
                    "child_run_id": child_run.run_id,
                },
            )
            store.append_event(
                child_run.run_id,
                "resource_download_started",
                {
                    "resource_uri": resource_uri,
                    "dataset_uri": dataset_uri,
                    "source": item.get("source"),
                    "parent_run_id": active_run.run_id,
                },
            )
            try:
                import_result = await _import_bisque_resource_for_user(
                    input_url=str(item.get("input_url") or resource_uri),
                    user_id=user_id,
                    bisque_root=bisque_root,
                    bisque_username=bisque_username,
                    bisque_password=bisque_password,
                    bisque_access_token=bisque_access_token,
                    bisque_cookie_header=bisque_cookie_header,
                    import_metadata={"dataset_uri": dataset_uri} if dataset_uri else None,
                )
                upload_row = dict(import_result.get("upload_row") or {})
                if upload_row:
                    materialized_path = _resolved_local_path_for_upload(
                        upload_row,
                        user_id=user_id,
                        bisque_auth=bisque_auth,
                    )
                    if not materialized_path:
                        raise ValueError(
                            "Imported BisQue resource could not be materialized for analysis."
                        )
                    imported_rows.append(upload_row)
                upload_record = import_result.get("uploaded_record")
                upload_payload = (
                    upload_record.model_dump(mode="json")
                    if hasattr(upload_record, "model_dump")
                    else upload_record
                )
                summary_payload = {
                    "resource_uri": resource_uri,
                    "dataset_uri": dataset_uri,
                    "status": str(import_result.get("status") or "imported"),
                    "download_source": str(import_result.get("download_source") or "unknown"),
                    "child_run_id": child_run.run_id,
                    "file_id": str(upload_row.get("file_id") or "").strip() or None,
                    "original_name": str(upload_row.get("original_name") or "").strip() or None,
                    "resource_kind": str(upload_row.get("resource_kind") or "").strip() or None,
                    "client_view_url": str(upload_row.get("client_view_url") or "").strip() or None,
                    "source_type": str(upload_row.get("source_type") or "").strip() or None,
                    "size_bytes": int(upload_row.get("size_bytes") or 0),
                }
                store.append_event(
                    active_run.run_id,
                    "resource_download_completed",
                    summary_payload,
                )
                store.append_event(
                    active_run.run_id,
                    "resource_result_summary",
                    {
                        **summary_payload,
                        "uploaded": upload_payload,
                    },
                )
                store.append_event(
                    child_run.run_id,
                    "resource_download_completed",
                    {
                        **summary_payload,
                        "parent_run_id": active_run.run_id,
                    },
                )
                store.update_status(child_run.run_id, RunStatus.SUCCEEDED)
                store.set_run_metadata(
                    child_run.run_id,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    workflow_kind="interactive_chat_bisque_resource",
                    mode="interactive",
                    parent_run_id=active_run.run_id,
                    planner_version="workgraph_v1",
                    agent_role="bisque_import",
                    checkpoint_state={
                        "phase": "completed",
                        "source_uri": resource_uri,
                        "dataset_uri": dataset_uri,
                        "file_id": summary_payload["file_id"],
                        "status": summary_payload["status"],
                    },
                    trace_group_id=str(active_run.trace_group_id or "").strip() or conversation_id,
                )
                import_summaries.append(summary_payload)
            except Exception as exc:
                error_payload = {
                    "resource_uri": resource_uri,
                    "dataset_uri": dataset_uri,
                    "source": item.get("source"),
                    "child_run_id": child_run.run_id,
                    "error": str(exc),
                }
                store.append_event(
                    active_run.run_id,
                    "resource_download_failed",
                    error_payload,
                    level="error",
                )
                store.append_event(
                    child_run.run_id,
                    "resource_download_failed",
                    {
                        **error_payload,
                        "parent_run_id": active_run.run_id,
                    },
                    level="error",
                )
                store.update_status(child_run.run_id, RunStatus.FAILED, error=str(exc))
                store.set_run_metadata(
                    child_run.run_id,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    workflow_kind="interactive_chat_bisque_resource",
                    mode="interactive",
                    parent_run_id=active_run.run_id,
                    planner_version="workgraph_v1",
                    agent_role="bisque_import",
                    checkpoint_state={
                        "phase": "failed",
                        "source_uri": resource_uri,
                        "dataset_uri": dataset_uri,
                        "error": str(exc),
                    },
                    trace_group_id=str(active_run.trace_group_id or "").strip() or conversation_id,
                )
                import_summaries.append(
                    {
                        "resource_uri": resource_uri,
                        "dataset_uri": dataset_uri,
                        "status": "error",
                        "child_run_id": child_run.run_id,
                        "error": str(exc),
                    }
                )

        if fanout_inputs:
            store.append_event(
                active_run.run_id,
                "dataset_fanout_completed",
                {
                    "dataset_count": len(dataset_summaries),
                    "resource_count": len(fanout_inputs),
                    "imported_count": sum(
                        1
                        for item in import_summaries
                        if str(item.get("status") or "").strip().lower() == "imported"
                    ),
                    "reused_count": sum(
                        1
                        for item in import_summaries
                        if str(item.get("status") or "").strip().lower() == "reused"
                    ),
                    "failed_count": sum(
                        1
                        for item in import_summaries
                        if str(item.get("status") or "").strip().lower() == "error"
                    ),
                },
            )

        successful_rows = list(imported_rows)
        if fanout_inputs and not successful_rows:
            failure_samples = [
                {
                    "resource_uri": item.get("resource_uri"),
                    "dataset_uri": item.get("dataset_uri"),
                    "error": item.get("error"),
                }
                for item in import_summaries[:10]
            ]
            raise HTTPException(
                status_code=502,
                detail={
                    "message": "BisQue imports failed before chat orchestration could start.",
                    "failures": failure_samples,
                },
            )

        return successful_rows, dataset_summaries, import_summaries

    @v1.post("/uploads/from-bisque", response_model=BisqueImportResponse)
    @legacy.post("/uploads/from-bisque", response_model=BisqueImportResponse)
    async def upload_files_from_bisque(
        req: BisqueImportRequest,
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> BisqueImportResponse:
        del _auth
        bisque_auth = _require_authenticated_bisque_auth(
            bisque_auth,
            detail="BisQue authentication is required before importing BisQue resources.",
        )
        user_id = _current_user_id(bisque_auth)
        bisque_root = str(getattr(settings, "bisque_root", "") or "").strip().rstrip("/")
        bisque_username, bisque_password, bisque_access_token, bisque_cookie_header = (
            _require_bisque_user_execution_access(bisque_auth)
        )
        resources = [str(item or "").strip() for item in req.resources if str(item or "").strip()]
        if not resources:
            raise HTTPException(status_code=400, detail="No BisQue resources provided.")
        if len(resources) > upload_max_files_per_request:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Too many resources in one request ({len(resources)}). "
                    f"Max allowed is {upload_max_files_per_request}."
                ),
            )

        uploaded_records: list[UploadedFileRecord] = []
        import_results: list[BisqueImportItem] = []

        for input_url in list(dict.fromkeys(resources)):
            try:
                import_result = await _import_bisque_resource_for_user(
                    input_url=input_url,
                    user_id=user_id,
                    bisque_root=bisque_root,
                    bisque_username=bisque_username,
                    bisque_password=bisque_password,
                    bisque_access_token=bisque_access_token,
                    bisque_cookie_header=bisque_cookie_header,
                )
                uploaded_record = import_result.get("uploaded_record")
                if isinstance(uploaded_record, UploadedFileRecord):
                    uploaded_records.append(uploaded_record)
                import_results.append(
                    BisqueImportItem(
                        input_url=input_url,
                        resource_uri=str(import_result.get("resource_uri") or "").strip() or None,
                        resource_uniq=str(import_result.get("resource_uniq") or "").strip() or None,
                        client_view_url=(
                            str(import_result.get("client_view_url") or "").strip() or None
                        ),
                        image_service_url=(
                            str(import_result.get("image_service_url") or "").strip() or None
                        ),
                        status=str(import_result.get("status") or "imported"),
                        download_source=(
                            str(import_result.get("download_source") or "").strip() or None
                        ),
                        uploaded=uploaded_record
                        if isinstance(uploaded_record, UploadedFileRecord)
                        else None,
                    )
                )
            except Exception as exc:
                links = _build_bisque_links(input_url, bisque_root)
                import_results.append(
                    BisqueImportItem(
                        input_url=input_url,
                        resource_uri=links.get("resource_uri"),
                        resource_uniq=links.get("resource_uniq"),
                        client_view_url=links.get("client_view_url"),
                        image_service_url=links.get("image_service_url"),
                        status="error",
                        error=str(exc),
                    )
                )

        return BisqueImportResponse(
            file_count=len(uploaded_records),
            uploaded=uploaded_records,
            imports=import_results,
        )

    @v1.get("/resources", response_model=ResourceListResponse)
    @legacy.get("/resources", response_model=ResourceListResponse)
    def list_resources(
        limit: int = Query(default=200, ge=1, le=1000),
        offset: int = Query(default=0, ge=0),
        q: str | None = Query(default=None),
        kind: str | None = Query(default=None),
        source: str | None = Query(default=None),
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> ResourceListResponse:
        del _auth
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        normalized_kind = str(kind or "").strip().lower() or None
        normalized_source = str(source or "").strip().lower() or None
        allowed_kinds = {"image", "video", "table", "file"}
        allowed_sources = {"upload", "bisque_import"}
        if normalized_kind and normalized_kind not in allowed_kinds:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid resource kind '{normalized_kind}'. Allowed: {sorted(allowed_kinds)}",
            )
        if normalized_source and normalized_source not in allowed_sources:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid source '{normalized_source}'. Allowed: {sorted(allowed_sources)}",
            )
        rows = store.list_uploads(
            user_id=user_id,
            limit=limit,
            offset=offset,
            query=q,
            kind=normalized_kind,
            source_type=normalized_source,
            include_deleted=False,
        )
        records = [_resource_record_from_upload(row) for row in rows]
        return ResourceListResponse(count=len(records), resources=records)

    @v1.post("/resources/reuse-suggestions", response_model=ResourceComputationLookupResponse)
    @legacy.post("/resources/reuse-suggestions", response_model=ResourceComputationLookupResponse)
    def resource_reuse_suggestions(
        req: ResourceComputationLookupRequest,
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> ResourceComputationLookupResponse:
        del _auth
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        requested_file_ids = [
            str(file_id or "").strip() for file_id in req.file_ids if str(file_id or "").strip()
        ]
        requested_file_ids = list(dict.fromkeys(requested_file_ids))
        if not requested_file_ids:
            return ResourceComputationLookupResponse(count=0, suggestions=[])

        requested_tools = _extract_reuse_tool_names(
            prompt=req.prompt,
            tool_names=req.tool_names,
        )
        if not requested_tools:
            return ResourceComputationLookupResponse(count=0, suggestions=[])

        requested_files: list[dict[str, Any]] = []
        for file_id in requested_file_ids:
            upload_row = store.get_upload(file_id, user_id=user_id)
            if not upload_row:
                continue
            requested_files.append(upload_row)
        if not requested_files:
            return ResourceComputationLookupResponse(count=0, suggestions=[])

        file_sha256s = [
            str(item.get("sha256") or "").strip().lower()
            for item in requested_files
            if str(item.get("sha256") or "").strip()
        ]
        file_name_norms = [
            " ".join(str(item.get("original_name") or "").strip().lower().split())
            for item in requested_files
            if str(item.get("original_name") or "").strip()
        ]
        candidate_limit = max(
            50,
            len(requested_files) * len(requested_tools) * max(5, int(req.limit_per_file_tool) * 3),
        )
        candidate_rows = store.list_resource_computations(
            user_id=user_id,
            tool_names=requested_tools,
            file_sha256s=file_sha256s,
            file_name_norms=file_name_norms,
            limit=candidate_limit,
        )
        if not candidate_rows:
            return ResourceComputationLookupResponse(count=0, suggestions=[])

        successful_rows = [
            row
            for row in candidate_rows
            if str(row.get("run_status") or "").strip().lower() == RunStatus.SUCCEEDED.value
        ]
        if not successful_rows:
            return ResourceComputationLookupResponse(count=0, suggestions=[])

        conversation_cache: dict[str, dict[str, Any] | None] = {}

        def _conversation_details(conversation_id: str | None) -> tuple[str | None, str | None]:
            conv_id = str(conversation_id or "").strip()
            if not conv_id:
                return None, None
            if conv_id not in conversation_cache:
                conversation_cache[conv_id] = store.get_conversation(
                    conversation_id=conv_id,
                    user_id=user_id,
                )
            payload = conversation_cache.get(conv_id)
            if not payload:
                return None, None
            title = str(payload.get("title") or "").strip() or None
            updated_at = str(payload.get("updated_at") or "").strip() or None
            return title, updated_at

        limit_per_file_tool = max(1, int(req.limit_per_file_tool))
        suggestions: list[ResourceComputationSuggestion] = []
        for file_row in requested_files:
            requested_file_id = str(file_row.get("file_id") or "").strip()
            requested_file_name = (
                str(file_row.get("original_name") or "").strip() or requested_file_id
            )
            requested_file_sha = str(file_row.get("sha256") or "").strip().lower()
            requested_file_name_norm = " ".join(requested_file_name.strip().lower().split())
            if not requested_file_id or not requested_file_name:
                continue
            for tool_name in requested_tools:
                selected: list[tuple[dict[str, Any], str]] = []
                seen_run_ids: set[str] = set()

                sha_matches = [
                    row
                    for row in successful_rows
                    if str(row.get("tool_name") or "").strip() == tool_name
                    and str(row.get("file_sha256") or "").strip().lower() == requested_file_sha
                ]
                for row in sha_matches:
                    run_id = str(row.get("run_id") or "").strip()
                    if not run_id or run_id in seen_run_ids:
                        continue
                    seen_run_ids.add(run_id)
                    selected.append((row, "sha256"))
                    if len(selected) >= limit_per_file_tool:
                        break

                if len(selected) < limit_per_file_tool and requested_file_name_norm:
                    name_matches = [
                        row
                        for row in successful_rows
                        if str(row.get("tool_name") or "").strip() == tool_name
                        and " ".join(str(row.get("file_name") or "").strip().lower().split())
                        == requested_file_name_norm
                    ]
                    for row in name_matches:
                        run_id = str(row.get("run_id") or "").strip()
                        if not run_id or run_id in seen_run_ids:
                            continue
                        seen_run_ids.add(run_id)
                        selected.append((row, "filename"))
                        if len(selected) >= limit_per_file_tool:
                            break

                for row, match_type in selected:
                    conversation_id = str(row.get("conversation_id") or "").strip() or None
                    conversation_title, conversation_updated_at = _conversation_details(
                        conversation_id
                    )
                    suggestions.append(
                        ResourceComputationSuggestion(
                            requested_file_id=requested_file_id,
                            requested_file_name=requested_file_name,
                            requested_file_sha256=requested_file_sha,
                            tool_name=tool_name,
                            run_id=str(row.get("run_id") or ""),
                            run_status=str(row.get("run_status") or ""),
                            run_goal=str(row.get("run_goal") or "").strip() or None,
                            run_updated_at=(
                                str(row.get("run_updated_at") or "").strip()
                                or str(row.get("updated_at") or "").strip()
                            ),
                            conversation_id=conversation_id,
                            conversation_title=conversation_title,
                            conversation_updated_at=conversation_updated_at,
                            match_type="sha256" if match_type == "sha256" else "filename",
                        )
                    )

        return ResourceComputationLookupResponse(count=len(suggestions), suggestions=suggestions)

    @v1.get("/resources/{file_id}/thumbnail")
    @legacy.get("/resources/{file_id}/thumbnail")
    async def resource_thumbnail(
        file_id: str,
        bisque_auth: dict[str, Any] = Depends(_require_authenticated_session),
    ) -> FileResponse:
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        upload_row = store.get_upload(file_id, user_id=user_id)
        if not upload_row:
            raise HTTPException(status_code=404, detail=f"Resource not found: {file_id}")

        resource_kind = str(upload_row.get("resource_kind") or "").strip().lower()
        if not resource_kind:
            resource_kind = _infer_resource_kind(
                original_name=str(upload_row.get("original_name") or ""),
                content_type=str(upload_row.get("content_type") or "").strip() or None,
                source_uri=str(upload_row.get("source_uri") or "").strip() or None,
            )
        if resource_kind != "image":
            raise HTTPException(
                status_code=404, detail="Thumbnail is only available for image resources."
            )

        thumbnail_path = await run_in_threadpool(
            _build_upload_thumbnail,
            upload_row=upload_row,
            user_id=user_id,
            bisque_auth=bisque_auth,
        )
        if thumbnail_path is None or not thumbnail_path.exists() or not thumbnail_path.is_file():
            raise HTTPException(
                status_code=404, detail="Thumbnail could not be generated for this resource."
            )

        return FileResponse(
            str(thumbnail_path),
            media_type="image/png",
            filename=f"{file_id}.png",
            headers={"Cache-Control": "public, max-age=600"},
        )

    @v1.delete("/resources/{file_id}")
    @legacy.delete("/resources/{file_id}")
    def delete_resource(
        file_id: str,
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> dict[str, Any]:
        del _auth
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        upload_row = store.get_upload(file_id, user_id=user_id, include_deleted=True)
        if not upload_row:
            raise HTTPException(status_code=404, detail=f"Resource not found: {file_id}")

        _delete_bisque_resource_for_upload(upload_row=upload_row, bisque_auth=bisque_auth)
        store.soft_delete_upload(file_id=file_id, user_id=user_id)
        cleanup_paths = {
            str(upload_row.get("stored_path") or "").strip(),
            str(upload_row.get("staging_path") or "").strip(),
            str(upload_row.get("cache_path") or "").strip(),
            str(upload_row.get("thumbnail_path") or "").strip(),
        }
        for path_value in cleanup_paths:
            safe_path = _safe_upload_local_path(path_value)
            if safe_path and safe_path.exists() and safe_path.is_file():
                safe_path.unlink(missing_ok=True)
        caption_cache.pop(file_id, None)
        raw_file_id = str(upload_row.get("file_id") or "").strip()
        canonical_file_id = str(upload_row.get("canonical_resource_uniq") or "").strip()
        if raw_file_id:
            caption_cache.pop(raw_file_id, None)
        if canonical_file_id:
            caption_cache.pop(canonical_file_id, None)
        _load_hdf5_viewer_manifest_cached.cache_clear()
        _load_hdf5_dataset_summary_cached.cache_clear()
        _load_hdf5_slice_png_cached.cache_clear()
        _load_hdf5_atlas_png_cached.cache_clear()
        _load_hdf5_scalar_volume_cached.cache_clear()
        _load_hdf5_histogram_cached.cache_clear()
        _load_hdf5_table_preview_cached.cache_clear()
        for path_value in cleanup_paths:
            if path_value:
                invalidate_file_derivatives(path_value)
                purge_scalar_volume_persistent_cache(path_value)
        return {"deleted": True, "file_id": file_id}

    def _resolve_upload_path(
        file_id: str,
        *,
        user_id: str | None = None,
        include_deleted: bool = False,
        bisque_auth: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], Path]:
        upload_record = store.get_upload(file_id, user_id=user_id, include_deleted=include_deleted)
        if not upload_record:
            raise HTTPException(status_code=404, detail=f"Upload not found: {file_id}")

        resolved_path = _resolved_local_path_for_upload(
            upload_record,
            user_id=user_id,
            bisque_auth=bisque_auth,
        )
        if not resolved_path:
            raise HTTPException(status_code=404, detail=f"Upload data missing on disk: {file_id}")
        stored_path = Path(resolved_path)
        return upload_record, stored_path

    def _metadata_from_image_payload(payload: dict[str, Any]) -> dict[str, Any]:
        metadata_payload = payload.get("metadata")
        metadata = metadata_payload if isinstance(metadata_payload, dict) else {}
        return {
            "reader": payload.get("reader") or "unknown",
            "dims_order": payload.get("dims_order") or "TCZYX",
            "array_shape": payload.get("array_shape") or [],
            "array_dtype": payload.get("array_dtype") or "unknown",
            "physical_spacing": payload.get("physical_spacing"),
            "scene": payload.get("scene"),
            "scene_count": len(payload.get("scenes") or []),
            "filename_hints": metadata.get("filename_hints") or {},
            "exif": metadata.get("exif") or {},
            "warnings": payload.get("warnings") or [],
        }

    def _fallback_metadata_caption(info: dict[str, Any]) -> str:
        if str(info.get("kind") or "").strip().lower() == "hdf5":
            hdf_payload = info.get("hdf5") if isinstance(info.get("hdf5"), dict) else {}
            summary = hdf_payload.get("summary") if isinstance(hdf_payload, dict) else {}
            dataset_count = int(summary.get("dataset_count") or 0)
            group_count = int(summary.get("group_count") or 0)
            geometry = summary.get("geometry") if isinstance(summary.get("geometry"), dict) else {}
            dimensions = geometry.get("dimensions") if isinstance(geometry, dict) else None
            dimension_text = ""
            if isinstance(dimensions, list) and dimensions:
                dimension_text = f" with geometry {'x'.join(str(value) for value in dimensions)}"
            return (
                f"Caption: HDF5 scientific container with {group_count} groups and "
                f"{dataset_count} datasets{dimension_text}, summarized from file structure metadata."
            )
        axis_sizes = info.get("axis_sizes") or {}
        z_size = int(axis_sizes.get("Z") or 1)
        c_size = int(axis_sizes.get("C") or 1)
        t_size = int(axis_sizes.get("T") or 1)
        y_size = int(axis_sizes.get("Y") or 0)
        x_size = int(axis_sizes.get("X") or 0)
        structure = "3D microscopy volume" if z_size > 1 else "2D microscopy image"
        channel_text = f"{c_size} channel{'s' if c_size != 1 else ''}"
        time_text = f"{t_size} timepoint{'s' if t_size != 1 else ''}"
        size_text = f"{x_size}x{y_size}" if x_size > 0 and y_size > 0 else "native resolution"
        depth_text = f", {z_size} z-slices" if z_size > 1 else ""
        return (
            f"Caption: {structure} ({size_text}{depth_text}) with {channel_text} and {time_text}, "
            "summarized from BioIO header metadata."
        )

    def _finalize_metadata_caption(text: str | None, info: dict[str, Any]) -> str:
        candidate = re.sub(r"\s+", " ", str(text or "").strip())
        if not candidate:
            candidate = _fallback_metadata_caption(info)
        if not candidate.lower().startswith("caption:"):
            candidate = f"Caption: {candidate}"
        return candidate

    def _llm_metadata_caption(info: dict[str, Any]) -> str:
        metadata = info.get("metadata") or {}
        try:
            client = get_openai_client()
            caption_client = client.with_options(
                timeout=min(4, int(settings.openai_timeout)),
                max_retries=0,
            )
            response = caption_client.chat.completions.create(
                model=settings.resolved_llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You write concise, professional scientific image captions for a research UI. "
                            "Return exactly one sentence prefixed with 'Caption:'; <= 24 words; plain text only."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Write a concise caption describing this uploaded image metadata for UI display.\n"
                            + json.dumps(info, default=str)
                            + "\nAdditional parser metadata:\n"
                            + json.dumps(metadata, default=str)
                        ),
                    },
                ],
                temperature=0.2,
                max_tokens=64,
                stream=False,
            )
            text = ""
            if response.choices:
                text = str(response.choices[0].message.content or "").strip()
            return _finalize_metadata_caption(text, info)
        except Exception:
            return _fallback_metadata_caption(info)

    def _png_stream_response(image_array: Any, *, file_name: str) -> StreamingResponse:
        png_bytes = _render_png_bytes(image_array)
        return _png_bytes_response(png_bytes, file_name=file_name)

    def _render_png_bytes(
        image_array: Any,
        *,
        already_u8: bool = False,
        max_dimension: int | None = upload_viewer_max_dimension,
    ) -> bytes:
        image_u8 = (
            np.asarray(image_array, dtype=np.uint8)
            if already_u8
            else _render_preview_image(image_array)
        )
        pil_image = Image.fromarray(image_u8)
        max_side = max(int(pil_image.width), int(pil_image.height))
        if max_dimension and max_side > int(max_dimension):
            scale = float(max_dimension) / float(max_side)
            target_width = max(1, round(pil_image.width * scale))
            target_height = max(1, round(pil_image.height * scale))
            resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
            pil_image = pil_image.resize((target_width, target_height), resample)
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        return buffer.getvalue()

    def _png_bytes_response(png_bytes: bytes, *, file_name: str) -> StreamingResponse:
        buffer = BytesIO(png_bytes)
        buffer.seek(0)
        headers = {
            "Cache-Control": "public, max-age=60",
            "Content-Disposition": f'inline; filename="{file_name}"',
        }
        return StreamingResponse(buffer, media_type="image/png", headers=headers)

    def _upload_file_path(path: Path) -> str:
        resolved = path.expanduser().resolve()
        return str(resolved)

    def _load_viewer_payload_cached(
        file_id: str,
        file_path: str,
    ) -> dict[str, Any]:
        del file_id
        payload = probe_scientific_image(
            file_path=str(file_path),
            array_mode="plane",
        )
        if not payload.get("success"):
            raise ValueError(str(payload.get("error") or "Failed to load upload for viewer."))
        payload.pop("_array", None)
        return payload

    def _get_viewer_payload(file_id: str, file_path: str) -> dict[str, Any]:
        del file_id
        return _load_viewer_payload_cached("", file_path)

    @lru_cache(maxsize=128)
    def _load_hdf5_viewer_manifest_cached(
        file_id: str,
        file_path: str,
        original_name: str,
        enabled: bool,
    ) -> dict[str, Any]:
        return build_hdf5_viewer_manifest(
            file_id=file_id,
            file_path=file_path,
            original_name=original_name,
            enabled=enabled,
        )

    def _get_hdf5_viewer_manifest(
        file_id: str,
        file_path: str,
        original_name: str,
    ) -> dict[str, Any]:
        with viewer_payload_cache_lock:
            payload = _load_hdf5_viewer_manifest_cached(
                file_id,
                file_path,
                original_name,
                viewer_hdf5_enabled,
            )
            return deepcopy(payload)

    @lru_cache(maxsize=512)
    def _load_hdf5_dataset_summary_cached(
        file_id: str,
        file_path: str,
        dataset_path: str,
    ) -> dict[str, Any]:
        payload = build_hdf5_dataset_summary(
            file_id=file_id,
            file_path=file_path,
            dataset_path=dataset_path,
        )
        payload["atlas_scheme"] = None
        if bool(payload.get("volume_eligible")):
            try:
                atlas_plan = _validate_hdf5_atlas_budget(payload)
            except ValueError as exc:
                payload["volume_eligible"] = False
                payload["volume_reason"] = str(exc)
                payload["capabilities"] = [
                    capability
                    for capability in list(payload.get("capabilities") or [])
                    if str(capability) != "volume"
                ]
            else:
                payload["atlas_scheme"] = atlas_plan.get("atlas_scheme")
        return payload

    def _get_hdf5_dataset_summary(
        file_id: str,
        file_path: str,
        dataset_path: str,
    ) -> dict[str, Any]:
        with viewer_payload_cache_lock:
            payload = _load_hdf5_dataset_summary_cached(
                file_id,
                file_path,
                dataset_path,
            )
            return deepcopy(payload)

    @lru_cache(maxsize=256)
    def _load_hdf5_slice_png_cached(
        file_id: str,
        file_path: str,
        dataset_path: str,
        axis: str,
        index: int | None,
        component: int | None,
    ) -> bytes:
        del file_id
        image_array = render_hdf5_dataset_slice(
            file_path=file_path,
            dataset_path=dataset_path,
            axis=axis,
            index=index,
            component=component,
        )
        return _render_png_bytes(image_array)

    def _get_hdf5_slice_png(
        file_id: str,
        file_path: str,
        dataset_path: str,
        axis: str,
        index: int | None,
        component: int | None,
    ) -> bytes:
        with viewer_hdf5_preview_cache_lock:
            return _load_hdf5_slice_png_cached(
                file_id, file_path, dataset_path, axis, index, component
            )

    def _format_texture_bytes(value: int) -> str:
        safe_value = max(0, int(value or 0))
        if safe_value >= 1024 * 1024:
            return f"{safe_value / (1024 * 1024):.1f} MiB"
        if safe_value >= 1024:
            return f"{safe_value / 1024:.1f} KiB"
        return f"{safe_value} B"

    def _validate_hdf5_atlas_budget(summary: dict[str, Any]) -> dict[str, Any]:
        if not bool(summary.get("volume_eligible")):
            raise ValueError(
                str(
                    summary.get("volume_reason")
                    or "Selected HDF5 dataset is not eligible for native 3D."
                )
            )
        axis_sizes = summary.get("axis_sizes")
        if not isinstance(axis_sizes, dict):
            raise ValueError(
                "Selected HDF5 dataset is missing normalized axis sizes for native 3D."
            )
        atlas_plan = plan_volume_source_atlas(
            axis_sizes=axis_sizes,
            atlas_max_dimension=upload_viewer_max_dimension,
        )
        voxel_count = int(atlas_plan.get("voxel_count") or 0)
        if voxel_count > viewer_hdf5_atlas_max_voxels:
            raise ValueError(
                "Selected HDF5 dataset exceeds the native 3D viewer voxel budget "
                f"({voxel_count:,} voxels > {viewer_hdf5_atlas_max_voxels:,}). "
                "Use the slice preview or downsample the dataset."
            )
        decoded_texture_bytes = int(atlas_plan.get("decoded_texture_bytes") or 0)
        if decoded_texture_bytes > viewer_hdf5_atlas_max_texture_bytes:
            raise ValueError(
                "Selected HDF5 dataset exceeds the native 3D atlas texture budget "
                f"({_format_texture_bytes(decoded_texture_bytes)} > "
                f"{_format_texture_bytes(viewer_hdf5_atlas_max_texture_bytes)}). "
                "Use the slice preview or reduce the viewer atlas size."
            )
        return atlas_plan

    @lru_cache(maxsize=128)
    def _load_hdf5_atlas_png_cached(
        file_id: str,
        file_path: str,
        dataset_path: str,
        enhancement: str,
        fusion_method: str,
        negative: bool,
        channels: tuple[int, ...],
    ) -> tuple[dict[str, Any], bytes]:
        summary = _load_hdf5_dataset_summary_cached(file_id, file_path, dataset_path)
        _validate_hdf5_atlas_budget(summary)
        payload, volume = load_view_volume_source(
            file_path=str(file_path),
            dataset_path=str(dataset_path),
            max_inline_elements=1024,
        )
        atlas_scheme, png_bytes = render_volume_source_atlas_png(
            payload=payload,
            volume=volume,
            enhancement=str(enhancement or "d"),
            fusion_method=str(fusion_method or "m"),
            negative=bool(negative),
            channel_indices=list(channels) if channels else None,
            atlas_max_dimension=upload_viewer_max_dimension,
        )
        return atlas_scheme, png_bytes

    def _get_hdf5_atlas_png(
        file_id: str,
        file_path: str,
        dataset_path: str,
        enhancement: str,
        fusion_method: str,
        negative: bool,
        channels: tuple[int, ...],
    ) -> tuple[dict[str, Any], bytes]:
        with viewer_hdf5_atlas_cache_lock:
            return _load_hdf5_atlas_png_cached(
                file_id,
                file_path,
                dataset_path,
                enhancement,
                fusion_method,
                negative,
                channels,
            )

    @lru_cache(maxsize=64)
    def _load_hdf5_scalar_volume_cached(
        file_id: str,
        file_path: str,
        dataset_path: str,
        channel: int | None,
    ) -> tuple[dict[str, Any], bytes]:
        del file_id
        return load_scalar_volume_texture(
            file_path=str(file_path),
            dataset_path=str(dataset_path),
            channel_index=channel,
            max_inline_elements=1024,
        )

    def _get_hdf5_scalar_volume(
        file_id: str,
        file_path: str,
        dataset_path: str,
        channel: int | None,
    ) -> tuple[dict[str, Any], bytes]:
        with viewer_hdf5_atlas_cache_lock:
            return _load_hdf5_scalar_volume_cached(
                file_id,
                file_path,
                dataset_path,
                channel,
            )

    @lru_cache(maxsize=256)
    def _load_hdf5_histogram_cached(
        file_id: str,
        file_path: str,
        dataset_path: str,
        component: int | None,
        bins: int,
    ) -> dict[str, Any]:
        return build_hdf5_dataset_histogram(
            file_id=file_id,
            file_path=file_path,
            dataset_path=dataset_path,
            component=component,
            bins=bins,
        )

    def _get_hdf5_histogram(
        file_id: str,
        file_path: str,
        dataset_path: str,
        component: int | None,
        bins: int,
    ) -> dict[str, Any]:
        with viewer_hdf5_preview_cache_lock:
            return deepcopy(
                _load_hdf5_histogram_cached(file_id, file_path, dataset_path, component, bins)
            )

    @lru_cache(maxsize=192)
    def _load_hdf5_table_preview_cached(
        file_id: str,
        file_path: str,
        dataset_path: str,
        offset: int,
        limit: int,
    ) -> dict[str, Any]:
        return build_hdf5_dataset_table_preview(
            file_id=file_id,
            file_path=file_path,
            dataset_path=dataset_path,
            offset=offset,
            limit=limit,
        )

    def _get_hdf5_table_preview(
        file_id: str,
        file_path: str,
        dataset_path: str,
        offset: int,
        limit: int,
    ) -> dict[str, Any]:
        with viewer_hdf5_preview_cache_lock:
            return deepcopy(
                _load_hdf5_table_preview_cached(file_id, file_path, dataset_path, offset, limit)
            )

    @lru_cache(maxsize=64)
    def _load_hdf5_materials_dashboard_cached(
        file_id: str,
        file_path: str,
    ) -> dict[str, Any]:
        return build_hdf5_materials_dashboard(
            file_id=file_id,
            file_path=file_path,
        )

    def _get_hdf5_materials_dashboard(
        file_id: str,
        file_path: str,
    ) -> dict[str, Any]:
        with viewer_hdf5_preview_cache_lock:
            return deepcopy(_load_hdf5_materials_dashboard_cached(file_id, file_path))

    def _load_slice_png_cached(
        file_id: str,
        file_path: str,
        axis: str,
        x: int | None,
        y: int | None,
        z: int | None,
        c: int | None,
        t: int | None,
        enhancement: str,
        fusion_method: str,
        negative: bool,
        channels: str | None,
        channel_colors: str | None,
        full_resolution: bool,
    ) -> bytes:
        del file_id
        del c
        channel_indices = _parse_viewer_channel_indices(channels)
        resolved_channel_colors = _parse_viewer_channel_colors(channel_colors)
        image_u8 = render_view_plane_image(
            file_path=str(file_path),
            axis=normalize_view_axis(axis),
            x_index=x,
            y_index=y,
            z_index=z,
            t_index=t,
            enhancement=enhancement,
            fusion_method=fusion_method,
            negative=negative,
            channel_indices=list(channel_indices),
            channel_colors=list(resolved_channel_colors),
            max_inline_elements=1024,
        )
        return _render_png_bytes(
            image_u8,
            already_u8=True,
            max_dimension=None if full_resolution else upload_viewer_max_dimension,
        )

    def _get_slice_png(
        file_id: str,
        file_path: str,
        axis: str,
        x: int | None,
        y: int | None,
        z: int | None,
        c: int | None,
        t: int | None,
        enhancement: str,
        fusion_method: str,
        negative: bool,
        channels: str | None,
        channel_colors: str | None,
        full_resolution: bool,
    ) -> bytes:
        return _load_slice_png_cached(
            file_id,
            file_path,
            axis,
            x,
            y,
            z,
            c,
            t,
            enhancement,
            fusion_method,
            negative,
            channels,
            channel_colors,
            full_resolution,
        )

    def _load_scalar_volume_cached(
        file_id: str,
        file_path: str,
        t: int | None,
        channel: int | None,
    ) -> tuple[dict[str, Any], bytes]:
        del file_id
        return load_scalar_volume_texture(
            file_path=str(file_path),
            t_index=t,
            channel_index=channel,
            max_inline_elements=1024,
        )

    def _get_scalar_volume(
        file_id: str,
        file_path: str,
        t: int | None,
        channel: int | None,
    ) -> tuple[dict[str, Any], bytes]:
        return _load_scalar_volume_cached(file_id, file_path, t, channel)

    def _load_tile_png_cached(
        file_id: str,
        file_path: str,
        axis: str,
        level: int,
        tile_x: int,
        tile_y: int,
        z: int | None,
        c: int | None,
        t: int | None,
    ) -> bytes:
        del file_id, c
        return render_view_tile_png(
            file_path=str(file_path),
            axis=normalize_view_axis(axis),
            level=int(level),
            tile_x=int(tile_x),
            tile_y=int(tile_y),
            z_index=z,
            t_index=t,
            tile_size=VIEWER_TILE_SIZE,
        )

    def _get_tile_png(
        file_id: str,
        file_path: str,
        axis: str,
        level: int,
        tile_x: int,
        tile_y: int,
        z: int | None,
        c: int | None,
        t: int | None,
    ) -> bytes:
        return _load_tile_png_cached(file_id, file_path, axis, level, tile_x, tile_y, z, c, t)

    def _parse_viewer_channel_indices(value: str | None) -> tuple[int, ...]:
        if not value:
            return ()
        output: list[int] = []
        for token in str(value).split(","):
            token = token.strip()
            if not token:
                continue
            try:
                channel = int(token)
            except Exception:
                continue
            if channel >= 0 and channel not in output:
                output.append(channel)
        return tuple(output)

    def _parse_viewer_channel_colors(value: str | None) -> tuple[str, ...]:
        if not value:
            return ()
        output: list[str] = []
        for token in str(value).split(","):
            safe = token.strip()
            if not safe:
                continue
            if safe.startswith("#"):
                safe = safe[1:]
            if len(safe) != 6 or any(ch not in "0123456789abcdefABCDEF" for ch in safe):
                continue
            output.append(f"#{safe.lower()}")
        return tuple(output)

    def _load_atlas_png_cached(
        file_id: str,
        file_path: str,
        enhancement: str,
        fusion_method: str,
        negative: bool,
        t: int | None,
        channels: tuple[int, ...],
        channel_colors: tuple[str, ...],
    ) -> tuple[dict[str, Any], bytes]:
        del file_id
        atlas_scheme, png_bytes = render_view_atlas_png(
            file_path=str(file_path),
            enhancement=str(enhancement or "d"),
            fusion_method=str(fusion_method or "m"),
            negative=bool(negative),
            t_index=t,
            channel_indices=list(channels) if channels else None,
            channel_colors=list(channel_colors) if channel_colors else None,
            atlas_max_dimension=upload_viewer_max_dimension,
        )
        return atlas_scheme, png_bytes

    def _get_atlas_png(
        file_id: str,
        file_path: str,
        enhancement: str,
        fusion_method: str,
        negative: bool,
        t: int | None,
        channels: tuple[int, ...],
        channel_colors: tuple[str, ...],
    ) -> tuple[dict[str, Any], bytes]:
        return _load_atlas_png_cached(
            file_id,
            file_path,
            enhancement,
            fusion_method,
            negative,
            t,
            channels,
            channel_colors,
        )

    def _load_histogram_cached(
        file_id: str,
        file_path: str,
        t: int | None,
        channels: tuple[int, ...],
        bins: int,
    ) -> dict[str, Any]:
        del file_id
        return render_view_histogram(
            file_path=str(file_path),
            t_index=t,
            channel_indices=list(channels) if channels else None,
            bins=bins,
        )

    def _get_histogram(
        file_id: str,
        file_path: str,
        t: int | None,
        channels: tuple[int, ...],
        bins: int,
    ) -> dict[str, Any]:
        return _load_histogram_cached(file_id, file_path, t, channels, bins)

    @v1.get("/uploads/{file_id}/viewer")
    @legacy.get("/uploads/{file_id}/viewer")
    async def upload_viewer_info(
        file_id: str,
        bisque_auth: dict[str, Any] = Depends(_require_authenticated_session),
    ) -> dict[str, Any]:
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        upload_record, stored_path = _resolve_upload_path(
            file_id,
            user_id=user_id,
            bisque_auth=bisque_auth,
        )
        cached_path = _upload_file_path(stored_path)
        original_name = upload_record.get("original_name") or stored_path.name
        if is_hdf5_viewer_path(stored_path):
            return await run_in_threadpool(
                _get_hdf5_viewer_manifest,
                str(file_id),
                cached_path,
                str(original_name),
            )
        try:
            payload = await run_in_threadpool(_get_viewer_payload, str(file_id), cached_path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return build_viewer_manifest(
            payload=payload,
            file_id=file_id,
            original_name=original_name,
            tile_size=VIEWER_TILE_SIZE,
            atlas_max_dimension=upload_viewer_max_dimension,
        )

    @v1.get("/uploads/{file_id}/hdf5/dataset")
    @legacy.get("/uploads/{file_id}/hdf5/dataset")
    async def upload_hdf5_dataset_summary(
        file_id: str,
        dataset_path: str = Query(..., min_length=1),
        bisque_auth: dict[str, Any] = Depends(_require_authenticated_session),
    ) -> dict[str, Any]:
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        _upload_record, stored_path = _resolve_upload_path(
            file_id,
            user_id=user_id,
            bisque_auth=bisque_auth,
        )
        if not is_hdf5_viewer_path(stored_path):
            raise HTTPException(
                status_code=400,
                detail="HDF5 dataset explorer is only available for HDF5 resources.",
            )
        cached_path = _upload_file_path(stored_path)
        try:
            return await run_in_threadpool(
                _get_hdf5_dataset_summary,
                str(file_id),
                cached_path,
                str(dataset_path),
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @v1.get("/uploads/{file_id}/hdf5/preview/slice")
    @legacy.get("/uploads/{file_id}/hdf5/preview/slice")
    def upload_hdf5_preview_slice(
        file_id: str,
        dataset_path: str = Query(..., min_length=1),
        axis: str = Query(default="z"),
        index: int | None = Query(default=None),
        component: int | None = Query(default=None),
        bisque_auth: dict[str, Any] = Depends(_require_authenticated_session),
    ) -> StreamingResponse:
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        upload_record, stored_path = _resolve_upload_path(
            file_id,
            user_id=user_id,
            bisque_auth=bisque_auth,
        )
        if not is_hdf5_viewer_path(stored_path):
            raise HTTPException(
                status_code=400,
                detail="HDF5 slice preview is only available for HDF5 resources.",
            )
        cached_path = _upload_file_path(stored_path)
        try:
            png_bytes = _get_hdf5_slice_png(
                str(file_id),
                cached_path,
                str(dataset_path),
                normalize_view_axis(axis),
                index,
                component,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _png_bytes_response(
            png_bytes,
            file_name=f"{upload_record.get('original_name') or stored_path.name}-hdf5-slice.png",
        )

    @v1.get("/uploads/{file_id}/hdf5/preview/atlas")
    @legacy.get("/uploads/{file_id}/hdf5/preview/atlas")
    def upload_hdf5_preview_atlas(
        file_id: str,
        dataset_path: str = Query(..., min_length=1),
        enhancement: str = Query(default="d"),
        fusion_method: str = Query(default="m"),
        negative: bool = Query(default=False),
        channels: str | None = Query(default=None),
        bisque_auth: dict[str, Any] = Depends(_require_authenticated_session),
    ) -> StreamingResponse:
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        upload_record, stored_path = _resolve_upload_path(
            file_id,
            user_id=user_id,
            bisque_auth=bisque_auth,
        )
        if not is_hdf5_viewer_path(stored_path):
            raise HTTPException(
                status_code=400,
                detail="HDF5 native 3D preview is only available for HDF5 resources.",
            )
        cached_path = _upload_file_path(stored_path)
        channel_indices = _parse_viewer_channel_indices(channels)
        try:
            _atlas_scheme, png_bytes = _get_hdf5_atlas_png(
                str(file_id),
                cached_path,
                str(dataset_path),
                enhancement,
                fusion_method,
                negative,
                channel_indices,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _png_bytes_response(
            png_bytes,
            file_name=f"{upload_record.get('original_name') or stored_path.name}-hdf5-atlas.png",
        )

    @v1.get("/uploads/{file_id}/hdf5/preview/scalar-volume")
    @legacy.get("/uploads/{file_id}/hdf5/preview/scalar-volume")
    def upload_hdf5_preview_scalar_volume(
        file_id: str,
        dataset_path: str = Query(..., min_length=1),
        channel: int | None = Query(default=None),
        bisque_auth: dict[str, Any] = Depends(_require_authenticated_session),
    ) -> FastAPIResponse:
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        _upload_record, stored_path = _resolve_upload_path(
            file_id,
            user_id=user_id,
            bisque_auth=bisque_auth,
        )
        if not is_hdf5_viewer_path(stored_path):
            raise HTTPException(
                status_code=400,
                detail="HDF5 scalar 3D preview is only available for HDF5 resources.",
            )
        cached_path = _upload_file_path(stored_path)
        try:
            metadata, volume_bytes = _get_hdf5_scalar_volume(
                str(file_id),
                cached_path,
                str(dataset_path),
                channel,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        headers = {
            "Cache-Control": "public, max-age=60",
            "X-Volume-Width": str(int(metadata["axis_sizes"]["X"])),
            "X-Volume-Height": str(int(metadata["axis_sizes"]["Y"])),
            "X-Volume-Depth": str(int(metadata["axis_sizes"]["Z"])),
            "X-Volume-Dtype": str(metadata["dtype"]),
            "X-Volume-Bytes-Per-Voxel": str(int(metadata["bytes_per_voxel"])),
            "X-Volume-Raw-Min": str(float(metadata["raw_min"])),
            "X-Volume-Raw-Max": str(float(metadata["raw_max"])),
        }
        if metadata.get("selected_channel") is not None:
            headers["X-Volume-Channel"] = str(int(metadata["selected_channel"]))
        return FastAPIResponse(
            content=volume_bytes, media_type="application/octet-stream", headers=headers
        )

    @v1.get("/uploads/{file_id}/hdf5/preview/histogram")
    @legacy.get("/uploads/{file_id}/hdf5/preview/histogram")
    async def upload_hdf5_preview_histogram(
        file_id: str,
        dataset_path: str = Query(..., min_length=1),
        component: int | None = Query(default=None),
        bins: int = Query(default=DEFAULT_HISTOGRAM_BINS),
        bisque_auth: dict[str, Any] = Depends(_require_authenticated_session),
    ) -> dict[str, Any]:
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        _upload_record, stored_path = _resolve_upload_path(
            file_id,
            user_id=user_id,
            bisque_auth=bisque_auth,
        )
        if not is_hdf5_viewer_path(stored_path):
            raise HTTPException(
                status_code=400,
                detail="HDF5 histogram preview is only available for HDF5 resources.",
            )
        cached_path = _upload_file_path(stored_path)
        try:
            return await run_in_threadpool(
                _get_hdf5_histogram,
                str(file_id),
                cached_path,
                str(dataset_path),
                component,
                max(8, int(bins or DEFAULT_HISTOGRAM_BINS)),
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @v1.get("/uploads/{file_id}/hdf5/materials/dashboard")
    @legacy.get("/uploads/{file_id}/hdf5/materials/dashboard")
    async def upload_hdf5_materials_dashboard(
        file_id: str,
        bisque_auth: dict[str, Any] = Depends(_require_authenticated_session),
    ) -> dict[str, Any]:
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        _upload_record, stored_path = _resolve_upload_path(
            file_id,
            user_id=user_id,
            bisque_auth=bisque_auth,
        )
        if not is_hdf5_viewer_path(stored_path):
            raise HTTPException(
                status_code=400,
                detail="Materials dashboard is only available for HDF5 resources.",
            )
        cached_path = _upload_file_path(stored_path)
        try:
            return await run_in_threadpool(
                _get_hdf5_materials_dashboard,
                str(file_id),
                cached_path,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @v1.get("/uploads/{file_id}/hdf5/preview/table")
    @legacy.get("/uploads/{file_id}/hdf5/preview/table")
    async def upload_hdf5_preview_table(
        file_id: str,
        dataset_path: str = Query(..., min_length=1),
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=32, ge=1, le=128),
        bisque_auth: dict[str, Any] = Depends(_require_authenticated_session),
    ) -> dict[str, Any]:
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        _upload_record, stored_path = _resolve_upload_path(
            file_id,
            user_id=user_id,
            bisque_auth=bisque_auth,
        )
        if not is_hdf5_viewer_path(stored_path):
            raise HTTPException(
                status_code=400,
                detail="HDF5 table preview is only available for HDF5 resources.",
            )
        cached_path = _upload_file_path(stored_path)
        try:
            return await run_in_threadpool(
                _get_hdf5_table_preview,
                str(file_id),
                cached_path,
                str(dataset_path),
                int(offset),
                int(limit),
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @v1.get("/uploads/{file_id}/caption")
    @legacy.get("/uploads/{file_id}/caption")
    async def upload_caption(
        file_id: str,
        bisque_auth: dict[str, Any] = Depends(_require_authenticated_session),
    ) -> dict[str, Any]:
        if file_id in caption_cache:
            return {"file_id": file_id, "caption": caption_cache[file_id], "source": "cache"}

        info = await upload_viewer_info(file_id=file_id, bisque_auth=bisque_auth)
        if str(info.get("kind") or "").strip().lower() == "hdf5":
            caption = _fallback_metadata_caption(info)
        else:
            caption = await run_in_threadpool(_llm_metadata_caption, info)
        caption_cache[file_id] = caption
        source = "llm" if caption != _fallback_metadata_caption(info) else "fallback"
        return {"file_id": file_id, "caption": caption, "source": source}

    @v1.get("/uploads/{file_id}/slice")
    @legacy.get("/uploads/{file_id}/slice")
    def upload_slice(
        file_id: str,
        axis: str = Query(default="z"),
        x: int | None = Query(default=None),
        y: int | None = Query(default=None),
        z: int | None = Query(default=None),
        c: int | None = Query(default=None),
        t: int | None = Query(default=None),
        enhancement: str = Query(default="d"),
        fusion_method: str = Query(default="m"),
        negative: bool = Query(default=False),
        channels: str | None = Query(default=None),
        channel_colors: str | None = Query(default=None),
        full_resolution: bool = Query(default=False),
        bisque_auth: dict[str, Any] = Depends(_require_authenticated_session),
    ) -> StreamingResponse:
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        upload_record, stored_path = _resolve_upload_path(
            file_id,
            user_id=user_id,
            bisque_auth=bisque_auth,
        )
        cached_path = _upload_file_path(stored_path)
        try:
            png_bytes = _get_slice_png(
                str(file_id),
                cached_path,
                axis,
                x,
                y,
                z,
                c,
                t,
                enhancement,
                fusion_method,
                negative,
                channels,
                channel_colors,
                full_resolution,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Slice preview unavailable for upload: {file_id}",
            ) from exc
        return _png_bytes_response(
            png_bytes,
            file_name=f"{upload_record.get('original_name') or stored_path.name}-slice.png",
        )

    @v1.get("/uploads/{file_id}/preview")
    @legacy.get("/uploads/{file_id}/preview")
    def upload_preview(
        file_id: str,
        bisque_auth: dict[str, Any] = Depends(_require_authenticated_session),
    ) -> StreamingResponse:
        return upload_slice(
            file_id=file_id,
            axis="z",
            x=None,
            y=None,
            z=None,
            c=None,
            t=None,
            enhancement="d",
            fusion_method="m",
            negative=False,
            channels=None,
            channel_colors=None,
            full_resolution=False,
            bisque_auth=bisque_auth,
        )

    @v1.get("/uploads/{file_id}/display")
    @legacy.get("/uploads/{file_id}/display")
    def upload_display_image(
        file_id: str,
        bisque_auth: dict[str, Any] = Depends(_require_authenticated_session),
    ) -> FileResponse:
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        upload_record, stored_path = _resolve_upload_path(
            file_id,
            user_id=user_id,
            bisque_auth=bisque_auth,
        )
        cached_path = _upload_file_path(stored_path)
        original_name = str(upload_record.get("original_name") or stored_path.name)
        if not is_ordinary_display_image_path(original_name):
            raise HTTPException(
                status_code=400,
                detail="Direct display is only available for ordinary 2D browser images.",
            )
        media_type = (
            str(upload_record.get("content_type") or "").strip()
            or mimetypes.guess_type(original_name)[0]
            or mimetypes.guess_type(cached_path.name)[0]
            or "application/octet-stream"
        )
        if not str(media_type).startswith("image/"):
            raise HTTPException(status_code=400, detail="Upload is not browser-displayable.")
        return FileResponse(
            path=str(cached_path),
            media_type=media_type,
            headers={"Cache-Control": "public, max-age=60"},
        )

    @v1.get("/uploads/{file_id}/scalar-volume")
    @legacy.get("/uploads/{file_id}/scalar-volume")
    def upload_scalar_volume(
        file_id: str,
        t: int | None = Query(default=None),
        channel: int | None = Query(default=None),
        bisque_auth: dict[str, Any] = Depends(_require_authenticated_session),
    ) -> FastAPIResponse:
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        _upload_record, stored_path = _resolve_upload_path(
            file_id,
            user_id=user_id,
            bisque_auth=bisque_auth,
        )
        cached_path = _upload_file_path(stored_path)
        try:
            metadata, volume_bytes = _get_scalar_volume(str(file_id), cached_path, t, channel)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Scalar volume unavailable for upload: {file_id}",
            ) from exc
        headers = {
            "Cache-Control": "public, max-age=60",
            "X-Volume-Width": str(int(metadata["axis_sizes"]["X"])),
            "X-Volume-Height": str(int(metadata["axis_sizes"]["Y"])),
            "X-Volume-Depth": str(int(metadata["axis_sizes"]["Z"])),
            "X-Volume-Dtype": str(metadata["dtype"]),
            "X-Volume-Bytes-Per-Voxel": str(int(metadata["bytes_per_voxel"])),
            "X-Volume-Raw-Min": str(float(metadata["raw_min"])),
            "X-Volume-Raw-Max": str(float(metadata["raw_max"])),
        }
        if metadata.get("selected_channel") is not None:
            headers["X-Volume-Channel"] = str(int(metadata["selected_channel"]))
        return FastAPIResponse(
            content=volume_bytes, media_type="application/octet-stream", headers=headers
        )

    @v1.get("/uploads/{file_id}/atlas")
    @legacy.get("/uploads/{file_id}/atlas")
    def upload_view_atlas(
        file_id: str,
        enhancement: str = Query(default="d"),
        fusion_method: str = Query(default="m"),
        negative: bool = Query(default=False),
        channels: str | None = Query(default=None),
        channel_colors: str | None = Query(default=None),
        t: int | None = Query(default=None),
        bisque_auth: dict[str, Any] = Depends(_require_authenticated_session),
    ) -> StreamingResponse:
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        upload_record, stored_path = _resolve_upload_path(
            file_id,
            user_id=user_id,
            bisque_auth=bisque_auth,
        )
        cached_path = _upload_file_path(stored_path)
        channel_indices = _parse_viewer_channel_indices(channels)
        resolved_channel_colors = _parse_viewer_channel_colors(channel_colors)
        try:
            _atlas_scheme, png_bytes = _get_atlas_png(
                str(file_id),
                cached_path,
                enhancement,
                fusion_method,
                negative,
                t,
                channel_indices,
                resolved_channel_colors,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Atlas preview unavailable for upload: {file_id}",
            ) from exc
        return _png_bytes_response(
            png_bytes,
            file_name=f"{upload_record.get('original_name') or stored_path.name}-atlas.png",
        )

    @v1.get("/uploads/{file_id}/histogram")
    @legacy.get("/uploads/{file_id}/histogram")
    def upload_view_histogram(
        file_id: str,
        channels: str | None = Query(default=None),
        t: int | None = Query(default=None),
        bins: int = Query(default=DEFAULT_HISTOGRAM_BINS),
        bisque_auth: dict[str, Any] = Depends(_require_authenticated_session),
    ) -> dict[str, Any]:
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        _upload_record, stored_path = _resolve_upload_path(
            file_id,
            user_id=user_id,
            bisque_auth=bisque_auth,
        )
        cached_path = _upload_file_path(stored_path)
        channel_indices = _parse_viewer_channel_indices(channels)
        safe_bins = max(8, min(256, int(bins or DEFAULT_HISTOGRAM_BINS)))
        try:
            histogram = _get_histogram(str(file_id), cached_path, t, channel_indices, safe_bins)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Histogram unavailable for upload: {file_id}",
            ) from exc
        return {
            "file_id": file_id,
            "bins": safe_bins,
            "histogram": histogram,
        }

    @v1.get("/uploads/{file_id}/tiles/{axis}/{level}/{tile_x}/{tile_y}")
    @legacy.get("/uploads/{file_id}/tiles/{axis}/{level}/{tile_x}/{tile_y}")
    def upload_view_tile(
        file_id: str,
        axis: str,
        level: int,
        tile_x: int,
        tile_y: int,
        z: int | None = Query(default=None),
        c: int | None = Query(default=None),
        t: int | None = Query(default=None),
        bisque_auth: dict[str, Any] = Depends(_require_authenticated_session),
    ) -> StreamingResponse:
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        upload_record, stored_path = _resolve_upload_path(
            file_id,
            user_id=user_id,
            bisque_auth=bisque_auth,
        )
        cached_path = _upload_file_path(stored_path)
        try:
            png_bytes = _get_tile_png(
                str(file_id),
                cached_path,
                axis,
                int(level),
                int(tile_x),
                int(tile_y),
                z,
                c,
                t,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Tile preview unavailable for upload: {file_id}",
            ) from exc
        return _png_bytes_response(
            png_bytes,
            file_name=(
                f"{upload_record.get('original_name') or stored_path.name}"
                f"-{normalize_view_axis(axis)}-tile-l{level}-{tile_x}-{tile_y}.png"
            ),
        )

    def _to_binary_prompt_label(value: Any) -> int:
        if isinstance(value, bool):
            return 1 if value else 0
        if isinstance(value, (int, float)):
            return 1 if float(value) > 0 else 0
        label = str(value or "").strip().lower()
        if label in {"0", "false", "negative", "exclude", "background", "bg", "no"}:
            return 0
        if label in {"1", "true", "positive", "include", "foreground", "fg", "yes"}:
            return 1
        return 1

    def _normalize_point_coordinates(
        x: float,
        y: float,
        *,
        width: int | None = None,
        height: int | None = None,
    ) -> dict[str, int] | None:
        try:
            point_x = round(float(x))
            point_y = round(float(y))
        except Exception:
            return None
        if width is not None and width > 0:
            point_x = max(0, min(point_x, int(width) - 1))
        if height is not None and height > 0:
            point_y = max(0, min(point_y, int(height) - 1))
        return {"x": int(point_x), "y": int(point_y)}

    def _normalize_box_coordinates(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        width: int | None = None,
        height: int | None = None,
    ) -> list[float] | None:
        left = float(round(min(x1, x2)))
        right = float(round(max(x1, x2)))
        top = float(round(min(y1, y2)))
        bottom = float(round(max(y1, y2)))
        if width is not None and width > 0:
            max_x = float(max(int(width) - 1, 0))
            left = min(max(left, 0.0), max_x)
            right = min(max(right, 0.0), max_x)
        if height is not None and height > 0:
            max_y = float(max(int(height) - 1, 0))
            top = min(max(top, 0.0), max_y)
            bottom = min(max(bottom, 0.0), max_y)
        if right - left < 1.0 or bottom - top < 1.0:
            return None
        return [left, top, right, bottom]

    def _resolve_annotation_canvas_size(file_path: str) -> tuple[int, int] | None:
        try:
            payload = load_scientific_image(
                file_path=str(file_path),
                array_mode="plane",
                generate_preview=False,
                save_array=False,
                include_array=False,
                return_array=False,
            )
        except Exception:
            return None
        if not bool(payload.get("success")):
            return None
        axis_sizes = (
            payload.get("axis_sizes") if isinstance(payload.get("axis_sizes"), dict) else {}
        )
        try:
            width = int(axis_sizes.get("X") or 0)
            height = int(axis_sizes.get("Y") or 0)
        except Exception:
            return None
        if width <= 0 or height <= 0:
            return None
        return int(width), int(height)

    def _adaptive_point_box_size(
        *,
        width: int | None,
        height: int | None,
        fallback: float,
    ) -> float:
        if width is None or height is None or width <= 0 or height <= 0:
            return max(2.0, float(fallback))
        shorter_edge = max(1.0, float(min(width, height)))
        return max(6.0, min(32.0, shorter_edge * 0.04))

    def _point_to_prompt_box(
        *,
        x: int,
        y: int,
        point_box_size: float,
        width: int | None,
        height: int | None,
    ) -> list[float] | None:
        half = max(1.0, float(point_box_size) / 2.0)
        return _normalize_box_coordinates(
            float(x) - half,
            float(y) - half,
            float(x) + half,
            float(y) + half,
            width=width,
            height=height,
        )

    def _prepare_interactive_annotation(
        *,
        annotation: dict[str, Any] | None,
        canvas_size: tuple[int, int] | None,
        point_box_size: float,
        tracker_prompt_mode: str,
    ) -> dict[str, Any]:
        canvas_width = int(canvas_size[0]) if canvas_size else None
        canvas_height = int(canvas_size[1]) if canvas_size else None
        resolved_points: list[dict[str, int]] = []
        resolved_boxes: list[dict[str, Any]] = []
        positive_points: list[list[float]] = []
        negative_points: list[list[float]] = []
        explicit_boxes: list[list[float]] = []
        explicit_box_labels: list[int] = []

        points = annotation.get("points") if isinstance(annotation, dict) else None
        if isinstance(points, list):
            for item in points:
                if not isinstance(item, dict):
                    continue
                try:
                    x = float(item.get("x"))
                    y = float(item.get("y"))
                except Exception:
                    continue
                label = _to_binary_prompt_label(item.get("label"))
                normalized = _normalize_point_coordinates(
                    x,
                    y,
                    width=canvas_width,
                    height=canvas_height,
                )
                if not normalized:
                    continue
                resolved_points.append(
                    {
                        "x": int(normalized["x"]),
                        "y": int(normalized["y"]),
                        "label": int(label),
                    }
                )
                point_coords = [float(normalized["x"]), float(normalized["y"])]
                if label > 0:
                    positive_points.append(point_coords)
                else:
                    negative_points.append(point_coords)

        raw_boxes = annotation.get("boxes") if isinstance(annotation, dict) else None
        if isinstance(raw_boxes, list):
            for item in raw_boxes:
                if not isinstance(item, dict):
                    continue
                try:
                    x1 = float(item.get("x1"))
                    y1 = float(item.get("y1"))
                    x2 = float(item.get("x2"))
                    y2 = float(item.get("y2"))
                except Exception:
                    continue
                label = _to_binary_prompt_label(item.get("label"))
                normalized = _normalize_box_coordinates(
                    x1,
                    y1,
                    x2,
                    y2,
                    width=canvas_width,
                    height=canvas_height,
                )
                if not normalized:
                    continue
                resolved_boxes.append(
                    {
                        "x1": int(normalized[0]),
                        "y1": int(normalized[1]),
                        "x2": int(normalized[2]),
                        "y2": int(normalized[3]),
                        "label": int(label),
                    }
                )
                explicit_boxes.append(normalized)
                explicit_box_labels.append(int(label))

        tracker_input_points: list[list[list[float]]] = []
        tracker_input_labels: list[list[int]] = []
        if positive_points:
            resolved_tracker_mode = (
                "per_positive_point_instance"
                if str(tracker_prompt_mode or "").strip().lower() == "per_positive_point_instance"
                else "single_object_refine"
            )
            if resolved_tracker_mode == "per_positive_point_instance":
                for point in positive_points:
                    tracker_input_points.append([point, *negative_points])
                    tracker_input_labels.append([1] + [0 for _ in negative_points])
            else:
                tracker_input_points.append([*positive_points, *negative_points])
                tracker_input_labels.append(
                    [1 for _ in positive_points] + [0 for _ in negative_points]
                )
        else:
            resolved_tracker_mode = (
                "per_positive_point_instance"
                if str(tracker_prompt_mode or "").strip().lower() == "per_positive_point_instance"
                else "single_object_refine"
            )

        adaptive_box_size = _adaptive_point_box_size(
            width=canvas_width,
            height=canvas_height,
            fallback=float(point_box_size),
        )
        point_prompt_boxes: list[list[float]] = []
        point_prompt_box_labels: list[int] = []
        for point in resolved_points:
            box = _point_to_prompt_box(
                x=int(point["x"]),
                y=int(point["y"]),
                point_box_size=adaptive_box_size,
                width=canvas_width,
                height=canvas_height,
            )
            if not box:
                continue
            point_prompt_boxes.append(box)
            point_prompt_box_labels.append(int(point["label"]))

        stats = {
            "point_count": len(resolved_points),
            "positive_point_count": len(positive_points),
            "negative_point_count": len(negative_points),
            "box_count": len(resolved_boxes),
            "prompt_count": int(len(resolved_points) + len(resolved_boxes)),
            "tracker_object_count": len(tracker_input_points),
            "tracker_prompt_mode": resolved_tracker_mode,
            "source_size_resolved": bool(canvas_size),
            "point_box_size_used": float(adaptive_box_size),
        }
        return {
            "source_size": (
                {"width": int(canvas_width), "height": int(canvas_height)}
                if canvas_width is not None and canvas_height is not None
                else None
            ),
            "resolved_points": resolved_points,
            "resolved_boxes": resolved_boxes,
            "tracker_input_points": tracker_input_points or None,
            "tracker_input_labels": tracker_input_labels or None,
            "explicit_boxes": explicit_boxes,
            "explicit_box_labels": explicit_box_labels,
            "point_prompt_boxes": point_prompt_boxes,
            "point_prompt_box_labels": point_prompt_box_labels,
            "stats": stats,
        }

    def _persist_sam3_annotation_metadata(
        *,
        file_id: str,
        user_id: str,
        run_id: str,
        annotation: dict[str, Any],
    ) -> None:
        row = store.get_upload(file_id, user_id=user_id, include_deleted=False)
        if not row:
            return
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        annotation_history = metadata.get("sam3_annotations")
        if not isinstance(annotation_history, list):
            annotation_history = []
        entry = {
            "run_id": run_id,
            "saved_at": datetime.utcnow().isoformat() + "Z",
            "annotation": annotation,
        }
        annotation_history.append(entry)
        metadata["sam3_annotations"] = annotation_history[-100:]

        _persist_upload_row(row, metadata=metadata)

    def _resolve_interactive_segmentation_model(
        value: str | None,
    ) -> tuple[str, str, str, str | None]:
        raw_value = str(value or "medsam").strip()
        normalized = raw_value.lower().replace("-", "").replace("_", "")
        if normalized in {"medsam", "medsam2"}:
            return ("medsam", "MedSAM2", "sam2_prompt_image", None)
        if normalized in {"sam", "sam3"}:
            return ("sam3", "SAM3", "segment_image_sam3", None)
        return (
            "medsam",
            "MedSAM2",
            "sam2_prompt_image",
            f"Unknown interactive segmentation model '{raw_value}', using MedSAM2.",
        )

    def _summarize_sam3_interactive_result(
        result: dict[str, Any],
        *,
        model_label: str = "SAM3",
    ) -> str:
        processed = int(result.get("processed") or 0)
        total_files = int(result.get("total_files") or 0)
        total_masks = int(result.get("total_masks_generated") or 0)
        parts = [
            f"{model_label} interactive segmentation processed {processed}/{total_files} file(s) "
            f"and generated {total_masks} reported mask(s)."
        ]
        coverage_mean = result.get("coverage_percent_mean")
        if isinstance(coverage_mean, (int, float)):
            parts.append(
                "Mean image-level coverage (union mask area divided by image area) is "
                f"{float(coverage_mean):.2f}%."
            )

        instance_cov_mean = result.get("instance_coverage_percent_mean")
        instance_cov_min = result.get("instance_coverage_percent_min")
        instance_cov_max = result.get("instance_coverage_percent_max")
        instance_measured = int(result.get("instance_count_measured_total") or 0)
        instance_scope_raw = str(result.get("instance_count_scope") or "").strip()
        instance_scope = (
            instance_scope_raw.replace("_", " ") if instance_scope_raw else "per-instance masks"
        )
        if isinstance(instance_cov_mean, (int, float)):
            range_text = ""
            if isinstance(instance_cov_min, (int, float)) and isinstance(
                instance_cov_max, (int, float)
            ):
                range_text = (
                    f" (min={float(instance_cov_min):.2f}%, max={float(instance_cov_max):.2f}%)"
                )
            parts.append(
                f"Per-instance coverage using {instance_scope}: mean={float(instance_cov_mean):.2f}%"
                f"{range_text} across {instance_measured} measured instance(s)."
            )

        mismatch_files = int(result.get("instance_count_mismatch_files") or 0)
        if mismatch_files > 0:
            parts.append(
                "Reported mask count and measured per-instance count differed in "
                f"{mismatch_files} file(s)."
            )
        file_rows = (
            result.get("files_processed") if isinstance(result.get("files_processed"), list) else []
        )
        volume_rows = [
            row
            for row in file_rows
            if isinstance(row, dict) and int(row.get("slice_count") or 0) > 1
        ]
        if volume_rows:
            total_slice_count = sum(int(row.get("slice_count") or 0) for row in volume_rows)
            total_slices_processed = sum(
                int(row.get("slices_processed") or 0) for row in volume_rows
            )
            processed_all_slices = all(bool(row.get("processed_all_slices")) for row in volume_rows)
            if processed_all_slices and total_slice_count > 0:
                parts.append(
                    f"All {total_slice_count} slice(s) across {len(volume_rows)} volumetric input(s) "
                    f"were segmented explicitly with {model_label}."
                )
            elif total_slice_count > 0:
                parts.append(
                    f"{model_label} processed {total_slices_processed}/{total_slice_count} slice(s) "
                    f"directly across {len(volume_rows)} volumetric input(s)."
                )
        return " ".join(parts)

    @v1.post("/segment/sam3/interactive", response_model=Sam3InteractiveResponse)
    @legacy.post("/segment/sam3/interactive", response_model=Sam3InteractiveResponse)
    async def segment_sam3_interactive(
        req: Sam3InteractiveRequest,
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> Sam3InteractiveResponse:
        del _auth
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        requested_file_ids = [
            str(file_id or "").strip() for file_id in req.file_ids if str(file_id or "").strip()
        ]
        if not requested_file_ids:
            raise HTTPException(status_code=400, detail="file_ids is required.")

        _, resolved_records, missing_file_ids = _resolve_file_ids(
            requested_file_ids,
            user_id=user_id,
            bisque_auth=bisque_auth,
        )
        if missing_file_ids:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Unknown or unavailable file_ids.",
                    "missing_file_ids": missing_file_ids,
                },
            )

        selected_model_token, selected_model_label, selected_tool_name, model_warning = (
            _resolve_interactive_segmentation_model(req.model)
        )

        run = WorkflowRun.new(
            goal=f"{selected_model_label} interactive segmentation on {len(resolved_records)} file(s)",
            plan=None,
        )
        store.create_run(run)
        store.set_run_metadata(
            run.run_id,
            user_id=user_id,
            conversation_id=str(req.conversation_id or "").strip() or None,
        )
        store.update_status(run.run_id, RunStatus.RUNNING)

        started = time.monotonic()
        annotations_by_file_id: dict[str, dict[str, Any]] = {}
        for item in req.annotations:
            payload = item.model_dump()
            file_id = str(payload.get("file_id") or "").strip()
            if file_id:
                annotations_by_file_id[file_id] = payload

        combined_files_processed: list[dict[str, Any]] = []
        combined_visualization_paths: list[dict[str, Any]] = []
        combined_preferred_upload_paths: list[str] = []
        combined_warnings: list[str] = []
        if model_warning:
            combined_warnings.append(model_warning)
        output_directories: list[str] = []
        annotation_summaries: list[dict[str, Any]] = []
        coverage_values: list[float] = []
        total_masks_generated = 0
        processed_files = 0
        concept_prompt_text = str(req.concept_prompt or "").strip() or None

        for row in resolved_records:
            file_id = str(row.get("file_id") or "").strip()
            stored_path = (
                str(row.get("_resolved_local_path") or "").strip()
                or _resolved_local_path_for_upload(
                    row,
                    user_id=user_id,
                    bisque_auth=bisque_auth,
                )
                or ""
            )
            original_name = str(row.get("original_name") or file_id)
            annotation_payload = annotations_by_file_id.get(
                file_id, {"file_id": file_id, "points": [], "boxes": []}
            )
            canvas_size = _resolve_annotation_canvas_size(stored_path)
            prompt_bundle = _prepare_interactive_annotation(
                annotation=annotation_payload,
                canvas_size=canvas_size,
                point_box_size=float(req.point_box_size),
                tracker_prompt_mode=str(req.tracker_prompt_mode or "single_object_refine"),
            )
            prompt_stats = prompt_bundle["stats"]
            resolved_points = (
                prompt_bundle.get("resolved_points")
                if isinstance(prompt_bundle.get("resolved_points"), list)
                else []
            )
            resolved_boxes = (
                prompt_bundle.get("resolved_boxes")
                if isinstance(prompt_bundle.get("resolved_boxes"), list)
                else []
            )
            tracker_input_points = prompt_bundle.get("tracker_input_points")
            tracker_input_labels = prompt_bundle.get("tracker_input_labels")
            explicit_boxes = (
                list(prompt_bundle.get("explicit_boxes"))
                if isinstance(prompt_bundle.get("explicit_boxes"), list)
                else []
            )
            explicit_box_labels = (
                list(prompt_bundle.get("explicit_box_labels"))
                if isinstance(prompt_bundle.get("explicit_box_labels"), list)
                else []
            )
            point_prompt_boxes = (
                list(prompt_bundle.get("point_prompt_boxes"))
                if isinstance(prompt_bundle.get("point_prompt_boxes"), list)
                else []
            )
            point_prompt_box_labels = (
                list(prompt_bundle.get("point_prompt_box_labels"))
                if isinstance(prompt_bundle.get("point_prompt_box_labels"), list)
                else []
            )
            has_positive_point = int(prompt_stats.get("positive_point_count") or 0) > 0
            has_points = int(prompt_stats.get("point_count") or 0) > 0
            has_boxes = int(prompt_stats.get("box_count") or 0) > 0
            prompt_strategy = "missing_prompts"
            prompt_warnings: list[str] = []
            if not bool(prompt_stats.get("source_size_resolved")) and (has_points or has_boxes):
                prompt_warnings.append(
                    f"Source image size could not be resolved for {original_name}; prompts were used without backend clamping."
                )

            try:
                if selected_model_token == "medsam":
                    prompt_strategy = "medsam_points_boxes"
                    if concept_prompt_text:
                        prompt_warnings.append(
                            "Concept prompt was ignored because MedSAM2 uses explicit points and boxes only."
                        )
                    if any(int(label) <= 0 for label in explicit_box_labels):
                        prompt_warnings.append(
                            "MedSAM2 treats interactive boxes as positive ROIs; negative box labels were preserved in provenance only."
                        )
                    medsam_result = await run_in_threadpool(
                        sam2_prompt_image,
                        file_path=stored_path,
                        input_points=tracker_input_points,
                        input_labels=tracker_input_labels,
                        input_boxes=explicit_boxes or None,
                        save_visualization=bool(req.save_visualizations),
                        model_id=None,
                        device=None,
                        multimask_output=False,
                        # Interactive medical/scientific segmentation should preserve
                        # full 3D coverage for uploaded CT/MRI/NIfTI volumes.
                        max_slices=0,
                    )
                    reported_mask_count = int(
                        medsam_result.get("total_masks_generated")
                        or medsam_result.get("instance_count_reported")
                        or medsam_result.get("instance_count_measured")
                        or 0
                    )
                    measured_mask_count = int(
                        medsam_result.get("instance_count_measured")
                        or medsam_result.get("total_masks_generated")
                        or reported_mask_count
                        or 0
                    )
                    if reported_mask_count <= 0 and bool(medsam_result.get("success")):
                        reported_mask_count = max(
                            1,
                            int(prompt_stats.get("positive_point_count") or 0),
                            len(explicit_boxes),
                        )
                    if measured_mask_count <= 0 and bool(medsam_result.get("success")):
                        measured_mask_count = reported_mask_count
                    medsam_visualizations = (
                        medsam_result.get("visualization_paths")
                        if isinstance(medsam_result.get("visualization_paths"), list)
                        else []
                    )
                    if (
                        not medsam_visualizations
                        and str(medsam_result.get("visualization_path") or "").strip()
                    ):
                        medsam_visualizations = [
                            {
                                "path": str(medsam_result.get("visualization_path") or "").strip(),
                                "title": "MedSAM2 prompt overlay",
                                "kind": "overlay",
                            }
                        ]
                    preferred_upload_paths = [
                        value
                        for value in [
                            str(medsam_result.get("preferred_upload_path") or "").strip(),
                            str(medsam_result.get("output_path") or "").strip(),
                        ]
                        if value
                    ]
                    medsam_file_row: dict[str, Any] = {
                        "file": original_name,
                        "success": bool(medsam_result.get("success")),
                        "total_masks": reported_mask_count
                        if bool(medsam_result.get("success"))
                        else 0,
                        "instance_count_reported": reported_mask_count
                        if bool(medsam_result.get("success"))
                        else 0,
                        "instance_count_measured": measured_mask_count
                        if bool(medsam_result.get("success"))
                        else 0,
                        "instance_count_scope": (
                            str(medsam_result.get("instance_count_scope") or "").strip()
                            or "prompt_object_masks"
                        ),
                        "instance_coverage_percent_mean": medsam_result.get(
                            "instance_coverage_percent_mean"
                        ),
                        "instance_coverage_percent_min": medsam_result.get(
                            "instance_coverage_percent_min"
                        ),
                        "instance_coverage_percent_max": medsam_result.get(
                            "instance_coverage_percent_max"
                        ),
                        "instance_coverage_percent_values": medsam_result.get(
                            "instance_coverage_percent_values"
                        ),
                        "coverage_percent": medsam_result.get("coverage_percent"),
                        "model": "MedSAM2",
                        "backend": str(medsam_result.get("backend") or "").strip() or None,
                        "resolved_model_ref": (
                            str(medsam_result.get("resolved_model_ref") or "").strip() or None
                        ),
                        "slice_count": int(medsam_result.get("slice_count") or 0) or None,
                        "slices_processed": int(medsam_result.get("slices_processed") or 0) or None,
                        "slice_stride": int(medsam_result.get("slice_stride") or 0) or None,
                        "processed_all_slices": bool(medsam_result.get("processed_all_slices")),
                        "prompt_strategy": prompt_strategy,
                    }
                    if medsam_result.get("error"):
                        medsam_file_row["error"] = str(medsam_result.get("error"))
                    tool_result = {
                        "success": bool(medsam_result.get("success")),
                        "processed": 1 if bool(medsam_result.get("success")) else 0,
                        "total_files": 1,
                        "total_masks_generated": reported_mask_count
                        if bool(medsam_result.get("success"))
                        else 0,
                        "files_processed": [medsam_file_row],
                        "preferred_upload_paths": preferred_upload_paths,
                        "visualization_paths": medsam_visualizations,
                        "output_directory": (
                            str(Path(preferred_upload_paths[0]).parent)
                            if preferred_upload_paths
                            else ""
                        ),
                        "warnings": (
                            medsam_result.get("warnings")
                            if isinstance(medsam_result.get("warnings"), list)
                            else []
                        )
                        + prompt_warnings,
                        "model": "MedSAM2",
                        "resolved_model_ref": (
                            str(medsam_result.get("resolved_model_ref") or "").strip() or None
                        ),
                    }
                else:
                    if has_positive_point and not has_boxes and not concept_prompt_text:
                        prompt_strategy = "sam3_tracker_points"
                        tool_result = await run_in_threadpool(
                            segment_image_sam3,
                            file_paths=[stored_path],
                            save_visualizations=bool(req.save_visualizations),
                            model_id=None,
                            device=None,
                            preset=req.preset,
                            concept_prompt=None,
                            input_points=tracker_input_points,
                            input_points_labels=tracker_input_labels,
                            threshold=req.threshold,
                            min_points=req.min_points,
                            max_points=req.max_points,
                            force_rerun=bool(req.force_rerun),
                        )
                    elif concept_prompt_text or has_points or has_boxes:
                        region_boxes = list(explicit_boxes)
                        region_box_labels = list(explicit_box_labels)
                        if has_points:
                            region_boxes.extend(point_prompt_boxes)
                            region_box_labels.extend(point_prompt_box_labels)
                        prompt_strategy = (
                            "sam3_concept_hybrid"
                            if concept_prompt_text and has_points
                            else "sam3_concept"
                            if concept_prompt_text
                            else "sam3_regions"
                        )
                        tool_result = await run_in_threadpool(
                            segment_image_sam3,
                            file_paths=[stored_path],
                            save_visualizations=bool(req.save_visualizations),
                            model_id=None,
                            device=None,
                            preset=req.preset,
                            concept_prompt=concept_prompt_text,
                            input_boxes=region_boxes or None,
                            input_boxes_labels=region_box_labels if region_box_labels else None,
                            threshold=req.threshold,
                            min_points=req.min_points,
                            max_points=req.max_points,
                            force_rerun=bool(req.force_rerun),
                        )
                    else:
                        tool_result = {
                            "success": False,
                            "error": (
                                "Interactive segmentation requires a concept prompt, a positive point, or a box."
                            ),
                            "files_processed": [],
                            "warnings": prompt_warnings,
                        }
                    if isinstance(tool_result.get("warnings"), list):
                        tool_result["warnings"] = [
                            *tool_result.get("warnings", []),
                            *prompt_warnings,
                        ]
            except Exception as exc:
                tool_result = {"success": False, "error": str(exc), "files_processed": []}

            file_rows = (
                tool_result.get("files_processed")
                if isinstance(tool_result.get("files_processed"), list)
                else []
            )
            if file_rows:
                for file_row in file_rows:
                    if not isinstance(file_row, dict):
                        continue
                    enriched_row = dict(file_row)
                    enriched_row["file_id"] = file_id
                    enriched_row.setdefault("file", original_name)
                    enriched_row["prompt_strategy"] = prompt_strategy
                    enriched_row["source_image_size"] = prompt_bundle.get("source_size")
                    combined_files_processed.append(enriched_row)
                    if bool(enriched_row.get("success")):
                        processed_files += 1
                        try:
                            coverage = enriched_row.get("coverage_percent")
                            if isinstance(coverage, (int, float)):
                                coverage_values.append(float(coverage))
                        except Exception:
                            pass
            else:
                fallback_row = {
                    "file_id": file_id,
                    "file": original_name,
                    "success": bool(tool_result.get("success")),
                    "prompt_strategy": prompt_strategy,
                    "source_image_size": prompt_bundle.get("source_size"),
                }
                if tool_result.get("error"):
                    fallback_row["error"] = str(tool_result.get("error"))
                combined_files_processed.append(fallback_row)
                if bool(tool_result.get("success")):
                    processed_files += 1

            total_masks_generated += int(tool_result.get("total_masks_generated") or 0)

            visualization_rows = (
                tool_result.get("visualization_paths")
                if isinstance(tool_result.get("visualization_paths"), list)
                else []
            )
            for visual in visualization_rows:
                if isinstance(visual, dict) and str(visual.get("path") or "").strip():
                    enriched_visual = dict(visual)
                    enriched_visual["file_id"] = file_id
                    combined_visualization_paths.append(enriched_visual)

            preferred_paths = (
                tool_result.get("preferred_upload_paths")
                if isinstance(tool_result.get("preferred_upload_paths"), list)
                else []
            )
            for preferred in preferred_paths:
                value = str(preferred or "").strip()
                if value:
                    combined_preferred_upload_paths.append(value)

            warnings = tool_result.get("warnings")
            if isinstance(warnings, list):
                combined_warnings.extend([str(item) for item in warnings if str(item).strip()])

            output_directory = str(tool_result.get("output_directory") or "").strip()
            if output_directory:
                output_directories.append(output_directory)

            annotation_record = {
                "file_id": file_id,
                "file_name": original_name,
                "points": (
                    annotation_payload.get("points")
                    if isinstance(annotation_payload.get("points"), list)
                    else []
                ),
                "boxes": (
                    annotation_payload.get("boxes")
                    if isinstance(annotation_payload.get("boxes"), list)
                    else []
                ),
                "point_box_size": float(req.point_box_size),
                "resolved_points": resolved_points,
                "resolved_boxes": resolved_boxes,
                "source_image_size": prompt_bundle.get("source_size"),
                "source_size_resolved": bool(prompt_stats.get("source_size_resolved")),
                "tracker_input_points": tracker_input_points,
                "tracker_input_labels": tracker_input_labels,
                "tracker_prompt_mode": prompt_stats.get("tracker_prompt_mode"),
                "point_prompt_boxes": point_prompt_boxes,
                "point_prompt_box_labels": point_prompt_box_labels,
                "prompt_strategy": prompt_strategy,
                "prompt_stats": prompt_stats,
                "concept_prompt": concept_prompt_text,
                "model": selected_model_label,
                "run_id": run.run_id,
            }
            annotation_summaries.append(annotation_record)
            _persist_sam3_annotation_metadata(
                file_id=file_id,
                user_id=user_id,
                run_id=run.run_id,
                annotation=annotation_record,
            )

        combined_preferred_upload_paths = list(dict.fromkeys(combined_preferred_upload_paths))
        combined_warnings = list(dict.fromkeys([item for item in combined_warnings if item]))
        output_directories = list(dict.fromkeys([item for item in output_directories if item]))
        prompt_strategies = sorted(
            {
                str(item.get("prompt_strategy") or "").strip()
                for item in annotation_summaries
                if str(item.get("prompt_strategy") or "").strip()
            }
        )

        coverage_mean = (
            round(float(sum(coverage_values) / len(coverage_values)), 6)
            if coverage_values
            else None
        )
        coverage_min = round(float(min(coverage_values)), 6) if coverage_values else None
        coverage_max = round(float(max(coverage_values)), 6) if coverage_values else None
        instance_count_reported_total = 0
        instance_count_measured_total = 0
        instance_count_mismatch_files = 0
        instance_area_weighted_sum = 0.0
        instance_coverage_weighted_sum = 0.0
        instance_area_min_global: float | None = None
        instance_area_max_global: float | None = None
        instance_coverage_min_global: float | None = None
        instance_coverage_max_global: float | None = None

        for row in combined_files_processed:
            if not isinstance(row, dict) or not bool(row.get("success")):
                continue

            try:
                reported_count = int(
                    row.get("instance_count_reported")
                    if row.get("instance_count_reported") is not None
                    else row.get("total_masks")
                    if row.get("total_masks") is not None
                    else 0
                )
            except Exception:
                reported_count = 0
            try:
                measured_count = int(row.get("instance_count_measured") or 0)
            except Exception:
                measured_count = 0

            instance_count_reported_total += max(0, int(reported_count))
            instance_count_measured_total += max(0, int(measured_count))
            if measured_count > 0 and reported_count != measured_count:
                instance_count_mismatch_files += 1

            area_mean = row.get("instance_area_voxels_mean")
            area_min = row.get("instance_area_voxels_min")
            area_max = row.get("instance_area_voxels_max")
            coverage_mean_row = row.get("instance_coverage_percent_mean")
            coverage_min_row = row.get("instance_coverage_percent_min")
            coverage_max_row = row.get("instance_coverage_percent_max")

            try:
                if measured_count > 0 and area_mean is not None:
                    instance_area_weighted_sum += float(area_mean) * float(measured_count)
            except Exception:
                pass
            try:
                if measured_count > 0 and coverage_mean_row is not None:
                    instance_coverage_weighted_sum += float(coverage_mean_row) * float(
                        measured_count
                    )
            except Exception:
                pass

            try:
                if area_min is not None:
                    area_min_value = float(area_min)
                    instance_area_min_global = (
                        area_min_value
                        if instance_area_min_global is None
                        else min(instance_area_min_global, area_min_value)
                    )
            except Exception:
                pass
            try:
                if area_max is not None:
                    area_max_value = float(area_max)
                    instance_area_max_global = (
                        area_max_value
                        if instance_area_max_global is None
                        else max(instance_area_max_global, area_max_value)
                    )
            except Exception:
                pass
            try:
                if coverage_min_row is not None:
                    coverage_min_value = float(coverage_min_row)
                    instance_coverage_min_global = (
                        coverage_min_value
                        if instance_coverage_min_global is None
                        else min(instance_coverage_min_global, coverage_min_value)
                    )
            except Exception:
                pass
            try:
                if coverage_max_row is not None:
                    coverage_max_value = float(coverage_max_row)
                    instance_coverage_max_global = (
                        coverage_max_value
                        if instance_coverage_max_global is None
                        else max(instance_coverage_max_global, coverage_max_value)
                    )
            except Exception:
                pass

        unique_instance_scopes = sorted(
            {
                str(row.get("instance_count_scope") or "").strip()
                for row in combined_files_processed
                if isinstance(row, dict)
                and bool(row.get("success"))
                and str(row.get("instance_count_scope") or "").strip()
            }
        )
        if not unique_instance_scopes:
            instance_count_scope = "unknown"
        elif len(unique_instance_scopes) == 1:
            instance_count_scope = unique_instance_scopes[0]
        else:
            instance_count_scope = "mixed"
        if instance_count_mismatch_files > 0:
            combined_warnings.append(
                "Reported mask count differed from measured per-instance count for at least one file."
            )
            combined_warnings = list(dict.fromkeys([item for item in combined_warnings if item]))

        summary = {
            "processed": int(processed_files),
            "total_files": len(resolved_records),
            "total_masks_generated": int(total_masks_generated),
            "coverage_percent_mean": coverage_mean,
            "coverage_percent_min": coverage_min,
            "coverage_percent_max": coverage_max,
            "instance_count_reported_total": int(instance_count_reported_total),
            "instance_count_measured_total": int(instance_count_measured_total),
            "instance_count_mismatch_files": int(instance_count_mismatch_files),
            "instance_count_scope": instance_count_scope,
            "instance_area_voxels_mean": (
                round(float(instance_area_weighted_sum) / float(instance_count_measured_total), 6)
                if instance_count_measured_total > 0
                else None
            ),
            "instance_area_voxels_min": (
                round(float(instance_area_min_global), 6)
                if instance_area_min_global is not None
                else None
            ),
            "instance_area_voxels_max": (
                round(float(instance_area_max_global), 6)
                if instance_area_max_global is not None
                else None
            ),
            "instance_coverage_percent_mean": (
                round(
                    float(instance_coverage_weighted_sum) / float(instance_count_measured_total), 6
                )
                if instance_count_measured_total > 0
                else None
            ),
            "instance_coverage_percent_min": (
                round(float(instance_coverage_min_global), 6)
                if instance_coverage_min_global is not None
                else None
            ),
            "instance_coverage_percent_max": (
                round(float(instance_coverage_max_global), 6)
                if instance_coverage_max_global is not None
                else None
            ),
            "files": combined_files_processed,
            "model": selected_model_label,
            "prompt_strategies": prompt_strategies,
            "min_points": req.min_points,
            "max_points": req.max_points,
        }
        progress_events = [
            {
                "event": "completed",
                "tool": selected_tool_name,
                "message": f"{selected_model_label} interactive segmentation complete.",
                "summary": summary,
                "artifacts": [
                    {
                        "path": str(item.get("path")),
                        "title": str(
                            item.get("title")
                            or item.get("kind")
                            or f"{selected_model_label} artifact"
                        ),
                    }
                    for item in combined_visualization_paths[:200]
                    if isinstance(item, dict) and str(item.get("path") or "").strip()
                ],
                "ts": datetime.utcnow().isoformat() + "Z",
            }
        ]

        input_snapshot_paths = [
            str(row.get("_resolved_local_path") or "").strip()
            for row in resolved_records
            if str(row.get("_resolved_local_path") or "").strip()
        ]
        _, upload_entries, _ = _snapshot_uploaded_files(run.run_id, input_snapshot_paths)
        tool_entries = _snapshot_progress_artifacts(run.run_id, progress_events)
        if upload_entries or tool_entries:
            _update_manifest_with_entries(run.run_id, upload_entries + tool_entries)

        response_result = {
            "processed": int(processed_files),
            "total_files": len(resolved_records),
            "total_masks_generated": int(total_masks_generated),
            "files_processed": combined_files_processed,
            "preferred_upload_paths": combined_preferred_upload_paths,
            "visualization_paths": combined_visualization_paths,
            "output_directories": output_directories,
            "coverage_percent_mean": coverage_mean,
            "coverage_percent_min": coverage_min,
            "coverage_percent_max": coverage_max,
            "instance_count_reported_total": summary["instance_count_reported_total"],
            "instance_count_measured_total": summary["instance_count_measured_total"],
            "instance_count_mismatch_files": summary["instance_count_mismatch_files"],
            "instance_count_scope": summary["instance_count_scope"],
            "instance_area_voxels_mean": summary["instance_area_voxels_mean"],
            "instance_area_voxels_min": summary["instance_area_voxels_min"],
            "instance_area_voxels_max": summary["instance_area_voxels_max"],
            "instance_coverage_percent_mean": summary["instance_coverage_percent_mean"],
            "instance_coverage_percent_min": summary["instance_coverage_percent_min"],
            "instance_coverage_percent_max": summary["instance_coverage_percent_max"],
            "model": selected_model_label,
            "prompt_strategies": prompt_strategies,
            "annotations": annotation_summaries,
            "run_id": run.run_id,
        }

        default_response_text = _summarize_sam3_interactive_result(
            response_result,
            model_label=selected_model_label,
        )
        response_text = default_response_text
        tool_insights_payload: list[dict[str, Any]] = []
        tool_synthesis_used = False
        synthesis_user_text = (
            f"User requested {selected_model_label} interactive segmentation with explicit annotations. "
            "Interpret segmentation outcomes, quantitative coverage, and caveats."
        )
        concept_prompt = str(req.concept_prompt or "").strip()
        if concept_prompt:
            synthesis_user_text += f"\nUser concept prompt: {concept_prompt}"
        try:
            synthesis_payload = await agno_runtime.synthesize_tool_endpoint_response(
                user_text=synthesis_user_text,
                domain_id="bio",
                tool_name=selected_tool_name,
                tool_result=response_result,
                default_response_text=default_response_text,
                conversation_id=str(req.conversation_id or "").strip() or None,
            )
            if isinstance(synthesis_payload, dict):
                candidate_text = str(synthesis_payload.get("response_text") or "").strip()
                if candidate_text:
                    response_text = candidate_text
                raw_insights = synthesis_payload.get("tool_insights")
                if isinstance(raw_insights, list):
                    tool_insights_payload = [
                        item for item in raw_insights if isinstance(item, dict)
                    ][:12]
                tool_synthesis_used = bool(synthesis_payload.get("synthesized"))
        except Exception as exc:
            logger.warning(
                "%s interactive synthesis fallback engaged: %s", selected_model_label, exc
            )

        response_result["tool_insights"] = tool_insights_payload
        contract_evidence = [
            {
                "source": str(item.get("tool") or "").strip() or selected_tool_name,
                "summary": str(item.get("headline") or "").strip() or None,
            }
            for item in tool_insights_payload
            if str(item.get("headline") or "").strip()
        ]
        contract_payload = {
            "result": response_text,
            "evidence": contract_evidence,
            "measurements": [
                {"name": "processed_files", "value": int(processed_files), "unit": "files"},
                {
                    "name": "total_masks_generated",
                    "value": int(total_masks_generated),
                    "unit": "masks",
                },
            ],
            "statistical_analysis": [],
            "confidence": {
                "level": "medium" if int(processed_files) > 0 else "low",
                "why": (
                    ["Masks were generated with user-provided prompts."]
                    if int(processed_files) > 0
                    else ["No successful segmentation result was produced."]
                ),
            },
            "qc_warnings": combined_warnings,
            "limitations": (
                [
                    "No masks were generated. Try adding positive points/boxes or increasing point density."
                ]
                if int(total_masks_generated) <= 0
                else []
            ),
            "next_steps": [
                {"action": "Review mask overlays and rerun with refined prompts if needed."},
                {"action": "Upload preferred mask artifacts to BisQue if results are acceptable."},
            ],
        }

        duration_seconds = max(0.0, time.monotonic() - started)
        done_payload = {
            "run_id": run.run_id,
            "model": llm_model_name,
            "response_text": response_text,
            "contract": contract_payload,
            "progress_events": progress_events,
            "duration_seconds": duration_seconds,
            "runtime_metadata": {
                "runtime": "agno_tool_endpoint",
                "tool_synthesis_used": bool(tool_synthesis_used),
                "tool_insights": tool_insights_payload,
            },
        }
        store.append_event(run.run_id, "chat_done_payload", {"response": done_payload})

        succeeded = int(processed_files) > 0
        store.update_status(
            run.run_id,
            RunStatus.SUCCEEDED if succeeded else RunStatus.FAILED,
            error=None if succeeded else "sam3 interactive segmentation failed",
        )
        latest_run = store.get_run(run.run_id)
        _index_run_resource_computations(
            run_id=run.run_id,
            user_id=user_id,
            conversation_id=str(req.conversation_id or "").strip() or None,
            run_goal=run.goal,
            run_status=(RunStatus.SUCCEEDED.value if succeeded else RunStatus.FAILED.value),
            run_created_at=(
                latest_run.created_at.isoformat()
                if latest_run is not None
                else run.created_at.isoformat()
            ),
            run_updated_at=(
                latest_run.updated_at.isoformat()
                if latest_run is not None
                else datetime.utcnow().isoformat()
            ),
            progress_events=progress_events,
            input_files=[
                {
                    "file_id": str(item.get("file_id") or "").strip() or None,
                    "file_name": str(item.get("original_name") or "").strip()
                    or Path(
                        str(item.get("_resolved_local_path") or item.get("stored_path") or "")
                    ).name,
                    "file_sha256": str(item.get("sha256") or "").strip().lower(),
                    "source_path": str(
                        item.get("_resolved_local_path") or item.get("stored_path") or ""
                    ).strip()
                    or None,
                }
                for item in resolved_records
                if str(item.get("sha256") or "").strip()
            ],
        )

        return Sam3InteractiveResponse(
            success=succeeded,
            run_id=run.run_id,
            response_text=response_text,
            progress_events=progress_events,
            result=response_result,
            warnings=combined_warnings,
        )

    @v1.get("/stats/tools", response_model=StatsToolsResponse)
    @legacy.get("/stats/tools", response_model=StatsToolsResponse)
    def stats_tools(
        _auth: None = Depends(_require_api_key),
    ) -> StatsToolsResponse:
        del _auth
        tools = list_curated_stat_tools()
        records = [
            StatsToolRecord(name=item["name"], description=item["description"]) for item in tools
        ]
        return StatsToolsResponse(tool_count=len(records), tools=records)

    @v1.post("/stats/run", response_model=StatsRunResponse)
    @legacy.post("/stats/run", response_model=StatsRunResponse)
    def stats_run(
        req: StatsRunRequest,
        _auth: None = Depends(_require_api_key),
    ) -> StatsRunResponse:
        del _auth
        result = run_stat_tool(req.tool_name, req.payload)
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error") or "stats tool failed")
        return StatsRunResponse(
            success=True,
            tool_name=req.tool_name,
            result=result.get("result"),
        )

    @v1.post("/data/load-image", response_model=ImageLoadResponse)
    @legacy.post("/data/load-image", response_model=ImageLoadResponse)
    def data_load_image(
        req: ImageLoadRequest,
        _auth: None = Depends(_require_api_key),
    ) -> ImageLoadResponse:
        del _auth
        payload = load_scientific_image(
            file_path=req.file_path,
            scene=req.scene,
            use_aicspylibczi=bool(req.use_aicspylibczi),
            array_mode=req.array_mode,
            t_index=req.t_index,
            c_index=req.c_index,
            z_index=req.z_index,
            save_array=bool(req.save_array),
            include_array=bool(req.include_array),
            max_inline_elements=int(req.max_inline_elements),
        )
        if not payload.get("success"):
            return ImageLoadResponse(
                success=False,
                result=None,
                error=str(payload.get("error") or "Failed to load image."),
            )
        payload.pop("_array", None)  # API surface stays JSON serializable.
        return ImageLoadResponse(success=True, result=payload, error=None)

    def _to_wire_payload(value: Any) -> Any:
        if hasattr(value, "model_dump"):
            try:
                return value.model_dump()
            except Exception:
                return value
        return value

    def _sse_event(event: str, payload: Any) -> str:
        wire = json.dumps(_to_wire_payload(payload), ensure_ascii=False, default=str)
        return f"event: {event}\ndata: {wire}\n\n"

    def _agent_runtime_event_record(payload: dict[str, Any]) -> dict[str, Any]:
        kind = str(payload.get("kind") or "").strip().lower()
        explicit_event_type = str(payload.get("event_type") or "").strip().lower()
        event_type = explicit_event_type or ("tool_event" if kind == "tool" else "graph_event")
        return {
            "ts": datetime.utcnow().isoformat(),
            "level": "info",
            "event_type": event_type,
            "payload": payload,
        }

    def _persist_agent_runtime_event(run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        record = _agent_runtime_event_record(payload)
        store.append_event(
            run_id,
            str(record.get("event_type") or "graph_event"),
            dict(record.get("payload") or {}),
            level=str(record.get("level") or "info"),
        )
        return record

    def _chat_selected_tool_names(req: ChatRequest) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()
        for raw in req.selected_tool_names or []:
            token = str(raw or "").strip()
            if not token or token in seen:
                continue
            seen.add(token)
            ordered.append(token)
        return ordered

    def _chat_workflow_hint_payload(req: ChatRequest) -> dict[str, Any] | None:
        if req.workflow_hint is None:
            return None
        payload = req.workflow_hint.model_dump(mode="json")
        workflow_id = str(payload.get("id") or "").strip()
        source = str(payload.get("source") or "").strip()
        if not workflow_id or not source:
            return None
        return {
            "id": workflow_id,
            "source": source,
        }

    def _chat_knowledge_context_payload(req: ChatRequest) -> dict[str, Any] | None:
        if req.knowledge_context is None:
            return None
        payload = req.knowledge_context.model_dump(mode="json")
        normalized: dict[str, Any] = {}
        for key in ("collaborator_id", "project_id"):
            value = str(payload.get(key) or "").strip()
            if value:
                normalized[key] = value
        raw_pack_ids = payload.get("pack_ids")
        if isinstance(raw_pack_ids, list):
            pack_ids = [str(item or "").strip() for item in raw_pack_ids if str(item or "").strip()]
            if pack_ids:
                normalized["pack_ids"] = pack_ids
        return normalized or None

    def _chat_selection_context_payload(req: ChatRequest) -> dict[str, Any] | None:
        if req.selection_context is None:
            return None
        payload = req.selection_context.model_dump(mode="json")
        normalized: dict[str, Any] = {}
        for key in (
            "context_id",
            "source",
            "originating_message_id",
            "originating_user_text",
            "suggested_domain",
        ):
            value = str(payload.get(key) or "").strip()
            if value:
                normalized[key] = value
        for key in ("focused_file_ids", "resource_uris", "dataset_uris", "suggested_tool_names"):
            raw = payload.get(key)
            if not isinstance(raw, list):
                continue
            items = [str(item or "").strip() for item in raw if str(item or "").strip()]
            if items:
                normalized[key] = items
        raw_handles = payload.get("artifact_handles")
        if isinstance(raw_handles, dict):
            normalized_handles: dict[str, list[str]] = {}
            for raw_key, raw_values in raw_handles.items():
                handle_key = str(raw_key or "").strip()
                if not handle_key:
                    continue
                values = raw_values if isinstance(raw_values, list) else [raw_values]
                cleaned = [str(item or "").strip() for item in values if str(item or "").strip()]
                if cleaned:
                    normalized_handles[handle_key] = list(dict.fromkeys(cleaned))
            if normalized_handles:
                normalized["artifact_handles"] = normalized_handles
        return normalized or None

    def _chat_debug_requested(req: ChatRequest) -> bool:
        return bool(req.debug)

    def _chat_memory_policy_payload(req: ChatRequest) -> dict[str, Any]:
        del req
        return agno_runtime.memory_service.normalize_policy(None).model_dump(mode="json")

    def _chat_knowledge_scope_payload(req: ChatRequest) -> dict[str, Any]:
        knowledge_context = _chat_knowledge_context_payload(req) or {}
        return agno_runtime.knowledge_hub.normalize_scope(
            {"project_id": knowledge_context.get("project_id")} if knowledge_context else None,
            default_project_id=str(knowledge_context.get("project_id") or "").strip() or None,
        ).model_dump(mode="json")

    def _chat_checkpoint_state_fragment(req: ChatRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        selected_tool_names = _chat_selected_tool_names(req)
        knowledge_context = _chat_knowledge_context_payload(req)
        workflow_hint = _chat_workflow_hint_payload(req)
        selection_context = _chat_selection_context_payload(req)
        if selected_tool_names:
            payload["selected_tool_names"] = selected_tool_names
        if knowledge_context is not None:
            payload["knowledge_context"] = knowledge_context
        if workflow_hint is not None:
            payload["workflow_hint"] = workflow_hint
        if selection_context is not None:
            payload["selection_context"] = selection_context
        return payload

    def _chat_hitl_decision(req: ChatRequest) -> str | None:
        latest_text = _latest_user_text([m.model_dump() for m in req.messages])
        text = str(latest_text or "").strip().lower()
        if not text:
            return None
        if re.match(r"^(approve|approved|yes|y|go ahead|continue|proceed)\b", text):
            return "approve"
        if re.match(r"^(reject|rejected|cancel|no|stop|deny)\b", text):
            return "reject"
        return None

    def _find_pending_hitl_chat_run(
        *,
        user_id: str | None,
        conversation_id: str | None,
    ) -> WorkflowRun | None:
        if not user_id or not conversation_id:
            return None
        recent_runs = store.list_runs_for_user(user_id=user_id, limit=200)
        for row in recent_runs:
            if str(row.get("conversation_id") or "").strip() != conversation_id:
                continue
            run_id = str(row.get("run_id") or "").strip()
            if not run_id:
                continue
            candidate = store.get_run(run_id)
            if candidate is None:
                continue
            checkpoint_state = (
                candidate.checkpoint_state if isinstance(candidate.checkpoint_state, dict) else {}
            )
            if str(checkpoint_state.get("phase") or "").strip().lower() != "pending_approval":
                continue
            pending_hitl = checkpoint_state.get("pending_hitl")
            if isinstance(pending_hitl, dict) and pending_hitl:
                return candidate
        return None

    def _resolve_chat_run_context(
        req: ChatRequest,
        *,
        user_id: str | None,
    ) -> tuple[WorkflowRun, dict[str, Any] | None]:
        conversation_id = str(req.conversation_id or "").strip() or None
        decision = _chat_hitl_decision(req)
        pending_run = _find_pending_hitl_chat_run(user_id=user_id, conversation_id=conversation_id)
        if pending_run is not None and decision is not None:
            checkpoint_state = (
                pending_run.checkpoint_state
                if isinstance(pending_run.checkpoint_state, dict)
                else {}
            )
            pending_hitl = (
                dict(checkpoint_state.get("pending_hitl") or {})
                if isinstance(checkpoint_state.get("pending_hitl"), dict)
                else {}
            )
            if pending_hitl:
                return pending_run, {
                    "decision": decision,
                    "pending_hitl": pending_hitl,
                }
        return _create_chat_run(req, user_id=user_id), None

    def _pending_hitl_tool_names(hitl_resume: dict[str, Any] | None) -> list[str]:
        if not isinstance(hitl_resume, dict):
            return []
        pending_hitl = (
            dict(hitl_resume.get("pending_hitl") or {})
            if isinstance(hitl_resume.get("pending_hitl"), dict)
            else {}
        )
        ordered: list[str] = []
        seen: set[str] = set()

        def _add(raw: Any) -> None:
            token = str(raw or "").strip()
            if not token or token in seen:
                return
            seen.add(token)
            ordered.append(token)

        interruptions = pending_hitl.get("interruptions")
        if isinstance(interruptions, list):
            for item in interruptions:
                if not isinstance(item, dict):
                    continue
                _add(item.get("tool_name"))

        selected_tool_names = pending_hitl.get("selected_tool_names")
        if isinstance(selected_tool_names, list):
            for item in selected_tool_names:
                _add(item)
        return ordered

    def _chunk_text_for_stream(text: str, *, chunk_size: int = 120) -> list[str]:
        body = str(text or "")
        if not body:
            return []
        return [body[index : index + chunk_size] for index in range(0, len(body), chunk_size)]

    def _chat_budget_state(req: ChatRequest, **extra: Any) -> dict[str, Any]:
        payload = {
            **req.budgets.model_dump(mode="json"),
            "reasoning_mode": str(req.reasoning_mode or "deep"),
        }
        selected_tool_names = _chat_selected_tool_names(req)
        knowledge_context = _chat_knowledge_context_payload(req)
        workflow_hint = _chat_workflow_hint_payload(req)
        selection_context = _chat_selection_context_payload(req)
        if selected_tool_names:
            payload["selected_tool_names"] = selected_tool_names
        if knowledge_context is not None:
            payload["knowledge_context"] = knowledge_context
        if workflow_hint is not None:
            payload["workflow_hint"] = workflow_hint
        if selection_context is not None:
            payload["selection_context"] = selection_context
        for key, value in extra.items():
            payload[key] = value
        return payload

    def _create_chat_run(req: ChatRequest, *, user_id: str | None = None) -> WorkflowRun:
        wire_messages = [m.model_dump() for m in req.messages]
        run_goal = req.goal or _latest_user_text(wire_messages)
        conversation_id = str(req.conversation_id or "").strip() or None
        run = WorkflowRun.new(
            goal=run_goal,
            plan=None,
            workflow_kind="interactive_chat",
            mode="interactive",
            planner_version="agno_v1",
            agent_role="triage",
            checkpoint_state=_chat_checkpoint_state_fragment(req) or None,
            budget_state=_chat_budget_state(req),
            trace_group_id=conversation_id,
        )
        store.create_run(run)
        store.set_run_metadata(
            run.run_id,
            user_id=user_id,
            conversation_id=conversation_id,
            workflow_kind="interactive_chat",
            mode="interactive",
            planner_version="agno_v1",
            agent_role="triage",
            checkpoint_state=_chat_checkpoint_state_fragment(req) or None,
            budget_state=_chat_budget_state(req),
            trace_group_id=conversation_id,
        )
        return run

    async def _run_chat_events(
        req: ChatRequest,
        *,
        user_id: str | None = None,
        run: WorkflowRun | None = None,
        bisque_auth: dict[str, Any] | None = None,
        hitl_resume: dict[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        if not req.messages:
            raise HTTPException(status_code=400, detail="messages cannot be empty")

        wire_messages = [m.model_dump() for m in req.messages]
        active_run = run or _create_chat_run(req, user_id=user_id)
        conversation_id = str(req.conversation_id or "").strip() or None
        store.update_status(active_run.run_id, RunStatus.RUNNING)
        run_artifact_dir = _ensure_run_artifact_dir(active_run.run_id)
        started = time.monotonic()
        response_chunks: list[str] = []

        _, resolved_id_records, missing_file_ids = _resolve_file_ids(
            req.file_ids,
            user_id=user_id,
            bisque_auth=bisque_auth,
        )
        if missing_file_ids:
            store.append_event(
                active_run.run_id,
                "chat_failed_invalid_file_ids",
                {"missing_file_ids": missing_file_ids[:200]},
                level="error",
            )
            store.update_status(active_run.run_id, RunStatus.FAILED, error="invalid file_ids")
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Unknown or unavailable file_ids.",
                    "missing_file_ids": missing_file_ids,
                },
            )

        selected_bisque_tools = _requires_bisque_auth_preflight(
            wire_messages,
            uploaded_files=req.uploaded_files,
            file_ids=req.file_ids,
        )
        selected_bisque_preflight_tools = list(selected_bisque_tools)
        for tool_name in _pending_hitl_tool_names(hitl_resume):
            if tool_name not in selected_bisque_preflight_tools:
                selected_bisque_preflight_tools.append(tool_name)
        if req.resource_uris or req.dataset_uris:
            selected_bisque_preflight_tools = []
        else:
            selected_bisque_preflight_tools = [
                tool_name
                for tool_name in selected_bisque_preflight_tools
                if tool_name != "upload_to_bisque"
            ]
        session_mode = str((bisque_auth or {}).get("mode") or "").strip().lower()
        if selected_bisque_preflight_tools and not _has_chat_bisque_execution_access(bisque_auth):
            configured_auth_mode = _bisque_auth_mode()
            missing_condition = "guest_session" if session_mode == "guest" else "missing_session"
            response_text = (
                "BisQue search, import, metadata, tagging, and annotation tools require a "
                "signed-in BisQue account with a live session. "
                f"Current session mode: {session_mode or 'none'}. "
                "Sign in with your BisQue account and retry."
            )
            progress_events = [
                {
                    "event": "error",
                    "tool": "bisque_auth",
                    "message": response_text,
                    "summary": {
                        "success": False,
                        "kind": "bisque_auth",
                        "error_code": "bisque_auth_required",
                        "configured_auth_mode": configured_auth_mode,
                        "session_mode": session_mode or None,
                        "missing_condition": missing_condition,
                        "selected_tools": selected_bisque_preflight_tools,
                    },
                    "artifacts": [],
                    "ts": datetime.utcnow().isoformat() + "Z",
                }
            ]
            duration = round(time.monotonic() - started, 3)
            store.append_event(
                active_run.run_id,
                "chat_preflight_failed",
                {
                    "reason": "bisque_auth_required",
                    "configured_auth_mode": configured_auth_mode,
                    "session_mode": session_mode or None,
                    "selected_tools": selected_bisque_preflight_tools,
                },
                level="warning",
            )
            store.append_event(
                active_run.run_id,
                "chat_completed",
                {
                    "duration_seconds": duration,
                    "tool_calls": 0,
                    "selected_domains": [],
                    "domain_output_count": 0,
                    "progress_events": len(progress_events),
                    "runtime": "preflight",
                },
            )
            store.update_status(active_run.run_id, RunStatus.SUCCEEDED)
            store.set_run_metadata(
                active_run.run_id,
                user_id=user_id,
                conversation_id=conversation_id,
                workflow_kind="interactive_chat",
                mode="interactive",
                planner_version="agno_v1",
                agent_role="triage",
                checkpoint_state={
                    **_chat_checkpoint_state_fragment(req),
                    "phase": "preflight_failed",
                    "reason": "bisque_auth_required",
                    "selected_tools": selected_bisque_preflight_tools,
                },
                budget_state=_chat_budget_state(req, tool_calls_used=0),
                trace_group_id=conversation_id,
            )
            response = ChatResponse(
                run_id=active_run.run_id,
                model=llm_model_name,
                response_text=response_text,
                duration_seconds=duration,
                progress_events=progress_events,
                metadata={
                    "runtime": "preflight",
                    "selected_tools": selected_bisque_preflight_tools,
                },
            )
            store.append_event(
                active_run.run_id,
                "chat_done_payload",
                {
                    "response": response.model_dump(mode="json"),
                    "progress_events": progress_events,
                    "selected_domains": [],
                    "domain_outputs": {},
                    "tool_calls": 0,
                    "runtime_metadata": {
                        "runtime": "preflight",
                        "selected_tools": selected_bisque_preflight_tools,
                    },
                },
            )
            yield {"event": "done", "data": response}
            return

        imported_bisque_rows: list[dict[str, Any]] = []
        dataset_import_summaries: list[dict[str, Any]] = []
        bisque_import_summaries: list[dict[str, Any]] = []
        if req.resource_uris or req.dataset_uris:
            (
                imported_bisque_rows,
                dataset_import_summaries,
                bisque_import_summaries,
            ) = await _prepare_bisque_chat_inputs(
                active_run=active_run,
                req=req,
                user_id=user_id,
                conversation_id=conversation_id,
                bisque_auth=bisque_auth,
            )

        source_meta: dict[str, dict[str, Any]] = {}
        all_sources: list[str] = []
        for row in resolved_id_records:
            source_path = (
                str(row.get("_resolved_local_path") or "").strip()
                or _resolved_local_path_for_upload(
                    row,
                    user_id=user_id,
                    bisque_auth=bisque_auth,
                )
                or ""
            )
            if not source_path:
                continue
            source_meta[source_path] = {
                "file_id": row.get("file_id"),
                "original_name": row.get("original_name"),
                "content_type": row.get("content_type"),
                "size_bytes": row.get("size_bytes"),
            }
            all_sources.append(source_path)
        for row in imported_bisque_rows:
            source_path = (
                str(row.get("_resolved_local_path") or "").strip()
                or _resolved_local_path_for_upload(
                    row,
                    user_id=user_id,
                    bisque_auth=bisque_auth,
                )
                or ""
            )
            if not source_path:
                continue
            source_meta[source_path] = {
                "file_id": row.get("file_id"),
                "original_name": row.get("original_name"),
                "content_type": row.get("content_type"),
                "size_bytes": row.get("size_bytes"),
                "source_type": row.get("source_type"),
                "source_uri": row.get("source_uri"),
                "client_view_url": row.get("client_view_url"),
                "resource_kind": row.get("resource_kind"),
                "metadata": row.get("metadata") if isinstance(row.get("metadata"), dict) else {},
            }
            all_sources.append(source_path)
        for raw in req.uploaded_files:
            source_path = str(raw or "").strip()
            if not source_path:
                continue
            all_sources.append(source_path)
            if source_path not in source_meta:
                try:
                    guessed_size = int(Path(source_path).expanduser().stat().st_size)
                except Exception:
                    guessed_size = 0
                source_meta[source_path] = {
                    "file_id": None,
                    "original_name": Path(source_path).name,
                    "content_type": mimetypes.guess_type(source_path)[0],
                    "size_bytes": guessed_size,
                }
        dedup_sources = list(dict.fromkeys(all_sources))

        runtime_uploaded_files, upload_entries, missing_uploaded = _snapshot_uploaded_files(
            active_run.run_id, dedup_sources
        )
        context_rows: list[dict[str, Any]] = []
        reuse_index_input_rows: list[dict[str, Any]] = []
        for entry in upload_entries:
            source_path = str(entry.get("source_path") or "").strip()
            rel_path = str(entry.get("path") or "").strip()
            effective_path = (
                str((run_artifact_dir / rel_path).resolve()) if rel_path else source_path
            )
            meta = source_meta.get(source_path, {})
            file_sha = str(entry.get("sha256") or "").strip().lower()
            original_name = str(meta.get("original_name") or Path(source_path).name)
            context_rows.append(
                {
                    "file_id": meta.get("file_id"),
                    "original_name": original_name,
                    "content_type": meta.get("content_type") or entry.get("mime_type"),
                    "size_bytes": int(meta.get("size_bytes") or entry.get("size_bytes") or 0),
                    "path": effective_path,
                    "source_type": meta.get("source_type"),
                    "source_uri": meta.get("source_uri"),
                    "client_view_url": meta.get("client_view_url"),
                    "resource_kind": meta.get("resource_kind"),
                    "metadata": meta.get("metadata")
                    if isinstance(meta.get("metadata"), dict)
                    else {},
                }
            )
            if file_sha:
                reuse_index_input_rows.append(
                    {
                        "file_id": str(meta.get("file_id") or "").strip() or None,
                        "file_name": original_name,
                        "file_sha256": file_sha,
                        "source_path": source_path,
                    }
                )
        system_context_messages: list[dict[str, str]] = []
        file_context = _build_file_context(context_rows)
        if file_context:
            system_context_messages.insert(0, {"role": "system", "content": file_context})
        history_context = _build_history_context(
            user_id=user_id,
            conversation_id=conversation_id,
            latest_user_text=_latest_user_text(wire_messages),
        )
        if history_context:
            system_context_messages.insert(0, {"role": "system", "content": history_context})
            store.append_event(
                active_run.run_id,
                "history_context_attached",
                {
                    "conversation_id": conversation_id,
                    "history_context_chars": len(history_context),
                },
            )
        followup_artifact_context, followup_artifact_paths = _build_followup_artifact_context(
            user_id=user_id,
            conversation_id=conversation_id,
            latest_user_text=_latest_user_text(wire_messages),
        )
        if followup_artifact_context:
            system_context_messages.insert(
                0, {"role": "system", "content": followup_artifact_context}
            )
            store.append_event(
                active_run.run_id,
                "followup_artifact_context_attached",
                {
                    "conversation_id": conversation_id,
                    "context_chars": len(followup_artifact_context),
                },
            )
        if followup_artifact_paths:
            for path in followup_artifact_paths:
                token = str(path or "").strip()
                if not token:
                    continue
                if token not in dedup_sources:
                    dedup_sources.append(token)
                if token not in runtime_uploaded_files:
                    runtime_uploaded_files.append(token)

        if upload_entries:
            _update_manifest_with_entries(active_run.run_id, upload_entries)
            store.append_event(
                active_run.run_id,
                "uploads_snapshot",
                {
                    "count": len(upload_entries),
                    "missing_count": len(missing_uploaded),
                    "file_id_count": len(resolved_id_records),
                    "bisque_import_count": len(imported_bisque_rows),
                    "uploads": [
                        {
                            "path": entry.get("path"),
                            "sha256": entry.get("sha256"),
                            "size_bytes": entry.get("size_bytes"),
                            "source_path": entry.get("source_path"),
                            "file_id": (
                                source_meta.get(str(entry.get("source_path") or "").strip(), {})
                            ).get("file_id"),
                            "original_name": (
                                source_meta.get(str(entry.get("source_path") or "").strip(), {})
                            ).get("original_name"),
                        }
                        for entry in upload_entries[:100]
                    ],
                },
            )
        elif req.uploaded_files:
            store.append_event(
                active_run.run_id,
                "uploads_snapshot",
                {
                    "count": 0,
                    "missing_count": len(missing_uploaded),
                    "file_id_count": len(resolved_id_records),
                    "missing": missing_uploaded[:100],
                },
                level="error",
            )

        conversation_messages: list[dict[str, Any]] = list(wire_messages)
        compaction_meta: dict[str, Any] = {
            "enabled": bool(context_compaction_enabled),
            "used": False,
            "reason": "not_evaluated",
            "original_message_count": len(conversation_messages),
            "compacted_message_count": len(conversation_messages),
            "token_estimate_before": (
                _estimate_messages_tokens(system_context_messages)
                + _estimate_messages_tokens(conversation_messages)
            ),
            "token_estimate_after": (
                _estimate_messages_tokens(system_context_messages)
                + _estimate_messages_tokens(conversation_messages)
            ),
            "summary_chars": 0,
            "summary_hash": "",
            "persisted_memory": False,
            "reused_summary": False,
        }
        try:
            conversation_messages, compaction_meta = _apply_conversation_compaction(
                user_id=user_id,
                conversation_id=conversation_id,
                messages=wire_messages,
                system_context_messages=system_context_messages,
            )
        except Exception as exc:
            compaction_meta = {
                **compaction_meta,
                "reason": "error",
                "error": str(exc),
            }
            store.append_event(
                active_run.run_id,
                "conversation_context_compaction_failed",
                {
                    "conversation_id": conversation_id,
                    "error": str(exc),
                },
                level="warning",
            )

        compaction_event_payload = {
            "conversation_id": conversation_id,
            "used": bool(compaction_meta.get("used")),
            "reason": str(compaction_meta.get("reason") or ""),
            "original_message_count": int(compaction_meta.get("original_message_count") or 0),
            "compacted_message_count": int(compaction_meta.get("compacted_message_count") or 0),
            "token_estimate_before": int(compaction_meta.get("token_estimate_before") or 0),
            "token_estimate_after": int(compaction_meta.get("token_estimate_after") or 0),
            "summary_chars": int(compaction_meta.get("summary_chars") or 0),
            "summary_hash": str(compaction_meta.get("summary_hash") or ""),
            "persisted_memory": bool(compaction_meta.get("persisted_memory")),
            "reused_summary": bool(compaction_meta.get("reused_summary")),
            "prefix_message_count": int(compaction_meta.get("prefix_message_count") or 0),
            "tail_message_count": int(compaction_meta.get("tail_message_count") or 0),
        }
        if compaction_event_payload["used"]:
            store.append_event(
                active_run.run_id,
                "conversation_context_compaction",
                compaction_event_payload,
            )

        messages_with_context = [*system_context_messages, *conversation_messages]
        store.append_event(
            active_run.run_id,
            "chat_started",
            {
                "message_count": len(messages_with_context),
                "raw_message_count": len(wire_messages),
                "max_tool_calls": req.budgets.max_tool_calls,
                "max_runtime_seconds": req.budgets.max_runtime_seconds,
                "uploaded_files": len(runtime_uploaded_files),
                "file_ids": len(req.file_ids),
                "resource_uri_count": len(req.resource_uris),
                "dataset_uri_count": len(req.dataset_uris),
                "bisque_import_count": len(imported_bisque_rows),
                "conversation_id": conversation_id,
                "user_id": user_id,
                "reasoning_mode": str(req.reasoning_mode or "deep"),
                "system_context_count": len(system_context_messages),
                "context_compaction": compaction_event_payload,
                "dataset_resolution": dataset_import_summaries,
                "bisque_imports": bisque_import_summaries,
                "runtime": "agno",
            },
        )

        try:
            legacy_stream = globals().get("stream_chat_completion")
            if callable(legacy_stream):
                progress_events: list[dict[str, Any]] = []
                tool_call_count = 0
                for chunk in legacy_stream(
                    messages_with_context,
                    uploaded_files=runtime_uploaded_files,
                ):
                    elapsed = time.monotonic() - started
                    if elapsed > float(req.budgets.max_runtime_seconds):
                        raise TimeoutError(
                            f"chat runtime exceeded budget ({req.budgets.max_runtime_seconds}s)"
                        )
                    progress = decode_progress_chunk(str(chunk or ""))
                    if progress:
                        progress_events.append(progress)
                        if str(progress.get("event") or "").strip().lower() == "started":
                            tool_call_count += 1
                            if tool_call_count > int(req.budgets.max_tool_calls):
                                raise RuntimeError(
                                    f"tool call budget exceeded ({req.budgets.max_tool_calls})"
                                )
                        continue
                    delta = str(chunk or "")
                    if delta:
                        response_chunks.append(delta)
                        yield {"event": "token", "data": {"delta": delta}}

                response_text = "".join(response_chunks)
                duration = round(time.monotonic() - started, 3)
                tool_artifact_entries = _snapshot_progress_artifacts(
                    active_run.run_id, progress_events
                )
                if tool_artifact_entries:
                    _update_manifest_with_entries(active_run.run_id, tool_artifact_entries)
                    store.append_event(
                        active_run.run_id,
                        "tool_outputs_snapshot",
                        {
                            "count": len(tool_artifact_entries),
                            "items": [
                                {
                                    "path": entry.get("path"),
                                    "tool": entry.get("tool"),
                                    "title": entry.get("title"),
                                    "source_path": entry.get("source_path"),
                                }
                                for entry in tool_artifact_entries[:100]
                            ],
                        },
                    )
                store.append_event(
                    active_run.run_id,
                    "chat_completed",
                    {
                        "duration_seconds": duration,
                        "tool_calls": tool_call_count,
                        "selected_domains": [],
                        "domain_output_count": 0,
                        "progress_events": len(progress_events),
                        "runtime": "legacy_test_hook",
                    },
                )
                store.update_status(active_run.run_id, RunStatus.SUCCEEDED)
                latest_run = store.get_run(active_run.run_id)
                _index_run_resource_computations(
                    run_id=active_run.run_id,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    run_goal=active_run.goal,
                    run_status=RunStatus.SUCCEEDED.value,
                    run_created_at=(
                        latest_run.created_at.isoformat()
                        if latest_run is not None
                        else active_run.created_at.isoformat()
                    ),
                    run_updated_at=(
                        latest_run.updated_at.isoformat()
                        if latest_run is not None
                        else datetime.utcnow().isoformat()
                    ),
                    progress_events=progress_events,
                    input_files=reuse_index_input_rows,
                )
                response = ChatResponse(
                    run_id=active_run.run_id,
                    model=llm_model_name,
                    response_text=response_text,
                    duration_seconds=duration,
                )
                store.append_event(
                    active_run.run_id,
                    "chat_done_payload",
                    {
                        "response": response.model_dump(mode="json"),
                        "progress_events": progress_events,
                        "tool_calls": tool_call_count,
                        "runtime_metadata": {"runtime": "legacy_test_hook"},
                    },
                )
                yield {"event": "done", "data": response}
                return

            saw_done = False
            workflow_hint_payload = _chat_workflow_hint_payload(req) or {}
            autonomous_transport_watchdog_seconds = float(
                getattr(settings, "pro_mode_autonomous_cycle_transport_watchdog_seconds", 1800)
                or 1800
            )
            transport_timeout_seconds = float(req.budgets.max_runtime_seconds)
            autonomous_transport_selected = False
            if str(workflow_hint_payload.get("id") or "").strip().lower() == "pro_mode":
                transport_timeout_seconds = max(
                    transport_timeout_seconds,
                    min(autonomous_transport_watchdog_seconds, 120.0),
                )

            def _maybe_enable_autonomous_transport(record: dict[str, Any]) -> None:
                nonlocal transport_timeout_seconds, autonomous_transport_selected
                payload = dict(record.get("payload") or {}) if isinstance(record, dict) else {}
                phase = str(payload.get("phase") or "").strip().lower()
                event_type = str(record.get("event_type") or "").strip().lower()
                nested_payload = (
                    dict(payload.get("payload") or {})
                    if isinstance(payload.get("payload"), dict)
                    else {}
                )
                execution_regime = (
                    str(
                        nested_payload.get("execution_regime")
                        or payload.get("execution_regime")
                        or ""
                    )
                    .strip()
                    .lower()
                )
                if (
                    phase == "execution_router"
                    and event_type == "pro_mode.phase_completed"
                    and execution_regime == "autonomous_cycle"
                ):
                    autonomous_transport_selected = True
                    transport_timeout_seconds = max(
                        transport_timeout_seconds,
                        autonomous_transport_watchdog_seconds,
                    )

            runtime_stream_kwargs: dict[str, Any] = {
                "messages": messages_with_context,
                "uploaded_files": runtime_uploaded_files,
                "max_tool_calls": int(req.budgets.max_tool_calls),
                "max_runtime_seconds": int(req.budgets.max_runtime_seconds),
                "conversation_id": conversation_id,
            }
            runtime_events_queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
            try:
                runtime_stream_signature = inspect.signature(agno_runtime.stream)
            except (TypeError, ValueError):
                runtime_stream_signature = None
            if (
                runtime_stream_signature is not None
                and "event_callback" in runtime_stream_signature.parameters
            ):

                def _emit_runtime_event(payload: dict[str, Any]) -> None:
                    record = _persist_agent_runtime_event(active_run.run_id, payload)
                    runtime_events_queue.put_nowait(("run_event", record))

                runtime_stream_kwargs["event_callback"] = _emit_runtime_event
            if (
                runtime_stream_signature is not None
                and "reasoning_mode" in runtime_stream_signature.parameters
            ):
                runtime_stream_kwargs["reasoning_mode"] = str(req.reasoning_mode or "deep")
            if (
                runtime_stream_signature is not None
                and "benchmark" in runtime_stream_signature.parameters
            ):
                runtime_stream_kwargs["benchmark"] = (
                    req.benchmark.model_dump(mode="json") if req.benchmark is not None else None
                )
            if (
                runtime_stream_signature is not None
                and "selected_tool_names" in runtime_stream_signature.parameters
            ):
                runtime_stream_kwargs["selected_tool_names"] = _chat_selected_tool_names(req)
            if (
                runtime_stream_signature is not None
                and "knowledge_context" in runtime_stream_signature.parameters
            ):
                runtime_stream_kwargs["knowledge_context"] = _chat_knowledge_context_payload(req)
            if (
                runtime_stream_signature is not None
                and "memory_policy" in runtime_stream_signature.parameters
            ):
                runtime_stream_kwargs["memory_policy"] = _chat_memory_policy_payload(req)
            if (
                runtime_stream_signature is not None
                and "knowledge_scope" in runtime_stream_signature.parameters
            ):
                runtime_stream_kwargs["knowledge_scope"] = _chat_knowledge_scope_payload(req)
            if (
                runtime_stream_signature is not None
                and "workflow_hint" in runtime_stream_signature.parameters
            ):
                runtime_stream_kwargs["workflow_hint"] = _chat_workflow_hint_payload(req)
            if (
                runtime_stream_signature is not None
                and "selection_context" in runtime_stream_signature.parameters
            ):
                runtime_stream_kwargs["selection_context"] = _chat_selection_context_payload(req)
            if (
                runtime_stream_signature is not None
                and "hitl_resume" in runtime_stream_signature.parameters
            ):
                runtime_stream_kwargs["hitl_resume"] = hitl_resume
            if (
                runtime_stream_signature is not None
                and "run_id" in runtime_stream_signature.parameters
            ):
                runtime_stream_kwargs["run_id"] = active_run.run_id
            if (
                runtime_stream_signature is not None
                and "user_id" in runtime_stream_signature.parameters
            ):
                runtime_stream_kwargs["user_id"] = user_id
            if (
                runtime_stream_signature is not None
                and "debug" in runtime_stream_signature.parameters
            ):
                runtime_stream_kwargs["debug"] = _chat_debug_requested(req)

            async def _runtime_stream_producer() -> None:
                try:
                    async for runtime_event in agno_runtime.stream(**runtime_stream_kwargs):
                        await runtime_events_queue.put(("runtime_event", runtime_event))
                except Exception as exc:
                    await runtime_events_queue.put(("runtime_error", exc))
                finally:
                    await runtime_events_queue.put(("runtime_complete", None))

            runtime_producer_task = asyncio.create_task(_runtime_stream_producer())
            try:
                while True:
                    queue_event_name, queued_payload = await runtime_events_queue.get()
                    if queue_event_name == "run_event" and isinstance(queued_payload, dict):
                        _maybe_enable_autonomous_transport(queued_payload)
                    elapsed = time.monotonic() - started
                    if elapsed > float(transport_timeout_seconds):
                        label = (
                            f"autonomous transport watchdog ({int(transport_timeout_seconds)}s)"
                            if autonomous_transport_selected
                            else f"chat runtime budget ({req.budgets.max_runtime_seconds}s)"
                        )
                        raise TimeoutError(f"chat runtime exceeded {label}")

                    if queue_event_name == "run_event":
                        yield {"event": "run_event", "data": queued_payload}
                        continue

                    if queue_event_name == "runtime_error":
                        raise cast(Exception, queued_payload)

                    if queue_event_name == "runtime_complete":
                        break

                    runtime_event = queued_payload
                    if not isinstance(runtime_event, dict):
                        delta = str(runtime_event or "")
                        if delta:
                            response_chunks.append(delta)
                            yield {"event": "token", "data": {"delta": delta}}
                        continue

                    event_name = str(runtime_event.get("event") or "").strip().lower()
                    data = runtime_event.get("data")

                    if event_name == "token":
                        delta = ""
                        if isinstance(data, dict):
                            delta = str(data.get("delta") or "")
                        if delta:
                            response_chunks.append(delta)
                            yield {"event": "token", "data": {"delta": delta}}
                        continue

                    if event_name == "error":
                        status_code = 500
                        detail: Any = "chat failed"
                        if isinstance(data, dict):
                            raw_status = data.get("status_code")
                            if isinstance(raw_status, int):
                                status_code = raw_status
                            elif isinstance(raw_status, str) and raw_status.isdigit():
                                status_code = int(raw_status)
                            detail = data.get("detail") or data
                        raise HTTPException(status_code=status_code, detail=detail)

                    if event_name != "done":
                        continue

                    payload = data if isinstance(data, dict) else {}
                    response_text = payload.get("response_text")
                    if response_text is None:
                        response_text = "".join(response_chunks)
                    else:
                        response_text = str(response_text)

                    selected_domains_raw = payload.get("selected_domains")
                    selected_domains = (
                        [
                            str(item).strip()
                            for item in selected_domains_raw
                            if str(item or "").strip()
                        ]
                        if isinstance(selected_domains_raw, list)
                        else []
                    )
                    raw_domain_outputs = payload.get("domain_outputs")
                    domain_outputs = (
                        {
                            str(key): str(value)
                            for key, value in raw_domain_outputs.items()
                            if str(key or "").strip()
                        }
                        if isinstance(raw_domain_outputs, dict)
                        else {}
                    )
                    tool_call_count = int(payload.get("tool_calls") or 0)
                    runtime_model = (
                        str(payload.get("model") or "").strip()
                        or agno_runtime.model
                        or llm_model_name
                    )
                    runtime_metadata_raw = payload.get("metadata")
                    runtime_metadata = (
                        dict(runtime_metadata_raw) if isinstance(runtime_metadata_raw, dict) else {}
                    )
                    benchmark_payload = (
                        payload.get("benchmark")
                        if isinstance(payload.get("benchmark"), dict)
                        else (
                            runtime_metadata.get("benchmark")
                            if isinstance(runtime_metadata.get("benchmark"), dict)
                            else None
                        )
                    )
                    tool_invocations = (
                        runtime_metadata.get("tool_invocations")
                        if isinstance(runtime_metadata.get("tool_invocations"), list)
                        else []
                    )
                    progress_events = _progress_events_from_tool_invocations(tool_invocations)
                    tool_artifact_entries = _snapshot_progress_artifacts(
                        active_run.run_id, progress_events
                    )
                    if tool_artifact_entries:
                        _update_manifest_with_entries(active_run.run_id, tool_artifact_entries)
                        store.append_event(
                            active_run.run_id,
                            "tool_outputs_snapshot",
                            {
                                "count": len(tool_artifact_entries),
                                "items": [
                                    {
                                        "path": entry.get("path"),
                                        "tool": entry.get("tool"),
                                        "title": entry.get("title"),
                                        "source_path": entry.get("source_path"),
                                    }
                                    for entry in tool_artifact_entries[:100]
                                ],
                            },
                        )

                    duration = round(time.monotonic() - started, 3)
                    interrupted = bool(runtime_metadata.get("interrupted"))
                    resume_decision = (
                        str(runtime_metadata.get("resume_decision") or "").strip().lower()
                    )
                    pending_hitl_payload = (
                        dict(runtime_metadata.get("pending_hitl") or {})
                        if isinstance(runtime_metadata.get("pending_hitl"), dict)
                        else {}
                    )
                    store.append_event(
                        active_run.run_id,
                        "chat_completed",
                        {
                            "duration_seconds": duration,
                            "tool_calls": tool_call_count,
                            "selected_domains": selected_domains,
                            "domain_output_count": len(domain_outputs),
                            "progress_events": len(progress_events),
                            "runtime": "agno",
                        },
                    )
                    if interrupted:
                        run_status = RunStatus.PENDING
                        checkpoint_phase = "pending_approval"
                        checkpoint_pending_hitl = pending_hitl_payload or None
                        agent_role = "domain_specialist"
                        run_error = None
                    elif resume_decision == "reject":
                        run_status = RunStatus.CANCELED
                        checkpoint_phase = "approval_rejected"
                        checkpoint_pending_hitl = None
                        agent_role = "domain_specialist"
                        run_error = str(response_text or "").strip() or "BisQue approval rejected."
                    else:
                        run_status = RunStatus.SUCCEEDED
                        checkpoint_phase = "completed"
                        checkpoint_pending_hitl = None
                        agent_role = "synthesizer"
                        run_error = None
                    store.update_status(active_run.run_id, run_status, error=run_error)
                    store.set_run_metadata(
                        active_run.run_id,
                        user_id=user_id,
                        conversation_id=conversation_id,
                        workflow_kind="interactive_chat",
                        mode="interactive",
                        planner_version="agno_v1",
                        agent_role=agent_role,
                        checkpoint_state={
                            **_chat_checkpoint_state_fragment(req),
                            "phase": checkpoint_phase,
                            "selected_domains": selected_domains,
                            "retry_count": int(runtime_metadata.get("retry_count") or 0),
                            "solve_mode": (
                                str(runtime_metadata.get("solve_mode") or "").strip() or None
                            ),
                            "pending_hitl": checkpoint_pending_hitl,
                            "resume_decision": resume_decision or None,
                        },
                        budget_state=_chat_budget_state(
                            req,
                            tool_calls_used=int(tool_call_count),
                            benchmark=(
                                req.benchmark.model_dump(mode="json")
                                if req.benchmark is not None
                                else None
                            ),
                        ),
                        trace_group_id=conversation_id,
                    )
                    latest_run = store.get_run(active_run.run_id)
                    if run_status == RunStatus.SUCCEEDED:
                        _index_run_resource_computations(
                            run_id=active_run.run_id,
                            user_id=user_id,
                            conversation_id=conversation_id,
                            run_goal=active_run.goal,
                            run_status=RunStatus.SUCCEEDED.value,
                            run_created_at=(
                                latest_run.created_at.isoformat()
                                if latest_run is not None
                                else active_run.created_at.isoformat()
                            ),
                            run_updated_at=(
                                latest_run.updated_at.isoformat()
                                if latest_run is not None
                                else datetime.utcnow().isoformat()
                            ),
                            progress_events=progress_events,
                            tool_invocations=tool_invocations,
                            input_files=reuse_index_input_rows,
                        )

                    response_metadata = dict(runtime_metadata or {})
                    latest_contract = next(
                        (
                            dict(event.get("contract") or {})
                            for event in reversed(progress_events)
                            if isinstance(event, dict)
                            and str(event.get("event") or "").strip().lower() == "workpad_contract"
                            and isinstance(event.get("contract"), dict)
                        ),
                        {},
                    )
                    if latest_contract and not isinstance(response_metadata.get("contract"), dict):
                        response_metadata["contract"] = latest_contract

                    response = ChatResponse(
                        run_id=active_run.run_id,
                        model=runtime_model,
                        response_text=str(response_text),
                        duration_seconds=duration,
                        progress_events=progress_events,
                        benchmark=benchmark_payload
                        if isinstance(benchmark_payload, dict)
                        else None,
                        metadata=response_metadata or None,
                    )
                    store.append_event(
                        active_run.run_id,
                        "chat_done_payload",
                        {
                            "response": response.model_dump(mode="json"),
                            "progress_events": progress_events,
                            "selected_domains": selected_domains,
                            "domain_outputs": domain_outputs,
                            "tool_calls": tool_call_count,
                            "runtime_metadata": response_metadata or runtime_metadata,
                        },
                    )
                    yield {"event": "done", "data": response}
                    saw_done = True
                    break
            finally:
                if not runtime_producer_task.done():
                    runtime_producer_task.cancel()
                with suppress(asyncio.CancelledError):
                    await runtime_producer_task

            if not saw_done:
                raise RuntimeError("chat failed: no completion payload")
        except asyncio.CancelledError:
            cancel_message = "Canceled by user."
            store.append_event(
                active_run.run_id,
                "chat_canceled",
                {"reason": cancel_message},
                level="warning",
            )
            store.update_status(active_run.run_id, RunStatus.CANCELED, error=cancel_message)
            raise
        except HTTPException as exc:
            error_text = str(exc.detail)
            store.append_event(
                active_run.run_id,
                "chat_failed",
                {"error": error_text, "status_code": int(exc.status_code)},
                level="error",
            )
            store.update_status(active_run.run_id, RunStatus.FAILED, error=error_text)
            raise
        except Exception as exc:
            error_text = str(exc)
            status_code = 500
            detail: Any = f"chat failed: {error_text}"
            if isinstance(exc, TimeoutError):
                status_code = 408
                detail = f"chat timeout: {error_text}"
            elif isinstance(exc, RuntimeError) and "budget exceeded" in error_text.lower():
                status_code = 429
                detail = f"chat budget exceeded: {error_text}"

            store.append_event(
                active_run.run_id,
                "chat_failed",
                {"error": error_text, "status_code": status_code},
                level="error",
            )
            store.update_status(active_run.run_id, RunStatus.FAILED, error=error_text)
            raise HTTPException(status_code=status_code, detail=detail) from exc

    def _v2_thread_title(messages: list[dict[str, Any]], fallback: str = "New thread") -> str:
        for message in messages:
            role = str(message.get("role") or "").strip().lower()
            if role != "user":
                continue
            content = re.sub(r"\s+", " ", str(message.get("content") or "")).strip()
            if content:
                return " ".join(content.split()[:8]).strip()[:120] or fallback
        return fallback

    def _normalize_v2_message_role(value: str | None) -> str:
        token = str(value or "").strip().lower()
        if token in {"system", "user", "assistant", "tool"}:
            return token
        if token == "developer":
            return "system"
        return "user"

    def _v2_message_payloads_to_snapshot(
        messages: list[V2ThreadMessage] | list[dict[str, Any]],
        *,
        thread_id: str,
        run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        now = datetime.utcnow().isoformat()
        snapshots: list[dict[str, Any]] = []
        for index, raw in enumerate(messages):
            payload = raw.model_dump(mode="json") if hasattr(raw, "model_dump") else dict(raw)
            snapshots.append(
                {
                    "id": str(payload.get("message_id") or f"{thread_id}:{index}"),
                    "role": _normalize_v2_message_role(str(payload.get("role") or "user")),
                    "content": str(payload.get("content") or ""),
                    "createdAt": (
                        str(payload.get("created_at") or "").strip()
                        or str(payload.get("createdAt") or "").strip()
                        or now
                    ),
                    "runId": str(
                        payload.get("run_id") or payload.get("runId") or run_id or ""
                    ).strip()
                    or None,
                    "metadata": (
                        dict(payload.get("metadata") or {})
                        if isinstance(payload.get("metadata"), dict)
                        else {}
                    ),
                }
            )
        return snapshots

    def _v2_thread_record(thread_row: dict[str, Any]) -> V2ThreadRecord:
        return V2ThreadRecord(
            thread_id=str(thread_row.get("thread_id") or ""),
            user_id=str(thread_row.get("user_id") or "").strip() or None,
            title=str(thread_row.get("title") or "").strip() or None,
            status=str(thread_row.get("status") or "active"),
            created_at=datetime.fromisoformat(
                str(thread_row.get("created_at") or datetime.utcnow().isoformat())
            ),
            updated_at=datetime.fromisoformat(
                str(thread_row.get("updated_at") or datetime.utcnow().isoformat())
            ),
            latest_run_id=str(thread_row.get("latest_run_id") or "").strip() or None,
            checkpoint_id=str(thread_row.get("checkpoint_id") or "").strip() or None,
            summary=str(thread_row.get("summary") or "").strip() or None,
            metadata=dict(thread_row.get("metadata") or {}),
        )

    def _v2_thread_message_record(message_row: dict[str, Any]) -> V2ThreadMessage:
        created_at_raw = str(message_row.get("created_at") or "").strip()
        created_at = datetime.fromisoformat(created_at_raw) if created_at_raw else None
        return V2ThreadMessage(
            message_id=str(message_row.get("message_id") or "").strip() or None,
            thread_id=str(message_row.get("thread_id") or "").strip() or None,
            role=_normalize_v2_message_role(str(message_row.get("role") or "user")),
            content=str(message_row.get("content") or ""),
            created_at=created_at,
            metadata=dict(message_row.get("metadata") or {}),
            run_id=str(message_row.get("run_id") or "").strip() or None,
        )

    def _map_run_status_to_v2(
        run: WorkflowRun, response_payload: ChatResponse | None = None
    ) -> str:
        checkpoint_state = run.checkpoint_state if isinstance(run.checkpoint_state, dict) else {}
        phase = str(checkpoint_state.get("phase") or "").strip().lower()
        if run.status == RunStatus.PENDING:
            if phase == "pending_approval":
                return "waiting_for_input"
            if phase in {"waiting_for_task", "task_pending"}:
                return "waiting_for_task"
            return "queued"
        if run.status == RunStatus.RUNNING:
            return "running"
        if run.status == RunStatus.SUCCEEDED:
            return "succeeded"
        if run.status == RunStatus.FAILED:
            return "failed"
        if run.status == RunStatus.CANCELED:
            return "canceled"
        if response_payload is not None:
            return "succeeded"
        return "queued"

    def _done_payload_for_run(run_id: str) -> ChatResponse | None:
        done_event = store.get_latest_event(run_id, "chat_done_payload")
        if not done_event:
            return None
        payload = done_event.get("payload") if isinstance(done_event, dict) else None
        response_payload = payload.get("response") if isinstance(payload, dict) else None
        if not isinstance(response_payload, dict):
            return None
        try:
            return ChatResponse.model_validate(response_payload)
        except Exception:
            return None

    def _v2_run_record(
        run: WorkflowRun,
        *,
        thread_id: str | None = None,
        user_id: str | None = None,
    ) -> V2RunRecord:
        metadata_row = store.get_run_metadata(run.run_id) or {}
        effective_thread_id = (
            str(thread_id or "").strip()
            or str(metadata_row.get("conversation_id") or "").strip()
            or None
        )
        effective_user_id = (
            str(user_id or "").strip() or str(metadata_row.get("user_id") or "").strip() or None
        )
        checkpoint_state = run.checkpoint_state if isinstance(run.checkpoint_state, dict) else None
        budget_state = run.budget_state if isinstance(run.budget_state, dict) else None
        response_payload = _done_payload_for_run(run.run_id)
        response_text = (
            str(response_payload.response_text or "").strip()
            if response_payload is not None
            else None
        )
        phase = str((checkpoint_state or {}).get("phase") or "").strip() or None
        checkpoint_id = str((checkpoint_state or {}).get("checkpoint_id") or "").strip() or None
        metadata: dict[str, Any] = {}
        if run.workflow_profile:
            metadata["workflow_profile"] = run.workflow_profile
        if run.pedagogy_level:
            metadata["pedagogy_level"] = run.pedagogy_level
        return V2RunRecord(
            run_id=run.run_id,
            thread_id=effective_thread_id,
            user_id=effective_user_id,
            goal=run.goal,
            status=_map_run_status_to_v2(run, response_payload=response_payload),
            workflow_kind=str(run.workflow_kind or "interactive_chat"),
            mode=str(run.mode or "").strip() or None,
            current_node=phase,
            parent_run_id=str(run.parent_run_id or "").strip() or None,
            planner_version=str(run.planner_version or "").strip() or None,
            agent_role=str(run.agent_role or "").strip() or None,
            trace_group_id=str(run.trace_group_id or "").strip() or None,
            checkpoint_id=checkpoint_id,
            checkpoint_state=checkpoint_state,
            budget_state=budget_state,
            response_text=response_text,
            error=str(run.error or "").strip() or None,
            created_at=run.created_at,
            updated_at=run.updated_at,
            started_at=run.created_at,
            completed_at=(
                run.updated_at
                if run.status in {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELED}
                else None
            ),
            metadata=metadata,
        )

    def _map_event_type_to_v2(event_type: str, payload: dict[str, Any]) -> str:
        token = str(event_type or "").strip()
        if token in {
            "run.started",
            "node.started",
            "node.completed",
            "state.updated",
            "task.queued",
            "task.progress",
            "task.completed",
            "artifact.created",
            "interrupt.raised",
            "checkpoint.created",
            "run.completed",
            "run.failed",
            "message.delta",
            "error",
        }:
            return token
        if token in {"run_created", "chat_started"}:
            return "run.started"
        if token in {"chat_completed", "chat_done_payload"}:
            return "run.completed"
        if token in {"chat_failed", "chat_canceled"}:
            return "run.failed"
        if token in {"tool_outputs_snapshot", "uploads_snapshot"}:
            return "artifact.created"
        if token == "tool_event" or str(payload.get("kind") or "").strip().lower() == "tool":
            return "task.progress"
        return "state.updated"

    def _v2_event_record(
        run_id: str,
        *,
        thread_id: str | None,
        event: dict[str, Any],
    ) -> V2GraphEventRecord:
        payload = dict(event.get("payload") or {})
        event_type = str(event.get("event_type") or "")
        event_kind = _map_event_type_to_v2(event_type, payload)
        ts_raw = str(event.get("ts") or "").strip()
        ts_value = datetime.fromisoformat(ts_raw) if ts_raw else None
        message = str(payload.get("message") or payload.get("error") or "").strip() or None
        return V2GraphEventRecord(
            run_id=run_id,
            thread_id=thread_id,
            event_kind=event_kind,  # type: ignore[arg-type]
            event_type=event_type or event_kind,
            node_name=(str(payload.get("node") or payload.get("phase") or "").strip() or None),
            task_id=str(payload.get("task_id") or "").strip() or None,
            checkpoint_id=str(payload.get("checkpoint_id") or "").strip() or None,
            scope_id=str(payload.get("scope_id") or "").strip() or None,
            agent_role=str(payload.get("agent_role") or "").strip() or None,
            level=str(event.get("level") or "").strip() or None,
            ts=ts_value,
            message=message,
            payload=payload,
        )

    def _v2_artifact_record(
        run_id: str,
        *,
        thread_id: str | None,
        artifact: ArtifactRecord,
    ) -> V2ArtifactRecord:
        artifact_path = str(artifact.path or "").strip()
        created_at = (
            artifact.modified_at
            if isinstance(artifact.modified_at, datetime)
            else datetime.utcnow()
        )
        return V2ArtifactRecord(
            artifact_id=f"{run_id}::{artifact_path}",
            run_id=run_id,
            thread_id=thread_id,
            kind="artifact",
            path=artifact_path or None,
            source_path=str(artifact.source_path or "").strip() or None,
            preview_path=None,
            title=str(artifact.title or "").strip() or None,
            result_group_id=str(artifact.result_group_id or "").strip() or None,
            mime_type=str(artifact.mime_type or "").strip() or None,
            size_bytes=int(artifact.size_bytes or 0),
            created_at=created_at,
            updated_at=created_at,
            metadata={},
        )

    def _chat_request_from_v2(thread_id: str, req: V2RunCreateRequest) -> ChatRequest:
        wire_messages = [
            {
                "role": _normalize_v2_message_role(message.role),
                "content": str(message.content or ""),
            }
            for message in req.messages
        ]
        if not wire_messages:
            wire_messages = [
                {
                    "role": "user",
                    "content": str(req.goal or "Start thread").strip() or "Start thread",
                }
            ]
        return ChatRequest.model_validate(
            {
                "messages": wire_messages,
                "uploaded_files": [],
                "file_ids": list(req.file_ids or []),
                "resource_uris": list(req.resource_uris or []),
                "dataset_uris": list(req.dataset_uris or []),
                "conversation_id": thread_id,
                "goal": req.goal,
                "selected_tool_names": list(req.selected_tool_names or []),
                "knowledge_context": (
                    req.knowledge_context.model_dump(mode="json")
                    if req.knowledge_context is not None
                    else None
                ),
                "selection_context": (
                    req.selection_context.model_dump(mode="json")
                    if req.selection_context is not None
                    else None
                ),
                "workflow_hint": (
                    req.workflow_hint.model_dump(mode="json")
                    if req.workflow_hint is not None
                    else None
                ),
                "reasoning_mode": str(req.reasoning_mode or "deep"),
                "budgets": req.budgets.model_dump(mode="json"),
                "benchmark": (
                    req.benchmark.model_dump(mode="json") if req.benchmark is not None else None
                ),
            }
        )

    def _sync_v2_thread_projection(
        *,
        thread_id: str,
        user_id: str,
        title: str,
        metadata: dict[str, Any],
        messages: list[dict[str, Any]],
        latest_run_id: str | None = None,
        checkpoint_id: str | None = None,
        summary: str | None = None,
    ) -> dict[str, Any]:
        existing = store.get_thread(thread_id=thread_id, user_id=user_id)
        created_at = str((existing or {}).get("created_at") or datetime.utcnow().isoformat())
        updated_at = datetime.utcnow().isoformat()
        thread_row = store.upsert_thread(
            thread_id=thread_id,
            user_id=user_id,
            conversation_id=thread_id,
            title=title,
            status="active",
            latest_run_id=latest_run_id,
            checkpoint_id=checkpoint_id,
            summary=summary,
            metadata=metadata,
            created_at=created_at,
            updated_at=updated_at,
        )
        store.replace_thread_messages(thread_id=thread_id, user_id=user_id, messages=messages)
        store.upsert_conversation(
            conversation_id=thread_id,
            user_id=user_id,
            title=title,
            created_at=created_at,
            updated_at=updated_at,
            state={
                "thread_id": thread_id,
                "metadata": metadata,
                "latest_run_id": latest_run_id,
                "checkpoint_id": checkpoint_id,
                "summary": summary,
            },
        )
        store.replace_conversation_messages(
            conversation_id=thread_id,
            user_id=user_id,
            messages=messages,
        )
        if summary:
            store.upsert_thread_summary(thread_id=thread_id, summary=summary, updated_at=updated_at)
        return thread_row

    async def _execute_v2_chat_run(
        *,
        thread_id: str,
        req: V2RunCreateRequest,
        request: FastAPIRequest,
        bisque_auth: dict[str, Any] | None,
    ) -> tuple[WorkflowRun, ChatResponse, str]:
        request_user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        chat_req = _chat_request_from_v2(thread_id, req)
        request_messages = _v2_message_payloads_to_snapshot(req.messages, thread_id=thread_id)
        thread_row = store.get_thread(thread_id=thread_id, user_id=request_user_id)
        title = (
            str((thread_row or {}).get("title") or "").strip()
            or str(req.metadata.get("title") or "").strip()
        )
        if not title:
            title = _v2_thread_title(request_messages)
        metadata = dict((thread_row or {}).get("metadata") or {})
        metadata.update(dict(req.metadata or {}))
        _sync_v2_thread_projection(
            thread_id=thread_id,
            user_id=request_user_id,
            title=title,
            metadata=metadata,
            messages=request_messages,
            latest_run_id=str((thread_row or {}).get("latest_run_id") or "").strip() or None,
            checkpoint_id=str((thread_row or {}).get("checkpoint_id") or "").strip() or None,
            summary=str((thread_row or {}).get("summary") or "").strip() or None,
        )

        auth_context = _build_request_bisque_auth_context(
            bisque_auth=bisque_auth,
            request=request,
        )
        context_token = set_request_bisque_auth(auth_context) if auth_context else None
        try:
            active_run, hitl_resume = _resolve_chat_run_context(chat_req, user_id=request_user_id)
            final_response: ChatResponse | None = None
            async for event in _run_chat_events(
                chat_req,
                user_id=request_user_id,
                run=active_run,
                bisque_auth=bisque_auth,
                hitl_resume=hitl_resume,
            ):
                if event.get("event") != "done":
                    continue
                payload = event.get("data")
                if isinstance(payload, ChatResponse):
                    final_response = payload
                elif isinstance(payload, dict):
                    final_response = ChatResponse.model_validate(payload)
            if final_response is None:
                raise HTTPException(status_code=500, detail="v2 run failed: no completion payload")
        finally:
            if context_token is not None:
                reset_request_bisque_auth(context_token)

        latest_run = store.get_run(active_run.run_id) or active_run
        assistant_message = {
            "id": f"{thread_id}:assistant:{latest_run.run_id}",
            "role": "assistant",
            "content": str(final_response.response_text or ""),
            "createdAt": datetime.utcnow().isoformat(),
            "runId": latest_run.run_id,
            "metadata": {},
        }
        checkpoint_state = (
            latest_run.checkpoint_state if isinstance(latest_run.checkpoint_state, dict) else {}
        )
        checkpoint_id = str(checkpoint_state.get("checkpoint_id") or "").strip() or None
        summary = str(final_response.response_text or "").strip()[:400] or None
        _sync_v2_thread_projection(
            thread_id=thread_id,
            user_id=request_user_id,
            title=title,
            metadata=metadata,
            messages=[*request_messages, assistant_message],
            latest_run_id=latest_run.run_id,
            checkpoint_id=checkpoint_id,
            summary=summary,
        )
        return latest_run, final_response, request_user_id

    @v2.post("/threads", response_model=V2ThreadRecord)
    def create_thread_v2(
        req: V2ThreadCreateRequest,
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> V2ThreadRecord:
        del _auth
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        thread_id = str(req.conversation_id or uuid4()).strip()
        messages = _v2_message_payloads_to_snapshot(req.initial_messages, thread_id=thread_id)
        title = str(req.title or "").strip() or _v2_thread_title(messages)
        thread_row = _sync_v2_thread_projection(
            thread_id=thread_id,
            user_id=user_id,
            title=title,
            metadata=dict(req.metadata or {}),
            messages=messages,
        )
        return _v2_thread_record(thread_row)

    @v2.get("/threads", response_model=V2ThreadListResponse)
    def list_threads_v2(
        limit: int = Query(default=200, ge=1, le=1000),
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> V2ThreadListResponse:
        del _auth
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        rows = store.list_threads(user_id=user_id, limit=limit)
        threads = [_v2_thread_record(row) for row in rows]
        return V2ThreadListResponse(count=len(threads), threads=threads)

    @v2.get("/threads/{thread_id}/messages", response_model=V2ThreadMessageListResponse)
    def get_thread_messages_v2(
        thread_id: str,
        limit: int = Query(default=500, ge=1, le=5000),
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> V2ThreadMessageListResponse:
        del _auth
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        thread_row = store.get_thread(thread_id=thread_id, user_id=user_id)
        if thread_row is None:
            raise HTTPException(status_code=404, detail="Thread not found")
        messages = [
            _v2_thread_message_record(row)
            for row in store.list_thread_messages(thread_id=thread_id, user_id=user_id, limit=limit)
        ]
        return V2ThreadMessageListResponse(
            thread_id=thread_id, count=len(messages), messages=messages
        )

    @v2.post("/threads/{thread_id}/runs", response_model=V2RunRecord)
    async def create_run_v2(
        thread_id: str,
        req: V2RunCreateRequest,
        request: FastAPIRequest,
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> V2RunRecord:
        del _auth
        request_user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        if not req.messages:
            thread_row = store.get_thread(thread_id=thread_id, user_id=request_user_id)
            title = str((thread_row or {}).get("title") or "").strip() or "New thread"
            metadata = dict((thread_row or {}).get("metadata") or {})
            metadata.update(dict(req.metadata or {}))
            _sync_v2_thread_projection(
                thread_id=thread_id,
                user_id=request_user_id,
                title=title,
                metadata=metadata,
                messages=store.list_thread_messages(
                    thread_id=thread_id, user_id=request_user_id, limit=1000
                ),
                latest_run_id=str((thread_row or {}).get("latest_run_id") or "").strip() or None,
                checkpoint_id=str((thread_row or {}).get("checkpoint_id") or "").strip() or None,
                summary=str((thread_row or {}).get("summary") or "").strip() or None,
            )
            run = WorkflowRun.new(
                goal=str(req.goal or "v2 scaffold run").strip() or "v2 scaffold run",
                plan=None,
                workflow_kind="interactive_chat",
                mode="interactive",
                planner_version="langgraph_v2",
                agent_role="triage",
                checkpoint_state={"phase": "queued", "thread_id": thread_id},
                budget_state=req.budgets.model_dump(mode="json"),
                trace_group_id=thread_id,
                workflow_profile="interactive_chat",
            )
            store.create_run(run)
            store.set_run_metadata(
                run.run_id,
                user_id=request_user_id,
                conversation_id=thread_id,
                workflow_kind="interactive_chat",
                mode="interactive",
                planner_version="langgraph_v2",
                agent_role="triage",
                checkpoint_state={"phase": "queued", "thread_id": thread_id},
                budget_state=req.budgets.model_dump(mode="json"),
                trace_group_id=thread_id,
                workflow_profile="interactive_chat",
            )
            store.append_event(
                run.run_id,
                "run.started",
                {
                    "thread_id": thread_id,
                    "workflow_kind": "interactive_chat",
                    "graph": "interactive_chat",
                    "status": "queued",
                },
            )
            store.upsert_thread(
                thread_id=thread_id,
                user_id=request_user_id,
                conversation_id=thread_id,
                title=title,
                status="active",
                latest_run_id=run.run_id,
                checkpoint_id=None,
                summary=str((thread_row or {}).get("summary") or "").strip() or None,
                metadata=metadata,
                created_at=str(
                    (thread_row or {}).get("created_at") or datetime.utcnow().isoformat()
                ),
                updated_at=datetime.utcnow().isoformat(),
            )
            return _v2_run_record(run, thread_id=thread_id, user_id=request_user_id)
        latest_run, _, request_user_id = await _execute_v2_chat_run(
            thread_id=thread_id,
            req=req,
            request=request,
            bisque_auth=bisque_auth,
        )
        _assert_run_owner_access(run_id=latest_run.run_id, request_user_id=request_user_id)
        return _v2_run_record(latest_run, thread_id=thread_id, user_id=request_user_id)

    @v2.get("/runs", response_model=V2RunListResponse)
    def list_runs_v2(
        limit: int = Query(default=100, ge=1, le=1000),
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> V2RunListResponse:
        del _auth
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        rows = store.list_runs_for_user(user_id=user_id, limit=limit)
        runs: list[V2RunRecord] = []
        for row in rows:
            run_id = str(row.get("run_id") or "").strip()
            if not run_id:
                continue
            run = store.get_run(run_id)
            if run is None:
                continue
            runs.append(
                _v2_run_record(
                    run,
                    thread_id=str(row.get("conversation_id") or "").strip() or None,
                    user_id=user_id,
                )
            )
        return V2RunListResponse(count=len(runs), runs=runs)

    @v2.get("/runs/{run_id}", response_model=V2RunRecord)
    def get_run_v2(
        run_id: str,
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> V2RunRecord:
        del _auth
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        request_user_id = _resolve_request_user_id(bisque_auth)
        _assert_run_owner_access(run_id=run_id, request_user_id=request_user_id)
        metadata_row = store.get_run_metadata(run_id) or {}
        return _v2_run_record(
            run,
            thread_id=str(metadata_row.get("conversation_id") or "").strip() or None,
            user_id=str(metadata_row.get("user_id") or "").strip() or None,
        )

    @v2.get("/runs/{run_id}/events", response_model=V2RunEventsResponse)
    def get_run_events_v2(
        run_id: str,
        limit: int = Query(default=500, ge=1, le=5000),
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> V2RunEventsResponse:
        del _auth
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        request_user_id = _resolve_request_user_id(bisque_auth)
        _assert_run_owner_access(run_id=run_id, request_user_id=request_user_id)
        metadata_row = store.get_run_metadata(run_id) or {}
        thread_id = str(metadata_row.get("conversation_id") or "").strip() or None
        events = [
            _v2_event_record(run_id, thread_id=thread_id, event=event)
            for event in store.list_events(run_id, limit=limit)
        ]
        return V2RunEventsResponse(run_id=run_id, count=len(events), events=events)

    @v2.post("/runs/{run_id}/cancel", response_model=V2RunRecord)
    def cancel_run_v2(
        run_id: str,
        req: V2RunCancelRequest,
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> V2RunRecord:
        del _auth
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        request_user_id = _resolve_request_user_id(bisque_auth)
        _assert_run_owner_access(run_id=run_id, request_user_id=request_user_id)
        checkpoint_state = run.checkpoint_state if isinstance(run.checkpoint_state, dict) else {}
        phase = str(checkpoint_state.get("phase") or "").strip().lower()
        if (
            run.status in {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELED}
            and phase != "pending_approval"
        ):
            raise HTTPException(status_code=409, detail="Run is already terminal")
        reason = str(req.reason or "Canceled by user.").strip() or "Canceled by user."
        store.update_status(run_id, RunStatus.CANCELED, error=reason)
        store.append_event(run_id, "run.failed", {"reason": reason}, level="warning")
        updated = store.get_run(run_id)
        if updated is None:
            raise HTTPException(status_code=500, detail="Run disappeared during cancel")
        metadata_row = store.get_run_metadata(run_id) or {}
        return _v2_run_record(
            updated,
            thread_id=str(metadata_row.get("conversation_id") or "").strip() or None,
            user_id=str(metadata_row.get("user_id") or "").strip() or None,
        )

    @v2.post("/runs/{run_id}/resume", response_model=V2RunRecord)
    async def resume_run_v2(
        run_id: str,
        req: V2RunResumeRequest,
        request: FastAPIRequest,
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> V2RunRecord:
        del _auth
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        request_user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        _assert_run_owner_access(run_id=run_id, request_user_id=request_user_id)
        checkpoint_state = run.checkpoint_state if isinstance(run.checkpoint_state, dict) else {}
        if str(checkpoint_state.get("phase") or "").strip().lower() != "pending_approval":
            raise HTTPException(status_code=409, detail="Run is not waiting for input")
        thread_id = str((store.get_run_metadata(run_id) or {}).get("conversation_id") or "").strip()
        if not thread_id:
            raise HTTPException(status_code=409, detail="Run is missing thread linkage")
        thread_messages = [
            _v2_thread_message_record(row)
            for row in store.list_thread_messages(
                thread_id=thread_id, user_id=request_user_id, limit=1000
            )
        ]
        decision = str(req.decision or "").strip().lower()
        if decision not in {"approve", "reject"}:
            note_text = str(req.note or "").strip().lower()
            if note_text.startswith(
                ("approve", "approved", "yes", "y", "go ahead", "continue", "proceed")
            ):
                decision = "approve"
            elif note_text.startswith(("reject", "rejected", "cancel", "no", "stop", "deny")):
                decision = "reject"
        if decision not in {"approve", "reject"}:
            decision = "approve"
        resume_message = V2ThreadMessage(
            role="user",
            content=str(req.note or decision),
            created_at=datetime.utcnow(),
            metadata=dict(req.metadata or {}),
        )
        budget_state = run.budget_state if isinstance(run.budget_state, dict) else {}
        checkpoint_state = run.checkpoint_state if isinstance(run.checkpoint_state, dict) else {}
        resumed_req = V2RunCreateRequest(
            goal=run.goal,
            messages=[*thread_messages, resume_message],
            selected_tool_names=list(
                budget_state.get("selected_tool_names")
                or checkpoint_state.get("selected_tool_names")
                or []
            ),
            knowledge_context=(
                req_model
                if isinstance(
                    (
                        req_model := (
                            checkpoint_state.get("knowledge_context")
                            if isinstance(checkpoint_state.get("knowledge_context"), dict)
                            else budget_state.get("knowledge_context")
                        )
                    ),
                    dict,
                )
                else None
            ),
            selection_context=(
                req_model
                if isinstance(
                    (
                        req_model := (
                            checkpoint_state.get("selection_context")
                            if isinstance(checkpoint_state.get("selection_context"), dict)
                            else budget_state.get("selection_context")
                        )
                    ),
                    dict,
                )
                else None
            ),
            workflow_hint=(
                req_model
                if isinstance(
                    (
                        req_model := (
                            checkpoint_state.get("workflow_hint")
                            if isinstance(checkpoint_state.get("workflow_hint"), dict)
                            else budget_state.get("workflow_hint")
                        )
                    ),
                    dict,
                )
                else None
            ),
            reasoning_mode=str(budget_state.get("reasoning_mode") or "deep"),
            budgets={
                "max_tool_calls": int(budget_state.get("max_tool_calls") or 12),
                "max_runtime_seconds": int(budget_state.get("max_runtime_seconds") or 900),
            },
            metadata={"resume_run_id": run_id, **dict(req.metadata or {})},
        )
        latest_run, _, _ = await _execute_v2_chat_run(
            thread_id=thread_id,
            req=V2RunCreateRequest.model_validate(resumed_req.model_dump(mode="json")),
            request=request,
            bisque_auth=bisque_auth,
        )
        return _v2_run_record(latest_run, thread_id=thread_id, user_id=request_user_id)

    @v2.get("/runs/{run_id}/artifacts", response_model=V2ArtifactListResponse)
    def get_run_artifacts_v2(
        run_id: str,
        limit: int = Query(default=500, ge=1, le=5000),
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> V2ArtifactListResponse:
        del _auth
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        request_user_id = _resolve_request_user_id(bisque_auth)
        _assert_run_owner_access(run_id=run_id, request_user_id=request_user_id)
        thread_id = (
            str((store.get_run_metadata(run_id) or {}).get("conversation_id") or "").strip() or None
        )
        artifacts = [
            _v2_artifact_record(run_id, thread_id=thread_id, artifact=artifact)
            for artifact in _list_artifacts(run_id, limit=limit)
        ]
        return V2ArtifactListResponse(run_id=run_id, count=len(artifacts), artifacts=artifacts)

    @v2.get("/artifacts/{artifact_id}", response_model=V2ArtifactResponse)
    def get_artifact_v2(
        artifact_id: str,
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> V2ArtifactResponse:
        del _auth
        run_id, separator, artifact_path = artifact_id.partition("::")
        if not separator or not run_id or not artifact_path:
            raise HTTPException(status_code=404, detail="Artifact not found")
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        request_user_id = _resolve_request_user_id(bisque_auth)
        _assert_run_owner_access(run_id=run_id, request_user_id=request_user_id)
        thread_id = (
            str((store.get_run_metadata(run_id) or {}).get("conversation_id") or "").strip() or None
        )
        artifacts = _list_artifacts(run_id, limit=5000)
        for artifact in artifacts:
            if str(artifact.path or "").strip() != artifact_path:
                continue
            return V2ArtifactResponse(
                artifact=_v2_artifact_record(run_id, thread_id=thread_id, artifact=artifact)
            )
        raise HTTPException(status_code=404, detail="Artifact not found")

    @v1.post("/chat/title", response_model=ChatTitleResponse)
    @legacy.post("/chat/title", response_model=ChatTitleResponse)
    def chat_title(
        req: ChatTitleRequest,
        _auth: None = Depends(_require_api_key),
    ) -> ChatTitleResponse:
        del _auth
        if not req.messages:
            raise HTTPException(status_code=400, detail="messages cannot be empty")
        wire_messages = [msg.model_dump() for msg in req.messages]
        try:
            title, strategy = generate_chat_title(
                wire_messages,
                max_words=int(req.max_words),
            )
        except Exception:
            seed = re.sub(r"\s+", " ", _latest_user_text(wire_messages)).strip()
            tokens = seed.split()
            title = " ".join(tokens[: int(req.max_words or 4)]).strip() or "New conversation"
            strategy = "fallback"
        return ChatTitleResponse(
            title=title,
            model=llm_model_name,
            strategy="llm" if strategy == "llm" else "fallback",
        )

    def _conversation_preview_from_state(state: dict[str, Any]) -> str:
        messages = state.get("messages")
        if isinstance(messages, list):
            latest_user: str | None = None
            latest_any: str | None = None
            for row in messages:
                if not isinstance(row, dict):
                    continue
                content = str(row.get("content") or "").strip()
                if not content:
                    continue
                latest_any = content
                if str(row.get("role") or "").strip() == "user":
                    latest_user = content
            candidate = latest_user or latest_any
            if candidate:
                return candidate.replace("\n", " ")[:160]
        return ""

    def _conversation_running_from_state(state: dict[str, Any]) -> bool:
        return bool(state.get("sending"))

    def _conversation_record_from_row(
        row: dict[str, Any],
        *,
        include_state: bool,
    ) -> ConversationRecord:
        state = dict(row.get("state") or {})
        messages = state.get("messages")
        message_count = len(messages) if isinstance(messages, list) else 0
        return ConversationRecord(
            conversation_id=str(row.get("conversation_id") or ""),
            title=str(row.get("title") or "New conversation"),
            created_at_ms=_ms_from_iso(str(row.get("created_at") or "")),
            updated_at_ms=_ms_from_iso(str(row.get("updated_at") or "")),
            preview=_conversation_preview_from_state(state),
            message_count=message_count,
            preferred_panel="chat",
            running=_conversation_running_from_state(state),
            state=state if include_state else {},
        )

    @v1.get("/conversations", response_model=ConversationListResponse)
    @legacy.get("/conversations", response_model=ConversationListResponse)
    def list_conversations(
        limit: int = Query(default=25, ge=1, le=1000),
        offset: int = Query(default=0, ge=0),
        include_state: bool = Query(default=False),
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> ConversationListResponse:
        del _auth
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        total_count = store.count_conversations(user_id=user_id)
        rows = store.list_conversations(user_id=user_id, limit=limit, offset=offset)
        records = [_conversation_record_from_row(row, include_state=include_state) for row in rows]
        return ConversationListResponse(
            count=len(records),
            total_count=total_count,
            limit=int(limit),
            offset=int(offset),
            has_more=offset + len(records) < total_count,
            conversations=records,
        )

    @v1.get("/conversations/search", response_model=ConversationSearchResponse)
    @legacy.get("/conversations/search", response_model=ConversationSearchResponse)
    def search_conversations(
        q: str = Query(..., min_length=1),
        limit: int = Query(default=50, ge=1, le=200),
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> ConversationSearchResponse:
        del _auth
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        matches = _search_conversation_messages_flexible(
            user_id=user_id,
            query_text=q,
            limit=limit,
        )
        return ConversationSearchResponse(query=q, count=len(matches), matches=matches)

    @v1.get("/conversations/{conversation_id}", response_model=ConversationRecord)
    @legacy.get("/conversations/{conversation_id}", response_model=ConversationRecord)
    def get_conversation(
        conversation_id: str,
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> ConversationRecord:
        del _auth
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        row = store.get_conversation(conversation_id=conversation_id, user_id=user_id)
        if not row:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        return _conversation_record_from_row(row, include_state=True)

    @v1.put("/conversations/{conversation_id}", response_model=ConversationRecord)
    @legacy.put("/conversations/{conversation_id}", response_model=ConversationRecord)
    def upsert_conversation(
        conversation_id: str,
        req: ConversationUpsertRequest,
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> ConversationRecord:
        del _auth
        if conversation_id != req.conversation_id:
            raise HTTPException(
                status_code=400, detail="Path conversation_id does not match payload."
            )
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        state_payload = dict(req.state or {})
        messages = state_payload.get("messages")
        messages_list = messages if isinstance(messages, list) else []
        if "messages" in state_payload:
            state_payload = {**state_payload, "messages": messages_list}
        indexed_messages = [message for message in messages_list if isinstance(message, dict)]

        store.upsert_conversation(
            conversation_id=conversation_id,
            user_id=user_id,
            title=str(req.title),
            created_at=_iso_from_ms(req.created_at_ms),
            updated_at=_iso_from_ms(req.updated_at_ms),
            state=state_payload,
        )
        store.replace_conversation_messages(
            conversation_id=conversation_id,
            user_id=user_id,
            messages=indexed_messages,
        )
        return _conversation_record_from_row(
            {
                "conversation_id": conversation_id,
                "title": str(req.title),
                "created_at": datetime.utcfromtimestamp(int(req.created_at_ms) / 1000).isoformat(),
                "updated_at": datetime.utcfromtimestamp(int(req.updated_at_ms) / 1000).isoformat(),
                "state": state_payload,
            },
            include_state=True,
        )

    @v1.delete("/conversations/{conversation_id}")
    @legacy.delete("/conversations/{conversation_id}")
    def delete_conversation(
        conversation_id: str,
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> dict[str, Any]:
        del _auth
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        store.delete_conversation(conversation_id=conversation_id, user_id=user_id)
        return {"deleted": True, "conversation_id": conversation_id}

    @v1.post("/chat", response_model=ChatResponse)
    @legacy.post("/chat", response_model=ChatResponse)
    async def chat(
        req: ChatRequest,
        request: FastAPIRequest,
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> ChatResponse:
        del _auth
        context_token = None
        request_user_id: str | None = None
        try:
            request_user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        except HTTPException:
            request_user_id = None
        auth_context = _build_request_bisque_auth_context(
            bisque_auth=bisque_auth,
            request=request,
        )
        if auth_context is not None:
            context_token = set_request_bisque_auth(auth_context)
        try:
            final_response: ChatResponse | None = None
            active_run, hitl_resume = _resolve_chat_run_context(req, user_id=request_user_id)
            async for event in _run_chat_events(
                req,
                user_id=request_user_id,
                run=active_run,
                bisque_auth=bisque_auth,
                hitl_resume=hitl_resume,
            ):
                if event.get("event") != "done":
                    continue
                payload = event.get("data")
                if isinstance(payload, ChatResponse):
                    final_response = payload
                elif isinstance(payload, dict):
                    final_response = ChatResponse.model_validate(payload)
            if final_response is None:
                raise HTTPException(status_code=500, detail="chat failed: no completion payload")
            return final_response
        finally:
            if context_token is not None:
                reset_request_bisque_auth(context_token)

    @v1.post("/chat/stream")
    @legacy.post("/chat/stream")
    async def chat_stream(
        req: ChatRequest,
        request: FastAPIRequest,
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> StreamingResponse:
        del _auth
        if not req.messages:
            raise HTTPException(status_code=400, detail="messages cannot be empty")
        request_user_id: str | None = None
        try:
            request_user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        except HTTPException:
            request_user_id = None
        run, hitl_resume = _resolve_chat_run_context(req, user_id=request_user_id)

        async def _event_generator() -> AsyncIterator[str]:
            auth_context = _build_request_bisque_auth_context(
                bisque_auth=bisque_auth,
                request=request,
            )
            context_token = set_request_bisque_auth(auth_context) if auth_context else None
            try:
                async for event in _run_chat_events(
                    req,
                    user_id=request_user_id,
                    run=run,
                    bisque_auth=bisque_auth,
                    hitl_resume=hitl_resume,
                ):
                    yield _sse_event(str(event.get("event") or "message"), event.get("data"))
            except HTTPException as exc:
                yield _sse_event(
                    "error",
                    {"status_code": exc.status_code, "detail": exc.detail},
                )
            except Exception as exc:
                yield _sse_event(
                    "error",
                    {"status_code": 500, "detail": str(exc)},
                )
            finally:
                if context_token is not None:
                    reset_request_bisque_auth(context_token)

        stream_response = StreamingResponse(
            _event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "X-Run-Id": run.run_id,
                "X-Model": agno_runtime.model,
            },
        )
        if bisque_auth and bool(bisque_auth.get("anonymous_session")):
            anonymous_session_id = str(bisque_auth.get("session_id") or "").strip().lower()
            if _is_hex_session_token(anonymous_session_id):
                stream_response.set_cookie(
                    key=anonymous_session_cookie_name,
                    value=anonymous_session_id,
                    max_age=anonymous_session_ttl_seconds,
                    httponly=True,
                    secure=_should_secure_cookie(request),
                    samesite="lax",
                    path="/",
                )
        return stream_response

    @v1.post("/runs", response_model=CreateRunResponse)
    @legacy.post("/runs", response_model=CreateRunResponse)
    async def create_run(
        req: CreateRunRequest,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> CreateRunResponse:
        del _auth
        request_user_id = _resolve_request_user_id(bisque_auth)
        request_hash = _request_hash(req.model_dump())
        endpoint = f"/v1/runs:{request_user_id or 'anonymous'}"
        idem_key = (idempotency_key or "").strip()
        if idem_key:
            existing = store.get_idempotency(endpoint, idem_key)
            if existing:
                if existing["request_hash"] != request_hash:
                    raise HTTPException(
                        status_code=409,
                        detail=("Idempotency-Key already used with a different request payload."),
                    )
                existing_run = store.get_run(existing["run_id"])
                existing_status = (
                    existing_run.status.value
                    if existing_run is not None
                    else RunStatus.PENDING.value
                )
                return CreateRunResponse(run_id=existing["run_id"], status=existing_status)

        plan = WorkflowPlan.from_dict(req.plan)
        if not plan.goal:
            plan = WorkflowPlan(goal=req.goal, steps=plan.steps)

        run = WorkflowRun.new(
            goal=req.goal,
            plan=plan,
            workflow_kind="workflow_plan",
            mode="durable",
        )
        store.create_run(run)
        store.set_run_metadata(
            run.run_id,
            user_id=request_user_id,
            conversation_id=None,
            workflow_kind="workflow_plan",
            mode="durable",
        )
        if idem_key:
            store.put_idempotency(endpoint, idem_key, request_hash, run.run_id)
        _ensure_run_artifact_dir(run.run_id)
        store.append_event(run.run_id, "run_created", {"goal": req.goal})

        _enqueue_plan_inline_thread(
            run_id=run.run_id,
            plan=plan,
            thread_name=f"workflow-run-{run.run_id[:8]}",
        )
        store.append_event(run.run_id, "enqueued", {"queue": "inline-thread"})

        latest = store.get_run(run.run_id)
        latest_status = latest.status.value if latest is not None else run.status.value
        return CreateRunResponse(run_id=run.run_id, status=latest_status)

    @v1.get("/runs/{run_id}", response_model=RunResponse)
    @legacy.get("/runs/{run_id}", response_model=RunResponse)
    def get_run(
        run_id: str,
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> RunResponse:
        del _auth
        run = store.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        request_user_id = _resolve_request_user_id(bisque_auth)
        _assert_run_owner_access(run_id=run_id, request_user_id=request_user_id)
        return RunResponse(
            run_id=run.run_id,
            goal=run.goal,
            status=run.status.value,
            created_at=run.created_at,
            updated_at=run.updated_at,
            error=run.error,
            workflow_kind=run.workflow_kind,
            mode=run.mode,
            parent_run_id=run.parent_run_id,
            planner_version=run.planner_version,
            agent_role=run.agent_role,
            checkpoint_state=run.checkpoint_state,
            budget_state=run.budget_state,
            trace_group_id=run.trace_group_id,
        )

    @v1.get("/runs/{run_id}/result", response_model=RunResultResponse)
    @legacy.get("/runs/{run_id}/result", response_model=RunResultResponse)
    def get_run_result(
        run_id: str,
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> RunResultResponse:
        del _auth
        run = store.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")

        request_user_id = _resolve_request_user_id(bisque_auth)
        _assert_run_owner_access(run_id=run_id, request_user_id=request_user_id)

        result_payload: ChatResponse | None = None
        done_event = store.get_latest_event(run_id, "chat_done_payload")
        if done_event:
            payload = done_event.get("payload") if isinstance(done_event, dict) else None
            response_payload = payload.get("response") if isinstance(payload, dict) else None
            if isinstance(response_payload, dict):
                try:
                    result_payload = ChatResponse.model_validate(response_payload)
                except Exception:
                    result_payload = None

        status_value = run.status.value
        checkpoint_state = run.checkpoint_state if isinstance(run.checkpoint_state, dict) else {}
        checkpoint_phase = str(checkpoint_state.get("phase") or "").strip().lower()
        if (
            result_payload is not None
            and checkpoint_phase != "pending_approval"
            and status_value
            in {
                RunStatus.PENDING.value,
                RunStatus.RUNNING.value,
            }
        ):
            status_value = RunStatus.SUCCEEDED.value

        return RunResultResponse(
            run_id=run.run_id,
            status=status_value,  # type: ignore[arg-type]
            result=result_payload,
        )

    @v1.get("/runs/{run_id}/events", response_model=RunEventsResponse)
    @legacy.get("/runs/{run_id}/events", response_model=RunEventsResponse)
    def get_events(
        run_id: str,
        limit: int = 200,
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> RunEventsResponse:
        del _auth
        run = store.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        request_user_id = _resolve_request_user_id(bisque_auth)
        _assert_run_owner_access(run_id=run_id, request_user_id=request_user_id)
        return RunEventsResponse(run_id=run_id, events=store.list_events(run_id, limit=limit))

    @v1.get("/history/analyses", response_model=AnalysisHistoryResponse)
    @legacy.get("/history/analyses", response_model=AnalysisHistoryResponse)
    def history_analyses(
        limit: int = Query(default=5, ge=1, le=50),
        q: str | None = Query(default=None),
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> AnalysisHistoryResponse:
        del _auth
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        query_terms = _extract_history_search_terms(str(q or ""), max_terms=8) if q else []
        analyses = _build_analysis_summaries(
            user_id=user_id,
            limit=limit,
            query_text=str(q or ""),
            query_terms=query_terms,
        )
        return AnalysisHistoryResponse(count=len(analyses), analyses=analyses)

    @v1.post("/evals/contracts", response_model=ContractAuditResponse)
    @legacy.post("/evals/contracts", response_model=ContractAuditResponse)
    def evaluate_contract_health(
        req: ContractAuditRequest,
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> ContractAuditResponse:
        del _auth
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        requested_run_ids = [
            str(run_id or "").strip() for run_id in req.run_ids if str(run_id or "").strip()
        ]
        if requested_run_ids:
            run_ids = requested_run_ids
        else:
            recent_runs = store.list_runs_for_user(user_id=user_id, limit=int(req.limit))
            run_ids = [
                str(row.get("run_id") or "").strip()
                for row in recent_runs
                if str(row.get("run_id") or "").strip()
            ]

        seen: set[str] = set()
        records: list[ContractAuditRecord] = []
        for run_id in run_ids:
            if run_id in seen:
                continue
            seen.add(run_id)
            run = store.get_run(run_id)
            if not run:
                continue
            try:
                _assert_run_owner_access(run_id=run_id, request_user_id=user_id)
            except HTTPException:
                continue
            done_event = store.get_latest_event(run_id, "chat_done_payload")
            payload = done_event.get("payload") if isinstance(done_event, dict) else None
            response_payload = payload.get("response") if isinstance(payload, dict) else None
            response_dict = response_payload if isinstance(response_payload, dict) else {}

            contract_audit = audit_contract_payload(response_dict)
            research_score = score_research_value(response_dict)
            recommendations = list(research_score.get("recommendations") or [])
            missing_fields = list(contract_audit.get("missing_fields") or [])
            if missing_fields:
                recommendations.insert(
                    0,
                    "Fix missing contract fields: " + ", ".join(missing_fields),
                )
            deduped_recommendations: list[str] = []
            seen_recommendations: set[str] = set()
            for recommendation in recommendations:
                text = str(recommendation or "").strip()
                if not text or text in seen_recommendations:
                    continue
                seen_recommendations.add(text)
                deduped_recommendations.append(text)

            records.append(
                ContractAuditRecord(
                    run_id=run_id,
                    status=run.status.value,
                    passed=bool(contract_audit.get("passed")),
                    checks=dict(contract_audit.get("checks") or {}),
                    missing_fields=missing_fields,
                    evidence_count=int(contract_audit.get("evidence_count") or 0),
                    measurement_count=int(contract_audit.get("measurement_count") or 0),
                    limitation_count=int(contract_audit.get("limitation_count") or 0),
                    next_step_count=int(contract_audit.get("next_step_count") or 0),
                    confidence_level=(
                        str(contract_audit.get("confidence_level") or "").strip() or None
                    ),
                    confidence_why_count=int(contract_audit.get("confidence_why_count") or 0),
                    research_score=int(research_score.get("score") or 0),
                    research_max_score=int(research_score.get("max_score") or 0),
                    research_summary=str(research_score.get("summary") or "").strip(),
                    recommendations=deduped_recommendations,
                )
            )

        passed = sum(1 for record in records if bool(record.passed))
        failed = len(records) - passed
        average_research_score = (
            round(sum(int(record.research_score) for record in records) / float(len(records)), 3)
            if records
            else 0.0
        )
        return ContractAuditResponse(
            count=len(records),
            passed=passed,
            failed=failed,
            average_research_score=average_research_score,
            records=records,
        )

    def _parse_datetime(value: Any) -> datetime:
        raw = str(value or "").strip()
        if not raw:
            return datetime.utcnow()
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except Exception:
            return datetime.utcnow()

    def _to_training_domain_record(row: dict[str, Any]) -> TrainingDomainRecord:
        scope_token = str(row.get("owner_scope") or "shared").strip().lower()
        if scope_token not in {"shared", "private"}:
            scope_token = "shared"
        return TrainingDomainRecord(
            domain_id=str(row.get("domain_id") or ""),
            name=str(row.get("name") or ""),
            description=str(row.get("description") or "").strip() or None,
            owner_scope=scope_token,  # type: ignore[arg-type]
            owner_user_id=str(row.get("owner_user_id") or ""),
            metadata=row.get("metadata") if isinstance(row.get("metadata"), dict) else {},
            created_at=_parse_datetime(row.get("created_at")),
            updated_at=_parse_datetime(row.get("updated_at")),
        )

    def _to_training_lineage_record(row: dict[str, Any]) -> TrainingLineageRecord:
        scope_token = str(row.get("scope") or "shared").strip().lower()
        if scope_token not in {"shared", "fork"}:
            scope_token = "shared"
        return TrainingLineageRecord(
            lineage_id=str(row.get("lineage_id") or ""),
            domain_id=str(row.get("domain_id") or ""),
            scope=scope_token,  # type: ignore[arg-type]
            owner_user_id=str(row.get("owner_user_id") or ""),
            model_key=str(row.get("model_key") or "").strip().lower(),
            parent_lineage_id=str(row.get("parent_lineage_id") or "").strip() or None,
            active_version_id=str(row.get("active_version_id") or "").strip() or None,
            metadata=row.get("metadata") if isinstance(row.get("metadata"), dict) else {},
            created_at=_parse_datetime(row.get("created_at")),
            updated_at=_parse_datetime(row.get("updated_at")),
        )

    def _to_training_model_version_record(row: dict[str, Any]) -> TrainingModelVersionRecord:
        status_token = normalize_version_status(row.get("status"))
        return TrainingModelVersionRecord(
            version_id=str(row.get("version_id") or ""),
            lineage_id=str(row.get("lineage_id") or ""),
            source_job_id=str(row.get("source_job_id") or "").strip() or None,
            artifact_run_id=str(row.get("artifact_run_id") or "").strip() or None,
            status=status_token,  # type: ignore[arg-type]
            metrics=row.get("metrics") if isinstance(row.get("metrics"), dict) else {},
            metadata=row.get("metadata") if isinstance(row.get("metadata"), dict) else {},
            created_at=_parse_datetime(row.get("created_at")),
            updated_at=_parse_datetime(row.get("updated_at")),
        )

    def _to_training_update_proposal_record(row: dict[str, Any]) -> TrainingUpdateProposalRecord:
        return TrainingUpdateProposalRecord(
            proposal_id=str(row.get("proposal_id") or ""),
            lineage_id=str(row.get("lineage_id") or ""),
            trigger_reason=str(row.get("trigger_reason") or "").strip() or "manual",
            trigger_snapshot=(
                row.get("trigger_snapshot") if isinstance(row.get("trigger_snapshot"), dict) else {}
            ),
            dataset_snapshot=(
                row.get("dataset_snapshot") if isinstance(row.get("dataset_snapshot"), dict) else {}
            ),
            config=row.get("config") if isinstance(row.get("config"), dict) else {},
            status=normalize_proposal_status(row.get("status")),  # type: ignore[arg-type]
            idempotency_key=str(row.get("idempotency_key") or "").strip() or None,
            approved_by=str(row.get("approved_by") or "").strip() or None,
            rejected_by=str(row.get("rejected_by") or "").strip() or None,
            linked_job_id=str(row.get("linked_job_id") or "").strip() or None,
            candidate_version_id=str(row.get("candidate_version_id") or "").strip() or None,
            error=str(row.get("error") or "").strip() or None,
            created_at=_parse_datetime(row.get("created_at")),
            updated_at=_parse_datetime(row.get("updated_at")),
            approved_at=(
                _parse_datetime(row.get("approved_at"))
                if str(row.get("approved_at") or "").strip()
                else None
            ),
            rejected_at=(
                _parse_datetime(row.get("rejected_at"))
                if str(row.get("rejected_at") or "").strip()
                else None
            ),
            started_at=(
                _parse_datetime(row.get("started_at"))
                if str(row.get("started_at") or "").strip()
                else None
            ),
            finished_at=(
                _parse_datetime(row.get("finished_at"))
                if str(row.get("finished_at") or "").strip()
                else None
            ),
        )

    def _to_training_merge_request_record(row: dict[str, Any]) -> TrainingMergeRequestRecord:
        return TrainingMergeRequestRecord(
            merge_id=str(row.get("merge_id") or ""),
            source_lineage_id=str(row.get("source_lineage_id") or ""),
            target_lineage_id=str(row.get("target_lineage_id") or ""),
            candidate_version_id=str(row.get("candidate_version_id") or ""),
            requested_by=str(row.get("requested_by") or ""),
            status=normalize_merge_status(row.get("status")),  # type: ignore[arg-type]
            decision_by=str(row.get("decision_by") or "").strip() or None,
            notes=str(row.get("notes") or "").strip() or None,
            evaluation=row.get("evaluation") if isinstance(row.get("evaluation"), dict) else {},
            linked_proposal_id=str(row.get("linked_proposal_id") or "").strip() or None,
            error=str(row.get("error") or "").strip() or None,
            created_at=_parse_datetime(row.get("created_at")),
            updated_at=_parse_datetime(row.get("updated_at")),
            decided_at=(
                _parse_datetime(row.get("decided_at"))
                if str(row.get("decided_at") or "").strip()
                else None
            ),
            executed_at=(
                _parse_datetime(row.get("executed_at"))
                if str(row.get("executed_at") or "").strip()
                else None
            ),
        )

    def _assert_training_domain_access(
        *, row: dict[str, Any] | None, user_id: str
    ) -> dict[str, Any]:
        if row is None:
            raise HTTPException(status_code=404, detail="Training domain not found.")
        owner_scope = str(row.get("owner_scope") or "").strip().lower()
        owner_user = str(row.get("owner_user_id") or "").strip()
        if owner_scope != "shared" and owner_user != user_id:
            raise HTTPException(status_code=404, detail="Training domain not found.")
        return row

    def _assert_training_lineage_access(
        *, row: dict[str, Any] | None, user_id: str
    ) -> dict[str, Any]:
        if row is None:
            raise HTTPException(status_code=404, detail="Training lineage not found.")
        scope_token = str(row.get("scope") or "").strip().lower()
        owner_user = str(row.get("owner_user_id") or "").strip()
        if scope_token != "shared" and owner_user != user_id:
            raise HTTPException(status_code=404, detail="Training lineage not found.")
        return row

    def _assert_training_lineage_mutation_access(
        *,
        row: dict[str, Any] | None,
        user_id: str,
        action: str,
    ) -> dict[str, Any]:
        lineage = _assert_training_lineage_access(row=row, user_id=user_id)
        scope_token = str(lineage.get("scope") or "").strip().lower()
        if scope_token != "shared":
            return lineage
        owner_user = str(lineage.get("owner_user_id") or "").strip()
        if owner_user == user_id:
            return lineage
        raise HTTPException(
            status_code=403,
            detail=(
                f"{action} is restricted to the shared-lineage owner. "
                "Ask the domain maintainer to perform this action."
            ),
        )

    def _seed_lineage_replay_items(
        *,
        lineage_row: dict[str, Any] | None,
        replace: bool = False,
    ) -> None:
        lineage = lineage_row if isinstance(lineage_row, dict) else {}
        lineage_id = str(lineage.get("lineage_id") or "").strip()
        model_token = str(lineage.get("model_key") or "").strip().lower()
        if not lineage_id or model_token != prairie_model_key:
            return
        if not replace:
            existing = store.list_training_replay_items(lineage_id=lineage_id, limit=1)
            if existing:
                return
        definition = get_model_definition(model_token)
        if definition is None:
            return
        config = _normalize_training_config(
            model_key=model_token,
            raw_config={},
            default_config=dict(definition.default_config),
        )
        adapter = training_runner.get_adapter(model_token)
        load_spec = getattr(adapter, "_load_canonical_benchmark_spec", None)
        collect_pool = getattr(adapter, "_collect_canonical_replay_pool", None)
        if not callable(load_spec) or not callable(collect_pool):
            return
        try:
            spec = load_spec(config=config)
            replay_pool = collect_pool(spec=spec)
        except Exception:
            logger.exception("Failed to seed replay items for lineage %s", lineage_id)
            return
        now = datetime.utcnow().isoformat()
        replay_items: list[dict[str, Any]] = []
        for row in replay_pool:
            sample_id = str(row.get("sample_id") or "").strip()
            image_path = str(row.get("image_path") or "").strip()
            if not sample_id or not image_path:
                continue
            class_tags = [
                str(tag).strip() for tag in list(row.get("class_tags") or []) if str(tag).strip()
            ]
            small_object_count = max(0, int(row.get("small_object_count") or 0))
            replay_items.append(
                {
                    "file_id": f"canonical:{sample_id}",
                    "sample_id": sample_id,
                    "weight": float(max(1, small_object_count + 1)),
                    "class_tag": ",".join(class_tags[:3]) or None,
                    "pinned": "prairie_dog" in class_tags,
                    "last_seen_at": now,
                }
            )
        if replay_items:
            store.upsert_training_replay_items(
                lineage_id=lineage_id,
                items=replay_items,
                replace=replace,
            )

    def _ensure_default_domain_and_lineage(
        *,
        user_id: str,
        model_key: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        model_token = str(model_key or "").strip().lower() or prairie_model_key
        domain_id = f"domain-{model_token}-default"
        domain_row = store.get_training_domain(domain_id=domain_id)
        if domain_row is None:
            domain_row = store.create_training_domain(
                domain_id=domain_id,
                name=f"{model_token.upper()} Shared Domain",
                description="Default shared domain lineage.",
                owner_scope="shared",
                owner_user_id=user_id,
                metadata={"model_key": model_token, "default": True},
            )
        lineages = store.list_training_lineages(
            domain_id=domain_id,
            model_key=model_token,
            include_shared=True,
            limit=50,
        )
        shared = next(
            (row for row in lineages if str(row.get("scope") or "").strip().lower() == "shared"),
            None,
        )
        if shared is None:
            lineage_id = f"lineage-{model_token}-shared"
            shared = store.create_training_lineage(
                lineage_id=lineage_id,
                domain_id=domain_id,
                scope="shared",
                owner_user_id=user_id,
                model_key=model_token,
                metadata={"default": True},
            )
        _seed_lineage_replay_items(lineage_row=shared)
        return domain_row, shared

    def _lineage_active_version_row(lineage_row: dict[str, Any]) -> dict[str, Any] | None:
        version_id = str(lineage_row.get("active_version_id") or "").strip()
        if not version_id:
            return None
        return store.get_training_model_version(version_id=version_id)

    def _to_training_job_record(row: dict[str, Any]) -> TrainingJobRecord:
        status_token = str(row.get("status") or "").strip().lower()
        if status_token not in {"queued", "running", "paused", "succeeded", "failed", "canceled"}:
            status_token = "failed"
        job_type = str(row.get("job_type") or "training").strip().lower()
        if job_type not in {"training", "inference"}:
            job_type = "training"
        return TrainingJobRecord(
            job_id=str(row.get("job_id") or ""),
            user_id=str(row.get("user_id") or ""),
            job_type=job_type,  # type: ignore[arg-type]
            dataset_id=str(row.get("dataset_id") or "").strip() or None,
            model_key=str(row.get("model_key") or "").strip().lower(),
            model_version=str(row.get("model_version") or "").strip() or None,
            status=status_token,  # type: ignore[arg-type]
            artifact_run_id=str(row.get("artifact_run_id") or "").strip() or None,
            error=str(row.get("error") or "").strip() or None,
            request=row.get("request") if isinstance(row.get("request"), dict) else {},
            result=row.get("result") if isinstance(row.get("result"), dict) else {},
            control=row.get("control") if isinstance(row.get("control"), dict) else {},
            created_at=_parse_datetime(row.get("created_at")),
            updated_at=_parse_datetime(row.get("updated_at")),
            started_at=(
                _parse_datetime(row.get("started_at"))
                if str(row.get("started_at") or "").strip()
                else None
            ),
            finished_at=(
                _parse_datetime(row.get("finished_at"))
                if str(row.get("finished_at") or "").strip()
                else None
            ),
            last_heartbeat_at=(
                _parse_datetime(row.get("last_heartbeat_at"))
                if str(row.get("last_heartbeat_at") or "").strip()
                else None
            ),
        )

    def _dataset_record_from_rows(
        *,
        dataset_row: dict[str, Any],
        item_rows: list[dict[str, Any]],
        manifest: dict[str, Any] | None = None,
    ) -> TrainingDatasetRecord:
        split_counts_raw = ((manifest or {}).get("counts") or {}) if manifest else {}
        split_counts = {
            split: int((split_counts_raw.get(split) or {}).get("samples") or 0)
            for split in ("train", "val", "test")
        }
        return TrainingDatasetRecord(
            dataset_id=str(dataset_row.get("dataset_id") or ""),
            user_id=str(dataset_row.get("user_id") or ""),
            name=str(dataset_row.get("name") or ""),
            description=str(dataset_row.get("description") or "").strip() or None,
            item_count=len(item_rows),
            split_counts=split_counts,
            created_at=_parse_datetime(dataset_row.get("created_at")),
            updated_at=_parse_datetime(dataset_row.get("updated_at")),
        )

    def _safe_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
        try:
            parsed = int(value)
        except Exception:
            parsed = default
        return max(minimum, min(maximum, parsed))

    def _safe_float(value: Any, *, default: float, minimum: float, maximum: float) -> float:
        try:
            parsed = float(value)
        except Exception:
            parsed = default
        if parsed < minimum:
            return minimum
        if parsed > maximum:
            return maximum
        return parsed

    def _normalize_training_config(
        *,
        model_key: str,
        raw_config: dict[str, Any],
        default_config: dict[str, Any],
    ) -> dict[str, Any]:
        model_token = str(model_key or "").strip().lower()
        if model_token == "yolov5_rarespot":
            # Prairie active learning uses a fixed server-side training profile.
            fixed_epochs = getattr(settings, "prairie_fixed_epochs", None)
            fixed_batch_size = getattr(settings, "prairie_fixed_batch_size", None)
            fixed_imgsz = getattr(settings, "prairie_fixed_imgsz", None)
            fixed_tile_size = getattr(settings, "prairie_fixed_tile_size", None)
            fixed_train_tile_overlap = getattr(settings, "prairie_training_tile_overlap", None)
            fixed_infer_tile_overlap = getattr(settings, "prairie_inference_tile_overlap", None)
            fixed_include_empty_tiles = getattr(settings, "prairie_include_empty_tiles", None)
            fixed_merge_iou = getattr(settings, "prairie_inference_merge_iou_threshold", None)
            fixed_min_box_pixels = getattr(settings, "prairie_min_box_pixels", None)
            fixed_conf = getattr(settings, "prairie_fixed_conf_threshold", None)
            fixed_iou = getattr(settings, "prairie_fixed_iou_threshold", None)
            benchmark_spec_path = getattr(settings, "prairie_canonical_benchmark_spec_path", None)
            conservative_patience = getattr(settings, "prairie_conservative_patience", None)
            conservative_freeze_layers = getattr(
                settings, "prairie_conservative_freeze_layers", None
            )
            small_dataset_object_threshold = getattr(
                settings, "prairie_small_dataset_object_threshold", None
            )
            small_dataset_epochs = getattr(settings, "prairie_small_dataset_epochs", None)
            guardrail_canonical_drop = getattr(
                settings, "prairie_guardrail_canonical_map50_drop_max", None
            )
            guardrail_prairie_drop = getattr(
                settings, "prairie_guardrail_prairie_dog_map50_drop_max", None
            )
            guardrail_active_drop = getattr(
                settings, "prairie_guardrail_active_map50_drop_max", None
            )
            guardrail_fp_increase = getattr(
                settings, "prairie_guardrail_canonical_fp_image_increase_max", None
            )
            hard_sample_bank_enabled = getattr(settings, "prairie_enable_hard_sample_bank", None)
            hard_sample_ratio = getattr(settings, "prairie_hard_sample_injection_ratio", None)
            small_object_weighting_enabled = getattr(
                settings, "prairie_enable_small_object_weighting", None
            )
            replay_new_ratio = getattr(settings, "prairie_replay_new_ratio", None)
            replay_old_ratio = getattr(settings, "prairie_replay_old_ratio", None)
            prefer_sahi = getattr(settings, "prairie_inference_prefer_sahi", None)
            hyp_path_default = (
                Path(__file__).resolve().parent.parent / "training" / "hyp.prairie_finetune.yaml"
            )
            return {
                "epochs": _safe_int(
                    fixed_epochs if fixed_epochs is not None else default_config.get("epochs"),
                    default=int(default_config.get("epochs") or 20),
                    minimum=1,
                    maximum=500,
                ),
                "batch_size": _safe_int(
                    (
                        fixed_batch_size
                        if fixed_batch_size is not None
                        else default_config.get("batch_size")
                    ),
                    default=int(default_config.get("batch_size") or 4),
                    minimum=1,
                    maximum=128,
                ),
                "imgsz": _safe_int(
                    fixed_imgsz if fixed_imgsz is not None else default_config.get("imgsz"),
                    default=int(default_config.get("imgsz") or 512),
                    minimum=128,
                    maximum=4096,
                ),
                "tile_size": _safe_int(
                    fixed_tile_size
                    if fixed_tile_size is not None
                    else default_config.get("tile_size"),
                    default=int(default_config.get("tile_size") or 512),
                    minimum=128,
                    maximum=4096,
                ),
                "train_tile_overlap": _safe_float(
                    (
                        fixed_train_tile_overlap
                        if fixed_train_tile_overlap is not None
                        else default_config.get("train_tile_overlap")
                    ),
                    default=float(default_config.get("train_tile_overlap") or 0.0),
                    minimum=0.0,
                    maximum=0.95,
                ),
                "tile_overlap": _safe_float(
                    (
                        fixed_infer_tile_overlap
                        if fixed_infer_tile_overlap is not None
                        else default_config.get("tile_overlap")
                    ),
                    default=float(default_config.get("tile_overlap") or 0.25),
                    minimum=0.0,
                    maximum=0.95,
                ),
                "include_empty_tiles": bool(
                    fixed_include_empty_tiles
                    if fixed_include_empty_tiles is not None
                    else bool(default_config.get("include_empty_tiles", True))
                ),
                "merge_iou": _safe_float(
                    fixed_merge_iou
                    if fixed_merge_iou is not None
                    else default_config.get("merge_iou"),
                    default=float(default_config.get("merge_iou") or 0.45),
                    minimum=0.05,
                    maximum=0.99,
                ),
                "min_box_pixels": _safe_float(
                    fixed_min_box_pixels
                    if fixed_min_box_pixels is not None
                    else default_config.get("min_box_pixels"),
                    default=float(default_config.get("min_box_pixels") or 4.0),
                    minimum=1.0,
                    maximum=64.0,
                ),
                "conf": _safe_float(
                    fixed_conf if fixed_conf is not None else default_config.get("conf"),
                    default=float(default_config.get("conf") or 0.25),
                    minimum=0.001,
                    maximum=0.99,
                ),
                "iou": _safe_float(
                    fixed_iou if fixed_iou is not None else default_config.get("iou"),
                    default=float(default_config.get("iou") or 0.45),
                    minimum=0.05,
                    maximum=0.99,
                ),
                "runtime_repo_path": str(
                    getattr(settings, "yolov5_runtime_path", "third_party/yolov5")
                ),
                "weights_path": str(
                    getattr(settings, "prairie_rarespot_weights_path", "RareSpotWeights.pt")
                ),
                "canonical_benchmark_spec_path": str(
                    benchmark_spec_path or "benchmark/canonical_rare_spot.yaml"
                ),
                "hyp_path": str(hyp_path_default.resolve()),
                "patience": _safe_int(
                    conservative_patience if conservative_patience is not None else 3,
                    default=3,
                    minimum=1,
                    maximum=100,
                ),
                "freeze_layers": _safe_int(
                    conservative_freeze_layers if conservative_freeze_layers is not None else 10,
                    default=10,
                    minimum=0,
                    maximum=64,
                ),
                "small_dataset_object_threshold": _safe_int(
                    small_dataset_object_threshold
                    if small_dataset_object_threshold is not None
                    else 300,
                    default=300,
                    minimum=1,
                    maximum=1_000_000,
                ),
                "small_dataset_epochs": _safe_int(
                    small_dataset_epochs if small_dataset_epochs is not None else 6,
                    default=6,
                    minimum=1,
                    maximum=300,
                ),
                "empty_tile_ratio": 1.0,
                "guardrail_canonical_map50_drop_max": _safe_float(
                    guardrail_canonical_drop if guardrail_canonical_drop is not None else 0.02,
                    default=0.02,
                    minimum=0.0,
                    maximum=1.0,
                ),
                "guardrail_prairie_dog_map50_drop_max": _safe_float(
                    guardrail_prairie_drop if guardrail_prairie_drop is not None else 0.03,
                    default=0.03,
                    minimum=0.0,
                    maximum=1.0,
                ),
                "guardrail_active_map50_drop_max": _safe_float(
                    guardrail_active_drop if guardrail_active_drop is not None else 0.02,
                    default=0.02,
                    minimum=0.0,
                    maximum=1.0,
                ),
                "guardrail_canonical_fp_image_increase_max": _safe_float(
                    guardrail_fp_increase if guardrail_fp_increase is not None else 0.25,
                    default=0.25,
                    minimum=0.0,
                    maximum=10.0,
                ),
                "enable_hard_sample_bank": bool(hard_sample_bank_enabled),
                "hard_sample_injection_ratio": _safe_float(
                    hard_sample_ratio if hard_sample_ratio is not None else 0.2,
                    default=0.2,
                    minimum=0.0,
                    maximum=1.0,
                ),
                "enable_small_object_weighting": bool(small_object_weighting_enabled),
                "replay_new_ratio": _safe_float(
                    replay_new_ratio if replay_new_ratio is not None else 0.6,
                    default=0.6,
                    minimum=0.01,
                    maximum=0.99,
                ),
                "replay_old_ratio": _safe_float(
                    replay_old_ratio if replay_old_ratio is not None else 0.4,
                    default=0.4,
                    minimum=0.0,
                    maximum=0.99,
                ),
                "enable_sahi": bool(prefer_sahi if prefer_sahi is not None else True),
                "execution_backend": "yolov5",
            }
        merged = {**default_config, **(raw_config or {})}
        normalized: dict[str, Any] = dict(merged)
        normalized["epochs"] = _safe_int(
            merged.get("epochs"),
            default=int(default_config.get("epochs") or 10),
            minimum=1,
            maximum=1_000,
        )
        normalized["batch_size"] = _safe_int(
            merged.get("batch_size"),
            default=int(default_config.get("batch_size") or 1),
            minimum=1,
            maximum=128,
        )
        normalized["learning_rate"] = _safe_float(
            merged.get("learning_rate"),
            default=float(default_config.get("learning_rate") or 1e-3),
            minimum=1e-8,
            maximum=1.0,
        )
        normalized["spatial_dims"] = normalize_spatial_dims(
            merged.get("spatial_dims"),
            default=str(default_config.get("spatial_dims") or "2d"),
        )
        normalized["num_classes"] = _safe_int(
            merged.get("num_classes"),
            default=int(default_config.get("num_classes") or 2),
            minimum=2,
            maximum=256,
        )
        backend = str(merged.get("execution_backend") or "auto").strip().lower()
        if backend not in {"auto", "simulated", "monai"}:
            backend = "auto"
        normalized["execution_backend"] = backend
        if model_token == "medsam":
            normalized["finetune"] = False
        return normalized

    def _training_control_transition_allowed(status: str, action: str) -> bool:
        status_token = str(status or "").strip().lower()
        action_token = str(action or "").strip().lower()
        if action_token == "pause":
            return status_token == "running"
        if action_token == "resume":
            return status_token == "paused"
        if action_token == "cancel":
            return status_token in {"queued", "running", "paused"}
        if action_token == "restart":
            return status_token in {"failed", "canceled", "succeeded"}
        return False

    def _format_preflight_failure_detail(preflight: dict[str, Any]) -> dict[str, Any]:
        checks = preflight.get("checks") if isinstance(preflight.get("checks"), list) else []
        failed_checks = [
            row
            for row in checks
            if isinstance(row, dict) and str(row.get("status") or "").strip().lower() == "fail"
        ]
        fail_details = [
            str(row.get("detail") or "").strip()
            for row in failed_checks
            if str(row.get("detail") or "").strip()
        ]
        message = "Preflight checks failed."
        if fail_details:
            message = "Preflight checks failed: " + "; ".join(fail_details[:3])
        return {"message": message, "preflight": preflight}

    def _resolve_training_dataset_upload(
        *,
        user_id: str,
        file_id: str,
    ) -> dict[str, Any]:
        upload = store.get_upload(file_id, user_id=user_id)
        if not upload:
            raise HTTPException(status_code=404, detail=f"Upload not found for file_id={file_id}")
        local_path_value = _resolved_local_path_for_upload(upload, user_id=user_id)
        if not local_path_value:
            raise HTTPException(
                status_code=404,
                detail=f"Upload file path is missing for file_id={file_id}",
            )
        local_path = Path(local_path_value)
        return {
            "file_id": str(upload.get("file_id") or ""),
            "path": str(local_path),
            "original_name": str(upload.get("original_name") or local_path.name),
            "sha256": str(upload.get("sha256") or "").strip() or None,
            "size_bytes": int(upload.get("size_bytes") or 0),
            "metadata": upload.get("metadata") if isinstance(upload.get("metadata"), dict) else {},
        }

    def _build_training_dataset_manifest(
        *,
        dataset_id: str,
        user_id: str,
        model_key: str | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
        dataset_row = store.get_training_dataset(dataset_id=dataset_id, user_id=user_id)
        if dataset_row is None:
            raise HTTPException(status_code=404, detail="Training dataset not found.")
        item_rows = store.list_training_dataset_items(dataset_id=dataset_id, user_id=user_id)
        model_token = str(model_key or "").strip().lower()
        task_type = "detection" if model_token == "yolov5_rarespot" else "segmentation"
        try:
            manifest = build_dataset_manifest(
                dataset_id=dataset_id,
                dataset_name=str(dataset_row.get("name") or dataset_id),
                items=item_rows,
                task_type=task_type,
            )
        except DatasetValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return dataset_row, item_rows, manifest

    prairie_source_type = "prairie_active_learning_dataset"
    prairie_model_key = "yolov5_rarespot"
    prairie_supported_classes = ("burrow", "prairie_dog")
    prairie_image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    def _prairie_source_id(user_id: str) -> str:
        token = hashlib.sha1(str(user_id or "").encode("utf-8")).hexdigest()[:16]
        return f"prairie-source-{token}"

    def _build_bisque_request_headers(
        *,
        bisque_username: str | None,
        bisque_password: str | None,
        bisque_access_token: str | None,
        bisque_cookie_header: str | None,
    ) -> dict[str, str]:
        headers = {"Accept": "application/xml,text/xml,*/*"}
        token = str(bisque_access_token or "").strip()
        cookie_header = str(bisque_cookie_header or "").strip()
        username = str(bisque_username or "").strip()
        password = str(bisque_password or "").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        elif cookie_header:
            headers["Cookie"] = cookie_header
        elif username and password:
            creds = f"{username}:{password}".encode()
            headers["Authorization"] = f"Basic {b64encode(creds).decode('ascii')}"
        return headers

    def _fetch_xml_url(
        *, url: str, headers: dict[str, str], timeout: int = 60
    ) -> ElementTree.Element:
        request = Request(url, headers=headers, method="GET")
        with urlopen(request, timeout=timeout) as response:
            payload = response.read() or b""
        if not payload:
            raise ValueError(f"Empty XML response from {url}")
        return ElementTree.fromstring(payload)

    def _extract_dataset_members_from_xml(
        *,
        root_node: ElementTree.Element,
        dataset_name: str,
        bisque_root: str,
    ) -> tuple[str | None, list[str]]:
        def _collect_member_uris(dataset_node: ElementTree.Element) -> list[str]:
            dataset_self_uris = {
                _normalize_bisque_resource_uri(
                    str(dataset_node.attrib.get("uri") or "").strip(),
                    bisque_root,
                )
            }
            dataset_uniq = str(dataset_node.attrib.get("resource_uniq") or "").strip()
            if dataset_uniq:
                dataset_self_uris.add(f"{bisque_root.rstrip('/')}/data_service/{dataset_uniq}")
            resources: list[str] = []
            seen: set[str] = set()
            for node in dataset_node.iter():
                candidates = [
                    str(node.attrib.get("uri") or "").strip(),
                    str(node.attrib.get("value") or "").strip(),
                    str(node.text or "").strip(),
                ]
                for raw in candidates:
                    if not raw:
                        continue
                    if "/data_service/" not in raw and not raw.startswith("/image_service/"):
                        continue
                    try:
                        normalized = _normalize_bisque_resource_uri(raw, bisque_root)
                    except Exception:
                        continue
                    lowered = normalized.lower()
                    if "/data_service/dataset/" in lowered:
                        continue
                    if normalized in dataset_self_uris:
                        continue
                    if "/tag/" in lowered:
                        continue
                    if "/data_service/" not in lowered:
                        continue
                    if normalized in seen:
                        continue
                    seen.add(normalized)
                    resources.append(normalized)
            return resources

        dataset_nodes: list[ElementTree.Element] = []
        if str(root_node.tag or "").strip().lower() == "dataset":
            dataset_nodes.append(root_node)
        dataset_nodes.extend(list(root_node.findall(".//dataset")))
        if not dataset_nodes:
            raise ValueError("No <dataset> nodes were found in BisQue response.")

        desired_name = str(dataset_name or "").strip()
        matching_nodes = [
            candidate
            for candidate in dataset_nodes
            if str(candidate.attrib.get("name") or "").strip() == desired_name
        ]
        if not matching_nodes and len(dataset_nodes) == 1:
            matching_nodes = dataset_nodes[:]
        if not matching_nodes:
            available = sorted(
                {
                    str(node.attrib.get("name") or "").strip()
                    for node in dataset_nodes
                    if str(node.attrib.get("name") or "").strip()
                }
            )
            raise ValueError(
                f"Dataset '{desired_name}' not found in BisQue response. Available={available[:10]}"
            )

        selected: ElementTree.Element | None = None
        resources: list[str] = []
        candidate_records: list[tuple[int, str, ElementTree.Element, list[str]]] = []
        for candidate in matching_nodes:
            candidate_resources = _collect_member_uris(candidate)
            candidate_timestamp = str(
                candidate.attrib.get("ts") or candidate.attrib.get("created") or ""
            ).strip()
            candidate_records.append(
                (
                    1 if candidate_resources else 0,
                    candidate_timestamp,
                    candidate,
                    candidate_resources,
                )
            )
        if candidate_records:
            _, _, selected, resources = max(candidate_records, key=lambda item: (item[0], item[1]))

        dataset_uri = str(selected.attrib.get("uri") or "").strip() or None
        if not dataset_uri:
            dataset_uniq = str(selected.attrib.get("resource_uniq") or "").strip()
            if dataset_uniq:
                dataset_uri = f"{bisque_root.rstrip('/')}/data_service/dataset/{dataset_uniq}"
        return dataset_uri, resources

    def _fetch_prairie_dataset_members_remote(
        *,
        dataset_name: str,
        bisque_root: str,
        headers: dict[str, str],
    ) -> tuple[str | None, list[str]]:
        normalized_root = str(bisque_root or "").strip().rstrip("/")
        if not normalized_root:
            raise ValueError("BISQUE_ROOT is not configured.")
        query_candidates = [
            {"tag_query": f'"name":"{dataset_name}"', "view": "deep"},
            {"name": dataset_name, "view": "deep"},
            {"tag_query": f'"name":"{dataset_name}"'},
            {"view": "deep"},
        ]
        errors: list[str] = []
        for params in query_candidates:
            query = urlencode(params)
            candidate_url = f"{normalized_root}/data_service/dataset"
            if query:
                candidate_url = f"{candidate_url}?{query}"
            try:
                root_node = _fetch_xml_url(url=candidate_url, headers=headers, timeout=45)
                dataset_uri, members = _extract_dataset_members_from_xml(
                    root_node=root_node,
                    dataset_name=dataset_name,
                    bisque_root=normalized_root,
                )
                if members:
                    return dataset_uri, members
                errors.append(f"{candidate_url}: dataset found but no members.")
            except Exception as exc:
                errors.append(f"{candidate_url}: {exc}")
        raise ValueError(
            "Unable to discover BisQue dataset members for "
            f"'{dataset_name}'. Attempts={errors[-3:]}"
        )

    globals()["_extract_dataset_members_from_xml"] = _extract_dataset_members_from_xml

    def _fetch_resource_deep_xml_bytes(
        *,
        resource_uri: str,
        headers: dict[str, str],
    ) -> bytes:
        deep_uri = f"{resource_uri}{'&' if '?' in resource_uri else '?'}view=deep"
        request = Request(deep_uri, headers=headers, method="GET")
        with urlopen(request, timeout=45) as response:
            payload = response.read() or b""
        if not payload:
            raise ValueError(f"Empty deep XML payload for {resource_uri}")
        return payload

    def _is_prairie_image_candidate(
        *,
        original_name: str | None,
        content_type: str | None,
    ) -> bool:
        content_type_token = str(content_type or "").strip().lower()
        if content_type_token.startswith("image/"):
            return True
        suffix = Path(str(original_name or "").strip()).suffix.lower()
        return bool(suffix and suffix in prairie_image_suffixes)

    def _parse_prairie_annotation_counts(
        annotation_path: Path,
    ) -> tuple[bool, dict[str, int], dict[str, int], tuple[Any, ...], str | None]:
        supported_counts = {name: 0 for name in prairie_supported_classes}
        unsupported_counts: dict[str, int] = {}
        annotation_signature: list[tuple[str, tuple[tuple[tuple[float, float], ...], ...]]] = []
        try:
            root = ElementTree.parse(str(annotation_path)).getroot()
        except Exception as exc:
            return False, supported_counts, unsupported_counts, tuple(), f"XML parse failed: {exc}"

        layers = [
            node
            for node in root.findall("./gobject")
            if str(node.attrib.get("name") or "").strip() == "gt2"
        ]
        if not layers:
            return (
                False,
                supported_counts,
                unsupported_counts,
                tuple(),
                "Missing gt2 annotation layer.",
            )

        for layer in layers:
            for class_node in layer.findall("./gobject"):
                class_name = str(class_node.attrib.get("name") or "").strip()
                if not class_name:
                    continue
                rectangle_signatures: list[tuple[tuple[float, float], ...]] = []
                for rectangle in class_node.findall("./rectangle"):
                    vertices: list[tuple[float, float]] = []
                    for vertex in rectangle.findall("./vertex"):
                        try:
                            x = round(float(vertex.attrib.get("x", "0.0")), 3)
                            y = round(float(vertex.attrib.get("y", "0.0")), 3)
                        except Exception:
                            x = 0.0
                            y = 0.0
                        vertices.append((x, y))
                    rectangle_signatures.append(tuple(sorted(vertices)))
                rect_count = len(rectangle_signatures)
                if class_name in supported_counts:
                    supported_counts[class_name] = supported_counts[class_name] + int(rect_count)
                else:
                    unsupported_counts[class_name] = unsupported_counts.get(class_name, 0) + int(
                        rect_count
                    )
                annotation_signature.append(
                    (
                        class_name,
                        tuple(sorted(rectangle_signatures)),
                    )
                )
        return True, supported_counts, unsupported_counts, tuple(sorted(annotation_signature)), None

    def _prairie_reviewed_sample_fingerprint(sample: dict[str, Any]) -> tuple[Any, ...]:
        image_row = sample.get("image") if isinstance(sample.get("image"), dict) else {}
        supported_counts = (
            sample.get("supported_counts")
            if isinstance(sample.get("supported_counts"), dict)
            else {}
        )
        unsupported_counts = (
            sample.get("unsupported_counts")
            if isinstance(sample.get("unsupported_counts"), dict)
            else {}
        )
        annotation_signature = (
            sample.get("annotation_signature")
            if isinstance(sample.get("annotation_signature"), (list, tuple))
            else ()
        )
        return (
            str(image_row.get("sha256") or image_row.get("file_id") or "").strip(),
            tuple(annotation_signature),
            tuple(
                (class_name, int(supported_counts.get(class_name) or 0))
                for class_name in prairie_supported_classes
            ),
            tuple(
                sorted(
                    (
                        str(class_name).strip(),
                        int(value or 0),
                    )
                    for class_name, value in unsupported_counts.items()
                    if str(class_name).strip()
                )
            ),
        )

    def _prairie_reviewed_sample_identity(sample: dict[str, Any]) -> str:
        image_row = sample.get("image") if isinstance(sample.get("image"), dict) else {}
        annotation_row = (
            sample.get("annotation") if isinstance(sample.get("annotation"), dict) else {}
        )
        image_sha = str(image_row.get("sha256") or "").strip()
        annotation_sha = str(annotation_row.get("sha256") or "").strip()
        if image_sha:
            return f"image:{image_sha}"
        if annotation_sha:
            return f"annotation:{annotation_sha}"
        sample_id = str(sample.get("sample_id") or "").strip()
        if sample_id:
            return f"sample:{sample_id}"
        return ""

    def _finalize_prairie_reviewed_samples(
        reviewed_candidates: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int], list[str], dict[str, int]]:
        unique_samples: dict[str, dict[str, Any]] = {}
        identity_to_sample_id: dict[str, str] = {}
        conflicted_sample_ids: set[str] = set()
        conflicted_identities: set[str] = set()
        class_counts = {name: 0 for name in prairie_supported_classes}
        unsupported_class_counts: dict[str, int] = {}
        messages: list[str] = []
        duplicate_samples_ignored = 0
        conflicting_samples_skipped = 0

        def _forget_existing(existing_sample_id: str) -> None:
            existing_sample = unique_samples.pop(existing_sample_id, None)
            if existing_sample is None:
                return
            identity = _prairie_reviewed_sample_identity(existing_sample)
            if identity and identity_to_sample_id.get(identity) == existing_sample_id:
                identity_to_sample_id.pop(identity, None)

        for sample in reviewed_candidates:
            sample_id = str(sample.get("sample_id") or "").strip()
            if not sample_id:
                messages.append(
                    "Encountered a reviewed prairie sample without a sample_id; skipped."
                )
                continue
            if sample_id in conflicted_sample_ids:
                duplicate_samples_ignored += 1
                continue
            identity = _prairie_reviewed_sample_identity(sample)
            if identity and identity in conflicted_identities:
                duplicate_samples_ignored += 1
                continue
            existing = unique_samples.get(sample_id)
            if existing is None:
                matched_sample_id = identity_to_sample_id.get(identity) if identity else None
                if matched_sample_id:
                    matched = unique_samples.get(matched_sample_id)
                    if matched is not None and _prairie_reviewed_sample_fingerprint(
                        matched
                    ) == _prairie_reviewed_sample_fingerprint(sample):
                        duplicate_samples_ignored += 1
                        messages.append(
                            f"{sample_id}: duplicate reviewed sample content detected; ignored to preserve unique counting."
                        )
                        continue
                    conflicting_samples_skipped += 1
                    conflicted_identities.add(identity)
                    conflicted_sample_ids.add(sample_id)
                    conflicted_sample_ids.add(matched_sample_id)
                    _forget_existing(matched_sample_id)
                    messages.append(
                        f"{sample_id}: conflicting duplicate reviewed image content detected; excluded from counts and training manifest."
                    )
                    continue
                unique_samples[sample_id] = sample
                if identity:
                    identity_to_sample_id[identity] = sample_id
                continue
            if _prairie_reviewed_sample_fingerprint(
                existing
            ) == _prairie_reviewed_sample_fingerprint(sample):
                duplicate_samples_ignored += 1
                messages.append(
                    f"{sample_id}: duplicate reviewed sample detected; ignored to preserve unique counting."
                )
                continue
            conflicting_samples_skipped += 1
            conflicted_sample_ids.add(sample_id)
            if identity:
                conflicted_identities.add(identity)
            _forget_existing(sample_id)
            messages.append(
                f"{sample_id}: conflicting duplicate reviewed sample detected; excluded from counts and training manifest."
            )

        finalized = [
            unique_samples[sample_id]
            for sample_id in sorted(unique_samples.keys(), key=lambda value: value.lower())
        ]
        for sample in finalized:
            supported_counts = (
                sample.get("supported_counts")
                if isinstance(sample.get("supported_counts"), dict)
                else {}
            )
            unsupported_counts = (
                sample.get("unsupported_counts")
                if isinstance(sample.get("unsupported_counts"), dict)
                else {}
            )
            for class_name in prairie_supported_classes:
                class_counts[class_name] = class_counts.get(class_name, 0) + int(
                    supported_counts.get(class_name) or 0
                )
            for class_name, value in unsupported_counts.items():
                token = str(class_name or "").strip()
                if not token:
                    continue
                unsupported_class_counts[token] = unsupported_class_counts.get(token, 0) + int(
                    value or 0
                )

        return (
            finalized,
            class_counts,
            unsupported_class_counts,
            messages,
            {
                "duplicate_samples_ignored": duplicate_samples_ignored,
                "conflicting_samples_skipped": conflicting_samples_skipped,
            },
        )

    globals()["_finalize_prairie_reviewed_samples"] = _finalize_prairie_reviewed_samples

    def _discover_local_prairie_samples(local_root: Path) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        if not local_root.exists() or not local_root.is_dir():
            return samples
        for image_path in sorted(local_root.iterdir()):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in prairie_image_suffixes:
                continue
            annotation_candidates = [
                Path(f"{image_path}.xml"),
                image_path.with_suffix(".xml"),
                image_path.parent / f"{image_path.stem}.xml",
            ]
            annotation_path = next((path for path in annotation_candidates if path.exists()), None)
            samples.append(
                {
                    "sample_id": image_path.stem,
                    "image_path": image_path.resolve(),
                    "annotation_path": annotation_path.resolve() if annotation_path else None,
                    "image_source_uri": f"local://{image_path.resolve()}",
                    "annotation_source_uri": (
                        f"local://{annotation_path.resolve()}" if annotation_path else None
                    ),
                    "client_view_url": None,
                    "image_service_url": None,
                }
            )
        return samples

    def _find_upload_by_source_uri(*, user_id: str, source_uri: str) -> dict[str, Any] | None:
        token = str(source_uri or "").strip()
        if not token:
            return None
        for row in store.list_uploads(
            user_id=user_id,
            query=token,
            limit=5000,
            include_deleted=False,
        ):
            if str(row.get("source_uri") or "").strip() != token:
                continue
            return row
        return None

    def _store_sync_upload_from_local_path(
        *,
        user_id: str,
        source_path: Path,
        original_name: str,
        source_uri: str,
        content_type: str | None,
        resource_kind: str,
        metadata: dict[str, Any] | None = None,
        client_view_url: str | None = None,
        image_service_url: str | None = None,
        consume_source: bool = False,
    ) -> dict[str, Any]:
        existing = _find_upload_by_source_uri(user_id=user_id, source_uri=source_uri)
        if existing is not None:
            return existing
        if not source_path.exists() or not source_path.is_file():
            raise ValueError(f"Source file is missing for sync upload: {source_path}")
        temp_path = source_path
        if not consume_source:
            temp_path = upload_store_root / f".prairie-sync-{uuid4().hex}.part"
            shutil.copy2(source_path, temp_path)
        record = _finalize_upload_from_path(
            file_id=uuid4().hex,
            original_name=original_name,
            content_type=content_type,
            source_path=temp_path,
            user_id=user_id,
            source_type="bisque_import",
            source_uri=source_uri,
            client_view_url=client_view_url,
            image_service_url=image_service_url,
            resource_kind=resource_kind,
            metadata=metadata,
        )
        persisted = store.get_upload(record.file_id, user_id=user_id)
        if persisted is None:
            raise RuntimeError(f"Failed to persist sync upload record for {source_uri}")
        return persisted

    def _build_prairie_split_membership(
        reviewed_samples: list[dict[str, Any]],
    ) -> tuple[set[str], set[str]]:
        ordered_ids = [
            str(item.get("sample_id") or "").strip()
            for item in sorted(
                reviewed_samples,
                key=lambda row: str(row.get("sample_id") or "").strip().lower(),
            )
            if str(item.get("sample_id") or "").strip()
        ]
        if not ordered_ids:
            return set(), set()
        if len(ordered_ids) == 1:
            token = ordered_ids[0]
            return {token}, {token}
        val_count = max(1, round(len(ordered_ids) * 0.2))
        val_members = set(ordered_ids[-val_count:])
        train_members = set(ordered_ids) - val_members
        if not train_members:
            train_members.add(ordered_ids[0])
        return train_members, val_members

    def _sync_prairie_active_learning_dataset(
        *,
        user_id: str,
        bisque_auth: dict[str, Any] | None,
        trigger: str,
    ) -> PrairieSyncResponse:
        dataset_name = str(
            getattr(settings, "prairie_active_learning_dataset_name", "Prairie_Dog_Active_Learning")
            or "Prairie_Dog_Active_Learning"
        ).strip()
        source_id = _prairie_source_id(user_id)
        source_row = store.get_training_managed_source(source_id=source_id, user_id=user_id)
        if source_row is None:
            source_row = store.upsert_training_managed_source(
                source_id=source_id,
                user_id=user_id,
                source_type=prairie_source_type,
                name=dataset_name,
                status="idle",
                metadata={"trigger": trigger},
            )
        store.update_training_managed_source(
            source_id=source_id,
            user_id=user_id,
            status="running",
            metadata={**(source_row.get("metadata") or {}), "trigger": trigger},
            error=None,
        )

        bisque_username, bisque_password = _get_effective_bisque_credentials(bisque_auth)
        bisque_access_token = _get_effective_bisque_access_token(bisque_auth)
        bisque_cookie_header = _get_effective_bisque_cookie_header(bisque_auth)
        bisque_root = str((bisque_auth or {}).get("bisque_root") or "").strip().rstrip("/") or str(
            getattr(settings, "bisque_root", "") or ""
        ).strip().rstrip("/")
        headers = _build_bisque_request_headers(
            bisque_username=bisque_username,
            bisque_password=bisque_password,
            bisque_access_token=bisque_access_token,
            bisque_cookie_header=bisque_cookie_header,
        )

        local_sync_dir = (
            str(os.getenv("PRAIRIE_SYNC_LOCAL_DIR") or "").strip()
            or str(os.getenv("PRAIRIE_ACTIVE_LEARNING_LOCAL_DIR") or "").strip()
        )
        discovered_samples: list[dict[str, Any]] = []
        sync_errors: list[str] = []
        bisque_dataset_uri: str | None = None
        skipped_non_image_members = 0
        if local_sync_dir:
            discovered_samples = _discover_local_prairie_samples(Path(local_sync_dir).expanduser())
            if not discovered_samples:
                sync_errors.append(f"No local prairie samples found in {local_sync_dir}.")
        else:
            try:
                bisque_dataset_uri, members = _fetch_prairie_dataset_members_remote(
                    dataset_name=dataset_name,
                    bisque_root=bisque_root,
                    headers=headers,
                )
                for resource_uri in members:
                    image_temp = upload_store_root / f".prairie-sync-image-{uuid4().hex}.part"
                    annotation_temp = upload_store_root / f".prairie-sync-xml-{uuid4().hex}.part"
                    try:
                        resource_uniq = _extract_bisque_resource_uniq(resource_uri)
                        download = _download_bisque_resource_to_temp(
                            resource_uri=resource_uri,
                            resource_uniq=resource_uniq,
                            temp_path=image_temp,
                            bisque_username=bisque_username,
                            bisque_password=bisque_password,
                            bisque_access_token=bisque_access_token,
                            bisque_cookie_header=bisque_cookie_header,
                            allow_settings_fallback=True,
                        )
                        original_name = str(download.get("original_name") or "").strip() or (
                            f"{resource_uniq or uuid4().hex}.jpg"
                        )
                        image_content_type = str(download.get("content_type") or "").strip() or None
                        if not _is_prairie_image_candidate(
                            original_name=original_name,
                            content_type=image_content_type,
                        ):
                            skipped_non_image_members += 1
                            image_temp.unlink(missing_ok=True)
                            annotation_temp.unlink(missing_ok=True)
                            continue
                        annotation_payload = _fetch_resource_deep_xml_bytes(
                            resource_uri=resource_uri,
                            headers=headers,
                        )
                        annotation_temp.write_bytes(annotation_payload)
                        discovered_samples.append(
                            {
                                "sample_id": Path(original_name).stem,
                                "image_path": image_temp.resolve(),
                                "annotation_path": annotation_temp.resolve(),
                                "image_source_uri": resource_uri,
                                "annotation_source_uri": f"{resource_uri}?view=deep",
                                "client_view_url": _build_bisque_links(
                                    resource_uri, bisque_root
                                ).get("client_view_url"),
                                "image_service_url": _build_bisque_links(
                                    resource_uri, bisque_root
                                ).get("image_service_url"),
                                "image_original_name": original_name,
                                "image_content_type": image_content_type,
                            }
                        )
                    except Exception as exc:
                        sync_errors.append(f"{resource_uri}: {exc}")
                        image_temp.unlink(missing_ok=True)
                        annotation_temp.unlink(missing_ok=True)
            except Exception as exc:
                sync_errors.append(str(exc))

        unreviewed_sample_ids: set[str] = set()
        reviewed_candidates: list[dict[str, Any]] = []
        temp_cleanup: list[Path] = []

        for sample in discovered_samples:
            image_path = Path(str(sample.get("image_path") or "")).expanduser()
            annotation_path_raw = sample.get("annotation_path")
            annotation_path = (
                Path(str(annotation_path_raw)).expanduser()
                if annotation_path_raw is not None
                else None
            )
            if not image_path.exists() or not image_path.is_file():
                sync_errors.append(f"{sample.get('sample_id')}: image path missing ({image_path}).")
                continue
            if (
                annotation_path is None
                or not annotation_path.exists()
                or not annotation_path.is_file()
            ):
                sample_id = str(sample.get("sample_id") or image_path.stem).strip()
                if sample_id:
                    unreviewed_sample_ids.add(sample_id)
                sync_errors.append(
                    f"{sample.get('sample_id')}: missing annotation XML (gt2 required)."
                )
                continue
            try:
                image_upload = _store_sync_upload_from_local_path(
                    user_id=user_id,
                    source_path=image_path,
                    original_name=str(sample.get("image_original_name") or image_path.name),
                    source_uri=str(sample.get("image_source_uri") or f"local://{image_path}"),
                    content_type=(
                        str(sample.get("image_content_type") or "").strip()
                        or mimetypes.guess_type(image_path.name)[0]
                        or "application/octet-stream"
                    ),
                    resource_kind="image",
                    metadata={"prairie_active_learning": True},
                    client_view_url=(str(sample.get("client_view_url") or "").strip() or None),
                    image_service_url=(str(sample.get("image_service_url") or "").strip() or None),
                    consume_source=image_path.name.startswith(".prairie-sync-image-"),
                )
                annotation_upload = _store_sync_upload_from_local_path(
                    user_id=user_id,
                    source_path=annotation_path,
                    original_name=f"{Path(str(sample.get('image_original_name') or image_path.name)).name}.xml",
                    source_uri=(
                        str(sample.get("annotation_source_uri") or "").strip()
                        or f"local://{annotation_path}"
                    ),
                    content_type="application/xml",
                    resource_kind="file",
                    metadata={"prairie_active_learning": True, "annotation_layer": "gt2"},
                    consume_source=annotation_path.name.startswith(".prairie-sync-xml-"),
                )
                sample_id = str(sample.get("sample_id") or image_path.stem).strip()
                temp_cleanup.extend([image_path, annotation_path])
                annotation_local_value = _resolved_local_path_for_upload(
                    annotation_upload,
                    user_id=user_id,
                )
                image_local_value = _resolved_local_path_for_upload(
                    image_upload,
                    user_id=user_id,
                )
                if not annotation_local_value or not image_local_value:
                    sync_errors.append(
                        f"{sample.get('sample_id')}: synced uploads missing local stored paths."
                    )
                    continue
                annotation_local = Path(annotation_local_value)
                image_local = Path(image_local_value)
                reviewed, supported, unsupported, annotation_signature, parse_error = (
                    _parse_prairie_annotation_counts(annotation_local)
                )
                if parse_error:
                    sync_errors.append(f"{sample.get('sample_id')}: {parse_error}")
                if reviewed:
                    reviewed_candidates.append(
                        {
                            "sample_id": sample_id or str(image_local.stem),
                            "image": {
                                "file_id": str(image_upload.get("file_id") or ""),
                                "path": str(image_local),
                                "original_name": str(
                                    image_upload.get("original_name") or image_local.name
                                ),
                                "sha256": str(image_upload.get("sha256") or "").strip() or None,
                                "size_bytes": int(image_upload.get("size_bytes") or 0),
                            },
                            "annotation": {
                                "file_id": str(annotation_upload.get("file_id") or ""),
                                "path": str(annotation_local),
                                "original_name": str(
                                    annotation_upload.get("original_name") or annotation_local.name
                                ),
                                "sha256": str(annotation_upload.get("sha256") or "").strip()
                                or None,
                                "size_bytes": int(annotation_upload.get("size_bytes") or 0),
                            },
                            "supported_counts": {
                                name: int(supported.get(name) or 0)
                                for name in prairie_supported_classes
                            },
                            "unsupported_counts": {
                                str(class_name): int(value or 0)
                                for class_name, value in unsupported.items()
                                if str(class_name).strip()
                            },
                            "annotation_signature": annotation_signature,
                        }
                    )
                elif sample_id:
                    unreviewed_sample_ids.add(sample_id)
            except Exception as exc:
                sync_errors.append(f"{sample.get('sample_id')}: {exc}")

        for path in temp_cleanup:
            if path.name.startswith(".prairie-sync-"):
                path.unlink(missing_ok=True)

        (
            reviewed_samples,
            class_counts,
            unsupported_class_counts,
            dedupe_messages,
            dedupe_stats,
        ) = _finalize_prairie_reviewed_samples(reviewed_candidates)
        if dedupe_messages:
            sync_errors.extend(dedupe_messages)
        reviewed_sample_ids = {
            str(sample.get("sample_id") or "").strip()
            for sample in reviewed_samples
            if str(sample.get("sample_id") or "").strip()
        }
        unreviewed_sample_ids -= reviewed_sample_ids
        synced_images = len(reviewed_sample_ids) + len(unreviewed_sample_ids)

        dataset_id = (
            str(source_row.get("training_dataset_id") or "").strip()
            or f"dataset-{uuid4().hex[:14]}"
        )
        store.create_training_dataset(
            dataset_id=dataset_id,
            user_id=user_id,
            name=dataset_name,
            description="Managed active-learning source for prairie dog YOLO workflow.",
            metadata={
                "managed_source_type": prairie_source_type,
                "model_key": prairie_model_key,
            },
        )

        train_members, val_members = _build_prairie_split_membership(reviewed_samples)
        assignment_rows: list[dict[str, Any]] = []
        for sample in reviewed_samples:
            sample_id = str(sample.get("sample_id") or "").strip()
            if not sample_id:
                continue
            split_targets: list[str] = []
            if sample_id in train_members:
                split_targets.append("train")
            if sample_id in val_members:
                split_targets.append("val")
            for split in split_targets:
                image_row = sample.get("image") if isinstance(sample.get("image"), dict) else {}
                annotation_row = (
                    sample.get("annotation") if isinstance(sample.get("annotation"), dict) else {}
                )
                assignment_rows.append(
                    {
                        "split": split,
                        "role": "image",
                        "sample_id": sample_id,
                        "file_id": str(image_row.get("file_id") or ""),
                        "path": str(image_row.get("path") or ""),
                        "original_name": str(image_row.get("original_name") or ""),
                        "sha256": image_row.get("sha256"),
                        "size_bytes": int(image_row.get("size_bytes") or 0),
                        "metadata": {"managed": True, "trigger": trigger},
                    }
                )
                assignment_rows.append(
                    {
                        "split": split,
                        "role": "annotation",
                        "sample_id": sample_id,
                        "file_id": str(annotation_row.get("file_id") or ""),
                        "path": str(annotation_row.get("path") or ""),
                        "original_name": str(annotation_row.get("original_name") or ""),
                        "sha256": annotation_row.get("sha256"),
                        "size_bytes": int(annotation_row.get("size_bytes") or 0),
                        "metadata": {"managed": True, "layer": "gt2", "trigger": trigger},
                    }
                )

        store.upsert_training_dataset_items(
            dataset_id=dataset_id,
            user_id=user_id,
            items=assignment_rows,
            replace=True,
        )

        manifest: dict[str, Any] = {}
        try:
            _, _, manifest = _build_training_dataset_manifest(
                dataset_id=dataset_id,
                user_id=user_id,
                model_key=prairie_model_key,
            )
        except HTTPException as exc:
            sync_errors.append(str(exc.detail))
            manifest = {}

        synced_at = datetime.utcnow().isoformat()
        reviewed_images = len(reviewed_samples)
        unreviewed_images = len(unreviewed_sample_ids)
        manifest_counts = manifest.get("counts") if isinstance(manifest.get("counts"), dict) else {}
        merged_metadata = {
            "managed_source_type": prairie_source_type,
            "model_key": prairie_model_key,
            "last_sync_at": synced_at,
            "manifest_counts": manifest_counts,
            "reviewed_images": reviewed_images,
            "unreviewed_images": unreviewed_images,
            "class_counts": class_counts,
            "unsupported_class_counts": unsupported_class_counts,
            "bisque_dataset_uri": bisque_dataset_uri,
            "dedupe": dedupe_stats,
        }
        store.create_training_dataset(
            dataset_id=dataset_id,
            user_id=user_id,
            name=dataset_name,
            description="Managed active-learning source for prairie dog YOLO workflow.",
            metadata=merged_metadata,
        )
        success = synced_images > 0
        stats = {
            "synced_images": synced_images,
            "reviewed_images": reviewed_images,
            "unreviewed_images": unreviewed_images,
            "class_counts": class_counts,
            "unsupported_class_counts": unsupported_class_counts,
            "manifest_counts": manifest_counts,
            "last_sync_trigger": trigger,
            "skipped_non_image_members": skipped_non_image_members,
            "dedupe": dedupe_stats,
        }
        store.update_training_managed_source(
            source_id=source_id,
            user_id=user_id,
            training_dataset_id=dataset_id,
            remote_uri=bisque_dataset_uri,
            status="idle" if success else "failed",
            stats=stats,
            metadata={
                **(
                    source_row.get("metadata")
                    if isinstance(source_row.get("metadata"), dict)
                    else {}
                ),
                "trigger": trigger,
            },
            error=None if success else "; ".join(sync_errors[:3]) or "No samples were synced.",
            last_sync_at=synced_at,
        )
        return PrairieSyncResponse(
            success=success,
            dataset_name=dataset_name,
            dataset_id=dataset_id,
            bisque_dataset_uri=bisque_dataset_uri,
            synced_images=synced_images,
            reviewed_images=reviewed_images,
            unreviewed_images=unreviewed_images,
            class_counts=class_counts,
            unsupported_class_counts=unsupported_class_counts,
            last_sync_at=synced_at,
            errors=sync_errors,
        )

    def _collect_prairie_detection_counts(user_id: str) -> dict[str, int]:
        counts = {name: 0 for name in prairie_supported_classes}
        jobs = store.list_training_jobs(
            user_id=user_id,
            job_type="inference",
            model_key=prairie_model_key,
            statuses=["succeeded"],
            limit=500,
        )
        for row in jobs:
            result = row.get("result") if isinstance(row.get("result"), dict) else {}
            per_class = (
                result.get("counts_by_class")
                if isinstance(result.get("counts_by_class"), dict)
                else {}
            )
            for class_name, value in per_class.items():
                token = str(class_name or "").strip()
                if token not in counts:
                    continue
                try:
                    counts[token] = counts.get(token, 0) + int(value)
                except Exception:
                    continue
        return counts

    def _metric_packet_complete(row: dict[str, Any] | None) -> bool:
        payload = row if isinstance(row, dict) else {}
        required = (
            payload.get("map50"),
            payload.get("map50_95"),
            payload.get("precision"),
            payload.get("recall"),
            payload.get("fp_per_image"),
            payload.get("fn_per_image"),
        )
        return all(isinstance(value, (float, int)) for value in required)

    def _promotion_benchmark_packet_complete(metrics: dict[str, Any] | None) -> bool:
        payload = metrics if isinstance(metrics, dict) else {}
        benchmark = payload.get("benchmark") if isinstance(payload.get("benchmark"), dict) else {}
        baseline = (
            benchmark.get("baseline_before_train")
            if isinstance(benchmark.get("baseline_before_train"), dict)
            else payload.get("benchmark_baseline")
            if isinstance(payload.get("benchmark_baseline"), dict)
            else {}
        )
        candidate = (
            benchmark.get("candidate_after_train")
            if isinstance(benchmark.get("candidate_after_train"), dict)
            else payload.get("benchmark_latest_candidate")
            if isinstance(payload.get("benchmark_latest_candidate"), dict)
            else {}
        )
        baseline_canonical = (
            baseline.get("canonical") if isinstance(baseline.get("canonical"), dict) else {}
        )
        baseline_active = baseline.get("active") if isinstance(baseline.get("active"), dict) else {}
        candidate_canonical = (
            candidate.get("canonical") if isinstance(candidate.get("canonical"), dict) else {}
        )
        candidate_active = (
            candidate.get("active") if isinstance(candidate.get("active"), dict) else {}
        )
        return bool(
            _metric_packet_complete(baseline_canonical)
            and _metric_packet_complete(baseline_active)
            and _metric_packet_complete(candidate_canonical)
            and _metric_packet_complete(candidate_active)
        )

    def _canonical_benchmark_packet_complete(metrics: dict[str, Any] | None) -> bool:
        payload = metrics if isinstance(metrics, dict) else {}
        manual_benchmark = (
            payload.get("manual_benchmark")
            if isinstance(payload.get("manual_benchmark"), dict)
            else {}
        )
        manual_canonical = (
            manual_benchmark.get("canonical")
            if isinstance(manual_benchmark.get("canonical"), dict)
            else {}
        )
        if _metric_packet_complete(manual_canonical):
            return True
        latest_candidate = (
            payload.get("benchmark_latest_candidate")
            if isinstance(payload.get("benchmark_latest_candidate"), dict)
            else {}
        )
        canonical_candidate = (
            latest_candidate.get("canonical")
            if isinstance(latest_candidate.get("canonical"), dict)
            else {}
        )
        if _metric_packet_complete(canonical_candidate):
            return True
        benchmark = payload.get("benchmark") if isinstance(payload.get("benchmark"), dict) else {}
        candidate_after = (
            benchmark.get("candidate_after_train")
            if isinstance(benchmark.get("candidate_after_train"), dict)
            else {}
        )
        candidate_canonical = (
            candidate_after.get("canonical")
            if isinstance(candidate_after.get("canonical"), dict)
            else {}
        )
        return _metric_packet_complete(candidate_canonical)

    def _evaluate_prairie_retrain_gate(stats: dict[str, Any] | None) -> dict[str, Any]:
        payload = stats if isinstance(stats, dict) else {}
        reviewed_images = int(payload.get("reviewed_images") or 0)
        class_counts = (
            payload.get("class_counts") if isinstance(payload.get("class_counts"), dict) else {}
        )
        class_object_counts = {
            class_name: int(class_counts.get(class_name) or 0)
            for class_name in prairie_supported_classes
        }
        total_objects = sum(int(value) for value in class_object_counts.values())
        min_reviewed = int(getattr(settings, "prairie_retrain_min_reviewed_samples", 30) or 30)
        min_total_objects = int(getattr(settings, "prairie_retrain_min_total_objects", 300) or 300)
        min_class_objects = int(getattr(settings, "prairie_retrain_min_class_objects", 80) or 80)
        reasons: list[str] = []
        if reviewed_images < min_reviewed:
            reasons.append(
                f"Need at least {min_reviewed} reviewed images (current {reviewed_images})."
            )
        if total_objects < min_total_objects:
            reasons.append(
                f"Need at least {min_total_objects} reviewed objects (current {total_objects})."
            )
        for class_name in prairie_supported_classes:
            count = int(class_object_counts.get(class_name) or 0)
            if count < min_class_objects:
                reasons.append(
                    f"Need at least {min_class_objects} '{class_name}' objects (current {count})."
                )
        return {
            "ready": len(reasons) == 0,
            "reasons": reasons,
            "counts": {
                "reviewed_images": reviewed_images,
                "total_objects": total_objects,
                **{
                    name: int(class_object_counts.get(name) or 0)
                    for name in prairie_supported_classes
                },
            },
            "thresholds": {
                "reviewed_images": min_reviewed,
                "total_objects": min_total_objects,
                "per_class_objects": min_class_objects,
            },
        }

    def _extract_benchmark_status(
        metrics: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, Any], str | None, bool, bool, bool]:
        payload = metrics if isinstance(metrics, dict) else {}
        baseline = (
            payload.get("benchmark_baseline")
            if isinstance(payload.get("benchmark_baseline"), dict)
            else {}
        )
        latest_candidate = (
            payload.get("benchmark_latest_candidate")
            if isinstance(payload.get("benchmark_latest_candidate"), dict)
            else {}
        )
        benchmark = {}
        if (not baseline or not latest_candidate) and isinstance(payload.get("benchmark"), dict):
            benchmark = (
                payload.get("benchmark") if isinstance(payload.get("benchmark"), dict) else {}
            )
        if not baseline and isinstance(benchmark.get("baseline_before_train"), dict):
            baseline = benchmark.get("baseline_before_train")
        if not latest_candidate and isinstance(benchmark.get("candidate_after_train"), dict):
            latest_candidate = benchmark.get("candidate_after_train")
        manual_benchmark = (
            payload.get("manual_benchmark")
            if isinstance(payload.get("manual_benchmark"), dict)
            else {}
        )
        last_benchmark_at = str(payload.get("last_benchmark_at") or "").strip() or None
        if not last_benchmark_at:
            last_benchmark_at = str(manual_benchmark.get("last_benchmark_at") or "").strip() or None
        canonical_ready = bool(payload.get("canonical_benchmark_ready"))
        promotion_ready = bool(payload.get("promotion_benchmark_ready"))
        benchmark_ready = bool(payload.get("benchmark_ready"))
        if not canonical_ready:
            canonical_ready = _canonical_benchmark_packet_complete(payload)
        if not promotion_ready:
            promotion_ready = _promotion_benchmark_packet_complete(payload)
        if not benchmark_ready and isinstance(payload.get("benchmark"), dict):
            benchmark_ready = bool((payload.get("benchmark") or {}).get("ready"))
        if not benchmark_ready:
            benchmark_ready = promotion_ready
        return (
            baseline,
            latest_candidate,
            last_benchmark_at,
            benchmark_ready,
            canonical_ready,
            promotion_ready,
        )

    def _build_prairie_gating_summary(result_payload: dict[str, Any]) -> dict[str, Any]:
        metrics = (
            result_payload.get("metrics") if isinstance(result_payload.get("metrics"), dict) else {}
        )
        benchmark = metrics.get("benchmark") if isinstance(metrics.get("benchmark"), dict) else {}
        comparison = (
            benchmark.get("comparison") if isinstance(benchmark.get("comparison"), dict) else {}
        )
        continuous = (
            result_payload.get("continuous_learning")
            if isinstance(result_payload.get("continuous_learning"), dict)
            else {}
        )
        guardrails = (
            continuous.get("guardrails")
            if isinstance(continuous.get("guardrails"), dict)
            else comparison
            if isinstance(comparison, dict)
            else {}
        )
        return {
            "benchmark_ready": bool(metrics.get("benchmark_ready") or benchmark.get("ready")),
            "canonical_benchmark_ready": bool(metrics.get("canonical_benchmark_ready")),
            "promotion_benchmark_ready": bool(
                metrics.get("promotion_benchmark_ready")
                or metrics.get("benchmark_ready")
                or benchmark.get("ready")
            ),
            "guardrails_passed": bool(guardrails.get("passed")),
            "reasons": list(guardrails.get("reasons") or []),
            "checks": guardrails.get("checks")
            if isinstance(guardrails.get("checks"), dict)
            else {},
        }

    def _list_prairie_retrain_jobs(user_id: str, *, limit: int = 200) -> list[dict[str, Any]]:
        rows = store.list_training_jobs(
            user_id=user_id,
            job_type="training",
            model_key=prairie_model_key,
            limit=limit,
        )
        output: list[dict[str, Any]] = []
        for row in rows:
            request_payload = row.get("request") if isinstance(row.get("request"), dict) else {}
            if not bool(request_payload.get("prairie_active_learning")):
                continue
            output.append(row)
        return output

    def _collect_job_output_artifacts(
        *,
        artifact_run_id: str,
        output_root: Path,
        category: str,
        tool_name: str,
    ) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        if not output_root.exists():
            return entries
        for path in sorted(output_root.rglob("*")):
            if not path.is_file():
                continue
            entry = _artifact_entry(artifact_run_id, path)
            entry["category"] = category
            entry["tool"] = tool_name
            entries.append(entry)
        return entries

    def _create_training_artifact_run(*, user_id: str, goal: str) -> str:
        run = WorkflowRun.new(goal=goal, plan=None)
        store.create_run(run)
        store.set_run_metadata(run.run_id, user_id=user_id, conversation_id=None)
        _ensure_run_artifact_dir(run.run_id)
        store.update_status(run.run_id, RunStatus.PENDING)
        store.append_event(run.run_id, "training_run_created", {"goal": goal})
        return run.run_id

    def _assert_training_job_owner(
        *,
        row: dict[str, Any] | None,
        user_id: str,
    ) -> dict[str, Any]:
        if row is None:
            raise HTTPException(status_code=404, detail="Training job not found.")
        owner = str(row.get("user_id") or "").strip()
        if owner != user_id:
            raise HTTPException(status_code=404, detail="Training job not found.")
        return row

    def _active_training_count(job_type: str) -> int:
        active = store.list_training_jobs(
            statuses=["queued", "running", "paused"],
            job_type=job_type,
            limit=50,
        )
        return len(active)

    def _training_thread_alive(job_id: str) -> bool:
        with training_job_threads_lock:
            thread = training_job_threads.get(job_id)
        return thread is not None and thread.is_alive()

    def _latest_checkpoint_from_job(row: dict[str, Any]) -> str | None:
        result_payload = row.get("result") if isinstance(row.get("result"), dict) else {}
        checkpoint_paths = result_payload.get("checkpoint_paths")
        if isinstance(checkpoint_paths, list) and checkpoint_paths:
            token = str(checkpoint_paths[-1] or "").strip()
            if token:
                return token
        last_progress = (
            result_payload.get("last_progress")
            if isinstance(result_payload.get("last_progress"), dict)
            else {}
        )
        token = str(last_progress.get("checkpoint_path") or "").strip()
        return token or None

    def _open_proposal_statuses() -> list[str]:
        return ["pending_approval", "approved", "running", "evaluating"]

    def _proposal_ready_statuses() -> list[str]:
        return ["ready_to_promote"]

    def _continuous_guardrail_policy() -> ContinuousLearningPolicy:
        return ContinuousLearningPolicy(
            canonical_map50_drop_max=float(
                getattr(settings, "prairie_guardrail_canonical_map50_drop_max", 0.02) or 0.02
            ),
            prairie_dog_map50_drop_max=float(
                getattr(settings, "prairie_guardrail_prairie_dog_map50_drop_max", 0.03) or 0.03
            ),
            active_map50_drop_max=float(
                getattr(settings, "prairie_guardrail_active_map50_drop_max", 0.02) or 0.02
            ),
            canonical_fp_image_increase_max=float(
                getattr(settings, "prairie_guardrail_canonical_fp_image_increase_max", 0.25) or 0.25
            ),
        )

    def _active_version_metrics(lineage_row: dict[str, Any]) -> dict[str, Any]:
        active_version_row = _lineage_active_version_row(lineage_row)
        if active_version_row is None:
            return {}
        return (
            active_version_row.get("metrics")
            if isinstance(active_version_row.get("metrics"), dict)
            else {}
        )

    def _derive_candidate_metrics(
        *,
        training_result: dict[str, Any],
        fallback_drift_score: float = 0.30,
    ) -> dict[str, Any]:
        metrics = (
            training_result.get("metrics")
            if isinstance(training_result.get("metrics"), dict)
            else {}
        )
        benchmark = metrics.get("benchmark") if isinstance(metrics.get("benchmark"), dict) else {}
        baseline_packet = (
            benchmark.get("baseline_before_train")
            if isinstance(benchmark.get("baseline_before_train"), dict)
            else {}
        )
        candidate_packet = (
            benchmark.get("candidate_after_train")
            if isinstance(benchmark.get("candidate_after_train"), dict)
            else {}
        )
        comparison_packet = (
            benchmark.get("comparison") if isinstance(benchmark.get("comparison"), dict) else {}
        )
        map50_value = (
            float(metrics.get("map50_candidate_eval"))
            if isinstance(metrics.get("map50_candidate_eval"), (float, int))
            else float(metrics.get("map50"))
            if isinstance(metrics.get("map50"), (float, int))
            else None
        )
        map50_baseline = (
            float(metrics.get("map50_baseline"))
            if isinstance(metrics.get("map50_baseline"), (float, int))
            else None
        )
        best_val = float(
            metrics.get("best_val_dice")
            if isinstance(metrics.get("best_val_dice"), (float, int))
            else metrics.get("val_dice")
            if isinstance(metrics.get("val_dice"), (float, int))
            else map50_value
            if isinstance(map50_value, (float, int))
            else 0.0
        )
        if best_val <= 0.0:
            best_val = 0.0
        legacy_val = float(metrics.get("legacy_val_dice") or max(0.0, best_val - 0.005))
        worst_class = float(metrics.get("worst_class_legacy_dice") or max(0.0, legacy_val - 0.01))
        drift_score = float(
            metrics.get("drift_score_mean")
            if isinstance(metrics.get("drift_score_mean"), (float, int))
            else max(0.01, min(1.0, 1.0 - best_val))
            if isinstance(metrics.get("map50"), (float, int))
            else fallback_drift_score
        )
        return {
            "dice_new_holdout": round(best_val, 6),
            "dice_legacy_holdout": round(legacy_val, 6),
            "worst_class_legacy_dice": round(worst_class, 6),
            "drift_score_mean": round(max(0.0, drift_score), 6),
            "best_val_dice": round(best_val, 6),
            "map50": round(map50_value, 6) if isinstance(map50_value, float) else None,
            "map50_baseline": round(map50_baseline, 6)
            if isinstance(map50_baseline, float)
            else None,
            "map50_delta_vs_baseline": round(map50_value - map50_baseline, 6)
            if isinstance(map50_value, float) and isinstance(map50_baseline, float)
            else None,
            "benchmark_ready": bool(metrics.get("benchmark_ready")),
            "canonical_benchmark_ready": bool(metrics.get("canonical_benchmark_ready")),
            "promotion_benchmark_ready": bool(
                metrics.get("promotion_benchmark_ready") or metrics.get("benchmark_ready")
            ),
            "benchmark": benchmark,
            "benchmark_baseline": baseline_packet,
            "benchmark_latest_candidate": candidate_packet,
            "benchmark_comparison": comparison_packet,
            "last_benchmark_at": str(metrics.get("last_benchmark_at") or "").strip() or None,
        }

    def _continuous_failure_update(
        *,
        request_payload: dict[str, Any],
        error: str,
        finished_at: str,
    ) -> None:
        continuous = (
            request_payload.get("continuous")
            if isinstance(request_payload.get("continuous"), dict)
            else {}
        )
        proposal_id = str(continuous.get("proposal_id") or "").strip()
        if not proposal_id:
            return
        proposal_row = store.get_training_update_proposal(proposal_id=proposal_id)
        if proposal_row is None:
            return
        current_status = normalize_proposal_status(proposal_row.get("status"))
        if proposal_transition_allowed(current_status, "failed"):
            store.update_training_update_proposal(
                proposal_id=proposal_id,
                status="failed",
                error=error,
                finished_at=finished_at,
            )

    def _continuous_success_update(
        *,
        job_row: dict[str, Any],
        request_payload: dict[str, Any],
        training_result: dict[str, Any],
        model_version: str,
        artifact_run_id: str | None,
        finished_at: str,
    ) -> dict[str, Any]:
        updated_result = dict(training_result)
        model_key = str(job_row.get("model_key") or "").strip().lower() or prairie_model_key
        user_id = str(job_row.get("user_id") or "").strip()
        continuous = (
            request_payload.get("continuous")
            if isinstance(request_payload.get("continuous"), dict)
            else {}
        )
        lineage_id = str(continuous.get("lineage_id") or "").strip()
        proposal_id = str(continuous.get("proposal_id") or "").strip()

        lineage_row: dict[str, Any] | None = None
        if lineage_id:
            lineage_row = store.get_training_lineage(lineage_id=lineage_id)
        if lineage_row is None:
            _, lineage_row = _ensure_default_domain_and_lineage(
                user_id=user_id, model_key=model_key
            )
            lineage_id = str(lineage_row.get("lineage_id") or "").strip()

        candidate_metrics = _derive_candidate_metrics(training_result=training_result)
        benchmark_ready = bool(
            candidate_metrics.get("promotion_benchmark_ready")
            or candidate_metrics.get("benchmark_ready")
        )
        checkpoint_paths = (
            list(updated_result.get("checkpoint_paths"))
            if isinstance(updated_result.get("checkpoint_paths"), list)
            else []
        )
        latest_checkpoint = str(checkpoint_paths[-1] or "").strip() if checkpoint_paths else None

        version_row = store.create_training_model_version(
            version_id=model_version,
            lineage_id=lineage_id,
            source_job_id=str(job_row.get("job_id") or "").strip(),
            artifact_run_id=artifact_run_id,
            status="candidate",
            metrics=candidate_metrics,
            metadata={
                "model_key": model_key,
                "model_version": model_version,
                "created_from_job_id": str(job_row.get("job_id") or "").strip(),
                "model_artifact_path": str(updated_result.get("model_artifact_path") or "").strip()
                or latest_checkpoint,
                "checkpoint_path": latest_checkpoint,
                "continuous": continuous,
            },
        )

        proposal_state_row = (
            store.get_training_update_proposal(proposal_id=proposal_id) if proposal_id else None
        )
        if proposal_state_row is not None:
            current_status = normalize_proposal_status(proposal_state_row.get("status"))
            if proposal_transition_allowed(current_status, "evaluating"):
                proposal_state_row = store.update_training_update_proposal(
                    proposal_id=proposal_id,
                    status="evaluating",
                )

        active_version_id = str(lineage_row.get("active_version_id") or "").strip()
        if not active_version_id:
            if benchmark_ready:
                store.update_training_model_version(version_id=model_version, status="active")
                store.update_training_lineage(
                    lineage_id=lineage_id,
                    active_version_id=model_version,
                )
                updated_result["continuous_learning"] = {
                    "lineage_id": lineage_id,
                    "candidate_version_id": model_version,
                    "promotion_state": "active_initial",
                }
            else:
                store.update_training_model_version(version_id=model_version, status="retired")
                updated_result["continuous_learning"] = {
                    "lineage_id": lineage_id,
                    "candidate_version_id": model_version,
                    "promotion_state": "rejected_guardrails",
                    "guardrails": {
                        "passed": False,
                        "reasons": ["Initial model version is missing complete benchmark packet."],
                    },
                }
            if proposal_id:
                proposal_row = proposal_state_row or store.get_training_update_proposal(
                    proposal_id=proposal_id
                )
                if proposal_row is not None:
                    current_status = normalize_proposal_status(proposal_row.get("status"))
                    if benchmark_ready:
                        if proposal_transition_allowed(current_status, "ready_to_promote"):
                            proposal_row = store.update_training_update_proposal(
                                proposal_id=proposal_id,
                                status="ready_to_promote",
                                candidate_version_id=model_version,
                                finished_at=finished_at,
                            )
                        current_status = normalize_proposal_status(
                            (proposal_row or {}).get("status")
                        )
                        if proposal_transition_allowed(current_status, "promoted"):
                            store.update_training_update_proposal(
                                proposal_id=proposal_id,
                                status="promoted",
                                candidate_version_id=model_version,
                                finished_at=finished_at,
                            )
                    elif proposal_transition_allowed(current_status, "rejected"):
                        store.update_training_update_proposal(
                            proposal_id=proposal_id,
                            status="rejected",
                            candidate_version_id=model_version,
                            error="Initial model version is missing complete benchmark packet.",
                            finished_at=finished_at,
                        )
            return updated_result

        active_metrics = _active_version_metrics(lineage_row)
        guardrails = evaluate_promotion_guardrails(
            active_metrics=active_metrics,
            candidate_metrics=candidate_metrics,
            policy=_continuous_guardrail_policy(),
        )
        guardrail_reasons = list(guardrails.get("reasons") or [])
        passed = bool(guardrails.get("passed"))

        current_version_metadata = (
            version_row.get("metadata") if isinstance(version_row.get("metadata"), dict) else {}
        )
        version_metadata = {
            **current_version_metadata,
            "guardrails": guardrails,
            "benchmark_ready": benchmark_ready,
        }
        if passed and benchmark_ready:
            store.update_training_model_version(
                version_id=model_version,
                status="canary",
                metrics=candidate_metrics,
                metadata=version_metadata,
            )
            updated_result["continuous_learning"] = {
                "lineage_id": lineage_id,
                "candidate_version_id": model_version,
                "promotion_state": "ready_to_promote",
                "guardrails": guardrails,
            }
            if proposal_id:
                proposal_row = proposal_state_row or store.get_training_update_proposal(
                    proposal_id=proposal_id
                )
                if proposal_row is not None:
                    current_status = normalize_proposal_status(proposal_row.get("status"))
                    if proposal_transition_allowed(current_status, "ready_to_promote"):
                        store.update_training_update_proposal(
                            proposal_id=proposal_id,
                            status="ready_to_promote",
                            candidate_version_id=model_version,
                            finished_at=finished_at,
                        )
        else:
            store.update_training_model_version(
                version_id=model_version,
                status="retired",
                metrics=candidate_metrics,
                metadata=version_metadata,
            )
            error_message = "Promotion guardrails failed."
            if not benchmark_ready:
                error_message = (
                    "Promotion guardrails failed. Missing complete benchmark packet "
                    "(canonical + active metrics required)."
                )
            if guardrail_reasons:
                error_message += " " + "; ".join(guardrail_reasons[:3])
            updated_result["continuous_learning"] = {
                "lineage_id": lineage_id,
                "candidate_version_id": model_version,
                "promotion_state": "rejected_guardrails",
                "guardrails": guardrails,
            }
            if proposal_id:
                proposal_row = proposal_state_row or store.get_training_update_proposal(
                    proposal_id=proposal_id
                )
                if proposal_row is not None:
                    current_status = normalize_proposal_status(proposal_row.get("status"))
                    if proposal_transition_allowed(current_status, "rejected"):
                        store.update_training_update_proposal(
                            proposal_id=proposal_id,
                            status="rejected",
                            candidate_version_id=model_version,
                            error=error_message,
                            finished_at=finished_at,
                        )
        return updated_result

    def _build_trigger_snapshot(
        *,
        lineage_id: str,
        dataset_id: str,
        approved_new_samples: int,
        class_counts: dict[str, int],
        health_status: str | None,
        last_promoted_at: str | None,
    ) -> dict[str, Any]:
        trigger = evaluate_trigger_policy(
            approved_new_samples=approved_new_samples,
            class_counts=class_counts,
            health_status=health_status,
            last_promoted_at=last_promoted_at,
        )
        return {
            **trigger,
            "lineage_id": lineage_id,
            "dataset_id": dataset_id,
            "health_status": health_status,
        }

    def _recover_stale_training_jobs() -> None:
        recoverable_jobs = store.list_training_jobs(
            statuses=["queued", "running", "paused"],
            limit=500,
        )
        if not recoverable_jobs:
            return
        stale_threshold_seconds = max(
            30,
            int(getattr(settings, "training_job_stale_heartbeat_seconds", 180) or 180),
        )
        now = datetime.utcnow()
        for row in recoverable_jobs:
            job_id = str(row.get("job_id") or "").strip()
            user_id = str(row.get("user_id") or "").strip()
            if not job_id or not user_id:
                continue
            if _training_thread_alive(job_id):
                continue
            status = str(row.get("status") or "").strip().lower()
            if status == "queued":
                _launch_training_worker(job_id=job_id, user_id=user_id)
                continue
            if status == "paused":
                # Paused jobs intentionally have no progressing heartbeat. Keep them resumable.
                continue
            heartbeat_reference = (
                str(row.get("last_heartbeat_at") or "").strip()
                or str(row.get("started_at") or "").strip()
                or str(row.get("updated_at") or "").strip()
            )
            heartbeat_dt = _parse_datetime(heartbeat_reference)
            stale_seconds = max(0.0, (now - heartbeat_dt).total_seconds())
            if stale_seconds < float(stale_threshold_seconds):
                continue
            checkpoint_hint = _latest_checkpoint_from_job(row)
            message = (
                "Worker restart detected. Job is recoverable; use restart action to resume "
                "from latest checkpoint. "
                f"Heartbeat stale for {int(stale_seconds)}s."
            )
            if checkpoint_hint:
                message += f" Latest checkpoint: {checkpoint_hint}"
            store.update_training_job(
                job_id=job_id,
                user_id=user_id,
                status="failed",
                error=message,
                finished_at=now.isoformat(),
                heartbeat_at=now.isoformat(),
            )
            store.append_training_job_event(
                job_id=job_id,
                user_id=user_id,
                event_type="recovery_required",
                payload={
                    "message": message,
                    "stale_seconds": int(stale_seconds),
                    "stale_threshold_seconds": stale_threshold_seconds,
                    "checkpoint_hint": checkpoint_hint,
                },
            )
            run_id = str(row.get("artifact_run_id") or "").strip()
            if run_id:
                with suppress(Exception):
                    store.update_status(run_id, RunStatus.FAILED, error=message)

    def _training_control_callback(job_id: str, user_id: str) -> None:
        paused_once = False
        while True:
            row = store.get_training_job(job_id=job_id, user_id=user_id)
            if row is None:
                raise TrainingCancelledError("Job no longer exists.")
            control = row.get("control") if isinstance(row.get("control"), dict) else {}
            action = str(control.get("action") or "").strip().lower()
            if action == "cancel":
                raise TrainingCancelledError("Job canceled by user.")
            if action == "pause":
                now = datetime.utcnow().isoformat()
                store.update_training_job(
                    job_id=job_id,
                    user_id=user_id,
                    status="paused",
                    heartbeat_at=now,
                )
                if not paused_once:
                    store.append_training_job_event(
                        job_id=job_id,
                        user_id=user_id,
                        event_type="paused",
                        payload={"message": "Job paused by user request."},
                    )
                    paused_once = True
                time.sleep(0.6)
                continue
            if paused_once and action in {"resume", "", "running"}:
                now = datetime.utcnow().isoformat()
                store.update_training_job(
                    job_id=job_id,
                    user_id=user_id,
                    status="running",
                    control={"action": "running"},
                    heartbeat_at=now,
                )
                store.append_training_job_event(
                    job_id=job_id,
                    user_id=user_id,
                    event_type="resumed",
                    payload={"message": "Job resumed."},
                )
            return

    def _launch_training_worker(job_id: str, user_id: str) -> None:
        def _worker() -> None:
            try:
                row = store.get_training_job(job_id=job_id, user_id=user_id)
                if row is None:
                    return
                initial_status = str(row.get("status") or "").strip().lower()
                initial_control = row.get("control") if isinstance(row.get("control"), dict) else {}
                initial_action = str(initial_control.get("action") or "").strip().lower()
                if initial_status in {"canceled", "failed"} or initial_action == "cancel":
                    return
                job_type = str(row.get("job_type") or "training").strip().lower()
                model_key = str(row.get("model_key") or "").strip().lower()
                request_payload = row.get("request") if isinstance(row.get("request"), dict) else {}
                config = (
                    request_payload.get("config")
                    if isinstance(request_payload.get("config"), dict)
                    else {}
                )
                artifact_run_id = str(row.get("artifact_run_id") or "").strip()
                output_root = _ensure_run_artifact_dir(artifact_run_id) / "model_jobs" / job_id
                output_root.mkdir(parents=True, exist_ok=True)
                now = datetime.utcnow().isoformat()
                store.update_training_job(
                    job_id=job_id,
                    user_id=user_id,
                    status="running",
                    started_at=now,
                    heartbeat_at=now,
                    control={"action": "running"},
                )
                if artifact_run_id:
                    store.update_status(artifact_run_id, RunStatus.RUNNING)
                store.append_training_job_event(
                    job_id=job_id,
                    user_id=user_id,
                    event_type="started",
                    payload={"job_type": job_type, "model_key": model_key},
                )
                continuous = (
                    request_payload.get("continuous")
                    if isinstance(request_payload.get("continuous"), dict)
                    else {}
                )
                proposal_id = str(continuous.get("proposal_id") or "").strip()
                if proposal_id and job_type == "training":
                    proposal_row = store.get_training_update_proposal(proposal_id=proposal_id)
                    if proposal_row is not None:
                        current_status = normalize_proposal_status(proposal_row.get("status"))
                        if proposal_transition_allowed(current_status, "running"):
                            store.update_training_update_proposal(
                                proposal_id=proposal_id,
                                status="running",
                                linked_job_id=job_id,
                                started_at=now,
                            )

                def _progress(event: dict[str, Any]) -> None:
                    event_payload = {**event, "ts": datetime.utcnow().isoformat()}
                    current = store.get_training_job(job_id=job_id, user_id=user_id) or {}
                    result_payload = (
                        dict(current.get("result"))
                        if isinstance(current.get("result"), dict)
                        else {}
                    )
                    result_payload["last_progress"] = event_payload
                    if isinstance(event_payload.get("epoch"), int) and isinstance(
                        event_payload.get("epochs"), int
                    ):
                        result_payload["progress"] = {
                            "epoch": int(event_payload["epoch"]),
                            "epochs": int(event_payload["epochs"]),
                            "ratio": float(event_payload["epoch"])
                            / float(max(1, int(event_payload["epochs"]))),
                        }
                    checkpoint_path = str(event_payload.get("checkpoint_path") or "").strip()
                    if checkpoint_path:
                        checkpoint_paths = (
                            list(result_payload.get("checkpoint_paths"))
                            if isinstance(result_payload.get("checkpoint_paths"), list)
                            else []
                        )
                        if checkpoint_path not in checkpoint_paths:
                            checkpoint_paths.append(checkpoint_path)
                        result_payload["checkpoint_paths"] = checkpoint_paths[-500:]
                    store.update_training_job(
                        job_id=job_id,
                        user_id=user_id,
                        result=result_payload,
                        heartbeat_at=datetime.utcnow().isoformat(),
                    )
                    store.append_training_job_event(
                        job_id=job_id,
                        user_id=user_id,
                        event_type="progress",
                        payload=event_payload,
                    )

                if job_type == "training":
                    dataset_id = str(row.get("dataset_id") or "").strip()
                    dataset_row, _, manifest = _build_training_dataset_manifest(
                        dataset_id=dataset_id,
                        user_id=user_id,
                        model_key=model_key,
                    )
                    initial_checkpoint_path = (
                        str(request_payload.get("initial_checkpoint_path") or "").strip() or None
                    )
                    training_result = training_runner.run_training(
                        model_key=model_key,
                        manifest=manifest,
                        config=config,
                        output_dir=output_root,
                        progress_callback=_progress,
                        control_callback=lambda: _training_control_callback(job_id, user_id),
                        initial_checkpoint_path=initial_checkpoint_path,
                    )
                    model_version = build_model_version(model_key, job_id)
                    training_result["model_version"] = model_version
                    training_result["manifest_counts"] = manifest.get("counts") or {}
                    training_result["dataset_name"] = str(dataset_row.get("name") or "")
                    entries = _collect_job_output_artifacts(
                        artifact_run_id=artifact_run_id,
                        output_root=output_root,
                        category="training_output",
                        tool_name="model_training",
                    )
                    if entries:
                        _update_manifest_with_entries(artifact_run_id, entries)
                    finished = datetime.utcnow().isoformat()
                    training_result = _continuous_success_update(
                        job_row=row,
                        request_payload=request_payload,
                        training_result=training_result,
                        model_version=model_version,
                        artifact_run_id=artifact_run_id or None,
                        finished_at=finished,
                    )
                    store.update_training_job(
                        job_id=job_id,
                        user_id=user_id,
                        status="succeeded",
                        model_version=model_version,
                        result=training_result,
                        finished_at=finished,
                        heartbeat_at=finished,
                        control={"action": "idle"},
                    )
                    store.append_training_job_event(
                        job_id=job_id,
                        user_id=user_id,
                        event_type="completed",
                        payload={"model_version": model_version, "output_root": str(output_root)},
                    )
                    if artifact_run_id:
                        store.update_status(artifact_run_id, RunStatus.SUCCEEDED)
                else:
                    file_ids = [
                        str(item or "").strip()
                        for item in (request_payload.get("file_ids") or [])
                        if str(item or "").strip()
                    ]
                    resolved_paths, _, missing = _resolve_file_ids(file_ids, user_id=user_id)
                    if missing:
                        raise RuntimeError(
                            "Inference file_ids are missing from upload store: "
                            + ", ".join(missing[:5])
                        )
                    model_version = str(row.get("model_version") or "").strip() or None
                    model_artifact_path = (
                        str(request_payload.get("model_artifact_path") or "").strip() or None
                    )
                    inference_result = training_runner.run_inference(
                        model_key=model_key,
                        model_artifact_path=model_artifact_path,
                        input_paths=resolved_paths,
                        config=config,
                        output_dir=output_root,
                        progress_callback=_progress,
                        control_callback=lambda: _training_control_callback(job_id, user_id),
                    )
                    inference_result["reviewed_samples"] = int(
                        request_payload.get("reviewed_samples") or 0
                    )
                    inference_result["reviewed_failures"] = int(
                        request_payload.get("reviewed_failures") or 0
                    )
                    inference_result["model_version"] = model_version or "builtin"
                    entries = _collect_job_output_artifacts(
                        artifact_run_id=artifact_run_id,
                        output_root=output_root,
                        category="inference_output",
                        tool_name="model_inference",
                    )
                    if entries:
                        _update_manifest_with_entries(artifact_run_id, entries)
                    finished = datetime.utcnow().isoformat()
                    store.update_training_job(
                        job_id=job_id,
                        user_id=user_id,
                        status="succeeded",
                        result=inference_result,
                        finished_at=finished,
                        heartbeat_at=finished,
                        control={"action": "idle"},
                    )
                    store.append_training_job_event(
                        job_id=job_id,
                        user_id=user_id,
                        event_type="completed",
                        payload={
                            "prediction_count": int(inference_result.get("prediction_count") or 0)
                        },
                    )
                    if artifact_run_id:
                        store.update_status(artifact_run_id, RunStatus.SUCCEEDED)
            except TrainingCancelledError as exc:
                finished = datetime.utcnow().isoformat()
                store.update_training_job(
                    job_id=job_id,
                    user_id=user_id,
                    status="canceled",
                    error=str(exc),
                    finished_at=finished,
                    heartbeat_at=finished,
                    control={"action": "idle"},
                )
                _continuous_failure_update(
                    request_payload=request_payload if isinstance(request_payload, dict) else {},
                    error=str(exc),
                    finished_at=finished,
                )
                store.append_training_job_event(
                    job_id=job_id,
                    user_id=user_id,
                    event_type="canceled",
                    payload={"error": str(exc)},
                )
                row_after = store.get_training_job(job_id=job_id, user_id=user_id)
                run_id = str((row_after or {}).get("artifact_run_id") or "").strip()
                if run_id:
                    store.update_status(run_id, RunStatus.CANCELED, error=str(exc))
            except Exception as exc:
                finished = datetime.utcnow().isoformat()
                current = store.get_training_job(job_id=job_id, user_id=user_id) or {}
                current_result = (
                    current.get("result") if isinstance(current.get("result"), dict) else {}
                )
                error_text = str(exc)
                failure_stage = (
                    "failed_pre_eval"
                    if "failed_pre_eval" in error_text.strip().lower()
                    else "failed"
                )
                store.update_training_job(
                    job_id=job_id,
                    user_id=user_id,
                    status="failed",
                    error=error_text,
                    result={**current_result, "failure_stage": failure_stage},
                    finished_at=finished,
                    heartbeat_at=finished,
                    control={"action": "idle"},
                )
                _continuous_failure_update(
                    request_payload=request_payload if isinstance(request_payload, dict) else {},
                    error=error_text,
                    finished_at=finished,
                )
                store.append_training_job_event(
                    job_id=job_id,
                    user_id=user_id,
                    event_type="failed",
                    payload={"error": error_text, "failure_stage": failure_stage},
                )
                row_after = store.get_training_job(job_id=job_id, user_id=user_id)
                run_id = str((row_after or {}).get("artifact_run_id") or "").strip()
                if run_id:
                    store.update_status(run_id, RunStatus.FAILED, error=error_text)
            finally:
                with training_job_threads_lock:
                    training_job_threads.pop(job_id, None)

        thread = Thread(target=_worker, name=f"training-job-{job_id[:10]}", daemon=True)
        with training_job_threads_lock:
            training_job_threads[job_id] = thread
        thread.start()

    _recover_stale_training_jobs()

    def _run_continuous_trigger_evaluator_once() -> None:
        if not bool(getattr(settings, "training_auto_proposals_enabled", False)):
            return
        if not store.acquire_training_scheduler_lease(
            lease_name="continuous-trigger-evaluator",
            owner_id=continuous_scheduler_owner,
            ttl_seconds=continuous_scheduler_ttl_seconds,
        ):
            return
        try:
            lineages = store.list_training_lineages(
                include_shared=True,
                limit=5000,
            )
            for lineage in lineages:
                model_key = str(lineage.get("model_key") or "").strip().lower()
                if model_key != prairie_model_key:
                    continue
                metadata = (
                    lineage.get("metadata") if isinstance(lineage.get("metadata"), dict) else {}
                )
                dataset_id = str(metadata.get("default_dataset_id") or "").strip()
                if not dataset_id:
                    continue
                lineage_id = str(lineage.get("lineage_id") or "").strip()
                if not lineage_id:
                    continue
                open_proposals = store.list_training_update_proposals(
                    lineage_id=lineage_id,
                    statuses=_open_proposal_statuses(),
                    include_shared=True,
                    limit=1,
                )
                if open_proposals:
                    continue
                owner_user_id = str(lineage.get("owner_user_id") or "").strip()
                if not owner_user_id:
                    continue
                try:
                    _, _, manifest = _build_training_dataset_manifest(
                        dataset_id=dataset_id,
                        user_id=owner_user_id,
                        model_key=model_key,
                    )
                except Exception:
                    continue
                manifest_counts = (
                    manifest.get("counts") if isinstance(manifest.get("counts"), dict) else {}
                )
                train_counts = (
                    manifest_counts.get("train")
                    if isinstance(manifest_counts.get("train"), dict)
                    else {}
                )
                approved_new_samples = int(train_counts.get("samples") or 0)
                class_counts = {
                    str(key): int(value)
                    for key, value in (
                        metadata.get("class_counts")
                        if isinstance(metadata.get("class_counts"), dict)
                        else {}
                    ).items()
                    if str(key).strip()
                }
                active_version = _lineage_active_version_row(lineage)
                last_promoted_at = (
                    str(active_version.get("updated_at") or "").strip()
                    if isinstance(active_version, dict)
                    else None
                ) or str(lineage.get("updated_at") or "").strip()
                health_status = str(metadata.get("health_status") or "").strip() or None
                trigger = _build_trigger_snapshot(
                    lineage_id=lineage_id,
                    dataset_id=dataset_id,
                    approved_new_samples=approved_new_samples,
                    class_counts=class_counts,
                    health_status=health_status,
                    last_promoted_at=last_promoted_at,
                )
                if not bool(trigger.get("should_trigger")):
                    continue
                _seed_lineage_replay_items(lineage_row=lineage)
                replay_items = store.list_training_replay_items(lineage_id=lineage_id, limit=5000)
                replay_mix = build_replay_mix_plan(
                    lineage_id=lineage_id,
                    new_samples=approved_new_samples,
                    replay_items=replay_items,
                )
                window = int(time.time() // max(300, continuous_scheduler_interval_seconds))
                idempotency_key = (
                    f"auto:{lineage_id}:{trigger.get('reason') or 'data_threshold'!s}:{window}"
                )
                store.create_training_update_proposal(
                    proposal_id=f"proposal-{uuid4().hex[:18]}",
                    lineage_id=lineage_id,
                    trigger_reason=str(trigger.get("reason") or "data_threshold"),
                    trigger_snapshot=trigger,
                    dataset_snapshot={
                        "dataset_id": dataset_id,
                        "manifest_counts": manifest_counts,
                        "approved_new_samples": approved_new_samples,
                        "class_counts": class_counts,
                        "replay_mix": replay_mix,
                    },
                    config=(
                        metadata.get("default_config")
                        if isinstance(metadata.get("default_config"), dict)
                        else {}
                    ),
                    status="pending_approval",
                    idempotency_key=idempotency_key,
                )
        finally:
            store.release_training_scheduler_lease(
                lease_name="continuous-trigger-evaluator",
                owner_id=continuous_scheduler_owner,
            )

    def _run_continuous_scheduler_iteration(
        evaluator: Callable[[], None] | None = None,
    ) -> None:
        runner = evaluator or _run_continuous_trigger_evaluator_once
        try:
            runner()
        except Exception:
            logger.exception(
                "Continuous trigger evaluator iteration failed.",
                extra={"scheduler_owner": continuous_scheduler_owner},
            )

    def _continuous_scheduler_loop() -> None:
        while not continuous_scheduler_stop.is_set():
            _run_continuous_scheduler_iteration()
            interrupted = continuous_scheduler_stop.wait(continuous_scheduler_interval_seconds)
            if interrupted:
                break

    def _start_continuous_scheduler() -> None:
        nonlocal continuous_scheduler_thread
        if continuous_scheduler_thread is not None and continuous_scheduler_thread.is_alive():
            return
        continuous_scheduler_stop.clear()
        continuous_scheduler_thread = Thread(
            target=_continuous_scheduler_loop,
            name="continuous-trigger-scheduler",
            daemon=True,
        )
        continuous_scheduler_thread.start()

    def _run_prairie_sync_scheduler_once() -> None:
        if not store.acquire_training_scheduler_lease(
            lease_name="prairie-sync-scheduler",
            owner_id=prairie_sync_scheduler_owner,
            ttl_seconds=max(120, prairie_sync_interval_seconds // 2),
        ):
            return
        try:
            managed_sources = store.list_training_managed_sources(
                source_type=prairie_source_type,
                limit=5000,
            )
            for source in managed_sources:
                user_id = str(source.get("user_id") or "").strip()
                dataset_name = str(source.get("name") or "").strip()
                if not user_id or not dataset_name:
                    continue
                try:
                    _sync_prairie_active_learning_dataset(
                        user_id=user_id,
                        bisque_auth=None,
                        trigger="auto",
                    )
                except Exception:
                    continue
        finally:
            store.release_training_scheduler_lease(
                lease_name="prairie-sync-scheduler",
                owner_id=prairie_sync_scheduler_owner,
            )

    def _prairie_sync_scheduler_loop() -> None:
        while not prairie_sync_scheduler_stop.is_set():
            with suppress(Exception):
                _run_prairie_sync_scheduler_once()
            interrupted = prairie_sync_scheduler_stop.wait(prairie_sync_interval_seconds)
            if interrupted:
                break

    def _start_prairie_sync_scheduler() -> None:
        nonlocal prairie_sync_scheduler_thread
        if prairie_sync_scheduler_thread is not None and prairie_sync_scheduler_thread.is_alive():
            return
        prairie_sync_scheduler_stop.clear()
        prairie_sync_scheduler_thread = Thread(
            target=_prairie_sync_scheduler_loop,
            name="prairie-sync-scheduler",
            daemon=True,
        )
        prairie_sync_scheduler_thread.start()

    @v1.get("/training/models", response_model=TrainingModelsResponse)
    @legacy.get("/training/models", response_model=TrainingModelsResponse)
    def list_training_models_endpoint(
        _auth: None = Depends(_require_api_key),
    ) -> TrainingModelsResponse:
        del _auth
        rows = [
            TrainingModelRecord(
                key=model.key,
                name=model.name,
                framework=model.framework,
                task_type=model.task_type,
                description=model.description,
                supports_training=model.supports_training,
                supports_finetune=model.supports_finetune,
                supports_inference=model.supports_inference,
                dimensions=list(model.dimensions),
                default_config=dict(model.default_config),
            )
            for model in list_model_definitions()
        ]
        return TrainingModelsResponse(count=len(rows), models=rows)

    @v1.post("/training/prairie/sync", response_model=PrairieSyncResponse)
    @legacy.post("/training/prairie/sync", response_model=PrairieSyncResponse)
    def sync_prairie_active_learning_dataset_endpoint(
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> PrairieSyncResponse:
        del _auth
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        return _sync_prairie_active_learning_dataset(
            user_id=user_id,
            bisque_auth=bisque_auth,
            trigger="manual",
        )

    @v1.get("/training/prairie/status", response_model=PrairieStatusResponse)
    @legacy.get("/training/prairie/status", response_model=PrairieStatusResponse)
    def get_prairie_active_learning_status_endpoint(
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> PrairieStatusResponse:
        del _auth
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        dataset_name = str(
            getattr(settings, "prairie_active_learning_dataset_name", "Prairie_Dog_Active_Learning")
            or "Prairie_Dog_Active_Learning"
        ).strip()
        source_id = _prairie_source_id(user_id)
        source_row = store.get_training_managed_source(source_id=source_id, user_id=user_id)
        if source_row is None:
            source_row = store.upsert_training_managed_source(
                source_id=source_id,
                user_id=user_id,
                source_type=prairie_source_type,
                name=dataset_name,
                status="idle",
                metadata={},
            )
        stats = source_row.get("stats") if isinstance(source_row.get("stats"), dict) else {}
        dataset_id = str(source_row.get("training_dataset_id") or "").strip() or None
        last_sync_at = str(source_row.get("last_sync_at") or "").strip() or None
        next_sync_at: str | None = None
        if last_sync_at:
            next_sync_at = (
                _parse_datetime(last_sync_at) + timedelta(seconds=prairie_sync_interval_seconds)
            ).isoformat()

        training_jobs = store.list_training_jobs(
            user_id=user_id,
            job_type="training",
            model_key=prairie_model_key,
            limit=500,
        )
        inference_jobs = store.list_training_jobs(
            user_id=user_id,
            job_type="inference",
            model_key=prairie_model_key,
            limit=500,
        )
        health_rows = compute_model_health_entries(
            training_jobs=training_jobs,
            inference_jobs=inference_jobs,
        )
        yolo_health = next(
            (
                row
                for row in health_rows
                if str(row.get("model_key") or "").strip().lower() == prairie_model_key
            ),
            None,
        )

        active_version_id: str | None = None
        latest_metrics: dict[str, Any] = {}
        active_version_metrics: dict[str, Any] = {}
        try:
            _, lineage_row = _ensure_default_domain_and_lineage(
                user_id=user_id,
                model_key=prairie_model_key,
            )
            active_version_id = str(lineage_row.get("active_version_id") or "").strip() or None
            active_version_row = _lineage_active_version_row(lineage_row)
            if isinstance(active_version_row, dict) and isinstance(
                active_version_row.get("metrics"), dict
            ):
                active_version_metrics = active_version_row.get("metrics") or {}
        except Exception:
            active_version_id = None
        succeeded_training = [
            row
            for row in training_jobs
            if str(row.get("status") or "").strip().lower() == "succeeded"
        ]
        if succeeded_training:
            latest_result = (
                succeeded_training[0].get("result")
                if isinstance(succeeded_training[0].get("result"), dict)
                else {}
            )
            latest_metrics = (
                latest_result.get("metrics")
                if isinstance(latest_result.get("metrics"), dict)
                else {}
            )
        benchmark_source = active_version_metrics if active_version_metrics else latest_metrics
        (
            benchmark_baseline,
            benchmark_latest_candidate,
            last_benchmark_at,
            benchmark_ready,
            canonical_ready,
            promotion_ready,
        ) = _extract_benchmark_status(benchmark_source)
        if not last_benchmark_at:
            (
                _,
                _,
                last_benchmark_at,
                inferred_ready,
                inferred_canonical_ready,
                inferred_promotion_ready,
            ) = _extract_benchmark_status(latest_metrics)
            if inferred_ready:
                benchmark_ready = True
            if inferred_canonical_ready:
                canonical_ready = True
            if inferred_promotion_ready:
                promotion_ready = True

        retrain_gate = _evaluate_prairie_retrain_gate(stats)

        detection_counts = _collect_prairie_detection_counts(user_id)
        return PrairieStatusResponse(
            dataset_name=dataset_name,
            dataset_id=dataset_id,
            last_sync_at=last_sync_at,
            next_sync_at=next_sync_at,
            active_model_version=active_version_id,
            model_health=(
                str(yolo_health.get("status") or "Needs Human Review")
                if isinstance(yolo_health, dict)
                else "Needs Human Review"
            ),
            reviewed_images=int(stats.get("reviewed_images") or 0),
            unreviewed_images=int(stats.get("unreviewed_images") or 0),
            class_counts={
                "burrow": int((stats.get("class_counts") or {}).get("burrow") or 0)
                if isinstance(stats.get("class_counts"), dict)
                else 0,
                "prairie_dog": int((stats.get("class_counts") or {}).get("prairie_dog") or 0)
                if isinstance(stats.get("class_counts"), dict)
                else 0,
            },
            unsupported_class_counts=(
                {
                    str(key): int(value)
                    for key, value in (stats.get("unsupported_class_counts") or {}).items()
                    if str(key).strip()
                }
                if isinstance(stats.get("unsupported_class_counts"), dict)
                else {}
            ),
            detection_counts=detection_counts,
            latest_metrics=latest_metrics,
            benchmark_baseline=benchmark_baseline,
            benchmark_latest_candidate=benchmark_latest_candidate,
            last_benchmark_at=last_benchmark_at,
            benchmark_ready=benchmark_ready,
            canonical_benchmark_ready=canonical_ready,
            promotion_benchmark_ready=promotion_ready,
            retrain_gate=bool(retrain_gate.get("ready")),
            retrain_gate_reasons=[
                str(item) for item in list(retrain_gate.get("reasons") or []) if str(item).strip()
            ],
            retrain_gate_counts={
                str(key): int(value)
                for key, value in (retrain_gate.get("counts") or {}).items()
                if str(key).strip()
            },
        )

    @v1.post("/training/prairie/benchmark/run", response_model=PrairieBenchmarkRunResponse)
    @legacy.post("/training/prairie/benchmark/run", response_model=PrairieBenchmarkRunResponse)
    def run_prairie_benchmark_endpoint(
        req: PrairieBenchmarkRunRequest = Body(default=PrairieBenchmarkRunRequest()),
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> PrairieBenchmarkRunResponse:
        del _auth
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        benchmark_mode = str(req.mode or "canonical_only").strip().lower() or "canonical_only"
        if benchmark_mode not in {"canonical_only", "promotion_packet"}:
            benchmark_mode = "canonical_only"
        definition = get_model_definition(prairie_model_key)
        if definition is None:
            raise HTTPException(
                status_code=400,
                detail="YOLOv5 RareSpot model is disabled. Enable TRAINING_ENABLE_YOLOV5_RARESPOT.",
            )
        _, lineage_row = _ensure_default_domain_and_lineage(
            user_id=user_id,
            model_key=prairie_model_key,
        )
        lineage_row = _assert_training_lineage_mutation_access(
            row=lineage_row,
            user_id=user_id,
            action="Benchmarking the shared prairie model",
        )
        active_version_id = str(lineage_row.get("active_version_id") or "").strip() or None
        model_artifact_path: str | None = None
        active_version_row = _lineage_active_version_row(lineage_row)
        if isinstance(active_version_row, dict):
            active_meta = (
                active_version_row.get("metadata")
                if isinstance(active_version_row.get("metadata"), dict)
                else {}
            )
            model_artifact_path = (
                str(active_meta.get("model_artifact_path") or "").strip()
                or str(active_meta.get("checkpoint_path") or "").strip()
                or None
            )
            if model_artifact_path:
                candidate_path = Path(model_artifact_path).expanduser()
                if not candidate_path.exists() or not candidate_path.is_file():
                    model_artifact_path = None
                else:
                    model_artifact_path = str(candidate_path.resolve())

        run_id = _create_training_artifact_run(
            user_id=user_id,
            goal=f"prairie-benchmark:{active_version_id or 'builtin'}",
        )
        output_root = (
            _ensure_run_artifact_dir(run_id) / "model_jobs" / f"benchmark-{uuid4().hex[:14]}"
        )
        output_root.mkdir(parents=True, exist_ok=True)
        store.update_status(run_id, RunStatus.RUNNING)
        config = _normalize_training_config(
            model_key=prairie_model_key,
            raw_config={},
            default_config=dict(definition.default_config),
        )
        config["benchmark_mode"] = benchmark_mode
        config["_active_packet_ready"] = _promotion_benchmark_packet_complete(
            active_version_row.get("metrics")
            if isinstance(active_version_row, dict)
            and isinstance(active_version_row.get("metrics"), dict)
            else {}
        )
        progress_events: list[dict[str, Any]] = []

        def _progress(event: dict[str, Any]) -> None:
            payload = {**event, "ts": datetime.utcnow().isoformat()}
            progress_events.append(payload)
            store.append_event(run_id, "benchmark_progress", payload)

        try:
            report = training_runner.run_benchmark(
                model_key=prairie_model_key,
                model_artifact_path=model_artifact_path,
                config=config,
                output_dir=output_root,
                progress_callback=_progress,
                control_callback=lambda: None,
            )
            entries = _collect_job_output_artifacts(
                artifact_run_id=run_id,
                output_root=output_root,
                category="benchmark_output",
                tool_name="model_benchmark",
            )
            if entries:
                _update_manifest_with_entries(run_id, entries)
            store.append_event(
                run_id,
                "benchmark_completed",
                {
                    "model_key": prairie_model_key,
                    "model_version": active_version_id or "builtin",
                    "mode": benchmark_mode,
                    "benchmark_ready": bool(report.get("benchmark_ready")),
                    "canonical_benchmark_ready": bool(report.get("canonical_benchmark_ready")),
                    "promotion_benchmark_ready": bool(report.get("promotion_benchmark_ready")),
                    "progress_events": len(progress_events),
                },
            )
            store.update_status(run_id, RunStatus.SUCCEEDED)

            if active_version_id and isinstance(active_version_row, dict):
                current_metrics = (
                    active_version_row.get("metrics")
                    if isinstance(active_version_row.get("metrics"), dict)
                    else {}
                )
                canonical_ready = bool(report.get("canonical_benchmark_ready"))
                promotion_ready = bool(report.get("promotion_benchmark_ready"))
                manual_benchmark = {
                    "mode": benchmark_mode,
                    "canonical": (
                        report.get("canonical") if isinstance(report.get("canonical"), dict) else {}
                    ),
                    "canonical_benchmark_ready": canonical_ready,
                    "promotion_benchmark_ready": promotion_ready,
                    "report_path": str(report.get("report_path") or "").strip() or None,
                    "last_benchmark_at": str(report.get("last_benchmark_at") or ""),
                }
                updated_metrics = {
                    **current_metrics,
                    "manual_benchmark": manual_benchmark,
                    "last_benchmark_at": str(report.get("last_benchmark_at") or ""),
                    "benchmark_ready": bool(current_metrics.get("benchmark_ready"))
                    or promotion_ready,
                    "canonical_benchmark_ready": bool(
                        current_metrics.get("canonical_benchmark_ready")
                    )
                    or canonical_ready,
                    "promotion_benchmark_ready": bool(
                        current_metrics.get("promotion_benchmark_ready")
                    )
                    or promotion_ready,
                }
                store.update_training_model_version(
                    version_id=active_version_id,
                    metrics=updated_metrics,
                )
            return PrairieBenchmarkRunResponse(
                run_id=run_id,
                model_version=active_version_id or "builtin",
                mode=benchmark_mode,  # type: ignore[arg-type]
                benchmark_ready=bool(report.get("benchmark_ready")),
                canonical_benchmark_ready=bool(report.get("canonical_benchmark_ready")),
                promotion_benchmark_ready=bool(report.get("promotion_benchmark_ready")),
                report=report if isinstance(report, dict) else {},
            )
        except Exception as exc:
            store.append_event(run_id, "benchmark_failed", {"error": str(exc)})
            store.update_status(run_id, RunStatus.FAILED, error=str(exc))
            raise HTTPException(status_code=500, detail=f"Benchmark run failed: {exc}") from exc

    @v1.post("/training/prairie/retrain-request", response_model=TrainingJobResponse)
    @legacy.post("/training/prairie/retrain-request", response_model=TrainingJobResponse)
    def create_prairie_retrain_request_endpoint(
        req: PrairieRetrainRequest,
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> TrainingJobResponse:
        del _auth
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        if not req.confirm_launch:
            raise HTTPException(
                status_code=400,
                detail="confirm_launch must be true for retraining launch.",
            )
        if _active_training_count("training") >= 1:
            raise HTTPException(
                status_code=409,
                detail="Another training job is active. v1 supports one training job at a time.",
            )
        definition = get_model_definition(prairie_model_key)
        if definition is None:
            raise HTTPException(
                status_code=400,
                detail="YOLOv5 RareSpot model is disabled. Enable TRAINING_ENABLE_YOLOV5_RARESPOT.",
            )
        source_row = store.get_training_managed_source(
            source_id=_prairie_source_id(user_id),
            user_id=user_id,
        )
        dataset_id = (
            str(source_row.get("training_dataset_id") or "").strip()
            if isinstance(source_row, dict)
            else ""
        )
        if not dataset_id:
            raise HTTPException(
                status_code=400,
                detail="Prairie dataset has not been synced yet. Run Sync now first.",
            )
        stats = source_row.get("stats") if isinstance(source_row.get("stats"), dict) else {}
        retrain_gate = _evaluate_prairie_retrain_gate(stats)
        if not bool(retrain_gate.get("ready")):
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Prairie retraining gate is not satisfied.",
                    "retrain_gate": retrain_gate,
                },
            )
        config = _normalize_training_config(
            model_key=prairie_model_key,
            raw_config={},
            default_config=dict(definition.default_config),
        )
        dataset_row, _, manifest = _build_training_dataset_manifest(
            dataset_id=dataset_id,
            user_id=user_id,
            model_key=prairie_model_key,
        )
        preflight = build_preflight_report(
            model_key=prairie_model_key,
            dataset_manifest=manifest,
            config=config,
            artifact_root=artifact_root,
            model_dimensions=[
                str(item or "").strip().lower()
                for item in definition.dimensions
                if str(item or "").strip()
            ],
            max_dimension_samples=int(
                getattr(settings, "training_dimension_check_max_samples", 512) or 512
            ),
        )
        if not bool(preflight.get("recommended_launch")):
            raise HTTPException(
                status_code=400,
                detail=_format_preflight_failure_detail(preflight),
            )
        _, lineage_row = _ensure_default_domain_and_lineage(
            user_id=user_id,
            model_key=prairie_model_key,
        )
        lineage_row = _assert_training_lineage_mutation_access(
            row=lineage_row,
            user_id=user_id,
            action="Launching prairie retraining",
        )
        active_version_row = _lineage_active_version_row(lineage_row)
        active_meta = (
            active_version_row.get("metadata")
            if isinstance(active_version_row, dict)
            and isinstance(active_version_row.get("metadata"), dict)
            else {}
        )
        initial_checkpoint_path = (
            str(active_meta.get("model_artifact_path") or "").strip()
            or str(active_meta.get("checkpoint_path") or "").strip()
            or None
        )
        if initial_checkpoint_path:
            checkpoint_path = Path(initial_checkpoint_path).expanduser()
            if not checkpoint_path.exists() or not checkpoint_path.is_file():
                initial_checkpoint_path = None
            else:
                initial_checkpoint_path = str(checkpoint_path.resolve())
        job_id = f"train-{uuid4().hex[:18]}"
        artifact_run_id = _create_training_artifact_run(
            user_id=user_id,
            goal=f"prairie-retrain:{dataset_row.get('name')}",
        )
        request_payload = {
            "dataset_id": dataset_id,
            "config": config,
            "preflight": preflight,
            "manifest_counts": manifest.get("counts") or {},
            "dataset_name": str(dataset_row.get("name") or ""),
            "confirmed": True,
            "prairie_active_learning": True,
            "note": str(req.note or "").strip() or None,
            "initial_checkpoint_path": initial_checkpoint_path,
            "retrain_gate": retrain_gate,
            "continuous": {
                "lineage_id": str(lineage_row.get("lineage_id") or ""),
                "trigger_reason": "manual_retrain",
            },
        }
        row = store.create_training_job(
            job_id=job_id,
            user_id=user_id,
            job_type="training",
            dataset_id=dataset_id,
            model_key=prairie_model_key,
            model_version=None,
            status="queued",
            request=request_payload,
            result={
                "preflight": preflight,
                "prairie_retrain_request": True,
                "retrain_gate": retrain_gate,
            },
            control={"action": "queued"},
            artifact_run_id=artifact_run_id,
        )
        store.append_training_job_event(
            job_id=job_id,
            user_id=user_id,
            event_type="prairie_retrain_requested",
            payload={"dataset_id": dataset_id, "note": str(req.note or "").strip() or None},
        )
        _launch_training_worker(job_id=job_id, user_id=user_id)
        return TrainingJobResponse(job=_to_training_job_record(row))

    @v1.get("/training/prairie/retrain-requests", response_model=PrairieRetrainListResponse)
    @legacy.get("/training/prairie/retrain-requests", response_model=PrairieRetrainListResponse)
    def list_prairie_retrain_requests_endpoint(
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> PrairieRetrainListResponse:
        del _auth
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        rows = _list_prairie_retrain_jobs(user_id, limit=500)
        records: list[PrairieRetrainRecord] = []
        for row in rows:
            request_payload = row.get("request") if isinstance(row.get("request"), dict) else {}
            result_payload = row.get("result") if isinstance(row.get("result"), dict) else {}
            retrain_gate_payload = (
                request_payload.get("retrain_gate")
                if isinstance(request_payload.get("retrain_gate"), dict)
                else result_payload.get("retrain_gate")
                if isinstance(result_payload.get("retrain_gate"), dict)
                else {}
            )
            status_token = str(row.get("status") or "").strip().lower()
            if status_token not in {
                "queued",
                "running",
                "paused",
                "succeeded",
                "failed",
                "canceled",
            }:
                status_token = "failed"
            artifact_run_id = str(row.get("artifact_run_id") or "").strip() or None
            gating_summary = _build_prairie_gating_summary(result_payload)
            if retrain_gate_payload:
                gating_summary = {
                    **gating_summary,
                    "retrain_gate_ready": bool(retrain_gate_payload.get("ready")),
                    "retrain_gate_reasons": list(retrain_gate_payload.get("reasons") or []),
                    "retrain_gate_counts": retrain_gate_payload.get("counts")
                    if isinstance(retrain_gate_payload.get("counts"), dict)
                    else {},
                    "retrain_gate_thresholds": retrain_gate_payload.get("thresholds")
                    if isinstance(retrain_gate_payload.get("thresholds"), dict)
                    else {},
                }
            records.append(
                PrairieRetrainRecord(
                    request_id=str(row.get("job_id") or ""),
                    training_job_id=str(row.get("job_id") or ""),
                    status=status_token,  # type: ignore[arg-type]
                    created_at=str(row.get("created_at") or ""),
                    started_at=str(row.get("started_at") or "").strip() or None,
                    finished_at=str(row.get("finished_at") or "").strip() or None,
                    model_version=str(row.get("model_version") or "").strip() or None,
                    note=str(request_payload.get("note") or "").strip() or None,
                    error=str(row.get("error") or "").strip() or None,
                    gating_summary=gating_summary,
                    benchmark_report_artifact_id=artifact_run_id,
                )
            )
        return PrairieRetrainListResponse(count=len(records), requests=records)

    @v1.post("/training/datasets", response_model=TrainingDatasetResponse)
    @legacy.post("/training/datasets", response_model=TrainingDatasetResponse)
    def create_training_dataset_endpoint(
        req: TrainingDatasetCreateRequest,
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> TrainingDatasetResponse:
        del _auth
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        dataset_id = f"dataset-{uuid4().hex[:14]}"
        row = store.create_training_dataset(
            dataset_id=dataset_id,
            user_id=user_id,
            name=req.name,
            description=req.description,
            metadata=req.metadata,
        )
        item_rows = store.list_training_dataset_items(dataset_id=dataset_id, user_id=user_id)
        record = _dataset_record_from_rows(dataset_row=row, item_rows=item_rows, manifest={})
        return TrainingDatasetResponse(dataset=record, manifest={})

    @v1.get("/training/datasets", response_model=TrainingDatasetListResponse)
    @legacy.get("/training/datasets", response_model=TrainingDatasetListResponse)
    def list_training_datasets_endpoint(
        limit: int = Query(default=200, ge=1, le=1000),
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> TrainingDatasetListResponse:
        del _auth
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        rows = store.list_training_datasets(user_id=user_id, limit=limit)
        records: list[TrainingDatasetRecord] = []
        for row in rows:
            item_rows = store.list_training_dataset_items(
                dataset_id=str(row.get("dataset_id") or ""),
                user_id=user_id,
            )
            manifest: dict[str, Any] = {}
            try:
                manifest = build_dataset_manifest(
                    dataset_id=str(row.get("dataset_id") or ""),
                    dataset_name=str(row.get("name") or ""),
                    items=item_rows,
                )
            except Exception:
                manifest = {}
            records.append(
                _dataset_record_from_rows(dataset_row=row, item_rows=item_rows, manifest=manifest)
            )
        return TrainingDatasetListResponse(count=len(records), datasets=records)

    @v1.get("/training/datasets/{dataset_id}", response_model=TrainingDatasetResponse)
    @legacy.get("/training/datasets/{dataset_id}", response_model=TrainingDatasetResponse)
    def get_training_dataset_endpoint(
        dataset_id: str,
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> TrainingDatasetResponse:
        del _auth
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        dataset_row, item_rows, manifest = _build_training_dataset_manifest(
            dataset_id=dataset_id,
            user_id=user_id,
        )
        record = _dataset_record_from_rows(
            dataset_row=dataset_row,
            item_rows=item_rows,
            manifest=manifest,
        )
        return TrainingDatasetResponse(dataset=record, manifest=manifest)

    @v1.post("/training/datasets/{dataset_id}/items", response_model=TrainingDatasetResponse)
    @legacy.post("/training/datasets/{dataset_id}/items", response_model=TrainingDatasetResponse)
    def assign_training_dataset_items_endpoint(
        dataset_id: str,
        req: TrainingDatasetItemsRequest,
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> TrainingDatasetResponse:
        del _auth
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        dataset_row = store.get_training_dataset(dataset_id=dataset_id, user_id=user_id)
        if dataset_row is None:
            raise HTTPException(status_code=404, detail="Training dataset not found.")
        resolved_items: list[dict[str, Any]] = []
        for assignment in req.items:
            upload_row = _resolve_training_dataset_upload(
                user_id=user_id, file_id=assignment.file_id
            )
            sample_id = str(assignment.sample_id or "").strip()
            if not sample_id:
                sample_id = Path(upload_row["original_name"]).stem or assignment.file_id
            resolved_items.append(
                {
                    "split": assignment.split,
                    "role": assignment.role,
                    "sample_id": sample_id,
                    "file_id": assignment.file_id,
                    "path": upload_row["path"],
                    "original_name": upload_row["original_name"],
                    "sha256": upload_row["sha256"],
                    "size_bytes": upload_row["size_bytes"],
                    "metadata": assignment.metadata,
                }
            )
        if not resolved_items and not req.replace:
            raise HTTPException(status_code=400, detail="No dataset items were provided.")
        store.upsert_training_dataset_items(
            dataset_id=dataset_id,
            user_id=user_id,
            items=resolved_items,
            replace=req.replace,
        )
        dataset_row, item_rows, manifest = _build_training_dataset_manifest(
            dataset_id=dataset_id,
            user_id=user_id,
        )
        merged_metadata = dict(dataset_row.get("metadata") or {})
        merged_metadata["manifest_counts"] = manifest.get("counts") or {}
        store.create_training_dataset(
            dataset_id=dataset_id,
            user_id=user_id,
            name=str(dataset_row.get("name") or ""),
            description=str(dataset_row.get("description") or ""),
            metadata=merged_metadata,
        )
        record = _dataset_record_from_rows(
            dataset_row=dataset_row,
            item_rows=item_rows,
            manifest=manifest,
        )
        return TrainingDatasetResponse(dataset=record, manifest=manifest)

    @v1.post("/training/preflight", response_model=TrainingPreflightResponse)
    @legacy.post("/training/preflight", response_model=TrainingPreflightResponse)
    def training_preflight_endpoint(
        req: TrainingPreflightRequest,
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> TrainingPreflightResponse:
        del _auth
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        model_key = str(req.model_key or "").strip().lower()
        definition = get_model_definition(model_key)
        if definition is None:
            raise HTTPException(status_code=400, detail=f"Unknown model_key: {req.model_key}")
        if not (definition.supports_training or definition.supports_finetune):
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model_key}' does not support training or finetuning.",
            )
        try:
            normalized_config = _normalize_training_config(
                model_key=model_key,
                raw_config=req.config,
                default_config=dict(definition.default_config),
            )
        except DatasetValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        requested_spatial_dims = str(normalized_config.get("spatial_dims") or "").strip().lower()
        supported_dimensions = [
            str(item or "").strip().lower()
            for item in definition.dimensions
            if str(item or "").strip()
        ]
        if (
            definition.task_type == "segmentation"
            and supported_dimensions
            and requested_spatial_dims not in supported_dimensions
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Model '{model_key}' supports {supported_dimensions}; "
                    f"requested spatial_dims={requested_spatial_dims}."
                ),
            )
        _, _, manifest = _build_training_dataset_manifest(
            dataset_id=req.dataset_id,
            user_id=user_id,
            model_key=model_key,
        )
        preflight = build_preflight_report(
            model_key=model_key,
            dataset_manifest=manifest,
            config=normalized_config,
            artifact_root=artifact_root,
            model_dimensions=supported_dimensions,
            max_dimension_samples=int(
                getattr(settings, "training_dimension_check_max_samples", 512) or 512
            ),
        )
        return TrainingPreflightResponse(
            dataset_id=req.dataset_id,
            model_key=model_key,
            config=normalized_config,
            recommended_launch=bool(preflight.get("recommended_launch")),
            report=preflight,
        )

    @v1.post("/training/jobs", response_model=TrainingJobResponse)
    @legacy.post("/training/jobs", response_model=TrainingJobResponse)
    def create_training_job_endpoint(
        req: TrainingJobCreateRequest,
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> TrainingJobResponse:
        del _auth
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        model_key = str(req.model_key or "").strip().lower()
        definition = get_model_definition(model_key)
        if definition is None:
            raise HTTPException(status_code=400, detail=f"Unknown model_key: {req.model_key}")
        if not (definition.supports_training or definition.supports_finetune):
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model_key}' does not support training or finetuning.",
            )
        if not req.confirm_launch:
            raise HTTPException(
                status_code=400,
                detail="confirm_launch must be true for human-in-the-loop training launch.",
            )
        if _active_training_count("training") >= 1:
            raise HTTPException(
                status_code=409,
                detail="Another training job is active. v1 supports one training job at a time.",
            )
        try:
            normalized_config = _normalize_training_config(
                model_key=model_key,
                raw_config=req.config,
                default_config=dict(definition.default_config),
            )
        except DatasetValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        requested_spatial_dims = str(normalized_config.get("spatial_dims") or "").strip().lower()
        supported_dimensions = [
            str(item or "").strip().lower()
            for item in definition.dimensions
            if str(item or "").strip()
        ]
        if (
            definition.task_type == "segmentation"
            and supported_dimensions
            and requested_spatial_dims not in supported_dimensions
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Model '{model_key}' supports {supported_dimensions}; "
                    f"requested spatial_dims={requested_spatial_dims}."
                ),
            )
        initial_checkpoint_path = str(req.initial_checkpoint_path or "").strip() or None
        if initial_checkpoint_path:
            checkpoint_path = Path(initial_checkpoint_path).expanduser()
            if not checkpoint_path.exists() or not checkpoint_path.is_file():
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "initial_checkpoint_path does not exist or is not a file: "
                        f"{initial_checkpoint_path}"
                    ),
                )
            initial_checkpoint_path = str(checkpoint_path.resolve())
        dataset_row, _, manifest = _build_training_dataset_manifest(
            dataset_id=req.dataset_id,
            user_id=user_id,
            model_key=model_key,
        )
        preflight = build_preflight_report(
            model_key=model_key,
            dataset_manifest=manifest,
            config=normalized_config,
            artifact_root=artifact_root,
            model_dimensions=supported_dimensions,
            max_dimension_samples=int(
                getattr(settings, "training_dimension_check_max_samples", 512) or 512
            ),
        )
        if not bool(preflight.get("recommended_launch")):
            raise HTTPException(
                status_code=400,
                detail=_format_preflight_failure_detail(preflight),
            )
        job_id = f"train-{uuid4().hex[:18]}"
        artifact_run_id = _create_training_artifact_run(
            user_id=user_id,
            goal=f"model-training:{model_key}:{dataset_row.get('name')}",
        )
        request_payload = {
            "dataset_id": req.dataset_id,
            "config": normalized_config,
            "preflight": preflight,
            "manifest_counts": manifest.get("counts") or {},
            "dataset_name": str(dataset_row.get("name") or ""),
            "initial_checkpoint_path": initial_checkpoint_path,
            "confirmed": True,
        }
        row = store.create_training_job(
            job_id=job_id,
            user_id=user_id,
            job_type="training",
            dataset_id=req.dataset_id,
            model_key=model_key,
            model_version=None,
            status="queued",
            request=request_payload,
            result={"preflight": preflight},
            control={"action": "queued"},
            artifact_run_id=artifact_run_id,
        )
        store.append_training_job_event(
            job_id=job_id,
            user_id=user_id,
            event_type="queued",
            payload={"preflight": preflight},
        )
        _launch_training_worker(job_id=job_id, user_id=user_id)
        return TrainingJobResponse(job=_to_training_job_record(row))

    @v1.get("/training/jobs/{job_id}", response_model=TrainingJobResponse)
    @legacy.get("/training/jobs/{job_id}", response_model=TrainingJobResponse)
    def get_training_job_endpoint(
        job_id: str,
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> TrainingJobResponse:
        del _auth
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        row = _assert_training_job_owner(
            row=store.get_training_job(job_id=job_id, user_id=user_id),
            user_id=user_id,
        )
        return TrainingJobResponse(job=_to_training_job_record(row))

    @v1.post("/training/jobs/{job_id}/control", response_model=TrainingJobResponse)
    @legacy.post("/training/jobs/{job_id}/control", response_model=TrainingJobResponse)
    def control_training_job_endpoint(
        job_id: str,
        req: TrainingJobControlRequest,
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> TrainingJobResponse:
        del _auth
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        row = _assert_training_job_owner(
            row=store.get_training_job(job_id=job_id, user_id=user_id),
            user_id=user_id,
        )
        action = str(req.action or "").strip().lower()
        current_status = str(row.get("status") or "").strip().lower()
        if not _training_control_transition_allowed(current_status, action):
            raise HTTPException(
                status_code=409,
                detail=(f"Action '{action}' is not allowed when job status is '{current_status}'."),
            )
        if action == "restart":
            if _active_training_count(str(row.get("job_type") or "training")) >= 1:
                raise HTTPException(
                    status_code=409,
                    detail="An active job already occupies this queue slot. Retry after it completes.",
                )
            request_payload = row.get("request") if isinstance(row.get("request"), dict) else {}
            result_payload = row.get("result") if isinstance(row.get("result"), dict) else {}
            checkpoint_paths = result_payload.get("checkpoint_paths")
            last_checkpoint = (
                str(checkpoint_paths[-1]).strip()
                if isinstance(checkpoint_paths, list) and checkpoint_paths
                else None
            )
            job_type = str(row.get("job_type") or "training").strip().lower()
            model_key = str(row.get("model_key") or "").strip().lower()
            if job_type == "training":
                definition = get_model_definition(model_key)
                if definition is None:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unknown model_key for restart: {model_key}",
                    )
                base_config = (
                    request_payload.get("config")
                    if isinstance(request_payload.get("config"), dict)
                    else {}
                )
                try:
                    normalized_config = _normalize_training_config(
                        model_key=model_key,
                        raw_config=base_config,
                        default_config=dict(definition.default_config),
                    )
                except DatasetValidationError as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
                dataset_id = str(row.get("dataset_id") or "").strip()
                dataset_row, _, manifest = _build_training_dataset_manifest(
                    dataset_id=dataset_id,
                    user_id=user_id,
                    model_key=model_key,
                )
                preflight = build_preflight_report(
                    model_key=model_key,
                    dataset_manifest=manifest,
                    config=normalized_config,
                    artifact_root=artifact_root,
                    model_dimensions=[
                        str(item or "").strip().lower()
                        for item in definition.dimensions
                        if str(item or "").strip()
                    ],
                    max_dimension_samples=int(
                        getattr(settings, "training_dimension_check_max_samples", 512) or 512
                    ),
                )
                if not bool(preflight.get("recommended_launch")):
                    raise HTTPException(
                        status_code=400,
                        detail=_format_preflight_failure_detail(preflight),
                    )
                request_payload = {
                    **request_payload,
                    "config": normalized_config,
                    "preflight": preflight,
                    "manifest_counts": manifest.get("counts") or {},
                    "dataset_name": str(dataset_row.get("name") or ""),
                }
            request_payload = {
                **request_payload,
                "initial_checkpoint_path": last_checkpoint,
            }
            new_job_id = f"{str(row.get('job_type') or 'job')[:5]}-{uuid4().hex[:18]}"
            artifact_run_id = _create_training_artifact_run(
                user_id=user_id,
                goal=f"job-restart:{row.get('model_key')}:{new_job_id}",
            )
            restarted = store.create_training_job(
                job_id=new_job_id,
                user_id=user_id,
                job_type=str(row.get("job_type") or "training"),
                dataset_id=str(row.get("dataset_id") or "").strip() or None,
                model_key=str(row.get("model_key") or "").strip(),
                model_version=str(row.get("model_version") or "").strip() or None,
                status="queued",
                request=request_payload,
                result={"restarted_from_job_id": job_id},
                control={"action": "queued"},
                artifact_run_id=artifact_run_id,
            )
            store.append_training_job_event(
                job_id=new_job_id,
                user_id=user_id,
                event_type="queued",
                payload={"restarted_from_job_id": job_id},
            )
            _launch_training_worker(job_id=new_job_id, user_id=user_id)
            return TrainingJobResponse(job=_to_training_job_record(restarted))

        control_payload = {"action": action, "requested_at": datetime.utcnow().isoformat()}
        status_override: str | None = None
        if action == "pause":
            status_override = "paused"
        elif action == "resume":
            status_override = "running"
        elif action == "cancel":
            status_override = "canceled" if current_status == "queued" else None
        updated = store.update_training_job(
            job_id=job_id,
            user_id=user_id,
            status=status_override,
            control=control_payload,
            finished_at=(
                datetime.utcnow().isoformat()
                if action == "cancel" and current_status == "queued"
                else None
            ),
            heartbeat_at=datetime.utcnow().isoformat(),
        )
        store.append_training_job_event(
            job_id=job_id,
            user_id=user_id,
            event_type="control_requested",
            payload={"action": action},
        )
        if updated is None:
            raise HTTPException(status_code=404, detail="Training job not found.")
        if action == "resume" and not _training_thread_alive(job_id):
            request_payload = (
                updated.get("request") if isinstance(updated.get("request"), dict) else {}
            )
            checkpoint_hint = _latest_checkpoint_from_job(updated)
            if checkpoint_hint:
                request_payload = {
                    **request_payload,
                    "initial_checkpoint_path": checkpoint_hint,
                }
                updated = (
                    store.update_training_job(
                        job_id=job_id,
                        user_id=user_id,
                        request=request_payload,
                    )
                    or updated
                )
            _launch_training_worker(job_id=job_id, user_id=user_id)
        if action == "cancel" and current_status == "queued":
            run_id = str(updated.get("artifact_run_id") or "").strip()
            if run_id:
                store.update_status(run_id, RunStatus.CANCELED, error="Canceled before execution.")
        return TrainingJobResponse(job=_to_training_job_record(updated))

    @v1.post("/inference/jobs", response_model=TrainingJobResponse)
    @legacy.post("/inference/jobs", response_model=TrainingJobResponse)
    def create_inference_job_endpoint(
        req: InferenceJobCreateRequest,
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> TrainingJobResponse:
        del _auth
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        model_key = str(req.model_key or "").strip().lower()
        definition = get_model_definition(model_key)
        if definition is None:
            raise HTTPException(status_code=400, detail=f"Unknown model_key: {req.model_key}")
        if not definition.supports_inference:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model_key}' does not support inference.",
            )
        if not req.confirm_launch:
            raise HTTPException(
                status_code=400,
                detail="confirm_launch must be true for human-in-the-loop inference launch.",
            )
        if _active_training_count("inference") >= 1:
            raise HTTPException(
                status_code=409,
                detail="Another inference job is active. v1 supports one inference job at a time.",
            )
        file_ids = list(
            dict.fromkeys(
                [str(item or "").strip() for item in req.file_ids if str(item or "").strip()]
            )
        )
        if not file_ids:
            raise HTTPException(status_code=400, detail="file_ids is required.")
        reviewed_samples = int(req.reviewed_samples)
        reviewed_failures = int(req.reviewed_failures)
        if reviewed_failures > reviewed_samples:
            raise HTTPException(
                status_code=400,
                detail="reviewed_failures cannot exceed reviewed_samples.",
            )

        model_version = str(req.model_version or "").strip() or None
        model_artifact_path: str | None = None
        selected_job: dict[str, Any] | None = None
        if model_version:
            explicit_jobs = store.list_training_jobs(
                user_id=user_id,
                job_type="training",
                model_key=model_key,
                model_version=model_version,
                statuses=["succeeded"],
                limit=1,
            )
            if not explicit_jobs:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"No succeeded training job found for model_key={model_key} "
                        f"and model_version={model_version}."
                    ),
                )
            selected_job = explicit_jobs[0]
        else:
            if definition.supports_training:
                lineage_row: dict[str, Any] | None = None
                try:
                    _, lineage_row = _ensure_default_domain_and_lineage(
                        user_id=user_id,
                        model_key=model_key,
                    )
                except Exception:
                    lineage_row = None
                active_version_id = str((lineage_row or {}).get("active_version_id") or "").strip()
                if active_version_id:
                    active_version_row = store.get_training_model_version(
                        version_id=active_version_id
                    )
                    source_job_id = str(
                        (active_version_row or {}).get("source_job_id") or ""
                    ).strip()
                    if source_job_id:
                        active_job = store.get_training_job(job_id=source_job_id)
                        active_status = str((active_job or {}).get("status") or "").strip().lower()
                        active_job_type = (
                            str((active_job or {}).get("job_type") or "").strip().lower()
                        )
                        active_model_key = (
                            str((active_job or {}).get("model_key") or "").strip().lower()
                        )
                        if (
                            active_job is not None
                            and active_status == "succeeded"
                            and active_job_type == "training"
                            and active_model_key == model_key
                        ):
                            selected_job = active_job
                            model_version = active_version_id
                    if selected_job is None:
                        active_jobs = store.list_training_jobs(
                            user_id=user_id,
                            job_type="training",
                            model_key=model_key,
                            model_version=active_version_id,
                            statuses=["succeeded"],
                            limit=1,
                        )
                        if active_jobs:
                            selected_job = active_jobs[0]
                            model_version = active_version_id
            if selected_job is None:
                if model_key == prairie_model_key:
                    # Prairie YOLO defaults to builtin baseline unless a specific
                    # model version is selected or currently promoted as active.
                    model_version = None
                else:
                    fallback_jobs = store.list_training_jobs(
                        user_id=user_id,
                        job_type="training",
                        model_key=model_key,
                        statuses=["succeeded"],
                        limit=25,
                    )
                    if fallback_jobs:
                        selected_job = fallback_jobs[0]
        if model_key == "dynunet" and selected_job is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "DynUNET inference requires a trained model version. "
                    "Launch training first or specify an existing model_version."
                ),
            )
        if selected_job is not None:
            selected = selected_job
            selected_result = (
                selected.get("result") if isinstance(selected.get("result"), dict) else {}
            )
            model_artifact_path = (
                str(selected_result.get("model_artifact_path") or "").strip() or None
            )
            model_version = str(selected.get("model_version") or "").strip() or model_version
        if model_key == "dynunet" and not model_artifact_path:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Selected DynUNET model version is missing a model artifact path. "
                    "Rerun training or choose a different model version."
                ),
            )

        inference_config: dict[str, Any] = dict(req.config or {})
        if model_key == "yolov5_rarespot":
            try:
                inference_config = _normalize_training_config(
                    model_key=model_key,
                    raw_config=req.config,
                    default_config=dict(definition.default_config),
                )
            except DatasetValidationError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        job_id = f"infer-{uuid4().hex[:18]}"
        artifact_run_id = _create_training_artifact_run(
            user_id=user_id,
            goal=f"model-inference:{model_key}:{model_version or 'builtin'}",
        )
        request_payload = {
            "file_ids": file_ids,
            "config": inference_config,
            "model_artifact_path": model_artifact_path,
            "reviewed_samples": reviewed_samples,
            "reviewed_failures": reviewed_failures,
            "confirm_launch": True,
        }
        row = store.create_training_job(
            job_id=job_id,
            user_id=user_id,
            job_type="inference",
            dataset_id=None,
            model_key=model_key,
            model_version=model_version or "builtin",
            status="queued",
            request=request_payload,
            result={},
            control={"action": "queued"},
            artifact_run_id=artifact_run_id,
        )
        store.append_training_job_event(
            job_id=job_id,
            user_id=user_id,
            event_type="queued",
            payload={"file_count": len(file_ids), "model_version": model_version or "builtin"},
        )
        _launch_training_worker(job_id=job_id, user_id=user_id)
        return TrainingJobResponse(job=_to_training_job_record(row))

    @v1.get("/inference/jobs/{job_id}/result", response_model=TrainingJobResponse)
    @legacy.get("/inference/jobs/{job_id}/result", response_model=TrainingJobResponse)
    def get_inference_job_result_endpoint(
        job_id: str,
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> TrainingJobResponse:
        del _auth
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        row = _assert_training_job_owner(
            row=store.get_training_job(job_id=job_id, user_id=user_id),
            user_id=user_id,
        )
        if str(row.get("job_type") or "").strip().lower() != "inference":
            raise HTTPException(status_code=400, detail="Requested job is not an inference job.")
        return TrainingJobResponse(job=_to_training_job_record(row))

    @v1.get("/model-health", response_model=ModelHealthResponse)
    @legacy.get("/model-health", response_model=ModelHealthResponse)
    def model_health_endpoint(
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> ModelHealthResponse:
        del _auth
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        training_jobs = store.list_training_jobs(user_id=user_id, job_type="training", limit=1000)
        inference_jobs = store.list_training_jobs(user_id=user_id, job_type="inference", limit=1000)
        health_rows = compute_model_health_entries(
            training_jobs=training_jobs,
            inference_jobs=inference_jobs,
        )
        records = [ModelHealthRecord.model_validate(row) for row in health_rows]
        return ModelHealthResponse(count=len(records), models=records)

    @v1.get("/admin/model-health", response_model=ModelHealthResponse)
    @legacy.get("/admin/model-health", response_model=ModelHealthResponse)
    def admin_model_health_endpoint(
        admin_auth: dict[str, Any] = Depends(_require_admin_session),
    ) -> ModelHealthResponse:
        del admin_auth
        training_jobs = store.list_training_jobs(job_type="training", limit=5000)
        inference_jobs = store.list_training_jobs(job_type="inference", limit=5000)
        health_rows = compute_model_health_entries(
            training_jobs=training_jobs,
            inference_jobs=inference_jobs,
        )
        records = [ModelHealthRecord.model_validate(row) for row in health_rows]
        return ModelHealthResponse(count=len(records), models=records)

    @v1.post("/training/domains", response_model=TrainingDomainRecord)
    @legacy.post("/training/domains", response_model=TrainingDomainRecord)
    def create_training_domain_endpoint(
        req: TrainingDomainCreateRequest,
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> TrainingDomainRecord:
        del _auth
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        domain_id = f"domain-{uuid4().hex[:16]}"
        row = store.create_training_domain(
            domain_id=domain_id,
            name=req.name,
            description=req.description,
            owner_scope=req.owner_scope,
            owner_user_id=user_id,
            metadata=req.metadata,
        )
        return _to_training_domain_record(row)

    @v1.get("/training/domains", response_model=TrainingDomainListResponse)
    @legacy.get("/training/domains", response_model=TrainingDomainListResponse)
    def list_training_domains_endpoint(
        limit: int = Query(default=200, ge=1, le=1000),
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> TrainingDomainListResponse:
        del _auth
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        rows = store.list_training_domains(
            owner_user_id=user_id,
            include_shared=True,
            limit=limit,
        )
        records = [_to_training_domain_record(row) for row in rows]
        return TrainingDomainListResponse(count=len(records), domains=records)

    @v1.get("/training/domains/{domain_id}/lineages", response_model=TrainingLineageListResponse)
    @legacy.get(
        "/training/domains/{domain_id}/lineages", response_model=TrainingLineageListResponse
    )
    def list_training_domain_lineages_endpoint(
        domain_id: str,
        limit: int = Query(default=200, ge=1, le=1000),
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> TrainingLineageListResponse:
        del _auth
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        domain_row = _assert_training_domain_access(
            row=store.get_training_domain(
                domain_id=domain_id, owner_user_id=user_id, include_shared=True
            ),
            user_id=user_id,
        )
        rows = store.list_training_lineages(
            domain_id=str(domain_row.get("domain_id") or ""),
            owner_user_id=user_id,
            include_shared=True,
            limit=limit,
        )
        records = [_to_training_lineage_record(row) for row in rows]
        return TrainingLineageListResponse(count=len(records), lineages=records)

    @v1.post("/training/lineages/{lineage_id}/fork", response_model=TrainingLineageRecord)
    @legacy.post("/training/lineages/{lineage_id}/fork", response_model=TrainingLineageRecord)
    def fork_training_lineage_endpoint(
        lineage_id: str,
        req: TrainingForkLineageRequest,
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> TrainingLineageRecord:
        del _auth
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        source = _assert_training_lineage_access(
            row=store.get_training_lineage(
                lineage_id=lineage_id, owner_user_id=user_id, include_shared=True
            ),
            user_id=user_id,
        )
        fork_id = f"lineage-fork-{uuid4().hex[:16]}"
        source_metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
        merged_metadata = {
            **source_metadata,
            **req.metadata,
            "forked_from_lineage_id": str(source.get("lineage_id") or ""),
            "forked_at": datetime.utcnow().isoformat(),
        }
        row = store.create_training_lineage(
            lineage_id=fork_id,
            domain_id=str(source.get("domain_id") or ""),
            scope="fork",
            owner_user_id=user_id,
            model_key=str(req.model_key or source.get("model_key") or prairie_model_key),
            parent_lineage_id=str(source.get("lineage_id") or ""),
            active_version_id=str(source.get("active_version_id") or "").strip() or None,
            metadata=merged_metadata,
        )
        return _to_training_lineage_record(row)

    @v1.get(
        "/training/lineages/{lineage_id}/versions", response_model=TrainingModelVersionListResponse
    )
    @legacy.get(
        "/training/lineages/{lineage_id}/versions", response_model=TrainingModelVersionListResponse
    )
    def list_training_lineage_versions_endpoint(
        lineage_id: str,
        limit: int = Query(default=200, ge=1, le=1000),
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> TrainingModelVersionListResponse:
        del _auth
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        lineage = _assert_training_lineage_access(
            row=store.get_training_lineage(
                lineage_id=lineage_id, owner_user_id=user_id, include_shared=True
            ),
            user_id=user_id,
        )
        rows = store.list_training_model_versions(
            lineage_id=str(lineage.get("lineage_id") or ""),
            limit=limit,
        )
        records = [_to_training_model_version_record(row) for row in rows]
        return TrainingModelVersionListResponse(count=len(records), versions=records)

    @v1.post(
        "/training/update-proposals/preview", response_model=TrainingUpdateProposalPreviewResponse
    )
    @legacy.post(
        "/training/update-proposals/preview", response_model=TrainingUpdateProposalPreviewResponse
    )
    def preview_training_update_proposal_endpoint(
        req: TrainingUpdateProposalPreviewRequest,
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> TrainingUpdateProposalPreviewResponse:
        del _auth
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        lineage = _assert_training_lineage_access(
            row=store.get_training_lineage(
                lineage_id=req.lineage_id,
                owner_user_id=user_id,
                include_shared=True,
            ),
            user_id=user_id,
        )
        if req.persist:
            lineage = _assert_training_lineage_mutation_access(
                row=lineage,
                user_id=user_id,
                action="Persisting update proposals",
            )
        model_key = str(lineage.get("model_key") or "").strip().lower() or prairie_model_key
        definition = get_model_definition(model_key)
        if definition is None:
            raise HTTPException(
                status_code=400, detail=f"Unknown model_key on lineage: {model_key}"
            )
        dataset_row, _, manifest = _build_training_dataset_manifest(
            dataset_id=req.dataset_id,
            user_id=user_id,
            model_key=model_key,
        )
        config = _normalize_training_config(
            model_key=model_key,
            raw_config=req.config,
            default_config=dict(definition.default_config),
        )
        preflight = build_preflight_report(
            model_key=model_key,
            dataset_manifest=manifest,
            config=config,
            artifact_root=artifact_root,
            model_dimensions=[str(item).lower() for item in definition.dimensions],
            max_dimension_samples=int(
                getattr(settings, "training_dimension_check_max_samples", 512) or 512
            ),
        )
        manifest_counts = manifest.get("counts") if isinstance(manifest.get("counts"), dict) else {}
        train_counts = (
            manifest_counts.get("train") if isinstance(manifest_counts.get("train"), dict) else {}
        )
        approved_new_samples = (
            int(req.approved_new_samples)
            if req.approved_new_samples is not None
            else int(train_counts.get("samples") or 0)
        )
        class_counts = {
            str(key): int(value)
            for key, value in (req.class_counts or {}).items()
            if str(key).strip()
        }
        active_version_row = _lineage_active_version_row(lineage)
        last_promoted_at = (
            str(active_version_row.get("updated_at") or "").strip()
            if isinstance(active_version_row, dict)
            else None
        ) or str(lineage.get("updated_at") or "").strip()
        health_status = req.health_status
        trigger = _build_trigger_snapshot(
            lineage_id=str(lineage.get("lineage_id") or ""),
            dataset_id=req.dataset_id,
            approved_new_samples=approved_new_samples,
            class_counts=class_counts,
            health_status=health_status,
            last_promoted_at=last_promoted_at,
        )
        _seed_lineage_replay_items(lineage_row=lineage)
        replay_items = store.list_training_replay_items(
            lineage_id=str(lineage.get("lineage_id") or ""),
            limit=5000,
        )
        replay_mix = build_replay_mix_plan(
            lineage_id=str(lineage.get("lineage_id") or ""),
            new_samples=approved_new_samples,
            replay_items=replay_items,
        )
        open_existing = store.list_training_update_proposals(
            owner_user_id=user_id,
            lineage_id=str(lineage.get("lineage_id") or ""),
            statuses=_open_proposal_statuses() + _proposal_ready_statuses(),
            include_shared=True,
            limit=5,
        )
        preview_payload: dict[str, Any] = {
            "lineage_id": str(lineage.get("lineage_id") or ""),
            "dataset_id": req.dataset_id,
            "dataset_name": str(dataset_row.get("name") or ""),
            "manifest_counts": manifest_counts,
            "approved_new_samples": approved_new_samples,
            "class_counts": class_counts,
            "trigger": trigger,
            "preflight": preflight,
            "replay_mix": replay_mix,
            "config": config,
            "open_proposal_exists": bool(open_existing),
        }
        proposal_record: TrainingUpdateProposalRecord | None = None
        if req.persist:
            if open_existing:
                proposal_record = _to_training_update_proposal_record(open_existing[0])
            else:
                reason = (
                    str(req.trigger_reason_override or trigger.get("reason") or "manual")
                    .strip()
                    .lower()
                    or "manual"
                )
                proposal_id = f"proposal-{uuid4().hex[:18]}"
                proposal_row = store.create_training_update_proposal(
                    proposal_id=proposal_id,
                    lineage_id=str(lineage.get("lineage_id") or ""),
                    trigger_reason=reason,
                    trigger_snapshot=trigger,
                    dataset_snapshot={
                        "dataset_id": req.dataset_id,
                        "dataset_name": str(dataset_row.get("name") or ""),
                        "manifest_counts": manifest_counts,
                        "approved_new_samples": approved_new_samples,
                        "class_counts": class_counts,
                        "replay_mix": replay_mix,
                    },
                    config=config,
                    status="pending_approval",
                    idempotency_key=req.idempotency_key,
                )
                proposal_record = _to_training_update_proposal_record(proposal_row)
        return TrainingUpdateProposalPreviewResponse(
            trigger=trigger,
            preview=preview_payload,
            proposal=proposal_record,
        )

    @v1.get("/training/update-proposals", response_model=TrainingUpdateProposalListResponse)
    @legacy.get("/training/update-proposals", response_model=TrainingUpdateProposalListResponse)
    def list_training_update_proposals_endpoint(
        lineage_id: str | None = Query(default=None),
        status: str | None = Query(default=None),
        limit: int = Query(default=200, ge=1, le=1000),
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> TrainingUpdateProposalListResponse:
        del _auth
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        statuses = [
            token.strip().lower() for token in str(status or "").split(",") if token.strip()
        ]
        rows = store.list_training_update_proposals(
            owner_user_id=user_id,
            lineage_id=str(lineage_id or "").strip() or None,
            statuses=statuses or None,
            include_shared=True,
            limit=limit,
        )
        records = [_to_training_update_proposal_record(row) for row in rows]
        return TrainingUpdateProposalListResponse(count=len(records), proposals=records)

    @v1.post(
        "/training/update-proposals/{proposal_id}/approve",
        response_model=TrainingUpdateProposalResponse,
    )
    @legacy.post(
        "/training/update-proposals/{proposal_id}/approve",
        response_model=TrainingUpdateProposalResponse,
    )
    def approve_training_update_proposal_endpoint(
        proposal_id: str,
        req: TrainingUpdateProposalDecisionRequest,
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> TrainingUpdateProposalResponse:
        del _auth
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        if not req.confirm_launch:
            raise HTTPException(
                status_code=400,
                detail="confirm_launch must be true for human-in-the-loop training launch.",
            )
        proposal = store.get_training_update_proposal(proposal_id=proposal_id)
        if proposal is None:
            raise HTTPException(status_code=404, detail="Training update proposal not found.")
        lineage = _assert_training_lineage_mutation_access(
            row=store.get_training_lineage(
                lineage_id=str(proposal.get("lineage_id") or ""),
                owner_user_id=user_id,
                include_shared=True,
            ),
            user_id=user_id,
            action="Approving update proposals",
        )
        current_status = normalize_proposal_status(proposal.get("status"))
        if not proposal_transition_allowed(current_status, "approved"):
            raise HTTPException(
                status_code=409,
                detail=f"Proposal cannot be approved from status '{current_status}'.",
            )
        if _active_training_count("training") >= 1:
            raise HTTPException(
                status_code=409,
                detail="Another training job is active. v1 supports one training job at a time.",
            )
        snapshot = (
            proposal.get("dataset_snapshot")
            if isinstance(proposal.get("dataset_snapshot"), dict)
            else {}
        )
        dataset_id = str(snapshot.get("dataset_id") or "").strip()
        if not dataset_id:
            raise HTTPException(
                status_code=400,
                detail="Proposal is missing dataset snapshot dataset_id.",
            )
        model_key = str(lineage.get("model_key") or "").strip().lower() or prairie_model_key
        definition = get_model_definition(model_key)
        if definition is None:
            raise HTTPException(status_code=400, detail=f"Unknown model_key: {model_key}")
        config_raw = proposal.get("config") if isinstance(proposal.get("config"), dict) else {}
        config = _normalize_training_config(
            model_key=model_key,
            raw_config=config_raw,
            default_config=dict(definition.default_config),
        )
        dataset_row, _, manifest = _build_training_dataset_manifest(
            dataset_id=dataset_id,
            user_id=user_id,
            model_key=model_key,
        )
        preflight = build_preflight_report(
            model_key=model_key,
            dataset_manifest=manifest,
            config=config,
            artifact_root=artifact_root,
            model_dimensions=[str(item).lower() for item in definition.dimensions],
            max_dimension_samples=int(
                getattr(settings, "training_dimension_check_max_samples", 512) or 512
            ),
        )
        if not bool(preflight.get("recommended_launch")):
            raise HTTPException(status_code=400, detail=_format_preflight_failure_detail(preflight))

        active_version_row = _lineage_active_version_row(lineage)
        active_meta = (
            active_version_row.get("metadata")
            if isinstance(active_version_row, dict)
            and isinstance(active_version_row.get("metadata"), dict)
            else {}
        )
        initial_checkpoint_path = (
            str(active_meta.get("model_artifact_path") or "").strip()
            or str(active_meta.get("checkpoint_path") or "").strip()
            or None
        )
        if initial_checkpoint_path:
            checkpoint_path = Path(initial_checkpoint_path).expanduser()
            if not checkpoint_path.exists() or not checkpoint_path.is_file():
                initial_checkpoint_path = None
            else:
                initial_checkpoint_path = str(checkpoint_path.resolve())

        job_id = f"train-{uuid4().hex[:18]}"
        artifact_run_id = _create_training_artifact_run(
            user_id=user_id,
            goal=f"continuous-update:{model_key}:{dataset_row.get('name')}",
        )
        request_payload = {
            "dataset_id": dataset_id,
            "config": config,
            "preflight": preflight,
            "manifest_counts": manifest.get("counts") or {},
            "dataset_name": str(dataset_row.get("name") or ""),
            "initial_checkpoint_path": initial_checkpoint_path,
            "confirmed": True,
            "continuous": {
                "proposal_id": proposal_id,
                "lineage_id": str(lineage.get("lineage_id") or ""),
                "trigger_reason": str(proposal.get("trigger_reason") or "manual"),
                "dataset_snapshot": snapshot,
                "note": str(req.note or "").strip() or None,
            },
        }
        store.create_training_job(
            job_id=job_id,
            user_id=user_id,
            job_type="training",
            dataset_id=dataset_id,
            model_key=model_key,
            model_version=None,
            status="queued",
            request=request_payload,
            result={"preflight": preflight, "proposal_id": proposal_id},
            control={"action": "queued"},
            artifact_run_id=artifact_run_id,
        )
        store.append_training_job_event(
            job_id=job_id,
            user_id=user_id,
            event_type="queued",
            payload={"proposal_id": proposal_id, "preflight": preflight},
        )
        approved_at = datetime.utcnow().isoformat()
        updated = store.update_training_update_proposal(
            proposal_id=proposal_id,
            status="approved",
            approved_by=user_id,
            linked_job_id=job_id,
            approved_at=approved_at,
        )
        _launch_training_worker(job_id=job_id, user_id=user_id)
        if updated is None:
            raise HTTPException(status_code=500, detail="Failed to update proposal.")
        return TrainingUpdateProposalResponse(proposal=_to_training_update_proposal_record(updated))

    @v1.post(
        "/training/update-proposals/{proposal_id}/reject",
        response_model=TrainingUpdateProposalResponse,
    )
    @legacy.post(
        "/training/update-proposals/{proposal_id}/reject",
        response_model=TrainingUpdateProposalResponse,
    )
    def reject_training_update_proposal_endpoint(
        proposal_id: str,
        req: TrainingUpdateProposalDecisionRequest,
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> TrainingUpdateProposalResponse:
        del _auth
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        proposal = store.get_training_update_proposal(proposal_id=proposal_id)
        if proposal is None:
            raise HTTPException(status_code=404, detail="Training update proposal not found.")
        _assert_training_lineage_mutation_access(
            row=store.get_training_lineage(
                lineage_id=str(proposal.get("lineage_id") or ""),
                owner_user_id=user_id,
                include_shared=True,
            ),
            user_id=user_id,
            action="Rejecting update proposals",
        )
        current_status = normalize_proposal_status(proposal.get("status"))
        if not proposal_transition_allowed(current_status, "rejected"):
            raise HTTPException(
                status_code=409,
                detail=f"Proposal cannot be rejected from status '{current_status}'.",
            )
        note = str(req.note or "").strip()
        updated = store.update_training_update_proposal(
            proposal_id=proposal_id,
            status="rejected",
            rejected_by=user_id,
            error=note or None,
            rejected_at=datetime.utcnow().isoformat(),
            finished_at=datetime.utcnow().isoformat(),
        )
        if updated is None:
            raise HTTPException(status_code=500, detail="Failed to update proposal.")
        return TrainingUpdateProposalResponse(proposal=_to_training_update_proposal_record(updated))

    @v1.post(
        "/training/model-versions/{version_id}/promote",
        response_model=TrainingModelVersionResponse,
    )
    @legacy.post(
        "/training/model-versions/{version_id}/promote",
        response_model=TrainingModelVersionResponse,
    )
    def promote_training_model_version_endpoint(
        version_id: str,
        req: TrainingVersionPromoteRequest,
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> TrainingModelVersionResponse:
        del _auth
        del req
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        version = store.get_training_model_version(version_id=version_id)
        if version is None:
            raise HTTPException(status_code=404, detail="Training model version not found.")
        lineage = _assert_training_lineage_mutation_access(
            row=store.get_training_lineage(
                lineage_id=str(version.get("lineage_id") or ""),
                owner_user_id=user_id,
                include_shared=True,
            ),
            user_id=user_id,
            action="Promoting model versions",
        )
        current_status = normalize_version_status(version.get("status"))
        if not version_transition_allowed(current_status, "active"):
            raise HTTPException(
                status_code=409,
                detail=f"Version cannot be promoted from status '{current_status}'.",
            )
        version_metrics = version.get("metrics") if isinstance(version.get("metrics"), dict) else {}
        version_metadata = (
            version.get("metadata") if isinstance(version.get("metadata"), dict) else {}
        )
        guardrails = (
            version_metadata.get("guardrails")
            if isinstance(version_metadata.get("guardrails"), dict)
            else {}
        )
        benchmark_ready = bool(version_metrics.get("benchmark_ready"))
        if not benchmark_ready and isinstance(version_metrics.get("benchmark"), dict):
            benchmark_ready = bool((version_metrics.get("benchmark") or {}).get("ready"))
        if not benchmark_ready:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Promotion blocked: benchmark packet is incomplete or missing. "
                    "Run `/v1/training/prairie/benchmark/run` and retrain if needed."
                ),
            )
        if guardrails and not bool(guardrails.get("passed")):
            reasons = (
                guardrails.get("reasons") if isinstance(guardrails.get("reasons"), list) else []
            )
            reason_text = "; ".join(str(item) for item in reasons[:3]) if reasons else ""
            detail = "Promotion blocked: guardrails did not pass."
            if reason_text:
                detail += f" {reason_text}"
            raise HTTPException(status_code=409, detail=detail)
        previous_active = str(lineage.get("active_version_id") or "").strip()
        if previous_active and previous_active != version_id:
            previous_row = store.get_training_model_version(version_id=previous_active)
            if previous_row is not None and version_transition_allowed(
                normalize_version_status(previous_row.get("status")),
                "retired",
            ):
                store.update_training_model_version(version_id=previous_active, status="retired")

        store.update_training_model_version(
            version_id=version_id,
            status="active",
            metadata={
                **version_metadata,
                "previous_active_version_id": previous_active or None,
                "promoted_at": datetime.utcnow().isoformat(),
            },
        )
        lineage_metadata = (
            lineage.get("metadata") if isinstance(lineage.get("metadata"), dict) else {}
        )
        store.update_training_lineage(
            lineage_id=str(lineage.get("lineage_id") or ""),
            active_version_id=version_id,
            metadata={
                **lineage_metadata,
                "previous_active_version_id": previous_active or None,
            },
        )
        proposals = store.list_training_update_proposals(
            owner_user_id=user_id,
            lineage_id=str(lineage.get("lineage_id") or ""),
            statuses=_proposal_ready_statuses(),
            include_shared=True,
            limit=50,
        )
        for proposal in proposals:
            if str(proposal.get("candidate_version_id") or "").strip() != version_id:
                continue
            current = normalize_proposal_status(proposal.get("status"))
            if proposal_transition_allowed(current, "promoted"):
                store.update_training_update_proposal(
                    proposal_id=str(proposal.get("proposal_id") or ""),
                    status="promoted",
                    finished_at=datetime.utcnow().isoformat(),
                )
        updated_version = store.get_training_model_version(version_id=version_id)
        updated_lineage = store.get_training_lineage(
            lineage_id=str(lineage.get("lineage_id") or ""),
            owner_user_id=user_id,
            include_shared=True,
        )
        if updated_version is None or updated_lineage is None:
            raise HTTPException(status_code=500, detail="Failed to update promotion state.")
        return TrainingModelVersionResponse(
            version=_to_training_model_version_record(updated_version),
            lineage=_to_training_lineage_record(updated_lineage),
        )

    @v1.post(
        "/training/model-versions/{version_id}/rollback",
        response_model=TrainingModelVersionResponse,
    )
    @legacy.post(
        "/training/model-versions/{version_id}/rollback",
        response_model=TrainingModelVersionResponse,
    )
    def rollback_training_model_version_endpoint(
        version_id: str,
        req: TrainingVersionRollbackRequest,
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> TrainingModelVersionResponse:
        del _auth
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        current_version = store.get_training_model_version(version_id=version_id)
        if current_version is None:
            raise HTTPException(status_code=404, detail="Training model version not found.")
        lineage = _assert_training_lineage_mutation_access(
            row=store.get_training_lineage(
                lineage_id=str(current_version.get("lineage_id") or ""),
                owner_user_id=user_id,
                include_shared=True,
            ),
            user_id=user_id,
            action="Rolling back model versions",
        )
        lineage_active_id = str(lineage.get("active_version_id") or "").strip()
        if not lineage_active_id:
            raise HTTPException(
                status_code=400, detail="Lineage has no active version to rollback."
            )
        target_version_id = str(req.target_version_id or "").strip()
        if not target_version_id:
            current_metadata = (
                current_version.get("metadata")
                if isinstance(current_version.get("metadata"), dict)
                else {}
            )
            lineage_metadata = (
                lineage.get("metadata") if isinstance(lineage.get("metadata"), dict) else {}
            )
            target_version_id = (
                str(current_metadata.get("previous_active_version_id") or "").strip()
                or str(lineage_metadata.get("previous_active_version_id") or "").strip()
            )
        if not target_version_id:
            raise HTTPException(
                status_code=400,
                detail="Rollback target is undefined. Supply target_version_id.",
            )
        target_version = store.get_training_model_version(version_id=target_version_id)
        if target_version is None:
            raise HTTPException(status_code=404, detail="Rollback target version not found.")
        if str(target_version.get("lineage_id") or "") != str(lineage.get("lineage_id") or ""):
            raise HTTPException(
                status_code=400,
                detail="Rollback target version does not belong to the same lineage.",
            )
        if lineage_active_id != target_version_id:
            active_row = store.get_training_model_version(version_id=lineage_active_id)
            if active_row is not None and version_transition_allowed(
                normalize_version_status(active_row.get("status")),
                "retired",
            ):
                store.update_training_model_version(version_id=lineage_active_id, status="retired")
        store.update_training_model_version(
            version_id=target_version_id,
            status="active",
        )
        lineage_metadata = (
            lineage.get("metadata") if isinstance(lineage.get("metadata"), dict) else {}
        )
        store.update_training_lineage(
            lineage_id=str(lineage.get("lineage_id") or ""),
            active_version_id=target_version_id,
            metadata={
                **lineage_metadata,
                "previous_active_version_id": lineage_active_id,
                "rollback_note": str(req.note or "").strip() or None,
            },
        )
        updated_lineage = store.get_training_lineage(
            lineage_id=str(lineage.get("lineage_id") or ""),
            owner_user_id=user_id,
            include_shared=True,
        )
        updated_version = store.get_training_model_version(version_id=target_version_id)
        if updated_lineage is None or updated_version is None:
            raise HTTPException(status_code=500, detail="Failed to complete rollback.")
        return TrainingModelVersionResponse(
            version=_to_training_model_version_record(updated_version),
            lineage=_to_training_lineage_record(updated_lineage),
        )

    @v1.post("/training/merge-requests", response_model=TrainingMergeRequestResponse)
    @legacy.post("/training/merge-requests", response_model=TrainingMergeRequestResponse)
    def create_training_merge_request_endpoint(
        req: TrainingMergeRequestCreateRequest,
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> TrainingMergeRequestResponse:
        del _auth
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        source = _assert_training_lineage_access(
            row=store.get_training_lineage(
                lineage_id=req.source_lineage_id,
                owner_user_id=user_id,
                include_shared=True,
            ),
            user_id=user_id,
        )
        target = _assert_training_lineage_access(
            row=store.get_training_lineage(
                lineage_id=req.target_lineage_id,
                owner_user_id=user_id,
                include_shared=True,
            ),
            user_id=user_id,
        )
        if (
            str(source.get("model_key") or "").strip().lower()
            != str(target.get("model_key") or "").strip().lower()
        ):
            raise HTTPException(
                status_code=400,
                detail="Source and target lineages must have the same model_key.",
            )
        candidate = store.get_training_model_version(version_id=req.candidate_version_id)
        if candidate is None:
            raise HTTPException(status_code=404, detail="Candidate version not found.")
        if (
            str(candidate.get("lineage_id") or "").strip()
            != str(source.get("lineage_id") or "").strip()
        ):
            raise HTTPException(
                status_code=400,
                detail="Candidate version must belong to source lineage.",
            )
        merge_row = store.create_training_merge_request(
            merge_id=f"merge-{uuid4().hex[:18]}",
            source_lineage_id=req.source_lineage_id,
            target_lineage_id=req.target_lineage_id,
            candidate_version_id=req.candidate_version_id,
            requested_by=user_id,
            status="open",
            notes=req.notes,
        )
        return TrainingMergeRequestResponse(
            merge_request=_to_training_merge_request_record(merge_row)
        )

    @v1.post(
        "/training/merge-requests/{merge_id}/approve",
        response_model=TrainingMergeRequestResponse,
    )
    @legacy.post(
        "/training/merge-requests/{merge_id}/approve",
        response_model=TrainingMergeRequestResponse,
    )
    def approve_training_merge_request_endpoint(
        merge_id: str,
        req: TrainingMergeRequestDecisionRequest,
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> TrainingMergeRequestResponse:
        del _auth
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        merge_row = store.get_training_merge_request(merge_id=merge_id)
        if merge_row is None:
            raise HTTPException(status_code=404, detail="Merge request not found.")
        target_lineage = _assert_training_lineage_mutation_access(
            row=store.get_training_lineage(
                lineage_id=str(merge_row.get("target_lineage_id") or ""),
                owner_user_id=user_id,
                include_shared=True,
            ),
            user_id=user_id,
            action="Approving merge requests",
        )
        current_status = normalize_merge_status(merge_row.get("status"))
        if not merge_transition_allowed(current_status, "approved"):
            raise HTTPException(
                status_code=409,
                detail=f"Merge request cannot be approved from status '{current_status}'.",
            )
        linked_proposal_id: str | None = None
        existing_open = store.list_training_update_proposals(
            owner_user_id=user_id,
            lineage_id=str(target_lineage.get("lineage_id") or ""),
            statuses=_open_proposal_statuses() + _proposal_ready_statuses(),
            include_shared=True,
            limit=1,
        )
        if existing_open:
            linked_proposal_id = str(existing_open[0].get("proposal_id") or "").strip() or None
        updated = store.update_training_merge_request(
            merge_id=merge_id,
            status="approved",
            decision_by=user_id,
            notes=req.notes,
            decided_at=datetime.utcnow().isoformat(),
            linked_proposal_id=linked_proposal_id,
        )
        if updated is None:
            raise HTTPException(status_code=500, detail="Failed to approve merge request.")
        return TrainingMergeRequestResponse(
            merge_request=_to_training_merge_request_record(updated)
        )

    @v1.post(
        "/training/merge-requests/{merge_id}/reject",
        response_model=TrainingMergeRequestResponse,
    )
    @legacy.post(
        "/training/merge-requests/{merge_id}/reject",
        response_model=TrainingMergeRequestResponse,
    )
    def reject_training_merge_request_endpoint(
        merge_id: str,
        req: TrainingMergeRequestDecisionRequest,
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> TrainingMergeRequestResponse:
        del _auth
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        merge_row = store.get_training_merge_request(merge_id=merge_id)
        if merge_row is None:
            raise HTTPException(status_code=404, detail="Merge request not found.")
        _assert_training_lineage_mutation_access(
            row=store.get_training_lineage(
                lineage_id=str(merge_row.get("target_lineage_id") or ""),
                owner_user_id=user_id,
                include_shared=True,
            ),
            user_id=user_id,
            action="Rejecting merge requests",
        )
        current_status = normalize_merge_status(merge_row.get("status"))
        if not merge_transition_allowed(current_status, "rejected"):
            raise HTTPException(
                status_code=409,
                detail=f"Merge request cannot be rejected from status '{current_status}'.",
            )
        updated = store.update_training_merge_request(
            merge_id=merge_id,
            status="rejected",
            decision_by=user_id,
            notes=req.notes,
            decided_at=datetime.utcnow().isoformat(),
        )
        if updated is None:
            raise HTTPException(status_code=500, detail="Failed to reject merge request.")
        return TrainingMergeRequestResponse(
            merge_request=_to_training_merge_request_record(updated)
        )

    @v1.get("/training/merge-requests", response_model=TrainingMergeRequestListResponse)
    @legacy.get("/training/merge-requests", response_model=TrainingMergeRequestListResponse)
    def list_training_merge_requests_endpoint(
        status: str | None = Query(default=None),
        limit: int = Query(default=200, ge=1, le=1000),
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> TrainingMergeRequestListResponse:
        del _auth
        user_id = _current_user_id(bisque_auth, allow_anonymous=True)
        statuses = [
            token.strip().lower() for token in str(status or "").split(",") if token.strip()
        ]
        rows = store.list_training_merge_requests(
            owner_user_id=user_id,
            statuses=statuses or None,
            include_shared=True,
            limit=limit,
        )
        records = [_to_training_merge_request_record(row) for row in rows]
        return TrainingMergeRequestListResponse(count=len(records), merge_requests=records)

    @v1.post("/workflows/repro-report", response_model=ReproReportResponse)
    @legacy.post("/workflows/repro-report", response_model=ReproReportResponse)
    def workflow_repro_report(
        req: ReproReportRequest,
        _auth: None = Depends(_require_api_key),
    ) -> ReproReportResponse:
        del _auth
        if req.run_id:
            run = store.get_run(req.run_id)
            if not run:
                raise HTTPException(status_code=404, detail="Run not found")

        report = generate_repro_report(
            run_id=req.run_id,
            title=req.title,
            result_summary=req.result_summary,
            measurements=req.measurements,
            statistical_analysis=req.statistical_analysis,
            qc_warnings=req.qc_warnings,
            limitations=req.limitations,
            provenance=req.provenance,
            next_steps=req.next_steps,
            output_dir=req.output_dir,
            artifact_root=str(artifact_root),
        )
        if not report.get("success"):
            raise HTTPException(
                status_code=500, detail=report.get("error") or "report generation failed"
            )

        if req.run_id:
            report_paths = []
            for key in ("report_markdown_path", "report_json_path"):
                raw = report.get(key)
                if raw:
                    path = Path(str(raw))
                    if path.exists() and path.is_file():
                        report_paths.append(path)
            if report_paths:
                entries = []
                for p in report_paths:
                    entry = _artifact_entry(req.run_id, p)
                    entry["category"] = "report"
                    entries.append(entry)
                _update_manifest_with_entries(req.run_id, entries)
                store.append_event(
                    req.run_id,
                    "report_generated",
                    {
                        "report_markdown_path": report.get("report_markdown_path"),
                        "report_json_path": report.get("report_json_path"),
                        "report_sha256": report.get("report_sha256"),
                    },
                )

        return ReproReportResponse(
            success=True,
            run_id=req.run_id,
            report_markdown_path=str(report.get("report_markdown_path")),
            report_json_path=str(report.get("report_json_path")),
            report_sha256=str(report.get("report_sha256")),
            report_bundle_sha256=str(report.get("report_bundle_sha256")),
        )

    @v1.get("/artifacts/{run_id}", response_model=ArtifactListResponse)
    @legacy.get("/artifacts/{run_id}", response_model=ArtifactListResponse)
    def list_artifacts(
        run_id: str,
        limit: int = Query(default=500, ge=1, le=5000),
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> ArtifactListResponse:
        del _auth
        run = store.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        request_user_id = _resolve_request_user_id(bisque_auth)
        _assert_run_owner_access(run_id=run_id, request_user_id=request_user_id)

        artifacts = _list_artifacts(run_id, limit=limit)
        run_root = str((_ensure_run_artifact_dir(run_id)).resolve())
        return ArtifactListResponse(
            run_id=run_id,
            root=run_root,
            artifact_count=len(artifacts),
            artifacts=artifacts,
        )

    @v1.get("/artifacts/{run_id}/manifest")
    @legacy.get("/artifacts/{run_id}/manifest")
    def get_artifact_manifest(
        run_id: str,
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> JSONResponse:
        del _auth
        run = store.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        request_user_id = _resolve_request_user_id(bisque_auth)
        _assert_run_owner_access(run_id=run_id, request_user_id=request_user_id)
        manifest = _materialize_manifest_if_missing(run_id)
        return JSONResponse(content=manifest)

    @v1.get("/artifacts/{run_id}/download")
    @legacy.get("/artifacts/{run_id}/download")
    def download_artifact(
        run_id: str,
        path: str = Query(..., min_length=1),
        bisque_auth: dict[str, Any] | None = Depends(_get_bisque_auth_optional),
        _auth: None = Depends(_require_api_key),
    ) -> FileResponse:
        del _auth
        run = store.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        request_user_id = _resolve_request_user_id(bisque_auth)
        _assert_run_owner_access(run_id=run_id, request_user_id=request_user_id)

        target, _resolved_rel_path = _resolve_artifact_path_alias(run_id, path)
        if target is None or not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail="Artifact not found")

        media_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        return FileResponse(path=str(target), filename=target.name, media_type=media_type)

    app.include_router(v1)
    app.include_router(v2)
    app.include_router(
        build_v3_router(_current_user_id, _get_bisque_auth_optional, _require_api_key)
    )
    app.include_router(legacy)
    app.state.agentic_v3 = agentic_v3_services
    app.state.run_continuous_scheduler_iteration = _run_continuous_scheduler_iteration
    return app


app = create_app()
