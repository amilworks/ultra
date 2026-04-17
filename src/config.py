"""Application configuration management using pydantic-settings."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _is_connection_url(value: str) -> bool:
    lowered = str(value or "").strip().lower()
    return lowered.startswith("postgres://") or lowered.startswith("postgresql://")


def _resolve_project_path(value: str) -> str:
    if _is_connection_url(value):
        return str(value).strip()
    raw = Path(str(value)).expanduser()
    if raw.is_absolute():
        return str(raw)
    return str((_PROJECT_ROOT / raw).resolve())


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=str((_PROJECT_ROOT / ".env").resolve()),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application settings
    app_name: str = Field(default="Bisque Ultra", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")
    environment: Literal["development", "staging", "production"] = Field(
        default="development", description="Environment"
    )
    debug: bool = Field(default=False, description="Debug mode")

    ui_show_diagnostics_default: bool = Field(
        default=False,
        description="Show diagnostic contract/progress/artifact details in the web UI by default.",
    )
    ui_auto_repro_report: bool = Field(
        default=False,
        description="Automatically generate reproducibility report artifacts for each run.",
    )
    ui_force_tool_visualizations: bool = Field(
        default=True,
        description=(
            "Force visualization-producing tools (SAM/YOLO) to persist preview artifacts "
            "so the chat UI can reliably render result cards."
        ),
    )

    # Logging settings
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format",
    )

    # Security settings
    enable_cors: bool = Field(default=False, description="Enable CORS")
    allowed_origins: list[str] = Field(default_factory=list, description="Allowed CORS origins")

    # Performance settings
    max_upload_size: int = Field(default=200, description="Max upload size in MB")
    # LLM provider routing (centralized)
    llm_provider: Literal["vllm", "openai", "ollama"] = Field(
        default="vllm",
        description="Active LLM provider preset used to resolve endpoint/model defaults.",
    )
    llm_base_url: str | None = Field(
        default=None,
        description="Optional override for the active provider OpenAI-compatible base URL.",
    )
    llm_api_key: str | None = Field(
        default=None,
        description="Optional override for the active provider API key.",
    )
    llm_model: str | None = Field(
        default=None,
        description="Optional override for the active provider model name.",
    )
    llm_response_verbosity: Literal["concise", "balanced", "detailed"] = Field(
        default="balanced",
        description=(
            "Default assistant response depth. "
            "Use detailed for longer, more explanatory scientific responses."
        ),
    )
    codegen_provider: Literal["vllm", "openai", "ollama"] | None = Field(
        default=None,
        description=(
            "Optional provider override used only for Python code generation tool calls. "
            "Falls back to llm_provider when unset."
        ),
    )
    codegen_base_url: str | None = Field(
        default=None,
        description=(
            "Optional OpenAI-compatible base URL override used only by code generation tools."
        ),
    )
    codegen_api_key: str | None = Field(
        default=None,
        description="Optional API key override used only by code generation tools.",
    )
    codegen_model: str | None = Field(
        default=None,
        description=(
            "Optional model override used only by code generation tools. "
            "When unset, falls back to resolved_llm_model."
        ),
    )
    codegen_timeout_seconds: int | None = Field(
        default=None,
        description=(
            "Optional request timeout override (seconds) for code generation model calls."
        ),
    )
    pro_mode_base_url: str | None = Field(
        default=None,
        description=(
            "Optional OpenAI-compatible base URL override used only by Pro Mode answer generation."
        ),
    )
    pro_mode_transport: Literal[
        "openai_compatible",
        "bedrock_published_api",
        "aws_bedrock_claude",
    ] = Field(
        default="openai_compatible",
        description=(
            "Transport used by the dedicated Pro Mode model path. "
            "Use `bedrock_published_api` for the custom Bedrock publish endpoint that serves "
            "`/conversation` instead of OpenAI chat/completions. Use "
            "`aws_bedrock_claude` to call Anthropic Claude on AWS Bedrock via Agno's "
            "native Bedrock provider."
        ),
    )
    pro_mode_api_key: str | None = Field(
        default=None,
        description="Optional API key override used only by Pro Mode answer generation.",
    )
    pro_mode_api_key_header: str | None = Field(
        default=None,
        description=(
            "Optional auth header name for dedicated Pro Mode gateways. "
            "Set this to X-API-Key for API Gateway style publishes; leave unset for "
            "standard OpenAI Bearer authentication."
        ),
    )
    pro_mode_api_key_prefix: str | None = Field(
        default=None,
        description=(
            "Optional prefix prepended to the Pro Mode API key when "
            "pro_mode_api_key_header is set. Leave blank for raw x-api-key headers."
        ),
    )
    pro_mode_model: str | None = Field(
        default=None,
        description=(
            "Optional model override used only by Pro Mode answer generation. "
            "When unset, falls back to resolved_llm_model."
        ),
    )
    pro_mode_default_headers: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Optional JSON object of additional default headers sent to the dedicated "
            "Pro Mode gateway."
        ),
    )
    pro_mode_default_query: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Optional JSON object of default query parameters sent to the dedicated "
            "Pro Mode gateway."
        ),
    )
    pro_mode_timeout_seconds: int | None = Field(
        default=None,
        ge=1,
        description=("Optional request timeout override (seconds) for Pro Mode model calls."),
    )
    pro_mode_fallback_enabled: bool = Field(
        default=True,
        description=(
            "When true, Pro Mode retries once on the default model path if the dedicated "
            "Pro Mode gateway is unavailable."
        ),
    )
    pro_mode_aws_region: str | None = Field(
        default=None,
        description=(
            "Optional AWS region override for the native Pro Mode Bedrock Claude transport. "
            "Falls back to AWS_REGION."
        ),
    )
    pro_mode_aws_profile: str | None = Field(
        default=None,
        description=(
            "Optional AWS profile name for the native Pro Mode Bedrock Claude transport. "
            "Falls back to AWS_PROFILE."
        ),
    )
    pro_mode_aws_access_key_id: str | None = Field(
        default=None,
        description=(
            "Optional AWS access key ID override for the native Pro Mode Bedrock Claude "
            "transport. Falls back to AWS_ACCESS_KEY_ID."
        ),
    )
    pro_mode_aws_secret_access_key: str | None = Field(
        default=None,
        description=(
            "Optional AWS secret access key override for the native Pro Mode Bedrock Claude "
            "transport. Falls back to AWS_SECRET_ACCESS_KEY."
        ),
    )
    pro_mode_aws_session_token: str | None = Field(
        default=None,
        description=(
            "Optional AWS session token override for the native Pro Mode Bedrock Claude "
            "transport. Falls back to AWS_SESSION_TOKEN."
        ),
    )
    pro_mode_aws_bearer_token: str | None = Field(
        default=None,
        description=(
            "Optional AWS Bedrock bearer token override for the native Pro Mode Bedrock "
            "Claude transport. Falls back to AWS_BEARER_TOKEN_BEDROCK and then "
            "AWS_BEDROCK_API_KEY for compatibility with current Agno docs. "
            "Do not point this at API Gateway X-API-Key publishes."
        ),
    )
    pro_mode_aws_sso_auth: bool = Field(
        default=False,
        description=(
            "When true, the native Pro Mode Bedrock Claude transport uses the current AWS "
            "SSO/profile session instead of explicit access keys."
        ),
    )
    pro_mode_max_tokens: int | None = Field(
        default=None,
        ge=256,
        description=(
            "Optional output token cap for the dedicated Pro Mode model path. "
            "Applied directly to native Claude transports and passed through on "
            "OpenAI-compatible routes when supported."
        ),
    )
    pro_mode_temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Optional sampling temperature for the dedicated Pro Mode model path. "
            "Prefer lower values for technical analysis and evaluation."
        ),
    )
    pro_mode_top_p: float | None = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description=("Optional nucleus sampling parameter for the dedicated Pro Mode model path."),
    )
    pro_mode_top_k: int | None = Field(
        default=None,
        ge=1,
        le=500,
        description=("Optional top-k sampling parameter for the dedicated Pro Mode model path."),
    )
    pro_mode_claude_thinking_enabled: bool = Field(
        default=True,
        description=(
            "When true, native Claude transports attach an explicit thinking budget "
            "for deep/high-reasoning phases."
        ),
    )
    pro_mode_claude_thinking_budget_tokens: int = Field(
        default=4096,
        ge=1024,
        le=128000,
        description=("Thinking-token budget used for native Claude deep/high-reasoning phases."),
    )
    pro_mode_claude_thinking_display: Literal["summarized", "omitted"] | None = Field(
        default=None,
        description=(
            "Optional Claude thinking display mode for native transports. "
            "Leave unset to use the provider default."
        ),
    )
    llm_mock_mode: bool = Field(
        default=False,
        description=(
            "Enable deterministic in-process mock LLM responses for local integration "
            "and browser E2E testing."
        ),
    )
    agents_model_triage: str | None = Field(
        default=None,
        description="Optional model override for the interactive triage/router agent.",
    )
    agents_model_planner: str | None = Field(
        default=None,
        description="Optional model override for chat deliberation planning.",
    )
    agents_model_domain_specialist: str | None = Field(
        default=None,
        description="Optional model override for general scientific specialist agents.",
    )
    agents_model_biology: str | None = Field(
        default=None,
        description="Optional model override for biology reasoning agents.",
    )
    agents_model_verifier: str | None = Field(
        default=None,
        description="Optional model override for verification agents.",
    )
    agents_model_synthesizer: str | None = Field(
        default=None,
        description="Optional model override for synthesis/reporting agents.",
    )
    agents_model_coder: str | None = Field(
        default=None,
        description="Optional model override for code-generation/execution specialists.",
    )
    agents_model_vision: str | None = Field(
        default=None,
        description="Optional model override for image-analysis specialists.",
    )
    agents_model_medical: str | None = Field(
        default=None,
        description="Optional model override for medical-imaging specialists.",
    )
    agents_medical_base_url: str | None = Field(
        default=None,
        description=(
            "Optional dedicated OpenAI-compatible endpoint used only by the medical domain "
            "agent when configured."
        ),
    )
    agents_medical_api_key: str | None = Field(
        default=None,
        description="Optional API key for the dedicated medical domain endpoint.",
    )
    agents_medical_timeout_seconds: int = Field(
        default=60,
        ge=1,
        description="Timeout for medical-domain model calls routed to the dedicated endpoint.",
    )
    agents_medical_circuit_breaker_cooldown_seconds: int = Field(
        default=60,
        ge=1,
        description=(
            "Cooldown window after a dedicated medical endpoint availability failure before "
            "probing it again."
        ),
    )
    agents_medical_multimodal_enabled: bool = Field(
        default=True,
        description=(
            "When true, eligible medical uploads are attached as inline images for the "
            "dedicated medical model."
        ),
    )
    agents_medical_max_images_per_prompt: int = Field(
        default=3,
        ge=0,
        le=12,
        description="Maximum number of inline medical images attached to one model prompt.",
    )
    agents_medical_max_inline_image_bytes: int = Field(
        default=1_500_000,
        ge=32_768,
        description="Maximum encoded byte size allowed for one inline medical image attachment.",
    )
    agents_model_safety_governor: str | None = Field(
        default=None,
        description="Optional model override for safety/governor checks.",
    )
    agno_tracing_enabled: bool | None = Field(
        default=None,
        validation_alias=AliasChoices("AGNO_TRACING_ENABLED", "AGENTS_SDK_TRACING_ENABLED"),
        description=(
            "Enable Agno tracing for chat orchestration runs. "
            "When unset, tracing defaults on outside development."
        ),
    )
    agno_trace_sensitive_data: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "AGNO_TRACE_SENSITIVE_DATA",
            "AGENTS_SDK_TRACE_SENSITIVE_DATA",
        ),
        description="Allow sensitive prompt/tool payloads in Agno traces when tracing is enabled.",
    )
    pro_mode_expert_council_enabled: bool = Field(
        default=False,
        description=(
            "Enable the legacy multi-agent expert council inside Pro Mode. "
            "Disabled by default so release traffic prefers the more reliable "
            "single-reasoner and iterative workflow paths."
        ),
    )
    pro_mode_autonomous_cycle_enabled: bool = Field(
        default=False,
        description=(
            "Enable the gated long-cycle autonomous Pro Mode regime. "
            "Disabled by default so the resumable autonomy path can be shadowed "
            "and benchmarked before it participates in default routing."
        ),
    )
    pro_mode_autonomous_cycle_shadow_enabled: bool = Field(
        default=False,
        description=(
            "Enable additional debug and shadow observability for the autonomous-cycle "
            "family without changing default user-visible routing."
        ),
    )
    pro_mode_autonomous_cycle_agno_controller_enabled: bool = Field(
        default=True,
        description=(
            "Enable the Agno-first controller overlay for the gated autonomous-cycle "
            "regime. This keeps the public Pro Mode contract stable while letting the "
            "experimental autonomy path use explicit think/run/analyze tool orchestration."
        ),
    )
    pro_mode_autonomous_cycle_max_cycles: int = Field(
        default=12,
        ge=1,
        le=32,
        description=(
            "Hidden safety-watchdog ceiling on autonomous-cycle iterations. "
            "RTD should stop semantically before this fires; this is not the controller's "
            "primary completion policy."
        ),
    )
    pro_mode_autonomous_cycle_watchdog_runtime_seconds: int = Field(
        default=1800,
        ge=60,
        le=7200,
        description=(
            "Hidden wall-clock watchdog for autonomous-cycle runs. "
            "This prevents runaway long-think execution without acting as the semantic stop rule."
        ),
    )
    pro_mode_autonomous_cycle_watchdog_tool_calls: int = Field(
        default=48,
        ge=1,
        le=256,
        description=(
            "Hidden tool-call watchdog for autonomous-cycle runs. "
            "This protects infrastructure but should not determine when RTD considers a task solved."
        ),
    )
    pro_mode_autonomous_cycle_phase_timeout_seconds: int = Field(
        default=240,
        ge=30,
        le=1800,
        description=(
            "Per-phase request timeout for autonomous-cycle think/analyze/tool steps. "
            "This is a transport safeguard, not the semantic stop policy."
        ),
    )
    pro_mode_autonomous_cycle_transport_watchdog_seconds: int = Field(
        default=1800,
        ge=60,
        le=7200,
        description=(
            "Public chat transport watchdog used after Pro Mode selects autonomous-cycle. "
            "Prevents premature HTTP 408 responses for long-think turns while keeping a hidden ceiling."
        ),
    )
    agents_benchmark_duplicate_solve_enabled: bool = Field(
        default=False,
        description="Default benchmark-mode self-consistency challenger pass for MCQ/scientific eval runs.",
    )
    agents_benchmark_duplicate_solve_passes: int = Field(
        default=2,
        ge=1,
        le=4,
        description="Default number of bounded independent solve passes in benchmark mode.",
    )
    agents_benchmark_strict_option_elimination: bool = Field(
        default=True,
        description="Default benchmark-mode requirement to eliminate options explicitly for MCQs.",
    )
    agents_benchmark_chemistry_reasoning_boost: bool = Field(
        default=True,
        description="Default benchmark-mode prompt boost for chemistry/theory-heavy reasoning.",
    )
    agents_benchmark_biology_reasoning_boost: bool = Field(
        default=False,
        description="Default benchmark-mode prompt boost for biology/mechanism-heavy reasoning.",
    )
    agents_benchmark_biology_quant_planner_enabled: bool = Field(
        default=False,
        description="Default benchmark-mode setting to enable the structured biology quantification planner.",
    )
    agents_benchmark_biology_parallel_critic_enabled: bool = Field(
        default=False,
        description="Default benchmark-mode setting to enable the conditional biology critic branch.",
    )
    agents_benchmark_force_verifier: bool = Field(
        default=True,
        description="Default benchmark-mode requirement to keep the verifier enabled when budgets allow.",
    )
    agents_benchmark_force_code_verification: bool = Field(
        default=False,
        description="Default benchmark-mode preference to force executable verification when exact checks are feasible.",
    )

    # Legacy OpenAI-compatible settings (used by provider presets/fallbacks)
    openai_api_key: str | None = Field(
        default=None, description="OpenAI API key (optional for vLLM)"
    )
    openai_base_url: str = Field(
        default="http://localhost:8001/v1",
        description="OpenAI API base URL or vLLM endpoint",
    )
    openai_model: str = Field(default="gpt-oss-120b", description="Model name")
    openai_timeout: int = Field(default=60, description="API request timeout in seconds")
    openai_max_retries: int = Field(default=2, description="Maximum number of retries")
    ollama_base_url: str = Field(
        default="http://localhost:11434/v1",
        description="Ollama OpenAI-compatible endpoint (must include /v1).",
    )
    ollama_model: str = Field(
        default="qwen2.5:14b-instruct",
        description="Default Ollama model used when LLM_PROVIDER=ollama and LLM_MODEL is unset.",
    )

    # BisQue settings
    bisque_root: str = Field(default="http://localhost:8080", description="BisQue root URL")
    bisque_user: str | None = Field(default=None, description="BisQue username")
    bisque_password: str | None = Field(default=None, description="BisQue password")
    bisque_auth_mode: Literal["local", "oidc", "dual"] = Field(
        default="local",
        validation_alias=AliasChoices("BISQUE_AUTH_MODE"),
        description=(
            "Authentication mode advertised by the BisQue deployment. "
            "Use dual when BisQue is configured for OIDC + local tokens."
        ),
    )
    bisque_auth_local_token_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("BISQUE_AUTH_LOCAL_TOKEN_ENABLED"),
        description=(
            "Prefer BisQue local token exchange (/auth_service/token) during "
            "username/password login before falling back to basic auth session checks."
        ),
    )
    admin_usernames: str = Field(
        default="",
        description=(
            "Comma-separated BisQue usernames granted access to admin console endpoints "
            "(case-insensitive)."
        ),
    )
    bisque_auth_session_ttl_seconds: int = Field(
        default=43200,
        description="Lifetime in seconds for frontend-authenticated BisQue sessions.",
    )
    anonymous_session_ttl_seconds: int = Field(
        default=2592000,
        description=(
            "Lifetime in seconds for anonymous browser sessions used to scope per-user data "
            "before explicit login."
        ),
    )
    bisque_auth_oidc_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("BISQUE_AUTH_OIDC_ENABLED", "BISQUE_OIDC_ENABLED"),
        description="Enable Keycloak/OIDC browser login flow for frontend authentication.",
    )
    bisque_auth_oidc_via_bisque_login: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "BISQUE_AUTH_OIDC_VIA_BISQUE_LOGIN",
            "BISQUE_OIDC_VIA_BISQUE_LOGIN",
        ),
        description=(
            "Use BisQue's /auth_service/oidc_login browser flow and rely on BisQue "
            "session cookies for downstream API/tool authorization."
        ),
    )
    bisque_auth_oidc_issuer_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("BISQUE_AUTH_OIDC_ISSUER_URL", "BISQUE_OIDC_ISSUER"),
        description="OIDC issuer URL (for example: https://auth.example.com/realms/bisque).",
    )
    bisque_auth_oidc_client_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("BISQUE_AUTH_OIDC_CLIENT_ID", "BISQUE_OIDC_CLIENT_ID"),
        description="OIDC client ID used for frontend login.",
    )
    bisque_auth_oidc_client_secret: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "BISQUE_AUTH_OIDC_CLIENT_SECRET",
            "BISQUE_OIDC_CLIENT_SECRET",
        ),
        description="Optional OIDC client secret for confidential clients.",
    )
    bisque_auth_oidc_redirect_uri: str | None = Field(
        default=None,
        validation_alias=AliasChoices("BISQUE_AUTH_OIDC_REDIRECT_URI", "BISQUE_OIDC_REDIRECT_URI"),
        description=(
            "Backend callback URI registered with the OIDC provider. "
            "If unset, the API derives it from the current request host "
            "(and falls back to ORCHESTRATOR_API_URL) as "
            "<api-origin>/v1/auth/oidc/callback."
        ),
    )
    bisque_auth_oidc_frontend_redirect_url: str = Field(
        default="http://127.0.0.1:5173/",
        validation_alias=AliasChoices(
            "BISQUE_AUTH_OIDC_FRONTEND_REDIRECT_URL",
            "BISQUE_OIDC_FRONTEND_REDIRECT_URL",
        ),
        description="Frontend URL to redirect to after successful OIDC callback.",
    )
    bisque_auth_oidc_scope: str = Field(
        default="openid profile email",
        validation_alias=AliasChoices("BISQUE_AUTH_OIDC_SCOPE", "BISQUE_OIDC_SCOPES"),
        description="Space-delimited scopes requested during OIDC login.",
    )
    bisque_auth_oidc_username_claim: str = Field(
        default="preferred_username",
        validation_alias=AliasChoices(
            "BISQUE_AUTH_OIDC_USERNAME_CLAIM",
            "BISQUE_OIDC_USERNAME_CLAIM",
        ),
        description="Claim name used to resolve display username from OIDC user info.",
    )
    bisque_auth_oidc_authorize_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "BISQUE_AUTH_OIDC_AUTHORIZE_URL",
            "BISQUE_OIDC_AUTHORIZATION_ENDPOINT",
        ),
        description="Optional override for OIDC authorize endpoint.",
    )
    bisque_auth_oidc_token_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("BISQUE_AUTH_OIDC_TOKEN_URL", "BISQUE_OIDC_TOKEN_ENDPOINT"),
        description="Optional override for OIDC token endpoint.",
    )
    bisque_auth_oidc_userinfo_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "BISQUE_AUTH_OIDC_USERINFO_URL",
            "BISQUE_OIDC_USERINFO_ENDPOINT",
        ),
        description="Optional override for OIDC userinfo endpoint.",
    )
    bisque_auth_oidc_logout_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "BISQUE_AUTH_OIDC_LOGOUT_URL",
            "BISQUE_OIDC_END_SESSION_ENDPOINT",
        ),
        description="Optional override for OIDC logout endpoint.",
    )
    bisque_auth_logout_redirect_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "BISQUE_AUTH_LOGOUT_REDIRECT_URL",
            "BISQUE_OIDC_POST_LOGOUT_REDIRECT_URI",
        ),
        description=(
            "Browser redirect target after logout. Defaults to BISQUE_ROOT/client_service/ "
            "when unset."
        ),
    )

    # Orchestration settings
    orchestrator_api_url: str = Field(
        default="http://localhost:8000",
        description="Base URL for the orchestrator API (FastAPI).",
    )
    orchestrator_api_timeout: int = Field(
        default=1800,
        description="Timeout in seconds for orchestrator API requests.",
    )
    orchestrator_api_key: str | None = Field(
        default=None,
        description="Optional API key required by FastAPI orchestrator endpoints.",
    )
    allow_query_api_key_compat: bool | None = Field(
        default=None,
        description=(
            "Temporary compatibility bridge for accepting `api_key` query parameters on "
            "protected endpoints. Defaults on in development and off in production."
        ),
    )
    run_store_path: str = Field(
        default="data/runs.db",
        description="SQLite path for run metadata and trace events.",
    )
    artifact_root: str = Field(
        default="data/artifacts",
        description="Root directory where run artifacts are stored.",
    )
    session_upload_root: str = Field(
        default="data/sessions",
        description="Root directory for session-scoped uploaded files.",
    )
    science_data_root: str = Field(
        default="data/science",
        description="Root directory for universal bioio-loaded arrays and previews.",
    )
    upload_store_root: str = Field(
        default="data/uploads",
        description="Root directory for API-managed uploaded files addressed by file_id.",
    )
    upload_max_files_per_request: int = Field(
        default=64,
        description="Maximum number of files accepted in a single /v1/uploads request.",
    )
    upload_max_file_size_mb: int = Field(
        default=2048,
        description="Maximum upload size per file in megabytes for /v1/uploads.",
    )
    upload_viewer_max_dimension: int = Field(
        default=2048,
        description=(
            "Maximum width or height (pixels) for rendered upload preview PNGs "
            "served to the web viewer."
        ),
    )
    viewer_hdf5_enabled: bool = Field(
        default=True,
        description=(
            "Enable typed HDF5 viewer manifests so .h5/.hdf5 materials-science files "
            "do not fall through the image viewer path."
        ),
    )
    viewer_hdf5_atlas_max_voxels: int = Field(
        default=16_777_216,
        description=(
            "Maximum voxel count allowed for native HDF5 atlas rendering before the viewer "
            "downgrades to slice-only mode."
        ),
    )
    viewer_hdf5_atlas_max_texture_mb: int = Field(
        default=16,
        description=(
            "Maximum decoded atlas texture size, in megabytes, allowed for native HDF5 3D "
            "before the viewer downgrades to slice-only mode."
        ),
    )
    training_job_stale_heartbeat_seconds: int = Field(
        default=180,
        description=(
            "Heartbeat staleness threshold for running training/inference jobs before "
            "they are marked recoverable-failed."
        ),
    )
    training_dimension_check_max_samples: int = Field(
        default=512,
        description=(
            "Maximum samples to inspect when profiling dataset dimensionality during "
            "training preflight checks."
        ),
    )
    training_auto_proposals_enabled: bool = Field(
        default=False,
        description="Enable automatic continuous-learning proposal generation scheduler.",
    )
    yolov5_runtime_path: str = Field(
        default="third_party/yolov5",
        description="Pinned local YOLOv5 runtime path for RareSpot training/inference.",
    )
    prairie_rarespot_weights_path: str = Field(
        default="RareSpotWeights.pt",
        validation_alias=AliasChoices(
            "PRAIRIE_RARESPOT_WEIGHTS_PATH",
            "YOLOV5_RARESPOT_WEIGHTS",
        ),
        description="Default RareSpot checkpoint path for prairie active-learning workflow.",
    )
    prairie_active_learning_dataset_name: str = Field(
        default="Prairie_Dog_Active_Learning",
        description="BisQue dataset name watched by prairie active-learning sync.",
    )
    prairie_sync_interval_seconds: int = Field(
        default=6 * 60 * 60,
        description="Prairie active-learning dataset sync cadence in seconds.",
    )
    prairie_fixed_epochs: int = Field(
        default=10,
        description="Fixed YOLO training epochs for prairie active-learning retraining jobs.",
    )
    prairie_fixed_batch_size: int = Field(
        default=4,
        description="Fixed YOLO training batch size for prairie active-learning retraining jobs.",
    )
    prairie_fixed_imgsz: int = Field(
        default=512,
        description="Fixed YOLO training/inference image size for prairie active-learning.",
    )
    prairie_fixed_tile_size: int = Field(
        default=512,
        description="Fixed sliding-window tile size for prairie YOLO training and inference.",
    )
    prairie_training_tile_overlap: float = Field(
        default=0.25,
        description="Tile overlap ratio for prairie YOLO training dataset generation.",
    )
    prairie_inference_tile_overlap: float = Field(
        default=0.25,
        description="Tile overlap ratio for prairie YOLO sliding-window inference.",
    )
    prairie_include_empty_tiles: bool = Field(
        default=True,
        description="Include empty background tiles in prairie YOLO training datasets.",
    )
    prairie_inference_merge_iou_threshold: float = Field(
        default=0.45,
        description="IoU threshold for post-tiling global NMS merge in prairie YOLO inference.",
    )
    prairie_min_box_pixels: float = Field(
        default=4.0,
        description="Minimum clipped box size (pixels) when projecting annotations to tiles.",
    )
    prairie_fixed_conf_threshold: float = Field(
        default=0.25,
        description="Default detection confidence threshold for prairie inference.",
    )
    prairie_fixed_iou_threshold: float = Field(
        default=0.45,
        description="Default detection IoU threshold for prairie inference NMS.",
    )
    prairie_retrain_min_reviewed_samples: int = Field(
        default=30,
        description="Minimum reviewed images required before prairie retraining launch.",
    )
    prairie_retrain_min_total_objects: int = Field(
        default=300,
        description="Minimum reviewed object count required before prairie retraining launch.",
    )
    prairie_retrain_min_class_objects: int = Field(
        default=80,
        description=(
            "Minimum reviewed object count required per supported class "
            "(burrow/prairie_dog) before prairie retraining launch."
        ),
    )
    prairie_canonical_benchmark_spec_path: str = Field(
        default="benchmark/canonical_rare_spot.yaml",
        description=(
            "Path to canonical RareSpot benchmark spec used for retrain gating "
            "(baseline-before-train vs candidate-after-train)."
        ),
    )
    prairie_conservative_patience: int = Field(
        default=3,
        description="Patience for conservative prairie YOLO finetune profile.",
    )
    prairie_conservative_freeze_layers: int = Field(
        default=10,
        description="Number of backbone layers to freeze for conservative prairie finetuning.",
    )
    prairie_small_dataset_object_threshold: int = Field(
        default=300,
        description="If train objects are below this threshold, reduce retrain epochs.",
    )
    prairie_small_dataset_epochs: int = Field(
        default=6,
        description="Conservative epoch cap used when train object count is small.",
    )
    prairie_guardrail_canonical_map50_drop_max: float = Field(
        default=0.02,
        description="Maximum allowed canonical mAP50 drop vs baseline before promotion is blocked.",
    )
    prairie_guardrail_prairie_dog_map50_drop_max: float = Field(
        default=0.03,
        description=(
            "Maximum allowed prairie_dog class mAP50 drop on canonical benchmark "
            "vs baseline before promotion is blocked."
        ),
    )
    prairie_guardrail_active_map50_drop_max: float = Field(
        default=0.02,
        description="Maximum allowed active-holdout mAP50 drop vs baseline before promotion is blocked.",
    )
    prairie_guardrail_canonical_fp_image_increase_max: float = Field(
        default=0.25,
        description=(
            "Maximum allowed canonical false-positive-per-image increase ratio "
            "vs baseline before promotion is blocked."
        ),
    )
    prairie_enable_hard_sample_bank: bool = Field(
        default=False,
        description="Enable hard-sample bank injection for prairie YOLO retrains (v1.1 flag).",
    )
    prairie_hard_sample_injection_ratio: float = Field(
        default=0.2,
        description="Target tile ratio from hard-sample bank when enabled.",
    )
    prairie_enable_small_object_weighting: bool = Field(
        default=False,
        description="Enable small-object weighting profile tweaks for prairie retrains (v1.1 flag).",
    )
    prairie_replay_new_ratio: float = Field(
        default=0.6,
        description="Target new-data ratio for prairie replay mixing (new vs replay tiles).",
    )
    prairie_replay_old_ratio: float = Field(
        default=0.4,
        description="Target replay-data ratio for prairie replay mixing (new vs replay tiles).",
    )
    prairie_inference_prefer_sahi: bool = Field(
        default=True,
        description=(
            "Prefer SAHI sliced inference for prairie YOLO when SAHI is installed; "
            "fallback to built-in tiled detect.py flow otherwise."
        ),
    )
    code_execution_enabled: bool = Field(
        default=False,
        description="Enable LLM-authored Python code execution tools.",
    )
    code_execution_default_backend: Literal["docker", "service"] = Field(
        default="docker",
        description=(
            "Default execution backend for execute_python_job. "
            "When the dedicated service URL is configured, production deployments should prefer "
            "`service`."
        ),
    )
    code_execution_durable_default: bool = Field(
        default=True,
        description=(
            "When true, execute_python_job submits execution to a durable run before polling "
            "for completion."
        ),
    )
    code_execution_service_url: str | None = Field(
        default=None,
        description="Private base URL for the dedicated code execution service.",
    )
    code_execution_service_api_key: str | None = Field(
        default=None,
        description="Bearer token used by the backend when calling the code execution service.",
    )
    code_execution_service_timeout_seconds: int = Field(
        default=60,
        description="HTTP timeout for individual code execution service requests.",
    )
    code_execution_service_poll_interval_seconds: float = Field(
        default=1.5,
        description="Poll interval when waiting for code execution service jobs.",
    )
    code_execution_service_wait_timeout_seconds: int = Field(
        default=7200,
        description="Maximum wait time for a code execution service job before timing out.",
    )
    code_execution_default_timeout_seconds: int = Field(
        default=900,
        description="Default wall-clock timeout for Python execution jobs.",
    )
    code_execution_max_timeout_seconds: int = Field(
        default=3600,
        description="Hard upper bound for Python execution timeout.",
    )
    code_execution_default_cpu_limit: float = Field(
        default=2.0,
        description="Default Docker CPU limit for Python execution jobs.",
    )
    code_execution_default_memory_mb: int = Field(
        default=4096,
        description="Default Docker memory limit (MB) for Python execution jobs.",
    )
    code_execution_docker_image: str = Field(
        default="bisque-ultra-codeexec:py311",
        description="Docker image used by local Python execution backend.",
    )
    code_execution_docker_network: str = Field(
        default="none",
        description="Docker network mode for code execution containers.",
    )
    code_execution_max_repair_cycles: int = Field(
        default=5,
        description="Maximum auto-repair cycles attempted by code execution workflow.",
    )
    code_execution_max_total_tool_calls: int = Field(
        default=12,
        description="Max codegen+execute tool calls budget for code execution workflows.",
    )
    code_execution_poll_interval_seconds: float = Field(
        default=1.0,
        description="Poll interval for durable code execution status checks.",
    )
    context_compaction_enabled: bool = Field(
        default=True,
        description="Enable backend conversation context compaction before model calls.",
    )
    context_compaction_min_messages: int = Field(
        default=18,
        description="Minimum number of conversation messages before compaction can trigger.",
    )
    context_compaction_trigger_tokens: int = Field(
        default=12000,
        description="Approximate input-token threshold where compaction starts.",
    )
    context_compaction_target_tokens: int = Field(
        default=7000,
        description="Approximate target token budget after compaction.",
    )
    context_compaction_recent_turns: int = Field(
        default=8,
        description="Number of recent user turns to keep verbatim when compacting.",
    )
    context_compaction_max_summary_chars: int = Field(
        default=5000,
        description="Maximum characters retained in compacted conversation summary text.",
    )
    tool_call_napkin_enabled: bool = Field(
        default=True,
        description="Enable persistent napkin memory injection/update for tool-calling behavior.",
    )
    tool_call_napkin_path: str = Field(
        default=".bisque/tool_call_memory.md",
        description="Repository-local markdown path used for persistent napkin memory notes.",
    )
    tool_call_napkin_skill_path: str = Field(
        default="SKILL.md",
        description="Skill markdown path used to seed baseline napkin guardrails for tool calls.",
    )
    tool_call_napkin_max_context_chars: int = Field(
        default=3200,
        description="Maximum characters from napkin memory injected into system context.",
    )
    prompt_workpad_pipeline_enabled: bool = Field(
        default=False,
        description=(
            "Enable prompt-scoped workpad orchestration controls for response-quality improvements."
        ),
    )
    prompt_workpad_refinement_mode: Literal["legacy", "phased"] = Field(
        default="legacy",
        description=(
            "Workpad refinement mode. "
            "'legacy' keeps the existing second-pass refinement behavior; "
            "'phased' skips mandatory rewrite so the primary answer is surfaced directly."
        ),
    )
    prompt_workpad_retain_on_success: bool = Field(
        default=False,
        description=(
            "When prompt workpad pipeline is enabled, retain scratchpad artifacts after successful runs."
        ),
    )
    prompt_workpad_retain_on_failure: bool = Field(
        default=True,
        description=(
            "When prompt workpad pipeline is enabled, retain scratchpad artifacts for failed runs."
        ),
    )
    prompt_workpad_phase_h_min_tokens: int = Field(
        default=1400,
        ge=256,
        le=24000,
        description=("Minimum max_tokens budget used by phased high-effort finalization."),
    )
    prompt_workpad_phase_h_max_tokens: int = Field(
        default=6000,
        ge=512,
        le=32000,
        description=("Maximum max_tokens budget used by phased high-effort finalization."),
    )
    prompt_workpad_quality_repair_enabled: bool = Field(
        default=True,
        description=(
            "Enable one-pass response quality repair when phased final output is overly meta or incomplete."
        ),
    )
    prompt_workpad_quality_max_meta_rate: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description=("Maximum tolerated meta-narration rate before quality repair triggers."),
    )
    prompt_workpad_quality_min_completeness: float = Field(
        default=0.58,
        ge=0.0,
        le=1.0,
        description=("Minimum answer completeness score required to skip quality repair."),
    )
    prompt_workpad_quality_min_numeric_density: float = Field(
        default=0.45,
        ge=0.0,
        le=25.0,
        description=(
            "Minimum numeric-detail density (numbers per 100 words) expected for numeric-heavy prompts."
        ),
    )

    deliberate_mode: bool = Field(
        default=False,
        description="Enable extra behind-the-scenes planning refinement.",
    )

    # OCR backend settings
    ocr_base_url: str = Field(
        default="http://localhost:8000/v1",
        description="Base URL for OCR model endpoint.",
    )
    ocr_api_key: str = Field(
        default="EMPTY",
        description="API key for OCR model endpoint.",
    )
    ocr_model: str = Field(
        default="deepseek-ai/DeepSeek-OCR",
        description="Model identifier for OCR requests.",
    )
    ocr_timeout: int = Field(
        default=3600,
        description="Timeout in seconds for OCR requests.",
    )

    # MedSAM2 defaults
    medsam2_model_id: str = Field(
        default="wanglab/MedSAM2",
        description="Default MedSAM2 model selector (e.g., wanglab/MedSAM2, latest, or a .pt path).",
    )
    medsam2_runtime_root: str = Field(
        default="third_party/MedSAM2",
        description=(
            "Directory containing the vendored MedSAM2 runtime (sam2 package and configs). "
            "Use an absolute host path in production if the runtime is installed outside the release tree."
        ),
    )
    medsam2_checkpoint_dir: str = Field(
        default="data/models/medsam2/checkpoints",
        description="Directory where MedSAM2 .pt checkpoints are stored.",
    )
    medsam2_default_checkpoint: str = Field(
        default="MedSAM2_latest.pt",
        description="Default MedSAM2 checkpoint filename loaded from medsam2_checkpoint_dir.",
    )
    medsam2_config_file: str = Field(
        default="sam2.1_hiera_t512.yaml",
        description="Official MedSAM2 config file used with checkpoint-based inference.",
    )
    medsam2_max_slices: int = Field(
        default=160,
        description="Maximum number of volume slices processed directly for MedSAM2.",
    )

    # Megaseg defaults
    megaseg_python: str | None = Field(
        default=None,
        description=(
            "Optional dedicated Python interpreter path used for Megaseg/cyto-dl inference. "
            "Prefer an absolute path in production."
        ),
    )
    cytodl_python: str | None = Field(
        default=None,
        description=("Backward-compatible alias for the Megaseg runtime Python interpreter path."),
    )
    megaseg_checkpoint_path: str | None = Field(
        default=None,
        description=("Optional Megaseg checkpoint path. Prefer an absolute path in production."),
    )
    megaseg_benchmark_root: str | None = Field(
        default=None,
        description=(
            "Optional Megaseg benchmark checkout root used to discover default "
            "checkpoint locations."
        ),
    )
    megaseg_service_url: str | None = Field(
        default=None,
        description=(
            "Optional private Megaseg inference service base URL. When set, "
            "segment_image_megaseg submits inference jobs to the remote service "
            "instead of launching the local runner subprocess."
        ),
    )
    megaseg_service_api_key: str | None = Field(
        default=None,
        description="Bearer token used when authenticating to the private Megaseg service.",
    )
    megaseg_service_timeout_seconds: float = Field(
        default=60.0,
        description="Per-request HTTP timeout in seconds when calling the Megaseg service.",
    )
    megaseg_service_poll_interval_seconds: float = Field(
        default=2.0,
        description="Polling interval in seconds while waiting for Megaseg service jobs.",
    )
    megaseg_service_wait_timeout_seconds: float = Field(
        default=7200.0,
        description="Maximum wall-clock time in seconds to wait for a Megaseg service job.",
    )
    megaseg_service_download_artifacts: bool = Field(
        default=True,
        description=(
            "When true, download Megaseg service artifacts into the app science output "
            "tree so downstream tools can use the usual local artifact layout."
        ),
    )

    # SAM3 defaults
    sam3_model_id: str = Field(
        default="facebook/sam3",
        description=(
            "Default SAM3 model reference. Prefer a local snapshot path in offline deployments."
        ),
    )
    sam3_allow_remote_download: bool = Field(
        default=False,
        description="Allow SAM3 model downloads from remote hubs when local snapshots are unavailable.",
    )
    sam3_max_slices: int = Field(
        default=192,
        description="Maximum number of slices processed explicitly for SAM3 slice-wise segmentation.",
    )
    sam3_window_size: int = Field(
        default=1024,
        description="Sliding-window size for SAM3 large-image inference.",
    )
    sam3_window_overlap: float = Field(
        default=0.25,
        description="Sliding-window overlap ratio for SAM3 inference.",
    )
    sam3_min_points: int = Field(
        default=8,
        description="Minimum auto-prompt points per SAM3 window.",
    )
    sam3_max_points: int = Field(
        default=64,
        description="Maximum auto-prompt points per SAM3 window.",
    )
    sam3_point_density: float = Field(
        default=0.0015,
        description="Density factor controlling SAM3 auto-prompt point count per window area.",
    )
    sam3_mask_threshold: float = Field(
        default=0.5,
        description="Mask binarization threshold for SAM3 tracker outputs.",
    )
    sam3_vote_threshold: float = Field(
        default=0.5,
        description="Per-pixel vote ratio required when merging overlapping SAM3 windows.",
    )
    sam3_min_component_area_ratio: float = Field(
        default=0.0001,
        description="Minimum connected-component area ratio retained after SAM3 segmentation.",
    )
    sam3_modality_hint: str = Field(
        default="auto",
        description="Preprocessing profile hint for SAM3 (auto, fluorescence, brightfield, ct_like, generic).",
    )
    sam3_refine_3d: bool = Field(
        default=True,
        description="Apply 3D temporal/component consistency refinement for volumetric SAM3 outputs.",
    )
    sam3_fallback_to_medsam2: bool = Field(
        default=True,
        description="Fallback to MedSAM2 if SAM3 initialization/inference is unavailable.",
    )
    sam3_auto_zero_mask_fallback_enabled: bool = Field(
        default=True,
        description=(
            "If SAM3 auto mode produces an empty mask, retry once with SAM3 concept mode "
            "using sam3_auto_zero_mask_fallback_prompt."
        ),
    )
    sam3_auto_zero_mask_fallback_prompt: str = Field(
        default="object",
        description=("Concept prompt used for SAM3 fallback when auto mode returns an empty mask."),
    )
    sam3_auto_zero_mask_fallback_prompts: str = Field(
        default="",
        description=(
            "Optional comma-separated concept prompts tried (in order) when SAM3 auto mode returns an empty mask."
        ),
    )
    sam3_auto_low_coverage_fallback_threshold_percent: float = Field(
        default=0.0,
        description=(
            "If > 0, SAM3 auto mode concept fallback is also triggered when coverage is below this percentage."
        ),
    )
    sequence_max_frames_per_file: int = Field(
        default=24,
        description="Maximum extracted frames per sequence/video file when running 2D tools on temporal inputs.",
    )
    sequence_frame_stride: int = Field(
        default=4,
        description="Frame stride used when sampling sequence/video files for 2D model tools.",
    )
    depth_pro_model_id: str = Field(
        default="apple/DepthPro-hf",
        description="DepthPro model reference used by estimate_depth_pro.",
    )
    depth_pro_use_fov_model: bool = Field(
        default=True,
        description="Enable DepthPro field-of-view head when supported by the selected checkpoint.",
    )

    def model_post_init(self, __context: Any) -> None:
        path_fields = (
            "run_store_path",
            "artifact_root",
            "session_upload_root",
            "science_data_root",
            "upload_store_root",
            "medsam2_runtime_root",
            "medsam2_checkpoint_dir",
            "megaseg_python",
            "cytodl_python",
            "megaseg_checkpoint_path",
            "megaseg_benchmark_root",
            "tool_call_napkin_path",
            "tool_call_napkin_skill_path",
        )
        for field_name in path_fields:
            value = getattr(self, field_name, None)
            if isinstance(value, str) and value.strip():
                setattr(self, field_name, _resolve_project_path(value))
        if str(self.environment).strip().lower() == "production":
            run_store_value = str(self.run_store_path or "").strip()
            if not _is_connection_url(run_store_value):
                raise ValueError(
                    "ENVIRONMENT=production requires RUN_STORE_PATH to be a Postgres URL."
                )

    @property
    def resolved_llm_base_url(self) -> str:
        """Resolved OpenAI-compatible base URL for the active LLM provider."""
        if self.llm_base_url and self.llm_base_url.strip():
            return self.llm_base_url.strip()
        if self.llm_provider == "ollama":
            return self.ollama_base_url.strip()
        return self.openai_base_url.strip()

    @property
    def resolved_llm_api_key(self) -> str | None:
        """Resolved API key for the active LLM provider."""
        if self.llm_api_key and self.llm_api_key.strip():
            return self.llm_api_key.strip()
        if self.llm_provider == "ollama":
            return None
        if self.openai_api_key and self.openai_api_key.strip():
            return self.openai_api_key.strip()
        return None

    @property
    def resolved_llm_model(self) -> str:
        """Resolved model name for the active LLM provider."""
        if self.llm_model and self.llm_model.strip():
            return self.llm_model.strip()
        if self.llm_provider == "ollama":
            return self.ollama_model.strip()
        return self.openai_model.strip()

    @property
    def resolved_codegen_provider(self) -> Literal["vllm", "openai", "ollama"]:
        """Resolved provider used by code-generation tools."""
        raw = str(self.codegen_provider or "").strip().lower()
        if raw in {"vllm", "openai", "ollama"}:
            return raw  # type: ignore[return-value]
        return self.llm_provider

    @property
    def resolved_codegen_base_url(self) -> str:
        """Resolved OpenAI-compatible base URL for code-generation tools."""
        if self.codegen_base_url and self.codegen_base_url.strip():
            return self.codegen_base_url.strip()
        if self.resolved_codegen_provider == self.llm_provider:
            return self.resolved_llm_base_url
        if self.resolved_codegen_provider == "ollama":
            return self.ollama_base_url.strip()
        return self.openai_base_url.strip()

    @property
    def resolved_codegen_api_key(self) -> str | None:
        """Resolved API key for code-generation tools."""
        if self.codegen_api_key and self.codegen_api_key.strip():
            return self.codegen_api_key.strip()
        if self.resolved_codegen_provider == self.llm_provider:
            return self.resolved_llm_api_key
        if self.resolved_codegen_provider == "ollama":
            return None
        if self.openai_api_key and self.openai_api_key.strip():
            return self.openai_api_key.strip()
        if self.resolved_codegen_provider in {"vllm", "ollama"}:
            return "EMPTY"
        return None

    @property
    def resolved_codegen_model(self) -> str:
        """Resolved model name for code-generation tools."""
        if self.codegen_model and self.codegen_model.strip():
            return self.codegen_model.strip()
        if self.resolved_codegen_provider == self.llm_provider:
            return self.resolved_llm_model
        if self.resolved_codegen_provider == "ollama":
            return self.ollama_model.strip()
        return self.openai_model.strip()

    @property
    def resolved_pro_mode_base_url(self) -> str:
        """Resolved OpenAI-compatible base URL for Pro Mode answer generation."""
        if self.pro_mode_base_url and self.pro_mode_base_url.strip():
            return self.pro_mode_base_url.strip()
        return self.resolved_llm_base_url

    @property
    def resolved_pro_mode_api_key(self) -> str | None:
        """Resolved API key for Pro Mode answer generation."""
        if self.pro_mode_api_key and self.pro_mode_api_key.strip():
            return self.pro_mode_api_key.strip()
        return self.resolved_llm_api_key

    @property
    def resolved_pro_mode_model(self) -> str:
        """Resolved model name for Pro Mode answer generation."""
        if self.pro_mode_model and self.pro_mode_model.strip():
            return self.pro_mode_model.strip()
        return self.resolved_llm_model

    @property
    def resolved_pro_mode_timeout_seconds(self) -> int:
        """Resolved timeout in seconds for Pro Mode model calls."""
        if self.pro_mode_timeout_seconds is not None:
            return max(1, int(self.pro_mode_timeout_seconds))
        return max(1, int(self.openai_timeout or 60))

    @property
    def resolved_allow_query_api_key_compat(self) -> bool:
        """Whether protected endpoints should still accept `api_key` query parameters."""
        if self.allow_query_api_key_compat is not None:
            return bool(self.allow_query_api_key_compat)
        return str(self.environment).strip().lower() != "production"

    @property
    def resolved_megaseg_python(self) -> str | None:
        """Resolved Python interpreter path used for Megaseg inference."""
        for candidate in (self.megaseg_python, self.cytodl_python):
            if candidate and candidate.strip():
                return candidate.strip()
        for env_name in ("MEGASEG_PYTHON", "CYTODL_PYTHON"):
            env_value = os.getenv(env_name)
            if env_value and env_value.strip():
                return _resolve_project_path(env_value)
        return None

    @property
    def resolved_megaseg_checkpoint_path(self) -> str | None:
        """Resolved Megaseg checkpoint path from settings or environment."""
        if self.megaseg_checkpoint_path and self.megaseg_checkpoint_path.strip():
            return self.megaseg_checkpoint_path.strip()
        env_value = os.getenv("MEGASEG_CHECKPOINT_PATH")
        if env_value and env_value.strip():
            return _resolve_project_path(env_value)
        return None

    @property
    def resolved_megaseg_benchmark_root(self) -> str | None:
        """Resolved Megaseg benchmark checkout root from settings or environment."""
        if self.megaseg_benchmark_root and self.megaseg_benchmark_root.strip():
            return self.megaseg_benchmark_root.strip()
        env_value = os.getenv("MEGASEG_BENCHMARK_ROOT")
        if env_value and env_value.strip():
            return _resolve_project_path(env_value)
        return None

    @property
    def resolved_megaseg_service_url(self) -> str | None:
        """Resolved Megaseg inference service base URL."""
        if self.megaseg_service_url and self.megaseg_service_url.strip():
            return self.megaseg_service_url.strip().rstrip("/")
        env_value = os.getenv("MEGASEG_SERVICE_URL")
        if env_value and env_value.strip():
            return env_value.strip().rstrip("/")
        return None

    @property
    def resolved_megaseg_service_api_key(self) -> str | None:
        """Resolved bearer token for the private Megaseg inference service."""
        if self.megaseg_service_api_key and self.megaseg_service_api_key.strip():
            return self.megaseg_service_api_key.strip()
        env_value = os.getenv("MEGASEG_SERVICE_API_KEY")
        if env_value and env_value.strip():
            return env_value.strip()
        return None

    @property
    def resolved_pro_mode_aws_region(self) -> str | None:
        """Resolved AWS region for the native Pro Mode Bedrock Claude transport."""
        if self.pro_mode_aws_region and self.pro_mode_aws_region.strip():
            return self.pro_mode_aws_region.strip()
        env_value = os.getenv("AWS_REGION")
        return env_value.strip() if env_value and env_value.strip() else None

    @property
    def resolved_pro_mode_aws_profile(self) -> str | None:
        """Resolved AWS profile for the native Pro Mode Bedrock Claude transport."""
        if self.pro_mode_aws_profile and self.pro_mode_aws_profile.strip():
            return self.pro_mode_aws_profile.strip()
        env_value = os.getenv("AWS_PROFILE")
        return env_value.strip() if env_value and env_value.strip() else None

    @property
    def resolved_pro_mode_aws_access_key_id(self) -> str | None:
        """Resolved AWS access key for the native Pro Mode Bedrock Claude transport."""
        if self.pro_mode_aws_access_key_id and self.pro_mode_aws_access_key_id.strip():
            return self.pro_mode_aws_access_key_id.strip()
        env_value = os.getenv("AWS_ACCESS_KEY_ID")
        return env_value.strip() if env_value and env_value.strip() else None

    @property
    def resolved_pro_mode_aws_secret_access_key(self) -> str | None:
        """Resolved AWS secret for the native Pro Mode Bedrock Claude transport."""
        if self.pro_mode_aws_secret_access_key and self.pro_mode_aws_secret_access_key.strip():
            return self.pro_mode_aws_secret_access_key.strip()
        env_value = os.getenv("AWS_SECRET_ACCESS_KEY")
        return env_value.strip() if env_value and env_value.strip() else None

    @property
    def resolved_pro_mode_aws_session_token(self) -> str | None:
        """Resolved AWS session token for the native Pro Mode Bedrock Claude transport."""
        if self.pro_mode_aws_session_token and self.pro_mode_aws_session_token.strip():
            return self.pro_mode_aws_session_token.strip()
        env_value = os.getenv("AWS_SESSION_TOKEN")
        return env_value.strip() if env_value and env_value.strip() else None

    @property
    def resolved_pro_mode_aws_bearer_token(self) -> str | None:
        """Resolved AWS Bedrock bearer token for the native Pro Mode transport."""
        if self.pro_mode_aws_bearer_token and self.pro_mode_aws_bearer_token.strip():
            return self.pro_mode_aws_bearer_token.strip()
        for env_name in ("AWS_BEARER_TOKEN_BEDROCK", "AWS_BEDROCK_API_KEY"):
            env_value = os.getenv(env_name)
            if env_value and env_value.strip():
                return env_value.strip()
        return None


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()
