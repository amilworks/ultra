"""Function calling tools for the LLM."""

import base64
import hashlib
import inspect
import io
import json
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from uuid import uuid4

import httpx
import numpy as np
from lxml import etree
from PIL import Image, ImageDraw, ImageOps

from src.auth import get_bisque_session, get_request_bisque_auth
from src.bqapi_bootstrap import LOCAL_BQAPI_READY
from src.config import get_settings
from src.logger import logger
from src.science.chemistry import (
    compare_structures,
    formula_balance_check,
    propose_reactive_sites,
    structure_report,
)
from src.science.imaging import load_scientific_image
from src.science.medsam2 import segment_array_with_medsam2
from src.science.megaseg_service_client import (
    MegasegServiceClient,
    MegasegServiceError,
    MegasegServiceTimeoutError,
)
from src.science.sam3 import (
    segment_array_with_sam3,
    segment_array_with_sam3_concept,
    segment_array_with_sam3_points,
)
from src.science.spectral_instability import (
    SPECTRAL_DEFAULT_CONF_THRES,
    SPECTRAL_DEFAULT_IOU_THRES,
    SPECTRAL_DEFAULT_PRESERVATION_RATIO,
    SpectralInstabilityConfig,
    score_spectral_instability,
)
from src.tooling.calculator import numpy_calculator
from src.tooling.code_execution import (
    execute_python_job_once,
    load_python_job_spec,
    prepare_python_job,
)
from src.tooling.domains import (
    analyze_csv,
    compare_conditions,
    plot_quantified_detections,
    quantify_objects,
    repro_report,
)
from src.tooling.domains import (
    stats_list_curated_tools as _domain_stats_list_curated_tools,
)
from src.tooling.domains import (
    stats_run_curated_tool as _domain_stats_run_curated_tool,
)

assert LOCAL_BQAPI_READY

_BISQUE_ENV_PLACEHOLDER_RE = re.compile(
    r"^(?:\$\{?(?:BI(?:S)?QUE_(?:USER|PASSWORD|ROOT)|ENV)\}?|BI(?:S)?QUE_(?:USER|PASSWORD|ROOT))$",
    re.IGNORECASE,
)


def _safe_slug(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", (name or "").strip())
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug or "model"


def _get_bisque_auth_material() -> tuple[
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
]:
    settings = get_settings()
    request_auth = get_request_bisque_auth()
    if request_auth:
        username = str(request_auth.username or "").strip() or None
        password = str(request_auth.password or "").strip() or None
        access_token = str(request_auth.access_token or "").strip() or None
        bisque_cookie_header = str(request_auth.bisque_cookie_header or "").strip() or None
        bisque_root = (
            str(request_auth.bisque_root or "").strip().rstrip("/")
            or str(getattr(settings, "bisque_root", "") or "").strip().rstrip("/")
            or None
        )
        return username, password, access_token, bisque_root, bisque_cookie_header

    user = settings.bisque_user
    password = settings.bisque_password
    bisque_root = str(getattr(settings, "bisque_root", "") or "").strip().rstrip("/") or None
    access_token = None
    bisque_cookie_header = None
    return user, password, access_token, bisque_root, bisque_cookie_header


def _get_bisque_credentials() -> tuple[str | None, str | None]:
    user, password, _, _, _ = _get_bisque_auth_material()
    return user, password


def _get_bisque_access_token() -> str | None:
    _, _, access_token, _, _ = _get_bisque_auth_material()
    return access_token


def _get_bisque_root() -> str | None:
    _, _, _, bisque_root, _ = _get_bisque_auth_material()
    return bisque_root


def _get_bisque_cookie_header() -> str | None:
    _, _, _, _, cookie_header = _get_bisque_auth_material()
    return cookie_header


def _request_bisque_local_access_token(
    *,
    username: str,
    password: str,
    bisque_root: str,
) -> str | None:
    normalized_root = str(bisque_root or "").strip().rstrip("/")
    if not normalized_root or not username or not password:
        return None
    try:
        response = httpx.post(
            f"{normalized_root}/auth_service/token",
            data={
                "username": str(username or "").strip(),
                "password": str(password or "").strip(),
            },
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json,*/*",
            },
            timeout=8.0,
            follow_redirects=True,
        )
        response.raise_for_status()
        payload = response.json()
        token = str(payload.get("access_token") or "").strip()
        return token or None
    except Exception as exc:  # noqa: BLE001
        logger.debug("BisQue local token bootstrap unavailable: %s", exc)
        return None


def _resolve_bisque_runtime_auth(
    *,
    explicit_user: str | None = None,
    explicit_password: str | None = None,
    explicit_root: str | None = None,
) -> tuple[str | None, str | None, str | None, str | None, str | None]:
    """
    Resolve BisQue auth for a tool call.

    When request-scoped auth exists (frontend login), do not fall back to .env
    credentials. This ensures user sessions are used for BisQue access.
    """
    settings = get_settings()
    request_auth = get_request_bisque_auth()
    (
        session_user,
        session_password,
        session_access_token,
        session_root,
        session_cookie_header,
    ) = _get_bisque_auth_material()

    username = _sanitize_bisque_credential(explicit_user) or _sanitize_bisque_credential(
        session_user
    )
    password = _sanitize_bisque_credential(explicit_password) or _sanitize_bisque_credential(
        session_password
    )
    access_token = _sanitize_bisque_credential(session_access_token)
    cookie_header = str(session_cookie_header or "").strip() or None
    root = (
        _sanitize_bisque_root(explicit_root)
        or _sanitize_bisque_root(session_root)
        or _sanitize_bisque_root(str(getattr(settings, "bisque_root", "") or ""))
    )

    if request_auth is None:
        username = username or _sanitize_bisque_credential(
            str(getattr(settings, "bisque_user", "") or "")
        )
        password = password or _sanitize_bisque_credential(
            str(getattr(settings, "bisque_password", "") or "")
        )
    elif not access_token and not cookie_header and username and password and root:
        access_token = _request_bisque_local_access_token(
            username=username,
            password=password,
            bisque_root=root,
        )

    return username, password, access_token, root, cookie_header


def _init_bisque_session_with_runtime_auth(
    *,
    bq: Any,
    explicit_user: str | None = None,
    explicit_password: str | None = None,
    explicit_root: str | None = None,
    preferred_auth_mode: str = "auto",
) -> tuple[str, str, bool]:
    """
    Initialize BQSession with request-scoped auth (token-first), with legacy
    .env fallback only when no request auth context exists.
    """
    username, password, access_token, root, cookie_header = _resolve_bisque_runtime_auth(
        explicit_user=explicit_user,
        explicit_password=explicit_password,
        explicit_root=explicit_root,
    )
    if not root:
        raise ValueError("BisQue root URL required. Set BISQUE_ROOT in .env")
    username, password, access_token, cookie_header = _apply_bisque_auth_preference(
        username=username,
        password=password,
        access_token=access_token,
        cookie_header=cookie_header,
        preferred_auth_mode=preferred_auth_mode,
    )
    token_mode, auth_mode = _init_bq_session(
        bq,
        username=username,
        password=password,
        access_token=access_token,
        bisque_root=root,
        cookie_header=cookie_header,
    )
    return root, auth_mode, token_mode


def _apply_bisque_auth_preference(
    *,
    username: str | None,
    password: str | None,
    access_token: str | None,
    cookie_header: str | None,
    preferred_auth_mode: str = "auto",
) -> tuple[str | None, str | None, str | None, str | None]:
    auth_preference = str(preferred_auth_mode or "auto").strip().lower()
    resolved_token = access_token
    resolved_cookie = cookie_header

    if auth_preference == "basic":
        if username and password:
            resolved_token = None
            resolved_cookie = None
        elif resolved_cookie:
            resolved_token = None
    elif auth_preference == "cookie" and resolved_cookie:
        resolved_token = None
    elif auth_preference in {"token", "bearer"} and resolved_token:
        resolved_cookie = None

    return username, password, resolved_token, resolved_cookie


def _init_bq_session(
    bq: Any,
    *,
    username: str | None,
    password: str | None,
    access_token: str | None,
    bisque_root: str,
    cookie_header: str | None = None,
) -> tuple[bool, str]:
    """
    Initialize BQSession in cookie/token/basic mode.

    Returns:
        (token_mode, auth_mode_label)
    """

    class _BearerTokenAuth:
        def __init__(self, token: str):
            self._token = token

        def __call__(self, request: Any) -> Any:
            request.headers["Authorization"] = f"Bearer {self._token}"
            return request

    def _init_with_bearer_auth() -> bool:
        if not hasattr(bq, "c") or not hasattr(bq, "_load_services"):
            return False
        if not hasattr(getattr(bq, "c", None), "auth"):
            return False
        bq.bisque_root = bisque_root
        bq.c.root = bisque_root
        bq.c.auth = _BearerTokenAuth(str(access_token))
        bq._load_services()
        session_ok = False
        if hasattr(bq, "_check_session"):
            try:
                session_ok = bool(bq._check_session())
            except Exception as exc:
                logger.warning("Bearer token session check failed, probing whoami: %s", exc)
                session_ok = False
        if not session_ok:
            try:
                whoami = bq.fetchxml(f"{bisque_root}/auth_service/whoami")
                if whoami is None:
                    raise ValueError("empty whoami response")
                session_ok = True
            except Exception as exc:
                raise ValueError(f"Bearer token authentication failed: {exc}") from exc
        return True

    def _init_with_cookie_auth() -> bool:
        header = str(cookie_header or "").strip()
        if not header:
            return False
        if not hasattr(bq, "c") or not hasattr(bq, "_load_services"):
            return False
        if not hasattr(getattr(bq, "c", None), "headers"):
            return False
        bq.bisque_root = bisque_root
        bq.c.root = bisque_root
        bq.c.headers["Cookie"] = header
        bq._load_services()
        try:
            whoami = bq.fetchxml(f"{bisque_root}/auth_service/whoami")
        except Exception as exc:
            raise ValueError(f"BisQue cookie authentication failed: {exc}") from exc
        resolved_user = _extract_bisque_user_from_xml(whoami)
        normalized_user = str(resolved_user or "").strip().lower()
        if not normalized_user or normalized_user in {
            "anonymous",
            "guest",
            "anon",
            "public",
            "none",
        }:
            raise ValueError("BisQue cookie authentication resolved to anonymous identity.")
        return True

    if access_token:
        token_error: Exception | None = None
        if hasattr(bq, "init_token"):
            try:
                token_user = str(username or "").strip() or None
                bq.init_token(
                    access_token,
                    bisque_root=bisque_root,
                    create_mex=False,
                    user=token_user,
                )
                return True, "token"
            except Exception as exc:
                token_error = exc
                logger.warning("BisQue init_token failed; trying bearer fallback: %s", exc)

        if _init_with_bearer_auth():
            return True, "bearer"

        if token_error is not None:
            logger.warning(
                "Token authentication unavailable/failed (%s); falling back to basic auth when possible.",
                token_error,
            )

    if cookie_header:
        cookie_error: Exception | None = None
        try:
            if _init_with_cookie_auth():
                return False, "cookie"
        except Exception as exc:
            cookie_error = exc
            logger.warning("BisQue cookie auth failed; trying basic auth fallback: %s", exc)
        if cookie_error is not None:
            logger.debug("Cookie auth exception details", exc_info=cookie_error)

    if username and password:
        bq.init_local(
            username,
            password,
            bisque_root=bisque_root,
            create_mex=False,
        )
        try:
            whoami = bq.fetchxml(f"{bisque_root}/auth_service/whoami")
        except Exception as exc:
            raise ValueError(f"BisQue basic authentication failed: {exc}") from exc
        resolved_user = _extract_bisque_user_from_xml(whoami)
        normalized_user = str(resolved_user or "").strip().lower()
        if not normalized_user or normalized_user in {
            "anonymous",
            "guest",
            "anon",
            "public",
            "none",
        }:
            raise ValueError("BisQue basic authentication resolved to anonymous identity.")
        return False, "basic"
    raise ValueError(
        "BisQue authentication required. Provide BisQue session cookie, API access token, or BISQUE_USER/BISQUE_PASSWORD."
    )


def _extract_bisque_resource_uniq(upload_response: Any) -> str | None:
    """Best-effort extraction of resource_uniq from bqapi postblob responses."""
    try:
        if hasattr(upload_response, "get"):
            ru = upload_response.get("resource_uniq")
            if ru:
                return str(ru)
    except Exception:
        pass

    try:
        if isinstance(upload_response, (bytes, bytearray)):
            text = upload_response.decode("utf-8", errors="ignore")
        else:
            text = str(upload_response)
    except Exception:
        return None

    m = re.search(r'resource_uniq="([^"]+)"', text)
    return m.group(1) if m else None


def _extract_bisque_resource_uri(upload_response: Any) -> str | None:
    """Best-effort extraction of full resource URI from bqapi postblob responses."""
    try:
        if hasattr(upload_response, "get"):
            uri = upload_response.get("uri") or upload_response.get("resource_uri")
            if uri:
                return str(uri)
    except Exception:
        pass

    try:
        if hasattr(upload_response, "attrib"):
            uri = upload_response.attrib.get("uri")
            if uri:
                return str(uri)
    except Exception:
        pass

    try:
        if isinstance(upload_response, (bytes, bytearray)):
            text = upload_response.decode("utf-8", errors="ignore")
        else:
            text = str(upload_response)
    except Exception:
        return None

    m = re.search(r'\buri="([^"]+)"', text)
    return m.group(1) if m else None


def _upload_response_error_hint(upload_response: Any) -> str | None:
    try:
        if isinstance(upload_response, (bytes, bytearray)):
            text = upload_response.decode("utf-8", errors="ignore")
        else:
            text = str(upload_response)
    except Exception:
        return None
    lowered = text.lower()
    if "401 unauthorized" in lowered:
        return "BisQue server returned 401 Unauthorized for upload."
    if "403 forbidden" in lowered:
        return "BisQue server returned 403 Forbidden for upload."
    if "500 internal server error" in lowered:
        return "BisQue server returned 500 Internal Server Error during upload."
    if "<html" in lowered and ("unauthorized" in lowered or "forbidden" in lowered):
        return "BisQue upload returned an HTML auth error page."
    return None


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest().lower()


def _request_scoped_bisque_user_id() -> str | None:
    request_auth = get_request_bisque_auth()
    if request_auth is None:
        return None
    username = str(getattr(request_auth, "username", "") or "").strip().lower()
    if not username:
        return None
    if username in {"anonymous", "guest", "anon", "public", "none"}:
        return None
    return f"bisque:{username}"


def _lookup_upload_session_auth_material_for_resource(
    resource_uri: str | None,
) -> tuple[str | None, str | None, str | None, str | None, str | None] | None:
    resource_token = str(resource_uri or "").strip()
    if not resource_token:
        return None
    settings = get_settings()
    run_store_path = str(getattr(settings, "run_store_path", "") or "").strip()
    if not run_store_path:
        return None
    try:
        from src.orchestration.store import RunStore
    except Exception:
        return None

    user_id = _request_scoped_bisque_user_id()
    try:
        store = RunStore(run_store_path)
        upload_row = store.get_upload(resource_token, user_id=user_id, include_deleted=False)
        if upload_row is None and user_id is not None:
            upload_row = store.get_upload(resource_token, user_id=None, include_deleted=False)
    except Exception as exc:
        logger.debug("Upload lookup failed for BisQue resource %s: %s", resource_token, exc)
        return None
    if not isinstance(upload_row, dict):
        return None
    metadata = upload_row.get("metadata") if isinstance(upload_row.get("metadata"), dict) else {}
    session_id = str(metadata.get("bisque_session_id") or "").strip()
    if not session_id:
        return None
    ttl_seconds = max(900, int(getattr(settings, "bisque_auth_session_ttl_seconds", 43200)))
    try:
        session = get_bisque_session(
            session_id,
            ttl_seconds=ttl_seconds,
            touch=True,
        )
    except Exception as exc:
        logger.debug("BisQue upload session lookup failed for %s: %s", resource_token, exc)
        return None
    if not isinstance(session, dict):
        return None
    return (
        str(session.get("username") or "").strip() or None,
        str(session.get("password") or "").strip() or None,
        str(session.get("access_token") or "").strip() or None,
        str(session.get("bisque_root") or "").strip().rstrip("/") or None,
        str(session.get("bisque_cookie_header") or "").strip() or None,
    )


def _lookup_canonical_bisque_upload_for_local_path(
    path: str | Path,
) -> dict[str, Any] | None:
    candidate = Path(path).expanduser()
    if not candidate.exists() or not candidate.is_file():
        return None
    try:
        digest = _sha256_file(candidate)
    except Exception:
        return None
    user_id = _request_scoped_bisque_user_id()
    if not user_id:
        return None
    try:
        from src.orchestration.store import RunStore
    except Exception:
        return None

    settings = get_settings()
    run_store_path = str(getattr(settings, "run_store_path", "") or "").strip()
    if not run_store_path:
        return None
    try:
        store = RunStore(run_store_path)
        row = store.find_upload_by_sha256(
            sha256=digest,
            user_id=user_id,
            include_deleted=False,
            require_canonical=True,
        )
    except Exception as exc:
        logger.debug("Canonical upload lookup failed for %s: %s", candidate, exc)
        return None
    if not isinstance(row, dict):
        return None
    canonical_resource_uri = str(row.get("canonical_resource_uri") or "").strip()
    sync_status = str(row.get("sync_status") or "").strip().lower()
    if not canonical_resource_uri or sync_status != "bisque_sync_succeeded":
        return None
    return row


def _bisque_resource_visible_in_session(
    *,
    bq: Any,
    bisque_root: str,
    resource_uri: str | None,
) -> bool:
    normalized_uri = str(resource_uri or "").strip()
    if not normalized_uri:
        return False
    try:
        target_uri = _normalize_bisque_resource_uri(normalized_uri, bisque_root)
    except Exception:
        return False
    try:
        _session_fetchxml_safe(bq, target_uri, view="short", cache="false")
    except Exception:
        return False
    return True


def _lookup_session_visible_canonical_bisque_upload_for_local_path(
    *,
    bq: Any,
    bisque_root: str,
    path: str | Path,
) -> dict[str, Any] | None:
    existing_upload = _lookup_canonical_bisque_upload_for_local_path(path)
    if existing_upload is None:
        return None
    canonical_resource_uri = str(existing_upload.get("canonical_resource_uri") or "").strip()
    if not canonical_resource_uri:
        return None
    if not _bisque_resource_visible_in_session(
        bq=bq,
        bisque_root=bisque_root,
        resource_uri=canonical_resource_uri,
    ):
        logger.info(
            "Skipping canonical BisQue upload reuse for %s because the current session cannot access %s",
            path,
            canonical_resource_uri,
        )
        return None
    return existing_upload


def _normalize_bisque_resource_uri(resource: str, bisque_root: str) -> str:
    """Normalize resource identifier to a data_service URL."""
    if not resource:
        return resource

    resource = str(resource).strip()

    if "resource=" in resource:
        candidate = resource.split("resource=", 1)[-1].split("&", 1)[0]
        candidate = unquote(candidate)
        if candidate:
            resource = candidate

    root = bisque_root.rstrip("/")

    if resource.startswith(root):
        resource = resource[len(root) :]

    if resource.startswith("/resource/"):
        resource = resource.replace("/resource/", "/data_service/", 1)
    if resource.startswith("/image_service/"):
        resource = resource.replace("/image_service/", "/data_service/", 1)

    if resource.startswith("http://") or resource.startswith("https://"):
        if "/image_service/" in resource:
            return resource.replace("/image_service/", "/data_service/", 1)
        return resource

    if resource.startswith("/data_service/"):
        return f"{root}{resource}"
    if resource.startswith("/image_service/"):
        return f"{root}{resource.replace('/image_service/', '/data_service/', 1)}"

    return f"{root}/data_service/{resource}"


def _build_bisque_resource_links(resource: str | None, bisque_root: str) -> dict[str, str | None]:
    """Build canonical BisQue links for UI and direct download workflows."""
    if not resource:
        return {
            "resource_uri": None,
            "resource_uniq": None,
            "image_service_url": None,
            "client_view_url": None,
        }

    normalized = _normalize_bisque_resource_uri(resource, bisque_root)
    parsed = urlparse(normalized)
    root = (
        f"{parsed.scheme}://{parsed.netloc}"
        if parsed.scheme and parsed.netloc
        else bisque_root.rstrip("/")
    )
    resource_uniq = _short_bisque_id(normalized) if normalized else None
    image_service_url: str | None
    if resource_uniq:
        image_service_url = f"{root}/image_service/{resource_uniq}"
    elif normalized:
        image_service_url = normalized.replace("/data_service/", "/image_service/", 1)
    else:
        image_service_url = None
    client_view_url = f"{root}/client_service/view?resource={normalized}" if normalized else None
    return {
        "resource_uri": normalized,
        "resource_uniq": resource_uniq,
        "image_service_url": image_service_url,
        "client_view_url": client_view_url,
    }


def _bisque_user_facing_resource_url(resource: str | None, bisque_root: str) -> str | None:
    return (
        str(
            _build_bisque_resource_links(resource, bisque_root).get("client_view_url") or ""
        ).strip()
        or None
    )


def _bisque_resource_tag_for_path(local_path: str | Path) -> str:
    path = Path(str(local_path))
    lowered_name = path.name.lower()
    image_suffixes = (
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
        ".nii",
        ".nii.gz",
        ".nrrd",
        ".mhd",
        ".mha",
        ".czi",
        ".lif",
        ".nd2",
        ".svs",
        ".dcm",
    )
    table_suffixes = (
        ".h5",
        ".hdf5",
        ".dream3d",
        ".csv",
        ".tsv",
        ".xlsx",
        ".xls",
        ".parquet",
    )
    if lowered_name.endswith(image_suffixes):
        return "image"
    if lowered_name.endswith(table_suffixes):
        return "table"
    return "file"


def _bisque_upload_resource_xml(local_path: str | Path) -> etree._Element:
    path = Path(str(local_path))
    return etree.Element(
        _bisque_resource_tag_for_path(path),
        name=path.name,
    )


def _normalize_bisque_url(url: str, bisque_root: str) -> str:
    """Normalize a BisQue URL or path using the configured root."""
    if not url:
        return url
    if url.startswith("http://") or url.startswith("https://"):
        return url
    if url.startswith("/"):
        return f"{bisque_root.rstrip('/')}{url}"
    return f"{bisque_root.rstrip('/')}/{url}"


def _is_bisque_env_placeholder(value: str | None) -> bool:
    token = str(value or "").strip()
    if not token:
        return False
    return bool(_BISQUE_ENV_PLACEHOLDER_RE.fullmatch(token))


def _sanitize_bisque_credential(value: str | None) -> str | None:
    token = str(value or "").strip()
    if not token:
        return None
    if _is_bisque_env_placeholder(token):
        return None
    return token


def _sanitize_bisque_root(value: str | None) -> str | None:
    token = str(value or "").strip().rstrip("/")
    if not token:
        return None
    if _is_bisque_env_placeholder(token):
        return None
    if token.startswith(("http://", "https://")):
        return token
    return None


_BISQUE_TAG_ALIASES = {
    # common dimension aliases -> BisQue metadata tag names
    "z": "image_num_z",
    "z_slices": "image_num_z",
    "num_z": "image_num_z",
    "depth": "image_num_z",
    "t": "image_num_t",
    "timepoints": "image_num_t",
    "num_t": "image_num_t",
    "c": "image_num_c",
    "channels": "image_num_c",
    "num_c": "image_num_c",
    "x": "image_num_x",
    "width": "image_num_x",
    "num_x": "image_num_x",
    "y": "image_num_y",
    "height": "image_num_y",
    "num_y": "image_num_y",
}


def _merge_filter_value(existing: Any, new_value: Any) -> Any:
    if existing is None:
        return new_value
    if isinstance(existing, (list, tuple)):
        merged = list(existing)
    else:
        merged = [existing]
    if isinstance(new_value, (list, tuple)):
        merged.extend(list(new_value))
    else:
        merged.append(new_value)
    return merged


def _coerce_bisque_filter_mapping(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        normalized: dict[str, Any] = {}
        for key, value in raw.items():
            name = str(key or "").strip()
            if not name or value is None or value == "":
                continue
            normalized[name] = _merge_filter_value(normalized.get(name), value)
        return normalized or None
    if not isinstance(raw, list):
        return None
    normalized: dict[str, Any] = {}
    for item in raw[:200]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        values: list[Any] = []
        raw_values = item.get("values")
        if isinstance(raw_values, list):
            values.extend(value for value in raw_values if value is not None and value != "")
        elif raw_values is not None and raw_values != "":
            values.append(raw_values)
        raw_value = item.get("value")
        if raw_value is not None and raw_value != "":
            values.append(raw_value)
        for value in values:
            normalized[name] = _merge_filter_value(normalized.get(name), value)
    return normalized or None


def _expand_bisque_tag_filters(filters: Any) -> dict[str, Any] | None:
    normalized_filters = _coerce_bisque_filter_mapping(filters)
    if not normalized_filters:
        return None
    expanded: dict[str, Any] = {}
    for key, value in normalized_filters.items():
        if value is None or value == "":
            continue
        normalized = _BISQUE_TAG_ALIASES.get(key, key)
        expanded[normalized] = _merge_filter_value(expanded.get(normalized), value)
    return expanded or None


_BISQUE_ATTR_KEYS = {
    "name",
    "resource_name",
    "type",
    "resource_user_type",
    "value",
    "resource_value",
    "hidden",
    "resource_hidden",
    "ts",
    "created",
    "unid",
    "resource_unid",
    "uniq",
    "resource_uniq",
    "mex",
    "owner",
    "owner_id",
}


def _quote_tag_value(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value)
    if s == "":
        return '""'
    if re.search(r'[\s():"\'&|<>~=]', s):
        s = s.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{s}"'
    return s


def _normalize_tag_query(tag_query: str | None) -> str | None:
    if not tag_query:
        return tag_query

    parts: list[tuple[str, bool]] = []
    buf: list[str] = []
    quote = None
    for ch in tag_query:
        if quote:
            buf.append(ch)
            if ch == quote:
                parts.append(("".join(buf), True))
                buf = []
                quote = None
        else:
            if ch in ('"', "'"):
                if buf:
                    parts.append(("".join(buf), False))
                    buf = []
                buf.append(ch)
                quote = ch
            else:
                buf.append(ch)
    if buf:
        parts.append(("".join(buf), quote is not None))

    attr_re = re.compile(
        r"(?<!@)\b(" + "|".join(map(re.escape, _BISQUE_ATTR_KEYS)) + r")\b(?=\s*:)"
    )

    def _normalize_segment(segment: str) -> str:
        segment = re.sub(
            r"(\b\w+\b)\s*(>=|<=|!=|==|>|<)\s*",
            r"\1:\2",
            segment,
        )
        segment = re.sub(r"(\b\w+\b)\s*=\s*", r"\1:", segment)
        return attr_re.sub(r"@\1", segment)

    rebuilt = []
    for segment, quoted in parts:
        rebuilt.append(segment if quoted else _normalize_segment(segment))
    return "".join(rebuilt)


def _rewrite_tag_query_aliases(tag_query: str | None) -> str | None:
    if not tag_query:
        return tag_query

    alias_pattern = re.compile(
        r"(?<![@\\w])("
        + "|".join(sorted(map(re.escape, _BISQUE_TAG_ALIASES.keys()), key=len, reverse=True))
        + r")(?=\\s*(::|:))"
    )

    parts = []
    buf = []
    quote = None
    for ch in tag_query:
        if quote:
            buf.append(ch)
            if ch == quote:
                parts.append(("".join(buf), True))
                buf = []
                quote = None
        else:
            if ch in ('"', "'"):
                if buf:
                    parts.append(("".join(buf), False))
                    buf = []
                buf.append(ch)
                quote = ch
            else:
                buf.append(ch)
    if buf:
        parts.append(("".join(buf), quote is not None))

    rewritten = []
    for segment, quoted in parts:
        if quoted:
            rewritten.append(segment)
        else:
            rewritten.append(
                alias_pattern.sub(
                    lambda m: _BISQUE_TAG_ALIASES.get(m.group(1), m.group(1)), segment
                )
            )
    return "".join(rewritten)


def _build_tag_query_compat(
    bq: Any,
    tag_query: str | None = None,
    tag_filters: dict[str, Any] | None = None,
    attr_filters: dict[str, Any] | None = None,
    text: str | None = None,
) -> str | None:
    if hasattr(bq, "build_tag_query"):
        return bq.build_tag_query(
            tag_query=tag_query,
            tag_filters=tag_filters,
            attr_filters=attr_filters,
            text=text,
        )

    parts: list[str] = []
    normalized = _normalize_tag_query(tag_query)
    if normalized:
        parts.append(normalized)

    if tag_filters:
        for key, value in tag_filters.items():
            if value is None or value == "":
                continue

            def _format_tag_value(v: Any) -> tuple[str | None, bool]:
                if v is None or v == "":
                    return None, False
                if isinstance(v, str):
                    op_match = re.match(r"^(>=|<=|!=|==|>|<)\s*(.+)$", v)
                    if op_match:
                        op, raw = op_match.groups()
                        raw = raw.strip()
                        val = _quote_tag_value(raw)
                        return (f"{key}:{op}{val}" if val else None), True
                val = _quote_tag_value(v)
                return (f"{key}:{val}" if val else None), False

            if isinstance(value, (list, tuple)):
                op_exprs: list[str] = []
                plain_vals: list[str] = []
                for v in value:
                    expr, is_op = _format_tag_value(v)
                    if not expr:
                        continue
                    if is_op:
                        op_exprs.append(expr)
                    else:
                        plain_vals.append(expr.replace(f"{key}:", "", 1))
                combined: list[str] = []
                if op_exprs:
                    combined.append("(" + " AND ".join(op_exprs) + ")")
                if plain_vals:
                    if len(plain_vals) == 1:
                        combined.append(f"{key}:{plain_vals[0]}")
                    else:
                        combined.append(f"{key}:(" + " OR ".join(plain_vals) + ")")
                if combined:
                    parts.append("(" + " AND ".join(combined) + ")")
            else:
                expr, _ = _format_tag_value(value)
                if expr:
                    parts.append(expr)

    if attr_filters:
        for key, value in attr_filters.items():
            if value is None or value == "":
                continue
            attr_key = key.lstrip("@")

            def _format_attr(v: Any) -> str | None:
                if v is None or v == "":
                    return None
                if isinstance(v, str):
                    op_match = re.match(r"^(>=|<=|!=|==|>|<)\s*(.+)$", v)
                    if op_match:
                        op, raw = op_match.groups()
                        val = _quote_tag_value(raw.strip())
                        return f"@{attr_key}:{op}{val}" if val else None
                val = _quote_tag_value(v)
                return f"@{attr_key}:{val}" if val else None

            if isinstance(value, (list, tuple)):
                exprs = [_format_attr(v) for v in value]
                exprs = [e for e in exprs if e]
                if exprs:
                    parts.append("(" + " AND ".join(exprs) + ")")
            else:
                expr = _format_attr(value)
                if expr:
                    parts.append(expr)

    if text:
        terms = list(text) if isinstance(text, (list, tuple)) else [text]
        text_exprs = []
        for term in terms:
            term = str(term).strip()
            if not term:
                continue
            val = _quote_tag_value(f"*{term}*")
            if val:
                text_exprs.append(f"@name:{val}")
                text_exprs.append(f"{val}")
        if text_exprs:
            parts.append("(" + " OR ".join(text_exprs) + ")")

    if not parts:
        return None
    return " AND ".join(parts)


def _ensure_dir(path: str | Path) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return str(path)


def _science_data_root_path() -> Path:
    settings = get_settings()
    root = str(getattr(settings, "science_data_root", "data/science") or "data/science").strip()
    return Path(root).expanduser()


def _science_output_root(*parts: str) -> str:
    root = _science_data_root_path()
    if parts:
        root = root.joinpath(*parts)
    return _ensure_dir(root)


_TOOLS_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _TOOLS_ROOT.parent
_MEGASEG_DEFAULT_BENCHMARK_ROOT = _REPO_ROOT / "data" / "models" / "megaseg" / "benchmark"
_MEGASEG_DEFAULT_CHECKPOINT = _MEGASEG_DEFAULT_BENCHMARK_ROOT / "checkpoints" / "epoch_650.ckpt"
_MEGASEG_DEFAULT_ALIAS_CHECKPOINT = (
    _MEGASEG_DEFAULT_BENCHMARK_ROOT / "checkpoints" / "megaseg" / "dynunet.ckpt"
)


def _resolve_megaseg_runner_script() -> Path:
    return (_TOOLS_ROOT / "science" / "megaseg_runner.py").resolve()


def _resolve_megaseg_python() -> str | None:
    settings = get_settings()
    active_venv = str(os.getenv("VIRTUAL_ENV") or "").strip()
    candidates = [
        str(getattr(settings, "resolved_megaseg_python", "") or "").strip(),
        str(Path(active_venv).expanduser() / "bin" / "python") if active_venv else "",
        str(Path(sys.executable).expanduser().resolve()),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.exists() and path.is_file():
            return str(path.absolute())
    return None


def _resolve_megaseg_checkpoint_path(explicit_path: str | None = None) -> str | None:
    settings = get_settings()
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(str(explicit_path)).expanduser())
    settings_checkpoint = str(
        getattr(settings, "resolved_megaseg_checkpoint_path", "") or ""
    ).strip()
    if settings_checkpoint:
        candidates.append(Path(settings_checkpoint).expanduser())
    candidates.extend(
        [
            _MEGASEG_DEFAULT_CHECKPOINT,
            _MEGASEG_DEFAULT_ALIAS_CHECKPOINT,
        ]
    )
    benchmark_root = str(getattr(settings, "resolved_megaseg_benchmark_root", "") or "").strip()
    if benchmark_root:
        benchmark_path = Path(benchmark_root).expanduser()
        candidates.extend(
            [
                benchmark_path / "checkpoints" / "epoch_650.ckpt",
                benchmark_path / "checkpoints" / "megaseg" / "dynunet.ckpt",
            ]
        )
        candidates.extend(
            sorted(
                benchmark_path.glob("checkpoints/epoch_*.ckpt"),
                key=lambda item: item.name,
                reverse=True,
            )
        )

    seen: set[str] = set()
    for candidate in candidates:
        normalized = str(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        if candidate.exists() and candidate.is_file():
            return str(candidate.resolve())
    return None


def _maybe_auto_adjust_megaseg_channels(
    *,
    file_paths: list[str],
    structure_channel: int,
    nucleus_channel: int | None,
    channel_index_base: int,
) -> tuple[int, int | None]:
    """
    Normalize Megaseg channel arguments for single-channel microscopy inputs.

    The Megaseg runner already supports ZYX volumes without a C axis by using the
    index-base channel as the structure input and omitting the nucleus channel.
    The top-level tool defaults remain tuned for Allen Cell-style multichannel
    microscopy, so we downshift them here when the inputs are explicitly
    single-channel.
    """
    normalized_paths = [str(path or "").strip() for path in file_paths if str(path or "").strip()]
    if not normalized_paths:
        return int(structure_channel), int(nucleus_channel) if nucleus_channel is not None else None

    for raw_path in normalized_paths:
        try:
            loaded = load_scientific_image(
                file_path=raw_path,
                generate_preview=False,
                save_array=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Megaseg channel preflight skipped for %s: %s", raw_path, exc)
            return int(structure_channel), int(
                nucleus_channel
            ) if nucleus_channel is not None else None

        if not bool(loaded.get("success")):
            return int(structure_channel), int(
                nucleus_channel
            ) if nucleus_channel is not None else None

        axis_sizes = loaded.get("axis_sizes") or {}
        channel_count = axis_sizes.get("C")
        is_multichannel = bool(loaded.get("is_multichannel"))
        try:
            resolved_channel_count = int(channel_count)
        except Exception:  # noqa: BLE001
            return int(structure_channel), int(
                nucleus_channel
            ) if nucleus_channel is not None else None

        if is_multichannel or resolved_channel_count > 1:
            return int(structure_channel), int(
                nucleus_channel
            ) if nucleus_channel is not None else None

    single_channel_number = int(channel_index_base)
    adjusted_structure = int(structure_channel)
    adjusted_nucleus = int(nucleus_channel) if nucleus_channel is not None else None
    changed = False

    if adjusted_structure != single_channel_number:
        adjusted_structure = single_channel_number
        changed = True
    if adjusted_nucleus is not None:
        adjusted_nucleus = None
        changed = True

    if changed:
        logger.info(
            "Megaseg auto-adjusted single-channel input to structure_channel=%s nucleus_channel=%s",
            adjusted_structure,
            adjusted_nucleus,
        )

    return adjusted_structure, adjusted_nucleus


def _models_root() -> str:
    settings = get_settings()
    explicit_root = str(os.getenv("YOLO_MODEL_ROOT") or "").strip()
    if explicit_root:
        return _ensure_dir(Path(explicit_root).expanduser())

    configured_default_model = str(os.getenv("YOLO_DEFAULT_MODEL") or "").strip()
    if configured_default_model:
        default_model_path = Path(configured_default_model).expanduser()
        if default_model_path.suffix:
            return _ensure_dir(default_model_path.parent)
        return _ensure_dir(default_model_path)

    configured_rarespot = str(getattr(settings, "prairie_rarespot_weights_path", "") or "").strip()
    if configured_rarespot:
        rarespot_path = Path(configured_rarespot).expanduser()
        if rarespot_path.suffix:
            return _ensure_dir(rarespot_path.parent)
        return _ensure_dir(rarespot_path)

    return _ensure_dir(Path("data") / "models" / "yolo")


def _finetuned_dir() -> str:
    legacy_dir = Path(_models_root()) / "finetuned"
    if legacy_dir.exists():
        return _ensure_dir(legacy_dir)
    return _ensure_dir(Path(_science_output_root("yolo", "models", "finetuned")))


def _require_ultralytics() -> Any:
    try:
        os.environ.setdefault(
            "YOLO_CONFIG_DIR", _ensure_dir(_science_data_root_path() / ".cache" / "ultralytics")
        )
        from ultralytics import YOLO  # type: ignore

        return YOLO
    except Exception as e:
        raise ImportError(
            "ultralytics is required for YOLO tools. Install with: uv pip install ultralytics"
        ) from e


def _default_yolo_pretrained_weights() -> str:
    """Return the default pretrained YOLO weights identifier/path.

    Can be overridden with `YOLO_DEFAULT_MODEL` (e.g., `yolo26x.pt`).
    """
    env = (os.getenv("YOLO_DEFAULT_MODEL") or "").strip()
    return env or "yolo26x.pt"


_PRAIRIE_YOLO_MODEL_KEY = "yolov5_rarespot"
_PRAIRIE_YOLO_MODEL_ALIASES = {
    _PRAIRIE_YOLO_MODEL_KEY,
    "prairie",
    "prairie_dog",
    "prairie-dog",
    "prairiedog",
    "burrow",
    "rarespot",
    "rarespotweights",
}
_PRAIRIE_YOLO_HINT_TOKENS = (
    "prairie dog",
    "prairie dogs",
    "prairie-dog",
    "prairie-dogs",
    "prairiedog",
    "prairiedogs",
    "burrow",
    "burrows",
    "rarespot",
)
_PRAIRIE_YOLO_PATH_HINT_TOKENS = ("prairie", "burrow", "rarespot")

_PRAIRIE_ECOLOGY_SURVEY_GUIDANCE = (
    "Treat detections as survey observations from a tiled orthomosaic, not as a full population census.",
    "Use prairie dog detections as occupancy or colony-activity evidence rather than a direct population estimate.",
    "Interpret burrows as habitat context that can support monitoring, but not as proof of current occupancy by itself.",
    "Keep image-space distances and counts separate from ground-distance or population-level inference.",
)

_PRAIRIE_ECOLOGY_CONFOUNDERS = (
    "soil texture and bare-ground patches",
    "vegetation clumps and grass cover",
    "shadows, rocks, and other dark patches",
    "small-object scale and partial occlusion",
    "border effects from cropped tiles",
)


def _pluralize_detection(count: int, singular: str, plural: str | None = None) -> str:
    normalized_plural = plural or f"{singular}s"
    return singular if int(count or 0) == 1 else normalized_plural


def _build_prairie_ecology_tile_observation(
    *,
    prairie_total: int,
    burrow_total: int,
    overall_context: dict[str, Any],
) -> str:
    if prairie_total <= 0 and burrow_total <= 0:
        return (
            "No prairie dogs or burrows were visible in this tile. "
            "On its own, this is a local nondetection for the visible scene, not evidence of absence from the broader colony or site."
        )

    if prairie_total > 0 and burrow_total > 0:
        nearest_mean = overall_context.get("nearest_burrow_distance_px_mean")
        if nearest_mean is not None:
            return (
                "Prairie dogs and burrows both appear in this tile. "
                f"The nearest detected burrow lies about {float(nearest_mean):.1f} px from each prairie dog on average, "
                "which is consistent with local co-occurrence in the visible scene."
            )
        return (
            "Prairie dogs and burrows both appear in this tile. "
            "Seeing both in the same tile is consistent with local colony activity in the visible scene."
        )

    if prairie_total > 0:
        return (
            "Prairie dogs are visible in this tile, but no burrows were detected in the current view. "
            "That supports local prairie-dog activity in the visible area, while the missing burrow boxes may reflect vegetation, shadows, or limited field of view rather than true burrow absence."
        )

    return (
        "A burrow is visible in this tile, but no prairie dogs were detected in the current view. "
        "That makes this tile evidence of burrow structure, and burrow detections alone do not establish current prairie-dog occupancy."
    )


def _build_prairie_ecology_review_note(review_flags: dict[str, Any]) -> str | None:
    flagged_reasons: list[str] = []
    if bool(review_flags.get("small_object_risk")):
        flagged_reasons.append("small-object scale")
    if bool(review_flags.get("border_effect_risk")):
        flagged_reasons.append("border effects")
    if bool(review_flags.get("burrow_context_missing")):
        flagged_reasons.append("missing burrow context")
    if bool(review_flags.get("low_confidence_review_recommended")):
        flagged_reasons.append("low-confidence detections")

    if flagged_reasons:
        if len(flagged_reasons) == 1:
            reasons_text = flagged_reasons[0]
        elif len(flagged_reasons) == 2:
            reasons_text = f"{flagged_reasons[0]} and {flagged_reasons[1]}"
        else:
            reasons_text = ", ".join(flagged_reasons[:-1]) + f", and {flagged_reasons[-1]}"
        return f"This tile warrants a closer look because it shows {reasons_text}."

    return (
        "The detection itself looks technically stable in this tile. "
        "No ecology review flags were raised for small-object scale, border effects, or low-confidence detections."
    )


def _normalize_yolo_context_text(value: Any) -> str:
    token = re.sub(r"[^a-z0-9]+", " ", str(value or "").strip().lower())
    return re.sub(r"\s+", " ", token).strip()


_PRAIRIE_YOLO_ALIAS_TOKENS = {
    _normalize_yolo_context_text(alias) for alias in _PRAIRIE_YOLO_MODEL_ALIASES
}


def _is_prairie_yolo_alias(model_name: str | None) -> bool:
    token = _normalize_yolo_context_text(model_name)
    return bool(token and token in _PRAIRIE_YOLO_ALIAS_TOKENS)


def _text_mentions_prairie_detection(value: Any) -> bool:
    normalized = _normalize_yolo_context_text(value)
    if not normalized:
        return False
    return any(hint in normalized for hint in _PRAIRIE_YOLO_HINT_TOKENS)


def _paths_suggest_prairie_detection(paths: list[str] | None) -> bool:
    for raw in paths or []:
        token = str(raw or "").strip()
        if not token:
            continue
        normalized_name = _normalize_yolo_context_text(Path(token).name)
        normalized_path = _normalize_yolo_context_text(token)
        if _text_mentions_prairie_detection(normalized_name) or _text_mentions_prairie_detection(
            normalized_path
        ):
            return True
        if any(hint in normalized_name for hint in _PRAIRIE_YOLO_PATH_HINT_TOKENS):
            return True
        if any(hint in normalized_path for hint in _PRAIRIE_YOLO_PATH_HINT_TOKENS):
            return True
    return False


def _text_requests_prediction_stability(value: Any) -> bool:
    lowered = str(value or "").strip().lower()
    if not lowered:
        return False
    return bool(
        re.search(
            r"\b("
            r"stability|unstable|uncertain|uncertainty|fragile|active learning|hard sample|"
            r"review candidate|review set|manual review|qa|quality assurance|"
            r"spectral|frequency|perturbation|confidence audit|uncertainty audit|"
            r"report|scientific report|summary report"
            r")\b",
            lowered,
        )
    )


def _resolve_prairie_builtin_weights_path() -> Path:
    settings = get_settings()
    configured = (
        str(os.getenv("YOLOV5_RARESPOT_WEIGHTS") or "").strip()
        or str(getattr(settings, "prairie_rarespot_weights_path", "") or "").strip()
        or "RareSpotWeights.pt"
    )
    return Path(configured).expanduser().resolve()


def _resolve_prairie_yolo_model_target() -> tuple[str | None, str | None, str]:
    """Resolve the preferred prairie detector weights.

    Prefer the promoted shared lineage checkpoint when available, otherwise
    fall back to the built-in RareSpot baseline path.
    """

    fallback_path = _resolve_prairie_builtin_weights_path()
    try:
        from src.orchestration.store import RunStore

        settings = get_settings()
        store = RunStore(settings.run_store_path)
        candidate_lineages: list[dict[str, Any]] = []
        default_shared = store.get_training_lineage(
            lineage_id=f"lineage-{_PRAIRIE_YOLO_MODEL_KEY}-shared"
        )
        if isinstance(default_shared, dict):
            candidate_lineages.append(default_shared)
        for row in store.list_training_lineages(
            model_key=_PRAIRIE_YOLO_MODEL_KEY,
            include_shared=True,
            limit=25,
        ):
            if not isinstance(row, dict):
                continue
            if row in candidate_lineages:
                continue
            if str(row.get("scope") or "").strip().lower() != "shared":
                continue
            candidate_lineages.append(row)

        for lineage in candidate_lineages:
            active_version_id = str(lineage.get("active_version_id") or "").strip()
            if not active_version_id:
                continue
            version_row = store.get_training_model_version(version_id=active_version_id)
            if not isinstance(version_row, dict):
                continue
            metadata = (
                version_row.get("metadata") if isinstance(version_row.get("metadata"), dict) else {}
            )
            source_job_id = str(version_row.get("source_job_id") or "").strip()
            candidate_paths: list[str] = []
            if source_job_id:
                job_row = store.get_training_job(job_id=source_job_id)
                job_result = (
                    job_row.get("result")
                    if isinstance(job_row, dict) and isinstance(job_row.get("result"), dict)
                    else {}
                )
                artifact_path = str(job_result.get("model_artifact_path") or "").strip()
                if artifact_path:
                    candidate_paths.append(artifact_path)
            metadata_path = str(metadata.get("model_artifact_path") or "").strip()
            if metadata_path:
                candidate_paths.append(metadata_path)
            for raw_path in candidate_paths:
                candidate = Path(raw_path).expanduser().resolve()
                if candidate.exists() and candidate.is_file():
                    return active_version_id, str(candidate), "active_shared_version"
    except Exception as exc:
        logger.warning("Prairie YOLO active-version lookup failed; using builtin fallback: %s", exc)

    if fallback_path.exists() and fallback_path.is_file():
        return _PRAIRIE_YOLO_MODEL_KEY, str(fallback_path), "builtin_rarespot"
    return _PRAIRIE_YOLO_MODEL_KEY, str(fallback_path), "builtin_missing"


def _default_yolo_fallback_candidates() -> list[str]:
    primary = _default_yolo_pretrained_weights()
    candidates = [
        primary,
        "yolo26x.pt",
        "yolo26l.pt",
        "yolo26m.pt",
        "yolo26n.pt",
        "yolov8x.pt",
        "yolov8n.pt",
    ]
    out: list[str] = []
    for item in candidates:
        name = str(item).strip()
        if name and name not in out:
            out.append(name)
    return out


def _finetuned_registry_path() -> Path:
    return Path(_finetuned_dir()) / "registry.jsonl"


def _read_finetuned_registry_entries(limit: int | None = None) -> list[dict[str, Any]]:
    path = _finetuned_registry_path()
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except Exception:
            continue
        if not isinstance(parsed, dict):
            continue
        model_path = parsed.get("model_path")
        if model_path and Path(str(model_path)).exists():
            entries.append(parsed)
            if limit is not None and len(entries) >= int(limit):
                break
    return entries


def _latest_finetuned_model() -> tuple[str | None, str | None]:
    entries = _read_finetuned_registry_entries(limit=1)
    if entries:
        item = entries[0]
        model_path = str(item.get("model_path") or "").strip()
        model_name = str(item.get("model_name") or "").strip() or Path(model_path).stem
        if model_path and Path(model_path).exists():
            return model_name, model_path

    candidates = sorted(
        Path(_finetuned_dir()).glob("*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        latest = candidates[0]
        return latest.stem, str(latest)
    return None, None


def _checkpoint_file_signature(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": None,
        "looks_html_or_xml": False,
        "hint": None,
    }
    if not path.exists() or not path.is_file():
        result["hint"] = "File does not exist."
        return result

    try:
        size = int(path.stat().st_size)
        result["size_bytes"] = size
        with path.open("rb") as handle:
            head = handle.read(512)
    except Exception as exc:  # noqa: BLE001
        result["hint"] = f"Unable to read file: {exc}"
        return result

    stripped = head.lstrip()
    if stripped.startswith(b"<"):
        preview = stripped[:120].decode("utf-8", errors="ignore").lower()
        html_markers = ("<html", "<!doctype", "<head", "<body", "<resource", "<error", "<?xml")
        if any(marker in preview for marker in html_markers):
            result["looks_html_or_xml"] = True
            result["hint"] = "File appears to be HTML/XML instead of a PyTorch checkpoint."
            return result

    if result["size_bytes"] is not None and int(result["size_bytes"]) <= 1024:
        result["hint"] = "Checkpoint file is unusually small."
    return result


def _looks_like_legacy_yolov5_checkpoint_error(message: str) -> bool:
    normalized = str(message or "").strip().lower()
    if not normalized:
        return False
    return (
        "appears to be an ultralytics yolov5 model originally trained with" in normalized
        or "not forwards compatible with yolov8" in normalized
        or "run a command with an official ultralytics model" in normalized
    )


def _run_legacy_yolov5_detection(
    *,
    model_artifact_path: str,
    input_paths: list[str],
    output_dir: Path,
    conf: float,
    iou: float,
) -> dict[str, Any]:
    from src.training.adapters import YOLOv5Adapter

    adapter = YOLOv5Adapter()
    return adapter.infer(
        model_artifact_path=str(model_artifact_path),
        input_paths=list(input_paths),
        output_dir=output_dir.resolve(),
        config={
            "enable_sahi": False,
            "tile_size": 512,
            "tile_overlap": 0.25,
            "conf": float(conf),
            "iou": float(iou),
            "merge_iou": float(iou),
            "imgsz": 512,
            "min_box_pixels": 4.0,
        },
        progress_callback=lambda _payload: None,
        control_callback=lambda: None,
    )


def _image_metadata_context(file_path: str, *, output_root: str | None = None) -> dict[str, Any]:
    try:
        loaded = load_scientific_image(
            file_path=str(file_path),
            array_mode="plane",
            generate_preview=False,
            save_array=False,
            include_array=False,
            output_root=output_root,
        )
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "error": str(exc)}
    if not isinstance(loaded, dict) or not bool(loaded.get("success")):
        return {
            "success": False,
            "error": str((loaded or {}).get("error") or "metadata_unavailable"),
        }
    axis_sizes = loaded.get("axis_sizes") if isinstance(loaded.get("axis_sizes"), dict) else {}
    metadata = loaded.get("metadata") if isinstance(loaded.get("metadata"), dict) else {}
    dimensions = {
        axis: int(value)
        for axis, value in axis_sizes.items()
        if str(axis).strip() and isinstance(value, (int, float))
    }
    summary = {
        "success": True,
        "reader": str(loaded.get("reader") or "").strip() or None,
        "dims_order": str(loaded.get("dims_order") or "").strip() or None,
        "array_shape": list(loaded.get("array_shape") or [])
        if isinstance(loaded.get("array_shape"), list)
        else [],
        "dimensions": dimensions,
        "header": dict(metadata.get("header") or {})
        if isinstance(metadata.get("header"), dict)
        else {},
        "exif": dict(metadata.get("exif") or {}) if isinstance(metadata.get("exif"), dict) else {},
        "geo": dict(metadata.get("geo") or {}) if isinstance(metadata.get("geo"), dict) else {},
        "filename_hints": (
            dict(metadata.get("filename_hints") or {})
            if isinstance(metadata.get("filename_hints"), dict)
            else {}
        ),
    }
    captured_at = (
        str(
            summary["exif"].get("DateTimeOriginal") or summary["exif"].get("DateTime") or ""
        ).strip()
        if isinstance(summary.get("exif"), dict)
        else ""
    )
    if captured_at:
        summary["captured_at"] = captured_at
    return summary


_YOLO_MUTED_CLASS_COLORS: dict[str, str] = {
    # Use a neon-but-readable palette for class labels while keeping box strokes
    # black. The label colors stay stable across runs so scientists can quickly
    # learn class-specific accents without depending on the prose.
    "prairie_dog": "#FF71CE",
    "burrow": "#01CDFE",
    "dog": "#FF71CE",
    "animal": "#FF71CE",
    "person": "#05FFA1",
    "cell": "#B967FF",
    "nucleus": "#FFD166",
    "background": "#7A748D",
}
_YOLO_MUTED_PALETTE: list[str] = [
    "#FF71CE",
    "#01CDFE",
    "#B967FF",
    "#05FFA1",
    "#FFD166",
    "#7A5CFA",
    "#FFB86C",
    "#7A748D",
]


def _yolo_class_color(class_name: str) -> str:
    normalized = str(class_name or "").strip().lower()
    if normalized in _YOLO_MUTED_CLASS_COLORS:
        return _YOLO_MUTED_CLASS_COLORS[normalized]
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
    return _YOLO_MUTED_PALETTE[int(digest, 16) % len(_YOLO_MUTED_PALETTE)]


def _preferred_yolo_label_font() -> str:
    candidates = [
        "JetBrains Mono",
        "JetBrainsMono Nerd Font",
        "JetBrainsMonoNL Nerd Font",
        "DejaVu Sans Mono",
        "Menlo",
        "monospace",
    ]
    try:
        from matplotlib import font_manager

        available = {
            str(font.name).strip().lower()
            for font in font_manager.fontManager.ttflist
            if str(font.name or "").strip()
        }
        for name in candidates:
            if name.lower() in available:
                return name
    except Exception:
        pass
    return candidates[0]


def _load_yolo_source_image(path: str) -> tuple[np.ndarray, int, int]:
    source_path = Path(str(path)).expanduser()
    with Image.open(source_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        array = np.asarray(image)
        width, height = image.size
    return array, int(width), int(height)


def _render_yolo_detection_figure(
    *,
    source_path: str,
    boxes: list[dict[str, Any]],
    output_path: Path,
) -> dict[str, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        image_array, width, height = _load_yolo_source_image(source_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dpi = 100
        figure = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, frameon=False)
        axes = figure.add_axes([0, 0, 1, 1])
        axes.imshow(image_array, interpolation="nearest", zorder=0)
        axes.set_xlim(0, width)
        axes.set_ylim(height, 0)
        axes.axis("off")
        label_font = _preferred_yolo_label_font()

        for box in boxes:
            if not isinstance(box, dict):
                continue
            xyxy = box.get("xyxy") if isinstance(box.get("xyxy"), list) else []
            if len(xyxy) < 4:
                continue
            try:
                x1, y1, x2, y2 = [float(value) for value in xyxy[:4]]
            except Exception:
                continue
            x1 = max(0.0, min(float(width), x1))
            y1 = max(0.0, min(float(height), y1))
            x2 = max(0.0, min(float(width), x2))
            y2 = max(0.0, min(float(height), y2))
            if x2 <= x1 or y2 <= y1:
                continue
            class_name = str(box.get("class_name") or "").strip() or "det"
            confidence = box.get("confidence")
            color = _yolo_class_color(class_name)
            axes.add_patch(
                Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=3.4,
                    edgecolor="#050505",
                    facecolor="none",
                    joinstyle="miter",
                    zorder=1,
                )
            )
            axes.add_patch(
                Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=1.0,
                    edgecolor="#1B1B1B",
                    facecolor="none",
                    alpha=0.95,
                    joinstyle="miter",
                    zorder=2,
                )
            )
            label = class_name.upper()
            try:
                if confidence is not None:
                    label = f"{label} {float(confidence) * 100:.0f}%"
            except Exception:
                pass
            label_y = y1 - 6 if y1 > 20 else y1 + 7
            label_vertical_alignment = "bottom" if y1 > 20 else "top"
            axes.text(
                x1 + 5,
                label_y,
                label,
                color=color,
                fontsize=9.5,
                fontweight="bold",
                fontfamily=label_font,
                va=label_vertical_alignment,
                ha="left",
                bbox={
                    "boxstyle": "square,pad=0.28",
                    "facecolor": "#050505",
                    "edgecolor": color,
                    "linewidth": 1.35,
                    "alpha": 0.98,
                },
                zorder=3,
                clip_on=True,
            )

        figure.savefig(
            output_path,
            dpi=dpi,
            facecolor="white",
            edgecolor="white",
            bbox_inches=None,
            pad_inches=0,
        )
        plt.close(figure)
        return {
            "success": True,
            "preview_path": str(output_path),
            "image_width": width,
            "image_height": height,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "success": False,
            "error": str(exc),
        }


def _build_prairie_ecology_context(
    *,
    counts_by_class: dict[str, int],
    spatial_analysis: dict[str, Any],
    inference_configuration: dict[str, Any],
    metadata_summary: dict[str, Any],
    prairie_model_requested: bool,
    finetune_recommended: bool,
) -> dict[str, Any] | None:
    prairie_total = int(counts_by_class.get("prairie_dog") or 0)
    burrow_total = int(counts_by_class.get("burrow") or 0)
    overall_context = (
        spatial_analysis.get("overall_prairie_burrow_context")
        if isinstance(spatial_analysis.get("overall_prairie_burrow_context"), dict)
        else {}
    )
    if not prairie_model_requested and prairie_total == 0 and burrow_total == 0:
        return None

    deployment_context = {
        "survey_platform": "fixed-wing drone",
        "survey_altitude_m": 100,
        "ground_sample_distance_cm_per_px": 2,
        "orthomosaic_overlap_percent": 70,
        "tile_size_px": int(inference_configuration.get("tile_size") or 512),
        "tile_overlap_ratio": float(inference_configuration.get("tile_overlap") or 0.25),
        "annotation_classes": ["prairie_dog", "burrow"],
        "inference_backend": str(inference_configuration.get("backend") or "").strip() or None,
        "tile_strategy": str(inference_configuration.get("tile_strategy") or "").strip() or None,
    }
    deployment_context = {
        key: value for key, value in deployment_context.items() if value is not None
    }

    image_size_values = [
        int(image.get("image_size", {}).get("width") or 0)
        for image in spatial_analysis.get("images", [])
        if isinstance(image, dict) and isinstance(image.get("image_size"), dict)
    ]
    image_height_values = [
        int(image.get("image_size", {}).get("height") or 0)
        for image in spatial_analysis.get("images", [])
        if isinstance(image, dict) and isinstance(image.get("image_size"), dict)
    ]
    image_width = max(image_size_values) if image_size_values else None
    image_height = max(image_height_values) if image_height_values else None
    border_touching_count = int(overall_context.get("border_touching_box_count") or 0)
    small_object_risk = bool(
        overall_context.get("prairie_dog_short_side_px_min") is not None
        and float(overall_context.get("prairie_dog_short_side_px_min") or 0.0) <= 30.0
    )
    burrow_context_missing = bool(prairie_total > 0 and burrow_total == 0)
    burrow_overlap_present = bool(
        int(overall_context.get("prairie_dogs_overlapping_burrows") or 0) > 0
    )
    border_effect_risk = bool(border_touching_count > 0)

    review_flags = {
        "manual_review_recommended": bool(
            finetune_recommended
            or small_object_risk
            or burrow_context_missing
            or border_effect_risk
        ),
        "small_object_risk": small_object_risk,
        "border_effect_risk": border_effect_risk,
        "burrow_context_missing": burrow_context_missing,
        "burrow_overlap_present": burrow_overlap_present,
        "low_confidence_review_recommended": bool(finetune_recommended),
        "georeferenced_context_present": bool(
            metadata_summary.get("first_latitude") is not None
            and metadata_summary.get("first_longitude") is not None
        ),
    }

    return {
        "study_focus": "prairie dog and burrow wildlife survey",
        "paper_context": {
            "model_family": "RareSpot / YOLOv5",
            "training_strategies": [
                "multi-scale consistency learning",
                "context-aware hard sample augmentation",
            ],
            "dataset_structure": [
                "expert-annotated prairie_dog and burrow bounding boxes",
                "flight-separated train/validation/test splits",
                "orthomosaic tiles from fixed-wing drone surveys",
            ],
        },
        "deployment_context": deployment_context,
        "survey_interpretation_guidance": list(_PRAIRIE_ECOLOGY_SURVEY_GUIDANCE),
        "common_confounders": list(_PRAIRIE_ECOLOGY_CONFOUNDERS),
        "review_flags": review_flags,
        "image_context": {
            "image_width_px": image_width,
            "image_height_px": image_height,
            "prairie_dog_count": prairie_total,
            "burrow_count": burrow_total,
            "nearest_burrow_distance_px_mean": overall_context.get(
                "nearest_burrow_distance_px_mean"
            ),
            "nearest_burrow_distance_px_min": overall_context.get("nearest_burrow_distance_px_min"),
        },
    }


def _build_prairie_ecology_result_message(
    *,
    counts_by_class: dict[str, int],
    ecology_context: dict[str, Any],
    spatial_analysis: dict[str, Any],
    finetune_recommended: bool,
) -> str:
    prairie_total = int(counts_by_class.get("prairie_dog") or 0)
    burrow_total = int(counts_by_class.get("burrow") or 0)
    review_flags = (
        dict(ecology_context.get("review_flags") or {})
        if isinstance(ecology_context.get("review_flags"), dict)
        else {}
    )
    overall_context = (
        dict(spatial_analysis.get("overall_prairie_burrow_context") or {})
        if isinstance(spatial_analysis.get("overall_prairie_burrow_context"), dict)
        else {}
    )
    lines: list[str] = []
    lines.append(
        _build_prairie_ecology_tile_observation(
            prairie_total=prairie_total,
            burrow_total=burrow_total,
            overall_context=overall_context,
        )
    )
    review_note = _build_prairie_ecology_review_note(review_flags)
    if review_note:
        lines.append(review_note)

    return " ".join(line.strip() for line in lines if str(line or "").strip()).strip()


def _detection_spatial_analysis(
    *,
    predictions: list[dict[str, Any]],
    metadata_context_by_path: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], str]:
    metadata_context_by_path = dict(metadata_context_by_path or {})
    per_image: list[dict[str, Any]] = []
    total_boxes = 0
    prairie_total = 0
    burrow_total = 0
    all_nearest_distances: list[float] = []
    all_overlap_count = 0
    capture_values: list[str] = []
    geo_values: list[tuple[float, float]] = []
    all_prairie_short_sides: list[float] = []
    all_burrow_short_sides: list[float] = []
    all_border_touching_count = 0

    def _xyxy_edge_distance(a: list[float], b: list[float]) -> float:
        ax1, ay1, ax2, ay2 = [float(value) for value in a[:4]]
        bx1, by1, bx2, by2 = [float(value) for value in b[:4]]
        dx = max(0.0, max(bx1 - ax2, ax1 - bx2))
        dy = max(0.0, max(by1 - ay2, ay1 - by2))
        return float((dx * dx + dy * dy) ** 0.5)

    for prediction in predictions:
        if not isinstance(prediction, dict):
            continue
        path = str(prediction.get("path") or prediction.get("input_path") or "").strip()
        boxes = prediction.get("boxes") if isinstance(prediction.get("boxes"), list) else []
        class_counts = (
            dict(prediction.get("class_counts") or {})
            if isinstance(prediction.get("class_counts"), dict)
            else {}
        )
        width = None
        height = None
        metadata_context = metadata_context_by_path.get(path) if path else None
        dimensions = (
            dict((metadata_context or {}).get("dimensions") or {})
            if isinstance((metadata_context or {}).get("dimensions"), dict)
            else {}
        )
        try:
            width = float(dimensions.get("X") or 0.0) or None
        except Exception:
            width = None
        try:
            height = float(dimensions.get("Y") or 0.0) or None
        except Exception:
            height = None
        if width is None or height is None:
            try:
                with Image.open(path) as image:
                    width, height = image.size
            except Exception:
                width, height = None, None

        dog_boxes: list[dict[str, Any]] = []
        burrow_boxes: list[dict[str, Any]] = []
        areas: list[float] = []
        image_border_touching_count = 0
        for box in boxes:
            if not isinstance(box, dict):
                continue
            xyxy = box.get("xyxy") if isinstance(box.get("xyxy"), list) else []
            if len(xyxy) < 4:
                continue
            x1, y1, x2, y2 = [float(value) for value in xyxy[:4]]
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            short_side = max(0.0, min(x2 - x1, y2 - y1))
            areas.append(area)
            class_name = str(box.get("class_name") or "").strip()
            normalized_box = {
                "class_name": class_name,
                "xyxy": [x1, y1, x2, y2],
                "area_px": area,
                "short_side_px": short_side,
            }
            if class_name == "prairie_dog":
                dog_boxes.append(normalized_box)
                all_prairie_short_sides.append(short_side)
            elif class_name == "burrow":
                burrow_boxes.append(normalized_box)
                all_burrow_short_sides.append(short_side)
            if width and height:
                if x1 <= 2.0 or y1 <= 2.0 or x2 >= float(width) - 2.0 or y2 >= float(height) - 2.0:
                    image_border_touching_count += 1

        prairie_total += len(dog_boxes)
        burrow_total += len(burrow_boxes)
        total_boxes += len(boxes)
        all_border_touching_count += image_border_touching_count

        image_summary: dict[str, Any] = {
            "path": path,
            "source_path": path,
            "source_name": Path(path).name if path else None,
            "class_counts": class_counts,
            "box_count": len(boxes),
        }
        if width and height:
            image_summary["image_size"] = {"width": int(width), "height": int(height)}
        if areas:
            image_summary["box_area_px_mean"] = round(sum(areas) / len(areas), 3)
        if dog_boxes and width and height:
            dog_area_mean = sum(float(box["area_px"]) for box in dog_boxes) / len(dog_boxes)
            image_summary["prairie_dog_area_fraction_mean"] = round(
                dog_area_mean / max(1.0, float(width) * float(height)),
                6,
            )
            image_summary["prairie_dog_short_side_px_min"] = round(
                min(float(box["short_side_px"]) for box in dog_boxes),
                3,
            )
            image_summary["prairie_dog_short_side_px_mean"] = round(
                sum(float(box["short_side_px"]) for box in dog_boxes) / len(dog_boxes),
                3,
            )
        if burrow_boxes and width and height:
            image_summary["burrow_short_side_px_min"] = round(
                min(float(box["short_side_px"]) for box in burrow_boxes),
                3,
            )
            image_summary["burrow_short_side_px_mean"] = round(
                sum(float(box["short_side_px"]) for box in burrow_boxes) / len(burrow_boxes),
                3,
            )
        if dog_boxes:
            overlap_count = 0
            nearest_records: list[dict[str, Any]] = []
            for dog_box in dog_boxes:
                if burrow_boxes:
                    nearest: dict[str, Any] | None = None
                    for burrow_box in burrow_boxes:
                        distance = _xyxy_edge_distance(dog_box["xyxy"], burrow_box["xyxy"])
                        candidate = {
                            "distance_px": float(distance),
                            "dog_xyxy": [float(value) for value in dog_box["xyxy"]],
                            "burrow_xyxy": [float(value) for value in burrow_box["xyxy"]],
                        }
                        if nearest is None or float(candidate["distance_px"]) < float(
                            nearest["distance_px"]
                        ):
                            nearest = candidate
                    if nearest is not None:
                        if float(nearest["distance_px"]) <= 0.0:
                            overlap_count += 1
                        nearest_records.append(nearest)
                        all_nearest_distances.append(float(nearest["distance_px"]))
            image_summary["prairie_burrow_context"] = {
                "prairie_dog_count": len(dog_boxes),
                "burrow_count": len(burrow_boxes),
                "prairie_dogs_overlapping_burrows": overlap_count,
                "prairie_dogs_without_burrow_overlap": max(0, len(dog_boxes) - overlap_count),
                "prairie_dog_short_side_px_min": (
                    round(min(float(box["short_side_px"]) for box in dog_boxes), 3)
                    if dog_boxes
                    else None
                ),
                "prairie_dog_short_side_px_mean": (
                    round(
                        sum(float(box["short_side_px"]) for box in dog_boxes) / len(dog_boxes),
                        3,
                    )
                    if dog_boxes
                    else None
                ),
                "burrow_short_side_px_min": (
                    round(min(float(box["short_side_px"]) for box in burrow_boxes), 3)
                    if burrow_boxes
                    else None
                ),
                "burrow_short_side_px_mean": (
                    round(
                        sum(float(box["short_side_px"]) for box in burrow_boxes)
                        / len(burrow_boxes),
                        3,
                    )
                    if burrow_boxes
                    else None
                ),
                "nearest_burrow_distance_px_min": (
                    round(min(float(row["distance_px"]) for row in nearest_records), 3)
                    if nearest_records
                    else None
                ),
                "nearest_burrow_distance_px_mean": (
                    round(
                        sum(float(row["distance_px"]) for row in nearest_records)
                        / len(nearest_records),
                        3,
                    )
                    if nearest_records
                    else None
                ),
                "nearest_burrow_distance_px_median": (
                    round(
                        statistics.median(float(row["distance_px"]) for row in nearest_records), 3
                    )
                    if nearest_records
                    else None
                ),
                "nearest_burrow_distance_px_max": (
                    round(max(float(row["distance_px"]) for row in nearest_records), 3)
                    if nearest_records
                    else None
                ),
                "nearest_burrow_distance_px_by_prairie_dog": nearest_records,
                "distance_metric": "edge_to_edge_px",
                "border_touching_box_count": image_border_touching_count,
            }
            all_overlap_count += overlap_count
        metadata_geo = (
            dict((metadata_context or {}).get("geo") or {})
            if isinstance((metadata_context or {}).get("geo"), dict)
            else {}
        )
        if metadata_geo:
            image_summary["geo"] = metadata_geo
            try:
                latitude = float(metadata_geo.get("latitude"))
                longitude = float(metadata_geo.get("longitude"))
            except Exception:
                latitude = None
                longitude = None
            if latitude is not None and longitude is not None:
                geo_values.append((latitude, longitude))
        captured_at = str((metadata_context or {}).get("captured_at") or "").strip()
        if captured_at:
            image_summary["captured_at"] = captured_at
            capture_values.append(captured_at)
        per_image.append(image_summary)

    overall_prairie_burrow_context: dict[str, Any] | None = None
    if prairie_total and burrow_total and all_nearest_distances:
        overall_prairie_burrow_context = {
            "prairie_dog_count": prairie_total,
            "burrow_count": burrow_total,
            "prairie_dogs_overlapping_burrows": all_overlap_count,
            "prairie_dogs_without_burrow_overlap": max(0, prairie_total - all_overlap_count),
            "prairie_dog_short_side_px_min": (
                round(min(all_prairie_short_sides), 3) if all_prairie_short_sides else None
            ),
            "prairie_dog_short_side_px_mean": (
                round(sum(all_prairie_short_sides) / len(all_prairie_short_sides), 3)
                if all_prairie_short_sides
                else None
            ),
            "burrow_short_side_px_min": (
                round(min(all_burrow_short_sides), 3) if all_burrow_short_sides else None
            ),
            "burrow_short_side_px_mean": (
                round(sum(all_burrow_short_sides) / len(all_burrow_short_sides), 3)
                if all_burrow_short_sides
                else None
            ),
            "nearest_burrow_distance_px_min": round(min(all_nearest_distances), 3),
            "nearest_burrow_distance_px_mean": round(
                sum(all_nearest_distances) / len(all_nearest_distances),
                3,
            ),
            "nearest_burrow_distance_px_median": round(
                statistics.median(all_nearest_distances),
                3,
            ),
            "nearest_burrow_distance_px_max": round(max(all_nearest_distances), 3),
            "distance_metric": "edge_to_edge_px",
            "border_touching_box_count": all_border_touching_count,
        }

    metadata_summary: dict[str, Any] = {}
    if geo_values:
        latitude, longitude = geo_values[0]
        metadata_summary.update(
            {
                "geo_image_count": len(geo_values),
                "first_latitude": round(latitude, 6),
                "first_longitude": round(longitude, 6),
            }
        )
    if capture_values:
        metadata_summary.update(
            {
                "capture_count": len(capture_values),
                "first_captured_at": capture_values[0],
            }
        )

    summary_parts: list[str] = []
    if total_boxes > 0:
        summary_parts.append(f"Detected {total_boxes} box(es)")
        if prairie_total:
            summary_parts.append(f"{prairie_total} prairie_dog")
        if burrow_total:
            summary_parts.append(f"{burrow_total} burrow")
        elif prairie_total:
            summary_parts.append("no burrows")
    else:
        summary_parts.append("No detections were produced")
    if overall_prairie_burrow_context:
        summary_parts.append(
            "Prairie-burrow proximity "
            f"mean {overall_prairie_burrow_context['nearest_burrow_distance_px_mean']} px, "
            f"median {overall_prairie_burrow_context['nearest_burrow_distance_px_median']} px, "
            f"min {overall_prairie_burrow_context['nearest_burrow_distance_px_min']} px."
        )
        if all_overlap_count:
            summary_parts.append(f"{all_overlap_count} prairie dog(s) overlap a burrow box.")
    if (
        metadata_summary.get("first_latitude") is not None
        and metadata_summary.get("first_longitude") is not None
    ):
        summary_parts.append(
            "GPS context "
            f"{metadata_summary['first_latitude']}, {metadata_summary['first_longitude']}"
        )
    if metadata_summary.get("first_captured_at"):
        summary_parts.append(f"captured at {metadata_summary['first_captured_at']}")
    result = {
        "images": per_image,
        "total_boxes": total_boxes,
        "prairie_dog_total": prairie_total,
        "burrow_total": burrow_total,
    }
    if overall_prairie_burrow_context:
        result["overall_prairie_burrow_context"] = overall_prairie_burrow_context
    if metadata_summary:
        result["metadata_summary"] = metadata_summary
    return (result, ". ".join(summary_parts).strip() + ".")


def _resolve_model_path(model_name: str | None = None, model_path: str | None = None) -> str:
    if model_path:
        return model_path
    if model_name:
        if _is_prairie_yolo_alias(model_name):
            _resolved_name, prairie_path, _resolution = _resolve_prairie_yolo_model_target()
            if prairie_path:
                return prairie_path
        normalized = str(model_name).strip().lower()
        if normalized in {"latest", "latest_finetuned", "newest", "recent"}:
            _, latest_path = _latest_finetuned_model()
            if latest_path:
                return latest_path
        candidate = Path(_finetuned_dir()) / f"{_safe_slug(model_name)}.pt"
        if candidate.exists():
            return str(candidate)
        model_name_path = Path(str(model_name))
        if model_name_path.suffix.lower() == ".pt":
            candidate_by_name = Path(_finetuned_dir()) / model_name_path.name
            if candidate_by_name.exists():
                return str(candidate_by_name)
    return _default_yolo_pretrained_weights()


_YOLO_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
_YOLO_LABEL_EXTS = {".txt"}
_UPLOADED_HASH_PREFIX_RE = re.compile(r"^(?P<prefix>[0-9a-f]{8,})__(?P<base>.+)$", re.IGNORECASE)
_SEQUENCE_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpg", ".mpeg"}
_SEGMENTATION_IMAGE_EXTS = _YOLO_IMAGE_EXTS.union(
    {
        ".czi",
        ".nd2",
        ".lif",
        ".lsm",
        ".svs",
        ".vsi",
        ".dv",
        ".r3d",
        ".ome.tif",
        ".ome.tiff",
        ".zarr",
        ".nii",
        ".nii.gz",
        ".nrrd",
        ".mha",
        ".mhd",
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".webm",
        ".mpg",
        ".mpeg",
    }
)

_REMOTE_INPUT_SCHEMES = {"http", "https", "s3"}

_DEFAULT_STEM_STRIP_TOKENS = (
    "input",
    "target",
    "label",
    "labels",
    "mask",
    "pred",
    "prediction",
    "gt",
    "groundtruth",
    "ground_truth",
    "sam3",
    "sam2",
    "medsam2",
    "megaseg",
    "dynunet",
)

_SEGMENTATION_RESULT_CACHE: dict[str, dict[str, Any]] = {}
_DEPTH_PRO_RESULT_CACHE: dict[str, dict[str, Any]] = {}
_DEPTH_PRO_RUNTIME_CACHE: dict[tuple[str, bool, str], tuple[Any, Any]] = {}


def _yolo_datasets_root() -> str:
    return _science_output_root("yolo", "datasets")


def _yolo_training_root() -> str:
    return _science_output_root("yolo", "training")


def _sequence_frames_root() -> str:
    return _science_output_root("sequence_frames")


def _is_sequence_media(path: Path) -> bool:
    return path.suffix.lower() in _SEQUENCE_VIDEO_EXTS


def _extract_sequence_frames(
    source_path: Path,
    *,
    max_frames: int,
    frame_stride: int,
) -> tuple[list[Path], str | None]:
    if not source_path.exists() or not source_path.is_file():
        return [], f"Sequence file not found: {source_path}"

    try:
        import cv2  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return [], f"opencv-python is required for sequence extraction: {exc}"

    max_frames = max(1, int(max_frames))
    frame_stride = max(1, int(frame_stride))
    signature = _path_signature(source_path)
    cache_payload = json.dumps(
        {
            "source": signature,
            "max_frames": max_frames,
            "frame_stride": frame_stride,
        },
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha256(cache_payload.encode("utf-8")).hexdigest()[:14]
    frame_dir = Path(_sequence_frames_root()) / f"{_safe_slug(source_path.stem)}_{digest}"
    manifest_path = frame_dir / "manifest.json"

    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
        cached_frame_paths = manifest.get("frame_paths")
        if isinstance(cached_frame_paths, list):
            cached = [Path(str(item)) for item in cached_frame_paths if str(item).strip()]
            if cached and all(item.exists() for item in cached):
                return cached, None

    frame_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        return [], f"Failed to open sequence file: {source_path.name}"

    saved_paths: list[Path] = []
    frame_index = 0
    try:
        while len(saved_paths) < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % frame_stride == 0:
                out_path = frame_dir / f"{_safe_slug(source_path.stem)}_frame_{frame_index:06d}.png"
                if not cv2.imwrite(str(out_path), frame):
                    return [], f"Failed to write extracted frame for {source_path.name}"
                saved_paths.append(out_path)
            frame_index += 1
    finally:
        capture.release()

    if not saved_paths:
        return [], f"No frames decoded from {source_path.name}"

    manifest = {
        "source_path": str(source_path.resolve()),
        "source_signature": signature,
        "max_frames": max_frames,
        "frame_stride": frame_stride,
        "frame_paths": [str(item.resolve()) for item in saved_paths],
    }
    try:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    except Exception:
        pass
    return saved_paths, None


def _expand_sequence_inputs_for_2d_models(
    paths: list[Path],
    *,
    max_frames_per_sequence: int | None = None,
    frame_stride: int | None = None,
) -> tuple[list[Path], list[dict[str, Any]], list[str]]:
    settings = get_settings()
    max_frames = int(
        max_frames_per_sequence
        if max_frames_per_sequence is not None
        else getattr(settings, "sequence_max_frames_per_file", 24)
    )
    stride = int(
        frame_stride if frame_stride is not None else getattr(settings, "sequence_frame_stride", 4)
    )
    max_frames = max(1, max_frames)
    stride = max(1, stride)

    expanded: list[Path] = []
    expansions: list[dict[str, Any]] = []
    warnings: list[str] = []

    for raw_path in paths:
        path = Path(str(raw_path)).expanduser()
        if not _is_sequence_media(path):
            expanded.append(path)
            continue
        frames, error = _extract_sequence_frames(
            path,
            max_frames=max_frames,
            frame_stride=stride,
        )
        if error:
            warnings.append(error)
            continue
        expanded.extend(frames)
        expansions.append(
            {
                "source_path": str(path),
                "frame_count": len(frames),
                "max_frames": max_frames,
                "frame_stride": stride,
                "frame_paths": [str(item) for item in frames],
            }
        )

    deduped: list[Path] = []
    seen: set[str] = set()
    for item in expanded:
        key = str(item)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped, expansions, warnings


def _is_yolo_image(path: Path) -> bool:
    if path.suffix.lower() in _YOLO_IMAGE_EXTS:
        return True
    if not path.exists() or not path.is_file():
        return False
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def _is_yolo_label(path: Path) -> bool:
    return path.suffix.lower() in _YOLO_LABEL_EXTS


def _looks_like_remote_input_path(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    raw = value.strip()
    if not raw:
        return False
    parsed = urlparse(raw)
    return bool(parsed.scheme and parsed.netloc and parsed.scheme.lower() in _REMOTE_INPUT_SCHEMES)


def _input_path_name(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if _looks_like_remote_input_path(raw):
        parsed = urlparse(raw)
        name = Path(unquote(parsed.path.rstrip("/"))).name
        if name:
            return name
        return unquote(parsed.netloc)
    return Path(raw).expanduser().name


def _is_directory_image_store(path: Path) -> bool:
    lower = path.name.lower()
    return path.is_dir() and (lower.endswith(".ome.zarr") or lower.endswith(".zarr"))


def _looks_like_segmentation_remote_path(value: str) -> bool:
    lower = _input_path_name(value).lower()
    if not lower or lower.endswith((".pdf", ".txt")):
        return False
    if lower.endswith(".ome.zarr") or lower.endswith(".zarr"):
        return True
    if lower.endswith(".ome.tif") or lower.endswith(".ome.tiff") or lower.endswith(".nii.gz"):
        return True
    return Path(lower).suffix.lower() in _SEGMENTATION_IMAGE_EXTS


_PROMPT_PATHISH_TOKEN_RE = re.compile(
    r"(?P<token>(?:s3|https?)://[^\s<>'\"`]+|(?:~|/|\./|\.\./)[^\s<>'\"`]+)",
    re.IGNORECASE,
)
_PROMPT_IMAGE_TEXT_SUFFIXES: tuple[str, ...] = (
    ".ome.zarr",
    ".ome.tiff",
    ".ome.tif",
    ".nii.gz",
    ".zarr",
    ".nrrd",
    ".mhd",
    ".mha",
    ".nd2",
    ".czi",
    ".svs",
    ".dicom",
    ".dcm",
    ".tiff",
    ".tif",
    ".nii",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".gif",
    ".webp",
)


def _clean_prompt_pathish_token(value: Any) -> str:
    token = str(value or "").strip()
    if not token:
        return ""
    token = token.strip("`")
    while token and token[0] in "\"'([{<":
        token = token[1:].strip()
    while token and token[-1] in "\"'`.,;:!?)]}>":
        token = token[:-1].rstrip()
    return token


def _image_text_suffix_target(value: str) -> str:
    token = _clean_prompt_pathish_token(value)
    if not token:
        return ""
    if _looks_like_remote_input_path(token):
        parsed = urlparse(token)
        return unquote(parsed.path).rstrip("/").lower()
    return token.rstrip("/").lower()


def _looks_like_image_path_text(value: Any) -> bool:
    target = _image_text_suffix_target(str(value or ""))
    if not target or target.endswith((".pdf", ".txt")):
        return False
    return any(target.endswith(suffix) for suffix in _PROMPT_IMAGE_TEXT_SUFFIXES)


def extract_scientific_image_paths_from_text(user_text: str | None) -> list[str]:
    text = str(user_text or "")
    if not text:
        return []
    ordered: list[str] = []
    seen: set[str] = set()
    for match in _PROMPT_PATHISH_TOKEN_RE.finditer(text):
        token = _clean_prompt_pathish_token(match.group("token"))
        if not token or token in seen or not _looks_like_image_path_text(token):
            continue
        seen.add(token)
        ordered.append(token)
    return ordered


def infer_scientific_image_inputs_from_text(user_text: str | None) -> list[str]:
    candidates = extract_scientific_image_paths_from_text(user_text)
    if not candidates:
        return []
    remote_candidates = [token for token in candidates if _looks_like_remote_input_path(token)]
    if remote_candidates:
        return remote_candidates
    existing_locals = [token for token in candidates if Path(token).expanduser().exists()]
    if existing_locals:
        return existing_locals
    return candidates


def _is_segmentation_image(path: Path) -> bool:
    lower = path.name.lower()
    if lower.endswith(".pdf") or lower.endswith(".txt"):
        return False
    if _is_sequence_media(path):
        return True
    if _is_directory_image_store(path):
        return True
    if lower.endswith(".ome.zarr") or lower.endswith(".zarr"):
        return path.is_dir()
    if lower.endswith(".nii.gz") or lower.endswith(".ome.tif") or lower.endswith(".ome.tiff"):
        return True
    if path.suffix.lower() in _SEGMENTATION_IMAGE_EXTS:
        return True
    if not path.exists() or not path.is_file():
        return False
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def _is_tabular_csv(path: Path) -> bool:
    lower = path.name.lower()
    return lower.endswith((".csv", ".tsv", ".tab", ".txt", ".csv.gz", ".tsv.gz"))


def _canonical_uploaded_filename(name: str) -> str:
    """Recover the original filename from session upload names like '<hash>__file.ext'."""
    match = _UPLOADED_HASH_PREFIX_RE.match(name or "")
    if not match:
        return name
    base = str(match.group("base") or "").strip()
    return base or name


_MASK_ARTIFACT_SAFE_EXTS = {
    ".npy",
    ".npz",
    ".nii",
    ".nii.gz",
}
_MASK_ARTIFACT_IMAGE_EXTS = {
    ".png",
    ".tif",
    ".tiff",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".webp",
}
_MASK_ARTIFACT_HINT_TOKENS = {
    "mask",
    "masks",
    "seg",
    "segmentation",
    "prediction",
    "pred",
    "binary",
}
_GROUND_TRUTH_HINT_TOKENS = {
    "gt",
    "label",
    "labels",
    "annotation",
    "annotations",
    "annot",
    "manual",
    "target",
    "truth",
    "reference",
}


def _compound_lower_suffix(path: str | Path) -> str:
    token = str(path or "").strip().lower()
    if token.endswith(".nii.gz"):
        return ".nii.gz"
    suffixes = Path(token).suffixes
    if len(suffixes) >= 2:
        double = "".join(suffixes[-2:]).lower()
        if double in {".ome.tif", ".ome.tiff"}:
            return double
    return Path(token).suffix.lower()


def _normalized_artifact_hint_text(path: str | Path) -> str:
    raw_name = Path(str(path or "")).name
    canonical = _canonical_uploaded_filename(raw_name)
    normalized = re.sub(r"[^a-z0-9]+", " ", canonical.lower()).strip()
    return re.sub(r"\s+", " ", normalized)


def _has_artifact_hint(path: str | Path, tokens: set[str]) -> bool:
    normalized = _normalized_artifact_hint_text(path)
    if not normalized:
        return False
    words = set(normalized.split())
    if words.intersection(tokens):
        return True
    return "ground truth" in normalized or "groundtruth" in normalized


def _looks_like_uploaded_ground_truth_artifact(path: str | Path) -> bool:
    suffix = _compound_lower_suffix(path)
    if suffix not in (_MASK_ARTIFACT_SAFE_EXTS | _MASK_ARTIFACT_IMAGE_EXTS):
        return False
    return _has_artifact_hint(path, _GROUND_TRUTH_HINT_TOKENS)


def _looks_like_uploaded_mask_artifact(path: str | Path) -> bool:
    suffix = _compound_lower_suffix(path)
    if suffix not in (_MASK_ARTIFACT_SAFE_EXTS | _MASK_ARTIFACT_IMAGE_EXTS):
        return False
    return _has_artifact_hint(path, _MASK_ARTIFACT_HINT_TOKENS | _GROUND_TRUTH_HINT_TOKENS)


def _looks_like_explicit_mask_path(path: str | Path) -> bool:
    suffix = _compound_lower_suffix(path)
    if suffix in _MASK_ARTIFACT_SAFE_EXTS:
        return True
    if suffix in _MASK_ARTIFACT_IMAGE_EXTS:
        return _has_artifact_hint(path, _MASK_ARTIFACT_HINT_TOKENS | _GROUND_TRUTH_HINT_TOKENS)
    return False


def _looks_like_explicit_ground_truth_path(path: str | Path) -> bool:
    suffix = _compound_lower_suffix(path)
    if suffix not in (_MASK_ARTIFACT_SAFE_EXTS | _MASK_ARTIFACT_IMAGE_EXTS):
        return False
    return _has_artifact_hint(path, _GROUND_TRUTH_HINT_TOKENS)


def _existing_local_paths(paths: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in paths:
        token = str(item or "").strip()
        if not token or token.lower().startswith(("http://", "https://")):
            continue
        candidate = Path(token).expanduser()
        try:
            if not candidate.exists() or not candidate.is_file():
                continue
            normalized = str(candidate.resolve())
        except Exception:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out


def _collect_latest_mask_paths_from_refs(latest_result_refs: dict[str, Any] | None) -> list[str]:
    if not isinstance(latest_result_refs, dict):
        return []
    candidates: list[str] = []
    for key in (
        "segment_image_sam2.mask_paths",
        "segment_image_sam2.preferred_upload_paths",
        "segment_image_sam3.mask_paths",
        "segment_image_sam3.preferred_upload_paths",
        "segment_image_megaseg.mask_paths",
        "segment_image_megaseg.preferred_upload_paths",
        "sam2_prompt_image.mask_paths",
        "sam2_prompt_image.preferred_upload_paths",
        "latest_segmentation_mask_paths",
    ):
        values = latest_result_refs.get(key)
        if isinstance(values, list):
            for item in values:
                token = str(item or "").strip()
                if token:
                    candidates.append(token)
    for key in (
        "segment_image_sam2.latest_mask_path",
        "segment_image_sam3.latest_mask_path",
        "segment_image_megaseg.latest_mask_path",
        "sam2_prompt_image.latest_mask_path",
        "latest_mask_path",
        "latest_segmentation_mask_path",
    ):
        token = str(latest_result_refs.get(key) or "").strip()
        if token:
            candidates.append(token)
    resolved = _existing_local_paths(candidates)
    return [path for path in resolved if _looks_like_explicit_mask_path(path)]


def _collect_latest_ground_truth_paths_from_refs(
    latest_result_refs: dict[str, Any] | None,
) -> list[str]:
    if not isinstance(latest_result_refs, dict):
        return []
    candidates: list[str] = []
    pairs = latest_result_refs.get("latest_eval_pairs")
    if isinstance(pairs, list):
        for row in pairs:
            if not isinstance(row, dict):
                continue
            token = str(row.get("ground_truth") or "").strip()
            if token:
                candidates.append(token)
    resolved = _existing_local_paths(candidates)
    return [path for path in resolved if _looks_like_explicit_ground_truth_path(path)]


def _strip_known_mask_extension(name: str) -> str:
    lower = (name or "").lower()
    for ext in (
        ".nii.gz",
        ".ome.tiff",
        ".ome.tif",
        ".tiff",
        ".tif",
        ".nrrd",
        ".mha",
        ".mhd",
        ".npy",
    ):
        if lower.endswith(ext):
            return name[: -len(ext)]
    return Path(name).stem


def _normalize_stem_tokens(tokens: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    merged = list(_DEFAULT_STEM_STRIP_TOKENS)
    for item in tokens or []:
        value = str(item).strip().lower()
        if value:
            merged.append(value)
    deduped = sorted(set(merged), key=len, reverse=True)
    return tuple(deduped)


def _normalized_pairing_stem(path: str, stem_strip_tokens: list[str] | None = None) -> str:
    name = _canonical_uploaded_filename(Path(str(path)).name)
    stem = _strip_known_mask_extension(name).lower()
    stem = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    if not stem:
        return ""

    strip_tokens = _normalize_stem_tokens(stem_strip_tokens)
    if not strip_tokens:
        return stem

    changed = True
    while changed and stem:
        changed = False
        for token in strip_tokens:
            suffix = f"_{token}"
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)].strip("_")
                changed = True
    return stem


def _path_signature(path: str | Path) -> dict[str, Any]:
    p = Path(str(path)).expanduser()
    try:
        stat = p.stat()
        return {
            "path": str(p.resolve()),
            "exists": True,
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    except Exception:
        return {
            "path": str(p),
            "exists": False,
        }


def _segmentation_cache_key(tool_name: str, args: dict[str, Any]) -> str:
    safe_args = dict(args)
    file_paths = [str(p) for p in safe_args.get("file_paths") or []]
    safe_args["file_signatures"] = [_path_signature(p) for p in file_paths]
    safe_args.pop("force_rerun", None)
    payload = json.dumps(
        {
            "tool": tool_name,
            "args": safe_args,
        },
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _segmentation_result_paths_exist(result: dict[str, Any]) -> bool:
    if not isinstance(result, dict):
        return False
    if result.get("success") is not True:
        return False
    paths: list[str] = []
    pref = result.get("preferred_upload_paths")
    if isinstance(pref, list):
        paths.extend(str(p) for p in pref if p)
    files_processed = result.get("files_processed")
    if isinstance(files_processed, list):
        for row in files_processed:
            if isinstance(row, dict) and row.get("preferred_upload_path"):
                paths.append(str(row.get("preferred_upload_path")))
    if not paths:
        return False
    return all(Path(p).expanduser().exists() for p in paths)


def _tool_result_refs_for_segmentation(result: dict[str, Any]) -> dict[str, Any]:
    refs: dict[str, Any] = {}
    result_group_id = str(result.get("result_group_id") or "").strip()
    if result_group_id:
        refs["latest_segmentation_result_group_id"] = result_group_id
    preferred = result.get("preferred_upload_paths")
    if isinstance(preferred, list) and preferred:
        refs["latest_segmentation_mask_path"] = str(preferred[-1])
        refs["latest_segmentation_mask_paths"] = [str(p) for p in preferred]
    files_processed = result.get("files_processed")
    if isinstance(files_processed, list):
        refs["latest_segmentation_manifest"] = [
            {
                "file": row.get("file"),
                "preferred_upload_path": row.get("preferred_upload_path"),
                "coverage_percent": row.get("coverage_percent"),
                "success": row.get("success"),
            }
            for row in files_processed
            if isinstance(row, dict)
        ]
    if result.get("output_directory"):
        refs["latest_segmentation_output_directory"] = str(result.get("output_directory"))
    return refs


def _stable_result_group_id(prefix: str, *parts: Any) -> str:
    normalized_parts = [str(part or "").strip() for part in parts if str(part or "").strip()]
    digest_input = "||".join(normalized_parts) or prefix
    digest = hashlib.sha1(digest_input.encode("utf-8")).hexdigest()[:12]
    safe_prefix = _safe_slug(prefix).replace("-", "_")
    return f"{safe_prefix}_{digest}"


def _segmentation_result_group_id(result: dict[str, Any], *, tool_name: str) -> str:
    explicit = str(result.get("result_group_id") or "").strip()
    if explicit:
        return explicit
    output_directory = str(result.get("output_directory") or "").strip()
    if output_directory:
        return _stable_result_group_id(tool_name, output_directory)
    preferred = result.get("preferred_upload_paths")
    preferred_paths = (
        [str(item or "").strip() for item in preferred if str(item or "").strip()]
        if isinstance(preferred, list)
        else []
    )
    files_processed = result.get("files_processed")
    file_tokens = [
        str((row or {}).get("file") or "").strip()
        for row in list(files_processed or [])
        if isinstance(row, dict) and str((row or {}).get("file") or "").strip()
    ]
    return _stable_result_group_id(tool_name, *preferred_paths[:8], *file_tokens[:8])


def _rewrite_megaseg_payload_paths(value: Any, path_map: dict[str, str]) -> Any:
    if isinstance(value, str):
        return path_map.get(value, value)
    if isinstance(value, list):
        return [_rewrite_megaseg_payload_paths(item, path_map) for item in value]
    if isinstance(value, dict):
        return {key: _rewrite_megaseg_payload_paths(item, path_map) for key, item in value.items()}
    return value


def _build_megaseg_tool_response(
    *,
    runner_result: dict[str, Any],
    output_dir: Path,
    checkpoint_display: str | None,
    structure_channel: int,
    nucleus_channel: int | None,
    channel_index_base: int,
    mask_threshold: float,
    requested_device: str | None,
    non_images: list[str],
    runner_stderr_text: str | None = None,
) -> dict[str, Any]:
    runner_files = (
        runner_result.get("files") if isinstance(runner_result.get("files"), list) else []
    )
    successful = [row for row in runner_files if isinstance(row, dict) and row.get("success")]
    files_processed: list[dict[str, Any]] = []
    preferred_upload_paths: list[str] = []
    preferred_upload_entries: list[dict[str, Any]] = []
    visualization_paths: list[dict[str, Any]] = []
    scientific_rows: list[dict[str, Any]] = []

    for row in runner_files:
        if not isinstance(row, dict):
            continue
        if row.get("success"):
            segmentation = (
                row.get("segmentation") if isinstance(row.get("segmentation"), dict) else {}
            )
            intensity_context = (
                row.get("intensity_context")
                if isinstance(row.get("intensity_context"), dict)
                else {}
            )
            preferred_path = str(row.get("mask_path") or "").strip() or None
            probability_path = str(row.get("probability_path") or "").strip() or None
            if preferred_path:
                preferred_upload_paths.append(preferred_path)
                preferred_upload_entries.append(
                    {
                        "file": row.get("file"),
                        "path": preferred_path,
                    }
                )
            row_visualizations = (
                row.get("visualizations") if isinstance(row.get("visualizations"), list) else []
            )
            for visualization in row_visualizations:
                if not isinstance(visualization, dict) or not visualization.get("path"):
                    continue
                visualization_paths.append(
                    {
                        "path": str(visualization.get("path")),
                        "file": row.get("file"),
                        "coverage_percent": segmentation.get("coverage_percent"),
                        "title": visualization.get("title"),
                        "kind": visualization.get("kind"),
                    }
                )
            files_processed.append(
                {
                    "file": row.get("file"),
                    "success": True,
                    "coverage_percent": segmentation.get("coverage_percent"),
                    "object_count": segmentation.get("object_count"),
                    "active_slice_count": segmentation.get("active_slice_count"),
                    "z_slice_count": segmentation.get("z_slice_count"),
                    "largest_component_voxels": segmentation.get("largest_component_voxels"),
                    "preferred_upload_path": preferred_path,
                    "probability_path": probability_path,
                    "visualization_saved": bool(row_visualizations),
                    "technical_summary": row.get("technical_summary"),
                }
            )
            scientific_rows.append(
                {
                    "file": row.get("file"),
                    "coverage_percent": segmentation.get("coverage_percent"),
                    "object_count": segmentation.get("object_count"),
                    "active_slice_count": segmentation.get("active_slice_count"),
                    "z_slice_count": segmentation.get("z_slice_count"),
                    "largest_component_voxels": segmentation.get("largest_component_voxels"),
                    "structure_inside_outside_ratio": intensity_context.get(
                        "structure_inside_outside_ratio"
                    ),
                    "nucleus_inside_outside_ratio": intensity_context.get(
                        "nucleus_inside_outside_ratio"
                    ),
                    "technical_summary": row.get("technical_summary"),
                }
            )
        else:
            files_processed.append(
                {
                    "file": row.get("file"),
                    "success": False,
                    "error": row.get("error", "Megaseg inference failed."),
                }
            )

    aggregate = (
        runner_result.get("aggregate") if isinstance(runner_result.get("aggregate"), dict) else {}
    )
    result_group_id = _stable_result_group_id(
        "megaseg",
        str(output_dir),
        *preferred_upload_paths[:8],
    )
    summary_payload = {
        "processed_files": len(successful),
        "total_files": len(runner_files),
        "mean_coverage_percent": aggregate.get("mean_coverage_percent"),
        "median_coverage_percent": aggregate.get("median_coverage_percent"),
        "mean_object_count": aggregate.get("mean_object_count"),
        "median_object_count": aggregate.get("median_object_count"),
    }

    ui_artifacts: list[dict[str, Any]] = [
        {
            "type": "metrics",
            "title": "Megaseg summary",
            "result_group_id": result_group_id,
            "payload": summary_payload,
        },
        {
            "type": "table",
            "title": "Megaseg per-file metrics",
            "result_group_id": result_group_id,
            "payload": scientific_rows[:200],
        },
    ]
    for item in visualization_paths[:24]:
        if item.get("path"):
            ui_artifacts.append(
                {
                    "type": "image",
                    "title": item.get("title") or "Megaseg overlay",
                    "path": item.get("path"),
                    "result_group_id": result_group_id,
                }
            )

    warnings: list[Any] = []
    if non_images:
        warnings.append({"ignored_non_images": non_images[:50]})
    if runner_stderr_text and "Failed to load image Python extension" not in runner_stderr_text:
        warnings.append({"runner_stderr_tail": runner_stderr_text[-2000:]})
    runner_warnings = runner_result.get("warnings")
    if isinstance(runner_warnings, list):
        warnings.extend(runner_warnings[:20])

    response: dict[str, Any] = {
        "success": len(successful) > 0,
        "processed": len(successful),
        "total_files": len(runner_files),
        "result_group_id": result_group_id,
        "files_processed": files_processed,
        "output_directory": str(output_dir),
        "model": "Megaseg DynUNet",
        "checkpoint_path": checkpoint_display,
        "device": runner_result.get("device") or requested_device or "auto",
        "structure_channel": int(structure_channel),
        "nucleus_channel": (int(nucleus_channel) if nucleus_channel is not None else None),
        "channel_index_base": int(channel_index_base),
        "mask_threshold": float(mask_threshold),
        "preferred_upload_paths": preferred_upload_paths,
        "preferred_upload_entries": preferred_upload_entries,
        "visualization_paths": visualization_paths,
        "summary_csv_path": runner_result.get("summary_csv_path"),
        "report_path": runner_result.get("report_path"),
        "scientific_summary": {
            "aggregate": aggregate,
            "files": scientific_rows[:200],
        },
        "ui_artifacts": ui_artifacts,
        "message": (
            "Megaseg DynUNet inference completed on the requested microscopy image set. "
            "Binary mask artifacts are ready for upload or downstream quantify_segmentation_masks, "
            "and probability volumes plus technical summaries were saved for each processed file."
            if successful
            else "Megaseg inference did not produce a successful segmentation result."
        ),
    }
    if warnings:
        response["warnings"] = warnings

    latest_refs = _tool_result_refs_for_segmentation(response)
    if preferred_upload_paths:
        latest_refs.update(
            {
                "segment_image_megaseg.mask_paths": [str(path) for path in preferred_upload_paths],
                "segment_image_megaseg.preferred_upload_paths": [
                    str(path) for path in preferred_upload_paths
                ],
                "segment_image_megaseg.latest_mask_path": str(preferred_upload_paths[-1]),
                "segment_image_megaseg.result_group_id": result_group_id,
            }
        )
    if response.get("report_path"):
        latest_refs["segment_image_megaseg.report_path"] = str(response.get("report_path"))
    if response.get("summary_csv_path"):
        latest_refs["segment_image_megaseg.summary_csv_path"] = str(
            response.get("summary_csv_path")
        )
    response["latest_result_refs"] = latest_refs
    return response


def _download_megaseg_service_result(
    *,
    client: MegasegServiceClient,
    job_payload: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    result = deepcopy(
        job_payload.get("result") if isinstance(job_payload.get("result"), dict) else {}
    )
    artifact_manifest = (
        job_payload.get("artifact_manifest")
        if isinstance(job_payload.get("artifact_manifest"), list)
        else []
    )
    remote_output_dir = str(result.get("output_directory") or "").strip()
    path_map: dict[str, str] = {}
    if remote_output_dir:
        path_map[remote_output_dir] = str(output_dir)

    for item in artifact_manifest:
        if not isinstance(item, dict):
            continue
        artifact_name = str(item.get("name") or "").strip()
        if not artifact_name:
            continue
        destination = output_dir / artifact_name
        client.download_artifact(
            job_id=str(job_payload.get("job_id") or ""),
            artifact_name=artifact_name,
            destination=destination,
        )
        if remote_output_dir:
            remote_path = str(Path(remote_output_dir) / artifact_name)
            path_map[remote_path] = str(destination)

    return _rewrite_megaseg_payload_paths(result, path_map)


def _run_megaseg_via_service(
    *,
    service_url: str,
    service_api_key: str | None,
    timeout_seconds: float,
    poll_interval_seconds: float,
    wait_timeout_seconds: float,
    download_artifacts: bool,
    local_image_paths: list[Path],
    remote_image_paths: list[str],
    output_dir: Path,
    structure_channel: int,
    nucleus_channel: int | None,
    channel_index_base: int,
    mask_threshold: float,
    save_visualizations: bool,
    generate_report: bool,
    device: str | None,
    checkpoint_path: str | None,
    structure_name: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    client = MegasegServiceClient(
        base_url=service_url,
        api_key=service_api_key,
        timeout_seconds=timeout_seconds,
    )
    request_payload: dict[str, Any] = {
        "sources": [{"uri": raw} for raw in remote_image_paths],
        "structure_channel": int(structure_channel),
        "nucleus_channel": (int(nucleus_channel) if nucleus_channel is not None else None),
        "channel_index_base": int(channel_index_base),
        "mask_threshold": float(mask_threshold),
        "save_visualizations": bool(save_visualizations),
        "generate_report": bool(generate_report),
        "device": str(device or "").strip() or None,
        "structure_name": str(structure_name or "structure"),
    }
    if checkpoint_path:
        request_payload["checkpoint_path"] = str(checkpoint_path)

    job_create = client.submit_job(
        request_payload=request_payload,
        local_upload_paths=local_image_paths,
    )
    job_id = str(job_create.get("job_id") or "").strip()
    if not job_id:
        raise MegasegServiceError("Megaseg service job creation did not return a job id.")
    job_payload = client.wait_for_job(
        job_id=job_id,
        poll_interval_seconds=poll_interval_seconds,
        wait_timeout_seconds=wait_timeout_seconds,
    )
    if not isinstance(job_payload.get("result"), dict):
        raise MegasegServiceError(f"Megaseg service job {job_id} did not return a result payload.")

    runner_result = (
        _download_megaseg_service_result(
            client=client,
            job_payload=job_payload,
            output_dir=output_dir,
        )
        if download_artifacts
        else deepcopy(job_payload["result"])
    )
    if not isinstance(runner_result, dict):
        raise MegasegServiceError(
            f"Megaseg service job {job_id} returned an invalid result payload."
        )
    return runner_result, job_payload


def segment_image_megaseg(
    file_paths: list[str],
    structure_channel: int = 4,
    nucleus_channel: int | None = 6,
    channel_index_base: int = 1,
    mask_threshold: float = 0.5,
    save_visualizations: bool = True,
    generate_report: bool = True,
    device: str | None = None,
    checkpoint_path: str | None = None,
    structure_name: str | None = None,
) -> dict[str, Any]:
    """Run Megaseg DynUNet inference on multichannel microscopy images."""
    if not file_paths:
        return {"success": False, "error": "file_paths is required."}

    settings = get_settings()
    service_url = str(getattr(settings, "resolved_megaseg_service_url", "") or "").strip()
    remote_inputs = [
        str(path).strip()
        for path in file_paths
        if str(path).strip() and _looks_like_remote_input_path(str(path))
    ]
    local_inputs = [
        str(path).strip()
        for path in file_paths
        if str(path).strip() and not _looks_like_remote_input_path(str(path))
    ]

    try:
        expanded = _expand_file_inputs(local_inputs)
    except Exception as exc:
        return {"success": False, "error": str(exc)}

    structure_channel, nucleus_channel = _maybe_auto_adjust_megaseg_channels(
        file_paths=[*expanded, *remote_inputs],
        structure_channel=structure_channel,
        nucleus_channel=nucleus_channel,
        channel_index_base=channel_index_base,
    )

    missing = [str(path) for path in expanded if not path.exists()]
    if missing:
        return {
            "success": False,
            "error": "Some files do not exist.",
            "missing": missing[:50],
        }

    local_image_paths: list[Path] = [
        path.resolve() for path in expanded if path.exists() and _is_segmentation_image(path)
    ]
    images: list[str] = [str(path) for path in local_image_paths]
    non_images = [
        str(path) for path in expanded if path.exists() and not _is_segmentation_image(path)
    ]
    for raw_path in remote_inputs:
        if _looks_like_segmentation_remote_path(raw_path):
            images.append(raw_path)
        else:
            non_images.append(raw_path)
    if not images:
        return {"success": False, "error": "No microscopy image files were found in file_paths."}

    run_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
    output_dir = Path(_science_output_root("megaseg_results")) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    runner_result: dict[str, Any]
    runner_stderr_text: str | None = None
    checkpoint_display: str | None = None
    explicit_checkpoint = str(checkpoint_path or "").strip() or None

    if service_url:
        service_api_key = (
            str(getattr(settings, "resolved_megaseg_service_api_key", "") or "").strip() or None
        )
        if not service_api_key:
            return {
                "success": False,
                "error": (
                    "MEGASEG_SERVICE_URL is configured but MEGASEG_SERVICE_API_KEY is missing. "
                    "Set the private Megaseg service bearer token before using remote inference."
                ),
                "output_directory": str(output_dir),
            }
        try:
            runner_result, job_payload = _run_megaseg_via_service(
                service_url=service_url,
                service_api_key=service_api_key,
                timeout_seconds=float(
                    getattr(settings, "megaseg_service_timeout_seconds", 60.0) or 60.0
                ),
                poll_interval_seconds=float(
                    getattr(settings, "megaseg_service_poll_interval_seconds", 2.0) or 2.0
                ),
                wait_timeout_seconds=float(
                    getattr(settings, "megaseg_service_wait_timeout_seconds", 7200.0) or 7200.0
                ),
                download_artifacts=bool(
                    getattr(settings, "megaseg_service_download_artifacts", True)
                ),
                local_image_paths=local_image_paths,
                remote_image_paths=[item for item in images if _looks_like_remote_input_path(item)],
                output_dir=output_dir,
                structure_channel=structure_channel,
                nucleus_channel=nucleus_channel,
                channel_index_base=channel_index_base,
                mask_threshold=mask_threshold,
                save_visualizations=save_visualizations,
                generate_report=generate_report,
                device=device,
                checkpoint_path=explicit_checkpoint,
                structure_name=structure_name,
            )
            checkpoint_display = (
                str(runner_result.get("checkpoint_path") or explicit_checkpoint or "").strip()
                or None
            )
            if isinstance(job_payload, dict):
                logger.info(
                    "Megaseg service job %s finished with status=%s",
                    job_payload.get("job_id"),
                    job_payload.get("status"),
                )
        except MegasegServiceTimeoutError as exc:
            return {
                "success": False,
                "error": str(exc),
                "output_directory": str(output_dir),
            }
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[-4000:] if exc.response is not None else str(exc)
            return {
                "success": False,
                "error": "Megaseg service request failed.",
                "details": detail or str(exc),
                "output_directory": str(output_dir),
            }
        except MegasegServiceError as exc:
            return {
                "success": False,
                "error": str(exc),
                "output_directory": str(output_dir),
            }
        except Exception as exc:
            return {
                "success": False,
                "error": f"Failed to execute Megaseg via remote service: {exc}",
                "output_directory": str(output_dir),
            }
    else:
        runner_script = _resolve_megaseg_runner_script()
        if not runner_script.exists():
            return {
                "success": False,
                "error": f"Megaseg runner script is missing: {runner_script}",
            }

        runner_python = _resolve_megaseg_python()
        if not runner_python:
            return {
                "success": False,
                "error": (
                    "Megaseg requires a Python runtime with cyto-dl/monai installed. "
                    "Set MEGASEG_PYTHON or CYTODL_PYTHON to a valid interpreter."
                ),
            }

        resolved_checkpoint = _resolve_megaseg_checkpoint_path(explicit_checkpoint)
        if not resolved_checkpoint:
            return {
                "success": False,
                "error": (
                    "No Megaseg checkpoint could be resolved. "
                    "Provide checkpoint_path or set MEGASEG_CHECKPOINT_PATH."
                ),
            }
        checkpoint_display = str(Path(resolved_checkpoint).expanduser().resolve())

        request_payload = {
            "file_paths": images,
            "output_dir": str(output_dir.resolve()),
            "structure_channel": int(structure_channel),
            "nucleus_channel": (int(nucleus_channel) if nucleus_channel is not None else None),
            "channel_index_base": int(channel_index_base),
            "mask_threshold": float(mask_threshold),
            "save_visualizations": bool(save_visualizations),
            "generate_report": bool(generate_report),
            "device": str(device or "").strip() or None,
            "checkpoint_path": checkpoint_display,
            "structure_name": str(structure_name or "structure"),
        }
        request_json_path = output_dir / "megaseg_request.json"
        request_json_path.write_text(
            json.dumps(request_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        try:
            completed = subprocess.run(
                [runner_python, str(runner_script), "--request-json", str(request_json_path)],
                capture_output=True,
                text=True,
                check=False,
                timeout=7200,
            )
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Megaseg inference timed out after 7200 seconds.",
                "output_directory": str(output_dir),
            }
        except Exception as exc:
            return {
                "success": False,
                "error": f"Failed to launch Megaseg inference: {exc}",
                "output_directory": str(output_dir),
            }

        stdout_text = str(completed.stdout or "").strip()
        runner_stderr_text = str(completed.stderr or "").strip()
        if completed.returncode != 0:
            return {
                "success": False,
                "error": "Megaseg inference subprocess failed.",
                "details": runner_stderr_text[-4000:] or stdout_text[-4000:] or None,
                "return_code": int(completed.returncode),
                "output_directory": str(output_dir),
            }

        try:
            runner_result = json.loads(stdout_text)
        except Exception as exc:
            return {
                "success": False,
                "error": f"Failed to parse Megaseg runner output: {exc}",
                "stdout_tail": stdout_text[-4000:] or None,
                "stderr_tail": runner_stderr_text[-4000:] or None,
                "output_directory": str(output_dir),
            }

    return _build_megaseg_tool_response(
        runner_result=runner_result,
        output_dir=output_dir,
        checkpoint_display=checkpoint_display,
        structure_channel=structure_channel,
        nucleus_channel=nucleus_channel,
        channel_index_base=channel_index_base,
        mask_threshold=mask_threshold,
        requested_device=device,
        non_images=non_images,
        runner_stderr_text=runner_stderr_text,
    )


def _yolo_candidate_stems(path: Path) -> list[str]:
    """Return candidate stems for pairing labels/images (exact + dehashed)."""
    stems: list[str] = []
    exact = path.stem
    if exact:
        stems.append(exact)

    canonical_name = _canonical_uploaded_filename(path.name)
    canonical_stem = Path(canonical_name).stem
    if canonical_stem and canonical_stem not in stems:
        stems.append(canonical_stem)
    return stems


def _expand_file_inputs(paths: list[str], *, max_files: int = 5000) -> list[Path]:
    """Expand a list of file and/or directory inputs to a de-duplicated list of files."""
    expanded: list[Path] = []
    for raw in paths:
        if raw is None:
            continue
        p = Path(str(raw)).expanduser()
        if _is_directory_image_store(p):
            expanded.append(p)
            continue
        if p.is_dir():
            for child in p.rglob("*"):
                if child.is_file():
                    expanded.append(child)
                    if len(expanded) > max_files:
                        raise ValueError(
                            f"Too many files found (>{max_files}) under directory: {p}"
                        )
        else:
            expanded.append(p)

    # De-dupe while preserving order
    seen: set[str] = set()
    deduped: list[Path] = []
    for p in expanded:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)
    return deduped


def _torch_cuda_available() -> bool:
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _resolve_pipeline_device(device: str | None) -> int:
    has_cuda = _torch_cuda_available()
    if device is None:
        return 0 if has_cuda else -1

    value = str(device).strip().lower()
    if value in {"cpu", "-1"}:
        return -1
    if value in {"cuda", "gpu"}:
        if not has_cuda:
            raise ValueError("CUDA device requested, but CUDA is not available.")
        return 0
    if value.startswith("cuda:"):
        value = value.split(":", 1)[1]
    try:
        idx = int(value)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Invalid device value: {device!r}. Use cpu, cuda, cuda:N, or N.") from e
    if idx < 0:
        return -1
    if not has_cuda:
        raise ValueError("CUDA index requested, but CUDA is not available.")
    return idx


def _resolve_torch_device(device: str | None) -> str:
    has_cuda = _torch_cuda_available()
    if device is None:
        return "cuda" if has_cuda else "cpu"

    value = str(device).strip().lower()
    if value in {"cpu", "-1"}:
        return "cpu"
    if value in {"cuda", "gpu"}:
        if not has_cuda:
            raise ValueError("CUDA device requested, but CUDA is not available.")
        return "cuda"
    if value.startswith("cuda:"):
        if not has_cuda:
            raise ValueError("CUDA device requested, but CUDA is not available.")
        return value
    if re.fullmatch(r"\d+", value):
        if not has_cuda:
            raise ValueError("CUDA index requested, but CUDA is not available.")
        return f"cuda:{value}"
    raise ValueError(f"Invalid device value: {device!r}. Use cpu, cuda, cuda:N, or N.")


def _depthpro_runtime(
    *,
    model_id: str,
    use_fov_model: bool,
    device: str | None,
) -> tuple[Any, Any, str]:
    try:
        pass  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError("torch is required for DepthPro inference.") from e

    try:
        from transformers import (  # type: ignore
            DepthProForDepthEstimation,
            DepthProImageProcessorFast,
        )

        processor_cls = DepthProImageProcessorFast
    except Exception:
        try:
            from transformers import (  # type: ignore
                DepthProForDepthEstimation,
                DepthProImageProcessor,
            )

            processor_cls = DepthProImageProcessor
        except Exception as e:  # noqa: BLE001
            raise ImportError(
                "DepthPro classes are unavailable in this transformers runtime. "
                "Upgrade transformers to a version that includes DepthPro."
            ) from e

    torch_device = _resolve_torch_device(device)
    cache_key = (str(model_id), bool(use_fov_model), torch_device)
    cached = _DEPTH_PRO_RUNTIME_CACHE.get(cache_key)
    if cached is not None:
        processor, model = cached
        return processor, model, torch_device

    processor = processor_cls.from_pretrained(model_id)
    try:
        model = DepthProForDepthEstimation.from_pretrained(
            model_id,
            use_fov_model=bool(use_fov_model),
        )
    except TypeError:
        model = DepthProForDepthEstimation.from_pretrained(model_id)
    model = model.to(torch_device)
    model.eval()
    _DEPTH_PRO_RUNTIME_CACHE[cache_key] = (processor, model)
    return processor, model, torch_device


def _depth_to_uint8(depth: np.ndarray) -> np.ndarray:
    arr = np.asarray(depth, dtype=np.float32)
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        arr = arr.reshape((arr.shape[-2], arr.shape[-1]))

    finite = arr[np.isfinite(arr)]
    if finite.size <= 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    lo = float(np.percentile(finite, 1.0))
    hi = float(np.percentile(finite, 99.0))
    if hi <= lo:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (norm * 255.0).astype(np.uint8)


def _depth_colorize(depth_u8: np.ndarray) -> Image.Image:
    gray = Image.fromarray(depth_u8, mode="L")
    return ImageOps.colorize(gray, black="#111827", mid="#2563eb", white="#f59e0b")


def upload_to_bisque(
    file_paths: list[str],
    dataset_uri: str | None = None,
    dataset_name: str | None = None,
    create_dataset_if_missing: bool = False,
    dataset_tags: dict[str, str] | list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """
    Upload files to BisQue image repository from file paths.
    Files should already be saved to the data folder.

    Args:
        file_paths: List of file paths to upload to BisQue
        dataset_uri: Optional existing dataset URI/resource_uniq to append uploads to
        dataset_name: Optional existing dataset name to resolve before appending
        create_dataset_if_missing: Whether to create the named dataset if not found
        dataset_tags: Optional tags to add if a new dataset is created

    Returns:
        Dictionary with upload results and metadata
    """
    try:
        from bqapi.comm import BQSession
    except ImportError:
        return {
            "success": False,
            "error": "bqapi package not installed. Install with: pip install bqapi",
        }

    try:
        # Initialize BisQue session
        bq = BQSession()
        root, _, _ = _init_bisque_session_with_runtime_auth(bq=bq)
        logger.info("Connecting to BisQue at %s", root)

        results = []
        requested_dataset_uri = str(dataset_uri or "").strip() or None
        requested_dataset_name = str(dataset_name or "").strip() or None

        for file_path in file_paths:
            try:
                # Check if file exists
                if not os.path.exists(file_path):
                    results.append(
                        {
                            "file": os.path.basename(file_path),
                            "success": False,
                            "error": f"File not found: {file_path}",
                        }
                    )
                    continue

                file_name = os.path.basename(file_path)
                file_size = os.path.getsize(file_path)
                existing_upload = _lookup_session_visible_canonical_bisque_upload_for_local_path(
                    bq=bq,
                    bisque_root=root,
                    path=file_path,
                )
                if existing_upload is not None:
                    logger.info(
                        "Reusing canonical BisQue resource for %s from upload registry",
                        file_path,
                    )
                    resource_uniq = (
                        str(existing_upload.get("canonical_resource_uniq") or "").strip() or None
                    )
                    resource_uri_raw = (
                        str(existing_upload.get("canonical_resource_uri") or "").strip() or None
                    )
                    upload_error_hint = None
                    response_value = resource_uri_raw or resource_uniq or file_path
                    reused_existing_upload = True
                else:
                    logger.info(f"Uploading {file_name} from {file_path} to BisQue")
                    response = bq.postblob(file_path, xml=_bisque_upload_resource_xml(file_path))
                    resource_uniq = _extract_bisque_resource_uniq(response)
                    resource_uri_raw = _extract_bisque_resource_uri(response)
                    upload_error_hint = _upload_response_error_hint(response)
                    response_value = response
                    reused_existing_upload = False

                links = _build_bisque_resource_links(
                    resource_uri_raw or resource_uniq,
                    root,
                )
                resolved_uri = str(links.get("resource_uri") or "").strip()
                if not resolved_uri:
                    results.append(
                        {
                            "file": file_name,
                            "local_path": file_path,
                            "success": False,
                            "error": upload_error_hint
                            or "Upload did not return a BisQue resource URI.",
                        }
                    )
                    logger.error(
                        "Failed to upload %s: missing resource URI (hint=%s)",
                        file_path,
                        upload_error_hint or "none",
                    )
                    continue

                results.append(
                    {
                        "file": file_name,
                        "local_path": file_path,
                        "success": True,
                        "size_bytes": file_size,
                        "resource_uniq": links.get("resource_uniq") or resource_uniq,
                        "resource_uri": resolved_uri,
                        "client_view_url": links.get("client_view_url"),
                        "image_service_url": links.get("image_service_url"),
                        "bisque_url": links.get("client_view_url") or str(response_value),
                        "reused_existing_upload": reused_existing_upload,
                        "canonical_upload_file_id": (
                            str(existing_upload.get("file_id") or "").strip()
                            if existing_upload is not None
                            else None
                        ),
                    }
                )

                logger.info(f"Successfully uploaded {file_name}")

            except Exception as file_error:
                results.append(
                    {
                        "file": os.path.basename(file_path),
                        "local_path": file_path,
                        "success": False,
                        "error": str(file_error),
                    }
                )
                logger.error(f"Failed to upload {file_path}: {str(file_error)}")

        # Summary
        successful = sum(1 for r in results if r.get("success"))
        total = len(results)
        dataset_result: dict[str, Any] | None = None
        if requested_dataset_uri or requested_dataset_name:
            uploaded_resource_uris = [
                str(row.get("resource_uri") or "").strip()
                for row in results
                if isinstance(row, dict)
                and row.get("success")
                and str(row.get("resource_uri") or "").strip()
            ]
            if uploaded_resource_uris:
                dataset_result = _organize_bisque_resources_into_dataset_with_session(
                    bq=bq,
                    bisque_root=root,
                    resource_uris=uploaded_resource_uris,
                    dataset_uri=requested_dataset_uri,
                    dataset_name=requested_dataset_name,
                    create_dataset_if_missing=create_dataset_if_missing,
                    tags=dataset_tags,
                )
            else:
                dataset_result = {
                    "success": False,
                    "action": "skipped",
                    "dataset_uri": requested_dataset_uri,
                    "dataset_name": requested_dataset_name,
                    "added": 0,
                    "total_resources": 0,
                    "error": "No uploaded resources were available to organize into a dataset.",
                    "message": "Upload failed before any resources could be organized into a dataset.",
                }

        overall_success = successful > 0 and (
            dataset_result is None or bool(dataset_result.get("success"))
        )
        if dataset_result and not dataset_result.get("success"):
            message = (
                f"Uploaded {successful} of {total} file(s) to BisQue, but dataset organization failed: "
                f"{dataset_result.get('error') or dataset_result.get('message') or 'unknown error'}"
            )
        elif dataset_result and dataset_result.get("action") == "created":
            message = (
                f"Uploaded {successful} of {total} file(s) to BisQue and created dataset "
                f"'{dataset_result.get('dataset_name') or requested_dataset_name or 'dataset'}'."
            )
        elif dataset_result and dataset_result.get("action") == "appended":
            message = (
                f"Uploaded {successful} of {total} file(s) to BisQue and added them to dataset "
                f"'{dataset_result.get('dataset_name') or requested_dataset_name or 'dataset'}'."
            )
        elif overall_success:
            message = f"Uploaded {successful} of {total} file(s) to BisQue."
        else:
            message = f"Failed to upload files to BisQue ({successful}/{total} succeeded)."

        return {
            "success": overall_success,
            "uploaded": successful,
            "total": total,
            "results": results,
            "bisque_root": root,
            "dataset_action": dataset_result.get("action")
            if isinstance(dataset_result, dict)
            else None,
            "dataset_success": dataset_result.get("success")
            if isinstance(dataset_result, dict)
            else None,
            "dataset_uri": dataset_result.get("dataset_uri")
            if isinstance(dataset_result, dict)
            else requested_dataset_uri,
            "dataset_name": dataset_result.get("dataset_name")
            if isinstance(dataset_result, dict)
            else requested_dataset_name,
            "dataset_members_added": (
                dataset_result.get("added") if isinstance(dataset_result, dict) else None
            ),
            "dataset_client_view_url": (
                dataset_result.get("dataset_client_view_url")
                if isinstance(dataset_result, dict)
                else None
            ),
            "dataset": dataset_result,
            "error": (
                str(dataset_result.get("error") or "").strip()
                if isinstance(dataset_result, dict) and not dataset_result.get("success")
                else None
            ),
            "message": message,
        }

    except Exception as e:
        logger.error(f"BisQue upload error: {str(e)}")
        return {"success": False, "error": f"BisQue connection/upload failed: {str(e)}"}


def bisque_ping() -> dict[str, Any]:
    """Validate BisQue connectivity and credentials."""
    try:
        from bqapi.comm import BQSession
    except ImportError:
        return {"success": False, "error": "bqapi package not installed"}

    try:
        bq = BQSession()
        root, _, _ = _init_bisque_session_with_runtime_auth(bq=bq)
        info = _session_fetchxml_safe(bq, f"{root}/auth_service/session")
        user_name = _extract_bisque_user_from_xml(info)
        if not user_name:
            whoami = _session_fetchxml_safe(bq, f"{root}/auth_service/whoami")
            user_name = _extract_bisque_user_from_xml(whoami)
        if not user_name:
            runtime_user, _, _, _, _ = _get_bisque_auth_material()
            user_name = runtime_user
        return {"success": True, "bisque_root": root, "user": user_name}
    except Exception as e:
        logger.error(f"BisQue ping failed: {e}")
        return {"success": False, "error": str(e)}


def _extract_bisque_user_from_xml(xml: etree._Element | None) -> str | None:
    if xml is None:
        return None

    for key in ("user", "username", "name"):
        for tag in [xml.find(f"./tag[@name='{key}']"), *xml.findall(f".//tag[@name='{key}']")]:
            if tag is None:
                continue
            value = str(tag.get("value") or tag.get("name") or tag.text or "").strip()
            if _looks_like_bisque_resource_uri(value):
                continue
            if value:
                return value

    user_nodes = [xml] if xml.tag == "user" else list(xml.findall(".//user"))
    for node in user_nodes:
        value = str(node.get("name") or node.get("value") or node.text or "").strip()
        if _looks_like_bisque_resource_uri(value):
            continue
        if value:
            return value

    for attr in ("name", "value", "user", "username"):
        value = str(xml.get(attr) or "").strip()
        if _looks_like_bisque_resource_uri(value):
            continue
        if value:
            return value

    text_value = str(xml.text or "").strip()
    if _looks_like_bisque_resource_uri(text_value):
        return None
    return text_value or None


def _looks_like_bisque_resource_uri(value: str | None) -> bool:
    token = str(value or "").strip().lower()
    if not token:
        return False
    return "/data_service/" in token or "/image_service/" in token


def _loaded_bisque_resource_type(
    *,
    resource: Any | None,
    resource_xml: etree._Element | None,
) -> str | None:
    candidates: list[str | None] = []
    if resource is not None:
        for attr in ("tag", "resource_type", "type", "xmltag"):
            if hasattr(resource, attr):
                candidates.append(str(getattr(resource, attr) or "").strip() or None)
        xmltree = getattr(resource, "xmltree", None)
        if xmltree is not None and getattr(xmltree, "tag", None) is not None:
            candidates.append(str(etree.QName(xmltree.tag).localname or "").strip() or None)
    if resource_xml is not None:
        candidates.append(str(etree.QName(resource_xml.tag).localname or "").strip() or None)

    for value in candidates:
        token = str(value or "").strip()
        if token:
            return token
    return None


def _short_bisque_id(uri: str | None) -> str | None:
    if not uri:
        return None
    return uri.rstrip("/").split("/")[-1]


def _bisque_file_ext(name: str | None) -> str | None:
    if not name:
        return None
    lower = name.lower()
    if lower.endswith(".nii.gz"):
        return "nii.gz"
    if lower.endswith(".ome.tif"):
        return "ome.tif"
    if lower.endswith(".ome.tiff"):
        return "ome.tiff"
    if "." not in lower:
        return None
    return lower.rsplit(".", 1)[-1]


def _bisque_media_label(name: str | None, resource_type: str | None) -> str:
    ext = _bisque_file_ext(name)
    video_exts = {"mp4", "mov", "avi", "mkv", "webm", "mpeg", "mpg"}
    image_exts = {
        "tif",
        "tiff",
        "png",
        "jpg",
        "jpeg",
        "bmp",
        "ome.tif",
        "ome.tiff",
        "czi",
        "nd2",
        "svs",
        "vsi",
        "lsm",
        "lif",
        "gif",
    }
    volume_exts = {"nii", "nii.gz", "nrrd", "mha", "mhd", "zarr"}
    dicom_exts = {"dcm", "dicom"}
    if ext in video_exts:
        return "video"
    if ext in volume_exts:
        return "volume"
    if ext in dicom_exts:
        return "dicom"
    if ext in image_exts:
        return "image"
    return resource_type or "resource"


def _format_bisque_timestamp(ts: str | None) -> str | None:
    if not ts:
        return None
    try:
        from datetime import datetime

        norm = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(norm)
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ts


def _bisque_query_text_value(text: str | list[str] | None) -> str | None:
    if isinstance(text, str):
        token = str(text).strip()
        return token or None
    if isinstance(text, list):
        parts = [str(item).strip() for item in text if str(item or "").strip()]
        if not parts:
            return None
        return ", ".join(parts[:8])
    return None


def _bisque_filter_summary(filters: dict[str, Any] | None) -> str | None:
    if not isinstance(filters, dict) or not filters:
        return None
    parts: list[str] = []
    for key in sorted(filters.keys())[:8]:
        value = filters.get(key)
        if isinstance(value, list):
            values = [str(item).strip() for item in value if str(item or "").strip()]
            rendered = ", ".join(values[:4])
        else:
            rendered = str(value).strip()
        if not rendered:
            continue
        parts.append(f"{key}={rendered}")
    return "; ".join(parts) if parts else None


def _bisque_query_scope_text(query: dict[str, Any] | None) -> str:
    if not isinstance(query, dict):
        return ""
    parts: list[str] = []
    requested_type = str(query.get("requested_resource_type") or "").strip()
    resolved_type = str(query.get("resource_type") or "").strip()
    if requested_type and requested_type != resolved_type:
        parts.append(f"requested type '{requested_type}' (resolved as '{resolved_type}')")
    elif resolved_type:
        parts.append(f"type '{resolved_type}'")
    text_value = _bisque_query_text_value(query.get("text"))
    if text_value:
        parts.append(f'text "{text_value}"')
    original_tag_query = str(query.get("original_tag_query") or "").strip()
    if original_tag_query:
        parts.append(f"tag query {original_tag_query}")
    tag_filters = _bisque_filter_summary(query.get("tag_filters"))
    if tag_filters:
        parts.append(f"tag filters {tag_filters}")
    metadata_filters = _bisque_filter_summary(query.get("metadata_filters"))
    if metadata_filters:
        parts.append(f"metadata filters {metadata_filters}")
    if not parts:
        return "the requested BisQue search"
    return ", ".join(parts)


def _bisque_search_summary(
    resources: list[dict[str, Any]],
    resource_type: str,
    *,
    query: dict[str, Any] | None = None,
) -> str:
    count = len(resources)
    scope_text = _bisque_query_scope_text(query)
    if count == 0:
        if scope_text:
            return f"No {resource_type} assets found for {scope_text}."
        return f"No {resource_type} assets found."

    names = []
    for res in resources[:3]:
        name = res.get("name") or _short_bisque_id(res.get("uri"))
        if name:
            names.append(name)

    ext_counts: dict[str, int] = {}
    for res in resources:
        ext = _bisque_file_ext(res.get("name"))
        if ext:
            ext_counts[ext] = ext_counts.get(ext, 0) + 1

    media_label = (
        _bisque_media_label(resources[0].get("name"), resource_type) if resources else resource_type
    )
    parts = [f"Found {count} {media_label} asset(s)."]
    if scope_text:
        parts.append(f"Search scope: {scope_text}.")
    if names:
        parts.append(f"Top matches: {', '.join(names)}.")
    if ext_counts:
        ext_summary = ", ".join(
            f"{ext} ({cnt})"
            for ext, cnt in sorted(ext_counts.items(), key=lambda x: (-x[1], x[0]))[:4]
        )
        parts.append(f"File types: {ext_summary}.")
    return " ".join(parts)


def _bisque_resource_table(resources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for res in resources:
        name = res.get("name") or "(unnamed)"
        media = _bisque_media_label(name, res.get("resource_type"))
        rows.append(
            {
                "Name": name,
                "Type": media,
                "Created": _format_bisque_timestamp(res.get("created")),
                "ID": _short_bisque_id(res.get("uri")),
            }
        )
    return rows


def _bisque_metadata_table(metadata_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for meta in metadata_list:
        name = meta.get("name") or "(unnamed)"
        media = _bisque_media_label(name, meta.get("resource_type"))
        rows.append(
            {
                "Name": name,
                "Type": media,
                "Created": _format_bisque_timestamp(meta.get("created")),
                "ID": _short_bisque_id(meta.get("uri")),
                "Tags": len(meta.get("tags") or []),
            }
        )
    return rows


def _bisque_metadata_summary(metadata: dict[str, Any]) -> str:
    name = metadata.get("name") or "Resource"
    media = _bisque_media_label(name, metadata.get("resource_type"))
    created = _format_bisque_timestamp(metadata.get("created"))
    tag_count = len(metadata.get("tags") or [])
    parts = [f"{name} ({media})."]
    if created:
        parts.append(f"Created {created}.")
    if tag_count:
        parts.append(f"{tag_count} tag(s) attached.")
    return " ".join(parts)


def _bisque_metadata_key_values(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, key in (
        ("Name", "name"),
        ("Type", "resource_type"),
        ("Created", "created"),
        ("Owner", "owner"),
        ("URI", "uri"),
    ):
        value = metadata.get(key)
        if value:
            if key == "created":
                value = _format_bisque_timestamp(str(value))
            rows.append({"Field": label, "Value": value})

    dims = metadata.get("dimensions") or {}
    if dims:
        dim_parts = []
        for key in ("width", "height", "depth", "channels", "timepoints"):
            val = dims.get(key)
            if val is not None:
                dim_parts.append(f"{key}={val}")
        if dim_parts:
            rows.append({"Field": "Dimensions", "Value": ", ".join(dim_parts)})
    return rows


def _should_retry_as_image(
    raw_resource_type: str | None,
    text: str | list[str] | None,
    tag_query: str | None,
) -> bool:
    if not raw_resource_type:
        return False
    key = str(raw_resource_type).strip().lower()
    if key not in {"file", "files", "dataset", "datasets"}:
        return False
    hints = {
        "tif",
        "tiff",
        "png",
        "jpg",
        "jpeg",
        "bmp",
        "ome.tif",
        "ome.tiff",
        "czi",
        "nd2",
        "svs",
        "vsi",
        "lsm",
        "lif",
        "nii",
        "nii.gz",
        "nrrd",
        "mha",
        "mhd",
        "dcm",
        "dicom",
    }
    tokens: list[str] = []
    if isinstance(text, str):
        tokens.append(text)
    elif isinstance(text, list):
        tokens.extend([str(t) for t in text])
    if tag_query:
        tokens.append(tag_query)
    haystack = " ".join(tokens).lower()
    return any(hint in haystack for hint in hints)


_BISQUE_RESOURCE_TYPE_ALIASES: dict[str, str] = {
    "image": "image",
    "images": "image",
    "video": "image",
    "videos": "image",
    "movie": "image",
    "movies": "image",
    "mp4": "image",
    "mov": "image",
    "avi": "image",
    "mkv": "image",
    "webm": "image",
    "dataset": "dataset",
    "datasets": "dataset",
    "file": "file",
    "files": "file",
    "table": "table",
    "tables": "table",
    "hdf5": "table",
    "h5": "table",
    ".h5": "table",
    ".hdf5": "table",
    "dream3d": "table",
    ".dream3d": "table",
    "csv": "table",
    "tsv": "table",
    "spreadsheet": "table",
    "tabular": "table",
}

_BISQUE_HDF5_HINT_RE = re.compile(
    r"(^|[^a-z0-9])(?:hdf5|dream3d|h5|\.h5|\.hdf5|\.dream3d)(?=$|[^a-z0-9])",
    re.IGNORECASE,
)
_BISQUE_NL_CATALOG_CUE_RE = re.compile(
    r"\b(?:what|which|list|show me|recent(?:ly)?|latest|uploaded)\b",
    re.IGNORECASE,
)
_BISQUE_NL_FILETYPE_ALIASES: dict[str, tuple[str, ...]] = {
    "jpg": (".jpg", ".jpeg"),
    "jpeg": (".jpg", ".jpeg"),
    "png": (".png",),
    "tif": (".tif", ".tiff"),
    "tiff": (".tif", ".tiff"),
    "bmp": (".bmp",),
    "gif": (".gif",),
    "webp": (".webp",),
}


def _normalize_bisque_resource_type(resource_type: str | None) -> str:
    token = str(resource_type or "").strip().lower()
    if not token:
        return "image"
    return _BISQUE_RESOURCE_TYPE_ALIASES.get(token, token)


def _has_hdf5_search_hint(
    text: str | list[str] | None,
    tag_query: str | None,
    raw_resource_type: str | None = None,
) -> bool:
    raw_type = str(raw_resource_type or "").strip().lower()
    if raw_type in {"table", "tables", "hdf5", "h5", ".h5", ".hdf5", "dream3d", ".dream3d"}:
        return True
    tokens: list[str] = []
    if isinstance(text, str):
        tokens.append(text)
    elif isinstance(text, list):
        tokens.extend([str(item) for item in text])
    if tag_query:
        tokens.append(tag_query)
    haystack = " ".join(token for token in tokens if token).strip().lower()
    if not haystack:
        return False
    return bool(_BISQUE_HDF5_HINT_RE.search(haystack))


def _normalize_bisque_natural_language_filetype_terms(
    text: str | list[str] | None,
    *,
    tag_query: str | None = None,
) -> list[str] | None:
    if tag_query or not isinstance(text, str):
        return None
    normalized = re.sub(r"[^a-z0-9.\s_-]", " ", str(text or "").strip().lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized or not _BISQUE_NL_CATALOG_CUE_RE.search(normalized):
        return None

    matched_terms: list[str] = []
    for token, aliases in _BISQUE_NL_FILETYPE_ALIASES.items():
        if re.search(
            rf"(?<![a-z0-9])(?:\.{re.escape(token)}|{re.escape(token)})(?![a-z0-9])", normalized
        ):
            for alias in aliases:
                if alias not in matched_terms:
                    matched_terms.append(alias)
    return matched_terms or None


def _expand_bisque_text_terms_for_file_types(
    text: str | list[str] | None,
    tag_query: str | None,
    raw_resource_type: str | None = None,
) -> str | list[str] | None:
    natural_language_terms = _normalize_bisque_natural_language_filetype_terms(
        text,
        tag_query=tag_query,
    )
    if natural_language_terms:
        return natural_language_terms
    if not _has_hdf5_search_hint(text, tag_query, raw_resource_type=raw_resource_type):
        return text

    if isinstance(text, str):
        terms = [text]
    elif isinstance(text, list):
        terms = [str(item) for item in text if str(item or "").strip()]
    else:
        terms = []

    haystack = " ".join(terms).lower()
    if any(token in haystack for token in (".h5", ".hdf5", ".dream3d")):
        return text

    expanded = list(terms)
    for token in (".h5", ".hdf5", ".dream3d", "dream3d"):
        if token not in haystack:
            expanded.append(token)
    if not expanded:
        expanded = [".h5", ".hdf5", ".dream3d", "dream3d"]
    return expanded


def _bisque_search_resource_types(
    raw_resource_type: str | None,
    text: str | list[str] | None,
    tag_query: str | None,
) -> list[str]:
    normalized = _normalize_bisque_resource_type(raw_resource_type)
    candidates: list[str] = []

    def _add(token: str | None) -> None:
        value = _normalize_bisque_resource_type(token)
        if value and value not in candidates:
            candidates.append(value)

    if _has_hdf5_search_hint(text, tag_query, raw_resource_type=raw_resource_type):
        _add("table")
        if normalized != "table":
            _add(normalized)
        return candidates or ["table"]

    _add(normalized)
    if _should_retry_as_image(raw_resource_type, text, tag_query):
        _add("image")
    return candidates or ["image"]


def _session_supports_fetchxml(bq: Any) -> bool:
    session_dict = getattr(bq, "__dict__", None)
    if isinstance(session_dict, dict) and callable(session_dict.get("fetchxml")):
        return True
    if callable(getattr(type(bq), "fetchxml", None)):
        return True
    comm = session_dict.get("c") if isinstance(session_dict, dict) else getattr(bq, "c", None)
    return callable(getattr(comm, "fetch", None))


def _collect_bisque_resource_nodes(
    root_node: etree._Element,
    *,
    fallback_type: str,
) -> list[etree._Element]:
    nodes: list[etree._Element] = []

    def _maybe_add(node: etree._Element) -> None:
        tag = str(getattr(node, "tag", "") or "").strip().lower()
        if not tag or tag in {"response", "value", "tag"}:
            return
        if tag == "resource":
            child_elements = [
                child for child in node if isinstance(getattr(child, "tag", None), str)
            ]
            explicit_type = str(node.get("resource_type") or node.get("type") or "").strip()
            if child_elements or not explicit_type:
                return
        raw_uri = str(node.get("uri") or "").strip()
        raw_uniq = str(node.get("resource_uniq") or "").strip()
        if not raw_uri and not raw_uniq:
            return
        nodes.append(node)

    _maybe_add(root_node)
    for node in root_node.iter():
        if node is root_node:
            continue
        _maybe_add(node)

    deduped: list[etree._Element] = []
    seen: set[str] = set()
    for node in nodes:
        marker = str(node.get("uri") or node.get("resource_uniq") or id(node))
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append(node)
    return deduped


def _bisque_resource_row_from_xml_node(
    node: etree._Element,
    *,
    fallback_type: str,
    bisque_root: str,
) -> dict[str, Any]:
    tag_name = str(getattr(node, "tag", "") or "").strip().lower()
    resource_type = (
        tag_name
        if tag_name not in {"resource", "response"}
        else str(node.get("resource_type") or node.get("type") or fallback_type).strip().lower()
        or fallback_type
    )
    raw_uri = str(node.get("uri") or "").strip()
    resource_uniq = str(node.get("resource_uniq") or "").strip() or None
    uri_source = raw_uri or resource_uniq
    links = _build_bisque_resource_links(uri_source, bisque_root) if uri_source else {}
    normalized_uri = str(links.get("resource_uri") or raw_uri or "").strip() or None
    return {
        "uri": normalized_uri,
        "resource_uri": normalized_uri,
        "client_view_url": str(links.get("client_view_url") or "").strip() or None,
        "image_service_url": str(links.get("image_service_url") or "").strip() or None,
        "name": str(node.get("name") or node.get("value") or "").strip() or None,
        "resource_type": resource_type or fallback_type,
        "owner": str(node.get("owner") or "").strip() or None,
        "created": str(node.get("ts") or node.get("created") or "").strip() or None,
    }


def _filter_bisque_resource_rows(
    rows: list[dict[str, Any]],
    *,
    filter_terms: list[str] | None,
) -> list[dict[str, Any]]:
    normalized_terms = [
        str(term or "").strip().lower() for term in (filter_terms or []) if str(term or "").strip()
    ]
    if not normalized_terms:
        return rows
    filtered: list[dict[str, Any]] = []
    for row in rows:
        haystack = " ".join(
            str(row.get(field) or "").strip().lower()
            for field in ("name", "uri", "resource_uri", "client_view_url")
        )
        if any(term in haystack for term in normalized_terms):
            filtered.append(row)
    return filtered


def _query_bisque_resources_with_fallback(
    *,
    session: Any,
    candidate_type: str,
    query_params: dict[str, Any],
    bisque_root: str,
    collect_query_rows: Callable[[Any, str], list[dict[str, Any]]],
    fallback_filter_terms: list[str] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    query_error: Exception | None = None
    try:
        results = session.query(candidate_type, **query_params)
        collected = collect_query_rows(results, candidate_type)
    except Exception as exc:  # noqa: BLE001
        query_error = exc
        collected = []
        logger.warning(
            "BisQue query(%s) failed; falling back to data_service XML: %s", candidate_type, exc
        )
    else:
        if collected:
            return collected, "query"

    if not _session_supports_fetchxml(session):
        if query_error is not None:
            raise query_error
        return collected, "query"

    fetch_url = f"{bisque_root.rstrip('/')}/data_service/{candidate_type}"
    try:
        payload = _session_fetchxml_safe(session, fetch_url, **query_params)
        fallback_rows = [
            _bisque_resource_row_from_xml_node(
                node,
                fallback_type=candidate_type,
                bisque_root=bisque_root,
            )
            for node in _collect_bisque_resource_nodes(payload, fallback_type=candidate_type)
        ]
        if not fallback_rows and fallback_filter_terms and query_params.get("tag_query"):
            broad_params = {key: value for key, value in query_params.items() if key != "tag_query"}
            broad_limit = max(int(query_params.get("limit") or 0), 50)
            broad_params["limit"] = max(broad_limit, 200)
            broad_params["offset"] = 0
            broad_payload = _session_fetchxml_safe(session, fetch_url, **broad_params)
            broad_rows = [
                _bisque_resource_row_from_xml_node(
                    node,
                    fallback_type=candidate_type,
                    bisque_root=bisque_root,
                )
                for node in _collect_bisque_resource_nodes(
                    broad_payload, fallback_type=candidate_type
                )
            ]
            fallback_rows = _filter_bisque_resource_rows(
                broad_rows,
                filter_terms=fallback_filter_terms,
            )[: max(1, int(query_params.get("limit") or 10))]
            if fallback_rows:
                return fallback_rows, "fetchxml_local_filter"
        return fallback_rows, "fetchxml_fallback"
    except Exception as exc:  # noqa: BLE001
        if query_error is not None:
            logger.warning(
                "BisQue data_service fallback for %s also failed after query error: %s",
                candidate_type,
                exc,
            )
            raise query_error
        logger.warning("BisQue data_service fallback for %s failed: %s", candidate_type, exc)
        return collected, "query"


def bisque_download_resource(resource_uri: str, output_path: str) -> dict[str, Any]:
    """Download a BisQue resource blob to a local path."""
    try:
        from bqapi.comm import BQSession
    except ImportError:
        return {"success": False, "error": "bqapi package not installed"}

    try:
        bq = BQSession()
        (
            runtime_user,
            runtime_password,
            runtime_access_token,
            runtime_root,
            runtime_cookie_header,
        ) = _resolve_bisque_runtime_auth()
        root, _, _ = _init_bisque_session_with_runtime_auth(
            bq=bq,
            explicit_user=runtime_user,
            explicit_password=runtime_password,
            explicit_root=runtime_root,
        )
        normalized_uri = _normalize_bisque_resource_uri(resource_uri, root)
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        resource = bq.load(normalized_uri, view="deep")
        blob_uri = getattr(resource, "blob", None) if resource is not None else None
        if blob_uri is None:
            blob_uri = f"{normalized_uri}/blob"
        links = _build_bisque_resource_links(normalized_uri, root)
        image_service_url = str(links.get("image_service_url") or "").strip()

        requested_name = str(getattr(resource, "name", "") or "").strip()
        if not requested_name:
            requested_name = target.name

        def _looks_like_binary_target() -> bool:
            binary_exts = {
                ".png",
                ".jpg",
                ".jpeg",
                ".tif",
                ".tiff",
                ".bmp",
                ".gif",
                ".webp",
                ".nii",
                ".nrrd",
                ".mha",
                ".mhd",
                ".h5",
                ".hdf5",
                ".zip",
                ".gz",
                ".tar",
                ".pt",
                ".pth",
                ".npy",
                ".npz",
                ".bin",
            }
            target_ext = target.suffix.lower()
            name_ext = Path(requested_name).suffix.lower()
            return target_ext in binary_exts or name_ext in binary_exts

        def _download_http_fallback(url: str) -> tuple[bool, str | None]:
            normalized_url = str(url or "").strip()
            if not normalized_url:
                return False, "empty fallback URL"

            headers: dict[str, str] = {}
            auth: tuple[str, str] | None = None
            timeout = 30.0
            if runtime_cookie_header:
                headers["Cookie"] = str(runtime_cookie_header)
            elif runtime_access_token:
                headers["Authorization"] = f"Bearer {runtime_access_token}"
            elif runtime_user and runtime_password:
                auth = (str(runtime_user), str(runtime_password))

            try:
                response = httpx.get(
                    normalized_url,
                    headers=headers or None,
                    auth=auth,
                    timeout=timeout,
                    follow_redirects=True,
                )
                if response.status_code >= 400:
                    return False, f"HTTP {response.status_code}"
                target.write_bytes(response.content)
                return True, None
            except Exception as exc:  # noqa: BLE001
                return False, str(exc)

        bq.fetchblob(blob_uri, path=str(target))
        signature = _checkpoint_file_signature(target)
        if signature.get("looks_html_or_xml"):
            fallback_attempts: list[str] = []
            fallback_urls: list[str] = []
            if image_service_url:
                fallback_urls.append(image_service_url)
            fallback_urls.append(str(blob_uri))

            seen_urls: set[str] = set()
            for url in fallback_urls:
                normalized_url = str(url or "").strip()
                if not normalized_url:
                    continue
                dedupe = normalized_url.lower()
                if dedupe in seen_urls:
                    continue
                seen_urls.add(dedupe)
                ok, reason = _download_http_fallback(normalized_url)
                status_text = "ok" if ok else f"failed ({reason or 'unknown'})"
                fallback_attempts.append(f"{normalized_url} -> {status_text}")
                if not ok:
                    continue
                signature = _checkpoint_file_signature(target)
                if not signature.get("looks_html_or_xml"):
                    break

            if signature.get("looks_html_or_xml") and _looks_like_binary_target():
                hint = (
                    "BisQue may have returned an XML metadata document instead of file bytes. "
                    "Verify permissions for this resource and retry."
                )
                if target.suffix.lower() == ".pt":
                    hint = (
                        "BisQue may have returned an error/login page instead of blob bytes. "
                        "Verify credentials and permissions for this resource, then retry."
                    )
                return {
                    "success": False,
                    "error": ("Downloaded file appears to be XML/HTML instead of binary content."),
                    "resource_uri": normalized_uri,
                    "blob_uri": blob_uri,
                    "image_service_url": image_service_url or None,
                    "output_path": str(target),
                    "signature": signature,
                    "fallback_attempts": fallback_attempts,
                    "hint": hint,
                }

        if target.suffix.lower() == ".pt" and signature.get("looks_html_or_xml"):
            return {
                "success": False,
                "error": "Downloaded file is not a valid .pt checkpoint (received HTML/XML content).",
                "resource_uri": normalized_uri,
                "blob_uri": blob_uri,
                "output_path": str(target),
                "hint": (
                    "BisQue may have returned an error/login page instead of blob bytes. "
                    "Verify credentials and permissions for this resource, then retry."
                ),
            }
        ui_artifacts = [
            {
                "type": "summary",
                "title": "Download complete",
                "payload": f"Saved `{target}` from `{normalized_uri}`.",
            }
        ]
        return {
            "success": True,
            "resource_uri": normalized_uri,
            "blob_uri": blob_uri,
            "output_path": str(target),
            "size_bytes": int(signature.get("size_bytes") or target.stat().st_size),
            "ui_artifacts": ui_artifacts,
        }
    except Exception as e:
        logger.error(f"BisQue download failed: {e}")
        return {"success": False, "error": str(e)}


def bisque_find_assets(
    resource_type: str = "image",
    tag_query: str | None = None,
    tag_filters: dict[str, Any] | list[dict[str, Any]] | None = None,
    metadata_filters: dict[str, Any] | list[dict[str, Any]] | None = None,
    text: str | None = None,
    limit: int = 5,
    offset: int = 0,
    order_by: str | None = "created",
    order: str = "desc",
    include_metadata: bool = True,
    max_metadata: int = 3,
    download: bool = False,
    download_dir: str = "data/bisque_downloads",
) -> dict[str, Any]:
    """
    Composite BisQue tool: search assets, optionally fetch metadata, optionally download.

    This chains search_bisque_resources -> load_bisque_resource -> bisque_download_resource
    into a single, reliable call for scientific workflows.
    """
    search_result = search_bisque_resources(
        resource_type=resource_type,
        tag_query=tag_query,
        tag_filters=tag_filters,
        metadata_filters=metadata_filters,
        text=text,
        limit=limit,
        offset=offset,
        order_by=order_by,
        order=order,
    )
    if not search_result.get("success"):
        return search_result

    resources = search_result.get("resources") or []
    metadata_results: list[dict[str, Any]] = []
    download_results: list[dict[str, Any]] = []

    def _resource_basename(uri: str | None, name: str | None) -> str:
        if name:
            return name
        if not uri:
            return "bisque_resource"
        return uri.rstrip("/").split("/")[-1]

    if include_metadata:
        for resource in resources[: max(1, int(max_metadata))]:
            uri = resource.get("uri") if isinstance(resource, dict) else None
            if not uri:
                continue
            meta = load_bisque_resource(uri, view="deep")
            if meta.get("success"):
                metadata_results.append(meta.get("resource") or {})

    if download:
        Path(download_dir).mkdir(parents=True, exist_ok=True)
        for resource in resources[: max(1, int(max_metadata))]:
            uri = resource.get("uri") if isinstance(resource, dict) else None
            if not uri:
                continue
            name = resource.get("name") if isinstance(resource, dict) else None
            file_name = _resource_basename(uri, name)
            output_path = str(Path(download_dir) / file_name)
            dl = bisque_download_resource(uri, output_path)
            download_results.append(dl)

    ui_artifacts = []
    if search_result.get("ui_artifacts"):
        ui_artifacts.extend(search_result.get("ui_artifacts", []))
    if metadata_results:
        ui_artifacts.append(
            {
                "type": "summary",
                "title": "Metadata retrieved",
                "payload": f"Loaded metadata for {len(metadata_results)} asset(s).",
            }
        )
        ui_artifacts.append(
            {
                "type": "table",
                "title": "Metadata (sample)",
                "payload": _bisque_metadata_table(metadata_results[:10]),
            }
        )
    if download_results:
        ui_artifacts.append(
            {
                "type": "table",
                "title": "Downloads",
                "payload": [
                    {
                        "Resource": _short_bisque_id(d.get("resource_uri")),
                        "Saved to": d.get("output_path"),
                        "Status": "ok" if d.get("success") else "failed",
                        "Error": d.get("error"),
                    }
                    for d in download_results
                ],
            }
        )

    return {
        "success": True,
        "query": search_result.get("query"),
        "count": search_result.get("count"),
        "resources": resources,
        "metadata": metadata_results,
        "downloads": download_results,
        "ui_artifacts": ui_artifacts,
    }


def search_bisque_resources(
    resource_type: str = "image",
    tag_query: str | None = None,
    tag_filters: dict[str, Any] | list[dict[str, Any]] | None = None,
    metadata_filters: dict[str, Any] | list[dict[str, Any]] | None = None,
    text: str | list[str] | None = None,
    limit: int = 10,
    offset: int = 0,
    order_by: str | None = "created",
    order: str = "desc",
) -> dict[str, Any]:
    """
    Search for resources in BisQue repository.

    Args:
        resource_type: Type of resource to search for (image, dataset, file, table, etc.).
            HDF5/DREAM3D aliases such as hdf5, h5, and dream3d are normalized to table.
        tag_query: BisQue tag_query string (e.g., "antibody:*GFP* AND image_num_z:>=160")
        tag_filters: Structured tag filters (supports operator prefixes like ">=160")
        metadata_filters: Convenience aliases for image metadata (z/t/c/x/y)
        text: Free-text search term (fuzzy match)
        limit: Maximum number of results to return
        offset: Number of results to skip (for pagination)

    Returns:
        Dictionary with search results and metadata
    """
    try:
        from bqapi.comm import BQSession
    except ImportError:
        return {"success": False, "error": "bqapi package not installed"}

    try:
        # Initialize session
        bq = BQSession()
        root, auth_mode, _ = _init_bisque_session_with_runtime_auth(bq=bq)
        logger.info("Searching BisQue for %s resources", resource_type)

        raw_resource_type = resource_type
        original_text = text
        original_tag_query = tag_query
        original_tag_filters = _coerce_bisque_filter_mapping(tag_filters)
        original_metadata_filters = _coerce_bisque_filter_mapping(metadata_filters)
        resource_type = _normalize_bisque_resource_type(resource_type)
        if hasattr(bq, "normalize_resource_type"):
            resource_type = _normalize_bisque_resource_type(
                bq.normalize_resource_type(resource_type)
            )

        # Seed a video hint when asked for videos but no text/filter provided.
        if (
            raw_resource_type
            and str(raw_resource_type).strip().lower()
            in {"video", "movie", "mp4", "mov", "avi", "mkv", "webm"}
            and not text
            and not tag_query
            and not tag_filters
            and not metadata_filters
        ):
            text = ["mp4", "mov", "avi", "mkv", "webm", "mpeg", "mpg"]

        text = _expand_bisque_text_terms_for_file_types(
            text,
            tag_query,
            raw_resource_type=raw_resource_type,
        )

        normalized_order = str(order or "desc").strip().lower()
        if normalized_order not in {"asc", "desc"}:
            normalized_order = "desc"

        normalized_order_by = str(order_by or "").strip().lower() or None
        if normalized_order_by in {"created_at", "created", "ts", "timestamp"}:
            normalized_order_by = "created"
        elif normalized_order_by in {"name", "owner"}:
            normalized_order_by = normalized_order_by
        elif normalized_order_by:
            normalized_order_by = None

        # Build query parameters
        query_params = {"limit": limit, "offset": offset}

        expanded_filters = _expand_bisque_tag_filters(tag_filters)
        expanded_meta = _expand_bisque_tag_filters(metadata_filters)
        if expanded_meta:
            for key, value in expanded_meta.items():
                expanded_filters = expanded_filters or {}
                expanded_filters[key] = _merge_filter_value(expanded_filters.get(key), value)

        normalized_tag_query = _rewrite_tag_query_aliases(tag_query)
        combined_tag_query = _build_tag_query_compat(
            bq,
            tag_query=normalized_tag_query,
            tag_filters=expanded_filters,
            text=text,
        )
        if combined_tag_query:
            query_params["tag_query"] = combined_tag_query
        if normalized_order_by:
            order_key = normalized_order_by
            if order_key in {"name", "created", "ts", "owner"}:
                order_key = f"@{order_key}"
            query_params["tag_order"] = f"{order_key}:{normalized_order}"

        def _collect_resources(results_iter, fallback_type: str) -> list[dict[str, Any]]:
            collected: list[dict[str, Any]] = []
            for resource in results_iter:
                links = _build_bisque_resource_links(
                    getattr(resource, "uri", None),
                    root,
                )
                collected.append(
                    {
                        "uri": resource.uri if hasattr(resource, "uri") else None,
                        "resource_uri": (
                            str(
                                links.get("resource_uri") or getattr(resource, "uri", None) or ""
                            ).strip()
                            or None
                        ),
                        "client_view_url": str(links.get("client_view_url") or "").strip() or None,
                        "image_service_url": str(links.get("image_service_url") or "").strip()
                        or None,
                        "name": resource.name if hasattr(resource, "name") else None,
                        "resource_type": (
                            resource.tag
                            if hasattr(resource, "tag")
                            else getattr(resource, "resource_type", None) or fallback_type
                        ),
                        "owner": resource.owner if hasattr(resource, "owner") else None,
                        "created": str(resource.ts) if hasattr(resource, "ts") else None,
                    }
                )
            return collected

        requested_dataset_name = (
            _extract_requested_dataset_name_for_search(
                text=original_text,
                tag_query=original_tag_query,
                tag_filters=original_tag_filters,
            )
            if resource_type == "dataset"
            else None
        )
        fallback_filter_terms: list[str] = []
        if isinstance(original_text, str) and str(original_text or "").strip():
            fallback_filter_terms.append(str(original_text).strip())
        elif isinstance(original_text, list):
            fallback_filter_terms.extend(
                str(item or "").strip() for item in original_text if str(item or "").strip()
            )
        if requested_dataset_name and requested_dataset_name not in fallback_filter_terms:
            fallback_filter_terms.append(requested_dataset_name)

        def _execute_search_with_session(
            session: Any,
            *,
            bisque_root: str,
        ) -> tuple[list[dict[str, Any]], list[str], str, dict[str, Any] | None, str]:
            resolved_resources: list[dict[str, Any]] = []
            types_tried: list[str] = []
            resolved_resource_type = resource_type
            resolution_fallback: dict[str, Any] | None = None
            search_backend = "query"
            for candidate_type in _bisque_search_resource_types(raw_resource_type, text, tag_query):
                types_tried.append(candidate_type)
                resolved_resources, search_backend = _query_bisque_resources_with_fallback(
                    session=session,
                    candidate_type=candidate_type,
                    query_params=query_params,
                    bisque_root=bisque_root,
                    collect_query_rows=_collect_resources,
                    fallback_filter_terms=fallback_filter_terms,
                )
                resolved_resource_type = candidate_type
                if resolved_resources:
                    break

            if (
                not resolved_resources
                and resolved_resource_type == "dataset"
                and requested_dataset_name
            ):
                resolved_dataset = _resolve_bisque_dataset_target_with_session(
                    bq=session,
                    bisque_root=bisque_root,
                    dataset_name=requested_dataset_name,
                )
                if isinstance(resolved_dataset, dict):
                    resolution_fallback = {
                        "dataset_name": requested_dataset_name,
                        "status": str(resolved_dataset.get("status") or "").strip() or "unknown",
                    }

                    def _dataset_result_row(item: dict[str, Any]) -> dict[str, Any]:
                        dataset_uri = str(item.get("dataset_uri") or "").strip() or None
                        links = (
                            _build_bisque_resource_links(dataset_uri, bisque_root)
                            if dataset_uri
                            else {}
                        )
                        return {
                            "uri": dataset_uri,
                            "resource_uri": dataset_uri,
                            "client_view_url": str(links.get("client_view_url") or "").strip()
                            or None,
                            "image_service_url": None,
                            "name": str(
                                item.get("dataset_name") or requested_dataset_name or ""
                            ).strip()
                            or None,
                            "resource_type": "dataset",
                            "owner": None,
                            "created": str(item.get("created") or "").strip() or None,
                        }

                    if str(resolved_dataset.get("status") or "").strip().lower() == "ambiguous":
                        resolved_resources = [
                            _dataset_result_row(item)
                            for item in list(resolved_dataset.get("candidate_datasets") or [])
                            if isinstance(item, dict)
                        ]
                    elif str(resolved_dataset.get("dataset_uri") or "").strip():
                        resolved_resources = [_dataset_result_row(resolved_dataset)]
            return (
                resolved_resources,
                types_tried,
                resolved_resource_type,
                resolution_fallback,
                search_backend,
            )

        # Execute query with file-type-aware fallback ordering.
        (
            resources,
            resource_types_tried,
            resource_type,
            dataset_resolution_fallback,
            search_backend,
        ) = _execute_search_with_session(bq, bisque_root=root)
        auth_retry_used = False
        if not resources and auth_mode != "basic":
            runtime_user, runtime_password, _, _, _ = _resolve_bisque_runtime_auth()
            if runtime_user and runtime_password:
                try:
                    basic_bq = BQSession()
                    basic_root, basic_auth_mode, _ = _init_bisque_session_with_runtime_auth(
                        bq=basic_bq,
                        preferred_auth_mode="basic",
                    )
                    if basic_auth_mode == "basic":
                        (
                            resources,
                            resource_types_tried,
                            resource_type,
                            dataset_resolution_fallback,
                            search_backend,
                        ) = _execute_search_with_session(basic_bq, bisque_root=basic_root)
                        root = basic_root
                        auth_mode = basic_auth_mode
                        auth_retry_used = True
                except Exception as auth_retry_exc:
                    logger.warning("BisQue search basic-auth retry failed: %s", auth_retry_exc)

        logger.info(f"Found {len(resources)} {resource_type} resources")

        ui_artifacts = [
            {
                "type": "summary",
                "title": "Search summary",
                "payload": _bisque_search_summary(
                    resources,
                    resource_type,
                    query={
                        "requested_resource_type": raw_resource_type,
                        "resource_type": resource_type,
                        "text": original_text,
                        "original_tag_query": original_tag_query,
                        "tag_filters": original_tag_filters,
                        "metadata_filters": original_metadata_filters,
                        "order_by": normalized_order_by,
                        "order": normalized_order,
                    },
                ),
            }
        ]
        if resources:
            ui_artifacts.append(
                {
                    "type": "metrics",
                    "title": "Search results",
                    "payload": {"matches": len(resources)},
                }
            )
            ui_artifacts.append(
                {
                    "type": "table",
                    "title": "Matches",
                    "payload": _bisque_resource_table(resources[:200]),
                }
            )

        return {
            "success": True,
            "count": len(resources),
            "resources": resources,
            "query": {
                "requested_resource_type": raw_resource_type,
                "normalized_resource_type": _normalize_bisque_resource_type(raw_resource_type),
                "resource_type": resource_type,
                "resource_types_tried": resource_types_tried,
                "text": original_text,
                "text_terms": text,
                "original_tag_query": original_tag_query,
                "tag_query": query_params.get("tag_query"),
                "tag_filters": original_tag_filters or {},
                "metadata_filters": original_metadata_filters or {},
                "expanded_tag_filters": expanded_filters or {},
                "dataset_resolution_fallback": dataset_resolution_fallback,
                "search_backend": search_backend,
                "auth_mode": auth_mode,
                "auth_retry_used": auth_retry_used,
                "limit": limit,
                "offset": offset,
                "order_by": normalized_order_by,
                "order": normalized_order,
            },
            "ui_artifacts": ui_artifacts,
        }

    except Exception as e:
        logger.error(f"BisQue search failed: {str(e)}")
        return {"success": False, "error": str(e)}


def load_bisque_resource(resource_uri: str, view: str = "deep") -> dict[str, Any]:
    """
    Load a specific resource from BisQue by URI.

    Args:
        resource_uri: Full URI of the resource to load
        view: Level of detail to load (short, full, deep)

    Returns:
        Dictionary with resource details
    """
    try:
        from bqapi.comm import BQSession
    except ImportError:
        return {"success": False, "error": "bqapi package not installed"}

    try:
        bq = BQSession()
        root, _, _ = _init_bisque_session_with_runtime_auth(bq=bq)
        normalized_uri = _normalize_bisque_resource_uri(resource_uri, root)
        logger.info(f"Loading resource from BisQue: {normalized_uri}")

        resource, resource_xml, load_error = _load_bisque_resource_with_probe(
            bq=bq,
            normalized_uri=normalized_uri,
            view=view,
        )

        if resource is None and resource_xml is None:
            return {
                "success": False,
                "error": (
                    str(load_error)
                    if load_error is not None
                    else f"Resource not found: {resource_uri}"
                ),
            }

        # Extract basic metadata
        metadata = {
            "uri": (
                resource.uri
                if resource is not None and hasattr(resource, "uri")
                else str(
                    resource_xml.get("uri") if resource_xml is not None else resource_uri or ""
                )
            ),
            "name": (
                resource.name
                if resource is not None and hasattr(resource, "name")
                else (resource_xml.get("name") if resource_xml is not None else None)
            ),
            "resource_type": (
                _loaded_bisque_resource_type(resource=resource, resource_xml=resource_xml)
            ),
            "owner": (
                resource.owner
                if resource is not None and hasattr(resource, "owner")
                else (resource_xml.get("owner") if resource_xml is not None else None)
            ),
            "created": (
                str(resource.ts)
                if resource is not None and hasattr(resource, "ts")
                else (str(resource_xml.get("ts") or resource_xml.get("created") or "") or None)
            ),
        }

        # Extract tags if available
        if resource is not None and hasattr(resource, "tags"):
            metadata["tags"] = [
                {
                    "name": tag.name if hasattr(tag, "name") else None,
                    "value": tag.value if hasattr(tag, "value") else None,
                }
                for tag in resource.tags
            ]

        # Extract dimensions for images
        if resource is not None and hasattr(resource, "image_num_x"):
            metadata["dimensions"] = {
                "width": resource.image_num_x,
                "height": resource.image_num_y,
                "depth": getattr(resource, "image_num_z", None),
                "channels": getattr(resource, "image_num_c", None),
                "timepoints": getattr(resource, "image_num_t", None),
            }

        logger.info(f"Successfully loaded resource: {metadata.get('name', resource_uri)}")

        ui_artifacts = [
            {
                "type": "summary",
                "title": "Resource overview",
                "payload": _bisque_metadata_summary(metadata),
            },
            {
                "type": "key_value",
                "title": "Key details",
                "payload": _bisque_metadata_key_values(metadata),
            },
        ]
        if metadata.get("tags"):
            ui_artifacts.append(
                {
                    "type": "table",
                    "title": "Tags",
                    "payload": [
                        {
                            "Name": tag.get("name"),
                            "Value": tag.get("value"),
                        }
                        for tag in metadata.get("tags")
                    ],
                }
            )

        links = _build_bisque_resource_links(metadata.get("uri"), root)

        return {
            "success": True,
            "resource": metadata,
            "view_url": links.get("client_view_url"),
            "client_view_url": links.get("client_view_url"),
            "image_service_url": links.get("image_service_url"),
            "ui_artifacts": ui_artifacts,
        }

    except Exception as e:
        logger.error(f"Failed to load resource: {str(e)}")
        return {"success": False, "error": str(e)}


def _load_bisque_resource_with_probe(
    *,
    bq: Any,
    normalized_uri: str,
    view: str = "deep",
    cache: str | None = "false",
) -> tuple[Any | None, etree._Element | None, Exception | None]:
    """
    Load a BisQue resource and distinguish a true missing resource from a swallowed
    auth failure. `bq.load(...)` can return None when the underlying request was
    actually rejected (for example 403 Forbidden), so we probe with fetchxml before
    concluding that the resource does not exist.
    """
    try:
        resource = bq.load(normalized_uri, view=view)
    except Exception as exc:  # noqa: BLE001
        return None, None, exc
    if resource is not None:
        return resource, None, None

    fetch_params: dict[str, Any] = {"view": view}
    if cache is not None:
        fetch_params["cache"] = cache
    try:
        fetched = _session_fetchxml_safe(bq, normalized_uri, **fetch_params)
    except Exception as exc:  # noqa: BLE001
        return None, None, exc
    return None, _unwrap_bisque_resource_document(fetched), None


def delete_bisque_resource(resource_uri: str) -> dict[str, Any]:
    """
    Delete a resource from BisQue.

    Args:
        resource_uri: Full URI of the resource to delete

    Returns:
        Dictionary with deletion status
    """
    try:
        from bqapi.comm import BQSession
    except ImportError:
        return {"success": False, "error": "bqapi package not installed"}

    try:

        def _run_delete(
            *,
            active_bq: Any,
            active_root: str,
            normalized_uri: str,
        ) -> dict[str, Any]:
            resource, resource_xml, load_error = _load_bisque_resource_with_probe(
                bq=active_bq,
                normalized_uri=normalized_uri,
                view="deep",
            )

            if resource is None and resource_xml is None:
                if load_error is not None:
                    raise load_error
                return {"success": False, "error": f"Resource not found: {resource_uri}"}

            resource_name = (
                resource.name
                if resource is not None and hasattr(resource, "name")
                else str(
                    resource_xml.get("name") if resource_xml is not None else resource_uri or ""
                )
            )

            cleanup_result = _remove_resource_from_all_bisque_datasets(
                bq=active_bq,
                bisque_root=active_root,
                resource_uri=normalized_uri,
            )
            cleanup_ok = int(cleanup_result.get("removed") or 0) >= int(
                cleanup_result.get("found") or 0
            )
            if not cleanup_ok:
                deleted_client_view_url = _bisque_user_facing_resource_url(
                    normalized_uri, active_root
                )
                return {
                    "success": False,
                    "deleted_uri": normalized_uri,
                    "deleted_client_view_url": deleted_client_view_url,
                    "client_view_url": deleted_client_view_url,
                    "resource_name": resource_name,
                    "deletion_verified": False,
                    "deletion_verification_attempts": 0,
                    "dataset_cleanup": cleanup_result,
                    "error": (
                        "BisQue delete was not attempted because one or more dataset memberships "
                        "still referenced the resource."
                    ),
                }

            delete_response = active_bq.deletexml(normalized_uri)
            deleted_ok = bool(getattr(delete_response, "ok", True))
            if not deleted_ok:
                raise ValueError(
                    f"BisQue delete failed for {normalized_uri}: "
                    f"{getattr(delete_response, 'status_code', 'unknown status')}"
                )

            _deleted_xml, deletion_verified, deletion_attempts = _wait_for_bisque_resource_state(
                bq=active_bq,
                resource_uri=normalized_uri,
                predicate=lambda _xml, exc: exc is not None
                and (_is_bisque_missing_resource_error(exc) or _is_bisque_forbidden_error(exc)),
            )

            deleted_and_clean = deletion_verified and cleanup_ok

            logger.info(f"Successfully deleted resource: {resource_name}")

            deleted_client_view_url = _bisque_user_facing_resource_url(normalized_uri, active_root)
            return {
                "success": deleted_and_clean,
                "deleted_uri": normalized_uri,
                "deleted_client_view_url": deleted_client_view_url,
                "client_view_url": deleted_client_view_url,
                "resource_name": resource_name,
                "deletion_verified": deletion_verified,
                "deletion_verification_attempts": deletion_attempts,
                "dataset_cleanup": cleanup_result,
                **(
                    {}
                    if deleted_and_clean
                    else {
                        "error": (
                            "BisQue delete request completed, but the resource did not fully disappear "
                            "or one or more dataset memberships still referenced it."
                        )
                    }
                ),
            }

        def _is_bisque_forbidden_error(exc: Exception | str | None) -> bool:
            message = str(exc or "").strip().lower()
            if not message:
                return False
            if "status=403" in message or " status 403" in message or "http 403" in message:
                return True
            if "403" in message and any(
                marker in message
                for marker in (
                    "forbidden",
                    "permission denied",
                    "not authorized",
                    "unauthorized to",
                )
            ):
                return True
            return False

        def _should_retry_with_cookie_fallback(exc: Exception) -> bool:
            if not _is_bisque_forbidden_error(exc):
                return False
            return _get_bisque_cookie_header() is not None

        def _try_upload_lineage_fallback(
            *,
            normalized_uri: str,
        ) -> dict[str, Any] | None:
            session_material = _lookup_upload_session_auth_material_for_resource(normalized_uri)
            if session_material is None:
                return None
            username, password, access_token, lineage_root, cookie_header = session_material
            active_root = lineage_root or root
            if not active_root:
                return None
            username, password, access_token, cookie_header = _apply_bisque_auth_preference(
                username=username,
                password=password,
                access_token=access_token,
                cookie_header=cookie_header,
                preferred_auth_mode="basic",
            )
            lineage_bq = BQSession()
            token_mode, lineage_auth_mode = _init_bq_session(
                lineage_bq,
                username=username,
                password=password,
                access_token=access_token,
                bisque_root=active_root,
                cookie_header=cookie_header,
            )
            return {
                "bq": lineage_bq,
                "root": active_root,
                "auth_mode": lineage_auth_mode,
                "token_mode": token_mode,
            }

        bq = BQSession()
        root, auth_mode, _ = _init_bisque_session_with_runtime_auth(
            bq=bq,
            preferred_auth_mode="auto",
        )
        normalized_uri = _normalize_bisque_resource_uri(resource_uri, root)
        logger.info(f"Deleting resource from BisQue: {normalized_uri}")

        try:
            return _run_delete(active_bq=bq, active_root=root, normalized_uri=normalized_uri)
        except Exception as exc:
            if not _is_bisque_forbidden_error(exc):
                raise

            if auth_mode != "basic":
                logger.warning(
                    "BisQue delete hit a 403 under %s auth; retrying with basic auth when credentials are available.",
                    auth_mode,
                )
                try:
                    basic_bq = BQSession()
                    basic_root, basic_auth_mode, _ = _init_bisque_session_with_runtime_auth(
                        bq=basic_bq,
                        explicit_root=root,
                        preferred_auth_mode="basic",
                    )
                    if basic_auth_mode == "basic":
                        return _run_delete(
                            active_bq=basic_bq,
                            active_root=basic_root,
                            normalized_uri=_normalize_bisque_resource_uri(resource_uri, basic_root),
                        )
                except Exception as basic_exc:
                    logger.warning("BisQue delete basic-auth retry failed: %s", basic_exc)

            if auth_mode != "cookie" and _should_retry_with_cookie_fallback(exc):
                logger.warning(
                    "BisQue delete hit a 403 under %s auth; retrying with request cookie session.",
                    auth_mode,
                )
                try:
                    cookie_bq = BQSession()
                    cookie_root, cookie_auth_mode, _ = _init_bisque_session_with_runtime_auth(
                        bq=cookie_bq,
                        explicit_root=root,
                        preferred_auth_mode="cookie",
                    )
                    if cookie_auth_mode == "cookie":
                        return _run_delete(
                            active_bq=cookie_bq,
                            active_root=cookie_root,
                            normalized_uri=_normalize_bisque_resource_uri(
                                resource_uri, cookie_root
                            ),
                        )
                except Exception as cookie_exc:
                    logger.warning("BisQue delete cookie-session retry failed: %s", cookie_exc)

            lineage_session = _try_upload_lineage_fallback(normalized_uri=normalized_uri)
            if lineage_session is not None:
                logger.warning(
                    "BisQue delete is retrying with upload-lineage auth mode %s for %s.",
                    lineage_session.get("auth_mode"),
                    normalized_uri,
                )
                return _run_delete(
                    active_bq=lineage_session["bq"],
                    active_root=str(lineage_session["root"] or root),
                    normalized_uri=_normalize_bisque_resource_uri(
                        resource_uri,
                        str(lineage_session["root"] or root),
                    ),
                )
            raise
    except Exception as e:
        logger.error(f"Failed to delete resource: {str(e)}")
        return {"success": False, "error": str(e)}


def _normalize_gobject_vertices(vertices: Any) -> list[Any]:
    if not isinstance(vertices, list):
        return []
    if not vertices:
        return []
    if all(isinstance(item, (int, float)) for item in vertices):
        if len(vertices) % 2 != 0:
            return []
        normalized_pairs: list[list[float]] = []
        for index in range(0, len(vertices), 2):
            normalized_pairs.append([float(vertices[index]), float(vertices[index + 1])])
        return normalized_pairs
    return vertices


def _bisque_gobject_xml_tag(requested_type: str, vertices: list[Any]) -> str:
    token = str(requested_type or "").strip().lower()
    vertex_count = len(vertices)
    if token in {"point", "label"}:
        return token
    if token in {"polyline", "line"}:
        return "polyline"
    if token in {"circle", "ellipse"}:
        return token
    if token in {"rectangle", "square"} and vertex_count <= 2:
        return token
    if token in {"polygon", "rectangle", "square"} and vertex_count >= 3:
        return "polygon"
    return "gobject"


def _build_bisque_gobject_request(
    gobjects: list[dict[str, Any]],
) -> tuple[etree._Element, list[dict[str, Any]]]:
    request = etree.Element("request")
    added: list[dict[str, Any]] = []

    for gobj in gobjects:
        requested_type = str(gobj.get("type") or "point").strip().lower()
        name = str(gobj.get("name") or "").strip() or None
        value = str(gobj.get("value") or "").strip() or None
        vertices = _normalize_gobject_vertices(gobj.get("vertices") or [])
        xml_tag = _bisque_gobject_xml_tag(requested_type, vertices)

        attrs: dict[str, str] = {}
        if name is not None:
            attrs["name"] = name
        if value is not None:
            attrs["value"] = value
        if xml_tag == "gobject":
            attrs["type"] = requested_type or "point"

        node = etree.SubElement(request, xml_tag, **attrs)

        vertex_count = 0
        for idx, vertex in enumerate(vertices):
            if isinstance(vertex, dict):
                x = vertex.get("x")
                y = vertex.get("y")
                z = vertex.get("z", 0)
                t = vertex.get("t", 0)
            elif isinstance(vertex, (list, tuple)) and len(vertex) >= 2:
                x = vertex[0]
                y = vertex[1]
                z = vertex[2] if len(vertex) > 2 else 0
                t = vertex[3] if len(vertex) > 3 else 0
            else:
                continue
            if x is None or y is None:
                continue
            etree.SubElement(
                node,
                "vertex",
                index=str(idx),
                x=str(float(x)),
                y=str(float(y)),
                z=str(float(z or 0)),
                t=str(float(t or 0)),
            )
            vertex_count += 1

        for tag in gobj.get("tags") or []:
            if isinstance(tag, dict) and tag.get("name") and tag.get("value") is not None:
                tag_attrs = {
                    "name": str(tag["name"]),
                    "value": str(tag["value"]),
                }
                tag_type = str(tag.get("type") or "").strip()
                if tag_type:
                    tag_attrs["type"] = tag_type
                etree.SubElement(node, "tag", **tag_attrs)

        added.append(
            {
                "type": requested_type,
                "xml_tag": xml_tag,
                "name": name,
                "vertices": vertex_count,
            }
        )

    return request, added


def bioio_load_image(
    file_path: str,
    scene: int | str | None = None,
    use_aicspylibczi: bool = False,
    array_mode: str = "plane",
    t_index: int | None = None,
    c_index: int | None = None,
    z_index: int | None = None,
    save_array: bool = True,
    include_array: bool = False,
    max_inline_elements: int = 16384,
) -> dict[str, Any]:
    """Load scientific image data via the universal bioio-backed data layer."""
    result = load_scientific_image(
        file_path=file_path,
        scene=scene,
        use_aicspylibczi=bool(use_aicspylibczi),
        array_mode="plane" if array_mode not in {"plane", "volume", "tczyx"} else array_mode,  # type: ignore[arg-type]
        t_index=t_index,
        c_index=c_index,
        z_index=z_index,
        save_array=bool(save_array),
        include_array=bool(include_array),
        max_inline_elements=int(max_inline_elements),
        return_array=False,
    )
    if not result.get("success"):
        return result

    artifacts: list[dict[str, Any]] = []
    preview_path = result.get("preview_path")
    if preview_path and Path(str(preview_path)).exists():
        artifacts.append(
            {
                "type": "image",
                "title": "Scientific preview",
                "path": str(preview_path),
            }
        )

    result["ui_artifacts"] = artifacts
    return result


def _normalize_uint8(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.zeros(x.shape, dtype=np.uint8)
    lo = float(np.percentile(finite, 1.0))
    hi = float(np.percentile(finite, 99.0))
    if hi <= lo:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
    if hi <= lo:
        return np.zeros(x.shape, dtype=np.uint8)
    y = np.clip((x.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    return (y * 255.0).astype(np.uint8)


def _overlay_from_mask(mask: np.ndarray, base_rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(base_rgb).astype(np.uint8)
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[..., None], 3, axis=-1)
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]

    overlay = rgb.astype(np.float32).copy()
    alpha = 0.45
    labels = _resize_mask_to_shape(mask, tuple(overlay.shape[:2]), preserve_labels=True)
    if labels.shape != overlay.shape[:2]:
        labels = np.zeros(overlay.shape[:2], dtype=np.uint16)
    unique_labels = [int(value) for value in np.unique(labels) if int(value) > 0]
    if not unique_labels:
        return np.clip(overlay, 0, 255).astype(np.uint8)
    if len(unique_labels) == 1:
        color_map = {unique_labels[0]: np.array([255.0, 80.0, 80.0], dtype=np.float32)}
    else:
        color_map = {
            label: _instance_preview_color(label).astype(np.float32) for label in unique_labels
        }
    for label in unique_labels:
        region = labels == label
        color = color_map[label]
        for channel in range(3):
            overlay[..., channel] = np.where(
                region,
                overlay[..., channel] * (1.0 - alpha) + color[channel] * alpha,
                overlay[..., channel],
            )
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _resize_mask_to_shape(
    mask: np.ndarray,
    target_shape: tuple[int, int],
    *,
    preserve_labels: bool = False,
) -> np.ndarray:
    arr = np.asarray(mask)
    if preserve_labels:
        if arr.shape == target_shape:
            return arr
        try:
            from PIL import Image

            max_value = int(np.max(arr)) if arr.size else 0
            label_array = arr.astype(np.uint16 if max_value > 255 else np.uint8)
            return np.array(
                Image.fromarray(label_array).resize(
                    (int(target_shape[1]), int(target_shape[0])),
                    Image.NEAREST,
                )
            )
        except Exception:
            return np.asarray(arr)

    m = arr > 0
    if m.shape == target_shape:
        return m
    try:
        from PIL import Image

        resized = np.array(
            Image.fromarray(m.astype(np.uint8)).resize(
                (int(target_shape[1]), int(target_shape[0])),
                Image.NEAREST,
            )
        )
        return resized > 0
    except Exception:
        return np.asarray(m)


_INSTANCE_PREVIEW_PALETTE = np.asarray(
    [
        [255, 45, 149],
        [77, 208, 225],
        [255, 196, 61],
        [138, 99, 255],
        [75, 227, 141],
        [255, 125, 74],
        [0, 203, 255],
        [255, 86, 181],
        [184, 255, 61],
        [255, 141, 214],
        [131, 244, 255],
        [255, 166, 0],
    ],
    dtype=np.uint8,
)


def _instance_preview_color(label_value: int) -> np.ndarray:
    if int(label_value) <= 0:
        return np.array([82, 220, 255], dtype=np.uint8)
    index = (int(label_value) - 1) % len(_INSTANCE_PREVIEW_PALETTE)
    return _INSTANCE_PREVIEW_PALETTE[index]


def _mask_rgba_from_plane(mask_plane: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    labels = _resize_mask_to_shape(mask_plane, target_shape, preserve_labels=True)
    h, w = target_shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    if labels.shape != (h, w):
        labels = np.asarray(labels)
        labels = labels[:h, :w] if labels.ndim == 2 else np.zeros((h, w), dtype=np.uint16)
    unique_labels = [int(value) for value in np.unique(labels) if int(value) > 0]
    if not unique_labels:
        return rgba
    if len(unique_labels) == 1:
        region = labels == unique_labels[0]
        rgba[region, 0] = 82
        rgba[region, 1] = 220
        rgba[region, 2] = 255
        rgba[region, 3] = 218
        return rgba
    for label in unique_labels:
        region = labels == label
        color = _instance_preview_color(label)
        rgba[region, 0] = color[0]
        rgba[region, 1] = color[1]
        rgba[region, 2] = color[2]
        rgba[region, 3] = 218
    return rgba


def _compose_side_by_side_preview(base_rgb: np.ndarray, mask_rgba: np.ndarray) -> np.ndarray:
    left = np.asarray(base_rgb).astype(np.uint8)
    if left.ndim == 2:
        left = np.repeat(left[..., None], 3, axis=-1)
    if left.shape[-1] == 4:
        left = left[..., :3]
    h, w = int(left.shape[0]), int(left.shape[1])
    right = np.asarray(mask_rgba).astype(np.uint8)
    if right.shape[:2] != (h, w):
        right = _mask_rgba_from_plane(
            right[..., 3] if right.ndim == 3 and right.shape[-1] >= 4 else right, (h, w)
        )

    pad = 10
    gap = 12
    canvas_h = h + (pad * 2)
    canvas_w = (w * 2) + gap + (pad * 2)
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[..., :] = np.array([12, 14, 17], dtype=np.uint8)

    y0 = pad
    x0 = pad
    canvas[y0 : y0 + h, x0 : x0 + w] = left

    right_x0 = pad + w + gap
    right_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    right_alpha = np.zeros((h, w), dtype=np.float32)
    if right.ndim == 3 and right.shape[-1] >= 4:
        right_rgb = right[..., :3]
        right_alpha = right[..., 3].astype(np.float32) / 255.0
    else:
        binary = _resize_mask_to_shape(right, (h, w))
        right_rgb[binary] = np.array([82, 220, 255], dtype=np.uint8)
        right_alpha[binary] = 0.85

    bg_patch = np.zeros((h, w, 3), dtype=np.uint8)
    bg_patch[...] = np.array([8, 9, 12], dtype=np.uint8)
    blended = np.where(
        right_alpha[..., None] > 0,
        (
            bg_patch.astype(np.float32) * (1.0 - right_alpha[..., None])
            + right_rgb.astype(np.float32) * right_alpha[..., None]
        ),
        bg_patch.astype(np.float32),
    )
    canvas[y0 : y0 + h, right_x0 : right_x0 + w] = np.clip(blended, 0, 255).astype(np.uint8)
    return canvas


def _preview_rgb_from_array(array: np.ndarray, order: str) -> np.ndarray:
    arr = np.asarray(array)
    ord_chars = [ch for ch in str(order).upper() if ch.isalpha()]

    # Select first timepoint for preview.
    if "T" in ord_chars:
        axis = ord_chars.index("T")
        arr = np.take(arr, 0, axis=axis)
        ord_chars.pop(axis)

    # Select middle z for preview if available.
    if "Z" in ord_chars:
        axis = ord_chars.index("Z")
        z_idx = int(arr.shape[axis] // 2)
        arr = np.take(arr, z_idx, axis=axis)
        ord_chars.pop(axis)

    # Bring to C,Y,X or Y,X.
    if "C" in ord_chars and "Y" in ord_chars and "X" in ord_chars:
        perm = [ord_chars.index("C"), ord_chars.index("Y"), ord_chars.index("X")]
        cyx = np.transpose(arr, perm)
        if cyx.shape[0] >= 3:
            rgb = np.stack([_normalize_uint8(cyx[i]) for i in range(3)], axis=-1)
        else:
            gray = _normalize_uint8(cyx[0])
            rgb = np.repeat(gray[..., None], 3, axis=-1)
        return rgb.astype(np.uint8)

    if "Y" in ord_chars and "X" in ord_chars:
        perm = [ord_chars.index("Y"), ord_chars.index("X")]
        yx = np.transpose(arr, perm)
        gray = _normalize_uint8(yx)
        return np.repeat(gray[..., None], 3, axis=-1).astype(np.uint8)

    flat = arr.reshape((-1,) + arr.shape[-2:])
    gray = _normalize_uint8(flat[0])
    return np.repeat(gray[..., None], 3, axis=-1).astype(np.uint8)


def _save_segmentation_artifacts(
    *,
    source_path: str,
    output_dir: Path,
    array: np.ndarray,
    array_order: str,
    mask: np.ndarray,
    save_visualization: bool,
    artifact_prefix: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(source_path).stem
    mask_array = np.asarray(mask)
    mask_max_label = int(np.max(mask_array)) if mask_array.size > 0 else 0
    storage_dtype = np.uint16 if mask_max_label > np.iinfo(np.uint8).max else np.uint8
    mask_u8 = mask_array.astype(storage_dtype)
    mask_path = output_dir / f"{stem}_{artifact_prefix}_mask.npy"
    np.save(mask_path, mask_u8)

    warnings: list[str] = []

    def _save_volume_mask() -> str | None:
        if np.asarray(mask_u8).ndim != 3:
            return None

        source = Path(source_path)
        lower = source.name.lower()

        # For NIfTI sources, write a NIfTI mask with source affine/header
        # so BisQue receives a true volumetric artifact.
        if lower.endswith(".nii") or lower.endswith(".nii.gz"):
            try:
                import nibabel as nib  # type: ignore

                src_img = nib.load(str(source))
                # Internal mask order is Z,Y,X; NIfTI data should be X,Y,Z.
                mask_xyz = np.transpose(mask_u8, (2, 1, 0))
                mask_img = nib.Nifti1Image(
                    mask_xyz, affine=src_img.affine, header=src_img.header.copy()
                )
                mask_img.set_data_dtype(np.uint8)
                nii_path = output_dir / f"{stem}_{artifact_prefix}_mask.nii.gz"
                nib.save(mask_img, str(nii_path))
                return str(nii_path)
            except Exception as e:
                warnings.append(f"NIfTI mask export failed: {e}")

        # Generic 3D fallback as TIFF stack.
        try:
            import tifffile  # type: ignore

            tiff_path = output_dir / f"{stem}_{artifact_prefix}_mask.tiff"
            tifffile.imwrite(str(tiff_path), mask_u8, photometric="minisblack")
            return str(tiff_path)
        except Exception as e:
            warnings.append(f"TIFF volume mask export failed: {e}")
            return None

    mask_volume_path = _save_volume_mask()

    overlay_path: str | None = None
    mask_preview_path: str | None = None
    side_by_side_path: str | None = None
    visualization_paths: list[dict[str, Any]] = []
    if save_visualization:
        base_rgb = _preview_rgb_from_array(array, array_order)
        if np.asarray(mask).ndim == 3:
            mask_plane = np.asarray(mask)[int(np.asarray(mask).shape[0] // 2)]
        else:
            mask_plane = np.asarray(mask)
        instance_label_count = int(
            len([value for value in np.unique(mask_plane) if int(value) > 0])
        )
        mask_rgba = _mask_rgba_from_plane(mask_plane, tuple(base_rgb.shape[:2]))
        is_multi_instance_preview = instance_label_count > 1

        mask_path_png = output_dir / f"{stem}_{artifact_prefix}_mask_preview.png"
        Image.fromarray(mask_rgba).save(mask_path_png)
        mask_preview_path = str(mask_path_png)
        visualization_paths.append(
            {
                "path": mask_preview_path,
                "file": Path(source_path).name,
                "title": (
                    "Instance mask preview (colored by label)"
                    if is_multi_instance_preview
                    else "Mask foreground preview"
                ),
                "kind": "mask_only",
            }
        )

        side_by_side = _compose_side_by_side_preview(base_rgb, mask_rgba)
        side_path = output_dir / f"{stem}_{artifact_prefix}_side_by_side.png"
        Image.fromarray(side_by_side).save(side_path)
        side_by_side_path = str(side_path)
        visualization_paths.append(
            {
                "path": side_by_side_path,
                "file": Path(source_path).name,
                "title": (
                    "Original + instance masks side by side"
                    if is_multi_instance_preview
                    else "Original + mask side by side"
                ),
                "kind": "side_by_side",
            }
        )

        overlay = _overlay_from_mask(mask_plane, base_rgb)
        vis_path = output_dir / f"{stem}_{artifact_prefix}_overlay.png"
        Image.fromarray(overlay).save(vis_path)
        overlay_path = str(vis_path)
        visualization_paths.append(
            {
                "path": overlay_path,
                "file": Path(source_path).name,
                "title": (
                    "Original with labeled instance overlay"
                    if is_multi_instance_preview
                    else "Original with segmentation overlay"
                ),
                "kind": "overlay",
            }
        )

    return {
        "mask_path": str(mask_path),
        "mask_volume_path": mask_volume_path,
        "overlay_path": overlay_path,
        "mask_preview_path": mask_preview_path,
        "side_by_side_path": side_by_side_path,
        "visualization_paths": visualization_paths,
        "warnings": warnings,
    }


def _save_medsam_artifacts(
    *,
    source_path: str,
    output_dir: Path,
    array: np.ndarray,
    array_order: str,
    mask: np.ndarray,
    save_visualization: bool,
) -> dict[str, Any]:
    return _save_segmentation_artifacts(
        source_path=source_path,
        output_dir=output_dir,
        array=array,
        array_order=array_order,
        mask=mask,
        save_visualization=save_visualization,
        artifact_prefix="medsam2",
    )


_MEDSAM2_CT_VOLUME_SUFFIXES = (".nii", ".nii.gz", ".nrrd", ".mha", ".mhd")
_MEDSAM2_GENERIC_MODEL_SELECTORS = {"", "latest", "wanglab/medsam2"}
_MEDSAM2_GENERIC_MODEL_BASENAMES = {"medsam2_latest.pt", "medsam2_latest"}


def _medsam2_percentile_window(
    array: np.ndarray,
    *,
    lower: float = 1.0,
    upper: float = 99.0,
    max_samples: int = 200_000,
) -> tuple[float | None, float | None]:
    arr = np.asarray(array)
    if arr.size <= 0:
        return None, None
    flat = arr.reshape(-1)
    if flat.size > max_samples:
        step = int(np.ceil(flat.size / float(max_samples)))
        flat = flat[:: max(step, 1)]
    finite = flat[np.isfinite(flat)]
    if finite.size <= 0:
        return None, None
    return float(np.percentile(finite, lower)), float(np.percentile(finite, upper))


def _is_generic_medsam2_selector(model_ref: str | None) -> bool:
    normalized = str(model_ref or "").strip().lower()
    if normalized in _MEDSAM2_GENERIC_MODEL_SELECTORS:
        return True
    return Path(normalized).name in _MEDSAM2_GENERIC_MODEL_BASENAMES


def _infer_ct_like_volume_profile(
    *,
    file_path: str,
    loaded: dict[str, Any],
    array: np.ndarray,
) -> dict[str, Any]:
    metadata = loaded.get("metadata") if isinstance(loaded.get("metadata"), dict) else {}
    header = metadata.get("header") if isinstance(metadata.get("header"), dict) else {}
    coordinates = (
        metadata.get("coordinates") if isinstance(metadata.get("coordinates"), dict) else {}
    )
    filename_hints = (
        metadata.get("filename_hints") if isinstance(metadata.get("filename_hints"), dict) else {}
    )
    tokens = filename_hints.get("tokens") if isinstance(filename_hints.get("tokens"), list) else []
    lower_tokens = {str(token).strip().lower() for token in tokens if str(token).strip()}

    lower_path = str(file_path or "").strip().lower()
    is_volume = np.asarray(array).ndim >= 3
    medical_format = str(header.get("Format") or "").strip().lower()
    coordinate_space = str(coordinates.get("space") or "").strip().lower()
    has_medical_volume_hint = bool(
        is_volume
        and (
            lower_path.endswith(_MEDSAM2_CT_VOLUME_SUFFIXES)
            or medical_format == "nifti"
            or coordinate_space == "patient"
        )
    )

    p01, p99 = _medsam2_percentile_window(array)
    hu_like = bool(p01 is not None and p99 is not None and p01 <= -300.0 and p99 >= 300.0)
    token_hint = bool(
        lower_tokens & {"ct", "headct", "brainct", "chestct", "abdomenct", "thoraxct", "lesionct"}
    )
    is_ct_like = bool(has_medical_volume_hint and (hu_like or token_hint))
    reason = None
    if is_ct_like:
        reason = "hu_like_volume" if hu_like else "ct_filename_hint"
    return {
        "is_ct_like": is_ct_like,
        "reason": reason,
        "is_volume": is_volume,
        "has_medical_volume_hint": has_medical_volume_hint,
        "hu_like": hu_like,
        "token_hint": token_hint,
        "p01": p01,
        "p99": p99,
    }


def _select_medsam2_model_for_input(
    *,
    file_path: str,
    loaded: dict[str, Any],
    array: np.ndarray,
    requested_model_id: str | None,
    default_model_id: str | None,
) -> tuple[str, dict[str, Any], list[str]]:
    explicit_model_id = str(requested_model_id or "").strip()
    if explicit_model_id:
        return (
            explicit_model_id,
            {
                "selection_mode": "explicit",
                "requested_model_id": explicit_model_id,
                "effective_model_id": explicit_model_id,
            },
            [],
        )

    default_selector = str(default_model_id or "wanglab/MedSAM2").strip() or "wanglab/MedSAM2"
    ct_profile = _infer_ct_like_volume_profile(file_path=file_path, loaded=loaded, array=array)
    warnings: list[str] = []
    effective_model_id = default_selector
    selection_mode = "default"
    selection_reason = None

    if ct_profile.get("is_ct_like") and _is_generic_medsam2_selector(default_selector):
        effective_model_id = "ct_lesion"
        selection_mode = "auto"
        selection_reason = str(ct_profile.get("reason") or "ct_volume")
        warnings.append(
            "Auto-selected MedSAM2 CT checkpoint (ct_lesion) for a CT-like volumetric input."
        )

    return (
        effective_model_id,
        {
            "selection_mode": selection_mode,
            "selection_reason": selection_reason,
            "requested_model_id": None,
            "default_model_id": default_selector,
            "effective_model_id": effective_model_id,
            "ct_like_profile": ct_profile,
        },
        warnings,
    )


def segment_image_sam2(
    file_paths: list[str],
    points_per_batch: int = 64,
    save_visualizations: bool = True,
    device: str | None = None,
    model_id: str | None = None,
    max_slices: int | None = None,
) -> dict[str, Any]:
    """Segment 2D/3D scientific images with MedSAM2 via the universal data layer."""
    settings = get_settings()
    requested_model_id = str(model_id or "").strip() or None
    default_model_id = str(getattr(settings, "medsam2_model_id", "wanglab/MedSAM2"))
    max_slices = int(max_slices or getattr(settings, "medsam2_max_slices", 160))

    if not file_paths:
        return {"success": False, "error": "file_paths is required"}

    run_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
    output_dir = Path(_science_output_root("medsam2_results")) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for file_path in file_paths:
        try:
            loaded = load_scientific_image(
                file_path=str(file_path),
                array_mode="volume",
                save_array=True,
                include_array=False,
                return_array=True,
            )
            if not loaded.get("success"):
                results.append(
                    {
                        "file": os.path.basename(str(file_path)),
                        "path": str(file_path),
                        "success": False,
                        "error": loaded.get("error") or "Failed to load image.",
                    }
                )
                continue

            array = np.asarray(loaded.pop("_array"))
            array_order = str(loaded.get("array_order") or "YX")
            selected_model_id, model_selection, selection_warnings = (
                _select_medsam2_model_for_input(
                    file_path=str(file_path),
                    loaded=loaded,
                    array=array,
                    requested_model_id=requested_model_id,
                    default_model_id=default_model_id,
                )
            )
            seg = segment_array_with_medsam2(
                array,
                order=array_order,
                model_id=selected_model_id,
                device=device,
                max_slices=max_slices,
            )
            if not seg.get("success"):
                results.append(
                    {
                        "file": os.path.basename(str(file_path)),
                        "path": str(file_path),
                        "success": False,
                        "error": seg.get("error") or "MedSAM2 inference failed.",
                    }
                )
                continue

            mask = np.asarray(seg.pop("_mask"))
            artifact_paths = _save_medsam_artifacts(
                source_path=str(file_path),
                output_dir=output_dir,
                array=array,
                array_order=array_order,
                mask=mask,
                save_visualization=bool(save_visualizations),
            )
            preferred_upload_path = artifact_paths.get("mask_volume_path") or artifact_paths.get(
                "mask_path"
            )
            segmented_voxels = int(np.count_nonzero(mask))
            total_voxels = int(mask.size)
            coverage_pct = float((segmented_voxels / total_voxels) * 100.0) if total_voxels else 0.0

            axis_sizes = (
                loaded.get("axis_sizes") if isinstance(loaded.get("axis_sizes"), dict) else {}
            )
            artifact_warnings = (
                artifact_paths.get("warnings")
                if isinstance(artifact_paths.get("warnings"), list)
                else []
            )
            result_data = {
                "file": os.path.basename(str(file_path)),
                "path": str(file_path),
                "success": True,
                "data": {
                    "scene": loaded.get("scene"),
                    "dims_order": loaded.get("dims_order"),
                    "axis_sizes": axis_sizes,
                    "selected_indices": loaded.get("selected_indices"),
                },
                "segmentation": {
                    "total_masks": 1,
                    "segmented_voxels": segmented_voxels,
                    "total_voxels": total_voxels,
                    "coverage_percent": round(coverage_pct, 4),
                    "mean_iou": seg.get("mean_iou"),
                    "slice_count": seg.get("slice_count"),
                    "slice_stride": seg.get("slice_stride"),
                },
                "mask_path": artifact_paths.get("mask_path"),
                "mask_volume_path": artifact_paths.get("mask_volume_path"),
                "preferred_upload_path": preferred_upload_path,
                "visualization": artifact_paths.get("overlay_path"),
                "visualizations": artifact_paths.get("visualization_paths") or [],
                "model": seg.get("resolved_model_ref") or seg.get("model_id") or selected_model_id,
                "model_selection": model_selection,
                "device": seg.get("device"),
                "warnings": (loaded.get("warnings") or [])
                + selection_warnings
                + (seg.get("warnings") or [])
                + artifact_warnings,
                "points_per_batch": points_per_batch,  # retained for backward compatibility
            }
            results.append(result_data)

        except Exception as file_error:
            results.append(
                {
                    "file": os.path.basename(str(file_path)),
                    "path": str(file_path),
                    "success": False,
                    "error": str(file_error),
                }
            )

    successful = [r for r in results if r.get("success")]
    results_summary: list[dict[str, Any]] = []
    visualization_paths: list[dict[str, Any]] = []
    preferred_upload_paths: list[str] = []
    preferred_upload_entries: list[dict[str, Any]] = []
    total_masks = 0

    for row in results:
        if row.get("success"):
            seg = row.get("segmentation") or {}
            auto_points = row.get("auto_points") if isinstance(row.get("auto_points"), dict) else {}
            total_masks += int(seg.get("total_masks", 0))
            results_summary.append(
                {
                    "file": row.get("file"),
                    "success": True,
                    "total_masks": int(seg.get("total_masks", 0)),
                    "coverage_percent": seg.get("coverage_percent"),
                    "avg_points_per_window": auto_points.get("avg_points_per_window"),
                    "min_points": auto_points.get("min_points"),
                    "max_points": auto_points.get("max_points"),
                    "point_density": auto_points.get("point_density"),
                    "visualization_saved": bool(
                        row.get("visualization") or row.get("visualizations")
                    ),
                    "preferred_upload_path": row.get("preferred_upload_path"),
                }
            )
            preferred_path = row.get("preferred_upload_path")
            if preferred_path:
                preferred_upload_paths.append(str(preferred_path))
                preferred_upload_entries.append(
                    {
                        "file": row.get("file"),
                        "path": str(preferred_path),
                    }
                )
            row_visualizations = (
                row.get("visualizations") if isinstance(row.get("visualizations"), list) else []
            )
            for visualization in row_visualizations:
                if not isinstance(visualization, dict) or not visualization.get("path"):
                    continue
                visualization_paths.append(
                    {
                        "path": str(visualization.get("path")),
                        "file": row.get("file"),
                        "coverage_percent": seg.get("coverage_percent"),
                        "title": visualization.get("title"),
                        "kind": visualization.get("kind"),
                    }
                )
            if row.get("visualization") and not row_visualizations:
                visualization_paths.append(
                    {
                        "path": row.get("visualization"),
                        "file": row.get("file"),
                        "coverage_percent": seg.get("coverage_percent"),
                        "title": "Original with segmentation overlay",
                        "kind": "overlay",
                    }
                )
        else:
            results_summary.append(
                {
                    "file": row.get("file"),
                    "success": False,
                    "error": row.get("error", "Unknown error"),
                }
            )

    summary_payload = {
        "processed_files": len(successful),
        "total_files": len(results),
        "total_masks_generated": int(total_masks),
    }
    coverage_rows = [
        {
            "file": row.get("file"),
            "coverage_percent": row.get("segmentation", {}).get("coverage_percent"),
            "total_masks": row.get("segmentation", {}).get("total_masks"),
        }
        for row in successful
    ]
    ui_artifacts: list[dict[str, Any]] = [
        {"type": "metrics", "title": "MedSAM2 segmentation summary", "payload": summary_payload},
        {
            "type": "table",
            "title": "Per-file segmentation coverage",
            "payload": coverage_rows[:200],
        },
    ]
    for item in visualization_paths[:40]:
        if item.get("path"):
            ui_artifacts.append(
                {
                    "type": "image",
                    "title": item.get("title") or "Segmentation visualization",
                    "path": item.get("path"),
                }
            )

    response = {
        "success": len(successful) > 0,
        "processed": len(successful),
        "total_files": len(results),
        "total_masks_generated": int(total_masks),
        "files_processed": results_summary,
        "output_directory": str(output_dir),
        "model": (
            (successful[0].get("model") if successful and isinstance(successful[0], dict) else None)
            or requested_model_id
            or default_model_id
        ),
        "device": device or "auto",
        "preferred_upload_paths": preferred_upload_paths,
        "preferred_upload_entries": preferred_upload_entries,
        "visualization_paths": visualization_paths,
        "ui_artifacts": ui_artifacts,
        "message": (
            "Segmentation performed with MedSAM2 using the bioio-backed universal loader. "
            "Use preferred_upload_paths (volume mask when available) for BisQue uploads. "
            "For promptable control, use sam2_prompt_image with points/boxes."
        ),
    }
    response["latest_result_refs"] = _tool_result_refs_for_segmentation(response)
    return response


def sam2_prompt_image(
    file_path: str,
    input_points: list[Any] | None = None,
    input_labels: list[Any] | None = None,
    input_boxes: list[Any] | None = None,
    model_id: str | None = None,
    multimask_output: bool = True,
    save_visualization: bool = True,
    device: str | None = None,
    max_slices: int | None = None,
) -> dict[str, Any]:
    """Prompted MedSAM2 segmentation for 2D/3D scientific images."""
    settings = get_settings()
    requested_model_id = str(model_id or "").strip() or None
    default_model_id = str(getattr(settings, "medsam2_model_id", "wanglab/MedSAM2"))
    resolved_max_slices = (
        int(max_slices)
        if max_slices is not None
        else int(getattr(settings, "medsam2_max_slices", 160))
    )

    if not file_path or not os.path.exists(file_path):
        return {"success": False, "error": f"File not found: {file_path}"}

    def _normalize_prompt_points(value: list[Any] | None) -> list[Any] | None:
        if not value:
            return None
        if (
            isinstance(value, list)
            and value
            and isinstance(value[0], (list, tuple))
            and len(value[0]) == 2
            and isinstance(value[0][0], (int, float))
        ):
            try:
                return [[float(point[0]), float(point[1])] for point in value]
            except Exception:
                return None

        groups: list[list[list[float]]] = []
        for group in value:
            if not isinstance(group, (list, tuple)):
                continue
            normalized_group: list[list[float]] = []
            for point in group:
                if not isinstance(point, (list, tuple)) or len(point) < 2:
                    continue
                try:
                    normalized_group.append([float(point[0]), float(point[1])])
                except Exception:
                    continue
            if normalized_group:
                groups.append(normalized_group)
        return groups or None

    def _normalize_prompt_labels(
        value: list[Any] | None,
        *,
        points_value: list[Any] | None,
    ) -> list[Any] | None:
        if not value:
            return None
        if (
            isinstance(points_value, list)
            and points_value
            and isinstance(points_value[0], (list, tuple))
            and len(points_value[0]) == 2
            and isinstance(points_value[0][0], (int, float))
        ):
            try:
                return [1 if int(item) > 0 else 0 for item in value]
            except Exception:
                return None

        groups: list[list[int]] = []
        for group in value:
            if not isinstance(group, (list, tuple)):
                continue
            normalized_group: list[int] = []
            for item in group:
                try:
                    normalized_group.append(1 if int(item) > 0 else 0)
                except Exception:
                    normalized_group.append(1)
            if normalized_group:
                groups.append(normalized_group)
        return groups or None

    points: list[Any] | None = None
    labels: list[Any] | None = None
    boxes: list[list[float]] | None = None
    if input_points:
        points = _normalize_prompt_points(input_points)
    if input_labels:
        labels = _normalize_prompt_labels(input_labels, points_value=points)
    if input_boxes:
        try:
            boxes = [
                [float(v) for v in box[:4]]
                for box in input_boxes
                if isinstance(box, (list, tuple)) and len(box) >= 4
            ]
        except Exception:
            boxes = None

    try:
        loaded = load_scientific_image(
            file_path=str(file_path),
            array_mode="volume",
            save_array=True,
            include_array=False,
            return_array=True,
        )
        if not loaded.get("success"):
            return {
                "success": False,
                "error": loaded.get("error") or "Failed to load image via bioio.",
            }

        array = np.asarray(loaded.pop("_array"))
        array_order = str(loaded.get("array_order") or "YX")
        selected_model_id, model_selection, selection_warnings = _select_medsam2_model_for_input(
            file_path=str(file_path),
            loaded=loaded,
            array=array,
            requested_model_id=requested_model_id,
            default_model_id=default_model_id,
        )
        seg = segment_array_with_medsam2(
            array,
            order=array_order,
            input_points=points,
            input_labels=labels,
            input_boxes=boxes,
            multimask_output=bool(multimask_output),
            model_id=selected_model_id,
            device=device,
            max_slices=resolved_max_slices,
        )
        if not seg.get("success"):
            return {"success": False, "error": seg.get("error") or "MedSAM2 segmentation failed."}

        mask = np.asarray(seg.pop("_mask"))
        output_dir = Path(_science_output_root("medsam2_prompt_results"))
        artifact_paths = _save_medsam_artifacts(
            source_path=str(file_path),
            output_dir=output_dir,
            array=array,
            array_order=array_order,
            mask=mask,
            save_visualization=bool(save_visualization),
        )
        preferred_upload_path = (
            artifact_paths.get("mask_volume_path")
            or artifact_paths.get("mask_path")
            or artifact_paths.get("overlay_path")
        )

        score_map = seg.get("_slice_scores") if isinstance(seg.get("_slice_scores"), dict) else {}
        best_scores = [float(v) for _, v in sorted(score_map.items())]
        mean_score = (
            float(np.mean(best_scores)) if best_scores else float(seg.get("mean_iou", 0.0) or 0.0)
        )
        total_masks_generated = int(
            seg.get("total_masks_generated") or seg.get("instance_count") or 0
        )
        if total_masks_generated <= 0 and bool(seg.get("success")):
            total_masks_generated = 1
        instance_count_scope = (
            str(seg.get("instance_count_scope") or "").strip() or "prompt_object_masks"
        )

        ui_artifacts = [
            {
                "type": "metrics",
                "title": "MedSAM2 prompt results",
                "payload": {
                    "model": seg.get("resolved_model_ref")
                    or seg.get("model_id")
                    or selected_model_id,
                    "selection_mode": model_selection.get("selection_mode"),
                    "selection_reason": model_selection.get("selection_reason"),
                    "slice_count": seg.get("slice_count"),
                    "slices_processed": seg.get("slices_processed"),
                    "processed_all_slices": seg.get("processed_all_slices"),
                    "slice_stride": seg.get("slice_stride"),
                    "total_masks_generated": total_masks_generated,
                    "instance_count_scope": instance_count_scope,
                    "best_iou_mean": round(mean_score, 4),
                    "coverage_percent": seg.get("coverage_percent"),
                    "instance_coverage_percent_mean": seg.get("instance_coverage_percent_mean"),
                    "instance_coverage_percent_min": seg.get("instance_coverage_percent_min"),
                    "instance_coverage_percent_max": seg.get("instance_coverage_percent_max"),
                },
            },
            {
                "type": "table",
                "title": "Input dimensionality",
                "payload": [
                    {
                        "dims_order": loaded.get("dims_order"),
                        "array_order": array_order,
                        "array_shape": loaded.get("array_shape"),
                        "scene": loaded.get("scene"),
                    }
                ],
            },
        ]
        visualizations = (
            artifact_paths.get("visualization_paths")
            if isinstance(artifact_paths.get("visualization_paths"), list)
            else []
        )
        if visualizations:
            for visualization in visualizations[:20]:
                if not isinstance(visualization, dict) or not visualization.get("path"):
                    continue
                ui_artifacts.append(
                    {
                        "type": "image",
                        "title": visualization.get("title") or "MedSAM2 prompt visualization",
                        "path": visualization.get("path"),
                    }
                )
        elif artifact_paths.get("overlay_path"):
            ui_artifacts.append(
                {
                    "type": "image",
                    "title": "MedSAM2 prompt overlay",
                    "path": artifact_paths["overlay_path"],
                }
            )

        return {
            "success": True,
            "file": os.path.basename(file_path),
            "output_path": preferred_upload_path,
            "mask_path": artifact_paths.get("mask_path"),
            "mask_volume_path": artifact_paths.get("mask_volume_path"),
            "preferred_upload_path": preferred_upload_path,
            "visualization_path": artifact_paths.get("overlay_path"),
            "visualization_paths": visualizations,
            "model": seg.get("resolved_model_ref") or seg.get("model_id") or selected_model_id,
            "resolved_model_ref": seg.get("resolved_model_ref")
            or seg.get("model_id")
            or selected_model_id,
            "model_selection": model_selection,
            "backend": seg.get("backend"),
            "slice_count": seg.get("slice_count"),
            "slices_processed": seg.get("slices_processed"),
            "slice_stride": seg.get("slice_stride"),
            "processed_all_slices": seg.get("processed_all_slices"),
            "total_masks_generated": total_masks_generated,
            "instance_count_reported": total_masks_generated,
            "instance_count_measured": int(seg.get("instance_count") or total_masks_generated),
            "instance_count_scope": instance_count_scope,
            "coverage_percent": seg.get("coverage_percent"),
            "instance_coverage_percent_mean": seg.get("instance_coverage_percent_mean"),
            "instance_coverage_percent_min": seg.get("instance_coverage_percent_min"),
            "instance_coverage_percent_max": seg.get("instance_coverage_percent_max"),
            "instance_coverage_percent_values": seg.get("instance_coverage_percent_values"),
            "best_iou_scores": best_scores,
            "warnings": (loaded.get("warnings") or [])
            + selection_warnings
            + (seg.get("warnings") or [])
            + (artifact_paths.get("warnings") or []),
            "ui_artifacts": ui_artifacts,
        }

    except Exception as exc:
        logger.error("MedSAM2 prompt failed: %s", exc)
        return {"success": False, "error": str(exc)}


def estimate_depth_pro(
    file_paths: list[str],
    model_id: str | None = None,
    device: str | None = None,
    use_fov_model: bool | None = None,
    save_visualizations: bool = True,
    save_raw_depth: bool = True,
    force_rerun: bool = False,
) -> dict[str, Any]:
    """Estimate depth maps with DepthPro and persist visualization artifacts."""
    settings = get_settings()
    model_ref = str(
        model_id or getattr(settings, "depth_pro_model_id", "apple/DepthPro-hf")
    ).strip()
    resolved_use_fov = bool(
        use_fov_model
        if use_fov_model is not None
        else getattr(settings, "depth_pro_use_fov_model", True)
    )

    if not file_paths:
        return {"success": False, "error": "file_paths is required"}
    if not bool(save_visualizations) and not bool(save_raw_depth):
        return {
            "success": False,
            "error": "At least one output mode is required (save_visualizations or save_raw_depth).",
        }

    try:
        expanded_inputs = _expand_file_inputs([str(p) for p in file_paths])
    except Exception as exc:
        return {"success": False, "error": str(exc)}

    expanded_inputs, sequence_expansions, sequence_warnings = _expand_sequence_inputs_for_2d_models(
        expanded_inputs
    )
    resolved_paths = [str(path) for path in expanded_inputs]
    if not resolved_paths:
        return {"success": False, "error": "No valid image or sequence frames found in file_paths."}
    missing_paths = [path for path in resolved_paths if not Path(path).expanduser().exists()]
    if missing_paths:
        return {
            "success": False,
            "error": "Some files do not exist",
            "missing": missing_paths,
        }

    cache_args = {
        "input_file_paths": [str(p) for p in file_paths],
        "file_paths": resolved_paths,
        "model_id": model_ref,
        "device": device,
        "use_fov_model": bool(resolved_use_fov),
        "save_visualizations": bool(save_visualizations),
        "save_raw_depth": bool(save_raw_depth),
        "force_rerun": bool(force_rerun),
    }
    cache_key = _segmentation_cache_key("estimate_depth_pro", cache_args)
    if not force_rerun:
        cached = _DEPTH_PRO_RESULT_CACHE.get(cache_key)
        if cached and _segmentation_result_paths_exist(cached):
            cached_result = dict(cached)
            cached_result["cached"] = True
            cached_result["cache_hit"] = True
            cached_result["message"] = (
                "Using cached DepthPro result for identical inputs/parameters."
            )
            return cached_result

    try:
        processor, model, torch_device = _depthpro_runtime(
            model_id=model_ref,
            use_fov_model=bool(resolved_use_fov),
            device=device,
        )
        import torch  # type: ignore
    except Exception as exc:
        return {"success": False, "error": str(exc)}

    run_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
    output_dir = Path(_science_output_root("depth_pro_results")) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    def _scalar(value: Any) -> float | None:
        if value is None:
            return None
        try:
            if hasattr(value, "detach"):
                value = value.detach().cpu()
            if hasattr(value, "item"):
                return float(value.item())
            return float(value)
        except Exception:
            return None

    results: list[dict[str, Any]] = []
    for file_path in resolved_paths:
        input_path = str(file_path or "").strip()
        if not input_path:
            continue
        try:
            loaded = load_scientific_image(
                file_path=input_path,
                array_mode="plane",
                save_array=False,
                include_array=False,
                return_array=False,
            )
            if not loaded.get("success"):
                results.append(
                    {
                        "file": os.path.basename(input_path),
                        "path": input_path,
                        "success": False,
                        "error": loaded.get("error") or "Failed to load image.",
                    }
                )
                continue

            preview_path = str(loaded.get("preview_path") or input_path)
            with Image.open(preview_path) as opened:
                image_rgb = opened.convert("RGB")

            model_inputs = processor(images=image_rgb, return_tensors="pt")
            if hasattr(model_inputs, "to"):
                model_inputs = model_inputs.to(model.device)
            elif isinstance(model_inputs, dict):
                model_inputs = {
                    key: (value.to(model.device) if hasattr(value, "to") else value)
                    for key, value in model_inputs.items()
                }
            else:
                raise RuntimeError("Unsupported processor output type for DepthPro inputs.")

            with torch.no_grad():
                outputs = model(**model_inputs)

            post_result = processor.post_process_depth_estimation(
                outputs,
                target_sizes=[(image_rgb.height, image_rgb.width)],
            )
            post = post_result[0] if isinstance(post_result, list) and post_result else {}
            depth_tensor = post.get("predicted_depth") if isinstance(post, dict) else None
            if depth_tensor is None:
                depth_tensor = getattr(outputs, "predicted_depth", None)
            if depth_tensor is None:
                raise RuntimeError("DepthPro did not return predicted_depth.")

            if hasattr(depth_tensor, "detach"):
                depth_array = np.asarray(depth_tensor.detach().float().cpu().numpy())
            else:
                depth_array = np.asarray(depth_tensor, dtype=np.float32)
            depth_array = np.squeeze(depth_array).astype(np.float32, copy=False)
            if depth_array.ndim != 2:
                raise RuntimeError(
                    f"DepthPro returned unsupported depth shape {depth_array.shape}; expected 2D."
                )

            file_name = Path(input_path).name
            stem = f"{_safe_slug(Path(file_name).stem)}_{uuid4().hex[:8]}"
            depth_map_path = output_dir / f"{stem}_depth_map.png"
            depth_heatmap_path = output_dir / f"{stem}_depth_heatmap.png"
            overlay_path = output_dir / f"{stem}_depth_overlay.png"
            side_by_side_path = output_dir / f"{stem}_depth_side_by_side.png"
            depth_npy_path = output_dir / f"{stem}_depth.npy"

            file_visualizations: list[dict[str, Any]] = []
            if bool(save_visualizations):
                depth_u8 = _depth_to_uint8(depth_array)
                depth_gray = Image.fromarray(depth_u8, mode="L")
                depth_color = _depth_colorize(depth_u8)
                overlay_image = Image.blend(image_rgb, depth_color, alpha=0.42)
                side_by_side = Image.new("RGB", (image_rgb.width * 2, image_rgb.height))
                side_by_side.paste(image_rgb, (0, 0))
                side_by_side.paste(depth_color, (image_rgb.width, 0))

                depth_gray.save(depth_map_path)
                depth_color.save(depth_heatmap_path)
                overlay_image.save(overlay_path)
                side_by_side.save(side_by_side_path)

                file_visualizations.extend(
                    [
                        {
                            "path": str(depth_map_path),
                            "file": file_name,
                            "title": f"{file_name} depth map (grayscale)",
                            "kind": "depth_map",
                        },
                        {
                            "path": str(depth_heatmap_path),
                            "file": file_name,
                            "title": f"{file_name} depth map (heatmap)",
                            "kind": "depth_heatmap",
                        },
                        {
                            "path": str(overlay_path),
                            "file": file_name,
                            "title": f"{file_name} depth overlay",
                            "kind": "depth_overlay",
                        },
                        {
                            "path": str(side_by_side_path),
                            "file": file_name,
                            "title": f"{file_name} depth side-by-side",
                            "kind": "depth_side_by_side",
                        },
                    ]
                )

            depth_npy_saved: str | None = None
            if bool(save_raw_depth):
                np.save(depth_npy_path, depth_array.astype(np.float32))
                depth_npy_saved = str(depth_npy_path)

            finite = depth_array[np.isfinite(depth_array)]
            depth_min = float(np.min(finite)) if finite.size else None
            depth_max = float(np.max(finite)) if finite.size else None
            depth_mean = float(np.mean(finite)) if finite.size else None
            depth_std = float(np.std(finite)) if finite.size else None
            preferred_upload_path = (
                str(depth_map_path) if bool(save_visualizations) else depth_npy_saved
            )

            results.append(
                {
                    "file": file_name,
                    "path": input_path,
                    "success": True,
                    "depth_map_path": str(depth_map_path) if bool(save_visualizations) else None,
                    "depth_heatmap_path": str(depth_heatmap_path)
                    if bool(save_visualizations)
                    else None,
                    "depth_overlay_path": str(overlay_path) if bool(save_visualizations) else None,
                    "depth_side_by_side_path": str(side_by_side_path)
                    if bool(save_visualizations)
                    else None,
                    "depth_npy_path": depth_npy_saved,
                    "preferred_upload_path": preferred_upload_path,
                    "visualizations": file_visualizations,
                    "model": model_ref,
                    "device": torch_device,
                    "field_of_view": _scalar(
                        post.get("field_of_view") if isinstance(post, dict) else None
                    ),
                    "focal_length": _scalar(
                        post.get("focal_length") if isinstance(post, dict) else None
                    ),
                    "depth_min": depth_min,
                    "depth_max": depth_max,
                    "depth_mean": depth_mean,
                    "depth_std": depth_std,
                    "warnings": loaded.get("warnings") or [],
                }
            )
        except Exception as file_error:
            logger.error("DepthPro inference failed for %s: %s", input_path, file_error)
            results.append(
                {
                    "file": os.path.basename(input_path),
                    "path": input_path,
                    "success": False,
                    "error": str(file_error),
                }
            )

    successful = [row for row in results if row.get("success")]
    results_summary: list[dict[str, Any]] = []
    visualization_paths: list[dict[str, Any]] = []
    preferred_upload_paths: list[str] = []
    preferred_upload_entries: list[dict[str, Any]] = []
    depth_map_paths: list[str] = []
    depth_npy_paths: list[str] = []
    depth_means: list[float] = []

    for row in results:
        if row.get("success"):
            preferred = str(row.get("preferred_upload_path") or "").strip()
            if preferred:
                preferred_upload_paths.append(preferred)
                preferred_upload_entries.append(
                    {
                        "file": row.get("file"),
                        "path": preferred,
                    }
                )
            depth_map_path = str(row.get("depth_map_path") or "").strip()
            if depth_map_path:
                depth_map_paths.append(depth_map_path)
            depth_npy_path = str(row.get("depth_npy_path") or "").strip()
            if depth_npy_path:
                depth_npy_paths.append(depth_npy_path)
            depth_mean = row.get("depth_mean")
            if isinstance(depth_mean, (int, float)):
                depth_means.append(float(depth_mean))
            row_visualizations = (
                row.get("visualizations") if isinstance(row.get("visualizations"), list) else []
            )
            for visualization in row_visualizations:
                if not isinstance(visualization, dict) or not visualization.get("path"):
                    continue
                visualization_paths.append(visualization)

            results_summary.append(
                {
                    "file": row.get("file"),
                    "success": True,
                    "depth_min": row.get("depth_min"),
                    "depth_max": row.get("depth_max"),
                    "depth_mean": row.get("depth_mean"),
                    "depth_std": row.get("depth_std"),
                    "field_of_view": row.get("field_of_view"),
                    "focal_length": row.get("focal_length"),
                    "preferred_upload_path": preferred,
                }
            )
        else:
            results_summary.append(
                {
                    "file": row.get("file"),
                    "success": False,
                    "error": row.get("error", "Unknown error"),
                }
            )

    summary_payload = {
        "processed_files": len(successful),
        "total_files": len(results),
        "depth_mean_average": round(float(np.mean(depth_means)), 6) if depth_means else None,
        "use_fov_model": bool(resolved_use_fov),
    }
    depth_table = [
        {
            "file": row.get("file"),
            "depth_min": row.get("depth_min"),
            "depth_max": row.get("depth_max"),
            "depth_mean": row.get("depth_mean"),
            "depth_std": row.get("depth_std"),
            "field_of_view": row.get("field_of_view"),
            "focal_length": row.get("focal_length"),
        }
        for row in successful
    ]

    ui_artifacts: list[dict[str, Any]] = [
        {"type": "metrics", "title": "DepthPro summary", "payload": summary_payload},
        {"type": "table", "title": "Per-file depth statistics", "payload": depth_table[:200]},
    ]
    for item in visualization_paths[:120]:
        if item.get("path"):
            ui_artifacts.append(
                {
                    "type": "image",
                    "title": item.get("title") or "DepthPro visualization",
                    "path": item.get("path"),
                }
            )

    latest_refs: dict[str, Any] = {
        "latest_depth_output_directory": str(output_dir),
        "latest_depth_manifest": [
            {
                "file": row.get("file"),
                "depth_map_path": row.get("depth_map_path"),
                "depth_npy_path": row.get("depth_npy_path"),
                "preferred_upload_path": row.get("preferred_upload_path"),
                "success": row.get("success"),
            }
            for row in results
            if isinstance(row, dict)
        ],
    }
    if depth_map_paths:
        latest_refs["latest_depth_map_path"] = depth_map_paths[-1]
        latest_refs["latest_depth_map_paths"] = depth_map_paths
    if depth_npy_paths:
        latest_refs["latest_depth_npy_path"] = depth_npy_paths[-1]
        latest_refs["latest_depth_npy_paths"] = depth_npy_paths

    response = {
        "success": len(successful) > 0,
        "processed": len(successful),
        "total_files": len(results),
        "files_processed": results_summary,
        "output_directory": str(output_dir),
        "model": model_ref,
        "device": torch_device,
        "use_fov_model": bool(resolved_use_fov),
        "preferred_upload_paths": preferred_upload_paths,
        "preferred_upload_entries": preferred_upload_entries,
        "depth_map_paths": depth_map_paths,
        "depth_npy_paths": depth_npy_paths,
        "sequence_expansions": sequence_expansions,
        "sequence_warnings": sequence_warnings,
        "visualization_paths": visualization_paths,
        "depth_mean_average": summary_payload["depth_mean_average"],
        "ui_artifacts": ui_artifacts,
        "latest_result_refs": latest_refs,
        "force_rerun": bool(force_rerun),
        "message": (
            "Depth estimation completed with DepthPro. "
            "Use depth_map_paths with segment_image_sam2 if you want to segment depth-derived structures."
            if successful
            else "DepthPro failed for all provided files."
        ),
    }
    if _segmentation_result_paths_exist(response):
        _DEPTH_PRO_RESULT_CACHE[cache_key] = json.loads(
            json.dumps(response, ensure_ascii=False, default=str)
        )
    return response


def segment_image_sam3(
    file_paths: list[str],
    save_visualizations: bool = True,
    model_id: str | None = None,
    device: str | None = None,
    preset: str | None = "balanced",
    concept_prompt: str | None = None,
    input_points: list[Any] | None = None,
    input_points_labels: list[Any] | None = None,
    input_boxes: list[list[float]] | None = None,
    input_boxes_labels: list[int] | None = None,
    threshold: float | None = None,
    slice_index: int | None = None,
    max_slices: int | None = None,
    window_size: int | None = None,
    window_overlap: float | None = None,
    min_points: int | None = None,
    max_points: int | None = None,
    point_density: float | None = None,
    mask_threshold: float | None = None,
    vote_threshold: float | None = None,
    min_component_area_ratio: float | None = None,
    modality_hint: str | None = None,
    preprocess: bool | None = None,
    refine_3d: bool | None = None,
    fallback_to_medsam2: bool | None = None,
    force_rerun: bool = False,
) -> dict[str, Any]:
    """Automatic first-pass segmentation for 2D/3D/4D scientific images using SAM3."""
    settings = get_settings()
    model_id = str(model_id or getattr(settings, "sam3_model_id", "facebook/sam3"))
    base_max_slices = int(getattr(settings, "sam3_max_slices", 192))
    base_window_size = int(getattr(settings, "sam3_window_size", 1024))
    base_window_overlap = float(getattr(settings, "sam3_window_overlap", 0.25))
    base_min_points = int(getattr(settings, "sam3_min_points", 8))
    base_max_points = int(getattr(settings, "sam3_max_points", 64))
    base_point_density = float(getattr(settings, "sam3_point_density", 0.0015))
    base_refine_3d = bool(getattr(settings, "sam3_refine_3d", True))
    base_preprocess = True if preprocess is None else bool(preprocess)

    preset_name = str(preset or "balanced").strip().lower().replace("-", "_")
    preset_defaults: dict[str, dict[str, Any]] = {
        "fast": {
            "max_slices": max(32, int(round(base_max_slices * 0.5))),
            "window_size": max(512, int(round(base_window_size * 0.75))),
            "window_overlap": max(0.05, min(0.35, base_window_overlap * 0.8)),
            "min_points": max(4, int(round(base_min_points * 0.75))),
            "max_points": max(16, int(round(base_max_points * 0.5))),
            "point_density": max(0.00025, base_point_density * 0.7),
            "preprocess": True,
            "refine_3d": False,
        },
        "balanced": {
            "max_slices": base_max_slices,
            "window_size": base_window_size,
            "window_overlap": base_window_overlap,
            "min_points": base_min_points,
            "max_points": base_max_points,
            "point_density": base_point_density,
            "preprocess": base_preprocess,
            "refine_3d": base_refine_3d,
        },
        "high_quality": {
            "max_slices": min(4096, max(base_max_slices, int(round(base_max_slices * 1.5)))),
            "window_size": max(base_window_size, 1024),
            "window_overlap": max(base_window_overlap, 0.25),
            "min_points": max(base_min_points, int(round(base_min_points * 1.5))),
            "max_points": min(512, max(base_max_points, int(round(base_max_points * 2.0)))),
            "point_density": min(0.05, max(base_point_density, base_point_density * 1.5)),
            "preprocess": True,
            "refine_3d": True,
        },
    }
    preset_warning: str | None = None
    if preset_name not in preset_defaults:
        preset_warning = f"Unknown preset '{preset_name}', using 'balanced'."
        preset_name = "balanced"
    preset_values = preset_defaults[preset_name]

    max_slices = int(max_slices if max_slices is not None else preset_values["max_slices"])
    window_size = int(window_size if window_size is not None else preset_values["window_size"])
    window_overlap = float(
        window_overlap if window_overlap is not None else preset_values["window_overlap"]
    )
    min_points = int(min_points if min_points is not None else preset_values["min_points"])
    max_points = int(max_points if max_points is not None else preset_values["max_points"])
    point_density = float(
        point_density if point_density is not None else preset_values["point_density"]
    )
    mask_threshold = float(
        mask_threshold
        if mask_threshold is not None
        else getattr(settings, "sam3_mask_threshold", 0.5)
    )
    vote_threshold = float(
        vote_threshold
        if vote_threshold is not None
        else getattr(settings, "sam3_vote_threshold", 0.5)
    )
    min_component_area_ratio = float(
        min_component_area_ratio
        if min_component_area_ratio is not None
        else getattr(settings, "sam3_min_component_area_ratio", 0.0001)
    )
    modality_hint = str(
        modality_hint
        if modality_hint is not None
        else getattr(settings, "sam3_modality_hint", "auto")
    )
    preprocess = bool(preprocess if preprocess is not None else preset_values["preprocess"])
    refine_3d = bool(refine_3d if refine_3d is not None else preset_values["refine_3d"])
    fallback_to_medsam2 = bool(
        fallback_to_medsam2
        if fallback_to_medsam2 is not None
        else getattr(settings, "sam3_fallback_to_medsam2", True)
    )
    allow_remote_download = bool(getattr(settings, "sam3_allow_remote_download", False))
    auto_zero_mask_fallback_enabled = bool(
        getattr(settings, "sam3_auto_zero_mask_fallback_enabled", True)
    )
    auto_zero_mask_fallback_prompt = str(
        getattr(settings, "sam3_auto_zero_mask_fallback_prompt", "object")
    ).strip()
    auto_zero_mask_fallback_prompt_candidates: list[str] = []
    if auto_zero_mask_fallback_prompt:
        auto_zero_mask_fallback_prompt_candidates.append(auto_zero_mask_fallback_prompt)
    raw_fallback_prompts = str(
        getattr(settings, "sam3_auto_zero_mask_fallback_prompts", "")
    ).strip()
    if raw_fallback_prompts:
        for candidate in re.split(r"[,\n;|]+", raw_fallback_prompts):
            prompt_value = str(candidate or "").strip()
            if prompt_value:
                auto_zero_mask_fallback_prompt_candidates.append(prompt_value)
    if not auto_zero_mask_fallback_prompt_candidates:
        auto_zero_mask_fallback_prompt_candidates = ["object"]
    auto_zero_mask_fallback_prompt_candidates = list(
        dict.fromkeys(auto_zero_mask_fallback_prompt_candidates)
    )
    auto_low_coverage_fallback_threshold_percent = float(
        getattr(settings, "sam3_auto_low_coverage_fallback_threshold_percent", 0.0)
    )
    threshold = float(threshold if threshold is not None else 0.5)
    concept_prompt = str(concept_prompt or "").strip()
    concept_warnings: list[str] = []
    normalized_input_points = (
        list(input_points) if isinstance(input_points, list) and input_points else []
    )
    normalized_input_point_labels = (
        list(input_points_labels)
        if isinstance(input_points_labels, list) and input_points_labels
        else None
    )
    normalized_input_boxes: list[list[float]] = []
    for idx, row in enumerate(input_boxes or []):
        values = list(row) if isinstance(row, (list, tuple)) else []
        if len(values) < 4:
            concept_warnings.append(
                f"Ignored input_boxes[{idx}] because it does not contain four coordinates."
            )
            continue
        try:
            x1, y1, x2, y2 = (
                float(values[0]),
                float(values[1]),
                float(values[2]),
                float(values[3]),
            )
        except Exception:
            concept_warnings.append(
                f"Ignored input_boxes[{idx}] because coordinates are not numeric."
            )
            continue
        normalized_input_boxes.append([x1, y1, x2, y2])

    normalized_input_boxes_labels: list[int] | None = None
    if input_boxes_labels is not None:
        if not normalized_input_boxes:
            concept_warnings.append(
                "input_boxes_labels was provided without valid input_boxes; ignored."
            )
        elif len(input_boxes_labels) != len(normalized_input_boxes):
            return {
                "success": False,
                "error": "input_boxes_labels must match input_boxes length.",
            }
        else:
            normalized_input_boxes_labels = []
            for label in input_boxes_labels:
                try:
                    normalized_input_boxes_labels.append(1 if int(label) > 0 else 0)
                except Exception:
                    normalized_input_boxes_labels.append(1)
    elif normalized_input_boxes:
        normalized_input_boxes_labels = [1 for _ in normalized_input_boxes]

    point_prompt_mode = bool(normalized_input_points)
    concept_mode = bool(concept_prompt) or bool(normalized_input_boxes)
    if point_prompt_mode and concept_mode:
        return {
            "success": False,
            "error": (
                "segment_image_sam3 does not accept mixed input_points with concept_prompt/input_boxes. "
                "Convert mixed prompts to prompt regions before calling."
            ),
        }

    if not file_paths:
        return {"success": False, "error": "file_paths is required"}

    try:
        expanded_inputs = _expand_file_inputs([str(p) for p in file_paths])
    except Exception as exc:
        return {"success": False, "error": str(exc)}
    expanded_inputs, sequence_expansions, sequence_warnings = _expand_sequence_inputs_for_2d_models(
        expanded_inputs
    )
    resolved_file_paths = [str(path) for path in expanded_inputs]
    if not resolved_file_paths:
        return {"success": False, "error": "No valid image or sequence frames found in file_paths."}
    missing_paths = [path for path in resolved_file_paths if not Path(path).expanduser().exists()]
    if missing_paths:
        return {
            "success": False,
            "error": "Some files do not exist",
            "missing": missing_paths,
        }

    cache_args = {
        "input_file_paths": [str(p) for p in file_paths],
        "file_paths": resolved_file_paths,
        "save_visualizations": bool(save_visualizations),
        "model_id": model_id,
        "device": device,
        "preset": preset_name,
        "mode": "tracker_points" if point_prompt_mode else "concept" if concept_mode else "auto",
        "concept_prompt": concept_prompt or None,
        "input_points": normalized_input_points,
        "input_points_labels": normalized_input_point_labels,
        "input_boxes": normalized_input_boxes,
        "input_boxes_labels": normalized_input_boxes_labels,
        "threshold": float(threshold),
        "slice_index": int(slice_index) if slice_index is not None else None,
        "auto_zero_mask_fallback_enabled": bool(auto_zero_mask_fallback_enabled),
        "auto_zero_mask_fallback_prompt": auto_zero_mask_fallback_prompt,
        "auto_zero_mask_fallback_prompts": auto_zero_mask_fallback_prompt_candidates,
        "auto_low_coverage_fallback_threshold_percent": float(
            auto_low_coverage_fallback_threshold_percent
        ),
        "max_slices": max_slices,
        "window_size": window_size,
        "window_overlap": window_overlap,
        "min_points": min_points,
        "max_points": max_points,
        "point_density": point_density,
        "mask_threshold": mask_threshold,
        "vote_threshold": vote_threshold,
        "min_component_area_ratio": min_component_area_ratio,
        "modality_hint": modality_hint,
        "preprocess": bool(preprocess),
        "refine_3d": bool(refine_3d),
        "fallback_to_medsam2": bool(fallback_to_medsam2),
        "force_rerun": bool(force_rerun),
    }
    cache_key = _segmentation_cache_key("segment_image_sam3", cache_args)
    if not force_rerun:
        cached = _SEGMENTATION_RESULT_CACHE.get(cache_key)
        if cached and _segmentation_result_paths_exist(cached):
            cached_result = dict(cached)
            cached_result["cached"] = True
            cached_result["cache_hit"] = True
            cached_result["message"] = (
                "Using cached SAM3 segmentation result for identical inputs/parameters."
            )
            return cached_result

    run_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
    output_dir = Path(_science_output_root("sam3_results")) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for file_path in resolved_file_paths:
        try:
            loaded = load_scientific_image(
                file_path=str(file_path),
                array_mode="volume",
                save_array=True,
                include_array=False,
                return_array=True,
            )
            if not loaded.get("success"):
                results.append(
                    {
                        "file": os.path.basename(str(file_path)),
                        "path": str(file_path),
                        "success": False,
                        "error": loaded.get("error") or "Failed to load image.",
                    }
                )
                continue

            array = np.asarray(loaded.pop("_array"))
            array_order = str(loaded.get("array_order") or "YX")
            seg: dict[str, Any] = {}
            for attempt in range(2):
                if point_prompt_mode:
                    seg = segment_array_with_sam3_points(
                        array,
                        order=array_order,
                        input_points=normalized_input_points,
                        input_labels=normalized_input_point_labels,
                        mask_threshold=float(mask_threshold),
                        model_id=model_id,
                        device=device,
                        slice_index=int(slice_index) if slice_index is not None else None,
                        preprocess=bool(preprocess),
                        allow_remote_download=allow_remote_download,
                    )
                elif concept_mode:
                    seg = segment_array_with_sam3_concept(
                        array,
                        order=array_order,
                        concept_prompt=concept_prompt or None,
                        input_boxes=normalized_input_boxes or None,
                        input_boxes_labels=normalized_input_boxes_labels,
                        threshold=float(threshold),
                        mask_threshold=float(mask_threshold),
                        model_id=model_id,
                        device=device,
                        slice_index=int(slice_index) if slice_index is not None else None,
                        preprocess=bool(preprocess),
                        allow_remote_download=allow_remote_download,
                    )
                else:
                    seg = segment_array_with_sam3(
                        array,
                        order=array_order,
                        model_id=model_id,
                        device=device,
                        max_slices=max_slices,
                        window_size=window_size,
                        window_overlap=window_overlap,
                        min_points=min_points,
                        max_points=max_points,
                        point_density=point_density,
                        mask_threshold=mask_threshold,
                        vote_threshold=vote_threshold,
                        min_component_area_ratio=min_component_area_ratio,
                        modality_hint=modality_hint,
                        preprocess=bool(preprocess),
                        refine_3d=bool(refine_3d),
                        fallback_to_medsam2=bool(fallback_to_medsam2),
                        allow_remote_download=allow_remote_download,
                    )
                if seg.get("success"):
                    break
                if attempt == 0:
                    logger.warning(
                        "SAM3 %s segmentation failed on first attempt for %s; retrying once.",
                        "tracker-points"
                        if point_prompt_mode
                        else "concept"
                        if concept_mode
                        else "auto",
                        file_path,
                    )
            if not seg.get("success"):
                results.append(
                    {
                        "file": os.path.basename(str(file_path)),
                        "path": str(file_path),
                        "success": False,
                        "error": seg.get("error") or "SAM3 inference failed.",
                    }
                )
                continue

            auto_fallback_used = False
            auto_fallback_note: str | None = None
            if (
                not concept_mode
                and not point_prompt_mode
                and bool(auto_zero_mask_fallback_enabled)
                and len(auto_zero_mask_fallback_prompt_candidates) > 0
            ):
                auto_mask_probe = np.asarray(seg.get("_mask"))
                auto_nonzero = (
                    int(np.count_nonzero(auto_mask_probe)) if auto_mask_probe.size > 0 else 0
                )
                auto_total = int(auto_mask_probe.size)
                auto_coverage_percent = (
                    float((auto_nonzero / auto_total) * 100.0) if auto_total > 0 else 0.0
                )
                fallback_triggered = auto_nonzero == 0 or (
                    float(auto_low_coverage_fallback_threshold_percent) > 0.0
                    and auto_coverage_percent <= float(auto_low_coverage_fallback_threshold_percent)
                )
                if fallback_triggered:
                    best_prompt: str | None = None
                    best_fallback_seg: dict[str, Any] | None = None
                    best_fallback_nonzero = 0
                    for fallback_prompt in auto_zero_mask_fallback_prompt_candidates:
                        fallback_seg = segment_array_with_sam3_concept(
                            array,
                            order=array_order,
                            concept_prompt=fallback_prompt,
                            threshold=float(threshold),
                            mask_threshold=float(mask_threshold),
                            model_id=model_id,
                            device=device,
                            slice_index=int(slice_index) if slice_index is not None else None,
                            preprocess=bool(preprocess),
                            allow_remote_download=allow_remote_download,
                        )
                        if not fallback_seg.get("success"):
                            continue
                        fallback_mask = np.asarray(fallback_seg.get("_mask"))
                        fallback_nonzero = (
                            int(np.count_nonzero(fallback_mask)) if fallback_mask.size > 0 else 0
                        )
                        if fallback_nonzero > best_fallback_nonzero:
                            best_fallback_nonzero = fallback_nonzero
                            best_prompt = fallback_prompt
                            best_fallback_seg = fallback_seg
                    if best_fallback_seg is not None and best_fallback_nonzero > auto_nonzero:
                        seg = best_fallback_seg
                        auto_fallback_used = True
                        auto_fallback_note = (
                            "SAM3 auto mode produced an empty/low-coverage mask; retried with concept mode "
                            f"(concept_prompt='{best_prompt}')."
                        )

            mask = np.asarray(seg.pop("_mask"))
            artifact_paths = _save_segmentation_artifacts(
                source_path=str(file_path),
                output_dir=output_dir,
                array=array,
                array_order=array_order,
                mask=mask,
                save_visualization=bool(save_visualizations),
                artifact_prefix="sam3",
            )
            preferred_upload_path = artifact_paths.get("mask_volume_path") or artifact_paths.get(
                "mask_path"
            )
            segmented_voxels = int(np.count_nonzero(mask))
            total_voxels = int(mask.size)
            coverage_pct = float((segmented_voxels / total_voxels) * 100.0) if total_voxels else 0.0
            instance_count_scope = "connected_components_of_final_mask"
            component_sizes: list[int] = []
            raw_instance_sizes = seg.get("instance_area_voxels")
            if isinstance(raw_instance_sizes, list):
                for value in raw_instance_sizes:
                    try:
                        parsed = int(value)
                    except Exception:
                        continue
                    if parsed > 0:
                        component_sizes.append(parsed)
            if component_sizes:
                scope_raw = str(seg.get("instance_measurement_scope") or "").strip()
                if scope_raw:
                    instance_count_scope = scope_raw
            else:
                component_sizes = _connected_component_sizes(mask > 0, min_component_size=1)
            instance_count_measured = int(len(component_sizes))
            instance_area_voxels_mean = float(np.mean(component_sizes)) if component_sizes else 0.0
            instance_area_voxels_min = float(np.min(component_sizes)) if component_sizes else 0.0
            instance_area_voxels_max = float(np.max(component_sizes)) if component_sizes else 0.0
            instance_coverage_values = (
                [float(size) / float(total_voxels) * 100.0 for size in component_sizes]
                if total_voxels and component_sizes
                else []
            )
            instance_coverage_percent_mean = (
                float(np.mean(instance_coverage_values)) if instance_coverage_values else 0.0
            )
            instance_coverage_percent_min = (
                float(np.min(instance_coverage_values)) if instance_coverage_values else 0.0
            )
            instance_coverage_percent_max = (
                float(np.max(instance_coverage_values)) if instance_coverage_values else 0.0
            )
            max_instance_values_reported = 4096
            instance_values_truncated = len(component_sizes) > max_instance_values_reported
            instance_area_voxels_values = [
                int(value) for value in component_sizes[:max_instance_values_reported]
            ]
            instance_coverage_percent_values = [
                round(float(value), 6)
                for value in instance_coverage_values[:max_instance_values_reported]
            ]

            axis_sizes = (
                loaded.get("axis_sizes") if isinstance(loaded.get("axis_sizes"), dict) else {}
            )
            artifact_warnings = (
                artifact_paths.get("warnings")
                if isinstance(artifact_paths.get("warnings"), list)
                else []
            )
            auto_summary = (
                seg.get("auto_point_summary")
                if isinstance(seg.get("auto_point_summary"), dict)
                else {}
            )
            effective_mode = (
                "interactive_points"
                if (
                    point_prompt_mode
                    or str(seg.get("backend") or "")
                    .strip()
                    .lower()
                    .startswith("sam3-tracker-points")
                )
                else "concept"
                if (
                    concept_mode
                    or bool(auto_fallback_used)
                    or str(seg.get("backend") or "").strip().lower().startswith("sam3-concept")
                )
                else "auto"
            )
            estimated_instances = seg.get("estimated_instances")
            try:
                total_masks = max(
                    0, int(estimated_instances if estimated_instances is not None else 0)
                )
            except Exception:
                total_masks = 0
            mask_count_scope_warning: str | None = None
            if (
                total_masks > 0
                and instance_count_measured > 0
                and total_masks != instance_count_measured
            ):
                mask_count_scope_warning = (
                    "Reported mask count differs from measured instance count in this output. "
                    f"Per-instance area metrics use `{instance_count_scope}`."
                )
            result_data = {
                "file": os.path.basename(str(file_path)),
                "path": str(file_path),
                "success": True,
                "data": {
                    "scene": loaded.get("scene"),
                    "dims_order": loaded.get("dims_order"),
                    "axis_sizes": axis_sizes,
                    "selected_indices": loaded.get("selected_indices"),
                },
                "segmentation": {
                    "mode": effective_mode,
                    "preset": preset_name,
                    "total_masks": total_masks,
                    "instance_count_reported": total_masks,
                    "instance_count_measured": instance_count_measured,
                    "instance_count_scope": instance_count_scope,
                    "segmented_voxels": segmented_voxels,
                    "total_voxels": total_voxels,
                    "coverage_scope": "union_mask_over_image",
                    "coverage_percent": round(coverage_pct, 4),
                    "instance_area_voxels_mean": round(instance_area_voxels_mean, 6),
                    "instance_area_voxels_min": round(instance_area_voxels_min, 6),
                    "instance_area_voxels_max": round(instance_area_voxels_max, 6),
                    "instance_area_voxels": instance_area_voxels_values,
                    "instance_coverage_percent_mean": round(instance_coverage_percent_mean, 6),
                    "instance_coverage_percent_min": round(instance_coverage_percent_min, 6),
                    "instance_coverage_percent_max": round(instance_coverage_percent_max, 6),
                    "instance_coverage_percent_values": instance_coverage_percent_values,
                    "instance_values_truncated": bool(instance_values_truncated),
                    "mean_score": seg.get("mean_score"),
                    "slice_count": seg.get("slice_count"),
                    "slice_stride": seg.get("slice_stride"),
                    "vote_threshold": seg.get("vote_threshold"),
                    "threshold": float(threshold) if effective_mode == "concept" else None,
                    "modality_profile": seg.get("modality_profile"),
                    "refine_3d": seg.get("refine_3d"),
                    "slice_index_used": seg.get("slice_index_used"),
                    "auto_zero_mask_fallback_used": bool(auto_fallback_used),
                },
                "auto_points": auto_summary,
                "modality_profile_stats": seg.get("modality_profile_stats"),
                "concept_prompt": seg.get("concept_prompt"),
                "input_points": seg.get("input_points") or normalized_input_points,
                "input_point_labels": seg.get("input_point_labels")
                or normalized_input_point_labels,
                "input_boxes": seg.get("input_boxes") or normalized_input_boxes,
                "input_boxes_labels": seg.get("input_boxes_labels")
                or normalized_input_boxes_labels,
                "mask_path": artifact_paths.get("mask_path"),
                "mask_volume_path": artifact_paths.get("mask_volume_path"),
                "preferred_upload_path": preferred_upload_path,
                "visualization": artifact_paths.get("overlay_path"),
                "visualizations": artifact_paths.get("visualization_paths") or [],
                "model": seg.get("resolved_model_ref") or seg.get("model_id") or model_id,
                "backend": seg.get("backend"),
                "device": seg.get("device") or device or "auto",
                "warnings": ([preset_warning] if preset_warning else [])
                + concept_warnings
                + ([auto_fallback_note] if auto_fallback_note else [])
                + ([mask_count_scope_warning] if mask_count_scope_warning else [])
                + (loaded.get("warnings") or [])
                + (seg.get("warnings") or [])
                + artifact_warnings,
            }
            results.append(result_data)

        except Exception as file_error:
            results.append(
                {
                    "file": os.path.basename(str(file_path)),
                    "path": str(file_path),
                    "success": False,
                    "error": str(file_error),
                }
            )

    successful = [r for r in results if r.get("success")]
    results_summary: list[dict[str, Any]] = []
    visualization_paths: list[dict[str, Any]] = []
    preferred_upload_paths: list[str] = []
    preferred_upload_entries: list[dict[str, Any]] = []
    total_masks = 0
    coverage_values: list[float] = []
    auto_fallback_used_count = 0
    instance_count_reported_total = 0
    instance_count_measured_total = 0
    instance_count_mismatch_files = 0
    instance_area_weighted_sum = 0.0
    instance_coverage_weighted_sum = 0.0
    instance_area_min_global: float | None = None
    instance_area_max_global: float | None = None
    instance_coverage_min_global: float | None = None
    instance_coverage_max_global: float | None = None

    for row in results:
        if row.get("success"):
            seg = row.get("segmentation") or {}
            auto_points = row.get("auto_points") if isinstance(row.get("auto_points"), dict) else {}
            reported_count = int(seg.get("total_masks", 0))
            measured_count = int(seg.get("instance_count_measured", 0))
            total_masks += reported_count
            instance_count_reported_total += reported_count
            instance_count_measured_total += measured_count
            if reported_count != measured_count:
                instance_count_mismatch_files += 1

            area_mean = seg.get("instance_area_voxels_mean")
            area_min = seg.get("instance_area_voxels_min")
            area_max = seg.get("instance_area_voxels_max")
            coverage_mean = seg.get("instance_coverage_percent_mean")
            coverage_min = seg.get("instance_coverage_percent_min")
            coverage_max = seg.get("instance_coverage_percent_max")
            try:
                if measured_count > 0 and area_mean is not None:
                    instance_area_weighted_sum += float(area_mean) * float(measured_count)
            except Exception:
                pass
            try:
                if measured_count > 0 and coverage_mean is not None:
                    instance_coverage_weighted_sum += float(coverage_mean) * float(measured_count)
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
                if coverage_min is not None:
                    coverage_min_value = float(coverage_min)
                    instance_coverage_min_global = (
                        coverage_min_value
                        if instance_coverage_min_global is None
                        else min(instance_coverage_min_global, coverage_min_value)
                    )
            except Exception:
                pass
            try:
                if coverage_max is not None:
                    coverage_max_value = float(coverage_max)
                    instance_coverage_max_global = (
                        coverage_max_value
                        if instance_coverage_max_global is None
                        else max(instance_coverage_max_global, coverage_max_value)
                    )
            except Exception:
                pass

            if bool(seg.get("auto_zero_mask_fallback_used")):
                auto_fallback_used_count += 1
            results_summary.append(
                {
                    "file": row.get("file"),
                    "success": True,
                    "mode": row.get("segmentation", {}).get("mode"),
                    "preset": row.get("segmentation", {}).get("preset"),
                    "total_masks": reported_count,
                    "instance_count_reported": reported_count,
                    "instance_count_measured": measured_count,
                    "instance_count_scope": seg.get("instance_count_scope"),
                    "coverage_scope": seg.get("coverage_scope"),
                    "coverage_percent": seg.get("coverage_percent"),
                    "instance_area_voxels_mean": area_mean,
                    "instance_area_voxels_min": area_min,
                    "instance_area_voxels_max": area_max,
                    "instance_area_voxels": seg.get("instance_area_voxels"),
                    "instance_coverage_percent_mean": coverage_mean,
                    "instance_coverage_percent_min": coverage_min,
                    "instance_coverage_percent_max": coverage_max,
                    "instance_coverage_percent_values": seg.get("instance_coverage_percent_values"),
                    "instance_values_truncated": bool(seg.get("instance_values_truncated")),
                    "avg_points_per_window": auto_points.get("avg_points_per_window"),
                    "min_points": auto_points.get("min_points"),
                    "max_points": auto_points.get("max_points"),
                    "point_density": auto_points.get("point_density"),
                    "visualization_saved": bool(
                        row.get("visualization") or row.get("visualizations")
                    ),
                    "preferred_upload_path": row.get("preferred_upload_path"),
                }
            )
            try:
                coverage_values.append(float(seg.get("coverage_percent")))
            except Exception:
                pass
            preferred_path = row.get("preferred_upload_path")
            if preferred_path:
                preferred_upload_paths.append(str(preferred_path))
                preferred_upload_entries.append(
                    {
                        "file": row.get("file"),
                        "path": str(preferred_path),
                    }
                )
            row_visualizations = (
                row.get("visualizations") if isinstance(row.get("visualizations"), list) else []
            )
            for visualization in row_visualizations:
                if not isinstance(visualization, dict) or not visualization.get("path"):
                    continue
                visualization_paths.append(
                    {
                        "path": str(visualization.get("path")),
                        "file": row.get("file"),
                        "coverage_percent": seg.get("coverage_percent"),
                        "title": visualization.get("title"),
                        "kind": visualization.get("kind"),
                    }
                )
            if row.get("visualization") and not row_visualizations:
                visualization_paths.append(
                    {
                        "path": row.get("visualization"),
                        "file": row.get("file"),
                        "coverage_percent": seg.get("coverage_percent"),
                        "title": "Original with segmentation overlay",
                        "kind": "overlay",
                    }
                )
        else:
            results_summary.append(
                {
                    "file": row.get("file"),
                    "success": False,
                    "error": row.get("error", "Unknown error"),
                }
            )

    coverage_table = [
        {
            "file": row.get("file"),
            "coverage_scope": row.get("segmentation", {}).get("coverage_scope"),
            "coverage_percent": row.get("segmentation", {}).get("coverage_percent"),
            "segmented_voxels": row.get("segmentation", {}).get("segmented_voxels"),
            "total_voxels": row.get("segmentation", {}).get("total_voxels"),
            "estimated_instances": row.get("segmentation", {}).get("total_masks"),
            "instance_count_scope": row.get("segmentation", {}).get("instance_count_scope"),
            "instance_count_measured": row.get("segmentation", {}).get("instance_count_measured"),
            "instance_coverage_percent_mean": row.get("segmentation", {}).get(
                "instance_coverage_percent_mean"
            ),
            "instance_coverage_percent_min": row.get("segmentation", {}).get(
                "instance_coverage_percent_min"
            ),
            "instance_coverage_percent_max": row.get("segmentation", {}).get(
                "instance_coverage_percent_max"
            ),
            "instance_area_voxels_mean": row.get("segmentation", {}).get(
                "instance_area_voxels_mean"
            ),
            "instance_area_voxels_min": row.get("segmentation", {}).get("instance_area_voxels_min"),
            "instance_area_voxels_max": row.get("segmentation", {}).get("instance_area_voxels_max"),
        }
        for row in successful
    ]
    unique_instance_scopes = sorted(
        {
            str(row.get("segmentation", {}).get("instance_count_scope") or "").strip()
            for row in successful
            if str(row.get("segmentation", {}).get("instance_count_scope") or "").strip()
        }
    )
    if not unique_instance_scopes:
        instance_count_scope = "unknown"
    elif len(unique_instance_scopes) == 1:
        instance_count_scope = unique_instance_scopes[0]
    else:
        instance_count_scope = "mixed"
    summary_payload = {
        "processed_files": len(successful),
        "total_files": len(results),
        "mode": "concept" if concept_mode else "auto",
        "mode_resolved": (
            "concept"
            if concept_mode
            else ("auto_with_concept_fallback" if auto_fallback_used_count > 0 else "auto")
        ),
        "input_file_paths": [str(p) for p in file_paths],
        "resolved_file_paths": resolved_file_paths,
        "preset": preset_name,
        "concept_prompt": concept_prompt or None,
        "input_boxes_count": int(len(normalized_input_boxes)),
        "auto_zero_mask_fallback_files": int(auto_fallback_used_count),
        "auto_zero_mask_fallback_prompts": auto_zero_mask_fallback_prompt_candidates,
        "auto_low_coverage_fallback_threshold_percent": float(
            auto_low_coverage_fallback_threshold_percent
        ),
        "total_masks_generated": int(total_masks),
        "instance_count_reported_total": int(instance_count_reported_total),
        "instance_count_measured_total": int(instance_count_measured_total),
        "instance_count_mismatch_files": int(instance_count_mismatch_files),
        "instance_count_scope": instance_count_scope,
        "coverage_scope": "union_mask_over_image",
        "coverage_percent_mean": round(float(np.mean(coverage_values)), 6)
        if coverage_values
        else None,
        "coverage_percent_min": round(float(np.min(coverage_values)), 6)
        if coverage_values
        else None,
        "coverage_percent_max": round(float(np.max(coverage_values)), 6)
        if coverage_values
        else None,
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
            round(float(instance_coverage_weighted_sum) / float(instance_count_measured_total), 6)
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
        "min_points": int(min_points),
        "max_points": int(max_points),
        "point_density": float(point_density),
    }
    ui_artifacts: list[dict[str, Any]] = [
        {"type": "metrics", "title": "SAM3 segmentation summary", "payload": summary_payload},
        {"type": "table", "title": "Per-file SAM3 coverage", "payload": coverage_table[:200]},
    ]
    for item in visualization_paths[:60]:
        if item.get("path"):
            ui_artifacts.append(
                {
                    "type": "image",
                    "title": item.get("title") or "SAM3 visualization",
                    "path": item.get("path"),
                }
            )

    response = {
        "success": len(successful) > 0,
        "processed": len(successful),
        "total_files": len(results),
        "total_masks_generated": int(total_masks),
        "files_processed": results_summary,
        "output_directory": str(output_dir),
        "model": (
            (successful[0].get("model") if successful and isinstance(successful[0], dict) else None)
            or model_id
        ),
        "backend": (
            (
                successful[0].get("backend")
                if successful and isinstance(successful[0], dict)
                else None
            )
            or ("sam3-concept" if concept_mode else "sam3-tracker")
        ),
        "mode": "concept" if concept_mode else "auto",
        "mode_resolved": (
            "concept"
            if concept_mode
            else ("auto_with_concept_fallback" if auto_fallback_used_count > 0 else "auto")
        ),
        "input_file_paths": [str(p) for p in file_paths],
        "resolved_file_paths": resolved_file_paths,
        "preset": preset_name,
        "concept_prompt": concept_prompt or None,
        "input_boxes": normalized_input_boxes,
        "input_boxes_labels": normalized_input_boxes_labels,
        "threshold": float(threshold),
        "slice_index": int(slice_index) if slice_index is not None else None,
        "auto_zero_mask_fallback_enabled": bool(auto_zero_mask_fallback_enabled),
        "auto_zero_mask_fallback_prompt": auto_zero_mask_fallback_prompt or None,
        "auto_zero_mask_fallback_prompts": auto_zero_mask_fallback_prompt_candidates,
        "auto_low_coverage_fallback_threshold_percent": float(
            auto_low_coverage_fallback_threshold_percent
        ),
        "auto_zero_mask_fallback_files": int(auto_fallback_used_count),
        "device": device or "auto",
        "preferred_upload_paths": preferred_upload_paths,
        "preferred_upload_entries": preferred_upload_entries,
        "visualization_paths": visualization_paths,
        "sequence_expansions": sequence_expansions,
        "sequence_warnings": sequence_warnings,
        "coverage_percent_mean": summary_payload["coverage_percent_mean"],
        "coverage_percent_min": summary_payload["coverage_percent_min"],
        "coverage_percent_max": summary_payload["coverage_percent_max"],
        "coverage_scope": summary_payload["coverage_scope"],
        "instance_count_reported_total": summary_payload["instance_count_reported_total"],
        "instance_count_measured_total": summary_payload["instance_count_measured_total"],
        "instance_count_mismatch_files": summary_payload["instance_count_mismatch_files"],
        "instance_count_scope": summary_payload["instance_count_scope"],
        "instance_area_voxels_mean": summary_payload["instance_area_voxels_mean"],
        "instance_area_voxels_min": summary_payload["instance_area_voxels_min"],
        "instance_area_voxels_max": summary_payload["instance_area_voxels_max"],
        "instance_coverage_percent_mean": summary_payload["instance_coverage_percent_mean"],
        "instance_coverage_percent_min": summary_payload["instance_coverage_percent_min"],
        "instance_coverage_percent_max": summary_payload["instance_coverage_percent_max"],
        "min_points": int(min_points),
        "max_points": int(max_points),
        "point_density": float(point_density),
        "ui_artifacts": ui_artifacts,
        "force_rerun": bool(force_rerun),
        "warnings": ([preset_warning] if preset_warning else []) + concept_warnings,
        "message": (
            (
                "Segmentation performed with SAM3 concept prompts. "
                "Use concept_prompt and/or input_boxes for targeted masks; use preferred_upload_paths for uploads."
                if concept_mode
                else (
                    (
                        "Segmentation performed with SAM3 automatic prompting and sliding-window support "
                        "with concept fallback on empty masks. "
                        "Use preferred_upload_paths (volume mask when available) for BisQue uploads."
                    )
                    if auto_fallback_used_count > 0
                    else (
                        "Segmentation performed with SAM3 automatic prompting and sliding-window support. "
                        "Use preferred_upload_paths (volume mask when available) for BisQue uploads."
                    )
                )
            )
            if successful
            else (
                "SAM3 concept segmentation failed for all provided files."
                if concept_mode
                else "SAM3 segmentation failed for all provided files."
            )
        ),
    }
    response["latest_result_refs"] = _tool_result_refs_for_segmentation(response)
    if _segmentation_result_paths_exist(response):
        _SEGMENTATION_RESULT_CACHE[cache_key] = json.loads(
            json.dumps(response, ensure_ascii=False, default=str)
        )
    return response


def _load_mask_array(path: str) -> np.ndarray:
    p = Path(str(path)).expanduser()
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"Mask file not found: {p}")

    lower = p.name.lower()
    if lower.endswith(".npy"):
        return np.asarray(np.load(p))
    if lower.endswith(".nii") or lower.endswith(".nii.gz"):
        import nibabel as nib  # type: ignore

        return np.asarray(nib.load(str(p)).get_fdata())
    if lower.endswith(".tif") or lower.endswith(".tiff"):
        import tifffile  # type: ignore

        return np.asarray(tifffile.imread(str(p)))

    # Fallback for common 2D image masks.
    img = Image.open(p)
    return np.asarray(img)


def _binarize_mask_array(arr: np.ndarray, threshold: float) -> np.ndarray:
    x = np.asarray(arr)
    if x.dtype == bool:
        return x
    if np.issubdtype(x.dtype, np.integer):
        return x > 0
    th = float(threshold)
    finite = x[np.isfinite(x)]
    if finite.size <= 0:
        return np.zeros(x.shape, dtype=bool)
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if hi <= lo:
        return np.zeros(x.shape, dtype=bool)
    norm = (x.astype(np.float32) - lo) / (hi - lo)
    return norm >= th


def _segmentation_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    p = np.asarray(pred).astype(bool)
    g = np.asarray(gt).astype(bool)
    if p.shape != g.shape:
        # Conservative nearest-neighbor resize only for 2D inputs.
        if p.ndim == 2 and g.ndim == 2:
            p = (
                np.array(
                    Image.fromarray(p.astype(np.uint8)).resize(
                        (g.shape[1], g.shape[0]), Image.NEAREST
                    )
                )
                > 0
            )
        else:
            raise ValueError(
                f"Shape mismatch between prediction and ground truth: {p.shape} vs {g.shape}"
            )

    tp = int(np.logical_and(p, g).sum())
    fp = int(np.logical_and(p, np.logical_not(g)).sum())
    fn = int(np.logical_and(np.logical_not(p), g).sum())

    dice = (2.0 * tp) / float(max(2 * tp + fp + fn, 1))
    iou = tp / float(max(tp + fp + fn, 1))
    precision = tp / float(max(tp + fp, 1))
    recall = tp / float(max(tp + fn, 1))

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


def _pairing_lookup_keys(path: str, stem_strip_tokens: list[str] | None = None) -> list[str]:
    p = Path(str(path)).expanduser()
    abs_path = str(p.resolve()) if p.exists() else str(p)
    base = p.name
    canonical_base = _canonical_uploaded_filename(base)
    stem = _normalized_pairing_stem(str(p), stem_strip_tokens=stem_strip_tokens)
    keys = [abs_path, str(p), base, canonical_base]
    if stem:
        keys.append(stem)
    return [k for k in keys if k]


def _resolve_pair_map_target(
    prediction_path: str,
    pair_map: dict[str, str],
    stem_strip_tokens: list[str] | None = None,
) -> str | None:
    if not isinstance(pair_map, dict) or not pair_map:
        return None
    for key in _pairing_lookup_keys(prediction_path, stem_strip_tokens=stem_strip_tokens):
        candidate = pair_map.get(key)
        if candidate:
            return str(candidate)
    return None


def _is_pairing_mismatch_error(result: dict[str, Any] | None) -> bool:
    if not isinstance(result, dict):
        return False
    error_text = str(result.get("error") or "").lower()
    if not error_text:
        return False
    return (
        "no prediction/ground-truth pairs matched" in error_text
        or "no prediction/ground truth pairs matched" in error_text
    )


def evaluate_segmentation_masks(
    prediction_paths: list[str],
    ground_truth_paths: list[str],
    threshold: float = 0.5,
    match_by_stem: bool = True,
    pair_map: dict[str, str] | None = None,
    stem_strip_tokens: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate predicted masks against ground truth masks."""
    if not prediction_paths:
        return {"success": False, "error": "prediction_paths is required."}
    if not ground_truth_paths:
        return {"success": False, "error": "ground_truth_paths is required."}

    pred_paths = [str(p) for p in prediction_paths]
    gt_paths = [str(p) for p in ground_truth_paths]

    pairs: list[tuple[str, str]] = []
    unmatched_predictions: list[str] = []
    unmatched_ground_truth: list[str] = []

    if isinstance(pair_map, dict) and pair_map:
        matched_ground_truth: set[str] = set()
        for pred_path in pred_paths:
            target = _resolve_pair_map_target(
                pred_path,
                pair_map=pair_map,
                stem_strip_tokens=stem_strip_tokens,
            )
            if target:
                pairs.append((pred_path, str(target)))
                matched_ground_truth.add(str(target))
            else:
                unmatched_predictions.append(pred_path)
        unmatched_ground_truth = sorted([g for g in gt_paths if str(g) not in matched_ground_truth])
    elif bool(match_by_stem):
        gt_by_key: dict[str, str] = {}
        for gt_path in gt_paths:
            key = _normalized_pairing_stem(gt_path, stem_strip_tokens=stem_strip_tokens)
            if key and key not in gt_by_key:
                gt_by_key[key] = gt_path
        for pred_path in pred_paths:
            key = _normalized_pairing_stem(pred_path, stem_strip_tokens=stem_strip_tokens)
            target = gt_by_key.pop(key, None) if key else None
            if target:
                pairs.append((pred_path, target))
            else:
                unmatched_predictions.append(pred_path)
        unmatched_ground_truth = sorted(gt_by_key.values())
    else:
        n = min(len(pred_paths), len(gt_paths))
        pairs = list(zip(pred_paths[:n], gt_paths[:n]))
        unmatched_predictions = pred_paths[n:]
        unmatched_ground_truth = gt_paths[n:]

    if not pairs:
        return {
            "success": False,
            "error": "No prediction/ground-truth pairs matched. Verify filenames or disable match_by_stem.",
            "unmatched_predictions": unmatched_predictions[:100],
            "unmatched_ground_truth": unmatched_ground_truth[:100],
            "latest_result_refs": {
                "latest_eval_rows": [],
                "latest_eval_metrics_mean": {},
                "latest_eval_pairs": [],
            },
        }

    rows: list[dict[str, Any]] = []
    metric_names = ("dice", "iou", "precision", "recall")
    aggregate = {k: [] for k in metric_names}

    for pred_path, gt_path in pairs:
        try:
            pred_raw = _load_mask_array(pred_path)
            gt_raw = _load_mask_array(gt_path)
            pred = _binarize_mask_array(pred_raw, threshold=float(threshold))
            gt = _binarize_mask_array(gt_raw, threshold=float(threshold))
            metrics = _segmentation_metrics(pred, gt)
            for k in metric_names:
                aggregate[k].append(float(metrics[k]))
            rows.append(
                {
                    "prediction": pred_path,
                    "ground_truth": gt_path,
                    "shape_pred": list(np.asarray(pred_raw).shape),
                    "shape_gt": list(np.asarray(gt_raw).shape),
                    **{k: round(float(metrics[k]), 6) for k in metric_names},
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "prediction": pred_path,
                    "ground_truth": gt_path,
                    "error": str(exc),
                }
            )

    means = {k: round(float(np.mean(v)), 6) if v else None for k, v in aggregate.items()}
    success_count = sum(1 for r in rows if not r.get("error"))

    result = {
        "success": success_count > 0,
        "pairs_evaluated": len(pairs),
        "successful_pairs": int(success_count),
        "metrics_mean": means,
        "rows": rows,
        "unmatched_predictions": unmatched_predictions[:100],
        "unmatched_ground_truth": unmatched_ground_truth[:100],
        "ui_artifacts": [
            {
                "type": "metrics",
                "title": "Segmentation metrics (mean)",
                "payload": means,
            },
            {
                "type": "table",
                "title": "Per-file segmentation metrics",
                "payload": rows[:200],
            },
        ],
    }
    result["latest_result_refs"] = {
        "latest_eval_rows": rows[:200],
        "latest_eval_metrics_mean": means,
        "latest_eval_pairs": [{"prediction": p, "ground_truth": g} for p, g in pairs[:200]],
    }
    return result


def _connected_component_sizes(binary: np.ndarray, min_component_size: int = 1) -> list[int]:
    arr = np.asarray(binary).astype(bool)
    if arr.size <= 0:
        return []
    min_component_size = max(int(min_component_size), 1)
    try:
        from scipy import ndimage as ndi  # type: ignore

        structure = ndi.generate_binary_structure(arr.ndim, 1)
        labels, num = ndi.label(arr, structure=structure)
        if num <= 0:
            return []
        sizes = np.bincount(labels.ravel())[1:]
        return [int(s) for s in sizes if int(s) >= min_component_size]
    except Exception:
        # Conservative fallback: treat any foreground as a single component.
        count = int(np.count_nonzero(arr))
        if count < min_component_size:
            return []
        return [count]


def quantify_segmentation_masks(
    mask_paths: list[str],
    ground_truth_paths: list[str] | None = None,
    pair_map: dict[str, str] | None = None,
    threshold: float = 0.5,
    match_by_stem: bool = True,
    min_component_size: int = 1,
    pixel_size: float | None = None,
    pixel_unit: str = "px",
    stem_strip_tokens: list[str] | None = None,
    result_group_id: str | None = None,
) -> dict[str, Any]:
    """Quantify segmentation masks into per-mask morphology summaries."""
    if not mask_paths:
        return {"success": False, "error": "mask_paths is required."}

    rows: list[dict[str, Any]] = []
    coverage_values: list[float] = []
    object_counts: list[int] = []
    all_component_sizes: list[int] = []

    for raw_path in mask_paths:
        mask_path = str(raw_path)
        try:
            mask_raw = _load_mask_array(mask_path)
            mask_bin = _binarize_mask_array(mask_raw, threshold=float(threshold))
        except Exception as exc:
            rows.append({"mask_path": mask_path, "success": False, "error": str(exc)})
            continue

        foreground = int(np.count_nonzero(mask_bin))
        total = int(mask_bin.size)
        coverage_fraction = float(foreground / total) if total else 0.0
        coverage_percent = float(coverage_fraction * 100.0)
        component_sizes = _connected_component_sizes(
            mask_bin, min_component_size=min_component_size
        )
        object_count = int(len(component_sizes))
        mean_component_size = float(np.mean(component_sizes)) if component_sizes else 0.0
        median_component_size = float(np.median(component_sizes)) if component_sizes else 0.0

        row: dict[str, Any] = {
            "mask_path": mask_path,
            "success": True,
            "shape": list(np.asarray(mask_raw).shape),
            "foreground_voxels": foreground,
            "total_voxels": total,
            "coverage_fraction": round(coverage_fraction, 8),
            "coverage_percent": round(coverage_percent, 6),
            "object_count": object_count,
            "mean_component_size_px": round(mean_component_size, 6),
            "median_component_size_px": round(median_component_size, 6),
        }
        scale: float | None = None
        if pixel_size is not None:
            try:
                scale = float(pixel_size)
            except (TypeError, ValueError):
                scale = None
        if scale is not None and scale > 0:
            unit = str(pixel_unit or "px")
            row[f"mean_component_size_{unit}"] = round(mean_component_size * scale, 6)
            row[f"median_component_size_{unit}"] = round(median_component_size * scale, 6)

        rows.append(row)
        coverage_values.append(coverage_percent)
        object_counts.append(object_count)
        all_component_sizes.extend(component_sizes)

    success_rows = [r for r in rows if r.get("success")]
    summary = {
        "mask_count": len(mask_paths),
        "successful_masks": len(success_rows),
        "mean_coverage_percent": round(float(np.mean(coverage_values)), 6)
        if coverage_values
        else 0.0,
        "median_coverage_percent": round(float(np.median(coverage_values)), 6)
        if coverage_values
        else 0.0,
        "mean_object_count": round(float(np.mean(object_counts)), 6) if object_counts else 0.0,
        "median_object_count": round(float(np.median(object_counts)), 6) if object_counts else 0.0,
    }

    eval_result: dict[str, Any] | None = None
    eval_pairing_fallback_used = False
    if ground_truth_paths:
        eval_result = evaluate_segmentation_masks(
            prediction_paths=[str(p) for p in mask_paths],
            ground_truth_paths=[str(p) for p in ground_truth_paths],
            threshold=float(threshold),
            match_by_stem=bool(match_by_stem),
            pair_map=pair_map,
            stem_strip_tokens=stem_strip_tokens,
        )
        if (
            _is_pairing_mismatch_error(eval_result)
            and len(mask_paths) == len(ground_truth_paths)
            and not isinstance(pair_map, dict)
        ):
            eval_result = evaluate_segmentation_masks(
                prediction_paths=[str(p) for p in mask_paths],
                ground_truth_paths=[str(p) for p in ground_truth_paths],
                threshold=float(threshold),
                match_by_stem=False,
                pair_map=pair_map,
                stem_strip_tokens=stem_strip_tokens,
            )
            eval_pairing_fallback_used = True

    measurements = [
        {"name": "mean_coverage_percent", "value": summary["mean_coverage_percent"], "unit": "%"},
        {
            "name": "median_coverage_percent",
            "value": summary["median_coverage_percent"],
            "unit": "%",
        },
        {"name": "mean_object_count", "value": summary["mean_object_count"], "unit": "count"},
    ]
    if eval_result and isinstance(eval_result.get("metrics_mean"), dict):
        metrics_mean = eval_result.get("metrics_mean") or {}
        for metric_name in ("dice", "iou", "precision", "recall"):
            metric_value = metrics_mean.get(metric_name)
            if isinstance(metric_value, (int, float)):
                measurements.append(
                    {"name": f"mean_{metric_name}", "value": float(metric_value), "unit": "score"}
                )

    normalized_result_group_id = (
        str(result_group_id or "").strip()
        or _stable_result_group_id("quantify_segmentation_masks", *list(mask_paths or [])[:8])
    )
    result: dict[str, Any] = {
        "success": len(success_rows) > 0,
        "result_group_id": normalized_result_group_id,
        "summary": summary,
        "rows": rows,
        "component_size_count": len(all_component_sizes),
        "measurements": measurements,
        "evaluation": eval_result,
        "evaluation_pairing_fallback_used": bool(eval_pairing_fallback_used),
        "ui_artifacts": [
            {
                "type": "metrics",
                "title": "Mask quantification summary",
                "result_group_id": normalized_result_group_id,
                "payload": summary,
            },
            {
                "type": "table",
                "title": "Per-mask quantification",
                "result_group_id": normalized_result_group_id,
                "payload": rows[:500],
            },
        ],
    }
    if eval_result and isinstance(eval_result, dict):
        metrics_payload = eval_result.get("metrics_mean")
        if isinstance(metrics_payload, dict):
            result["ui_artifacts"].append(
                {
                    "type": "metrics",
                    "title": "Overlap metrics (mean)",
                    "result_group_id": normalized_result_group_id,
                    "payload": metrics_payload,
                }
            )
    result["latest_result_refs"] = {
        "latest_segmentation_quant_rows": rows[:500],
        "latest_segmentation_quant_summary": summary,
        "latest_eval_rows": ((eval_result or {}).get("rows") or [])[:200]
        if isinstance(eval_result, dict)
        else [],
        "latest_eval_metrics_mean": (eval_result or {}).get("metrics_mean")
        if isinstance(eval_result, dict)
        else {},
        "latest_segmentation_result_group_id": normalized_result_group_id,
    }
    return result


def segment_evaluate_batch(
    image_paths: list[str],
    ground_truth_paths: list[str],
    pair_map: dict[str, str] | None = None,
    save_visualizations: bool = True,
    model_id: str | None = None,
    device: str | None = None,
    threshold: float = 0.5,
    match_by_stem: bool = True,
    stem_strip_tokens: list[str] | None = None,
    force_rerun: bool = False,
) -> dict[str, Any]:
    """Run SAM3 segmentation then evaluate masks in one workflow call."""
    if not image_paths:
        return {"success": False, "error": "image_paths is required."}
    if not ground_truth_paths:
        return {"success": False, "error": "ground_truth_paths is required."}

    seg = segment_image_sam3(
        file_paths=[str(p) for p in image_paths],
        save_visualizations=bool(save_visualizations),
        model_id=model_id,
        device=device,
        force_rerun=bool(force_rerun),
    )
    if not seg.get("success"):
        return {
            "success": False,
            "error": seg.get("error") or "Segmentation failed.",
            "segmentation": seg,
        }

    preferred_paths = [str(p) for p in seg.get("preferred_upload_paths") or [] if p]
    if not preferred_paths:
        files_processed = (
            seg.get("files_processed") if isinstance(seg.get("files_processed"), list) else []
        )
        preferred_paths = [
            str(item.get("preferred_upload_path"))
            for item in files_processed
            if isinstance(item, dict) and item.get("preferred_upload_path")
        ]

    eval_result = evaluate_segmentation_masks(
        prediction_paths=preferred_paths,
        ground_truth_paths=[str(p) for p in ground_truth_paths],
        threshold=float(threshold),
        match_by_stem=bool(match_by_stem),
        pair_map=pair_map,
        stem_strip_tokens=stem_strip_tokens,
    )
    eval_pairing_fallback_used = False
    if (
        _is_pairing_mismatch_error(eval_result)
        and len(preferred_paths) == len(ground_truth_paths)
        and not isinstance(pair_map, dict)
    ):
        eval_result = evaluate_segmentation_masks(
            prediction_paths=preferred_paths,
            ground_truth_paths=[str(p) for p in ground_truth_paths],
            threshold=float(threshold),
            match_by_stem=False,
            pair_map=pair_map,
            stem_strip_tokens=stem_strip_tokens,
        )
        eval_pairing_fallback_used = True

    result: dict[str, Any] = {
        "success": bool(seg.get("success")) and bool(eval_result.get("success")),
        "segmentation": seg,
        "evaluation": eval_result,
        "evaluation_pairing_fallback_used": bool(eval_pairing_fallback_used),
        "prediction_paths": preferred_paths,
        "ground_truth_paths": [str(p) for p in ground_truth_paths],
        "metrics_mean": eval_result.get("metrics_mean"),
        "ui_artifacts": [],
    }
    if isinstance(seg.get("ui_artifacts"), list):
        result["ui_artifacts"].extend(seg.get("ui_artifacts"))
    if isinstance(eval_result.get("ui_artifacts"), list):
        result["ui_artifacts"].extend(eval_result.get("ui_artifacts"))
    result["latest_result_refs"] = {
        **(_tool_result_refs_for_segmentation(seg) if isinstance(seg, dict) else {}),
        "latest_eval_rows": (eval_result.get("rows") or [])[:200]
        if isinstance(eval_result, dict)
        else [],
        "latest_eval_metrics_mean": (eval_result.get("metrics_mean") or {})
        if isinstance(eval_result, dict)
        else {},
    }
    return result


def yolo_list_finetuned_models(limit: int = 10) -> dict[str, Any]:
    """List locally available finetuned YOLO checkpoints (newest first)."""
    n = max(1, min(int(limit), 100))
    entries = _read_finetuned_registry_entries(limit=n)
    models: list[dict[str, Any]] = []

    for item in entries:
        model_path = str(item.get("model_path") or "").strip()
        if not model_path or not Path(model_path).exists():
            continue
        metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
        map_value = metrics.get("map")
        if not isinstance(map_value, (int, float)):
            map_value = metrics.get("metrics/mAP50-95(B)")
        models.append(
            {
                "model_name": item.get("model_name") or Path(model_path).stem,
                "model_path": model_path,
                "base_model": item.get("base_model"),
                "created_at_utc": item.get("created_at_utc"),
                "dataset_id": item.get("dataset_id"),
                "map": float(map_value) if isinstance(map_value, (int, float)) else None,
                "class_names": item.get("class_names")
                if isinstance(item.get("class_names"), list)
                else [],
            }
        )

    if not models:
        candidates = sorted(
            Path(_finetuned_dir()).glob("*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:n]
        for candidate in candidates:
            models.append(
                {
                    "model_name": candidate.stem,
                    "model_path": str(candidate),
                    "base_model": None,
                    "created_at_utc": datetime.utcfromtimestamp(
                        candidate.stat().st_mtime
                    ).isoformat()
                    + "Z",
                    "dataset_id": None,
                    "map": None,
                    "class_names": [],
                }
            )

    latest = models[0] if models else None
    ui_artifacts: list[dict[str, Any]] = []
    if models:
        ui_artifacts.append(
            {
                "type": "table",
                "title": "Local finetuned YOLO models",
                "payload": [
                    {
                        "model_name": m.get("model_name"),
                        "created_at_utc": m.get("created_at_utc"),
                        "dataset_id": m.get("dataset_id"),
                        "map": m.get("map"),
                    }
                    for m in models
                ],
            }
        )

    return {
        "success": True,
        "count": len(models),
        "models": models,
        "latest_model_name": latest.get("model_name") if latest else None,
        "latest_model_path": latest.get("model_path") if latest else None,
        "registry_path": str(_finetuned_registry_path()),
        "ui_artifacts": ui_artifacts,
    }


def _module_emit_progress(message: str, *, event: str = "log", **extra: Any) -> None:
    try:
        from src.tooling.progress import emit_progress

        emit_progress(str(message), tool="run_bisque_module", event=event, **extra)
    except Exception:
        return


def _coerce_xml_element(value: Any) -> etree._Element:
    if value is None:
        raise ValueError("Expected XML payload but received None")
    if isinstance(value, etree._Element):
        return value
    if hasattr(value, "xmltree") and isinstance(getattr(value, "xmltree"), etree._Element):
        return getattr(value, "xmltree")
    if isinstance(value, (bytes, bytearray)):
        payload = bytes(value)
    else:
        payload = str(value).encode("utf-8", errors="ignore")
    return etree.XML(payload)


def _xml_payload_bytes(value: Any) -> bytes:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, etree._Element):
        return etree.tostring(value)
    if hasattr(value, "xmltree") and isinstance(getattr(value, "xmltree"), etree._Element):
        return etree.tostring(getattr(value, "xmltree"))
    if isinstance(value, str):
        return value.encode("utf-8")
    return str(value).encode("utf-8", errors="ignore")


def _session_postxml_safe(
    bq: Any,
    url: str,
    xml: Any,
    *,
    method: str = "POST",
    **params: Any,
) -> etree._Element:
    try:
        return _coerce_xml_element(bq.postxml(url, xml, method=method, **params))
    except Exception:
        comm = getattr(bq, "c", None)
        if comm is None or not hasattr(comm, "prepare_url") or not hasattr(comm, "push"):
            raise
        request_url = comm.prepare_url(url, **params)
        raw = comm.push(
            request_url,
            content=_xml_payload_bytes(xml),
            method=method,
            headers={"Content-Type": "text/xml", "Accept": "text/xml"},
        )
        return _coerce_xml_element(raw)


def _session_fetchxml_safe(
    bq: Any,
    url: str,
    **params: Any,
) -> etree._Element:
    try:
        return _coerce_xml_element(bq.fetchxml(url, **params))
    except Exception:
        comm = getattr(bq, "c", None)
        if comm is None or not hasattr(comm, "prepare_url") or not hasattr(comm, "fetch"):
            raise
        request_url = comm.prepare_url(url, **params)
        raw = comm.fetch(
            request_url,
            headers={"Content-Type": "text/xml", "Accept": "text/xml"},
        )
        return _coerce_xml_element(raw)


def _looks_like_bisque_resource(value: str) -> bool:
    token = str(value or "").strip()
    if not token:
        return False
    if token.startswith(("http://", "https://", "/data_service/", "/image_service/", "/resource/")):
        return True
    return bool(re.fullmatch(r"00-[A-Za-z0-9]+", token))


def _safe_temp_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name or "").strip())
    cleaned = cleaned.strip("._")
    return cleaned or "uploaded.bin"


def _resolve_uploaded_local_path(
    resource_value: str,
    *,
    uploaded_files: list[Any] | None,
    staging_dir: Path,
) -> Path | None:
    token = str(resource_value or "").strip()
    if not token:
        return None

    direct = Path(token).expanduser()
    if direct.exists() and direct.is_file():
        return direct.resolve()

    requested_name = Path(token).name
    requested_name_lower = requested_name.lower()
    requested_canonical = _canonical_uploaded_filename(requested_name).lower()

    for item in uploaded_files or []:
        if isinstance(item, str):
            candidate = Path(item).expanduser()
            if not candidate.exists() or not candidate.is_file():
                continue
            aliases = {
                candidate.name.lower(),
                _canonical_uploaded_filename(candidate.name).lower(),
            }
            candidate_resolved = str(candidate.resolve())
            if (
                token == str(candidate)
                or token == candidate_resolved
                or requested_name_lower in aliases
                or requested_canonical in aliases
            ):
                return candidate.resolve()
            continue

        file_name = str(getattr(item, "name", "") or "").strip()
        if not file_name or not hasattr(item, "getvalue"):
            continue
        aliases = {
            file_name.lower(),
            _canonical_uploaded_filename(file_name).lower(),
        }
        if requested_name_lower not in aliases and requested_canonical not in aliases:
            continue
        staging_dir.mkdir(parents=True, exist_ok=True)
        out_path = staging_dir / f"{uuid4().hex[:8]}__{_safe_temp_filename(file_name)}"
        out_path.write_bytes(item.getvalue())
        return out_path.resolve()

    fallback_candidates = [
        staging_dir / requested_name,
        _science_data_root_path() / requested_name,
    ]
    for candidate in fallback_candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    return None


def _build_module_mex_payload(
    *,
    processed_inputs: dict[str, str],
    module_params: dict[str, str] | None,
) -> etree._Element:
    mex = etree.Element("mex")
    inputs_tag = etree.SubElement(mex, "tag", name="inputs")
    for input_name, input_value in processed_inputs.items():
        etree.SubElement(
            inputs_tag,
            "tag",
            name=str(input_name),
            type="resource",
            value=str(input_value),
        )
    for param_name, param_value in (module_params or {}).items():
        etree.SubElement(
            inputs_tag,
            "tag",
            name=str(param_name),
            value=str(param_value),
        )
    return mex


def _normalize_module_input_names(
    module_name: str,
    input_resources: dict[str, str],
) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in (input_resources or {}).items():
        name = str(key or "").strip()
        token = str(value or "").strip()
        if name and token:
            normalized[name] = token

    module = str(module_name or "").strip().lower()
    if module != "edgedetection":
        return normalized

    if "Input Image" in normalized:
        return normalized

    alias_order = [
        "input image",
        "input_image",
        "inputimage",
        "image",
        "image_url",
        "input",
    ]
    lowered_to_original = {k.lower(): k for k in normalized}
    for alias in alias_order:
        match = lowered_to_original.get(alias)
        if match:
            value = normalized.pop(match)
            normalized["Input Image"] = value
            return normalized

    if len(normalized) == 1:
        only_value = next(iter(normalized.values()))
        return {"Input Image": only_value}

    return normalized


def _collect_mex_error_messages(mex_tree: etree._Element) -> list[str]:
    messages: list[str] = []
    for attr in ("error_message", "error", "message", "stderr", "detail"):
        value = str(mex_tree.get(attr) or "").strip()
        if value:
            messages.append(value)
    for node in mex_tree.xpath(
        './/tag[@name="error_message" or @name="error" or @name="stderr" or @name="message" or @name="detail"]'
    ):
        value = str(node.get("value") or "").strip()
        if value:
            messages.append(value)
        text = str(node.text or "").strip()
        if text:
            messages.append(text)
    deduped: list[str] = []
    seen: set[str] = set()
    for message in messages:
        key = message.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(message)
    return deduped


def _extract_mex_status(mex_tree: etree._Element) -> str:
    candidates: list[str] = [
        str(mex_tree.get("value") or "").strip(),
        str(mex_tree.get("status") or "").strip(),
    ]
    for node in mex_tree.xpath(
        './/tag[@name="status" or @name="state" or @name="mex_status" or @name="execution_status"]'
    ):
        candidates.append(str(node.get("value") or "").strip())
        candidates.append(str(node.text or "").strip())
    for raw in candidates:
        value = str(raw or "").strip().upper()
        if value:
            return value
    return "UNKNOWN"


def _extract_mex_uri(payload: Any) -> str | None:
    tree = _coerce_xml_element(payload)
    candidates: list[str] = [str(tree.get("uri") or "").strip()]
    for node in tree.xpath(".//mex[@uri]"):
        candidates.append(str(node.get("uri") or "").strip())
    for node in tree.xpath('.//tag[@name="mex_url" or @name="mex_uri"]'):
        candidates.append(str(node.get("value") or "").strip())
    for candidate in candidates:
        value = str(candidate or "").strip()
        if not value:
            continue
        if "/module_service/" in value or "/data_service/mex" in value:
            return value
        if value.startswith("/"):
            return value
        if value.startswith(("http://", "https://")):
            return value
    return None


def _iter_xml_resource_candidates(node: etree._Element) -> list[str]:
    out: list[str] = []
    for attr in ("value", "uri", "resource", "resource_uri", "href"):
        value = str(node.get(attr) or "").strip()
        if value:
            out.append(value)
    text = str(node.text or "").strip()
    if text:
        out.append(text)
    return out


def _looks_like_bisque_output_resource(value: str) -> bool:
    token = str(value or "").strip()
    if not token:
        return False
    if _looks_like_bisque_resource(token):
        return True
    lowered = token.lower()
    return "/data_service/" in lowered or "/image_service/" in lowered


def _is_terminal_mex_status(status: str) -> bool:
    normalized = str(status or "").strip().upper()
    return normalized in {
        "FINISHED",
        "COMPLETED",
        "DONE",
        "SUCCESS",
        "FAILED",
        "FAILURE",
        "ERROR",
        "CANCELED",
        "CANCELLED",
        "TERMINATED",
    }


def _is_success_mex_status(status: str) -> bool:
    return str(status or "").strip().upper() in {
        "FINISHED",
        "COMPLETED",
        "DONE",
        "SUCCESS",
    }


def _canonical_mex_status(status: str) -> str:
    normalized = str(status or "").strip().upper()
    if normalized in {"COMPLETED", "DONE", "SUCCESS"}:
        return "FINISHED"
    if normalized in {"FAILURE", "ERROR"}:
        return "FAILED"
    return normalized or "UNKNOWN"


def _poll_mex_until_terminal(
    bq: Any,
    *,
    mex_uri: str,
    timeout_seconds: int,
    poll_interval_seconds: float,
) -> tuple[etree._Element, list[str]]:
    timeout = max(5, int(timeout_seconds))
    interval = max(0.25, float(poll_interval_seconds))
    deadline = time.time() + timeout

    statuses: list[str] = []
    last_status: str | None = None
    while time.time() <= deadline:
        mex_tree = _session_fetchxml_safe(bq, mex_uri, view="deep")
        status = _canonical_mex_status(_extract_mex_status(mex_tree))
        if status != last_status:
            statuses.append(status)
            _module_emit_progress(
                f"MEX status: {status}",
                event="mex_status",
                mex_url=mex_uri,
                mex_status=status,
            )
            last_status = status
        if _is_terminal_mex_status(status):
            return mex_tree, statuses
        time.sleep(interval)

    raise TimeoutError(f"MEX timeout after {timeout}s (last_status={last_status or 'UNKNOWN'})")


def _extract_output_resource_uri_from_mex(mex_tree: etree._Element) -> str | None:
    candidates: list[str] = []
    xpaths = [
        './tag[@name="outputs"]//tag[@type="image" or @type="resource" or @type="file"]',
        './tag[@name="outputs"]//image',
        './tag[@name="outputs"]//tag',
        './/tag[@name="outputs"]//tag',
        ".//outputs//tag",
        './/tag[@name="outputs"]//*',
    ]
    for expr in xpaths:
        for node in mex_tree.xpath(expr):
            if not isinstance(node, etree._Element):
                continue
            candidates.extend(_iter_xml_resource_candidates(node))

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        value = str(candidate or "").strip()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)

    for candidate in deduped:
        if _looks_like_bisque_output_resource(candidate):
            return candidate
    return None


def _download_module_output(
    bq: Any,
    *,
    output_resource_uri: str,
    bisque_root: str,
    output_dir: Path,
) -> str | None:
    try:
        from bqapi.util import fetch_blob
    except Exception:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_uri = _normalize_bisque_resource_uri(output_resource_uri, bisque_root)
    try:
        downloaded = fetch_blob(bq, normalized_uri, dest=str(output_dir))
        if isinstance(downloaded, dict):
            direct = downloaded.get(normalized_uri)
            if direct:
                return str(Path(str(direct)).resolve())
            for _, value in downloaded.items():
                if value:
                    return str(Path(str(value)).resolve())
    except Exception:
        return None
    return None


def run_bisque_module(
    module_name: str,
    input_resources: dict[str, str],
    module_params: dict[str, str] | None = None,
    bisque_user: str | None = None,
    bisque_password: str | None = None,
    bisque_root: str | None = None,
    uploaded_files: list[Any] | None = None,
    wait_for_completion: bool = True,
    timeout_seconds: int = 600,
    poll_interval_seconds: float = 2.0,
    download_output: bool = True,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """
    Execute a BisQue module using token-first auth with robust MEX polling/output capture.
    """
    try:
        from bqapi.comm import BQSession
    except ImportError:
        return {
            "success": False,
            "error": "bqapi package not installed. Install with: pip install bqapi",
        }

    user, password, access_token, root, cookie_header = _resolve_bisque_runtime_auth(
        explicit_user=str(bisque_user or ""),
        explicit_password=str(bisque_password or ""),
        explicit_root=str(bisque_root or ""),
    )
    root = str(root or "").strip()
    module = str(module_name or "").strip()

    if not module:
        return {
            "success": False,
            "error": "module_name is required.",
        }
    if not root:
        return {
            "success": False,
            "error": "BisQue root URL required. Set BISQUE_ROOT in .env",
        }

    requested_inputs: dict[str, str] = {}
    if isinstance(input_resources, dict):
        for key, value in input_resources.items():
            input_name = str(key or "").strip()
            if not input_name:
                continue
            input_value = str(value or "").strip()
            if input_value:
                requested_inputs[input_name] = input_value
    requested_inputs = _normalize_module_input_names(module, requested_inputs)

    if not requested_inputs and uploaded_files:
        first = uploaded_files[0]
        if isinstance(first, str):
            requested_inputs = {"Input Image": Path(first).name}
        elif hasattr(first, "name"):
            requested_inputs = {"Input Image": str(getattr(first, "name") or "input").strip()}

    if not requested_inputs:
        return {
            "success": False,
            "error": "input_resources is required. Provide at least one module input (for example {'Input Image': '<resource or file>'}).",
        }

    staged_inputs_dir = Path(_science_output_root("module_inputs"))
    module_out_dir = (
        Path(output_dir).expanduser()
        if output_dir
        else Path(_science_output_root("bisque_module_outputs", _safe_slug(module)))
    )

    bq: Any | None = None
    auth_mode = "unknown"
    token_mode = False
    mex_url: str | None = None
    try:
        _module_emit_progress("Initializing BisQue session…", event="init")
        bq = BQSession()
        token_mode, auth_mode = _init_bq_session(
            bq,
            username=user,
            password=password,
            access_token=access_token,
            bisque_root=root,
            cookie_header=cookie_header,
        )

        processed_inputs: dict[str, str] = {}
        for input_name, raw_value in requested_inputs.items():
            token = str(raw_value or "").strip()
            if _looks_like_bisque_resource(token):
                processed_inputs[input_name] = _normalize_bisque_resource_uri(token, root)
                continue

            local_path = _resolve_uploaded_local_path(
                token,
                uploaded_files=uploaded_files,
                staging_dir=staged_inputs_dir,
            )
            if local_path is None or not local_path.exists() or not local_path.is_file():
                return {
                    "success": False,
                    "error": f"File not found for input '{input_name}': {token}",
                    "module_name": module,
                }

            _module_emit_progress(
                f"Uploading input for {input_name}: {local_path.name}",
                event="upload_input",
                input_name=input_name,
            )
            resource_xml = etree.Element("image", name=local_path.name)
            upload_response = bq.postblob(str(local_path), xml=resource_xml)
            upload_uri = _extract_bisque_resource_uri(upload_response)
            if not upload_uri:
                upload_element = _coerce_xml_element(upload_response)
                upload_uri = str(upload_element.get("uri") or "").strip()
                if not upload_uri:
                    child = upload_element.find("./*")
                    upload_uri = str(child.get("uri") if child is not None else "").strip()
            if not upload_uri:
                return {
                    "success": False,
                    "error": f"Failed to upload local input '{local_path}'.",
                    "module_name": module,
                }
            processed_inputs[input_name] = _normalize_bisque_resource_uri(upload_uri, root)

        mex_payload = _build_module_mex_payload(
            processed_inputs=processed_inputs,
            module_params=module_params,
        )
        _module_emit_progress(
            f"Submitting module execution for {module}…",
            event="submit",
            module_name=module,
        )
        execute_response = _session_postxml_safe(
            bq,
            f"/module_service/{module}/execute",
            mex_payload,
        )
        mex_url_raw = _extract_mex_uri(execute_response) or ""
        mex_url = _normalize_bisque_url(mex_url_raw, root) if mex_url_raw else ""
        if not mex_url:
            return {
                "success": False,
                "error": f"Module '{module}' did not return a MEX URI in execute response.",
                "module_name": module,
            }
        mex_id = _short_bisque_id(mex_url)

        if not wait_for_completion:
            return {
                "success": True,
                "module_name": module,
                "mex_url": mex_url,
                "mex_id": mex_id,
                "status": "RUNNING",
                "auth_mode": auth_mode,
                "token_mode": token_mode,
                "inputs": processed_inputs,
                "message": f"Module {module} submitted. MEX is running at {mex_url}.",
            }

        final_mex, status_chain = _poll_mex_until_terminal(
            bq,
            mex_uri=mex_url,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )
        final_status = _canonical_mex_status(_extract_mex_status(final_mex))

        if not _is_success_mex_status(final_status):
            errors = _collect_mex_error_messages(final_mex)
            return {
                "success": False,
                "module_name": module,
                "mex_url": mex_url,
                "mex_id": mex_id,
                "status": final_status,
                "status_chain": status_chain,
                "errors": errors,
                "error": (
                    f"Module execution failed with status={final_status}"
                    + (f"; errors={errors[:3]}" if errors else "")
                ),
                "inputs": processed_inputs,
            }

        output_resource_uri = _extract_output_resource_uri_from_mex(final_mex)
        if not output_resource_uri:
            return {
                "success": False,
                "module_name": module,
                "mex_url": mex_url,
                "mex_id": mex_id,
                "status": final_status,
                "status_chain": status_chain,
                "error": "Module finished but no output image/resource URI was found in MEX outputs.",
                "inputs": processed_inputs,
            }
        output_links = _build_bisque_resource_links(output_resource_uri, root)
        output_resource_uri_norm = str(output_links.get("resource_uri") or "").strip()

        local_output_path: str | None = None
        if download_output and output_resource_uri_norm:
            _module_emit_progress("Downloading module output…", event="download_output")
            local_output_path = _download_module_output(
                bq,
                output_resource_uri=output_resource_uri_norm,
                bisque_root=root,
                output_dir=module_out_dir,
            )

        ui_artifacts: list[dict[str, Any]] = [
            {
                "type": "summary",
                "title": f"{module} execution complete",
                "payload": (
                    f"MEX `{mex_id or mex_url}` finished. "
                    f"Output resource: `{str(output_links.get('client_view_url') or output_resource_uri_norm).strip()}`."
                ),
            }
        ]
        if local_output_path:
            ui_artifacts.append(
                {
                    "type": "image",
                    "title": f"{module} output",
                    "path": local_output_path,
                }
            )

        result: dict[str, Any] = {
            "success": True,
            "module_name": module,
            "mex_url": mex_url,
            "mex_id": mex_id,
            "status": final_status,
            "status_chain": status_chain,
            "auth_mode": auth_mode,
            "token_mode": token_mode,
            "inputs": processed_inputs,
            "output_resource_uri": output_resource_uri_norm,
            "output_resource_uniq": output_links.get("resource_uniq"),
            "output_client_view_url": output_links.get("client_view_url"),
            "output_image_service_url": output_links.get("image_service_url"),
            "output_path": local_output_path,
            "output_local_path": local_output_path,
            "preferred_upload_path": local_output_path,
            "output_directory": str(module_out_dir.resolve()) if local_output_path else None,
            "latest_result_refs": {
                "latest_module_output_path": local_output_path,
                "latest_module_output_resource_uri": output_resource_uri_norm,
                "latest_module_mex_url": mex_url,
            },
            "ui_artifacts": ui_artifacts,
            "message": (
                f"Module {module} completed successfully."
                + (f" Output downloaded to {local_output_path}." if local_output_path else "")
            ),
        }
        return result
    except Exception as exc:
        logger.exception("Error running BisQue module %s", module)
        return {
            "success": False,
            "module_name": module,
            "mex_url": mex_url,
            "error": f"Exception running module: {exc}",
        }
    finally:
        if bq is not None and hasattr(bq, "close"):
            try:
                bq.close()
            except Exception:
                pass


def segment_video_sam2(
    file_paths: list[str],
    track_points: list[list[int]] = None,
    save_visualizations: bool = True,
    device: str | None = None,
    model_id: str | None = None,
) -> dict[str, Any]:
    """
    Segment and track objects in videos using SAM2 (Segment Anything Model 2).
    Automatically tracks objects through all frames with point-based initialization.

    Args:
        file_paths: List of video file paths to segment
        track_points: Optional list of [x, y] coordinates to track. If None, uses center point.
        save_visualizations: Whether to save video visualization overlays (default: True)

    Returns:
        Dictionary with segmentation results and metadata
    """
    try:
        import cv2
        import torch
        from transformers import Sam2VideoModel, Sam2VideoProcessor
        from transformers.video_utils import load_video
    except ImportError:
        return {
            "success": False,
            "error": "Required packages not installed. Install with: pip install transformers opencv-python torch",
        }

    settings = get_settings()
    requested_model = str(model_id or getattr(settings, "medsam2_model_id", "wanglab/MedSAM2"))
    torch_device = "cpu"
    fallback_used = False
    try:
        # Initialize SAM-family video model with GPU if available.
        logger.info("Initializing video segmentation model: %s", requested_model)
        torch_device = _resolve_torch_device(device)
        dtype = torch.float16 if torch_device.startswith("cuda") else torch.float32
        try:
            model = Sam2VideoModel.from_pretrained(
                requested_model,
                dtype=dtype,
                low_cpu_mem_usage=False,
            ).to(torch_device)
            processor = Sam2VideoProcessor.from_pretrained(requested_model)
            resolved_model = requested_model
        except Exception:
            fallback_model = "facebook/sam2-hiera-large"
            model = Sam2VideoModel.from_pretrained(
                fallback_model,
                dtype=dtype,
                low_cpu_mem_usage=False,
            ).to(torch_device)
            processor = Sam2VideoProcessor.from_pretrained(fallback_model)
            resolved_model = fallback_model
            fallback_used = True
        logger.info("Video model initialized on %s", torch_device)
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to initialize video segmentation model: {str(e)}. Ensure transformers and torch are installed with CUDA support.",
        }

    # Create output directory for visualizations
    output_dir = os.path.join("data", "sam2_video_results")
    os.makedirs(output_dir, exist_ok=True)

    results = []

    for file_path in file_paths:
        try:
            # Validate file exists
            if not os.path.exists(file_path):
                results.append(
                    {
                        "file": os.path.basename(file_path),
                        "success": False,
                        "error": f"File not found: {file_path}",
                    }
                )
                continue

            file_name = os.path.basename(file_path)
            base_name = os.path.splitext(file_name)[0]
            logger.info(f"Processing video {file_name} with SAM2")

            # Load video frames
            video_frames, video_metadata = load_video(file_path)
            num_frames = len(video_frames)

            # Extract FPS from metadata
            if hasattr(video_metadata, "fps"):
                fps = float(video_metadata.fps)
            elif hasattr(video_metadata, "frame_rate"):
                fps = float(video_metadata.frame_rate)
            else:
                # Default to 30 fps if not available
                fps = 30.0
                logger.warning(f"Could not extract FPS from video metadata, using default: {fps}")

            logger.info(f"Loaded {num_frames} frames from {file_name} (FPS: {fps})")

            # Initialize video inference session
            inference_session = processor.init_video_session(
                video=video_frames,
                inference_device=torch_device,
                dtype=dtype,
            )

            # Determine tracking point (use center if not specified)
            video_height = inference_session.video_height
            video_width = inference_session.video_width

            if track_points is None or len(track_points) == 0:
                # Use center point by default
                track_points = [[video_width // 2, video_height // 2]]

            logger.info(f"Tracking {len(track_points)} point(s)")
            track_x, track_y = track_points[0]

            # Add click on first frame to select object
            ann_frame_idx = 0
            ann_obj_id = 1
            points = [[track_points]]
            labels = [[[1 for _ in track_points]]]  # 1 = foreground point

            processor.add_inputs_to_inference_session(
                inference_session=inference_session,
                frame_idx=ann_frame_idx,
                obj_ids=ann_obj_id,
                input_points=points,
                input_labels=labels,
            )

            # Segment the object on the first frame
            model(
                inference_session=inference_session,
                frame_idx=ann_frame_idx,
            )

            # Propagate through the entire video
            logger.info(f"Propagating segmentation through {num_frames} frames...")
            video_segments = {}
            for sam2_video_output in model.propagate_in_video_iterator(inference_session):
                video_res_masks = processor.post_process_masks(
                    [sam2_video_output.pred_masks],
                    original_sizes=[[video_height, video_width]],
                    binarize=False,
                )[0]
                video_segments[sam2_video_output.frame_idx] = video_res_masks

            logger.info(f"Successfully tracked object through {len(video_segments)} frames")

            # Create visualization if requested
            visualization_path = None
            if save_visualizations:
                try:
                    vis_filename = f"{base_name}_sam2_tracked.mp4"
                    visualization_path = os.path.join(output_dir, vis_filename)

                    # Get frame dimensions from first frame (already numpy array)
                    first_frame = video_frames[0]
                    if isinstance(first_frame, np.ndarray):
                        frame_height, frame_width = first_frame.shape[:2]
                    else:
                        # Fallback if it's a PIL Image
                        first_frame_array = np.array(first_frame.convert("RGB"))
                        frame_height, frame_width = first_frame_array.shape[:2]

                    # Use H264 codec for better compatibility
                    fourcc = cv2.VideoWriter_fourcc(*"avc1")  # H264 codec
                    out_video = cv2.VideoWriter(
                        visualization_path, fourcc, float(fps), (frame_width, frame_height)
                    )

                    if not out_video.isOpened():
                        logger.warning("Failed to open video writer with avc1, trying mp4v")
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        out_video = cv2.VideoWriter(
                            visualization_path, fourcc, float(fps), (frame_width, frame_height)
                        )

                    # Generate color for the tracked object
                    np.random.seed(42)
                    color = np.random.randint(0, 255, size=3).tolist()

                    logger.info(f"Writing {len(video_frames)} frames to {visualization_path}...")

                    # Apply masks to each frame
                    for frame_idx, frame in enumerate(video_frames):
                        # Handle both numpy arrays and PIL Images
                        if isinstance(frame, np.ndarray):
                            # Already numpy array - ensure it's RGB and correct shape
                            if len(frame.shape) == 3 and frame.shape[2] == 3:
                                # Assume it's already RGB
                                frame_array = frame.astype(np.uint8)
                            elif len(frame.shape) == 3 and frame.shape[2] == 4:
                                # RGBA - convert to RGB
                                frame_array = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                            else:
                                # Grayscale or other - convert to RGB
                                frame_array = (
                                    cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                                    if len(frame.shape) == 2
                                    else frame
                                )
                        else:
                            # PIL Image - convert to numpy
                            frame_array = np.array(frame.convert("RGB"))

                        # Ensure frame dimensions match expected dimensions
                        if (
                            frame_array.shape[0] != frame_height
                            or frame_array.shape[1] != frame_width
                        ):
                            logger.warning(
                                f"Frame {frame_idx} size mismatch. Expected ({frame_height}, {frame_width}), got {frame_array.shape[:2]}. Resizing..."
                            )
                            frame_array = cv2.resize(frame_array, (frame_width, frame_height))

                        if frame_idx in video_segments:
                            # Get mask for this frame
                            mask = video_segments[frame_idx][0, 0].cpu().numpy()  # [H, W]

                            # Ensure mask dimensions match frame dimensions
                            if mask.shape[0] != frame_height or mask.shape[1] != frame_width:
                                logger.warning(
                                    f"Mask size mismatch at frame {frame_idx}. Resizing mask from {mask.shape} to ({frame_height}, {frame_width})"
                                )
                                mask = cv2.resize(mask, (frame_width, frame_height))

                            mask_binary = (mask > 0).astype(np.uint8)

                            # Create colored overlay
                            overlay = frame_array.copy().astype(np.float32)
                            for c in range(3):
                                overlay[:, :, c] = np.where(
                                    mask_binary,
                                    overlay[:, :, c] * 0.5 + color[c] * 0.5,
                                    overlay[:, :, c],
                                )

                            # Draw tracking point on first frame
                            if frame_idx == 0:
                                overlay_uint8 = overlay.astype(np.uint8)
                                cv2.circle(overlay_uint8, (track_x, track_y), 8, (0, 0, 255), -1)
                                cv2.circle(
                                    overlay_uint8, (track_x, track_y), 10, (255, 255, 255), 2
                                )
                                frame_array = overlay_uint8
                            else:
                                frame_array = overlay.astype(np.uint8)

                        # Ensure frame is uint8 and correct shape before writing
                        frame_array = frame_array.astype(np.uint8)

                        # Convert RGB to BGR for OpenCV
                        frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)

                        # Verify frame shape before writing
                        if frame_bgr.shape[0] != frame_height or frame_bgr.shape[1] != frame_width:
                            logger.error(
                                f"Frame {frame_idx} shape mismatch before writing: {frame_bgr.shape}"
                            )
                            frame_bgr = cv2.resize(frame_bgr, (frame_width, frame_height))

                        out_video.write(frame_bgr)

                    # Properly close the video writer
                    out_video.release()
                    cv2.destroyAllWindows()

                    # Verify the file was created and has size
                    if os.path.exists(visualization_path):
                        file_size = os.path.getsize(visualization_path)
                        if file_size > 0:
                            logger.info(
                                f"Successfully saved video visualization to {visualization_path} ({file_size:,} bytes)"
                            )
                        else:
                            logger.error(f"Video file created but is empty: {visualization_path}")
                            visualization_path = None
                    else:
                        logger.error(f"Video file was not created: {visualization_path}")
                        visualization_path = None

                except Exception as vis_error:
                    logger.error(f"Failed to create video visualization: {str(vis_error)}")
                    import traceback

                    logger.error(traceback.format_exc())
                    visualization_path = None

            # Compile result
            result_data = {
                "file": file_name,
                "path": file_path,
                "success": True,
                "video_info": {
                    "width": video_width,
                    "height": video_height,
                    "frames": num_frames,
                    "fps": fps,
                },
                "tracking": {
                    "point": [track_x, track_y],
                    "frames_tracked": len(video_segments),
                    "tracking_success_rate": round(len(video_segments) / num_frames * 100, 1),
                },
                "visualization": visualization_path,
            }

            results.append(result_data)
            logger.info(
                f"Successfully segmented video {file_name}: tracked through {len(video_segments)} frames"
            )

        except Exception as file_error:
            results.append(
                {
                    "file": os.path.basename(file_path),
                    "path": file_path,
                    "success": False,
                    "error": str(file_error),
                }
            )
            logger.error(f"Failed to segment video {file_path}: {str(file_error)}")

    # Generate summary statistics
    successful = [r for r in results if r.get("success")]
    total_frames = sum(r["video_info"]["frames"] for r in successful)

    # Create minimal summary for LLM
    results_summary = []
    visualization_paths = []

    for r in results:
        if r.get("success"):
            results_summary.append(
                {
                    "file": r["file"],
                    "success": True,
                    "frames": r["video_info"]["frames"],
                    "tracking_rate": r["tracking"]["tracking_success_rate"],
                    "visualization_saved": r.get("visualization") is not None,
                }
            )
            if r.get("visualization"):
                visualization_paths.append(
                    {
                        "path": r["visualization"],
                        "file": r["file"],
                        "frames": r["video_info"]["frames"],
                        "tracking_rate": r["tracking"]["tracking_success_rate"],
                    }
                )
        else:
            results_summary.append(
                {"file": r["file"], "success": False, "error": r.get("error", "Unknown error")}
            )

    return {
        "success": len(successful) > 0,
        "processed": len(successful),
        "total_files": len(results),
        "total_frames_processed": total_frames,
        "files_processed": results_summary,
        "output_directory": output_dir,
        "model": resolved_model,
        "fallback_used": fallback_used,
        "device": torch_device,
        "visualization_paths": visualization_paths,
    }


def yolo_detect(
    file_paths: list[str],
    model_name: str | None = None,
    model_path: str | None = None,
    use_latest_finetuned_if_available: bool = False,
    conf: float = 0.25,
    iou: float = 0.7,
    include_stability_audit: bool = False,
    stability_top_k: int = 3,
    stability_preservation_ratio: float = SPECTRAL_DEFAULT_PRESERVATION_RATIO,
    save_visualizations: bool = True,
) -> dict[str, Any]:
    """Run YOLO detection on one or more images.

    Uses a finetuned model if `model_name` exists in data/models/yolo/finetuned.
    Otherwise defaults to a pretrained model (default: yolo26x.pt, configurable via YOLO_DEFAULT_MODEL).
    """
    try:
        yolo_class = _require_ultralytics()
    except ImportError as e:
        return {"success": False, "error": str(e)}

    if not file_paths:
        return {"success": False, "error": "file_paths is required"}

    try:
        expanded = _expand_file_inputs([str(p) for p in file_paths])
    except Exception as e:
        return {"success": False, "error": str(e)}

    expanded, sequence_expansions, sequence_warnings = _expand_sequence_inputs_for_2d_models(
        expanded
    )

    missing = [str(p) for p in expanded if not p.exists()]
    if missing:
        return {"success": False, "error": "Some files do not exist", "missing": missing}

    images = [p for p in expanded if _is_yolo_image(p)]
    non_images = [str(p) for p in expanded if p.exists() and not _is_yolo_image(p)]
    if not images:
        return {"success": False, "error": "No images found in file_paths."}

    paths = [str(p) for p in images]

    run_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
    out_dir_path = Path(_science_output_root("yolo", "predictions")) / run_id
    out_dir_path.mkdir(parents=True, exist_ok=True)
    out_dir = str(out_dir_path.resolve())

    warnings: dict[str, Any] = {}
    if sequence_expansions:
        warnings["sequence_expansions"] = sequence_expansions
    if sequence_warnings:
        warnings["sequence_warnings"] = sequence_warnings

    prairie_model_requested = _is_prairie_yolo_alias(model_name)
    resolved_model_name = model_name
    explicit_model_path = str(model_path).strip() if model_path else None
    if prairie_model_requested and not explicit_model_path:
        prairie_model_name, prairie_model_path, prairie_resolution = (
            _resolve_prairie_yolo_model_target()
        )
        if prairie_model_path:
            explicit_model_path = prairie_model_path
            resolved_model_name = prairie_model_name or _PRAIRIE_YOLO_MODEL_KEY
            warnings["prairie_model_resolution"] = {
                "reason": prairie_resolution,
                "model_name": resolved_model_name,
                "model_path": prairie_model_path,
            }

    if use_latest_finetuned_if_available and not model_name and not explicit_model_path:
        latest_name, latest_path = _latest_finetuned_model()
        if latest_path:
            model_name = "latest"
            resolved_model_name = latest_name or "latest"
            warnings["auto_selected_model"] = {
                "reason": "No model specified; using latest local finetuned model.",
                "model_name": latest_name,
                "model_path": latest_path,
            }

    weights = _resolve_model_path(model_name=model_name, model_path=explicit_model_path)
    resolved_weights = weights

    candidate_paths: list[str] = [weights]
    if not model_name and not explicit_model_path:
        for candidate in _default_yolo_fallback_candidates():
            if candidate not in candidate_paths:
                candidate_paths.append(candidate)
    latest_name, latest_path = _latest_finetuned_model()
    if latest_path and latest_path not in candidate_paths:
        if explicit_model_path:
            signature = _checkpoint_file_signature(Path(explicit_model_path))
            if not prairie_model_requested and (
                signature.get("looks_html_or_xml")
                or signature.get("hint") == "File does not exist."
            ):
                candidate_paths.append(latest_path)
                warnings["model_path_invalid_fallback"] = {
                    "provided_model_path": explicit_model_path,
                    "reason": signature.get("hint") or "Downloaded file is not a valid checkpoint.",
                    "fallback_model_name": latest_name,
                    "fallback_model_path": latest_path,
                }
        elif use_latest_finetuned_if_available:
            candidate_paths.append(latest_path)

    model = None
    load_errors: list[str] = []
    legacy_runtime_used = False
    legacy_inference_backend: str | None = None
    legacy_inference_configuration: dict[str, Any] = {}
    legacy_logs: dict[str, Any] = {}
    for idx, candidate in enumerate(candidate_paths):
        signature = _checkpoint_file_signature(Path(candidate))
        if signature.get("looks_html_or_xml"):
            load_errors.append(f"{candidate}: {signature.get('hint') or 'invalid checkpoint file'}")
            continue
        try:
            model = yolo_class(candidate)
            resolved_weights = candidate
            if idx > 0:
                warnings["model_fallback_used"] = {
                    "fallback_model_path": candidate,
                    "fallback_model_name": Path(candidate).stem,
                }
            break
        except Exception as exc:  # noqa: BLE001
            load_errors.append(f"{candidate}: {exc}")

    prairie_tiled_runtime_forced = prairie_model_requested
    if prairie_tiled_runtime_forced and model is not None:
        warnings["legacy_runtime_forced"] = {
            "reason": "prairie_model_uses_tiled_legacy_runtime",
            "model_name": resolved_model_name or _PRAIRIE_YOLO_MODEL_KEY,
            "tile_size": 512,
            "tile_overlap": 0.25,
        }
        model = None

    predictions: list[dict[str, Any]] = []
    counts_by_class: dict[str, int] = {}
    conf_values: list[float] = []
    results = None

    if model is None:
        attempted_existing = [
            str(Path(candidate).expanduser().resolve())
            for candidate in candidate_paths
            if Path(candidate).expanduser().exists()
        ]
        should_try_legacy = bool(prairie_model_requested) or any(
            _looks_like_legacy_yolov5_checkpoint_error(error) for error in load_errors
        )
        if should_try_legacy and attempted_existing:
            legacy_model_path = attempted_existing[0]
            try:
                legacy_result = _run_legacy_yolov5_detection(
                    model_artifact_path=legacy_model_path,
                    input_paths=paths,
                    output_dir=out_dir_path,
                    conf=float(conf),
                    iou=float(iou),
                )
                legacy_runtime_used = True
                legacy_inference_backend = (
                    str(legacy_result.get("inference_backend") or "yolov5_detect_tiled").strip()
                    or "yolov5_detect_tiled"
                )
                legacy_logs = {
                    key: legacy_result.get(key)
                    for key in ("stdout_log_path", "stderr_log_path")
                    if legacy_result.get(key)
                }
                legacy_inference_configuration = {
                    "tile_size": legacy_result.get("tile_size"),
                    "tile_overlap": legacy_result.get("tile_overlap"),
                    "merge_iou": legacy_result.get("merge_iou"),
                    "tile_count": legacy_result.get("tile_count"),
                }
                resolved_weights = legacy_model_path
                resolved_model_name = resolved_model_name or Path(legacy_model_path).stem
                warnings["legacy_runtime_used"] = {
                    "reason": "legacy_yolov5_checkpoint",
                    "model_path": legacy_model_path,
                    "inference_backend": legacy_inference_backend,
                }
                predictions_json = str(legacy_result.get("predictions_json") or "").strip()
                if not predictions_json:
                    predictions_json = str(Path(out_dir) / "predictions.json")
                for pred in legacy_result.get("predictions") or []:
                    if not isinstance(pred, dict):
                        continue
                    boxes = pred.get("boxes") if isinstance(pred.get("boxes"), list) else []
                    per_image = {
                        "path": str(pred.get("input_path") or "").strip() or None,
                        "boxes": [],
                        "class_counts": {},
                    }
                    per_image_counts = (
                        dict(pred.get("class_counts") or {})
                        if isinstance(pred.get("class_counts"), dict)
                        else {}
                    )
                    per_image["class_counts"] = {
                        str(key): int(value)
                        for key, value in per_image_counts.items()
                        if str(key).strip()
                    }
                    for box in boxes:
                        if not isinstance(box, dict):
                            continue
                        cls_name = str(box.get("class_name") or "").strip()
                        confidence = float(box.get("confidence") or 0.0)
                        xyxy = box.get("xyxy") if isinstance(box.get("xyxy"), list) else []
                        counts_by_class[cls_name] = counts_by_class.get(cls_name, 0) + 1
                        conf_values.append(confidence)
                        per_image["boxes"].append(
                            {
                                "class_id": int(box.get("class_id") or 0),
                                "class_name": cls_name,
                                "confidence": confidence,
                                "xyxy": [float(value) for value in xyxy[:4]],
                            }
                        )
                    predictions.append(per_image)
            except Exception as exc:  # noqa: BLE001
                load_errors.append(f"legacy_runtime: {exc}")

        if not predictions:
            detail = "; ".join(load_errors[-3:]) if load_errors else "unknown error"
            return {
                "success": False,
                "error": (
                    "Failed to load YOLO weights. "
                    "If this model came from BisQue, verify that the downloaded file is a real .pt checkpoint, "
                    "not an HTML/XML response."
                ),
                "attempted_model_paths": candidate_paths,
                "details": detail,
            }
    else:
        results = model.predict(
            source=paths,
            conf=float(conf),
            iou=float(iou),
            save=bool(save_visualizations),
            project=str(Path(out_dir).resolve()),
            name="predict",
            verbose=False,
        )

        for r in results:
            per_image = {
                "path": getattr(r, "path", None),
                "boxes": [],
                "class_counts": {},
            }
            names = getattr(r, "names", {}) or {}
            boxes = getattr(r, "boxes", None)
            if boxes is not None:
                cls_list = boxes.cls.tolist() if hasattr(boxes, "cls") else []
                conf_list = boxes.conf.tolist() if hasattr(boxes, "conf") else []
                xyxy_list = boxes.xyxy.tolist() if hasattr(boxes, "xyxy") else []
                for cls_id, score, xyxy in zip(cls_list, conf_list, xyxy_list):
                    cls_name = str(names.get(int(cls_id), str(int(cls_id))))
                    counts_by_class[cls_name] = counts_by_class.get(cls_name, 0) + 1
                    per_image["class_counts"][cls_name] = (
                        per_image["class_counts"].get(cls_name, 0) + 1
                    )
                    conf_values.append(float(score))
                    per_image["boxes"].append(
                        {
                            "class_id": int(cls_id),
                            "class_name": cls_name,
                            "confidence": float(score),
                            "xyxy": [float(x) for x in xyxy],
                        }
                    )
            predictions.append(per_image)

        predictions_json = str(Path(out_dir) / "predictions.json")
        Path(predictions_json).write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "model_path": resolved_weights,
                    "model_name": resolved_model_name,
                    "conf": conf,
                    "iou": iou,
                    "predictions": predictions,
                    "counts_by_class": counts_by_class,
                },
                indent=2,
            )
        )

    prediction_images_raw: list[str] = []
    prediction_images: list[str] = []
    prediction_image_records: list[dict[str, Any]] = []
    render_failures: list[dict[str, Any]] = []
    annotated_dir = out_dir_path / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    for index, prediction in enumerate(predictions):
        if not isinstance(prediction, dict):
            continue
        source_path = str(prediction.get("path") or prediction.get("input_path") or "").strip()
        if not source_path and index < len(paths):
            source_path = str(paths[index]).strip()
        if not source_path:
            continue
        raw_source_path = source_path
        raw_source_name = Path(raw_source_path).name or raw_source_path
        source_name = raw_source_name
        class_counts = prediction.get("class_counts")
        normalized_class_counts = (
            {
                str(key): int(raw_value)
                for key, raw_value in class_counts.items()
                if str(key).strip()
            }
            if isinstance(class_counts, dict)
            else {}
        )
        raw_boxes = prediction.get("boxes")
        boxes = raw_boxes if isinstance(raw_boxes, list) else []
        preview_name = (
            f"{index:03d}-{_safe_slug(Path(raw_source_name).stem)}__matplotlib_annotated.png"
        )
        preview_path = annotated_dir / preview_name
        render_result = _render_yolo_detection_figure(
            source_path=raw_source_path,
            boxes=boxes,
            output_path=preview_path,
        )
        if bool(render_result.get("success")):
            record_preview_path = str(render_result.get("preview_path") or preview_path)
            preview_kind = "matplotlib_annotated"
            prediction_images.append(record_preview_path)
            image_width = render_result.get("image_width")
            image_height = render_result.get("image_height")
        else:
            record_preview_path = raw_source_path
            preview_kind = "original_fallback"
            prediction_images.append(raw_source_path)
            image_width = None
            image_height = None
            render_failures.append(
                {
                    "source_path": raw_source_path,
                    "error": str(render_result.get("error") or "render_failed"),
                }
            )
        prediction_images_raw.append(raw_source_path)
        prediction_image_records.append(
            {
                "index": index,
                "source_path": raw_source_path,
                "source_name": source_name,
                "preview_path": record_preview_path,
                "preview_name": Path(record_preview_path).name or record_preview_path,
                "preview_kind": preview_kind,
                "raw_source_path": raw_source_path,
                "raw_source_name": raw_source_name,
                "image_width": int(image_width) if isinstance(image_width, (int, float)) else None,
                "image_height": int(image_height)
                if isinstance(image_height, (int, float))
                else None,
                "box_count": len(boxes),
                "class_counts": normalized_class_counts,
            }
        )

    if render_failures:
        warnings["visualization_render_failed"] = {
            "count": len(render_failures),
            "images": render_failures[:12],
        }

    avg_conf = (sum(conf_values) / len(conf_values)) if conf_values else 0.0
    total_boxes = sum(counts_by_class.values())
    finetune_recommended = total_boxes == 0 or avg_conf < 0.35
    inference_backend = legacy_inference_backend if legacy_runtime_used else "ultralytics"
    metadata_context_by_path = {
        str(path): _image_metadata_context(
            str(path),
            output_root=str((out_dir_path / "metadata").resolve()),
        )
        for path in paths
    }
    spatial_analysis, analysis_summary = _detection_spatial_analysis(
        predictions=predictions,
        metadata_context_by_path=metadata_context_by_path,
    )
    spatial_images = (
        spatial_analysis.get("images", []) if isinstance(spatial_analysis, dict) else []
    )

    counts_rows = [
        {"class": k, "count": int(v)}
        for k, v in sorted(counts_by_class.items(), key=lambda x: (-x[1], x[0]))
    ]

    ui_artifacts: list[dict[str, Any]] = []
    if counts_rows:
        ui_artifacts.append(
            {
                "type": "chart",
                "kind": "bar",
                "title": "YOLO counts by class",
                "data": counts_rows,
                "x": "class",
                "y": "count",
            }
        )
        ui_artifacts.append(
            {
                "type": "write",
                "title": "YOLO counts table",
                "payload": counts_rows,
            }
        )
    ui_artifacts.append(
        {
            "type": "metrics",
            "title": "YOLO summary",
            "payload": {
                "total_boxes": total_boxes,
                "avg_confidence": round(avg_conf, 4),
                "finetune_recommended": finetune_recommended,
            },
        }
    )
    if analysis_summary:
        ui_artifacts.append(
            {
                "type": "write",
                "title": "Detection context",
                "payload": {
                    "summary": analysis_summary,
                    "spatial_analysis": spatial_analysis,
                    "metadata_context_by_image": metadata_context_by_path,
                },
            }
        )
    for record in prediction_image_records[:24]:
        preview_path = str(record.get("preview_path") or "").strip()
        if not preview_path:
            continue
        preview_kind = str(record.get("preview_kind") or "").strip().lower()
        preview_title = str(record.get("preview_name") or record.get("source_name") or "").strip()
        ui_artifacts.append(
            {
                "type": "image",
                "kind": "image",
                "path": preview_path,
                "title": preview_title or "YOLO annotated detections",
                "caption": (
                    "Black bounding boxes with class-colored labels over the detected objects."
                    if preview_kind == "matplotlib_annotated"
                    else "Original image fallback because the annotated detection render failed."
                ),
                "payload": {
                    "preview_kind": preview_kind or None,
                    "source_path": str(
                        record.get("raw_source_path") or record.get("source_path") or ""
                    ).strip()
                    or None,
                    "box_count": int(record.get("box_count") or 0),
                    "class_counts": (
                        record.get("class_counts")
                        if isinstance(record.get("class_counts"), dict)
                        else {}
                    ),
                },
            }
        )

    inference_configuration = {
        "backend": inference_backend,
        "tile_size": legacy_inference_configuration.get("tile_size")
        if legacy_runtime_used
        else None,
        "tile_overlap": legacy_inference_configuration.get("tile_overlap")
        if legacy_runtime_used
        else None,
        "conf": round(float(conf), 4),
        "iou": round(float(iou), 4),
        "merge_iou": (
            legacy_inference_configuration.get("merge_iou") if legacy_runtime_used else None
        ),
        "tile_count": legacy_inference_configuration.get("tile_count")
        if legacy_runtime_used
        else None,
    }
    if legacy_runtime_used:
        inference_configuration["tile_strategy"] = "sliding_window_overlap"

    stability_audit: dict[str, Any] | None = None
    stability_warnings: dict[str, Any] = {}
    supports_stability_audit = bool(
        prairie_model_requested
        or _is_prairie_yolo_alias(resolved_model_name)
        or _text_mentions_prairie_detection(resolved_model_name)
        or _text_mentions_prairie_detection(resolved_weights)
    )
    if include_stability_audit:
        if supports_stability_audit:
            stability_audit = analyze_prediction_stability_tool(
                file_paths=paths,
                model_name=resolved_model_name or _PRAIRIE_YOLO_MODEL_KEY,
                model_path=str(resolved_weights) if str(resolved_weights or "").strip() else None,
                top_k=max(1, int(stability_top_k)),
                preservation_ratio=float(stability_preservation_ratio),
                conf=float(conf),
                iou=float(iou),
            )
            if not bool(stability_audit.get("success")):
                stability_warnings["stability_audit_failed"] = {
                    "reason": str(stability_audit.get("error") or "unknown_error"),
                }
                stability_audit = None
        else:
            stability_warnings["stability_audit_skipped"] = {
                "reason": (
                    "The current prediction-stability backend is validated for the prairie/RareSpot detector path. "
                    "Use analyze_prediction_stability explicitly if you want to test another detector."
                ),
                "model_name": resolved_model_name or Path(str(resolved_weights)).stem,
            }

    overall_spatial_context = (
        spatial_analysis.get("overall_prairie_burrow_context")
        if isinstance(spatial_analysis, dict)
        and isinstance(spatial_analysis.get("overall_prairie_burrow_context"), dict)
        else {}
    )
    metadata_summary = (
        spatial_analysis.get("metadata_summary")
        if isinstance(spatial_analysis, dict)
        and isinstance(spatial_analysis.get("metadata_summary"), dict)
        else {}
    )
    ecology_context = _build_prairie_ecology_context(
        counts_by_class=counts_by_class,
        spatial_analysis=spatial_analysis if isinstance(spatial_analysis, dict) else {},
        inference_configuration=inference_configuration,
        metadata_summary=metadata_summary,
        prairie_model_requested=prairie_model_requested,
        finetune_recommended=finetune_recommended,
    )

    scientific_summary = {
        "inference": inference_configuration,
        "overall": {
            "image_count": len(predictions),
            "total_boxes": total_boxes,
            "prairie_dog_count": counts_by_class.get("prairie_dog", 0),
            "burrow_count": counts_by_class.get("burrow", 0),
            "nearest_burrow_distance_px_mean": overall_spatial_context.get(
                "nearest_burrow_distance_px_mean"
            ),
        },
        "per_image": spatial_images,
        "metadata": metadata_summary,
        "image_records": prediction_image_records,
    }
    if stability_audit:
        scientific_summary["prediction_stability"] = {
            "summary": stability_audit.get("summary"),
            "review_candidates": stability_audit.get("review_candidates"),
            "backend_method": stability_audit.get("backend_method"),
            "active_learning_note": stability_audit.get("active_learning_note"),
        }
    if ecology_context:
        scientific_summary["ecology_context"] = ecology_context

    result_message = (
        _build_prairie_ecology_result_message(
            counts_by_class=counts_by_class,
            ecology_context=ecology_context,
            spatial_analysis=(spatial_analysis if isinstance(spatial_analysis, dict) else {}),
            finetune_recommended=finetune_recommended,
        )
        if ecology_context
        else (
            f"{analysis_summary} "
            + (
                "Finetuning is recommended for better scientific accuracy. "
                if finetune_recommended
                else "If you'd like higher accuracy, you can finetune on labeled examples. "
            )
            + "For finetuning, upload training images and YOLO-format label .txt files (same base filename)."
        )
    )
    if stability_audit and isinstance(stability_audit.get("review_candidates"), list):
        review_candidates = list(stability_audit.get("review_candidates") or [])
        top_review_bits = ", ".join(
            f"{item.get('file_name')} ({float(item.get('score') or 0.0):.3f})"
            for item in review_candidates[:3]
            if str(item.get("file_name") or "").strip()
        )
        if top_review_bits:
            result_message = (
                f"{result_message} Stability audit: the best manual-review candidates are {top_review_bits}, "
                "because their detections changed most under controlled perturbation."
            ).strip()
        else:
            result_message = (
                f"{result_message} Stability audit: this image remained comparatively stable under the perturbation "
                "used here, so it is not a high-priority review candidate by this particular signal."
            ).strip()

    result = {
        "success": True,
        "model_path": resolved_weights,
        "model_name": resolved_model_name or Path(str(resolved_weights)).stem,
        "output_directory": out_dir,
        "predictions_json": predictions_json,
        "prediction_images": prediction_images,
        "prediction_images_raw": prediction_images_raw,
        "prediction_image_records": prediction_image_records,
        "counts_by_class": counts_by_class,
        "ui_artifacts": ui_artifacts,
        "inference_configuration": inference_configuration,
        "metrics": {
            "total_boxes": total_boxes,
            "avg_confidence": round(avg_conf, 4),
            "finetune_recommended": finetune_recommended,
            "prairie_dog_count": counts_by_class.get("prairie_dog", 0),
            "burrow_count": counts_by_class.get("burrow", 0),
            "nearest_burrow_distance_px_mean": overall_spatial_context.get(
                "nearest_burrow_distance_px_mean"
            ),
        },
        "analysis_summary": analysis_summary,
        "spatial_analysis": spatial_analysis,
        "metadata_context_by_image": metadata_context_by_path,
        "scientific_summary": scientific_summary,
        "prediction_stability_audit": stability_audit,
        "predictions": predictions[:10],
        "sequence_expansions": sequence_expansions,
        "sequence_warnings": sequence_warnings,
        "message": result_message,
    }
    if ecology_context:
        result["ecology_context"] = ecology_context
    if legacy_runtime_used:
        result["inference_backend"] = legacy_inference_backend
        if legacy_logs:
            result["runtime_logs"] = legacy_logs
    if prairie_tiled_runtime_forced:
        result["warnings"] = {
            **result.get("warnings", {}),
            "legacy_runtime_forced": warnings.get("legacy_runtime_forced"),
        }
    if non_images:
        warnings["ignored_non_images"] = non_images[:50]
    if stability_warnings:
        warnings.update(stability_warnings)
    if warnings:
        result["warnings"] = warnings
    if stability_audit and isinstance(stability_audit.get("ui_artifacts"), list):
        result["ui_artifacts"] = (
            list(result.get("ui_artifacts") or [])
            + list(stability_audit.get("ui_artifacts") or [])[:12]
        )
    return result


def _prediction_stability_review_candidates(
    results: list[dict[str, Any]] | None,
    *,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for item in list(results or []):
        if not isinstance(item, dict):
            continue
        score = float(item.get("score") or 0.0)
        if score <= 0:
            continue
        file_name = str(item.get("file_name") or "").strip()
        if not file_name:
            continue
        lost = int(item.get("lost_detection_count") or 0)
        new = int(item.get("new_detection_count") or 0)
        class_jitter = float(item.get("class_jitter") or 0.0)
        spatial_jitter = float(item.get("spatial_jitter") or 0.0)
        confidence_jitter = float(item.get("confidence_jitter") or 0.0)
        if lost > 0:
            reason = f"{lost} detection(s) disappeared under perturbation"
            dominant = "lost_detections"
        elif class_jitter > 0:
            reason = f"class identity changed for {class_jitter:.1f} matched detection(s)"
            dominant = "class_jitter"
        elif spatial_jitter > 0:
            reason = (
                f"box locations shifted under perturbation (spatial jitter {spatial_jitter:.2f})"
            )
            dominant = "spatial_jitter"
        elif confidence_jitter > 0:
            reason = f"detection confidence shifted under perturbation (aggregate shift {confidence_jitter:.2f})"
            dominant = "confidence_jitter"
        elif new > 0:
            reason = f"{new} new detection(s) appeared under perturbation"
            dominant = "new_detections"
        else:
            reason = "predictions changed under perturbation"
            dominant = "mixed"
        candidates.append(
            {
                "file_name": file_name,
                "file_path": str(item.get("file_path") or "").strip() or None,
                "score": score,
                "normalized_score": float(item.get("normalized_score") or 0.0),
                "reason": reason,
                "dominant_instability": dominant,
                "original_detection_count": int(item.get("original_detection_count") or 0),
                "filtered_detection_count": int(item.get("filtered_detection_count") or 0),
                "lost_detection_count": lost,
                "new_detection_count": new,
                "class_jitter": class_jitter,
                "spatial_jitter": spatial_jitter,
                "confidence_jitter": confidence_jitter,
                "mean_confidence_shift": float(item.get("mean_confidence_shift") or 0.0),
                "mean_confidence_drop": float(item.get("mean_confidence_drop") or 0.0),
                "matched_iou_mean": float(item.get("matched_iou_mean") or 0.0),
                "class_consistency_rate": float(item.get("class_consistency_rate") or 0.0),
                "retained_energy_fraction": float(item.get("retained_energy_fraction") or 0.0),
            }
        )
    candidates.sort(key=lambda row: (-float(row["score"]), str(row["file_name"])))
    return candidates[: max(1, int(top_k))]


def analyze_prediction_stability_tool(
    file_paths: list[str],
    model_name: str | None = None,
    model_path: str | None = None,
    method: str = "auto",
    top_k: int = 5,
    preservation_ratio: float = SPECTRAL_DEFAULT_PRESERVATION_RATIO,
    conf: float = SPECTRAL_DEFAULT_CONF_THRES,
    iou: float = SPECTRAL_DEFAULT_IOU_THRES,
    imgsz: int = 640,
    batch_size: int = 4,
) -> dict[str, Any]:
    """Analyze prediction stability and surface review candidates."""
    normalized_method = str(method or "auto").strip().lower() or "auto"
    if normalized_method not in {"auto", "spectral"}:
        return {
            "success": False,
            "error": f"Unsupported prediction stability method '{method}'.",
            "supported_methods": ["auto", "spectral"],
        }

    spectral_result = score_spectral_instability_tool(
        file_paths=file_paths,
        model_name=model_name,
        model_path=model_path,
        preservation_ratio=preservation_ratio,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        batch_size=batch_size,
    )
    if not bool(spectral_result.get("success")):
        return spectral_result

    review_candidates = _prediction_stability_review_candidates(
        spectral_result.get("results"),
        top_k=int(top_k),
    )
    summary = dict(spectral_result.get("summary") or {})
    summary["review_candidate_count"] = len(review_candidates)
    top_bits = ", ".join(
        f"{item.get('file_name')}: {float(item.get('score') or 0.0):.3f}"
        for item in review_candidates[:3]
        if str(item.get("file_name") or "").strip()
    )
    message = (
        "Prediction stability analysis completed. "
        "Higher scores indicate detections that changed more under controlled perturbation."
    )
    if top_bits:
        message += f" Best review candidates: {top_bits}."

    ui_artifacts = list(spectral_result.get("ui_artifacts") or [])
    ui_artifacts.insert(
        0,
        {
            "type": "metrics",
            "title": "Prediction stability audit",
            "payload": {
                "review_candidate_count": len(review_candidates),
                "image_count": int(summary.get("image_count") or 0),
                "nonzero_score_count": int(summary.get("nonzero_score_count") or 0),
                "max_score": float(summary.get("max_score") or 0.0),
                "mean_score": float(summary.get("mean_score") or 0.0),
            },
        },
    )
    if review_candidates:
        ui_artifacts.append(
            {
                "type": "table",
                "title": "Best manual-review candidates",
                "payload": review_candidates,
            }
        )

    return {
        "success": True,
        "message": message,
        "analysis_kind": "prediction_stability",
        "method": "prediction_stability",
        "backend_method": spectral_result.get("method"),
        "method_version": spectral_result.get("method_version"),
        "model_name": spectral_result.get("model_name"),
        "model_path": spectral_result.get("model_path"),
        "output_directory": spectral_result.get("output_directory"),
        "scores_json": spectral_result.get("scores_json"),
        "output_files": spectral_result.get("output_files"),
        "artifacts": spectral_result.get("artifacts"),
        "visualization_paths": spectral_result.get("visualization_paths"),
        "summary": summary,
        "results": spectral_result.get("results"),
        "review_candidates": review_candidates,
        "ui_artifacts": ui_artifacts,
        "warnings": spectral_result.get("warnings") or {},
        "active_learning_note": (
            "This audit ranks images by prediction fragility under perturbation. "
            "Use it to prioritize review, not as a direct estimate of model error."
        ),
    }


def score_spectral_instability_tool(
    file_paths: list[str],
    model_name: str | None = None,
    model_path: str | None = None,
    preservation_ratio: float = SPECTRAL_DEFAULT_PRESERVATION_RATIO,
    conf: float = SPECTRAL_DEFAULT_CONF_THRES,
    iou: float = SPECTRAL_DEFAULT_IOU_THRES,
    imgsz: int = 640,
    batch_size: int = 4,
) -> dict[str, Any]:
    """Score RareSpot-style spectral instability for one or more images."""
    if not file_paths:
        return {"success": False, "error": "file_paths is required"}

    try:
        expanded = _expand_file_inputs([str(p) for p in file_paths])
    except Exception as exc:
        return {"success": False, "error": str(exc)}

    expanded, sequence_expansions, sequence_warnings = _expand_sequence_inputs_for_2d_models(
        expanded
    )
    missing = [str(path) for path in expanded if not path.exists()]
    if missing:
        return {"success": False, "error": "Some files do not exist", "missing": missing}

    images = [path for path in expanded if _is_yolo_image(path)]
    non_images = [str(path) for path in expanded if path.exists() and not _is_yolo_image(path)]
    if not images:
        return {"success": False, "error": "No images found in file_paths."}

    prairie_model_requested = (
        _is_prairie_yolo_alias(model_name) or not str(model_name or "").strip()
    )
    explicit_model_path = str(model_path).strip() if model_path else ""
    resolved_model_name = str(model_name or "").strip() or _PRAIRIE_YOLO_MODEL_KEY
    resolved_weights_path = explicit_model_path
    warnings: dict[str, Any] = {}

    if prairie_model_requested and not explicit_model_path:
        prairie_model_name, prairie_model_path, prairie_resolution = (
            _resolve_prairie_yolo_model_target()
        )
        if prairie_model_path:
            resolved_weights_path = prairie_model_path
            resolved_model_name = prairie_model_name or _PRAIRIE_YOLO_MODEL_KEY
            warnings["prairie_model_resolution"] = {
                "reason": prairie_resolution,
                "model_name": resolved_model_name,
                "model_path": prairie_model_path,
            }
    if not resolved_weights_path:
        resolved_weights_path = str(_resolve_prairie_builtin_weights_path())

    run_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
    output_root = (Path(_science_output_root("spectral_instability")) / run_id).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    try:
        payload = score_spectral_instability(
            image_paths=[str(path.resolve()) for path in images],
            weights_path=resolved_weights_path,
            output_dir=output_root,
            config=SpectralInstabilityConfig(
                imgsz=int(imgsz),
                batch_size=int(batch_size),
                conf_thres=float(conf),
                iou_thres=float(iou),
                preservation_ratio=float(preservation_ratio),
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "success": False,
            "error": str(exc),
            "model_path": resolved_weights_path,
        }

    if not bool(payload.get("success")):
        return payload

    summary = dict(payload.get("summary") or {})
    results = list(payload.get("results") or [])
    visualization_paths = (
        list(payload.get("visualization_paths") or [])
        if isinstance(payload.get("visualization_paths"), list)
        else []
    )
    top_ranked = list(summary.get("top_ranked") or [])
    if top_ranked:
        top_bits = ", ".join(
            f"{item.get('file_name')}: {float(item.get('score') or 0.0):.3f}"
            for item in top_ranked[:3]
            if str(item.get("file_name") or "").strip()
        )
        message = (
            "Spectral instability scoring completed. Higher scores indicate images whose RareSpot detections "
            f"change more under spectral feature filtering. Top-ranked images: {top_bits}."
        )
    else:
        message = "Spectral instability scoring completed, but no images produced nonzero instability under the current settings."

    if sequence_expansions:
        warnings["sequence_expansions"] = sequence_expansions
    if sequence_warnings:
        warnings["sequence_warnings"] = sequence_warnings
    if non_images:
        warnings["ignored_non_images"] = non_images[:50]

    ui_artifacts = [
        {
            "type": "chart",
            "kind": "bar",
            "title": "Spectral instability ranking",
            "data": [
                {
                    "image": str(item.get("file_name") or ""),
                    "score": float(item.get("score") or 0.0),
                }
                for item in results[:12]
                if str(item.get("file_name") or "").strip()
            ],
            "x": "image",
            "y": "score",
        },
        {
            "type": "write",
            "title": "Spectral instability details",
            "payload": results[:24],
        },
        {
            "type": "metrics",
            "title": "Spectral instability summary",
            "payload": {
                "image_count": int(summary.get("image_count") or 0),
                "nonzero_score_count": int(summary.get("nonzero_score_count") or 0),
                "max_score": float(summary.get("max_score") or 0.0),
                "mean_score": float(summary.get("mean_score") or 0.0),
                "median_score": float(summary.get("median_score") or 0.0),
                "max_normalized_score": float(summary.get("max_normalized_score") or 0.0),
                "mean_normalized_score": float(summary.get("mean_normalized_score") or 0.0),
                "median_normalized_score": float(summary.get("median_normalized_score") or 0.0),
            },
        },
    ]
    for item in visualization_paths[:8]:
        if not isinstance(item, dict) or not str(item.get("path") or "").strip():
            continue
        ui_artifacts.append(
            {
                "type": "image",
                "title": str(item.get("title") or "Spectral instability visualization"),
                "path": str(item.get("path")),
                "caption": str(item.get("caption") or "").strip() or None,
            }
        )

    return {
        "success": True,
        "message": message,
        "method": payload.get("method"),
        "method_version": payload.get("method_version"),
        "model_name": resolved_model_name,
        "model_path": str(Path(resolved_weights_path).expanduser().resolve()),
        "output_directory": str(output_root),
        "scores_json": payload.get("output_json"),
        "output_files": [payload.get("output_json")] if payload.get("output_json") else [],
        "artifacts": (
            [{"path": payload.get("output_json"), "title": "Spectral instability scores"}]
            if payload.get("output_json")
            else []
        ),
        "visualization_paths": visualization_paths,
        "summary": summary,
        "results": results,
        "ui_artifacts": ui_artifacts,
        "warnings": warnings,
        "active_learning_note": (
            "This is a standalone acquisition signal: it ranks images by detector instability under spectral feature perturbation, "
            "but it does not by itself update the labeled set or retrain the model."
        ),
    }


def yolo_finetune_detect(
    file_paths: list[str] | None = None,
    image_paths: list[str] | None = None,
    label_paths: list[str] | None = None,
    class_names: list[str] | None = None,
    base_model: str | None = None,
    epochs: int = 10,
    imgsz: int = 640,
    batch: int = 4,
    val_split: float = 0.2,
    seed: int = 42,
    device: str | None = None,
    prepare_only: bool = False,
) -> dict[str, Any]:
    """Finetune a pretrained YOLO26 *detection* model on a tiny custom dataset.

    Inputs are image files + YOLO-format label `.txt` files (same base filename).
    The tool will create a temporary COCO-style dataset folder under `data/yolo/datasets/<id>/`
    and then run Ultralytics training starting from pretrained weights.
    """
    from src.tooling.progress import emit_progress

    def _emit(message: str, **extra: Any) -> None:
        emit_progress(str(message), tool="yolo_finetune_detect", **extra)

    base_weights = (base_model or _default_yolo_pretrained_weights()).strip()
    if base_weights.lower().endswith((".yaml", ".yml")):
        return {
            "success": False,
            "error": "base_model must point to pretrained weights (.pt), not a model YAML.",
        }

    if epochs < 1 or epochs > 300:
        return {"success": False, "error": "epochs must be between 1 and 300"}
    if imgsz < 32 or imgsz > 2048:
        return {"success": False, "error": "imgsz must be between 32 and 2048"}
    if batch < 1 or batch > 256:
        return {"success": False, "error": "batch must be between 1 and 256"}
    if not (0.05 <= float(val_split) <= 0.5):
        return {"success": False, "error": "val_split must be between 0.05 and 0.5"}

    raw_inputs: list[str] = []
    if file_paths:
        raw_inputs.extend([str(p) for p in file_paths])
    if image_paths:
        raw_inputs.extend([str(p) for p in image_paths])
    if label_paths:
        raw_inputs.extend([str(p) for p in label_paths])
    if not raw_inputs:
        return {
            "success": False,
            "error": "Provide file_paths (images + labels) or image_paths/label_paths.",
        }

    _emit("Collecting training inputs…", event="collect_inputs")
    try:
        expanded = _expand_file_inputs(raw_inputs)
    except Exception as e:
        return {"success": False, "error": str(e)}

    missing = [str(p) for p in expanded if not p.exists()]
    if missing:
        return {"success": False, "error": "Some inputs do not exist", "missing": missing}

    images = [p for p in expanded if _is_yolo_image(p)]
    labels = [p for p in expanded if _is_yolo_label(p)]
    unknown = [str(p) for p in expanded if p not in images and p not in labels]
    warnings: dict[str, Any] = {}
    if unknown:
        warnings["ignored_unsupported_files"] = unknown[:50]

    _emit(f"Found {len(images)} image(s) and {len(labels)} label file(s).", event="inputs_count")

    if not images:
        return {"success": False, "error": "No images found in inputs."}
    if not labels:
        return {"success": False, "error": "No label .txt files found in inputs."}

    # Ensure image filenames are unique after normalizing session hash prefixes.
    normalized_image_names = [_canonical_uploaded_filename(p.name) for p in images]
    dup_images = [n for n, c in Counter(normalized_image_names).items() if c > 1]
    if dup_images:
        return {
            "success": False,
            "error": (
                "Duplicate image filenames detected after upload-name normalization. "
                "Rename images so basenames are unique."
            ),
            "duplicates": dup_images,
        }

    # Build label lookup by exact and canonical stems, so uploads named
    # '<hash>__filename' can still pair with their corresponding images.
    labels_by_key: dict[str, Path] = {}
    duplicate_label_keys: set[str] = set()
    for lbl in labels:
        for key in _yolo_candidate_stems(lbl):
            if not key:
                continue
            existing = labels_by_key.get(key)
            if existing and existing != lbl:
                duplicate_label_keys.add(key)
                continue
            labels_by_key[key] = lbl
    if duplicate_label_keys:
        return {
            "success": False,
            "error": (
                "Duplicate label basenames detected after upload-name normalization. "
                "Ensure one .txt per image stem."
            ),
            "duplicates": sorted(duplicate_label_keys),
        }

    def _match_label_for_image(image_path: Path) -> tuple[Path | None, str | None]:
        for key in _yolo_candidate_stems(image_path):
            label = labels_by_key.get(key)
            if label is not None:
                return label, key
        return None, None

    def _parse_label_file(path: Path) -> list[tuple[int, float, float, float, float]]:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"Failed to read label file: {path}: {e}") from e
        rows: list[tuple[int, float, float, float, float]] = []
        for line_no, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(
                    f"Invalid YOLO detection label in {path}:{line_no}. "
                    f"Expected 5 columns (class x_center y_center width height), got {len(parts)}."
                )
            try:
                cls_id = int(float(parts[0]))
            except Exception as e:  # noqa: BLE001
                raise ValueError(f"Invalid class id in {path}:{line_no}: {parts[0]!r}") from e
            if cls_id < 0:
                raise ValueError(f"Class id must be >= 0 in {path}:{line_no}, got {cls_id}.")
            try:
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
            except Exception as e:  # noqa: BLE001
                raise ValueError(
                    f"Invalid bbox coordinates in {path}:{line_no}: {' '.join(parts[1:])!r}"
                ) from e
            coords = (x_center, y_center, width, height)
            if any(v < -1e-6 or v > 1.0 + 1e-6 for v in coords):
                raise ValueError(
                    f"YOLO bbox coordinates must be normalized to [0,1] in {path}:{line_no}, got {coords}."
                )
            if width <= 0.0 or height <= 0.0:
                raise ValueError(
                    f"YOLO bbox width/height must be > 0 in {path}:{line_no}, got {(width, height)}."
                )
            rows.append((cls_id, x_center, y_center, width, height))
        return rows

    # Parse labels + infer classes from actual IDs present.
    label_rows_by_stem: dict[str, list[tuple[int, float, float, float, float]]] = {}
    labels_for_image_stem: dict[str, Path] = {}
    image_class_ids_raw: dict[int, set[int]] = {}
    matched_via_canonical = 0
    unique_class_ids: set[int] = set()
    total_boxes = 0
    for idx, img in enumerate(images):
        lbl, matched_key = _match_label_for_image(img)
        if not lbl:
            image_class_ids_raw[idx] = set()
            continue
        if matched_key and matched_key != img.stem:
            matched_via_canonical += 1
        labels_for_image_stem[img.stem] = lbl
        try:
            rows = _parse_label_file(lbl)
        except Exception as e:
            return {"success": False, "error": str(e)}
        label_rows_by_stem[img.stem] = rows
        cls_ids = {r[0] for r in rows}
        image_class_ids_raw[idx] = cls_ids
        unique_class_ids.update(cls_ids)
        total_boxes += len(rows)
    if total_boxes <= 0:
        return {
            "success": False,
            "error": (
                "No bounding boxes found in labels after pairing images with labels by filename stem. "
                "If you uploaded files through the app, ensure each image has a matching label with the "
                "same original basename before the upload hash prefix."
            ),
        }
    if matched_via_canonical > 0:
        warnings["paired_via_upload_normalization"] = {
            "count": int(matched_via_canonical),
            "note": "Matched image/label pairs using dehashed upload filenames.",
        }

    sorted_class_ids = sorted(unique_class_ids)
    class_id_map: dict[int, int] = {raw_id: idx for idx, raw_id in enumerate(sorted_class_ids)}
    reverse_class_id_map: dict[int, int] = {idx: raw_id for raw_id, idx in class_id_map.items()}

    inferred_num_classes = len(sorted_class_ids)
    max_class_id = sorted_class_ids[-1]
    if class_names:
        provided_class_names = [str(c).strip() for c in class_names if str(c).strip()]
        if not provided_class_names:
            return {"success": False, "error": "class_names was provided but empty."}
        raw_ids_are_contiguous = sorted_class_ids == list(range(inferred_num_classes))
        if raw_ids_are_contiguous and len(provided_class_names) == inferred_num_classes:
            class_names = provided_class_names
        elif max_class_id < len(provided_class_names):
            class_names = [provided_class_names[raw_id] for raw_id in sorted_class_ids]
        else:
            return {
                "success": False,
                "error": (
                    "class_names does not match detected classes. "
                    "Pass either one name per detected class when raw class ids are contiguous from 0, "
                    "or pass a list indexed by raw class id."
                ),
                "max_class_id": int(max_class_id),
                "detected_num_classes": int(inferred_num_classes),
                "class_names_len": len(provided_class_names),
            }
    else:
        class_names = [f"class{raw_id}" for raw_id in sorted_class_ids]

    preview = ", ".join(class_names[:10])
    if len(class_names) > 10:
        preview += ", ..."
    _emit(f"Inferred {len(class_names)} class(es): {preview}", event="classes_inferred")
    if sorted_class_ids != list(range(len(sorted_class_ids))):
        raw_preview = ", ".join(str(cid) for cid in sorted_class_ids[:10])
        if len(sorted_class_ids) > 10:
            raw_preview += ", ..."
        _emit(
            f"Remapping sparse raw class ids to contiguous ids: {raw_preview}",
            event="class_remap",
            raw_class_ids=sorted_class_ids,
        )

    # Create COCO-style dataset folder (images/train2017, labels/train2017, ...)
    dataset_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
    dataset_dir = Path(_yolo_datasets_root()) / dataset_id
    train_img_dir = dataset_dir / "images" / "train2017"
    val_img_dir = dataset_dir / "images" / "val2017"
    train_lbl_dir = dataset_dir / "labels" / "train2017"
    val_lbl_dir = dataset_dir / "labels" / "val2017"
    for d in (train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir):
        d.mkdir(parents=True, exist_ok=True)

    image_class_ids = {
        idx: {class_id_map[cid] for cid in image_class_ids_raw.get(idx, set())}
        for idx in range(len(images))
    }

    def _union_classes(sample_indices: list[int]) -> set[int]:
        classes: set[int] = set()
        for i in sample_indices:
            classes.update(image_class_ids.get(i, set()))
        return classes

    def _split_score(train_idx: list[int], val_idx: list[int]) -> tuple[int, int, int]:
        train_classes = _union_classes(train_idx)
        val_classes = _union_classes(val_idx)
        unseen_val = val_classes - train_classes
        overlap = val_classes & train_classes
        return (len(unseen_val), -len(overlap), -len(val_classes))

    rng = random.Random(int(seed))
    indices = list(range(len(images)))
    split_strategy = "single_image_duplicate"
    if len(images) == 1:
        train_indices = [0]
        val_indices = [0]
    else:
        n_val = max(1, int(round(len(images) * float(val_split))))
        n_val = min(len(images) - 1, n_val)
        best_train: list[int] | None = None
        best_val: list[int] | None = None
        best_score: tuple[int, int, int] | None = None
        attempts = max(32, min(512, len(images) * 32))
        for _ in range(attempts):
            shuffled = indices[:]
            rng.shuffle(shuffled)
            cand_val = shuffled[:n_val]
            cand_train = shuffled[n_val:]
            if not cand_train:
                continue
            score = _split_score(cand_train, cand_val)
            if best_score is None or score < best_score:
                best_train = cand_train
                best_val = cand_val
                best_score = score
                if score[0] == 0 and score[1] <= -1:
                    break
        if best_train is None or best_val is None:
            shuffled = indices[:]
            rng.shuffle(shuffled)
            best_val = shuffled[:n_val]
            best_train = shuffled[n_val:]
        train_indices = best_train
        val_indices = best_val
        split_strategy = "overlap_optimized"

    def _split_diagnostics(train_idx: list[int], val_idx: list[int]) -> dict[str, Any]:
        train_classes = sorted(_union_classes(train_idx))
        val_classes = sorted(_union_classes(val_idx))
        unseen_val = sorted(set(val_classes) - set(train_classes))
        return {
            "train_class_ids": train_classes,
            "val_class_ids": val_classes,
            "unseen_val_class_ids": unseen_val,
            "train_raw_class_ids": [int(reverse_class_id_map[c]) for c in train_classes],
            "val_raw_class_ids": [int(reverse_class_id_map[c]) for c in val_classes],
            "unseen_val_raw_class_ids": [int(reverse_class_id_map[c]) for c in unseen_val],
        }

    split_diagnostics = _split_diagnostics(train_indices, val_indices)
    if split_diagnostics["unseen_val_class_ids"]:
        if len(images) <= 8 and len(train_indices) >= 1:
            fallback_candidates = sorted(
                train_indices,
                key=lambda idx: (-len(image_class_ids.get(idx, set())), idx),
            )
            fallback_val = fallback_candidates[: max(1, len(val_indices))]
            if fallback_val:
                val_indices = fallback_val
                split_strategy = "train_overlap_fallback"
                split_diagnostics = _split_diagnostics(train_indices, val_indices)
                warnings["val_split_adjusted"] = {
                    "reason": "val classes were not present in train; reused train samples for validation.",
                    "strategy": split_strategy,
                    "seed": int(seed),
                }
                _emit(
                    "Adjusted validation split to avoid unseen validation classes on tiny dataset.",
                    event="split_adjusted",
                    strategy=split_strategy,
                )

    if split_diagnostics["unseen_val_class_ids"]:
        warnings["val_classes_unseen_in_train"] = {
            "class_ids": split_diagnostics["unseen_val_class_ids"],
            "raw_class_ids": split_diagnostics["unseen_val_raw_class_ids"],
        }
        _emit(
            "Validation contains classes not present in train; mAP can remain near zero until split is fixed.",
            event="split_warning",
            unseen_val_class_ids=split_diagnostics["unseen_val_class_ids"],
            unseen_val_raw_class_ids=split_diagnostics["unseen_val_raw_class_ids"],
        )

    def _write_mapped_label(stem: str, dst_lbl: Path) -> None:
        rows = label_rows_by_stem.get(stem) or []
        if not rows:
            dst_lbl.write_text("", encoding="utf-8")
            return
        lines = [
            f"{class_id_map[row[0]]} {row[1]:.6f} {row[2]:.6f} {row[3]:.6f} {row[4]:.6f}"
            for row in rows
        ]
        dst_lbl.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _copy_pair(img: Path, img_dst_dir: Path, lbl_dst_dir: Path) -> dict[str, Any]:
        dst_img = img_dst_dir / img.name
        shutil.copy2(img, dst_img)
        src_lbl = labels_for_image_stem.get(img.stem)
        dst_lbl = lbl_dst_dir / f"{img.stem}.txt"
        if src_lbl and src_lbl.exists():
            _write_mapped_label(img.stem, dst_lbl)
        else:
            dst_lbl.write_text("", encoding="utf-8")
        return {"image": str(dst_img), "label": str(dst_lbl), "has_label": bool(src_lbl)}

    manifest = {"train": [], "val": []}
    for idx in train_indices:
        manifest["train"].append(_copy_pair(images[idx], train_img_dir, train_lbl_dir))
    for idx in val_indices:
        manifest["val"].append(_copy_pair(images[idx], val_img_dir, val_lbl_dir))

    dataset_yaml_path = dataset_dir / "data.yaml"
    abs_dataset_dir = dataset_dir.resolve()
    names_lines = "\n".join(f"  {i}: {json.dumps(name)}" for i, name in enumerate(class_names))
    dataset_yaml = (
        f"path: {abs_dataset_dir}\n"
        f"train: images/train2017\n"
        f"val: images/val2017\n"
        f"nc: {len(class_names)}\n"
        f"names:\n{names_lines}\n"
    )
    dataset_yaml_path.write_text(dataset_yaml, encoding="utf-8")

    meta_path = dataset_dir / "meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "created_at_utc": datetime.utcnow().isoformat() + "Z",
                "base_model": base_weights,
                "num_images": len(images),
                "num_boxes": int(total_boxes),
                "train_images": len(manifest["train"]),
                "val_images": len(manifest["val"]),
                "class_names": class_names,
                "raw_class_ids": sorted_class_ids,
                "class_id_remap": {str(raw): int(mapped) for raw, mapped in class_id_map.items()},
                "val_split": float(val_split),
                "seed": int(seed),
                "split_strategy": split_strategy,
                "split_diagnostics": split_diagnostics,
                "inputs": {
                    "images": [str(p) for p in images[:200]],
                    "labels": [str(p) for p in labels[:200]],
                },
                "manifest": manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    _emit(
        (
            f"Prepared dataset {dataset_id}: train={len(manifest['train'])} image(s), "
            f"val={len(manifest['val'])} image(s), unseen_val_classes={len(split_diagnostics['unseen_val_class_ids'])}."
        ),
        event="dataset_prepared",
        dataset_id=dataset_id,
        dataset_dir=str(dataset_dir),
        dataset_yaml_path=str(dataset_yaml_path),
        split_strategy=split_strategy,
        split_diagnostics=split_diagnostics,
    )

    if prepare_only:
        _emit("prepare_only=true; skipping training.", event="prepare_only")
        result = {
            "success": True,
            "prepared_only": True,
            "dataset_id": dataset_id,
            "dataset_dir": str(dataset_dir),
            "dataset_yaml_path": str(dataset_yaml_path),
            "class_names": class_names,
            "raw_class_ids": sorted_class_ids,
            "class_id_remap": {str(raw): int(mapped) for raw, mapped in class_id_map.items()},
            "split_strategy": split_strategy,
            "split_diagnostics": split_diagnostics,
            "message": "Dataset prepared. Re-run with prepare_only=false to start finetuning.",
        }
        if warnings:
            result["warnings"] = warnings
        return result

    # Train starting from pretrained weights.
    try:
        yolo_class = _require_ultralytics()
    except ImportError as e:
        return {"success": False, "error": str(e)}

    _emit(f"Loading pretrained weights: {base_weights}", event="load_model")
    try:
        model = yolo_class(base_weights)
    except Exception as e:  # noqa: BLE001
        return {
            "success": False,
            "error": f"Failed to load pretrained weights: {base_weights}. Provide a local .pt path or ensure Ultralytics can download it. ({e})",
        }

    train_project = Path(_yolo_training_root())
    train_name = dataset_id
    train_args: dict[str, Any] = {
        "data": str(dataset_yaml_path),
        "epochs": int(epochs),
        "imgsz": int(imgsz),
        "batch": int(batch),
        "project": str(train_project),
        "name": train_name,
        "exist_ok": True,
        "verbose": False,
        "plots": False,
        "seed": int(seed),
    }
    if device:
        train_args["device"] = str(device)

    def _first_metric(metrics: dict[str, Any], keys: list[str]) -> float | None:
        for key in keys:
            val = metrics.get(key)
            if isinstance(val, (int, float)):
                return float(val)
        return None

    def _safe_float(val: Any) -> float | None:
        try:
            return float(val)
        except Exception:
            return None

    def _on_train_start(trainer: Any) -> None:
        save_dir = getattr(trainer, "save_dir", None)
        if save_dir:
            _emit(f"Logging results to: {save_dir}", event="train_start", save_dir=str(save_dir))

    def _on_fit_epoch_end(trainer: Any) -> None:
        try:
            epoch0 = getattr(trainer, "epoch", None)
            epochs_total = getattr(trainer, "epochs", None)
            if epoch0 is None or epochs_total is None:
                return
            epoch = int(epoch0) + 1
            total = int(epochs_total)

            losses: dict[str, float] = {}
            try:
                tloss = getattr(trainer, "tloss", None)
                label_loss_items = getattr(trainer, "label_loss_items", None)
                if tloss is not None and callable(label_loss_items):
                    items = label_loss_items(tloss) or {}
                    if isinstance(items, dict):
                        for k in ("box_loss", "cls_loss", "dfl_loss"):
                            v = _safe_float(items.get(k))
                            if v is not None:
                                losses[k] = v
            except Exception:
                pass

            metrics = getattr(trainer, "metrics", None) or {}
            metrics = metrics if isinstance(metrics, dict) else {}
            map50 = _first_metric(metrics, ["metrics/mAP50(B)", "metrics/mAP50"])
            map5095 = _first_metric(metrics, ["metrics/mAP50-95(B)", "metrics/mAP50-95"])
            precision = _first_metric(metrics, ["metrics/precision(B)", "metrics/precision"])
            recall = _first_metric(metrics, ["metrics/recall(B)", "metrics/recall"])

            parts: list[str] = [f"Epoch {epoch}/{total}"]
            if "box_loss" in losses:
                parts.append(f"box={losses['box_loss']:.4f}")
            if "cls_loss" in losses:
                parts.append(f"cls={losses['cls_loss']:.4f}")
            if "dfl_loss" in losses:
                parts.append(f"dfl={losses['dfl_loss']:.4f}")
            if precision is not None:
                parts.append(f"P={precision:.4f}")
            if recall is not None:
                parts.append(f"R={recall:.4f}")
            if map50 is not None:
                parts.append(f"mAP50={map50:.4f}")
            if map5095 is not None:
                parts.append(f"mAP50-95={map5095:.4f}")

            _emit(" | ".join(parts), event="epoch_end", epoch=epoch, epochs=total)
        except Exception:
            return

    try:
        model.add_callback("on_train_start", _on_train_start)
        model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)
    except Exception:
        pass

    _emit(
        f"Starting training: epochs={int(epochs)}, imgsz={int(imgsz)}, batch={int(batch)}, device={device or 'auto'}",
        event="train_begin",
    )

    try:
        model.train(**train_args)
    except Exception as e:  # noqa: BLE001
        return {"success": False, "error": f"Training failed: {e}", "dataset_dir": str(dataset_dir)}

    trainer = getattr(model, "trainer", None)
    trainer_save_dir = getattr(trainer, "save_dir", None)

    candidate_train_dirs: list[Path] = []
    if trainer_save_dir:
        candidate_train_dirs.append(Path(str(trainer_save_dir)))
    candidate_train_dirs.extend(
        [
            (train_project / train_name),
            (Path("runs") / "detect" / train_project / train_name),
            (Path("runs") / "detect" / train_name),
        ]
    )

    best_weights: Path | None = None
    train_dir: Path | None = None
    for candidate in candidate_train_dirs:
        weights_path = candidate / "weights" / "best.pt"
        if weights_path.exists():
            train_dir = candidate.resolve()
            best_weights = weights_path.resolve()
            break

    if best_weights is None:
        # Last-resort search: look for "<dataset_id>/weights/best.pt" under runs/
        runs_root = Path("runs")
        if runs_root.exists():
            for hit in runs_root.rglob("best.pt"):
                try:
                    if hit.parent.name == "weights" and hit.parent.parent.name == dataset_id:
                        train_dir = hit.parent.parent.resolve()
                        best_weights = hit.resolve()
                        break
                except Exception:
                    continue

    if best_weights is None:
        searched = [str(p.resolve()) for p in candidate_train_dirs]
        return {
            "success": False,
            "error": "Training completed but best.pt was not found.",
            "searched_train_dirs": searched,
            "hint": (
                "Ultralytics may write training outputs under runs/detect/... depending on configuration. "
                "Check the 'save_dir=' path in the training logs and confirm best.pt exists under weights/."
            ),
        }

    _emit(
        f"Training complete. Best checkpoint: {best_weights}",
        event="train_complete",
        train_dir=str(train_dir or ""),
        best_weights=str(best_weights),
    )

    # Validate to obtain a reliable metric for naming/registry.
    metrics: dict[str, Any] = {}
    try:
        val_model = yolo_class(str(best_weights))
        val_out = val_model.val(data=str(dataset_yaml_path), verbose=False)
        results_dict = getattr(val_out, "results_dict", None)
        if isinstance(results_dict, dict):
            metrics.update(
                {str(k): float(v) for k, v in results_dict.items() if isinstance(v, (int, float))}
            )
        box = getattr(val_out, "box", None)
        if box is not None:
            for k in ("map", "map50", "map75"):
                v = getattr(box, k, None)
                if isinstance(v, (int, float)):
                    metrics[k] = float(v)
    except Exception as e:  # noqa: BLE001
        metrics["warning"] = f"Validation metrics unavailable: {e}"

    def _metric_token(value: float | None) -> str:
        if value is None:
            return "na"
        try:
            return f"{float(value):.3f}".replace(".", "p")
        except Exception:
            return "na"

    base_tag = _safe_slug(Path(base_weights).stem)
    date_tag = datetime.utcnow().strftime("%Y%m%d")
    classes_tag = _safe_slug("-".join(class_names[:3])) if class_names else "classes"
    map_value = metrics.get("map") if isinstance(metrics.get("map"), (int, float)) else None
    metric_tag = f"map_{_metric_token(map_value)}"
    suffix = dataset_id.split("-")[-1]
    model_name = _safe_slug(f"{base_tag}-ft_{date_tag}_{classes_tag}_{metric_tag}_{suffix}")

    dst_weights = Path(_finetuned_dir()) / f"{model_name}.pt"
    if dst_weights.exists():
        model_name = _safe_slug(f"{model_name}-{uuid4().hex[:6]}")
        dst_weights = Path(_finetuned_dir()) / f"{model_name}.pt"
    shutil.copy2(best_weights, dst_weights)
    _emit(
        f"Saved finetuned weights: {dst_weights}",
        event="weights_saved",
        model_name=model_name,
        model_path=str(dst_weights),
    )

    registry_path = Path(_finetuned_dir()) / "registry.jsonl"
    registry_entry = {
        "model_name": model_name,
        "model_path": str(dst_weights),
        "created_at_utc": datetime.utcnow().isoformat() + "Z",
        "base_model": base_weights,
        "dataset_id": dataset_id,
        "dataset_dir": str(dataset_dir),
        "dataset_yaml_path": str(dataset_yaml_path),
        "class_names": class_names,
        "raw_class_ids": sorted_class_ids,
        "class_id_remap": {str(raw): int(mapped) for raw, mapped in class_id_map.items()},
        "split_strategy": split_strategy,
        "split_diagnostics": split_diagnostics,
        "metrics": metrics,
        "train_args": {
            "epochs": int(epochs),
            "imgsz": int(imgsz),
            "batch": int(batch),
            "val_split": float(val_split),
            "seed": int(seed),
            "device": device,
        },
        "train_dir": str(train_dir or ""),
        "best_weights_source": str(best_weights),
    }
    try:
        with registry_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(registry_entry) + "\n")
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to update YOLO finetune registry: %s", e)

    result = {
        "success": True,
        "dataset_id": dataset_id,
        "dataset_dir": str(dataset_dir),
        "dataset_yaml_path": str(dataset_yaml_path),
        "train_dir": str(train_dir or ""),
        "model_name": model_name,
        "model_path": str(dst_weights),
        "registry_path": str(registry_path),
        "class_names": class_names,
        "raw_class_ids": sorted_class_ids,
        "class_id_remap": {str(raw): int(mapped) for raw, mapped in class_id_map.items()},
        "split_strategy": split_strategy,
        "split_diagnostics": split_diagnostics,
        "metrics": metrics,
        "message": (
            "Finetuning complete. "
            f"Use yolo_detect with model_name='{model_name}' to run inference with the finetuned weights."
        ),
    }
    if warnings:
        result["warnings"] = warnings
    return result


class DeepSeekPostProcessor:
    """Post-processor for DeepSeek OCR responses with layout detection."""

    def __init__(self, output_dir="./data/ocr_results"):
        self.output_dir = output_dir
        os.makedirs(f"{output_dir}/crops", exist_ok=True)
        os.makedirs(f"{output_dir}/annotated", exist_ok=True)

    def _re_match(self, text):
        """Extracts reference and detection tags from the raw text."""
        pattern = r"(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)"
        matches = re.findall(pattern, text, re.DOTALL)

        matches_image = []
        matches_other = []

        for full_match, ref_content, det_content in matches:
            if "image" in ref_content:
                matches_image.append((full_match, ref_content, det_content))
            else:
                matches_other.append((full_match, ref_content, det_content))
        return matches, matches_image, matches_other

    def _extract_coords(self, det_content):
        """Parses the coordinate string [x1, y1, x2, y2]"""
        if det_content is None:
            return None
        raw = str(det_content).strip()
        if not raw:
            return None

        def _normalize(parsed: Any) -> list[list[float]] | None:
            if (
                isinstance(parsed, (list, tuple))
                and len(parsed) == 4
                and all(isinstance(v, (int, float)) for v in parsed)
            ):
                return [[float(v) for v in parsed]]
            if (
                isinstance(parsed, (list, tuple))
                and parsed
                and all(
                    isinstance(item, (list, tuple))
                    and len(item) == 4
                    and all(isinstance(v, (int, float)) for v in item)
                    for item in parsed
                )
            ):
                return [[float(v) for v in item] for item in parsed]
            return None

        try:
            parsed = json.loads(raw)
            normalized = _normalize(parsed)
            if normalized is not None:
                return normalized
        except Exception:
            pass

        # Fallback parser for non-JSON text: extract strict [x1,y1,x2,y2] groups.
        groups = re.findall(
            r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]",
            raw,
        )
        if not groups:
            return None
        return [[float(v1), float(v2), float(v3), float(v4)] for v1, v2, v3, v4 in groups]

    def process(self, original_image, raw_response_text, page_index=0):
        """Process API response and extract layout elements."""
        width, height = original_image.size
        draw_image = original_image.copy()
        draw = ImageDraw.Draw(draw_image, "RGBA")

        # Parse the special tokens
        _, match_imgs, match_others = self._re_match(raw_response_text)
        clean_markdown = raw_response_text

        # Process Figures/Images (Cropping)
        for idx, (full_string, label, det) in enumerate(match_imgs):
            coords_list = self._extract_coords(det)
            if not coords_list:
                continue

            for coords in coords_list:
                # DeepSeek outputs normalized coords (0-999). Scale them to image size.
                x1 = int(coords[0] / 999 * width)
                y1 = int(coords[1] / 999 * height)
                x2 = int(coords[2] / 999 * width)
                y2 = int(coords[3] / 999 * height)

                # Crop and Save
                try:
                    crop = original_image.crop((x1, y1, x2, y2))
                    crop_filename = f"crop_{page_index}_{idx}.jpg"
                    crop_path = os.path.join(self.output_dir, "crops", crop_filename)
                    crop.save(crop_path)

                    # Update Markdown to link to local image
                    clean_markdown = clean_markdown.replace(
                        full_string, f"\n![Figure {idx}]({crop_path})\n"
                    )
                except Exception as e:
                    logger.warning(f"Error cropping: {e}")

                # Draw Box (Blue for images)
                self._draw_box(draw, x1, y1, x2, y2, label, (0, 0, 255))

        # Process Other Elements (Headers, Tables)
        for full_string, label, det in match_others:
            coords_list = self._extract_coords(det)
            if not coords_list:
                continue

            # Remove the special tokens from markdown
            clean_markdown = clean_markdown.replace(full_string, "")

            for coords in coords_list:
                x1 = int(coords[0] / 999 * width)
                y1 = int(coords[1] / 999 * height)
                x2 = int(coords[2] / 999 * width)
                y2 = int(coords[3] / 999 * height)

                # Draw Box (Red for text/other)
                self._draw_box(draw, x1, y1, x2, y2, label, (255, 0, 0))

        # Save the Annotated Layout Image
        layout_path = os.path.join(self.output_dir, "annotated", f"layout_{page_index}.jpg")
        draw_image.save(layout_path)

        return clean_markdown, layout_path

    def _draw_box(self, draw, x1, y1, x2, y2, label, color):
        """Draw bounding box with semi-transparent fill."""
        fill_color = color + (30,)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2, fill=fill_color)


def ocr_document(
    file_paths: list[str], process_all_pages: bool = True, max_pages: int = 10
) -> dict[str, Any]:
    """
    OCR documents (PDFs or images) using DeepSeek-OCR with layout detection.
    Automatically extracts text, tables, figures, and preserves document structure.

    Args:
        file_paths: List of file paths to process (PDFs or images)
        process_all_pages: For PDFs, whether to process all pages or just first page
        max_pages: Maximum number of pages to process per PDF (default: 10)

    Returns:
        Dictionary with OCR results, extracted text, and visualization paths
    """
    try:
        from openai import OpenAI
        from pdf2image import convert_from_path
    except ImportError:
        return {
            "success": False,
            "error": "Required packages not installed. Install with: pip install openai pdf2image pillow",
        }

    settings = get_settings()

    # Initialize OCR client
    try:
        ocr_client = OpenAI(
            api_key=settings.ocr_api_key,
            base_url=settings.ocr_base_url,
            timeout=float(settings.ocr_timeout),
            max_retries=settings.openai_max_retries,
        )
        logger.info("OCR client initialized at %s", settings.ocr_base_url)
    except Exception as e:
        return {"success": False, "error": f"Failed to initialize OCR client: {str(e)}"}

    # Initialize post-processor
    processor = DeepSeekPostProcessor()

    results = []

    for file_path in file_paths:
        try:
            if not os.path.exists(file_path):
                results.append(
                    {
                        "file": os.path.basename(file_path),
                        "success": False,
                        "error": f"File not found: {file_path}",
                    }
                )
                continue

            file_name = os.path.basename(file_path)
            file_ext = os.path.splitext(file_name)[1].lower()

            logger.info(f"Processing {file_name} with DeepSeek OCR")

            # Determine if PDF or image
            is_pdf = file_ext == ".pdf"

            if is_pdf:
                # Convert PDF pages to images
                logger.info("Converting PDF to images...")
                images = convert_from_path(file_path, dpi=200)

                if not process_all_pages:
                    images = images[:1]
                else:
                    images = images[:max_pages]

                logger.info(f"Processing {len(images)} page(s)")
            else:
                # Single image
                images = [Image.open(file_path)]

            # Process each page/image
            page_results = []
            all_markdown = []

            for page_idx, img in enumerate(images):
                # Encode image to base64
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG")
                img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                img_url = f"data:image/jpeg;base64,{img_b64}"

                # OCR request
                logger.info(f"OCR processing page {page_idx + 1}/{len(images)}")
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": img_url}},
                            {
                                "type": "text",
                                "text": "Transcribe this document into markdown format. Extract all text, tables, and figures with their locations.",
                            },
                        ],
                    }
                ]

                response = ocr_client.chat.completions.create(
                    model=settings.ocr_model,
                    messages=messages,
                    max_tokens=4096,
                    temperature=0.0,
                    extra_body={
                        "skip_special_tokens": False,
                        "vllm_xargs": {
                            "ngram_size": 30,
                            "window_size": 90,
                            "whitelist_token_ids": [128821, 128822],
                        },
                    },
                )

                raw_text = response.choices[0].message.content

                # Post-process to extract layout and clean markdown
                clean_markdown, layout_path = processor.process(
                    original_image=img, raw_response_text=raw_text, page_index=page_idx
                )

                page_results.append(
                    {"page": page_idx + 1, "markdown": clean_markdown, "layout_image": layout_path}
                )

                # Add page marker for multi-page documents
                if len(images) > 1:
                    all_markdown.append(f"\n\n--- Page {page_idx + 1} ---\n\n{clean_markdown}")
                else:
                    all_markdown.append(clean_markdown)

            # Combine all pages
            combined_markdown = "\n".join(all_markdown)

            results.append(
                {
                    "file": file_name,
                    "path": file_path,
                    "success": True,
                    "type": "pdf" if is_pdf else "image",
                    "pages_processed": len(images),
                    "markdown": combined_markdown,
                    "page_details": page_results,
                    "output_directory": processor.output_dir,
                }
            )

            logger.info(f"Successfully processed {file_name}: {len(images)} page(s)")

        except Exception as file_error:
            results.append(
                {
                    "file": os.path.basename(file_path),
                    "path": file_path,
                    "success": False,
                    "error": str(file_error),
                }
            )
            logger.error(f"Failed to process {file_path}: {str(file_error)}")

    # Summary
    successful = [r for r in results if r.get("success")]
    total_pages = sum(r.get("pages_processed", 0) for r in successful)

    return {
        "success": len(successful) > 0,
        "processed_files": len(successful),
        "total_files": len(results),
        "total_pages_processed": total_pages,
        "results": results,
        "output_directory": processor.output_dir,
        "model": settings.ocr_model,
    }


def annotation_add(
    resource, ann="tag", ann_name=None, ann_value=None, ann_type=None, add_if_exists=False
):
    """
    Helper function to add annotation tags to a BisQue resource XML.

    Args:
        resource: XML element representing the resource
        ann: Annotation element type (default: 'tag')
        ann_name: Name of the annotation
        ann_value: Value of the annotation
        ann_type: Type of annotation (default: 'annotation')
        add_if_exists: Whether to add duplicate annotations

    Returns:
        List of modified elements
    """
    modified = []
    if ann_name is not None and ann_value is not None:
        # Check if annotation already exists
        xpath = f'//{ann}[@name="{ann_name}" and @value="{ann_value}"]'
        if ann_type is not None:
            xpath = xpath.replace("]", f' and @type="{ann_type}"]')
        anns = resource.xpath(xpath)

        # Add annotation if it doesn't exist or if add_if_exists is True
        if len(anns) < 1 or add_if_exists is True:
            g = etree.SubElement(resource, ann, name=ann_name, value=ann_value)
            if ann_type is not None:
                g.set("type", ann_type)
            modified.append({"g": g})
    return modified


def add_tags_to_resource(resource_uri: str, tags: list[dict[str, str]]) -> dict[str, Any]:
    """
    Add metadata tags to a BisQue resource.

    Args:
        resource_uri: Full URI of the resource
        tags: List of tag dictionaries with 'name' and 'value' keys

    Returns:
        Dictionary with operation status
    """
    try:
        from bqapi.comm import BQSession
    except ImportError:
        return {"success": False, "error": "bqapi package not installed"}

    try:
        bq = BQSession()
        root, _, _ = _init_bisque_session_with_runtime_auth(bq=bq)
        normalized = _normalize_bisque_resource_uri(resource_uri, root)

        # Initialize session
        logger.info(f"Adding tags to resource: {normalized}")

        resource = bq.load(normalized, view="deep")

        if resource is None:
            return {
                "success": False,
                "error": f"Resource not found or failed to load: {resource_uri}",
            }

        valid_tags: list[dict[str, str]] = []
        for tag_dict in tags:
            if "name" not in tag_dict or "value" not in tag_dict:
                logger.warning(f"Skipping invalid tag: {tag_dict}")
                continue
            name = str(tag_dict["name"]).strip()
            value = str(tag_dict["value"]).strip()
            if not name:
                logger.warning("Skipping tag with empty name: %s", tag_dict)
                continue
            resource.add_tag(name=name, value=value, type="annotation")
            valid_tags.append({"name": name, "value": value})
            logger.info("Added tag via BisQue resource save: %s=%s", name, value)

        if not valid_tags:
            return {
                "success": False,
                "error": "No valid tags were provided.",
                "resource_uri": normalized,
            }

        bq.save(resource)

        def _count_verified_tags(xml: etree._Element | None) -> int:
            if xml is None:
                return 0
            return sum(
                1
                for tag_dict in valid_tags
                if xml.xpath(
                    "./tag[@name=$name and @value=$value and (@type='annotation' or not(@type))]",
                    name=tag_dict["name"],
                    value=tag_dict["value"],
                )
            )

        verification_xml, verified_state, verification_attempts = _wait_for_bisque_resource_state(
            bq=bq,
            resource_uri=normalized,
            predicate=lambda xml, _exc: _count_verified_tags(xml) == len(valid_tags),
        )
        verified_tags = _count_verified_tags(verification_xml)

        ui_artifacts = [
            {
                "type": "metrics",
                "title": "Tags added",
                "payload": {
                    "requested_tags": len(valid_tags),
                    "verified_tags": verified_tags,
                    "verification_attempts": verification_attempts,
                },
            },
            {
                "type": "table",
                "title": "Applied tags",
                "payload": valid_tags,
            },
        ]

        client_view_url = _bisque_user_facing_resource_url(normalized, root)
        return {
            "success": verified_state and verified_tags == len(valid_tags),
            "resource_uri": normalized,
            "client_view_url": client_view_url,
            "tags_added": valid_tags,
            "total_tags": len(valid_tags),
            "verified_tags": verified_tags,
            "verification_attempts": verification_attempts,
            "ui_artifacts": ui_artifacts,
        }

    except Exception as e:
        logger.error(f"Failed to add tags: {str(e)}")
        return {"success": False, "error": str(e)}


def bisque_fetch_xml(
    resource_uri: str,
    view: str = "deep",
    output_path: str | None = None,
    max_chars: int = 20000,
) -> dict[str, Any]:
    """
    Fetch raw XML for a BisQue resource or endpoint.

    Args:
        resource_uri: Resource URI, resource_uniq, or BisQue path.
        view: BisQue view level (short, full, deep). Default: deep.
        output_path: Optional path to save XML to disk.
        max_chars: Max characters to return inline (truncated if longer).
    """
    try:
        from bqapi.comm import BQSession
    except ImportError:
        return {"success": False, "error": "bqapi package not installed"}

    try:
        bq = BQSession()
        root, _, _ = _init_bisque_session_with_runtime_auth(bq=bq)

        if (
            resource_uri
            and not resource_uri.startswith(("http://", "https://"))
            and "/" not in resource_uri
        ):
            normalized = _normalize_bisque_resource_uri(resource_uri, root)
        else:
            normalized = _normalize_bisque_url(resource_uri, root)
        xml = (
            _session_fetchxml_safe(bq, normalized, view=view)
            if view
            else _session_fetchxml_safe(bq, normalized)
        )
        if hasattr(xml, "tag") and xml.tag == "response" and len(xml):
            xml = xml[0]

        try:
            xml_bytes = etree.tostring(xml, pretty_print=True)
        except TypeError:
            xml_bytes = etree.tostring(xml)
        except Exception:
            xml_bytes = str(xml)
        if isinstance(xml_bytes, bytes):
            xml_text = xml_bytes.decode("utf-8", errors="ignore")
        else:
            xml_text = str(xml_bytes)

        saved_path = None
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text(xml_text, encoding="utf-8")
            saved_path = str(Path(output_path))

        truncated = len(xml_text) > max_chars
        preview = xml_text[:max_chars] + ("...(truncated)" if truncated else "")

        ui_artifacts = [
            {
                "type": "write",
                "title": "XML preview",
                "payload": preview,
            }
        ]

        return {
            "success": True,
            "resource_uri": normalized,
            "xml_preview": preview,
            "truncated": truncated,
            "saved_path": saved_path,
            "ui_artifacts": ui_artifacts,
        }

    except Exception as e:
        logger.error(f"Failed to fetch XML: {str(e)}")
        return {"success": False, "error": str(e)}


def bisque_download_dataset(
    dataset_uri: str,
    output_dir: str,
    limit: int | None = None,
    use_localpath: bool = False,
) -> dict[str, Any]:
    """
    Download all images in a BisQue dataset to a local directory.

    Args:
        dataset_uri: Dataset URI or resource_uniq
        output_dir: Local output directory
        limit: Optional max number of members to download
        use_localpath: If running on BisQue host, use local paths (faster)
    """
    try:
        from bqapi.comm import BQSession
        from bqapi.util import fetch_image_pixels
    except ImportError:
        return {"success": False, "error": "bqapi package not installed"}

    try:
        dest = Path(output_dir)
        dest.mkdir(parents=True, exist_ok=True)

        bq = BQSession()
        root, _, _ = _init_bisque_session_with_runtime_auth(bq=bq)

        normalized = _normalize_bisque_resource_uri(dataset_uri, root)
        dataset_xml = _session_fetchxml_safe(bq, normalized, view="deep")
        if hasattr(dataset_xml, "tag") and dataset_xml.tag == "response" and len(dataset_xml):
            dataset_xml = dataset_xml[0]

        members = dataset_xml.findall('.//value[@type="object"]')
        member_uris = [_normalize_bisque_resource_uri(m.text, root) for m in members if m.text]
        if limit:
            member_uris = member_uris[: int(limit)]

        results = []
        for uri in member_uris:
            try:
                fetched = fetch_image_pixels(bq, uri, str(dest), uselocalpath=use_localpath)
                local_path = fetched.get(uri)
                results.append({"uri": uri, "path": local_path, "success": True})
            except Exception as e:
                results.append({"uri": uri, "success": False, "error": str(e)})

        success_count = sum(1 for r in results if r.get("success"))
        ui_artifacts = [
            {
                "type": "metrics",
                "title": "Dataset download",
                "payload": {
                    "downloaded": success_count,
                    "failed": len(results) - success_count,
                },
            },
            {
                "type": "table",
                "title": "Downloaded dataset members",
                "payload": results[:200],
            },
        ]

        return {
            "success": success_count > 0,
            "dataset_uri": normalized,
            "output_dir": str(dest),
            "total_members": len(member_uris),
            "downloaded": success_count,
            "results": results,
            "ui_artifacts": ui_artifacts,
        }

    except Exception as e:
        logger.error(f"Failed to download dataset: {str(e)}")
        return {"success": False, "error": str(e)}


def _coerce_named_value_mapping(raw: Any) -> dict[str, str]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return {
            str(key).strip(): str(value)
            for key, value in raw.items()
            if str(key or "").strip() and value is not None
        }
    if not isinstance(raw, list):
        return {}
    normalized: dict[str, str] = {}
    for item in raw[:200]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        value = item.get("value")
        if value is None:
            values = item.get("values")
            if isinstance(values, list) and values:
                value = values[0]
        if value is None:
            continue
        normalized[name] = str(value)
    return normalized


def _normalize_bisque_name_key(value: str | None) -> str:
    token = str(value or "").strip().lower()
    return re.sub(r"[^a-z0-9]+", "", token)


def _collect_bisque_dataset_nodes(root_node: etree._Element) -> list[etree._Element]:
    nodes: list[etree._Element] = []
    if str(root_node.tag or "").strip().lower() == "dataset":
        nodes.append(root_node)
    nodes.extend(list(root_node.findall(".//dataset")))
    deduped: list[etree._Element] = []
    seen: set[int] = set()
    for node in nodes:
        marker = id(node)
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append(node)
    return deduped


def _dataset_match_score(candidate_name: str, requested_name: str) -> tuple[int, str | None]:
    candidate = str(candidate_name or "").strip()
    requested = str(requested_name or "").strip()
    if not candidate or not requested:
        return 0, None
    if candidate == requested:
        return 4, "exact_name"
    if candidate.lower() == requested.lower():
        return 3, "casefold_name"
    candidate_key = _normalize_bisque_name_key(candidate)
    requested_key = _normalize_bisque_name_key(requested)
    if candidate_key and candidate_key == requested_key:
        return 2, "normalized_name"
    if requested_key and requested_key in candidate_key:
        return 1, "partial_name"
    return 0, None


def _resolve_bisque_dataset_target_with_session(
    *,
    bq: Any,
    bisque_root: str,
    dataset_uri: str | None = None,
    dataset_name: str | None = None,
) -> dict[str, Any] | None:
    requested_dataset_uri = str(dataset_uri or "").strip()
    if requested_dataset_uri:
        normalized_uri = _normalize_bisque_resource_uri(requested_dataset_uri, bisque_root)
        return {
            "dataset_uri": normalized_uri,
            "dataset_name": str(dataset_name or "").strip() or None,
            "dataset_uniq": _short_bisque_id(normalized_uri),
            "match_type": "uri",
            "candidate_names": [],
        }

    requested_dataset_name = str(dataset_name or "").strip()
    if not requested_dataset_name:
        return None

    query_candidates = [
        {"tag_query": f'"name":"{requested_dataset_name}"', "view": "deep"},
        {"name": requested_dataset_name, "view": "deep"},
        {"tag_query": f'"name":"{requested_dataset_name}"'},
        {"view": "deep"},
    ]
    candidates: list[dict[str, Any]] = []
    seen_uris: set[str] = set()
    for params in query_candidates:
        try:
            payload = _session_fetchxml_safe(bq, f"{bisque_root}/data_service/dataset", **params)
        except Exception:
            continue
        for node in _collect_bisque_dataset_nodes(payload):
            candidate_name = str(node.get("name") or "").strip()
            raw_uri = str(node.get("uri") or "").strip()
            resource_uniq = str(node.get("resource_uniq") or "").strip()
            normalized_uri = None
            if raw_uri:
                try:
                    normalized_uri = _normalize_bisque_resource_uri(raw_uri, bisque_root)
                except Exception:
                    normalized_uri = raw_uri
            elif resource_uniq:
                normalized_uri = f"{bisque_root.rstrip('/')}/data_service/dataset/{resource_uniq}"
            if not normalized_uri or normalized_uri in seen_uris:
                continue
            seen_uris.add(normalized_uri)
            score, match_type = _dataset_match_score(candidate_name, requested_dataset_name)
            candidates.append(
                {
                    "dataset_uri": normalized_uri,
                    "dataset_name": candidate_name or requested_dataset_name,
                    "dataset_uniq": resource_uniq or _short_bisque_id(normalized_uri),
                    "match_type": match_type,
                    "match_score": score,
                    "created": str(node.get("ts") or node.get("created") or "").strip(),
                }
            )

    matches = [item for item in candidates if int(item.get("match_score") or 0) > 0]
    if not matches:
        return None
    ordered_matches = sorted(
        matches,
        key=lambda item: (
            int(item.get("match_score") or 0),
            str(item.get("created") or ""),
            str(item.get("dataset_name") or ""),
        ),
        reverse=True,
    )
    best_score = int(ordered_matches[0].get("match_score") or 0)
    best_matches = [
        item for item in ordered_matches if int(item.get("match_score") or 0) == best_score
    ]
    candidate_names = sorted(
        {
            str(item.get("dataset_name") or "").strip()
            for item in matches
            if str(item.get("dataset_name") or "").strip()
        }
    )[:10]
    if len(best_matches) > 1:
        return {
            "status": "ambiguous",
            "dataset_name": requested_dataset_name,
            "candidate_names": candidate_names,
            "candidate_datasets": [
                {
                    "dataset_uri": str(item.get("dataset_uri") or "").strip() or None,
                    "dataset_name": str(item.get("dataset_name") or "").strip() or None,
                    "dataset_uniq": str(item.get("dataset_uniq") or "").strip() or None,
                    "match_type": str(item.get("match_type") or "").strip() or None,
                    "match_score": int(item.get("match_score") or 0),
                    "created": str(item.get("created") or "").strip() or None,
                }
                for item in best_matches[:8]
            ],
            "error": (
                f"Dataset name '{requested_dataset_name}' is ambiguous in BisQue. "
                f"{len(best_matches)} matching datasets were found."
            ),
            "message": (
                f"Dataset '{requested_dataset_name}' matched multiple BisQue datasets. "
                "Choose one of the returned dataset URIs before mutating data."
            ),
        }
    best = dict(best_matches[0])
    best["candidate_names"] = candidate_names
    best["status"] = "resolved"
    return best


def _extract_requested_dataset_name_for_search(
    *,
    text: str | list[str] | None,
    tag_query: str | None,
    tag_filters: dict[str, Any] | list[dict[str, Any]] | None,
) -> str | None:
    normalized_filters = _coerce_bisque_filter_mapping(tag_filters) or {}
    for key in ("name", "resource_name"):
        raw_value = normalized_filters.get(key)
        values = raw_value if isinstance(raw_value, list) else [raw_value]
        for item in values:
            candidate = _clean_inferred_bisque_dataset_target(str(item or ""))
            if not candidate:
                continue
            if re.match(r"^(>=|<=|!=|==|>|<)", candidate):
                continue
            if "*" in candidate or ":" in candidate:
                continue
            return candidate

    if isinstance(text, str):
        for pattern in (
            r"\b(?:named|called)\s+[\"“']([^\"”']{2,160})[\"”']",
            r"[\"“']([^\"”']{2,160})[\"”']",
        ):
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                candidate = _clean_inferred_bisque_dataset_target(match.group(1))
                if candidate:
                    return candidate
        if text.strip() and len(text.strip()) <= 180 and not re.search(r"[.!?]", text):
            return _clean_inferred_bisque_dataset_target(text)

    if isinstance(tag_query, str):
        for pattern in (
            r"(?:@?name|resource_name)\s*:\s*\"([^\"]{2,160})\"",
            r"(?:@?name|resource_name)\s*:\s*([A-Za-z0-9_.:-]{2,160})",
        ):
            match = re.search(pattern, tag_query, flags=re.IGNORECASE)
            if match:
                candidate = _clean_inferred_bisque_dataset_target(match.group(1))
                if candidate and "*" not in candidate:
                    return candidate

    return None


def _normalize_bisque_dataset_resource_inputs(
    *,
    bq: Any,
    bisque_root: str,
    resource_uris: list[str],
) -> tuple[list[str], list[str]]:
    normalized: list[str] = []
    errors: list[str] = []
    seen: set[str] = set()
    for raw in resource_uris:
        token = str(raw or "").strip()
        if not token:
            continue
        local_path = Path(token).expanduser()
        if not local_path.exists():
            candidate = _science_data_root_path() / token
            if candidate.exists():
                local_path = candidate
        try:
            if local_path.exists() and local_path.is_file():
                existing_upload = _lookup_session_visible_canonical_bisque_upload_for_local_path(
                    bq=bq,
                    bisque_root=bisque_root,
                    path=local_path,
                )
                if existing_upload is not None:
                    normalized_value = _normalize_bisque_resource_uri(
                        str(existing_upload.get("canonical_resource_uri") or ""),
                        bisque_root,
                    )
                else:
                    upload_response = bq.postblob(
                        str(local_path),
                        xml=_bisque_upload_resource_xml(local_path),
                    )
                    upload_uri = _extract_bisque_resource_uri(upload_response)
                    if not upload_uri:
                        upload_xml = _coerce_xml_element(upload_response)
                        upload_uri = str(upload_xml.get("uri") or "").strip()
                        if not upload_uri and len(upload_xml):
                            upload_uri = str(upload_xml[0].get("uri") or "").strip()
                    if not upload_uri:
                        raise ValueError(f"Failed to upload local resource {local_path}")
                    normalized_value = _normalize_bisque_resource_uri(upload_uri, bisque_root)
            else:
                normalized_value = _normalize_bisque_resource_uri(token, bisque_root)
            if normalized_value in seen:
                continue
            seen.add(normalized_value)
            normalized.append(normalized_value)
        except Exception as exc:
            errors.append(f"{token}: {exc}")
    return normalized, errors


def _build_bisque_dataset_ui_artifacts(
    *,
    title: str,
    dataset_name: str | None,
    dataset_uri: str | None,
    resource_uris: list[str],
    added: int | None = None,
    failed: int | None = None,
) -> list[dict[str, Any]]:
    metrics_payload: dict[str, Any] = {
        "dataset_name": dataset_name,
        "dataset_uri": dataset_uri,
        "members": len(resource_uris),
    }
    if added is not None:
        metrics_payload["added"] = int(added)
    if failed is not None:
        metrics_payload["failed"] = int(failed)
    return [
        {
            "type": "metrics",
            "title": title,
            "payload": metrics_payload,
        },
        {
            "type": "table",
            "title": "Dataset members",
            "payload": [{"uri": uri} for uri in resource_uris[:200]],
        },
    ]


def _extract_bisque_dataset_document(xml: etree._Element) -> etree._Element:
    if str(xml.tag or "").strip().lower() == "dataset":
        return etree.fromstring(etree.tostring(xml))
    dataset = xml.find(".//dataset")
    if dataset is None:
        raise ValueError("BisQue dataset payload did not contain a <dataset> document.")
    return etree.fromstring(etree.tostring(dataset))


def _dataset_member_values(dataset_xml: etree._Element) -> list[etree._Element]:
    return list(dataset_xml.findall("./value"))


def _dataset_member_replace_request(
    *,
    dataset_xml: etree._Element,
    bisque_root: str,
    remove_resource_uri: str,
) -> tuple[etree._Element, etree._Element, bool]:
    request = etree.Element("request")
    updated_dataset = etree.fromstring(etree.tostring(dataset_xml))
    for value_node in list(_dataset_member_values(updated_dataset)):
        updated_dataset.remove(value_node)
    removed = False
    next_index = 0
    for value_node in _dataset_member_values(dataset_xml):
        text = str(value_node.text or "").strip()
        try:
            candidate_uri = _normalize_bisque_resource_uri(text, bisque_root)
        except Exception:
            candidate_uri = text
        if candidate_uri == remove_resource_uri:
            removed = True
            continue
        cloned = etree.fromstring(etree.tostring(value_node))
        cloned.set("index", str(next_index))
        request.append(cloned)
        updated_dataset.append(etree.fromstring(etree.tostring(cloned)))
        next_index += 1
    return request, updated_dataset, removed


def _dataset_member_uri_set(dataset_xml: etree._Element, bisque_root: str) -> set[str]:
    members: set[str] = set()
    for value_node in _dataset_member_values(dataset_xml):
        text = str(value_node.text or "").strip()
        if not text:
            continue
        try:
            members.add(_normalize_bisque_resource_uri(text, bisque_root))
        except Exception:
            members.add(text)
    return members


_BISQUE_SETTLE_MAX_ATTEMPTS = max(
    1,
    int(os.getenv("BISQUE_SETTLE_MAX_ATTEMPTS", "12") or 12),
)
_BISQUE_SETTLE_INTERVAL_SECONDS = max(
    0.1,
    float(os.getenv("BISQUE_SETTLE_INTERVAL_SECONDS", "0.5") or 0.5),
)


def _unwrap_bisque_resource_document(xml: etree._Element) -> etree._Element:
    if hasattr(xml, "tag") and str(xml.tag or "").strip().lower() == "response" and len(xml):
        return xml[0]
    return xml


def _is_bisque_missing_resource_error(exc: Exception | str | None) -> bool:
    message = str(exc or "").strip().lower()
    if not message:
        return False
    return bool(
        re.search(
            r"\b(404|not found|missing|no such resource|resource does not exist|unknown resource)\b",
            message,
        )
    )


def _wait_for_bisque_dataset_membership_state(
    *,
    bq: Any,
    dataset_uri: str,
    bisque_root: str,
    expected_present: set[str] | None = None,
    expected_absent: set[str] | None = None,
    max_attempts: int | None = None,
    poll_interval_seconds: float | None = None,
) -> tuple[etree._Element, set[str], int]:
    attempts = max(1, int(max_attempts or _BISQUE_SETTLE_MAX_ATTEMPTS))
    interval = max(
        0.1,
        float(
            poll_interval_seconds
            if poll_interval_seconds is not None
            else _BISQUE_SETTLE_INTERVAL_SECONDS
        ),
    )
    expected_present = set(expected_present or set())
    expected_absent = set(expected_absent or set())
    last_document: etree._Element | None = None
    last_members: set[str] = set()

    for attempt_index in range(attempts):
        payload = _session_fetchxml_safe(bq, dataset_uri, view="deep", cache="false")
        last_document = _extract_bisque_dataset_document(payload)
        last_members = _dataset_member_uri_set(last_document, bisque_root)
        if expected_present.issubset(last_members) and expected_absent.isdisjoint(last_members):
            return last_document, last_members, attempt_index + 1
        if attempt_index < attempts - 1:
            time.sleep(interval)

    if last_document is None:
        raise ValueError(f"Unable to reload BisQue dataset {dataset_uri} after update.")
    return last_document, last_members, attempts


def _wait_for_bisque_resource_state(
    *,
    bq: Any,
    resource_uri: str,
    predicate: Callable[[etree._Element | None, Exception | None], bool],
    cache: str | None = None,
    max_attempts: int | None = None,
    poll_interval_seconds: float | None = None,
) -> tuple[etree._Element | None, bool, int]:
    attempts = max(1, int(max_attempts or _BISQUE_SETTLE_MAX_ATTEMPTS))
    interval = max(
        0.1,
        float(
            poll_interval_seconds
            if poll_interval_seconds is not None
            else _BISQUE_SETTLE_INTERVAL_SECONDS
        ),
    )
    last_xml: etree._Element | None = None
    last_error: Exception | None = None

    for attempt_index in range(attempts):
        try:
            fetch_params: dict[str, Any] = {"view": "deep"}
            if cache is not None:
                fetch_params["cache"] = cache
            fetched = _session_fetchxml_safe(bq, resource_uri, **fetch_params)
            last_xml = _unwrap_bisque_resource_document(fetched)
            last_error = None
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            last_xml = None
            if predicate(None, exc):
                return None, True, attempt_index + 1
        else:
            if predicate(last_xml, None):
                return last_xml, True, attempt_index + 1
        if attempt_index < attempts - 1:
            time.sleep(interval)

    if (
        last_error is not None
        and last_xml is None
        and not _is_bisque_missing_resource_error(last_error)
    ):
        raise last_error
    return last_xml, False, attempts


def _find_bisque_datasets_containing_resource(
    *,
    bq: Any,
    bisque_root: str,
    resource_uri: str,
) -> list[dict[str, str]]:
    normalized_resource_uri = _normalize_bisque_resource_uri(resource_uri, bisque_root)
    matches: list[dict[str, str]] = []
    seen_dataset_uris: set[str] = set()
    query_candidates = [
        {"view": "deep", "limit": 200, "cache": "false"},
        {"view": "deep", "cache": "false"},
        {"view": "deep", "limit": 200},
        {"view": "deep"},
    ]

    for params in query_candidates:
        try:
            payload = _session_fetchxml_safe(bq, f"{bisque_root}/data_service/dataset", **params)
        except Exception:
            continue

        for node in _collect_bisque_dataset_nodes(payload):
            try:
                document = _extract_bisque_dataset_document(node)
            except Exception:
                continue
            members = _dataset_member_uri_set(document, bisque_root)
            if normalized_resource_uri not in members:
                continue

            raw_uri = str(node.get("uri") or "").strip()
            resource_uniq = str(node.get("resource_uniq") or "").strip()
            dataset_uri = (
                _normalize_bisque_resource_uri(raw_uri, bisque_root)
                if raw_uri
                else f"{bisque_root.rstrip('/')}/data_service/dataset/{resource_uniq}"
            )
            if not dataset_uri or dataset_uri in seen_dataset_uris:
                continue
            seen_dataset_uris.add(dataset_uri)
            matches.append(
                {
                    "dataset_uri": dataset_uri,
                    "dataset_uniq": resource_uniq or _short_bisque_id(dataset_uri),
                    "dataset_name": str(node.get("name") or "").strip(),
                }
            )

        if matches:
            break

    return matches


def _remove_resource_from_all_bisque_datasets(
    *,
    bq: Any,
    bisque_root: str,
    resource_uri: str,
) -> dict[str, Any]:
    normalized_resource_uri = _normalize_bisque_resource_uri(resource_uri, bisque_root)
    resource_uniq = _short_bisque_id(normalized_resource_uri)
    containing_datasets = _find_bisque_datasets_containing_resource(
        bq=bq,
        bisque_root=bisque_root,
        resource_uri=normalized_resource_uri,
    )
    if not containing_datasets:
        return {
            "searched": True,
            "found": 0,
            "removed": 0,
            "results": [],
        }

    dataset_service = None
    try:
        dataset_service = bq.service("dataset_service")
    except Exception:
        dataset_service = None

    results: list[dict[str, Any]] = []
    removed = 0
    for dataset in containing_datasets:
        dataset_uri = str(dataset.get("dataset_uri") or "").strip()
        dataset_uniq = str(dataset.get("dataset_uniq") or "").strip() or _short_bisque_id(
            dataset_uri
        )
        dataset_name = str(dataset.get("dataset_name") or "").strip() or None
        removed_here = False
        error_message: str | None = None
        settle_attempts = 0

        try:
            payload = _session_fetchxml_safe(bq, dataset_uri, view="deep", cache="false")
            document = _extract_bisque_dataset_document(payload)
            replace_request, updated_document, member_present = _dataset_member_replace_request(
                dataset_xml=document,
                bisque_root=bisque_root,
                remove_resource_uri=normalized_resource_uri,
            )

            strategy_errors: list[str] = []
            if member_present:
                strategies: list[tuple[str, Callable[[], Any]]] = []
                if dataset_service is not None and dataset_uniq:
                    strategies.append(
                        (
                            "dataset_service_delete_member",
                            lambda: dataset_service.delete_member(dataset_uniq, resource_uniq),
                        )
                    )
                strategies.extend(
                    [
                        (
                            "value_replace",
                            lambda: _session_postxml_safe(
                                bq,
                                f"{dataset_uri.rstrip('/')}/value",
                                replace_request,
                                method="PUT",
                                view="deep",
                            ),
                        ),
                        (
                            "dataset_put",
                            lambda: _session_postxml_safe(
                                bq, dataset_uri, updated_document, method="PUT"
                            ),
                        ),
                    ]
                )

                for strategy_name, strategy in strategies:
                    try:
                        strategy()
                        _verification_document, remaining_members, settle_attempts = (
                            _wait_for_bisque_dataset_membership_state(
                                bq=bq,
                                dataset_uri=dataset_uri,
                                bisque_root=bisque_root,
                                expected_absent={normalized_resource_uri},
                            )
                        )
                        removed_here = normalized_resource_uri not in remaining_members
                        if removed_here:
                            removed += 1
                            break
                        strategy_errors.append(
                            f"{strategy_name}: dataset still referenced {normalized_resource_uri} after update"
                        )
                    except Exception as exc:
                        strategy_errors.append(f"{strategy_name}: {exc}")
            else:
                removed_here = True
                removed += 1

            if not removed_here:
                error_message = "; ".join(strategy_errors) or (
                    f"Dataset {dataset_uri} still referenced {normalized_resource_uri} after cleanup."
                )
        except Exception as exc:
            error_message = str(exc)

        results.append(
            {
                "dataset_uri": dataset_uri,
                "dataset_uniq": dataset_uniq,
                "dataset_name": dataset_name,
                "verification_attempts": settle_attempts,
                "success": removed_here,
                **({} if removed_here else {"error": error_message or "dataset cleanup failed"}),
            }
        )

    return {
        "searched": True,
        "found": len(containing_datasets),
        "removed": removed,
        "results": results,
    }


def _append_members_to_dataset_document(
    *,
    dataset_xml: etree._Element,
    bisque_root: str,
    resource_uris: list[str],
) -> tuple[etree._Element, list[str]]:
    existing_members = _dataset_member_uri_set(dataset_xml, bisque_root)
    added: list[str] = []

    for normalized_uri in resource_uris:
        if normalized_uri in existing_members:
            continue
        value = etree.SubElement(dataset_xml, "value", type="object")
        value.text = normalized_uri
        existing_members.add(normalized_uri)
        added.append(normalized_uri)

    for index, value_node in enumerate(_dataset_member_values(dataset_xml)):
        value_node.set("index", str(index))

    return dataset_xml, added


def _create_bisque_dataset_with_session(
    *,
    bq: Any,
    bisque_root: str,
    name: str,
    resource_uris: list[str],
    tags: dict[str, str] | list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    dataset_name = str(name or "").strip()
    if not dataset_name:
        return {"success": False, "error": "name is required."}

    normalized_uris, input_errors = _normalize_bisque_dataset_resource_inputs(
        bq=bq,
        bisque_root=bisque_root,
        resource_uris=list(resource_uris or []),
    )
    if not normalized_uris:
        error = "resource_uris cannot be empty." if not input_errors else input_errors[0]
        return {"success": False, "error": error, "input_errors": input_errors}

    dataset = etree.Element("dataset", name=dataset_name)
    for key, value in _coerce_named_value_mapping(tags).items():
        etree.SubElement(dataset, "tag", name=str(key), value=str(value))

    for normalized_uri in normalized_uris:
        value = etree.SubElement(dataset, "value", type="object")
        value.text = normalized_uri

    created = _session_postxml_safe(bq, f"{bisque_root}/data_service/dataset", dataset)
    dataset_uri = str(created.get("uri") or "").strip()
    if not dataset_uri and len(created):
        dataset_uri = str(created[0].get("uri") or "").strip()
    dataset_links = _build_bisque_resource_links(dataset_uri, bisque_root) if dataset_uri else {}
    resolved_dataset_uri = (
        str(dataset_links.get("resource_uri") or dataset_uri or "").strip() or None
    )
    dataset_client_view_url = (
        str(dataset_links.get("client_view_url") or "").strip() or None if dataset_links else None
    )
    message = f"Created BisQue dataset '{dataset_name}' with {len(normalized_uris)} resource(s)."
    return {
        "success": resolved_dataset_uri is not None,
        "action": "created",
        "dataset_uri": resolved_dataset_uri,
        "dataset_name": dataset_name,
        "dataset_uniq": _short_bisque_id(resolved_dataset_uri),
        "dataset_client_view_url": dataset_client_view_url,
        "members": len(normalized_uris),
        "resource_uris": normalized_uris,
        "input_errors": input_errors,
        "message": message,
        "ui_artifacts": _build_bisque_dataset_ui_artifacts(
            title="Dataset created",
            dataset_name=dataset_name,
            dataset_uri=resolved_dataset_uri,
            resource_uris=normalized_uris,
        ),
    }


def _append_resources_to_bisque_dataset_with_session(
    *,
    bq: Any,
    bisque_root: str,
    dataset_uri: str,
    dataset_name: str | None,
    resource_uris: list[str],
    match_type: str | None = None,
) -> dict[str, Any]:
    normalized_dataset_uri = _normalize_bisque_resource_uri(dataset_uri, bisque_root)
    dataset_uniq = _short_bisque_id(normalized_dataset_uri)
    normalized_uris, input_errors = _normalize_bisque_dataset_resource_inputs(
        bq=bq,
        bisque_root=bisque_root,
        resource_uris=list(resource_uris or []),
    )
    if not normalized_uris:
        error = "resource_uris cannot be empty." if not input_errors else input_errors[0]
        return {"success": False, "error": error, "input_errors": input_errors}

    results: list[dict[str, Any]] = []
    verification_missing: list[str] = []
    try:
        dataset_payload = _session_fetchxml_safe(bq, normalized_dataset_uri, view="deep")
        dataset_document = _extract_bisque_dataset_document(dataset_payload)
        dataset_document, newly_added_uris = _append_members_to_dataset_document(
            dataset_xml=dataset_document,
            bisque_root=bisque_root,
            resource_uris=normalized_uris,
        )
        _session_postxml_safe(bq, normalized_dataset_uri, dataset_document, method="PUT")

        _verification_document, verified_members, settle_attempts = (
            _wait_for_bisque_dataset_membership_state(
                bq=bq,
                dataset_uri=normalized_dataset_uri,
                bisque_root=bisque_root,
                expected_present=set(normalized_uris),
            )
        )

        for normalized_uri in normalized_uris:
            resource_uniq = _short_bisque_id(normalized_uri)
            is_verified = normalized_uri in verified_members
            if not is_verified:
                verification_missing.append(normalized_uri)
            results.append(
                {
                    "resource_uri": normalized_uri,
                    "resource_uniq": resource_uniq,
                    "success": is_verified,
                    "verification_attempts": settle_attempts,
                    **(
                        {}
                        if is_verified
                        else {
                            "error": (
                                "BisQue dataset update completed, but the dataset still did not "
                                f"list member {normalized_uri} when reloaded."
                            )
                        }
                    ),
                    "already_present": normalized_uri not in newly_added_uris,
                }
            )
    except Exception as exc:
        failure_message = str(exc)
        for normalized_uri in normalized_uris:
            results.append(
                {
                    "resource_uri": normalized_uri,
                    "resource_uniq": _short_bisque_id(normalized_uri),
                    "success": False,
                    "error": failure_message,
                }
            )

    added = sum(1 for row in results if row.get("success") and not bool(row.get("already_present")))
    failed = len(results) - sum(1 for row in results if row.get("success"))
    dataset_links = _build_bisque_resource_links(normalized_dataset_uri, bisque_root)
    resolved_dataset_name = str(dataset_name or "").strip() or None
    message = (
        f"Added {added} resource(s) to BisQue dataset "
        f"'{resolved_dataset_name or dataset_uniq or 'dataset'}'."
    )
    error = None
    if failed > 0 or input_errors:
        first_failure = next(
            (
                str(row.get("error") or "").strip()
                for row in results
                if isinstance(row, dict) and not row.get("success")
            ),
            "",
        )
        error = first_failure or (
            f"Dataset membership verification failed for {len(verification_missing)} resource(s)."
            if verification_missing
            else (input_errors[0] if input_errors else None)
        )

    return {
        "success": failed == 0 and not input_errors,
        "action": "appended",
        "dataset_uri": str(dataset_links.get("resource_uri") or normalized_dataset_uri),
        "dataset_name": resolved_dataset_name,
        "dataset_uniq": dataset_uniq,
        "dataset_client_view_url": str(dataset_links.get("client_view_url") or "").strip() or None,
        "match_type": match_type,
        "added": added,
        "failed": failed + len(input_errors),
        "total_resources": len(normalized_uris),
        "resource_uris": normalized_uris,
        "results": results,
        "input_errors": input_errors,
        "error": error,
        "message": message,
        "ui_artifacts": _build_bisque_dataset_ui_artifacts(
            title="Dataset updated",
            dataset_name=resolved_dataset_name,
            dataset_uri=str(dataset_links.get("resource_uri") or normalized_dataset_uri),
            resource_uris=normalized_uris,
            added=added,
            failed=failed + len(input_errors),
        ),
    }


def _organize_bisque_resources_into_dataset_with_session(
    *,
    bq: Any,
    bisque_root: str,
    resource_uris: list[str],
    dataset_uri: str | None = None,
    dataset_name: str | None = None,
    create_dataset_if_missing: bool = False,
    tags: dict[str, str] | list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    requested_dataset_uri = str(dataset_uri or "").strip() or None
    requested_dataset_name = str(dataset_name or "").strip() or None
    if not requested_dataset_uri and not requested_dataset_name:
        return {
            "success": False,
            "error": "dataset_uri or dataset_name is required to organize resources.",
        }

    resolved = _resolve_bisque_dataset_target_with_session(
        bq=bq,
        bisque_root=bisque_root,
        dataset_uri=requested_dataset_uri,
        dataset_name=requested_dataset_name,
    )
    if resolved is not None:
        if str(resolved.get("status") or "").strip().lower() == "ambiguous":
            return {
                "success": False,
                "action": "ambiguous",
                "dataset_uri": requested_dataset_uri,
                "dataset_name": requested_dataset_name,
                "candidate_names": list(resolved.get("candidate_names") or []),
                "candidate_datasets": list(resolved.get("candidate_datasets") or []),
                "error": str(resolved.get("error") or "").strip() or "Dataset target is ambiguous.",
                "message": str(resolved.get("message") or "").strip()
                or "Dataset target is ambiguous.",
            }
        return _append_resources_to_bisque_dataset_with_session(
            bq=bq,
            bisque_root=bisque_root,
            dataset_uri=str(resolved.get("dataset_uri") or requested_dataset_uri or ""),
            dataset_name=str(resolved.get("dataset_name") or requested_dataset_name or "").strip()
            or None,
            resource_uris=list(resource_uris or []),
            match_type=str(resolved.get("match_type") or "").strip() or None,
        )

    if create_dataset_if_missing:
        if not requested_dataset_name:
            return {
                "success": False,
                "error": "dataset_name is required when create_dataset_if_missing=true.",
            }
        return _create_bisque_dataset_with_session(
            bq=bq,
            bisque_root=bisque_root,
            name=requested_dataset_name,
            resource_uris=list(resource_uris or []),
            tags=tags,
        )

    return {
        "success": False,
        "action": "not_found",
        "dataset_uri": requested_dataset_uri,
        "dataset_name": requested_dataset_name,
        "error": (
            f"Dataset '{requested_dataset_name}' was not found in BisQue. "
            "Set create_dataset_if_missing=true to create it."
            if requested_dataset_name
            else "Dataset could not be resolved in BisQue."
        ),
        "message": (
            f"Dataset '{requested_dataset_name}' could not be resolved."
            if requested_dataset_name
            else "Dataset could not be resolved."
        ),
    }


def bisque_create_dataset(
    name: str,
    resource_uris: list[str],
    tags: dict[str, str] | list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """
    Create a BisQue dataset from a list of resource URIs.

    Args:
        name: Dataset name
        resource_uris: List of BisQue resource URIs or resource_uniq values
        tags: Optional metadata tags to add to the dataset
    """
    try:
        from bqapi.comm import BQSession
    except ImportError:
        return {"success": False, "error": "bqapi package not installed"}

    try:
        if not resource_uris:
            return {"success": False, "error": "resource_uris cannot be empty"}

        bq = BQSession()
        root, _, _ = _init_bisque_session_with_runtime_auth(bq=bq)
        return _create_bisque_dataset_with_session(
            bq=bq,
            bisque_root=root,
            name=name,
            resource_uris=resource_uris,
            tags=tags,
        )
    except Exception as e:
        logger.error(f"Failed to create dataset: {str(e)}")
        return {"success": False, "error": str(e)}


def bisque_add_to_dataset(
    resource_uris: list[str],
    dataset_uri: str | None = None,
    dataset_name: str | None = None,
    create_dataset_if_missing: bool = False,
    tags: dict[str, str] | list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """
    Add existing resources to a BisQue dataset, optionally creating the dataset.

    Args:
        resource_uris: List of BisQue resource URIs or resource_uniq values
        dataset_uri: Optional existing dataset URI or resource_uniq
        dataset_name: Optional existing dataset name to resolve first
        create_dataset_if_missing: Whether to create the named dataset if not found
        tags: Optional metadata tags to add when creating a new dataset
    """
    try:
        from bqapi.comm import BQSession
    except ImportError:
        return {"success": False, "error": "bqapi package not installed"}

    try:
        if not resource_uris:
            return {"success": False, "error": "resource_uris cannot be empty"}

        bq = BQSession()
        root, _, _ = _init_bisque_session_with_runtime_auth(bq=bq)
        return _organize_bisque_resources_into_dataset_with_session(
            bq=bq,
            bisque_root=root,
            resource_uris=resource_uris,
            dataset_uri=dataset_uri,
            dataset_name=dataset_name,
            create_dataset_if_missing=create_dataset_if_missing,
            tags=tags,
        )
    except Exception as e:
        logger.error(f"Failed to add resources to dataset: {str(e)}")
        return {"success": False, "error": str(e)}


def bisque_add_gobjects(
    resource_uri: str,
    gobjects: list[dict[str, Any]],
    replace_existing: bool = False,
) -> dict[str, Any]:
    """
    Add graphical objects (gobjects) to a BisQue resource.

    Args:
        resource_uri: Resource URI to annotate
        gobjects: List of gobject specs: {type, name?, value?, vertices, tags?}
        replace_existing: Whether to remove existing gobjects before adding
    """
    try:
        from bqapi.comm import BQSession
    except ImportError:
        return {"success": False, "error": "bqapi package not installed"}

    try:
        bq = BQSession()
        root, _, _ = _init_bisque_session_with_runtime_auth(bq=bq)

        normalized = _normalize_bisque_resource_uri(resource_uri, root)
        resource = bq.load(normalized, view="short")
        if resource is None:
            return {
                "success": False,
                "error": f"Resource not found or failed to load: {resource_uri}",
            }
        request_xml, added = _build_bisque_gobject_request(gobjects)
        if not added:
            return {
                "success": False,
                "error": "No valid gobjects were provided.",
                "resource_uri": normalized,
            }

        _session_postxml_safe(
            bq,
            f"{normalized.rstrip('/')}/gobject",
            request_xml,
            method="PUT" if replace_existing else "POST",
            view="deep",
        )

        def _count_verified_gobjects(xml: etree._Element | None) -> int:
            if xml is None:
                return 0
            verified_count = 0
            for row in added:
                gtype = str(row.get("type") or "").strip().lower()
                xml_tag = str(row.get("xml_tag") or "").strip().lower()
                name = str(row.get("name") or "").strip()
                vertex_count = int(row.get("vertices") or 0)
                found = False
                for match in xml.xpath(
                    "./gobject | ./point | ./label | ./polyline | ./polygon | ./circle | ./ellipse | ./rectangle | ./square"
                ):
                    local_name = etree.QName(match.tag).localname.lower()
                    match_type = str(match.get("type") or "").strip().lower()
                    match_name = str(match.get("name") or "").strip()
                    if local_name not in {xml_tag, gtype, "gobject"}:
                        continue
                    if local_name == "gobject" and gtype and match_type != gtype:
                        continue
                    if name and match_name != name:
                        continue
                    if len(match.findall("./vertex")) >= vertex_count:
                        found = True
                        break
                if found:
                    verified_count += 1
            return verified_count

        verification_xml, verified_state, verification_attempts = _wait_for_bisque_resource_state(
            bq=bq,
            resource_uri=normalized,
            predicate=lambda xml, _exc: _count_verified_gobjects(xml) == len(added),
            cache="false",
        )
        verified = _count_verified_gobjects(verification_xml)

        counts = Counter([a["xml_tag"] for a in added])
        ui_artifacts = [
            {
                "type": "metrics",
                "title": "GObjects added",
                "payload": {
                    "total": len(added),
                    "verified": verified,
                    "verification_attempts": verification_attempts,
                    **counts,
                },
            },
            {
                "type": "table",
                "title": "GObject summary",
                "payload": added,
            },
        ]

        client_view_url = _bisque_user_facing_resource_url(normalized, root)
        return {
            "success": len(added) > 0 and verified_state and verified == len(added),
            "resource_uri": normalized,
            "client_view_url": client_view_url,
            "added": added,
            "verified": verified,
            "verification_attempts": verification_attempts,
            "ui_artifacts": ui_artifacts,
        }

    except Exception as e:
        logger.error(f"Failed to add gobjects: {str(e)}")
        return {"success": False, "error": str(e)}


def bisque_advanced_search(
    resource_type: str = "image",
    tag_query: str | None = None,
    tag_filters: dict[str, Any] | None = None,
    metadata_filters: dict[str, Any] | None = None,
    text: str | None = None,
    owner: str | None = None,
    permission: str | None = None,
    created_after: str | None = None,
    created_before: str | None = None,
    modified_after: str | None = None,
    modified_before: str | None = None,
    order_by: str | None = None,
    order: str = "desc",
    limit: int = 50,
    offset: int = 0,
    view: str = "short",
) -> dict[str, Any]:
    """
    Advanced BisQue query with structured filters.
    """
    try:
        from bqapi.comm import BQSession
    except ImportError:
        return {"success": False, "error": "bqapi package not installed"}

    try:
        bq = BQSession()
        _root, _, _ = _init_bisque_session_with_runtime_auth(bq=bq)

        query_params: dict[str, Any] = {
            "limit": limit,
            "offset": offset,
            "view": view,
        }

        attr_filters: dict[str, Any] = {}
        if owner:
            attr_filters["owner"] = owner
        if created_after or created_before:
            created_range = []
            if created_after:
                created_range.append(f">={created_after}")
            if created_before:
                created_range.append(f"<={created_before}")
            attr_filters["created"] = created_range
        if modified_after or modified_before:
            modified_range = []
            if modified_after:
                modified_range.append(f">={modified_after}")
            if modified_before:
                modified_range.append(f"<={modified_before}")
            attr_filters["ts"] = modified_range

        expanded_filters = _expand_bisque_tag_filters(tag_filters)
        expanded_meta = _expand_bisque_tag_filters(metadata_filters)
        if expanded_meta:
            for key, value in expanded_meta.items():
                expanded_filters = expanded_filters or {}
                expanded_filters[key] = _merge_filter_value(expanded_filters.get(key), value)

        normalized_tag_query = _rewrite_tag_query_aliases(tag_query)
        combined_tag_query = _build_tag_query_compat(
            bq,
            tag_query=normalized_tag_query,
            tag_filters=expanded_filters,
            attr_filters=attr_filters,
            text=text,
        )
        if permission:
            # Treat permission as a tag filter unless server supports @perm.
            perm_query = bq.build_tag_query(tag_filters={"permission": permission})
            combined_tag_query = (
                f"{combined_tag_query} AND {perm_query}"
                if combined_tag_query and perm_query
                else combined_tag_query or perm_query
            )

        if combined_tag_query:
            query_params["tag_query"] = combined_tag_query
        if order_by:
            order_key = order_by
            if order_key in {"name", "created", "ts", "owner"}:
                order_key = f"@{order_key}"
            query_params["tag_order"] = f"{order_key}:{order}"

        results = bq.query(resource_type, **query_params)
        resources = []
        for resource in results:
            resources.append(
                {
                    "uri": getattr(resource, "uri", None),
                    "name": getattr(resource, "name", None),
                    "owner": getattr(resource, "owner", None),
                    "created": str(getattr(resource, "ts", None)),
                }
            )

        ui_artifacts = [
            {
                "type": "metrics",
                "title": "Query results",
                "payload": {"count": len(resources)},
            },
            {
                "type": "table",
                "title": "Query matches",
                "payload": resources[:200],
            },
        ]

        return {
            "success": True,
            "count": len(resources),
            "resources": resources,
            "query": {
                "resource_type": resource_type,
                "tag_query": query_params.get("tag_query"),
                "limit": limit,
                "offset": offset,
                "view": view,
                "order_by": order_by,
                "order": order,
            },
            "ui_artifacts": ui_artifacts,
        }

    except Exception as e:
        logger.error(f"Advanced BisQue search failed: {str(e)}")
        return {"success": False, "error": str(e)}


_PAPER_SEGMENTATION_STATS_ALIASES = {
    "paper_segmentation_stats",
    "paper-ready-segmentation-stats",
    "segmentation_paper_stats",
    "segmentation_publication_stats",
}


def _coerce_numeric_list(values: Any) -> list[float]:
    if not isinstance(values, list):
        return []
    out: list[float] = []
    for value in values:
        try:
            x = float(value)
        except Exception:
            continue
        if np.isfinite(x):
            out.append(float(x))
    return out


def _run_paper_segmentation_stats(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = payload if isinstance(payload, dict) else {}
    metrics_payload = data.get("metrics")
    metrics_input: dict[str, list[float]] = {}
    if isinstance(metrics_payload, dict):
        for key, values in metrics_payload.items():
            coerced = _coerce_numeric_list(values)
            if coerced:
                metrics_input[str(key)] = coerced
    alias_inputs = {
        "dice": data.get("dice_scores") or data.get("dice"),
        "iou": data.get("iou_scores") or data.get("iou"),
        "precision": data.get("precision_scores") or data.get("precision"),
        "recall": data.get("recall_scores") or data.get("recall"),
        "coverage_percent": data.get("coverage_percent"),
        "object_count": data.get("object_count"),
    }
    for metric_name, raw_values in alias_inputs.items():
        coerced = _coerce_numeric_list(raw_values)
        if coerced and metric_name not in metrics_input:
            metrics_input[metric_name] = coerced

    metric_summaries: dict[str, Any] = {}
    measurements: list[dict[str, Any]] = []
    limitations: list[str] = []

    for metric_name, values in metrics_input.items():
        summary_result = _domain_stats_run_curated_tool(
            tool_name="summary_statistics",
            payload={"values": values},
        )
        if not summary_result.get("success"):
            metric_summaries[metric_name] = {"success": False, "error": summary_result.get("error")}
            continue
        summary = (
            summary_result.get("result") if isinstance(summary_result.get("result"), dict) else {}
        )
        summary["n"] = len(values)
        metric_summaries[metric_name] = summary
        mean_value = summary.get("mean")
        if isinstance(mean_value, (int, float)):
            measurements.append(
                {
                    "name": f"{metric_name}_mean",
                    "value": float(mean_value),
                    "unit": "score",
                }
            )
        if len(values) < 5:
            limitations.append(
                f"{metric_name} has only {len(values)} observations; inferential claims are low power."
            )

    comparisons: list[dict[str, Any]] = []
    compare_rows = data.get("comparisons")
    if isinstance(compare_rows, list):
        for row in compare_rows:
            if not isinstance(row, dict):
                continue
            group_a = _coerce_numeric_list(row.get("group_a"))
            group_b = _coerce_numeric_list(row.get("group_b"))
            if len(group_a) < 2 or len(group_b) < 2:
                continue
            metric_name = str(row.get("metric_name") or row.get("metric") or "metric")
            alpha = row.get("alpha", data.get("alpha", 0.05))
            comp_result = _domain_stats_run_curated_tool(
                tool_name="compare_two_groups",
                payload={
                    "group_a": group_a,
                    "group_b": group_b,
                    "metric_name": metric_name,
                    "alpha": alpha,
                },
            )
            if comp_result.get("success"):
                result_obj = comp_result.get("result")
                if isinstance(result_obj, dict):
                    comparisons.append(result_obj)

    if not comparisons:
        default_group_a = _coerce_numeric_list(data.get("group_a"))
        default_group_b = _coerce_numeric_list(data.get("group_b"))
        if len(default_group_a) >= 2 and len(default_group_b) >= 2:
            metric_name = str(data.get("metric_name") or "metric")
            alpha = data.get("alpha", 0.05)
            comp_result = _domain_stats_run_curated_tool(
                tool_name="compare_two_groups",
                payload={
                    "group_a": default_group_a,
                    "group_b": default_group_b,
                    "metric_name": metric_name,
                    "alpha": alpha,
                },
            )
            if comp_result.get("success"):
                result_obj = comp_result.get("result")
                if isinstance(result_obj, dict):
                    comparisons.append(result_obj)

    publication_readiness: dict[str, Any] = {
        "ready_for_methods_draft": bool(metric_summaries),
        "ready_for_inferential_claims": bool(comparisons),
        "notes": [],
    }
    dice_summary = metric_summaries.get("dice")
    if isinstance(dice_summary, dict) and isinstance(dice_summary.get("mean"), (int, float)):
        mean_dice = float(dice_summary["mean"])
        publication_readiness["mean_dice"] = mean_dice
        if mean_dice < 0.7:
            publication_readiness["notes"].append(
                "Mean Dice is below 0.7; segmentation quality may be insufficient for publication claims."
            )
    if not comparisons:
        publication_readiness["notes"].append(
            "No two-group inferential comparison was provided; avoid significance claims."
        )

    if not metric_summaries:
        return {
            "success": False,
            "error": (
                "paper_segmentation_stats requires segmentation metric arrays "
                "(e.g., dice_scores, iou_scores, precision_scores, recall_scores) "
                "or payload.metrics with numeric lists."
            ),
        }

    return {
        "success": True,
        "tool_name": "paper_segmentation_stats",
        "result": {
            "metric_summaries": metric_summaries,
            "group_comparisons": comparisons,
            "publication_readiness": publication_readiness,
        },
        "measurements": measurements,
        "limitations": sorted(set(limitations)),
        "ui_artifacts": [
            {
                "type": "table",
                "title": "Paper segmentation metric summaries",
                "payload": [
                    {"metric": k, **v} for k, v in metric_summaries.items() if isinstance(v, dict)
                ],
            },
            {
                "type": "table",
                "title": "Paper segmentation two-group comparisons",
                "payload": comparisons[:100],
            },
            {
                "type": "write",
                "title": "Publication readiness notes",
                "payload": publication_readiness,
            },
        ],
    }


def stats_list_curated_tools() -> dict[str, Any]:
    base = _domain_stats_list_curated_tools()
    if not isinstance(base, dict):
        return {"success": False, "error": "Unexpected curated-tool response format."}
    if not base.get("success"):
        return base
    tools = base.get("tools")
    if not isinstance(tools, list):
        return base

    alias_name = "paper_segmentation_stats"
    if not any(isinstance(item, dict) and item.get("name") == alias_name for item in tools):
        tools.append(
            {
                "name": alias_name,
                "description": (
                    "Paper-focused segmentation summary preset. Accepts segmentation metric arrays "
                    "(dice/iou/precision/recall), optional coverage/object counts, and optional two-group "
                    "comparisons for inferential reporting."
                ),
                "required_payload": "dice_scores/iou_scores/... or payload.metrics; optional group_a/group_b.",
            }
        )
    base["tools"] = tools
    base["count"] = len(tools)
    base["ui_artifacts"] = [
        {
            "type": "table",
            "title": "Curated statistical tools",
            "payload": tools,
        }
    ]
    return base


def stats_run_curated_tool(tool_name: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    normalized = str(tool_name or "").strip().lower()
    if normalized in _PAPER_SEGMENTATION_STATS_ALIASES:
        return _run_paper_segmentation_stats(payload=payload)
    return _domain_stats_run_curated_tool(tool_name=tool_name, payload=payload)


# Tool schema dictionaries now live under src/tooling/domains/.


def codegen_python_plan(
    task_summary: str,
    job_id: str | None = None,
    inputs: list[dict[str, Any]] | None = None,
    constraints: dict[str, Any] | None = None,
    attempt_index: int = 1,
    previous_failure: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate/repair and persist a Python code job package."""
    settings = get_settings()
    if not bool(getattr(settings, "code_execution_enabled", False)):
        return {
            "success": False,
            "error": "Code execution tools are disabled. Set CODE_EXECUTION_ENABLED=true.",
            "error_class": "feature_disabled",
        }
    max_repair_cycles = max(1, int(getattr(settings, "code_execution_max_repair_cycles", 5)))
    normalized_attempt = int(max(1, int(attempt_index or 1)))
    if normalized_attempt > (max_repair_cycles + 1):
        return {
            "success": False,
            "error": (
                f"attempt_index={normalized_attempt} exceeds max repair cycles "
                f"({max_repair_cycles})."
            ),
            "error_class": "repair_budget_exhausted",
            "max_repair_cycles": max_repair_cycles,
        }
    try:
        result = prepare_python_job(
            task_summary=str(task_summary or "").strip(),
            job_id=job_id,
            inputs=inputs if isinstance(inputs, list) else None,
            constraints=constraints if isinstance(constraints, dict) else None,
            attempt_index=normalized_attempt,
            previous_failure=previous_failure if isinstance(previous_failure, dict) else None,
        )
        result.setdefault("success", True)
        result["message"] = (
            f"Prepared Python job {result.get('job_id')} (attempt {result.get('attempt_index')})."
        )
        return result
    except Exception as exc:  # noqa: BLE001
        logger.exception("codegen_python_plan failed")
        return {
            "success": False,
            "error": str(exc),
            "error_class": "codegen_failed",
        }


def _build_code_execution_plan(
    *,
    job_id: str,
    execution_backend: str,
    timeout_seconds: int | None,
    cpu_limit: float | str | None,
    memory_mb: int | None,
    auto_repair: bool,
    max_repair_cycles: int | None,
) -> tuple[Any, Any]:
    from src.orchestration.models import ToolStep, WorkflowPlan

    step = ToolStep(
        id=f"codeexec-{uuid4().hex[:8]}",
        tool_name="_execute_python_job_once",
        arguments={
            "job_id": job_id,
            "execution_backend": execution_backend,
            "timeout_seconds": timeout_seconds,
            "cpu_limit": cpu_limit,
            "memory_mb": memory_mb,
            "auto_repair": bool(auto_repair),
            "max_repair_cycles": max_repair_cycles,
        },
        description="Durable Python sandbox execution",
        timeout_seconds=int(timeout_seconds or 3600),
        retries=0,
    )
    plan = WorkflowPlan(goal=f"Execute Python sandbox job {job_id}", steps=[step])
    return plan, step


def _wait_for_run_completion(
    *,
    store: Any,
    run_id: str,
    wait_timeout_seconds: int,
    poll_interval_seconds: float,
) -> tuple[str, dict[str, Any] | None]:
    started = time.time()
    while time.time() - started <= wait_timeout_seconds:
        run = store.get_run(run_id)
        if run is None:
            return "missing", None
        raw_status = getattr(run, "status", "")
        status_value = str(getattr(raw_status, "value", raw_status) or "").strip().lower()
        if status_value in {"succeeded", "failed", "canceled"}:
            step_event = (
                store.get_latest_event(run_id, "step_succeeded")
                or store.get_latest_event(run_id, "step_completed")
                or store.get_latest_event(run_id, "step_failed")
                or store.get_latest_event(run_id, "step_failed_attempt")
            )
            payload = step_event.get("payload") if isinstance(step_event, dict) else None
            result = payload.get("result") if isinstance(payload, dict) else None
            return status_value, result if isinstance(result, dict) else None
        time.sleep(max(0.1, float(poll_interval_seconds)))
    return "timeout_waiting", None


def execute_python_job(
    job_id: str,
    execution_backend: str = "docker",
    durable_execution: bool | None = None,
    wait_for_completion: bool = True,
    wait_timeout_seconds: int | None = None,
    timeout_seconds: int | None = None,
    cpu_limit: float | str | None = None,
    memory_mb: int | None = None,
    auto_repair: bool = True,
    max_repair_cycles: int | None = None,
) -> dict[str, Any]:
    """Execute a generated Python job in Docker or submit durable execution."""
    settings = get_settings()
    if not bool(getattr(settings, "code_execution_enabled", False)):
        return {
            "success": False,
            "error": "Code execution tools are disabled. Set CODE_EXECUTION_ENABLED=true.",
            "error_class": "feature_disabled",
        }
    job_token = str(job_id or "").strip()
    if not job_token:
        return {"success": False, "error": "job_id is required", "error_class": "invalid_request"}
    try:
        load_python_job_spec(job_token)
    except Exception as exc:  # noqa: BLE001
        return {
            "success": False,
            "error": str(exc),
            "error_class": "invalid_job_id",
            "job_id": job_token,
        }

    backend = (
        str(execution_backend or getattr(settings, "code_execution_default_backend", "docker"))
        .strip()
        .lower()
    )
    effective_durable = (
        bool(getattr(settings, "code_execution_durable_default", True))
        if durable_execution is None
        else bool(durable_execution)
    )
    timeout_default = int(getattr(settings, "code_execution_default_timeout_seconds", 900))
    timeout_cap = int(getattr(settings, "code_execution_max_timeout_seconds", 3600))
    effective_timeout_seconds = int(timeout_seconds or timeout_default)
    effective_timeout_seconds = max(1, min(effective_timeout_seconds, timeout_cap))
    wait_timeout = int(wait_timeout_seconds or (effective_timeout_seconds + 120))
    poll_every = float(getattr(settings, "code_execution_poll_interval_seconds", 1.0) or 1.0)

    # Non-durable path (direct execution inside current process).
    if not effective_durable:
        result = execute_python_job_once(
            job_id=job_token,
            execution_backend=backend,
            timeout_seconds=effective_timeout_seconds,
            cpu_limit=cpu_limit,
            memory_mb=memory_mb,
            auto_repair=bool(auto_repair),
            max_repair_cycles=max_repair_cycles,
        )
        result["durable_execution"] = False
        return result

    # Durable execution path: submit a child workflow run that survives client disconnects.
    from src.orchestration.models import WorkflowRun
    from src.orchestration.store import RunStore

    store = RunStore(settings.run_store_path)
    plan, _step = _build_code_execution_plan(
        job_id=job_token,
        execution_backend=backend,
        timeout_seconds=effective_timeout_seconds,
        cpu_limit=cpu_limit,
        memory_mb=memory_mb,
        auto_repair=bool(auto_repair),
        max_repair_cycles=max_repair_cycles,
    )
    child_run = WorkflowRun.new(goal=plan.goal, plan=plan)
    store.create_run(child_run)
    store.set_run_metadata(child_run.run_id, user_id=None, conversation_id=None)
    store.append_event(
        child_run.run_id,
        "code_execution_submitted",
        {
            "job_id": job_token,
            "execution_backend": backend,
            "timeout_seconds": effective_timeout_seconds,
        },
    )

    submitted_via = "inline-thread"
    from src.orchestration.executor import PlanExecutor

    def _run_plan() -> None:
        executor = PlanExecutor(store)
        executor.execute(child_run.run_id, plan)

    thread = threading.Thread(
        target=_run_plan,
        name=f"codeexec-run-{child_run.run_id[:8]}",
        daemon=True,
    )
    thread.start()
    store.append_event(
        child_run.run_id,
        "code_execution_enqueued",
        {"queue": "inline-thread", "job_id": job_token},
    )

    if not wait_for_completion:
        return {
            "success": True,
            "job_id": job_token,
            "durable_execution": True,
            "wait_for_completion": False,
            "durable_run_id": child_run.run_id,
            "durable_status": "submitted",
            "submitted_via": submitted_via,
            "message": (
                f"Submitted durable Python execution run {child_run.run_id}. "
                "Use run events/result endpoints to monitor completion."
            ),
        }

    status, step_result = _wait_for_run_completion(
        store=store,
        run_id=child_run.run_id,
        wait_timeout_seconds=wait_timeout,
        poll_interval_seconds=poll_every,
    )
    if status == "succeeded" and isinstance(step_result, dict):
        enriched = dict(step_result)
        enriched["job_id"] = job_token
        enriched["durable_execution"] = True
        enriched["durable_run_id"] = child_run.run_id
        enriched["durable_status"] = status
        enriched["submitted_via"] = submitted_via
        return enriched

    return {
        "success": False,
        "job_id": job_token,
        "durable_execution": True,
        "durable_run_id": child_run.run_id,
        "durable_status": status,
        "submitted_via": submitted_via,
        "error_class": "durable_execution_incomplete",
        "error_message": (
            "Durable execution did not reach succeeded state within wait timeout."
            if status == "timeout_waiting"
            else f"Durable execution ended with status={status}."
        ),
        "repair_hint": (
            "Retry execute_python_job with wait_for_completion=false and poll /v1/runs/{run_id} "
            "or increase wait_timeout_seconds."
        ),
    }


def _execute_python_job_once(
    job_id: str,
    execution_backend: str = "docker",
    timeout_seconds: int | None = None,
    cpu_limit: float | str | None = None,
    memory_mb: int | None = None,
    auto_repair: bool = True,
    max_repair_cycles: int | None = None,
) -> dict[str, Any]:
    """Internal deterministic execution entrypoint used by durable workflow plans."""
    return execute_python_job_once(
        job_id=job_id,
        execution_backend=execution_backend,
        timeout_seconds=timeout_seconds,
        cpu_limit=cpu_limit,
        memory_mb=memory_mb,
        auto_repair=bool(auto_repair),
        max_repair_cycles=max_repair_cycles,
    )


# Map function names to actual functions
AVAILABLE_TOOLS: dict[str, Callable] = {
    "upload_to_bisque": upload_to_bisque,
    "bisque_ping": bisque_ping,
    "bisque_download_resource": bisque_download_resource,
    "bisque_find_assets": bisque_find_assets,
    "search_bisque_resources": search_bisque_resources,
    "load_bisque_resource": load_bisque_resource,
    "delete_bisque_resource": delete_bisque_resource,
    "add_tags_to_resource": add_tags_to_resource,
    "bisque_fetch_xml": bisque_fetch_xml,
    "bisque_download_dataset": bisque_download_dataset,
    "bisque_create_dataset": bisque_create_dataset,
    "bisque_add_to_dataset": bisque_add_to_dataset,
    "bisque_add_gobjects": bisque_add_gobjects,
    "bisque_advanced_search": bisque_advanced_search,
    "bioio_load_image": bioio_load_image,
    "segment_image_megaseg": segment_image_megaseg,
    "segment_image_sam2": segment_image_sam2,
    "sam2_prompt_image": sam2_prompt_image,
    "estimate_depth_pro": estimate_depth_pro,
    "segment_image_sam3": segment_image_sam3,
    "evaluate_segmentation_masks": evaluate_segmentation_masks,
    "segment_evaluate_batch": segment_evaluate_batch,
    "segment_video_sam2": segment_video_sam2,
    "run_bisque_module": run_bisque_module,
    "yolo_list_finetuned_models": yolo_list_finetuned_models,
    "yolo_detect": yolo_detect,
    "analyze_prediction_stability": analyze_prediction_stability_tool,
    "score_spectral_instability": score_spectral_instability_tool,
    "yolo_finetune_detect": yolo_finetune_detect,
    "analyze_csv": analyze_csv,
    "quantify_objects": quantify_objects,
    "plot_quantified_detections": plot_quantified_detections,
    "quantify_segmentation_masks": quantify_segmentation_masks,
    "compare_conditions": compare_conditions,
    "stats_list_curated_tools": stats_list_curated_tools,
    "stats_run_curated_tool": stats_run_curated_tool,
    "repro_report": repro_report,
    "codegen_python_plan": codegen_python_plan,
    "numpy_calculator": numpy_calculator,
    "execute_python_job": execute_python_job,
    "_execute_python_job_once": _execute_python_job_once,
    "structure_report": structure_report,
    "compare_structures": compare_structures,
    "propose_reactive_sites": propose_reactive_sites,
    "formula_balance_check": formula_balance_check,
}


def _clean_inferred_bisque_dataset_target(value: str) -> str:
    token = str(value or "").strip().strip("\"'`")
    token = re.sub(r"\s+(?:on|in)\s+bisque$", "", token, flags=re.IGNORECASE)
    token = token.strip(" \t\r\n.,;:!?")
    return token


def _infer_bisque_catalog_search_from_text(user_text: str | None) -> dict[str, Any] | None:
    text = str(user_text or "").strip()
    lowered = text.lower()
    if not text or not any(
        cue in lowered
        for cue in (
            "bisque",
            "dataset",
            "datasets",
            "resource",
            "resources",
            "image",
            "images",
            "file",
            "files",
            "table",
            "tables",
            "hdf5",
            "h5",
            "dream3d",
        )
    ):
        return None

    resource_type: str | None = None
    if re.search(r"\bdatasets?\b", lowered):
        resource_type = "dataset"
    elif re.search(r"\b(?:table|tables|hdf5|h5|dream3d)\b", lowered):
        resource_type = "table"
    elif re.search(r"\b(?:image|images|png|jpg|jpeg|tiff?|ome[-\s]?tiff?)\b", lowered):
        resource_type = "image"

    exact_name: str | None = None
    for pattern in (
        r"\b(?:named|called)\s+[\"“']([^\"”']{2,160})[\"”']",
        r"[\"“']([^\"”']{2,160})[\"”']",
    ):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        candidate = _clean_inferred_bisque_dataset_target(match.group(1))
        if candidate and candidate.lower() not in {
            "bisque",
            "dataset",
            "datasets",
            "resource",
            "resources",
        }:
            exact_name = candidate
            break

    inferred: dict[str, Any] = {}
    if resource_type:
        inferred["resource_type"] = resource_type
    if exact_name:
        inferred["text"] = exact_name
        inferred["tag_filters"] = {"name": exact_name}
    return inferred or None


def _infer_bisque_dataset_target_from_text(user_text: str | None) -> dict[str, Any] | None:
    text = str(user_text or "").strip()
    lowered = text.lower()
    if not text or not any(
        cue in lowered
        for cue in (
            "upload",
            "save",
            "store",
            "put ",
            "add ",
            "dataset",
            "organize",
            "aggregate",
            "group ",
            "collect ",
        )
    ):
        return None

    pattern_specs = [
        (
            r"\b(?:create|make)\s+(?:a\s+)?new\s+dataset(?:\s+(?:named|called))?\s+[\"“']?([^\"”'\n\r.!?]+)",
            True,
        ),
        (
            r"\b(?:dataset|collection)\s+(?:named|called)\s+[\"“']([^\"”']{2,160})[\"”']",
            False,
        ),
        (
            r"\b(?:upload|save|store|put|add|organize|group|aggregate)\b.{0,40}\b(?:to|into|in)\s+(?:the\s+)?(?:dataset|collection)\s+[\"“']?([^\"”'\n\r.!?]+)",
            False,
        ),
    ]
    for pattern, implies_create in pattern_specs:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        candidate = _clean_inferred_bisque_dataset_target(match.group(1))
        if not candidate or candidate.lower() in {"bisque", "dataset"}:
            continue
        create_if_missing = bool(
            implies_create
            or "new dataset" in lowered
            or "if it doesn't exist" in lowered
            or "if it does not exist" in lowered
            or "create it" in lowered
        )
        if _looks_like_bisque_resource(candidate):
            return {
                "dataset_uri": candidate,
                "create_dataset_if_missing": create_if_missing,
            }
        return {
            "dataset_name": candidate,
            "create_dataset_if_missing": create_if_missing,
        }
    return None


_SAM3_SEGMENTATION_GENERIC_REFERENTS = {
    "all",
    "all objects",
    "everything",
    "image",
    "picture",
    "photo",
    "frame",
    "scan",
    "volume",
    "scene",
    "file",
    "this",
    "that",
    "it",
    "this image",
    "that image",
    "the image",
    "the whole image",
    "the full image",
    "the entire image",
    "this picture",
    "that picture",
    "this photo",
    "that photo",
    "the frame",
    "this frame",
    "that frame",
    "the scene",
    "this scene",
    "that scene",
}


def _clean_inferred_sam3_concept_prompt(value: str | None) -> str | None:
    candidate = re.sub(r"\s+", " ", str(value or "").strip())
    if not candidate:
        return None

    candidate = re.sub(r"^[\"'“”‘’\s]+|[\"'“”‘’\s]+$", "", candidate)
    candidate = re.sub(
        r"\s+(?:in|on|from|within)\s+(?:the|this|that|my|our|uploaded|current|provided)?\s*"
        r"(?:image|photo|picture|frame|scan|volume|scene|file)\b.*$",
        "",
        candidate,
        flags=re.IGNORECASE,
    )
    candidate = re.sub(
        r"\s+(?:using|with|via)\s+sam3\b.*$",
        "",
        candidate,
        flags=re.IGNORECASE,
    )
    candidate = re.sub(
        r"\s+(?:please|for me)\b.*$",
        "",
        candidate,
        flags=re.IGNORECASE,
    )
    candidate = re.sub(
        r"\s+(?:and|then)\s+(?:tell|return|show|upload|quantif(?:y|ication)|measure|"
        r"report|compute|analy[sz]e|give|save|compare|summari[sz]e)\b.*$",
        "",
        candidate,
        flags=re.IGNORECASE,
    )
    candidate = re.sub(r"^[,;:\-\s]+|[,;:\-\s]+$", "", candidate).strip()
    if not candidate:
        return None

    lowered = candidate.lower()
    if lowered in _SAM3_SEGMENTATION_GENERIC_REFERENTS:
        return None
    if len(re.sub(r"[^a-z0-9]+", "", lowered)) < 2:
        return None
    return candidate


def _infer_sam3_concept_prompt_from_text(user_text: str | None) -> str | None:
    text = re.sub(r"\s+", " ", str(user_text or "").strip())
    if not text:
        return None

    lowered = text.lower()
    if not re.search(
        r"\b(segment|segmentation|mask|isolate|extract|outline|delineate|highlight|cut out)\b",
        lowered,
    ):
        return None

    pattern_specs = [
        r"\b(?:segment|mask|isolate|extract|outline|delineate|highlight|cut\s*out)\s+(?P<target>.+)",
        r"\b(?:find|get|show)\s+(?:me\s+)?(?:the\s+)?mask\s+(?:for|of)\s+(?P<target>.+)",
        r"\b(?:run|use)\s+sam3\s+(?:to\s+)?(?:segment|mask)\s+(?P<target>.+)",
    ]
    for pattern in pattern_specs:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        cleaned = _clean_inferred_sam3_concept_prompt(match.group("target"))
        if cleaned:
            return cleaned
    return None


def execute_tool_call(
    tool_name: str,
    arguments: str | dict,
    uploaded_files: list = None,
    user_text: str | None = None,
    latest_result_refs: dict[str, Any] | None = None,
    selection_context: dict[str, Any] | None = None,
) -> str:
    """
    Execute a tool call from the LLM.

    Args:
        tool_name: Name of the tool to execute
        arguments: JSON string or dict of arguments
        uploaded_files: Optional list of uploaded files from the current request/session

    Returns:
        JSON string result
    """
    try:
        # Parse arguments if string
        if isinstance(arguments, str):
            args = json.loads(arguments)
        else:
            args = arguments
        if args is None:
            args = {}
        if not isinstance(args, dict):
            return json.dumps(
                {
                    "success": False,
                    "error": f"Tool arguments must be a JSON object, got {type(args).__name__}",
                }
            )

        # Get the function
        if tool_name not in AVAILABLE_TOOLS:
            return json.dumps({"success": False, "error": f"Tool '{tool_name}' not found"})

        func = AVAILABLE_TOOLS[tool_name]
        normalized_selection_context = (
            dict(selection_context) if isinstance(selection_context, dict) else {}
        )
        selection_artifact_handles = (
            dict(normalized_selection_context.get("artifact_handles") or {})
            if isinstance(normalized_selection_context.get("artifact_handles"), dict)
            else {}
        )
        selection_resource_uris = [
            str(value or "").strip()
            for value in list(normalized_selection_context.get("resource_uris") or [])
            if str(value or "").strip()
        ]
        selection_dataset_uris = [
            str(value or "").strip()
            for value in list(normalized_selection_context.get("dataset_uris") or [])
            if str(value or "").strip()
        ]

        if tool_name in {"search_bisque_resources", "bisque_find_assets"}:
            inferred_catalog_target = _infer_bisque_catalog_search_from_text(user_text)
            if isinstance(inferred_catalog_target, dict):
                inferred_resource_type = str(
                    inferred_catalog_target.get("resource_type") or ""
                ).strip()
                current_resource_type = _normalize_bisque_resource_type(
                    str(args.get("resource_type") or "")
                )
                if inferred_resource_type and (
                    not str(args.get("resource_type") or "").strip()
                    or (current_resource_type == "image" and inferred_resource_type != "image")
                    or (inferred_resource_type == "dataset" and current_resource_type != "dataset")
                ):
                    args["resource_type"] = inferred_resource_type
                    logger.info(
                        "Inferred BisQue search resource_type for %s: %s",
                        tool_name,
                        inferred_resource_type,
                    )

                inferred_text = str(inferred_catalog_target.get("text") or "").strip()
                existing_text = args.get("text")
                if inferred_text and (
                    not existing_text or str(existing_text).strip() == str(user_text or "").strip()
                ):
                    args["text"] = inferred_text

                inferred_tag_filters = (
                    _coerce_bisque_filter_mapping(inferred_catalog_target.get("tag_filters")) or {}
                )
                # When we already inferred a precise quoted target into `text`,
                # let search_bisque_resources build one canonical query from that
                # text alone. Merging an extra inferred name tag-filter can
                # generate malformed BisQue tag_query expressions.
                if (
                    inferred_tag_filters
                    and not inferred_text
                    and not str(args.get("tag_query") or "").strip()
                ):
                    merged_tag_filters = (
                        _coerce_bisque_filter_mapping(args.get("tag_filters")) or {}
                    )
                    for key, value in inferred_tag_filters.items():
                        merged_tag_filters.setdefault(key, value)
                    args["tag_filters"] = merged_tag_filters
                    logger.info(
                        "Inferred BisQue catalog target for %s: resource_type=%s text=%s tag_filters=%s",
                        tool_name,
                        args.get("resource_type"),
                        args.get("text"),
                        args.get("tag_filters"),
                    )

        inferred_sam3_concept_prompt: str | None = None
        if tool_name in {"segment_image_sam2", "segment_image_sam3"}:
            depth_map_paths = args.pop("depth_map_paths", None)
            if (not args.get("file_paths")) and isinstance(depth_map_paths, list):
                normalized_depth_paths = [
                    str(path).strip() for path in depth_map_paths if str(path or "").strip()
                ]
                if normalized_depth_paths:
                    args["file_paths"] = normalized_depth_paths
                    logger.info(
                        "Mapped %s depth_map_paths -> file_paths (%s item(s))",
                        tool_name,
                        len(normalized_depth_paths),
                    )
        if tool_name == "segment_image_sam3":
            has_explicit_prompting = bool(
                str(args.get("concept_prompt") or "").strip()
                or list(args.get("input_boxes") or [])
                or list(args.get("input_points") or [])
            )
            if not has_explicit_prompting:
                inferred_sam3_concept_prompt = _infer_sam3_concept_prompt_from_text(user_text)
                if inferred_sam3_concept_prompt:
                    args["concept_prompt"] = inferred_sam3_concept_prompt
                    logger.info(
                        "Inferred SAM3 concept prompt from user request: %s",
                        inferred_sam3_concept_prompt,
                    )

        settings = get_settings()
        force_tool_visualizations = bool(getattr(settings, "ui_force_tool_visualizations", True))
        if force_tool_visualizations and tool_name in {
            "segment_image_megaseg",
            "segment_image_sam2",
            "segment_image_sam3",
            "estimate_depth_pro",
            "segment_evaluate_batch",
            "yolo_detect",
            "segment_video_sam2",
        }:
            args["save_visualizations"] = True

        def _looks_like_remote_path(value: Any) -> bool:
            return _looks_like_remote_input_path(value)

        def _list_arg(name: str) -> list[str]:
            value = args.get(name)
            if not isinstance(value, list):
                return []
            out: list[str] = []
            for item in value:
                token = str(item or "").strip()
                if token:
                    out.append(token)
            return out

        def _paths_need_current_local_replacement(paths: list[str]) -> bool:
            if not paths:
                return False
            has_local_existing = any(
                Path(path).expanduser().exists()
                for path in paths
                if not _looks_like_remote_path(path)
            )
            has_local_missing = any(
                not Path(path).expanduser().exists()
                for path in paths
                if not _looks_like_remote_path(path)
            )
            return has_local_missing and not has_local_existing

        def _image_paths_need_current_local_replacement(paths: list[str]) -> bool:
            if _paths_need_current_local_replacement(paths):
                return True
            if not paths:
                return False
            has_local_existing = any(
                Path(path).expanduser().exists()
                for path in paths
                if not _looks_like_remote_path(path)
            )
            has_supported_remote = any(
                _looks_like_remote_path(path) and _looks_like_segmentation_remote_path(path)
                for path in paths
            )
            has_unsupported_remote = any(
                _looks_like_remote_path(path) and not _looks_like_segmentation_remote_path(path)
                for path in paths
            )
            # BisQue data_service/image_service URLs and other non-file remote placeholders
            # are not directly loadable scientific inputs for the segmentation stack. When the
            # current turn already has concrete uploaded/selection-local image files, prefer
            # those over stale conversational placeholders. Keep valid remote microscopy paths
            # like s3://...ome.zarr intact.
            return has_unsupported_remote and not has_supported_remote and not has_local_existing

        def _safe_download_token(value: Any, *, default: str) -> str:
            token = str(value or "").strip().rstrip("/").split("/")[-1]
            token = re.sub(r"[^A-Za-z0-9._-]+", "_", token)
            return token or default

        def _existing_selection_local_image_files() -> list[str]:
            candidates: list[str] = []
            for key in ("image_files", "downloaded_files"):
                values = selection_artifact_handles.get(key)
                if not isinstance(values, list):
                    continue
                for raw_value in values:
                    token = str(raw_value or "").strip()
                    if (
                        not token
                        or _looks_like_remote_path(token)
                        or _looks_like_bisque_resource(token)
                    ):
                        continue
                    path = Path(token).expanduser()
                    if not path.exists() or not _is_segmentation_image(path):
                        continue
                    resolved = str(path.resolve())
                    if resolved not in candidates:
                        candidates.append(resolved)
            return candidates

        def _selection_context_image_files() -> list[str]:
            local_candidates = _existing_selection_local_image_files()
            if local_candidates:
                return local_candidates
            for resource_uri in selection_resource_uris[:1]:
                resource_token = _safe_download_token(resource_uri, default="resource")
                managed_output = str(
                    (Path(_science_output_root("bisque_downloads")) / resource_token).resolve()
                )
                download_result = bisque_download_resource(resource_uri, managed_output)
                if not isinstance(download_result, dict) or not bool(
                    download_result.get("success")
                ):
                    continue
                candidate_paths: list[str] = []
                for key in ("local_path", "output_path"):
                    token = str(download_result.get(key) or "").strip()
                    if token:
                        candidate_paths.append(token)
                for row in list(download_result.get("download_rows") or []):
                    if not isinstance(row, dict):
                        continue
                    token = str(
                        row.get("output_path") or row.get("local_path") or row.get("path") or ""
                    ).strip()
                    if token:
                        candidate_paths.append(token)
                resolved_candidates: list[str] = []
                for raw_path in candidate_paths:
                    path = Path(raw_path).expanduser()
                    if not path.exists() or not _is_segmentation_image(path):
                        continue
                    resolved = str(path.resolve())
                    if resolved not in resolved_candidates:
                        resolved_candidates.append(resolved)
                if resolved_candidates:
                    logger.info(
                        "Resolved selection-context BisQue resource into %s local scientific image(s) for tool execution",
                        len(resolved_candidates),
                    )
                    return resolved_candidates
            return []

        if tool_name in {
            "load_bisque_resource",
            "delete_bisque_resource",
            "add_tags_to_resource",
            "bisque_add_gobjects",
            "bisque_fetch_xml",
            "bisque_download_resource",
        } and not args.get("resource_uri"):
            if selection_resource_uris:
                args["resource_uri"] = selection_resource_uris[0]
                logger.info(
                    "Injected selection-context resource_uri into %s: %s",
                    tool_name,
                    args["resource_uri"],
                )
            elif tool_name == "load_bisque_resource" and selection_dataset_uris:
                args["resource_uri"] = selection_dataset_uris[0]
                logger.info(
                    "Injected selection-context dataset_uri as resource_uri into %s: %s",
                    tool_name,
                    args["resource_uri"],
                )

        if tool_name == "bisque_download_dataset" and not args.get("dataset_uri"):
            if selection_dataset_uris:
                args["dataset_uri"] = selection_dataset_uris[0]
                logger.info(
                    "Injected selection-context dataset_uri into %s: %s",
                    tool_name,
                    args["dataset_uri"],
                )
        if tool_name == "bisque_download_dataset" and not args.get("output_dir"):
            dataset_token = _safe_download_token(args.get("dataset_uri"), default="dataset")
            args["output_dir"] = str(
                (Path(_science_output_root("bisque_downloads")) / dataset_token).resolve()
            )
            logger.info(
                "Injected managed output_dir into %s: %s",
                tool_name,
                args["output_dir"],
            )
        if (
            tool_name == "bisque_download_resource"
            and not args.get("output_path")
            and args.get("resource_uri")
        ):
            resource_token = _safe_download_token(args.get("resource_uri"), default="resource")
            args["output_path"] = str(
                (Path(_science_output_root("bisque_downloads")) / resource_token).resolve()
            )
            logger.info(
                "Injected managed output_path into %s: %s",
                tool_name,
                args["output_path"],
            )

        # Special handling for tools that commonly operate on uploaded files.
        if tool_name in {"upload_to_bisque", "bisque_add_to_dataset"} and not (
            args.get("dataset_uri") or args.get("dataset_name")
        ):
            if selection_dataset_uris:
                args["dataset_uri"] = selection_dataset_uris[0]
                logger.info(
                    "Injected selection-context dataset_uri into %s: %s",
                    tool_name,
                    args["dataset_uri"],
                )
            else:
                inferred_dataset = _infer_bisque_dataset_target_from_text(user_text)
                if isinstance(inferred_dataset, dict):
                    if inferred_dataset.get("dataset_uri"):
                        args["dataset_uri"] = inferred_dataset.get("dataset_uri")
                    if inferred_dataset.get("dataset_name"):
                        args["dataset_name"] = inferred_dataset.get("dataset_name")
                    if inferred_dataset.get("create_dataset_if_missing"):
                        args["create_dataset_if_missing"] = True
                    logger.info(
                        "Inferred BisQue dataset target for %s: uri=%s name=%s create_if_missing=%s",
                        tool_name,
                        args.get("dataset_uri"),
                        args.get("dataset_name"),
                        args.get("create_dataset_if_missing"),
                    )

        selection_image_files: list[str] = []
        if tool_name in {
            "bioio_load_image",
            "segment_image_megaseg",
            "segment_image_sam2",
            "segment_image_sam3",
            "estimate_depth_pro",
        }:
            selection_image_files = _selection_context_image_files()
        prompt_image_files = (
            infer_scientific_image_inputs_from_text(user_text)
            if tool_name
            in {
                "bioio_load_image",
                "segment_image_megaseg",
                "segment_image_sam2",
                "segment_image_sam3",
                "estimate_depth_pro",
            }
            else []
        )
        if tool_name == "bioio_load_image":
            provided_file_path = str(args.get("file_path") or "").strip()
            compatible_uploaded_files = [
                str(Path(str(p)).expanduser().resolve())
                for p in uploaded_files or []
                if _is_segmentation_image(Path(str(p)))
            ]
            compatible_selection_files = list(selection_image_files)
            resolved_uploaded_local: Path | None = None
            if provided_file_path:
                resolved_uploaded_local = _resolve_uploaded_local_path(
                    provided_file_path,
                    uploaded_files=uploaded_files,
                    staging_dir=Path(_science_output_root("tool_inputs")),
                )
            provided_local_exists = bool(
                provided_file_path
                and not _looks_like_remote_path(provided_file_path)
                and not _looks_like_bisque_resource(provided_file_path)
                and Path(provided_file_path).expanduser().exists()
            )
            if not provided_file_path and compatible_uploaded_files:
                args["file_path"] = compatible_uploaded_files[0]
                logger.info(
                    "Injected uploaded scientific image '%s' into bioio_load_image tool call",
                    Path(args["file_path"]).name,
                )
            elif not provided_file_path and compatible_selection_files:
                args["file_path"] = compatible_selection_files[0]
                logger.info(
                    "Injected selection-context scientific image '%s' into bioio_load_image tool call",
                    Path(args["file_path"]).name,
                )
            elif not provided_file_path and prompt_image_files:
                args["file_path"] = prompt_image_files[0]
                logger.info(
                    "Inferred prompt-scoped scientific image '%s' into bioio_load_image tool call",
                    args["file_path"],
                )
            elif resolved_uploaded_local is not None:
                args["file_path"] = str(resolved_uploaded_local)
                logger.info(
                    "Resolved bioio_load_image file_path '%s' to uploaded local file '%s'",
                    provided_file_path,
                    args["file_path"],
                )
            elif compatible_uploaded_files and (
                _looks_like_bisque_resource(provided_file_path)
                or (provided_file_path and not provided_local_exists)
            ):
                args["file_path"] = compatible_uploaded_files[0]
                logger.info(
                    "Replaced unavailable bioio_load_image file_path '%s' with uploaded local file '%s'",
                    provided_file_path,
                    args["file_path"],
                )
            elif compatible_selection_files and (
                _looks_like_bisque_resource(provided_file_path)
                or (provided_file_path and not provided_local_exists)
            ):
                args["file_path"] = compatible_selection_files[0]
                logger.info(
                    "Replaced unavailable bioio_load_image file_path '%s' with selection-context local file '%s'",
                    provided_file_path,
                    args["file_path"],
                )
            elif prompt_image_files and (
                not provided_file_path
                or (
                    _looks_like_bisque_resource(provided_file_path)
                    or (provided_file_path and not provided_local_exists)
                )
            ):
                args["file_path"] = prompt_image_files[0]
                logger.info(
                    "Replaced unavailable bioio_load_image file_path '%s' with prompt-scoped image '%s'",
                    provided_file_path,
                    args["file_path"],
                )

        def _apply_quantify_segmentation_defaults() -> str | None:
            provided_mask_paths = _list_arg("mask_paths")
            provided_gt_paths = _list_arg("ground_truth_paths")
            latest_mask_paths = _collect_latest_mask_paths_from_refs(latest_result_refs)
            latest_gt_paths = _collect_latest_ground_truth_paths_from_refs(latest_result_refs)
            uploaded_mask_files = [
                str(p)
                for p in uploaded_files
                if _looks_like_uploaded_mask_artifact(Path(str(p)))
            ]
            uploaded_gt_files = [
                str(p)
                for p in uploaded_files
                if _looks_like_uploaded_ground_truth_artifact(Path(str(p)))
            ]
            explicit_mask_locals = _existing_local_paths(provided_mask_paths)
            explicit_gt_locals = _existing_local_paths(provided_gt_paths)
            explicit_mask_is_plausible = any(
                _looks_like_explicit_mask_path(path) for path in explicit_mask_locals
            )
            explicit_gt_is_plausible = any(
                _looks_like_explicit_ground_truth_path(path) for path in explicit_gt_locals
            )
            replacement_mask_paths = latest_mask_paths or uploaded_mask_files
            replacement_gt_paths = latest_gt_paths or uploaded_gt_files
            latest_result_group_id = str(
                (latest_result_refs or {}).get("latest_segmentation_result_group_id") or ""
            ).strip()

            if not provided_mask_paths:
                if replacement_mask_paths:
                    args["mask_paths"] = replacement_mask_paths[:8]
                    logger.info(
                        "Injected %s mask artifact path(s) into quantify_segmentation_masks tool call",
                        len(args["mask_paths"]),
                    )
                else:
                    return json.dumps(
                        {
                            "success": False,
                            "error": (
                                "No segmentation mask artifacts are available for quantify_segmentation_masks. "
                                "Pass mask_paths explicitly or run segment_image_sam2 first."
                            ),
                            "next_step": "Run segment_image_sam2 or provide mask_paths.",
                        },
                        ensure_ascii=False,
                    )
            elif replacement_mask_paths and (
                not explicit_mask_locals or not explicit_mask_is_plausible
            ):
                args["mask_paths"] = replacement_mask_paths[:8]
                logger.info(
                    "Replaced ambiguous/non-mask inputs in quantify_segmentation_masks with %s inferred mask artifact(s)",
                    len(args["mask_paths"]),
                )
            elif provided_mask_paths and not explicit_mask_locals:
                return json.dumps(
                    {
                        "success": False,
                        "error": (
                            "quantify_segmentation_masks mask_paths must reference local mask artifacts for this turn. "
                            "The provided mask paths were not available locally."
                        ),
                        "next_step": "Use local mask artifact paths or run segment_image_sam2 first.",
                    },
                    ensure_ascii=False,
                )
            elif explicit_mask_locals:
                args["mask_paths"] = explicit_mask_locals

            if not provided_gt_paths and replacement_gt_paths:
                args["ground_truth_paths"] = replacement_gt_paths[:8]
                logger.info(
                    "Injected %s ground-truth mask path(s) into quantify_segmentation_masks tool call",
                    len(args["ground_truth_paths"]),
                )
            elif (
                provided_gt_paths
                and replacement_gt_paths
                and (not explicit_gt_locals or not explicit_gt_is_plausible)
            ):
                args["ground_truth_paths"] = replacement_gt_paths[:8]
                logger.info(
                    "Replaced ambiguous/non-ground-truth inputs in quantify_segmentation_masks with %s inferred label artifact(s)",
                    len(args["ground_truth_paths"]),
                )
            elif provided_gt_paths and not explicit_gt_locals:
                return json.dumps(
                    {
                        "success": False,
                        "error": (
                            "quantify_segmentation_masks ground_truth_paths must reference local label masks for this turn. "
                            "The provided ground-truth paths were not available locally."
                        ),
                        "next_step": "Use local label-mask paths or re-upload the ground-truth masks.",
                    },
                    ensure_ascii=False,
                )
            elif explicit_gt_locals:
                args["ground_truth_paths"] = explicit_gt_locals

            if latest_result_group_id and not str(args.get("result_group_id") or "").strip():
                args["result_group_id"] = latest_result_group_id
            return None

        if uploaded_files:
            if tool_name == "upload_to_bisque":
                if not args.get("file_paths") or args.get("file_paths") == []:
                    args["file_paths"] = uploaded_files
                    logger.info(f"Injected {len(uploaded_files)} uploaded file(s) into tool call")
            elif tool_name == "yolo_detect":
                provided_paths = _list_arg("file_paths")
                compatible_files = [
                    p
                    for p in uploaded_files
                    if _is_yolo_image(Path(str(p))) or _is_sequence_media(Path(str(p)))
                ]
                if not args.get("file_paths") or args.get("file_paths") == []:
                    if compatible_files:
                        args["file_paths"] = compatible_files
                        logger.info(
                            "Injected %s uploaded image/sequence file(s) into yolo_detect tool call",
                            len(compatible_files),
                        )
                else:
                    if compatible_files and _paths_need_current_local_replacement(provided_paths):
                        args["file_paths"] = compatible_files
                        logger.info(
                            "Replaced unavailable yolo_detect inputs with %s uploaded image/sequence file(s)",
                            len(compatible_files),
                        )
                if (
                    args.get("file_paths")
                    and not args.get("model_name")
                    and not args.get("model_path")
                    and _text_mentions_prairie_detection(user_text)
                ) or (
                    args.get("file_paths")
                    and not args.get("model_name")
                    and not args.get("model_path")
                    and _paths_suggest_prairie_detection(
                        [str(item) for item in list(args.get("file_paths") or [])]
                    )
                ):
                    args["model_name"] = _PRAIRIE_YOLO_MODEL_KEY
                    logger.info("Auto-selected prairie YOLO model for yolo_detect")
                if (
                    args.get("file_paths")
                    and not bool(args.get("include_stability_audit"))
                    and _is_prairie_yolo_alias(str(args.get("model_name") or ""))
                    and _text_requests_prediction_stability(user_text)
                ):
                    args["include_stability_audit"] = True
                    args["stability_top_k"] = int(args.get("stability_top_k") or 3)
                    logger.info("Auto-enabled prediction stability audit for yolo_detect")
            elif tool_name == "analyze_prediction_stability":
                provided_paths = _list_arg("file_paths")
                compatible_files = [p for p in uploaded_files if _is_yolo_image(Path(str(p)))]
                if not args.get("file_paths") or args.get("file_paths") == []:
                    if compatible_files:
                        args["file_paths"] = compatible_files
                        logger.info(
                            "Injected %s uploaded image file(s) into analyze_prediction_stability tool call",
                            len(compatible_files),
                        )
                else:
                    if compatible_files and _paths_need_current_local_replacement(provided_paths):
                        args["file_paths"] = compatible_files
                        logger.info(
                            "Replaced unavailable analyze_prediction_stability inputs with %s uploaded image file(s)",
                            len(compatible_files),
                        )
                if (
                    args.get("file_paths")
                    and not args.get("model_name")
                    and not args.get("model_path")
                ):
                    args["model_name"] = _PRAIRIE_YOLO_MODEL_KEY
                    logger.info("Auto-selected prairie YOLO model for analyze_prediction_stability")
            elif tool_name == "score_spectral_instability":
                provided_paths = _list_arg("file_paths")
                compatible_files = [p for p in uploaded_files if _is_yolo_image(Path(str(p)))]
                if not args.get("file_paths") or args.get("file_paths") == []:
                    if compatible_files:
                        args["file_paths"] = compatible_files
                        logger.info(
                            "Injected %s uploaded image/sequence file(s) into score_spectral_instability tool call",
                            len(compatible_files),
                        )
                else:
                    if compatible_files and _paths_need_current_local_replacement(provided_paths):
                        args["file_paths"] = compatible_files
                        logger.info(
                            "Replaced unavailable score_spectral_instability inputs with %s uploaded image/sequence file(s)",
                            len(compatible_files),
                        )
                if (
                    args.get("file_paths")
                    and not args.get("model_name")
                    and not args.get("model_path")
                ):
                    args["model_name"] = _PRAIRIE_YOLO_MODEL_KEY
                    logger.info("Auto-selected prairie YOLO model for score_spectral_instability")
            elif tool_name == "yolo_finetune_detect":
                has_any = any(
                    args.get(k)
                    for k in (
                        "file_paths",
                        "image_paths",
                        "label_paths",
                    )
                )
                if not has_any:
                    args["file_paths"] = uploaded_files
                    logger.info(
                        "Injected %s uploaded file(s) into yolo_finetune_detect tool call",
                        len(uploaded_files),
                    )
            elif tool_name == "run_bisque_module":
                request_auth = get_request_bisque_auth()
                has_request_bound_auth = bool(
                    request_auth
                    and (
                        str(getattr(request_auth, "access_token", "") or "").strip()
                        or str(getattr(request_auth, "password", "") or "").strip()
                        or str(getattr(request_auth, "username", "") or "").strip()
                    )
                )
                input_resources = args.get("input_resources")
                normalized_inputs: dict[str, str] = {}
                if isinstance(input_resources, dict):
                    for key, value in input_resources.items():
                        name = str(key or "").strip()
                        token = str(value or "").strip()
                        if name and token:
                            normalized_inputs[name] = token
                if not normalized_inputs:
                    alias_resource = str(args.pop("resource_uri", "") or "").strip()
                    if alias_resource:
                        normalized_inputs = {"Input Image": alias_resource}
                if not normalized_inputs:
                    first_uploaded = next(
                        (str(path).strip() for path in uploaded_files if str(path).strip()),
                        "",
                    )
                    if first_uploaded:
                        normalized_inputs = {
                            "Input Image": Path(first_uploaded).name or first_uploaded
                        }
                        logger.info(
                            "Injected uploaded file '%s' into run_bisque_module input_resources",
                            Path(first_uploaded).name or first_uploaded,
                        )
                if normalized_inputs:
                    args["input_resources"] = normalized_inputs
                if "uploaded_files" not in args:
                    args["uploaded_files"] = uploaded_files
                if has_request_bound_auth:
                    # Prevent model-supplied credentials from overriding the authenticated session.
                    args.pop("bisque_root", None)
                    args.pop("bisque_user", None)
                    args.pop("bisque_password", None)
                else:
                    if "bisque_root" in args:
                        sanitized_root = _sanitize_bisque_root(str(args.get("bisque_root") or ""))
                        if sanitized_root:
                            args["bisque_root"] = sanitized_root
                        else:
                            args.pop("bisque_root", None)
                    if "bisque_user" in args:
                        sanitized_user = _sanitize_bisque_credential(
                            str(args.get("bisque_user") or "")
                        )
                        if sanitized_user:
                            args["bisque_user"] = sanitized_user
                        else:
                            args.pop("bisque_user", None)
                    if "bisque_password" in args:
                        sanitized_password = _sanitize_bisque_credential(
                            str(args.get("bisque_password") or "")
                        )
                        if sanitized_password:
                            args["bisque_password"] = sanitized_password
                        else:
                            args.pop("bisque_password", None)
            elif tool_name == "analyze_csv":
                provided_paths = _list_arg("file_paths")
                csv_files = [p for p in uploaded_files if _is_tabular_csv(Path(str(p)))]
                if (not args.get("file_paths") or args.get("file_paths") == []) and csv_files:
                    args["file_paths"] = csv_files
                    logger.info(
                        "Injected %s uploaded CSV/tabular file(s) into analyze_csv tool call",
                        len(csv_files),
                    )
                elif csv_files and _paths_need_current_local_replacement(provided_paths):
                    args["file_paths"] = csv_files
                    logger.info(
                        "Replaced unavailable analyze_csv inputs with %s uploaded CSV/tabular file(s)",
                        len(csv_files),
                    )
            elif tool_name in {
                "segment_image_megaseg",
                "segment_image_sam2",
                "segment_image_sam3",
                "estimate_depth_pro",
            }:
                provided_paths = _list_arg("file_paths")
                if not args.get("file_paths") or args.get("file_paths") == []:
                    image_files = [
                        p for p in uploaded_files if _is_segmentation_image(Path(str(p)))
                    ]
                    if image_files:
                        args["file_paths"] = image_files
                        logger.info(
                            "Injected %s uploaded scientific image(s) into %s tool call",
                            len(image_files),
                            tool_name,
                        )
                else:
                    image_files = [
                        p for p in uploaded_files if _is_segmentation_image(Path(str(p)))
                    ]
                    if image_files and _image_paths_need_current_local_replacement(
                        provided_paths
                    ):
                        args["file_paths"] = image_files
                        logger.info(
                            "Replaced unavailable %s inputs with %s uploaded image(s)",
                            tool_name,
                            len(image_files),
                        )
            elif tool_name == "segment_evaluate_batch":
                provided_image_paths = _list_arg("image_paths")
                provided_gt_paths = _list_arg("ground_truth_paths")
                provided_image_has_remote = any(
                    _looks_like_remote_path(path) for path in provided_image_paths
                )
                provided_image_has_local = bool(_existing_local_paths(provided_image_paths))
                provided_gt_has_remote = any(
                    _looks_like_remote_path(path) for path in provided_gt_paths
                )
                provided_gt_has_local = bool(_existing_local_paths(provided_gt_paths))
                uploaded_gt_files = [
                    str(p)
                    for p in uploaded_files
                    if _looks_like_uploaded_ground_truth_artifact(Path(str(p)))
                ]
                uploaded_image_files = [
                    str(p)
                    for p in uploaded_files
                    if _is_segmentation_image(Path(str(p)))
                    and not _looks_like_uploaded_mask_artifact(Path(str(p)))
                ]
                if (
                    not provided_image_paths
                    or provided_image_has_remote
                    or not provided_image_has_local
                ) and uploaded_image_files:
                    args["image_paths"] = uploaded_image_files
                    logger.info(
                        "Injected %s uploaded scientific image(s) into segment_evaluate_batch tool call",
                        len(uploaded_image_files),
                    )
                elif provided_image_paths and not provided_image_has_local:
                    return json.dumps(
                        {
                            "success": False,
                            "error": (
                                "segment_evaluate_batch image_paths must reference local uploaded files for this turn. "
                                "The provided image paths were not available locally."
                            ),
                            "next_step": "Use uploaded local image paths or re-upload the images.",
                        },
                        ensure_ascii=False,
                    )
                if (
                    not provided_gt_paths or provided_gt_has_remote or not provided_gt_has_local
                ) and uploaded_gt_files:
                    args["ground_truth_paths"] = uploaded_gt_files
                    logger.info(
                        "Injected %s uploaded ground-truth mask(s) into segment_evaluate_batch tool call",
                        len(uploaded_gt_files),
                    )
                elif provided_gt_paths and not provided_gt_has_local:
                    return json.dumps(
                        {
                            "success": False,
                            "error": (
                                "segment_evaluate_batch ground_truth_paths must reference local uploaded label files for this turn. "
                                "The provided ground-truth paths were not available locally."
                            ),
                            "next_step": "Use uploaded local label paths or re-upload the ground-truth masks.",
                        },
                        ensure_ascii=False,
                    )
                if not args.get("ground_truth_paths"):
                    return json.dumps(
                        {
                            "success": False,
                            "error": (
                                "segment_evaluate_batch requires ground_truth_paths. "
                                "Upload label/ground-truth masks or run segment_image_sam2 for segmentation-only analysis first."
                            ),
                            "next_step": "Provide ground_truth_paths or call segment_image_sam2 instead.",
                        },
                        ensure_ascii=False,
                    )
            elif tool_name == "quantify_segmentation_masks":
                quantify_error = _apply_quantify_segmentation_defaults()
                if quantify_error:
                    return quantify_error

        if (not uploaded_files) and tool_name in {
            "segment_image_megaseg",
            "segment_image_sam2",
            "segment_image_sam3",
            "estimate_depth_pro",
        }:
            provided_paths = _list_arg("file_paths")
            if (
                not args.get("file_paths") or args.get("file_paths") == []
            ) and selection_image_files:
                args["file_paths"] = selection_image_files
                logger.info(
                    "Injected %s selection-context scientific image(s) into %s tool call",
                    len(selection_image_files),
                    tool_name,
                )
            elif (
                not args.get("file_paths") or args.get("file_paths") == []
            ) and prompt_image_files:
                args["file_paths"] = prompt_image_files
                logger.info(
                    "Inferred %s prompt-scoped scientific image(s) into %s tool call",
                    len(prompt_image_files),
                    tool_name,
                )
            elif selection_image_files and _image_paths_need_current_local_replacement(
                provided_paths
            ):
                args["file_paths"] = selection_image_files
                logger.info(
                    "Replaced unavailable %s inputs with %s selection-context image(s)",
                    tool_name,
                    len(selection_image_files),
                )
            elif prompt_image_files and _image_paths_need_current_local_replacement(
                provided_paths
            ):
                args["file_paths"] = prompt_image_files
                logger.info(
                    "Replaced unavailable %s inputs with %s prompt-scoped image(s)",
                    tool_name,
                    len(prompt_image_files),
                )
        if (not uploaded_files) and tool_name == "quantify_segmentation_masks":
            quantify_error = _apply_quantify_segmentation_defaults()
            if quantify_error:
                return quantify_error

        signature = inspect.signature(func)
        accepts_var_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if not accepts_var_kwargs:
            accepted = {
                name
                for name, parameter in signature.parameters.items()
                if parameter.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
            }
            dropped_args = [name for name in list(args.keys()) if name not in accepted]
            if dropped_args:
                for key in dropped_args:
                    args.pop(key, None)
                logger.info(
                    "Dropped unsupported tool arguments: tool=%s keys=%s",
                    tool_name,
                    ",".join(sorted(dropped_args)),
                )

        # Execute and return result
        result = func(**args)
        if (
            tool_name == "segment_image_sam3"
            and inferred_sam3_concept_prompt
            and isinstance(result, dict)
        ):
            result.setdefault("concept_prompt", inferred_sam3_concept_prompt)
            result["concept_prompt_source"] = "inferred_from_user_request"
            result["inferred_from_user_request"] = True
            warnings = result.get("warnings") if isinstance(result.get("warnings"), list) else []
            note = (
                "SAM3 concept prompt was inferred from the natural-language segmentation request: "
                f"'{inferred_sam3_concept_prompt}'."
            )
            if note not in warnings:
                result["warnings"] = list(warnings) + [note]
        return json.dumps(result)

    except Exception as e:
        logger.error(f"Tool execution error: {str(e)}")
        return json.dumps({"success": False, "error": str(e)})
