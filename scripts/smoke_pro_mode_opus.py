#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from typing import Any

import httpx

from src.config import Settings


def _print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def _mask_secret(value: str | None) -> str:
    token = str(value or "").strip()
    if not token:
        return "(empty)"
    if len(token) <= 8:
        return "*" * len(token)
    return f"{token[:4]}...{token[-4:]}"


def _build_gateway_headers(settings: Settings) -> tuple[dict[str, str], dict[str, str]]:
    headers = {str(k): str(v) for k, v in (settings.pro_mode_default_headers or {}).items()}
    params = {str(k): str(v) for k, v in (settings.pro_mode_default_query or {}).items()}
    api_key = str(settings.resolved_pro_mode_api_key or "").strip()
    header_name = str(settings.pro_mode_api_key_header or "").strip()
    if header_name and api_key:
        prefix = settings.pro_mode_api_key_prefix
        if prefix is None:
            headers[header_name] = api_key
        else:
            prefix_text = str(prefix)
            headers[header_name] = f"{prefix_text}{api_key}" if prefix_text else api_key
    elif api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers, params


def _probe_gateway(settings: Settings) -> int:
    _print_header("Gateway Probe")
    base_url = str(settings.resolved_pro_mode_base_url or "").strip().rstrip("/")
    if not base_url:
        print("No PRO_MODE_BASE_URL configured.")
        return 1
    headers, params = _build_gateway_headers(settings)
    timeout = max(float(settings.resolved_pro_mode_timeout_seconds or 60), 30.0)
    payload = {
        "message": {
            "model": str(settings.resolved_pro_mode_model or "").strip(),
            "content": [{"contentType": "text", "body": "Reply with exactly: opus smoke ok"}],
        },
        "continueGenerate": False,
        "enableReasoning": True,
    }
    print(f"transport={settings.pro_mode_transport}")
    print(f"base_url={base_url}")
    print(f"model={settings.resolved_pro_mode_model}")
    print(f"auth_header={settings.pro_mode_api_key_header or 'Authorization'}")
    print(f"api_key={_mask_secret(settings.resolved_pro_mode_api_key)}")
    print(f"default_query={json.dumps(params, sort_keys=True)}")
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.post(
                f"{base_url}/conversation",
                headers=headers,
                params=params,
                json=payload,
            )
        print(f"status={response.status_code}")
        body = response.text[:1200]
        if response.is_success:
            print(body)
            return 0
        print(body)
        return 2
    except Exception as exc:  # pragma: no cover - smoke script
        print(f"gateway_error={exc}")
        return 3


def _probe_native_bedrock_auth(settings: Settings) -> int:
    _print_header("Native Bedrock Auth Probe")
    region = str(settings.resolved_pro_mode_aws_region or "").strip()
    profile = str(settings.resolved_pro_mode_aws_profile or "").strip()
    has_access_key = bool(str(settings.resolved_pro_mode_aws_access_key_id or "").strip())
    has_secret_key = bool(str(settings.resolved_pro_mode_aws_secret_access_key or "").strip())
    has_session_token = bool(str(settings.resolved_pro_mode_aws_session_token or "").strip())
    print(f"transport={settings.pro_mode_transport}")
    print(f"model={settings.resolved_pro_mode_model}")
    print(f"aws_region={region or '(unset)'}")
    print(f"aws_profile={profile or '(unset)'}")
    print(f"aws_sso_auth={settings.pro_mode_aws_sso_auth}")
    print(f"has_access_key={has_access_key}")
    print(f"has_secret_key={has_secret_key}")
    print(f"has_session_token={has_session_token}")
    print(f"env_AWS_PROFILE={'set' if os.getenv('AWS_PROFILE') else 'unset'}")
    print(f"env_AWS_REGION={'set' if os.getenv('AWS_REGION') else 'unset'}")
    try:
        import boto3
    except Exception as exc:  # pragma: no cover - smoke script
        print(f"aws_dependency_error={exc}")
        return 8
    try:
        session_kwargs: dict[str, Any] = {}
        if profile:
            session_kwargs["profile_name"] = profile
        if region:
            session_kwargs["region_name"] = region
        session = boto3.Session(**session_kwargs)
        sts_client = session.client("sts", region_name=region or None)
        identity = sts_client.get_caller_identity()
        arn = str(identity.get("Arn") or "")
        print(f"sts_account={str(identity.get('Account') or '')[-4:] or '(unknown)'}")
        print(f"sts_arn_suffix={arn.split('/')[-1] if arn else '(unknown)'}")
        return 0
    except Exception as exc:  # pragma: no cover - smoke script
        print(f"aws_auth_error={exc}")
        return 9


def _probe_local_backend(settings: Settings) -> int:
    _print_header("Local Backend Probe")
    api_root = str(settings.orchestrator_api_url or "http://127.0.0.1:8000").strip().rstrip("/")
    payload: dict[str, Any] = {
        "messages": [
            {
                "role": "user",
                "content": "Briefly explain why Otsu thresholding and watershed segmentation make different tradeoffs.",
            }
        ],
        "conversation_id": "local-opus-smoke",
        "workflow_hint": {"id": "pro_mode", "source": "slash_menu"},
        "reasoning_mode": "deep",
        "debug": True,
        "budgets": {"max_tool_calls": 8, "max_runtime_seconds": 180},
    }
    try:
        with httpx.Client(timeout=180.0) as client:
            response = client.post(f"{api_root}/v1/chat", json=payload)
        print(f"status={response.status_code}")
        body = response.text[:1200]
        if not response.is_success:
            print(body)
            return 4
        parsed = response.json()
        metadata = parsed.get("metadata") or {}
        pro_mode_meta = metadata.get("pro_mode") or {}
        model_route = pro_mode_meta.get("model_route") or {}
        runtime_status = str(pro_mode_meta.get("runtime_status") or "").strip().lower()
        published_api = pro_mode_meta.get("published_api") or model_route.get("published_api") or {}
        print(f"response_model={parsed.get('model')}")
        print(f"execution_path={pro_mode_meta.get('execution_path')}")
        print(f"runtime_status={runtime_status or '(missing)'}")
        print(f"transport={model_route.get('transport')}")
        print(f"active_model={model_route.get('active_model')}")
        print(f"fallback_used={model_route.get('fallback_used')}")
        print(f"published_api={json.dumps(published_api, sort_keys=True)}")
        print(f"response_text={str(parsed.get('response_text') or '')[:300]}")
        if runtime_status != "completed":
            print("Local backend returned a visible answer, but Pro Mode runtime_status was not completed.")
            return 6
        if settings.pro_mode_transport == "bedrock_published_api":
            conversation_id = str(published_api.get("conversation_id") or "").strip()
            message_id = str(published_api.get("message_id") or "").strip()
            if not conversation_id or not message_id:
                print(
                    "Local backend did not surface published_api conversation/message ids for the "
                    "Bedrock transport."
                )
                return 7
        if settings.pro_mode_transport == "aws_bedrock_claude":
            if str(model_route.get("transport") or "").strip() != "aws_bedrock_claude":
                print("Local backend did not report the native Bedrock Claude transport in metadata.")
                return 10
        return 0
    except Exception as exc:  # pragma: no cover - smoke script
        print(f"local_backend_error={exc}")
        return 5


def _post_local_chat(
    api_root: str,
    payload: dict[str, Any],
    *,
    timeout_seconds: float = 240.0,
) -> dict[str, Any]:
    with httpx.Client(timeout=timeout_seconds) as client:
        response = client.post(f"{api_root}/v1/chat", json=payload)
    response.raise_for_status()
    return response.json()


def _tool_invocations(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = parsed.get("metadata") or {}
    invocations = metadata.get("tool_invocations")
    if isinstance(invocations, list):
        return [item for item in invocations if isinstance(item, dict)]
    return []


def _tool_name(invocation: dict[str, Any]) -> str:
    return (
        str(
            invocation.get("tool")
            or invocation.get("tool_name")
            or invocation.get("name")
            or invocation.get("display_name")
            or ""
        )
        .strip()
    )


def _probe_local_code_tools(settings: Settings) -> int:
    _print_header("Local Code Tool Probe")
    api_root = str(settings.orchestrator_api_url or "http://127.0.0.1:8000").strip().rstrip("/")
    payload: dict[str, Any] = {
        "messages": [
            {
                "role": "user",
                "content": (
                    "In Pro Mode, write Python to compute the mean and standard deviation of "
                    "the list [1, 2, 3, 4, 5], run it, and explain the result briefly."
                ),
            }
        ],
        "conversation_id": "local-opus-code-smoke",
        "workflow_hint": {"id": "pro_mode", "source": "slash_menu"},
        "reasoning_mode": "deep",
        "debug": True,
        "budgets": {"max_tool_calls": 8, "max_runtime_seconds": 240},
    }
    try:
        parsed = _post_local_chat(api_root, payload, timeout_seconds=300.0)
        metadata = parsed.get("metadata") or {}
        pro_mode_meta = metadata.get("pro_mode") or {}
        tool_invocations = _tool_invocations(parsed)
        tool_names = [_tool_name(item) for item in tool_invocations]
        writer_route = (
            (pro_mode_meta.get("model_routes") or {}).get("pro_mode_final_writer")
            if isinstance(pro_mode_meta.get("model_routes"), dict)
            else {}
        ) or {}
        print(f"response_model={parsed.get('model')}")
        print(f"execution_path={pro_mode_meta.get('execution_path')}")
        print(f"execution_regime={pro_mode_meta.get('execution_regime')}")
        print(f"tool_runtime_model={pro_mode_meta.get('tool_runtime_model')}")
        print(f"writer_transport={writer_route.get('transport')}")
        print(f"writer_model={writer_route.get('active_model')}")
        print(f"tool_names={json.dumps(tool_names)}")
        print(f"response_text={str(parsed.get('response_text') or '')[:300]}")
        if "codegen_python_plan" not in tool_names or "execute_python_job" not in tool_names:
            print("Expected Pro Mode code workflow to run both codegen_python_plan and execute_python_job.")
            return 11
        if str(writer_route.get("transport") or "").strip() not in {"bedrock_published_api", "aws_bedrock_claude"}:
            print("Expected Pro Mode final writer to stay on the dedicated Opus transport.")
            return 12
        return 0
    except Exception as exc:  # pragma: no cover - smoke script
        print(f"local_code_tool_error={exc}")
        return 13


def _probe_local_bisque_tools(settings: Settings) -> int:
    _print_header("Local BisQue Tool Probe")
    if not str(settings.bisque_user or "").strip() or not str(settings.bisque_password or "").strip():
        print("Skipping BisQue tool probe because BISQUE_USER/BISQUE_PASSWORD are not configured.")
        return 0
    api_root = str(settings.orchestrator_api_url or "http://127.0.0.1:8000").strip().rstrip("/")
    payload: dict[str, Any] = {
        "messages": [
            {
                "role": "user",
                "content": "In Pro Mode, search BisQue for images and summarize one result briefly.",
            }
        ],
        "conversation_id": "local-opus-bisque-smoke",
        "workflow_hint": {"id": "pro_mode", "source": "slash_menu"},
        "reasoning_mode": "deep",
        "debug": True,
        "budgets": {"max_tool_calls": 8, "max_runtime_seconds": 240},
    }
    try:
        parsed = _post_local_chat(api_root, payload, timeout_seconds=300.0)
        metadata = parsed.get("metadata") or {}
        pro_mode_meta = metadata.get("pro_mode") or {}
        tool_invocations = _tool_invocations(parsed)
        tool_names = [_tool_name(item) for item in tool_invocations]
        text = str(parsed.get("response_text") or "")
        print(f"response_model={parsed.get('model')}")
        print(f"execution_path={pro_mode_meta.get('execution_path')}")
        print(f"tool_runtime_model={pro_mode_meta.get('tool_runtime_model')}")
        print(f"tool_names={json.dumps(tool_names)}")
        print(f"response_text={text[:300]}")
        if "search_bisque_resources" not in tool_names:
            print("Expected Pro Mode BisQue workflow to call search_bisque_resources.")
            return 14
        if "BisQue authentication required" in text:
            print("BisQue tool path still reports missing authentication.")
            return 15
        return 0
    except Exception as exc:  # pragma: no cover - smoke script
        print(f"local_bisque_tool_error={exc}")
        return 16


def main() -> int:
    settings = Settings()
    exit_code = 0
    if settings.pro_mode_transport == "bedrock_published_api":
        gateway_status = _probe_gateway(settings)
        if gateway_status != 0:
            exit_code = gateway_status
    elif settings.pro_mode_transport == "aws_bedrock_claude":
        auth_status = _probe_native_bedrock_auth(settings)
        if auth_status != 0:
            exit_code = auth_status
    else:
        print(
            "Configured PRO_MODE_TRANSPORT is neither 'bedrock_published_api' nor "
            "'aws_bedrock_claude'. The Opus smoke target is intended for those transports."
        )
        exit_code = 1
    local_status = _probe_local_backend(settings)
    if local_status != 0 and exit_code == 0:
        exit_code = local_status
    code_tool_status = _probe_local_code_tools(settings)
    if code_tool_status != 0 and exit_code == 0:
        exit_code = code_tool_status
    bisque_tool_status = _probe_local_bisque_tools(settings)
    if bisque_tool_status != 0 and exit_code == 0:
        exit_code = bisque_tool_status
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
