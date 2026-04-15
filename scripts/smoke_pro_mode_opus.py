#!/usr/bin/env python3
from __future__ import annotations

import json
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
        published_api = pro_mode_meta.get("published_api") or {}
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
        return 0
    except Exception as exc:  # pragma: no cover - smoke script
        print(f"local_backend_error={exc}")
        return 5


def main() -> int:
    settings = Settings()
    exit_code = 0
    if settings.pro_mode_transport != "bedrock_published_api":
        print(
            "Configured PRO_MODE_TRANSPORT is not 'bedrock_published_api'. "
            "Local Opus smoke is intended for the Bedrock-published path."
        )
        exit_code = 1
    gateway_status = _probe_gateway(settings)
    if gateway_status != 0:
        exit_code = gateway_status
    local_status = _probe_local_backend(settings)
    if local_status != 0 and exit_code == 0:
        exit_code = local_status
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
