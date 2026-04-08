"""External MCP configuration helpers for legacy chat integrations."""

from __future__ import annotations

import json
import os
from typing import Any

from src.logger import logger


def load_external_mcp_server_specs() -> list[dict[str, Any]]:
    """Load external MCP specs from environment.

    Environment
    ----------
    `AGENTS_EXTERNAL_MCP_SERVERS_JSON`
        JSON array of server spec objects.
    """

    raw = str(os.getenv("AGENTS_EXTERNAL_MCP_SERVERS_JSON") or "").strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except Exception as exc:
        logger.warning("Invalid AGENTS_EXTERNAL_MCP_SERVERS_JSON: %s", exc)
        return []
    if not isinstance(parsed, list):
        logger.warning("AGENTS_EXTERNAL_MCP_SERVERS_JSON must be a JSON array.")
        return []
    out: list[dict[str, Any]] = []
    for item in parsed:
        if isinstance(item, dict):
            out.append(dict(item))
    return out


def build_external_mcp_servers(specs: list[dict[str, Any]]) -> list[Any]:
    """Return normalized MCP server specs for legacy integrations.

    Supported spec shapes
    ---------------------
    Streamable HTTP
        `{"transport":"streamable_http","url":"https://...","name":"optional","headers":{...}}`
    SSE
        `{"transport":"sse","url":"https://...","name":"optional","headers":{...}}`
    Stdio
        `{"transport":"stdio","command":"npx","args":[...],"name":"optional","env":{...}}`
    """

    servers: list[Any] = []
    for index, item in enumerate(specs):
        transport = str(
            item.get("transport")
            or item.get("type")
            or item.get("protocol")
            or "streamable_http"
        ).strip().lower()
        name = str(item.get("name") or item.get("label") or "").strip() or None
        try:
            if transport in {"streamable_http", "streamablehttp", "http"}:
                url = str(item.get("url") or "").strip()
                if not url:
                    raise ValueError("missing `url` for streamable_http MCP server")
                params: dict[str, Any] = {"transport": "streamable_http", "url": url}
                headers = item.get("headers")
                if isinstance(headers, dict) and headers:
                    params["headers"] = {str(k): str(v) for k, v in headers.items()}
                timeout = item.get("timeout")
                if timeout is not None:
                    params["timeout"] = float(timeout)
                if name:
                    params["name"] = name
                servers.append(params)
                continue

            if transport == "sse":
                url = str(item.get("url") or "").strip()
                if not url:
                    raise ValueError("missing `url` for SSE MCP server")
                params = {"transport": "sse", "url": url}
                headers = item.get("headers")
                if isinstance(headers, dict) and headers:
                    params["headers"] = {str(k): str(v) for k, v in headers.items()}
                timeout = item.get("timeout")
                if timeout is not None:
                    params["timeout"] = float(timeout)
                if name:
                    params["name"] = name
                servers.append(params)
                continue

            if transport == "stdio":
                command = str(item.get("command") or "").strip()
                if not command:
                    raise ValueError("missing `command` for stdio MCP server")
                params = {"transport": "stdio", "command": command}
                args = item.get("args")
                if isinstance(args, list):
                    params["args"] = [str(token) for token in args]
                env = item.get("env")
                if isinstance(env, dict):
                    params["env"] = {str(k): str(v) for k, v in env.items()}
                cwd = str(item.get("cwd") or "").strip()
                if cwd:
                    params["cwd"] = cwd
                if name:
                    params["name"] = name
                servers.append(params)
                continue

            raise ValueError(f"unsupported transport `{transport}`")
        except Exception as exc:
            logger.warning("Skipping invalid MCP server spec #%s: %s", index, exc)
            continue

    return servers
