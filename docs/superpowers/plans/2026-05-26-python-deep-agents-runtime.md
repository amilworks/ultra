# Python Deep Agents Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first Python Deep Agents runtime package that Go workers can call to execute researcher runs with typed context, vLLM model wiring, sandbox-backed code execution, and normalized run events.

**Architecture:** Keep Go as the durable run-control plane and add a separate Python package under `backend/deepagents_runtime/`. The runtime constructs Deep Agents using official APIs: `create_deep_agent`, typed runtime `context_schema`, direct backend instances, `CompositeBackend` for memory routing, sandbox backend for `execute`, and Deep Agents event streaming normalized into Go `run_event` records.

**Tech Stack:** Python 3.11+, `deepagents>=0.6.3,<0.7`, `langchain-openai`, `langchain-core`, `langgraph`, `pydantic`, `pytest`, Docker sandbox, OpenAI-compatible vLLM endpoint via `ChatOpenAI(base_url=..., model=...)`.

---

## Documentation Anchors

This plan must stay grounded in these official docs:

- Deep Agents context engineering says runtime context is per-run configuration passed with `context`, propagates to subagents, and should carry user metadata/API keys/config without being automatically injected into the prompt: https://docs.langchain.com/oss/python/deepagents/context-engineering
- Deep Agents backends now prefer pre-constructed backend instances instead of deprecated backend factories as of `deepagents` 0.5.0: https://docs.langchain.com/oss/python/deepagents/backends
- Deep Agents memory is filesystem-backed and long-term memory is controlled through backend routing and memory paths: https://docs.langchain.com/oss/python/deepagents/memory
- Async subagents are preview in `deepagents` 0.5.0 and are for long-running concurrent work; the first milestone should declare sync subagent specs and keep async wiring behind config until an Agent Protocol server is added: https://docs.langchain.com/oss/python/deepagents/async-subagents
- Deep Agents sandboxes are backends that expose filesystem tools plus `execute` and should isolate credentials, files, and network access: https://docs.langchain.com/oss/python/deepagents/sandboxes
- Deep Agents event streaming exposes messages, tool calls, subagents, and final output through `stream_events(..., version="v3")`: https://docs.langchain.com/oss/python/deepagents/event-streaming
- Deep Agents models can receive a configured model instance; LangChain `ChatOpenAI` supports `base_url` for OpenAI-compatible servers such as vLLM: https://docs.langchain.com/oss/python/deepagents/models and https://reference.langchain.com/python/langchain-openai/langchain_openai/chat_models/base/ChatOpenAI/

## File Structure

Create a self-contained runtime package so the dirty legacy `src/` deletion does not block progress:

```text
backend/deepagents_runtime/
  pyproject.toml
  README.md
  src/ultra_deepagents/__init__.py
  src/ultra_deepagents/config.py
  src/ultra_deepagents/context.py
  src/ultra_deepagents/model.py
  src/ultra_deepagents/agent.py
  src/ultra_deepagents/events.py
  src/ultra_deepagents/code_execution/__init__.py
  src/ultra_deepagents/code_execution/cleanup.py
  src/ultra_deepagents/code_execution/docker.py
  src/ultra_deepagents/code_execution/paths.py
  tests/test_config.py
  tests/test_model.py
  tests/test_agent_factory.py
  tests/test_event_normalizer.py
  tests/test_code_execution.py
```

Modify root files:

```text
Makefile
.env.example
```

## Task 1: Scaffold Runtime Package, Config, and Context

**Files:**
- Create: `backend/deepagents_runtime/pyproject.toml`
- Create: `backend/deepagents_runtime/src/ultra_deepagents/__init__.py`
- Create: `backend/deepagents_runtime/src/ultra_deepagents/config.py`
- Create: `backend/deepagents_runtime/src/ultra_deepagents/context.py`
- Create: `backend/deepagents_runtime/tests/test_config.py`
- Modify: `Makefile`
- Modify: `.env.example`

- [x] **Step 1: Write the failing config/context tests**

Create `backend/deepagents_runtime/tests/test_config.py`:

```python
from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.context import AgentRunContext


def test_runtime_settings_load_vllm_defaults(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://vrl-h200.ece.ucsb.edu:9393/v1")
    monkeypatch.setenv("OPENAI_MODEL", "deepseek_v4")
    settings = RuntimeSettings.from_env()
    assert settings.openai_base_url == "http://vrl-h200.ece.ucsb.edu:9393/v1"
    assert settings.openai_model == "deepseek_v4"
    assert settings.openai_api_key == "EMPTY"
    assert settings.sandbox_network == "none"


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
```

- [x] **Step 2: Run test to verify it fails**

Run:

```bash
cd backend/deepagents_runtime && uv run --python 3.11 --extra dev pytest tests/test_config.py -q
```

Expected: FAIL because package files do not exist.

- [x] **Step 3: Add package metadata and runtime config/context**

Create `backend/deepagents_runtime/pyproject.toml`:

```toml
[project]
name = "ultra-deepagents-runtime"
version = "0.1.0"
description = "Python Deep Agents runtime for BisQue Ultra"
requires-python = ">=3.11"
dependencies = [
    "deepagents>=0.6.3,<0.7",
    "langchain-core>=1.4.0,<2",
    "langchain-openai>=1.2.0,<2",
    "langgraph>=1.1.3,<2",
    "pydantic>=2.5.0",
    "pydantic-settings>=2.1.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/ultra_deepagents"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

Create `backend/deepagents_runtime/src/ultra_deepagents/__init__.py`:

```python
from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.context import AgentRunContext

__all__ = ["AgentRunContext", "RuntimeSettings"]
```

Create `backend/deepagents_runtime/src/ultra_deepagents/config.py`:

```python
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeSettings:
    openai_base_url: str
    openai_model: str
    openai_api_key: str = "EMPTY"
    request_timeout_seconds: float = 120.0
    max_retries: int = 1
    sandbox_image: str = "bisque-ultra-codeexec:py311"
    sandbox_network: str = "none"
    sandbox_cpus: float = 2.0
    sandbox_memory: str = "4g"
    sandbox_pids_limit: int = 256
    sandbox_timeout_seconds: int = 900
    sandbox_output_limit_bytes: int = 200_000

    @classmethod
    def from_env(cls) -> "RuntimeSettings":
        return cls(
            openai_base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:8001/v1"),
            openai_model=os.getenv("OPENAI_MODEL", "deepseek_v4"),
            openai_api_key=os.getenv("OPENAI_API_KEY") or "EMPTY",
            request_timeout_seconds=float(os.getenv("ULTRA_DEEPAGENTS_TIMEOUT_SECONDS", "120")),
            max_retries=int(os.getenv("ULTRA_DEEPAGENTS_MAX_RETRIES", "1")),
            sandbox_image=os.getenv("ULTRA_DEEPAGENTS_SANDBOX_IMAGE", "bisque-ultra-codeexec:py311"),
            sandbox_network=os.getenv("ULTRA_DEEPAGENTS_SANDBOX_NETWORK", "none"),
            sandbox_cpus=float(os.getenv("ULTRA_DEEPAGENTS_SANDBOX_CPUS", "2.0")),
            sandbox_memory=os.getenv("ULTRA_DEEPAGENTS_SANDBOX_MEMORY", "4g"),
            sandbox_pids_limit=int(os.getenv("ULTRA_DEEPAGENTS_SANDBOX_PIDS_LIMIT", "256")),
            sandbox_timeout_seconds=int(os.getenv("ULTRA_DEEPAGENTS_SANDBOX_TIMEOUT_SECONDS", "900")),
            sandbox_output_limit_bytes=int(
                os.getenv("ULTRA_DEEPAGENTS_SANDBOX_OUTPUT_LIMIT_BYTES", "200000")
            ),
        )
```

Create `backend/deepagents_runtime/src/ultra_deepagents/context.py`:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class AgentRunContext:
    assistant_id: str
    org_id: str
    user_id: str
    project_id: str
    thread_id: str
    run_id: str
    model_profile: str = "vllm"
    selected_file_ids: tuple[str, ...] = field(default_factory=tuple)
    selected_resource_uris: tuple[str, ...] = field(default_factory=tuple)
    selected_dataset_uris: tuple[str, ...] = field(default_factory=tuple)
    allowed_tool_packs: tuple[str, ...] = field(default_factory=tuple)
    budget: dict[str, Any] = field(default_factory=dict)
    auth_claims: dict[str, Any] = field(default_factory=dict)
    artifact_root: str = "/outputs"
    sandbox_policy: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["selected_file_ids"] = list(self.selected_file_ids)
        payload["selected_resource_uris"] = list(self.selected_resource_uris)
        payload["selected_dataset_uris"] = list(self.selected_dataset_uris)
        payload["allowed_tool_packs"] = list(self.allowed_tool_packs)
        return payload
```

Modify root `Makefile`:

```makefile
.PHONY: deepagents-test deepagents-smoke

deepagents-test:
	cd backend/deepagents_runtime && uv run --python 3.11 --extra dev pytest -q

deepagents-smoke:
	cd backend/deepagents_runtime && OPENAI_BASE_URL=$${OPENAI_BASE_URL:-http://vrl-h200.ece.ucsb.edu:9393/v1} OPENAI_MODEL=$${OPENAI_MODEL:-deepseek_v4} uv run --python 3.11 python -m ultra_deepagents.smoke
```

Append `.env.example` entries:

```bash
# Python Deep Agents runtime
ULTRA_DEEPAGENTS_TIMEOUT_SECONDS=120
ULTRA_DEEPAGENTS_MAX_RETRIES=1
ULTRA_DEEPAGENTS_SANDBOX_IMAGE=bisque-ultra-codeexec:py311
ULTRA_DEEPAGENTS_SANDBOX_NETWORK=none
ULTRA_DEEPAGENTS_SANDBOX_CPUS=2.0
ULTRA_DEEPAGENTS_SANDBOX_MEMORY=4g
ULTRA_DEEPAGENTS_SANDBOX_PIDS_LIMIT=256
ULTRA_DEEPAGENTS_SANDBOX_TIMEOUT_SECONDS=900
ULTRA_DEEPAGENTS_SANDBOX_OUTPUT_LIMIT_BYTES=200000
```

- [x] **Step 4: Run tests**

Run:

```bash
cd backend/deepagents_runtime && uv run --python 3.11 --extra dev pytest tests/test_config.py -q
```

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add .env.example Makefile backend/deepagents_runtime
git commit -m "feat: scaffold python deep agents runtime"
```

## Task 2: Port Cold Docker Sandbox Backend

**Files:**
- Create: `backend/deepagents_runtime/src/ultra_deepagents/code_execution/__init__.py`
- Create: `backend/deepagents_runtime/src/ultra_deepagents/code_execution/paths.py`
- Create: `backend/deepagents_runtime/src/ultra_deepagents/code_execution/docker.py`
- Create: `backend/deepagents_runtime/src/ultra_deepagents/code_execution/cleanup.py`
- Create: `backend/deepagents_runtime/tests/test_code_execution.py`

- [x] **Step 1: Write failing sandbox tests**

Create `backend/deepagents_runtime/tests/test_code_execution.py` using the existing `ultra_agent` sandbox tests as the source of truth, replacing imports with `ultra_deepagents`.

- [x] **Step 2: Run test to verify it fails**

Run:

```bash
cd backend/deepagents_runtime && uv run --python 3.11 --extra dev pytest tests/test_code_execution.py -q
```

Expected: FAIL because `code_execution` package does not exist.

- [x] **Step 3: Port the previous sandbox**

Port these files from `/Users/macbook/Documents/ultra_agent/src/ultra_agent/code_execution` with imports changed to `ultra_deepagents` and behavior preserved by tests:

```text
cleanup.py
docker.py
paths.py
__init__.py
```

Keep the existing Docker isolation behavior intact: no network by default, `/workspace` mount, CPU/memory/pid/time/output limits, read-only root, `/tmp` tmpfs, path escape prevention, upload/download helpers, and recursive-root-search rejection.

- [x] **Step 4: Run sandbox tests**

Run:

```bash
cd backend/deepagents_runtime && uv run --python 3.11 --extra dev pytest tests/test_code_execution.py -q
```

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add backend/deepagents_runtime
git commit -m "feat: port deep agents sandbox backend"
```

## Task 3: Add vLLM Model Factory

**Files:**
- Create: `backend/deepagents_runtime/src/ultra_deepagents/model.py`
- Create: `backend/deepagents_runtime/src/ultra_deepagents/smoke.py`
- Create: `backend/deepagents_runtime/tests/test_model.py`

- [x] **Step 1: Write failing model tests**

Create `backend/deepagents_runtime/tests/test_model.py`:

```python
from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.model import build_chat_model


def test_build_chat_model_uses_openai_compatible_base_url():
    settings = RuntimeSettings(
        openai_base_url="http://vrl-h200.ece.ucsb.edu:9393/v1",
        openai_model="deepseek_v4",
        openai_api_key="EMPTY",
    )
    model = build_chat_model(settings)
    assert model.openai_api_base == "http://vrl-h200.ece.ucsb.edu:9393/v1"
    assert model.model_name == "deepseek_v4"
```

- [x] **Step 2: Run test to verify it fails**

Run:

```bash
cd backend/deepagents_runtime && uv run --python 3.11 --extra dev pytest tests/test_model.py -q
```

Expected: FAIL because `model.py` does not exist.

- [x] **Step 3: Implement model factory and smoke module**

Use `langchain_openai.ChatOpenAI` with explicit `base_url`, `api_key`, `model`, `timeout`, and `max_retries`. Add `smoke.py` that invokes the model with a one-sentence prompt and prints the response length.

- [x] **Step 4: Run model test and optional live smoke**

Run:

```bash
cd backend/deepagents_runtime && uv run --python 3.11 --extra dev pytest tests/test_model.py -q
cd backend/deepagents_runtime && OPENAI_BASE_URL=http://vrl-h200.ece.ucsb.edu:9393/v1 OPENAI_MODEL=deepseek_v4 uv run --python 3.11 python -m ultra_deepagents.smoke
```

Expected: test PASS; live smoke prints a non-empty response or a clear connection/auth error.

- [x] **Step 5: Commit**

```bash
git add backend/deepagents_runtime
git commit -m "feat: add vllm chat model factory"
```

## Task 4: Add Deep Agents Factory

**Files:**
- Create: `backend/deepagents_runtime/src/ultra_deepagents/agent.py`
- Create: `backend/deepagents_runtime/tests/test_agent_factory.py`

- [x] **Step 1: Write failing agent factory tests**

Create tests that monkeypatch `ultra_deepagents.agent.create_deep_agent` and assert:

- `context_schema` is `AgentRunContext`.
- `model` is the configured vLLM chat model or injected fake model.
- `system_prompt` mentions `/memories/`, `/outputs/`, subagents, and sandbox execution.
- `backend` is a direct backend instance, not a deprecated runtime factory.
- subagents include `literature-reviewer`, `methods-critic`, `imaging-analyst`, and `statistics-analyst`.

- [x] **Step 2: Run test to verify it fails**

Run:

```bash
cd backend/deepagents_runtime && uv run --python 3.11 --extra dev pytest tests/test_agent_factory.py -q
```

Expected: FAIL because `agent.py` does not exist.

- [x] **Step 3: Implement factory**

Create `build_sandbox_backend(settings, workspace_dir=...)` plus `build_research_agent(settings, model=None, backend=None, workspace_dir=None, tools=None)`. The factory builds or accepts the model/backend, creates a direct `DockerSandboxBackend` when `workspace_dir` is supplied, otherwise falls back to a direct `StateBackend()` instance, and passes:

```python
create_deep_agent(
    name="ultra-research-agent",
    model=model or build_chat_model(settings),
    tools=list(tools or []),
    system_prompt=SYSTEM_PROMPT,
    context_schema=AgentRunContext,
    subagents=SUBAGENTS,
    backend=resolved_backend,
    memory=["/memories/preferences.md", "/memories/research_context.md"],
)
```

- [x] **Step 4: Run tests**

Run:

```bash
cd backend/deepagents_runtime && uv run --python 3.11 --extra dev pytest tests/test_agent_factory.py -q
```

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add backend/deepagents_runtime
git commit -m "feat: add deep agents research factory"
```

## Task 5: Normalize Deep Agents Events for Go

**Files:**
- Create: `backend/deepagents_runtime/src/ultra_deepagents/events.py`
- Create: `backend/deepagents_runtime/tests/test_event_normalizer.py`

- [x] **Step 1: Write failing event tests**

Create tests that pass synthetic message, tool-call, subagent, artifact, completion, and failure events into the normalizer and assert Go-compatible fields: `run_id`, `thread_id`, `event_kind`, `message`, `payload`, `node_name`, `task_id`, `agent_role`, and `level`.

- [x] **Step 2: Run test to verify it fails**

Run:

```bash
cd backend/deepagents_runtime && uv run --python 3.11 --extra dev pytest tests/test_event_normalizer.py -q
```

Expected: FAIL because `events.py` does not exist.

- [x] **Step 3: Implement normalizer**

Implement `RunEvent` dataclass and functions:

```python
normalize_message_delta(context, text, source="coordinator")
normalize_tool_call(context, tool_name, status, payload)
normalize_subagent_status(context, name, task_id, status, payload)
normalize_run_completed(context, response_text)
normalize_run_failed(context, error)
```

Return JSON-serializable dictionaries matching the Go `V2GraphEventRecord` contract.

- [x] **Step 4: Run event tests**

Run:

```bash
cd backend/deepagents_runtime && uv run --python 3.11 --extra dev pytest tests/test_event_normalizer.py -q
```

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add backend/deepagents_runtime
git commit -m "feat: normalize deep agents run events"
```

## Task 6: Final Verification

- [x] **Step 1: Run Python runtime tests**

```bash
make deepagents-test
```

- [x] **Step 2: Run Go control-plane tests**

```bash
make control-test
```

- [x] **Step 3: Run optional live vLLM smoke**

```bash
OPENAI_BASE_URL=http://vrl-h200.ece.ucsb.edu:9393/v1 OPENAI_MODEL=deepseek_v4 make deepagents-smoke
```

Expected: non-empty model response or a precise connection/auth/tool-calling error recorded in the final report.

- [x] **Step 4: Scan and diff-check**

```bash
rg -n "TB""D|TO""DO|FIX""ME|place""holder" docs/superpowers/plans/2026-05-26-python-deep-agents-runtime.md
git diff --check
```

- [x] **Step 5: Commit fixes if needed**

If formatting or generated lock files changed, commit them:

```bash
git add backend/deepagents_runtime docs/superpowers/plans/2026-05-26-python-deep-agents-runtime.md
git commit -m "chore: verify python deep agents runtime"
```

## Follow-Up Plans

This milestone intentionally does not implement the warm Docker pool or Go-to-Python job transport. The next plan should replace per-command `docker run --rm` with a warm per-run container lease and then add a Go worker client that calls the Python runtime and ingests normalized events.
