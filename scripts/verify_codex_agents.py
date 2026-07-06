#!/usr/bin/env python3
"""Verify Ultra's project-scoped Codex agent configuration.

The contract is intentionally narrow:
- Project config must not pin a model; agents inherit the parent model.
- Default reasoning effort is xhigh for this critical scientific codebase.
- Subagent nesting stays at depth 1 unless a future change updates this guard.
- Custom agents are focused, documented, and never set their own model.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    print(
        "ERROR: Python 3.11+ is required to parse TOML. "
        "Use the bundled Codex Python runtime or another modern Python.",
        file=sys.stderr,
    )
    raise SystemExit(2)


ROOT = Path(__file__).resolve().parents[1]
CODEX_DIR = ROOT / ".codex"
AGENTS_DIR = CODEX_DIR / "agents"
MEMORY_DIR = CODEX_DIR / "memory"
SKILLS_DIR = CODEX_DIR / "skills"
CONFIG_PATH = CODEX_DIR / "config.toml"
WORKFLOW_PATH = CODEX_DIR / "README.md"

REQUIRED_AGENTS = {
    "ultra-architect.toml": "ultra_architect",
    "ultra-control-plane.toml": "ultra_control_plane",
    "ultra-deepagents-runtime.toml": "ultra_deepagents_runtime",
    "ultra-explorer.toml": "ultra_explorer",
    "ultra-frontend-trace.toml": "ultra_frontend_trace",
    "ultra-imaging-pipeline.toml": "ultra_imaging_pipeline",
    "ultra-implementer.toml": "ultra_implementer",
    "ultra-nats-expert.toml": "ultra_nats_expert",
    "ultra-perf-engineer.toml": "ultra_perf_engineer",
    "ultra-reviewer.toml": "ultra_reviewer",
    "ultra-security-data.toml": "ultra_security_data",
    "ultra-test-verifier.toml": "ultra_test_verifier",
    "ultra-trace-debugger.toml": "ultra_trace_debugger",
}

READ_ONLY_AGENTS = {
    "ultra_architect",
    "ultra_control_plane",
    "ultra_deepagents_runtime",
    "ultra_explorer",
    "ultra_frontend_trace",
    "ultra_imaging_pipeline",
    "ultra_nats_expert",
    "ultra_perf_engineer",
    "ultra_reviewer",
    "ultra_security_data",
    "ultra_trace_debugger",
}

REQUIRED_MEMORY = {
    "ultra-imaging-pipeline.md": (
        "convert once",
        "derive_pyramid",
        "viewerinfo",
    ),
    "ultra-nats-expert.md": (
        "natsdocs",
        "at-least-once",
        "benchmark",
    ),
    "ultra-orchestrator.md": (
        "fan-out",
        "one writer",
        "repair loops",
    ),
    "ultra-perf-engineer.md": (
        "baseline",
        "regression gate",
        "p95",
    ),
    "ultra-trace-debugger.md": (
        "run forensics query pack",
        "blast-radius",
        "control_run_events",
    ),
}

REQUIRED_SKILLS = {
    "complex-software-development",
    "go-nats-development",
}

ALLOWED_AGENT_REASONING = {"high", "xhigh"}


def load_toml(path: Path) -> dict[str, Any]:
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        fail(f"Missing required file: {path.relative_to(ROOT)}")
    except tomllib.TOMLDecodeError as exc:
        fail(f"Invalid TOML in {path.relative_to(ROOT)}: {exc}")


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def verify_config() -> None:
    config = load_toml(CONFIG_PATH)
    require("model" not in config, ".codex/config.toml must not pin model")
    require(
        config.get("model_reasoning_effort") == "xhigh",
        ".codex/config.toml must set model_reasoning_effort = \"xhigh\"",
    )
    mcp_servers = config.get("mcp_servers")
    require(isinstance(mcp_servers, dict), ".codex/config.toml must define [mcp_servers]")
    nats_docs = mcp_servers.get("natsDocs")
    require(isinstance(nats_docs, dict), ".codex/config.toml must define [mcp_servers.natsDocs]")
    require(
        nats_docs.get("url") == "https://docs.nats.io/~gitbook/mcp",
        "mcp_servers.natsDocs.url must point at the official NATS docs MCP endpoint",
    )
    agents = config.get("agents")
    require(isinstance(agents, dict), ".codex/config.toml must define [agents]")
    require(agents.get("max_depth") == 1, "agents.max_depth must remain 1")
    require(
        int(agents.get("max_threads", 0)) == 10,
        "agents.max_threads must be 10: 13-agent roster, review waves run up to "
        "7 read-only reviewers in parallel, recon stays at 2-5 sharp briefs",
    )
    require(
        int(agents.get("job_max_runtime_seconds", 0)) >= 7200,
        "agents.job_max_runtime_seconds must be at least 7200",
    )


def verify_agents() -> None:
    require(AGENTS_DIR.is_dir(), "Missing required directory: .codex/agents")
    actual = {path.name for path in AGENTS_DIR.glob("*.toml")}
    missing = sorted(set(REQUIRED_AGENTS) - actual)
    extra = sorted(actual - set(REQUIRED_AGENTS))
    require(not missing, f"Missing required agent files: {', '.join(missing)}")
    require(not extra, f"Unexpected agent files: {', '.join(extra)}")

    for filename, expected_name in sorted(REQUIRED_AGENTS.items()):
        path = AGENTS_DIR / filename
        agent = load_toml(path)
        label = path.relative_to(ROOT)
        require(agent.get("name") == expected_name, f"{label} has wrong name")
        require(isinstance(agent.get("description"), str) and agent["description"].strip(), f"{label} missing description")
        instructions = agent.get("developer_instructions")
        require(isinstance(instructions, str) and instructions.strip(), f"{label} missing developer_instructions")
        require("model" not in agent, f"{label} must inherit the parent model")
        effort = agent.get("model_reasoning_effort")
        require(
            effort is None or effort in ALLOWED_AGENT_REASONING,
            f"{label} reasoning effort must be omitted, high, or xhigh",
        )
        if expected_name in READ_ONLY_AGENTS:
            require(
                agent.get("sandbox_mode") == "read-only",
                f"{label} must be sandbox_mode = \"read-only\"",
            )
        if expected_name == "ultra_implementer":
            require(
                "sandbox_mode" not in agent,
                f"{label} must inherit parent sandbox/write policy",
            )


def verify_workflow_doc() -> None:
    try:
        text = WORKFLOW_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        fail("Missing required file: .codex/README.md")
    lowered = text.lower()
    for phrase in (
        "single writer",
        "max_depth = 1",
        "inherit the parent model",
        "read-only",
        "two repair",
        "ultra_trace_debugger",
        "ultra_nats_expert",
        "ultra_imaging_pipeline",
        "ultra_perf_engineer",
        "natsdocs",
        ".codex/memory",
    ):
        require(phrase in lowered, f".codex/README.md must mention {phrase!r}")


def verify_memory() -> None:
    require(MEMORY_DIR.is_dir(), "Missing required directory: .codex/memory")
    actual = {path.name for path in MEMORY_DIR.glob("*.md")}
    missing = sorted(set(REQUIRED_MEMORY) - actual)
    require(not missing, f"Missing required memory files: {', '.join(missing)}")

    for filename, phrases in sorted(REQUIRED_MEMORY.items()):
        path = MEMORY_DIR / filename
        text = path.read_text(encoding="utf-8")
        require(text.strip(), f"{path.relative_to(ROOT)} must not be empty")
        lowered = text.lower()
        for phrase in phrases:
            require(
                phrase in lowered,
                f"{path.relative_to(ROOT)} must mention {phrase!r}",
            )


def verify_skills() -> None:
    require(SKILLS_DIR.is_dir(), "Missing required directory: .codex/skills")
    actual = {path.name for path in SKILLS_DIR.iterdir() if path.is_dir()}
    missing = sorted(REQUIRED_SKILLS - actual)
    require(not missing, f"Missing required skills: {', '.join(missing)}")

    import re

    for folder in sorted(actual):
        skill_path = SKILLS_DIR / folder / "SKILL.md"
        label = f".codex/skills/{folder}/SKILL.md"
        require(skill_path.is_file(), f"Missing required file: {label}")
        text = skill_path.read_text(encoding="utf-8")
        frontmatter = re.match(r"\A---\n(.*?)\n---\n", text, re.DOTALL)
        require(frontmatter is not None, f"{label} must start with YAML frontmatter")
        name = re.search(r"^name:\s*(\S+)\s*$", frontmatter.group(1), re.MULTILINE)
        require(
            name is not None and name.group(1) == folder,
            f"{label} frontmatter name must match its folder ({folder!r})",
        )
        description = re.search(r"^description:", frontmatter.group(1), re.MULTILINE)
        require(description is not None, f"{label} frontmatter must include a description")


def main() -> None:
    verify_config()
    verify_agents()
    verify_workflow_doc()
    verify_memory()
    verify_skills()
    print("Codex agent configuration verified.")


if __name__ == "__main__":
    main()
