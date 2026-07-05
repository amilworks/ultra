from __future__ import annotations

from pathlib import Path


def test_autonomy_live_smoke_requires_tool_capability_evidence():
    makefile = Path(__file__).resolve().parents[1] / "Makefile"
    text = makefile.read_text()
    target = text.split("autonomy-live-smoke:", 1)[1].split("\nautonomy-gate:", 1)[0]

    assert "--require-tool-capability-quality" in target
    assert "--capability-matrix" in target
    assert target.count("tool_capability_manifest") >= 2


def test_autonomy_gate_includes_frontend_refresh_and_artifact_hydration_checks():
    makefile = Path(__file__).resolve().parents[1] / "Makefile"
    text = makefile.read_text()

    gate_line = next(
        line for line in text.splitlines() if line.startswith("autonomy-gate:")
    )
    assert "frontend-autonomy-test" in gate_line

    frontend_target = text.split("frontend-autonomy-test:", 1)[1].split(
        "\nautonomy-live-smoke:", 1
    )[0]
    for expected in [
        "src/features/chat/run-artifact-hydration.test.ts",
        "src/features/chat/run-stream-recovery-app.test.ts",
        "src/features/chat/stale-conversation.test.ts",
        "src/lib/api.test.ts",
    ]:
        assert expected in frontend_target


def test_autonomy_gate_includes_deterministic_deepagents_autonomy_quality_checks():
    makefile = Path(__file__).resolve().parents[1] / "Makefile"
    text = makefile.read_text()

    gate_line = next(
        line for line in text.splitlines() if line.startswith("autonomy-gate:")
    )
    assert "deepagents-autonomy-test" in gate_line

    target = text.split("deepagents-autonomy-test:", 1)[1].split(
        "\ndeepagents-smoke:", 1
    )[0]
    for expected in [
        "tests/test_live_trace.py",
        "tests/test_runner_paper_preload.py",
    ]:
        assert expected in target


def test_autonomy_gate_workflow_installs_frontend_dependencies():
    workflow = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "autonomy-gate.yml"
    text = workflow.read_text()

    assert "pnpm/action-setup" in text
    assert "actions/setup-node" in text
    assert "pnpm --dir frontend install --frozen-lockfile" in text
    assert '"frontend/src/features/chat/**"' in text
    assert '"frontend/src/App.tsx"' in text
