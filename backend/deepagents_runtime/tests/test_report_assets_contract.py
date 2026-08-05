"""The vendored report-asset contract, pinned in both directions.

The scientific-reporting skill and the coordinator prompt promise offline
IIFE bundles at /opt/report-assets/; the sandbox Dockerfile is what actually
ships them. These stay in lockstep or pages die silently at read time — a
skill that names an asset the image lacks produces confident dead tags, and
an image asset no guidance names is dead weight nobody uses.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCKERFILE = REPO_ROOT / "deploy" / "docker" / "deepagents-sandbox.Dockerfile"
SKILL = (
    REPO_ROOT
    / "backend"
    / "deepagents_runtime"
    / "skills"
    / "scientific-reporting"
    / "SKILL.md"
)
AGENT = (
    REPO_ROOT
    / "backend"
    / "deepagents_runtime"
    / "src"
    / "ultra_deepagents"
    / "agent.py"
)

VENDORED_ASSETS = (
    "/opt/report-assets/three.iife.min.js",
    "/opt/report-assets/chart.iife.min.js",
)


def test_dockerfile_builds_and_ships_every_promised_asset() -> None:
    dockerfile = DOCKERFILE.read_text()
    for asset in VENDORED_ASSETS:
        assert asset in dockerfile, f"Dockerfile does not ship {asset}"
    # Pinned versions, not floating tags — rebuilds must be reproducible.
    assert "chart.js@4.5.1" in dockerfile
    assert "chartjs-adapter-date-fns@3.0.0" in dockerfile
    assert "three@0.172.0" in dockerfile
    # Every bundle stage carries its license into the image.
    assert "CHART_LICENSE" in dockerfile
    assert "THREE_LICENSE" in dockerfile
    # Size gates keep a silently-empty or accidentally-bloated bundle from
    # shipping.
    assert dockerfile.count("size out of expected range") >= 2


def test_skill_and_prompt_teach_every_shipped_asset() -> None:
    skill = SKILL.read_text()
    agent = AGENT.read_text()
    for asset in VENDORED_ASSETS:
        assert asset in skill, f"SKILL.md never mentions {asset}"
        assert asset in agent, f"agent.py prompt never mentions {asset}"
    # The consumption contract is inline-the-file-contents; a path/src
    # reference is a dead tag under the reading canvas CSP.
    assert "FILE CONTENTS" in skill
    # Chart.js pages must know the global shape they get.
    assert "new Chart(ctx, config)" in skill
