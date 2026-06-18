import os
import subprocess
from pathlib import Path

import pytest
from ultra_deepagents.code_execution import docker
from ultra_deepagents.code_execution.cleanup import cleanup_expired_code_workspaces
from ultra_deepagents.code_execution.docker import DockerSandboxBackend, DockerSandboxConfig
from ultra_deepagents.code_execution.paths import code_workspace_path, resolve_workspace_file
from ultra_deepagents.context import AgentRunContext


def test_code_workspace_path_is_scoped_by_org_user_and_run(tmp_path: Path):
    context = AgentRunContext(
        assistant_id="ultra-research",
        org_id="frontier/lab",
        user_id="faculty A",
        project_id="project-1",
        thread_id="thread-1",
        run_id="run:123",
    )

    path = code_workspace_path(root_dir=tmp_path, context=context)

    assert path == tmp_path / "frontier_lab" / "faculty_A" / "run_123" / "workspace"


def test_resolve_workspace_file_rejects_paths_outside_workspace(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    assert resolve_workspace_file(workspace, "/workspace/plots/a.png") == workspace / "plots/a.png"
    assert resolve_workspace_file(workspace, "plots/a.png") == workspace / "plots/a.png"
    with pytest.raises(ValueError, match="outside /workspace"):
        resolve_workspace_file(workspace, "/etc/passwd")
    with pytest.raises(ValueError, match="outside /workspace"):
        resolve_workspace_file(workspace, "../secret.txt")


def test_docker_sandbox_command_enforces_isolation_and_limits(tmp_path: Path):
    backend = DockerSandboxBackend(
        workspace_dir=tmp_path / "workspace",
        config=DockerSandboxConfig(
            image="ultra-agent-sandbox:test",
            cpus=1.5,
            memory="2g",
            pids_limit=128,
            timeout_seconds=9,
            output_limit_bytes=4096,
        ),
    )

    command = backend.build_docker_command("python analysis.py")

    assert command[:3] == ["docker", "run", "--rm"]
    assert "--network" in command
    assert command[command.index("--network") + 1] == "none"
    assert "--cpus" in command
    assert command[command.index("--cpus") + 1] == "1.5"
    assert "--memory" in command
    assert command[command.index("--memory") + 1] == "2g"
    assert "--pids-limit" in command
    assert command[command.index("--pids-limit") + 1] == "128"
    assert "--cap-drop" in command
    assert command[command.index("--cap-drop") + 1] == "ALL"
    assert "--security-opt" in command
    assert command[command.index("--security-opt") + 1] == "no-new-privileges"
    assert "--read-only" in command
    assert "--tmpfs" in command
    assert command[command.index("--tmpfs") + 1] == "/tmp:rw,nosuid,nodev,size=512m"
    assert "--workdir" in command
    assert command[command.index("--workdir") + 1] == "/workspace"
    assert "ultra-agent-sandbox:test" in command
    assert command[-3:] == ["bash", "-lc", "python analysis.py"]


def test_docker_sandbox_omits_resource_limits_when_unset(tmp_path: Path):
    backend = DockerSandboxBackend(
        workspace_dir=tmp_path / "workspace",
        config=DockerSandboxConfig(
            image="ultra-agent-sandbox:test",
            cpus=0,
            memory="",
            pids_limit=0,
            timeout_seconds=0,
            output_limit_bytes=0,
        ),
    )

    command = backend.build_docker_command("python train.py")

    assert "--cpus" not in command
    assert "--memory" not in command
    assert "--pids-limit" not in command
    assert command[-3:] == ["bash", "-lc", "python train.py"]


def test_docker_sandbox_ignores_tool_timeout_when_admin_timeout_is_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    captured: dict[str, object] = {}

    def fake_run(
        command: list[str],
        *,
        capture_output: bool,
        text: bool,
        timeout: int | None,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        captured["timeout"] = timeout
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    backend = DockerSandboxBackend(
        workspace_dir=tmp_path / "workspace",
        config=DockerSandboxConfig(image="ultra-agent-sandbox:test", timeout_seconds=0),
    )

    result = backend.execute("python train.py", timeout=7)

    assert captured["timeout"] is None
    assert result.exit_code == 0
    assert result.output == "ok"


def test_docker_sandbox_admin_timeout_overrides_tool_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    captured: dict[str, object] = {}

    def fake_run(
        command: list[str],
        *,
        capture_output: bool,
        text: bool,
        timeout: int | None,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        captured["timeout"] = timeout
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    backend = DockerSandboxBackend(
        workspace_dir=tmp_path / "workspace",
        config=DockerSandboxConfig(image="ultra-agent-sandbox:test", timeout_seconds=1800),
    )

    result = backend.execute("python train.py", timeout=7)

    assert captured["timeout"] == 1800
    assert result.exit_code == 0


def test_unlimited_output_limit_does_not_truncate():
    output, truncated = docker._truncate_output("x" * 1_000_000, 0)

    assert output == "x" * 1_000_000
    assert truncated is False


def test_docker_sandbox_rejects_recursive_root_globs_before_launch(tmp_path: Path):
    backend = DockerSandboxBackend(
        workspace_dir=tmp_path / "workspace",
        config=DockerSandboxConfig(image="ultra-agent-sandbox:test"),
    )

    unsafe = "python3 -c \"import glob; glob.glob('/**/*wetland*', recursive=True)\""
    violation = docker.validate_sandbox_command(unsafe)
    result = backend.execute(unsafe)

    assert violation is not None
    assert result.exit_code == 126
    assert "Recursive searches must stay under /workspace" in result.output


def test_docker_sandbox_rejects_shell_timeout_wrappers_before_launch(tmp_path: Path):
    backend = DockerSandboxBackend(
        workspace_dir=tmp_path / "workspace",
        config=DockerSandboxConfig(image="ultra-agent-sandbox:test"),
    )

    unsafe = "cd /workspace && timeout 200 python3 train.py"
    violation = docker.validate_sandbox_command(unsafe)
    result = backend.execute(unsafe)

    assert violation is not None
    assert result.exit_code == 126
    assert "Do not wrap sandbox commands with shell timeout" in result.output


def test_docker_sandbox_allows_workspace_scoped_recursive_searches():
    assert docker.validate_sandbox_command("find /workspace -name '*.csv'") is None
    assert (
        docker.validate_sandbox_command(
            "python3 -c \"import glob; glob.glob('/workspace/**/*.csv', recursive=True)\""
        )
        is None
    )


def test_docker_sandbox_upload_and_download_use_workspace_paths(tmp_path: Path):
    backend = DockerSandboxBackend(
        workspace_dir=tmp_path / "workspace",
        config=DockerSandboxConfig(image="ultra-agent-sandbox:test"),
    )

    upload = backend.upload_files([("/workspace/data/result.csv", b"x,y\n1,2\n")])[0]
    download = backend.download_files(["data/result.csv"])[0]

    assert upload.error is None
    assert download.error is None
    assert download.content == b"x,y\n1,2\n"


def test_cleanup_expired_code_workspaces_removes_old_run_dirs(tmp_path: Path):
    # Live layout is flat: <workspace_root>/<run_id> per run.
    old_run = tmp_path / "old-run"
    fresh_run = tmp_path / "fresh-run"
    old_run.mkdir(parents=True)
    fresh_run.mkdir(parents=True)
    (old_run / "plot.png").write_bytes(b"old")
    (fresh_run / "plot.png").write_bytes(b"fresh")
    old_mtime = 100.0
    fresh_mtime = 1_000.0
    os.utime(old_run, (old_mtime, old_mtime))
    os.utime(fresh_run, (fresh_mtime, fresh_mtime))

    removed = cleanup_expired_code_workspaces(
        root_dir=tmp_path,
        retention_seconds=100,
        now_seconds=1_000,
    )

    assert old_run in removed
    assert not old_run.exists()
    assert fresh_run.exists()


def test_cleanup_expired_code_workspaces_disabled_when_retention_not_positive(tmp_path: Path):
    run_dir = tmp_path / "ancient-run"
    run_dir.mkdir(parents=True)
    os.utime(run_dir, (1.0, 1.0))
    # retention 0 disables the sweep entirely (no accidental "delete everything").
    assert cleanup_expired_code_workspaces(root_dir=tmp_path, retention_seconds=0, now_seconds=1_000) == []
    assert run_dir.exists()
