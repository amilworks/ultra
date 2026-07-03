import base64
import io
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest
from ultra_deepagents.code_execution import cleanup, docker
from ultra_deepagents.code_execution.cleanup import (
    cleanup_expired_code_workspaces,
    reap_orphaned_sandbox_containers,
)
from ultra_deepagents.code_execution.docker import DockerSandboxBackend, DockerSandboxConfig
from ultra_deepagents.code_execution.paths import code_workspace_path, resolve_workspace_file
from ultra_deepagents.context import AgentRunContext


def _rfc3339(epoch: float) -> str:
    """Docker-style RFC3339 UTC timestamp (trailing Z) for a given epoch second."""
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _fake_docker(stopped, running, inspect_map, removed_calls):
    """A subprocess.run stand-in that emulates docker ps/inspect/rm for the reaper."""

    def fake_run(cmd, *, capture_output, text, timeout, check):
        assert cmd[0] == "docker"
        sub = cmd[1]
        if sub == "ps":
            # The label filter is the multi-tenant safety boundary — assert it is ALWAYS present.
            assert "label=ultra.sandbox=1" in cmd
            if "-a" in cmd:
                assert "status=exited" in cmd and "status=created" in cmd and "status=dead" in cmd
                return subprocess.CompletedProcess(cmd, 0, stdout="\n".join(stopped) + "\n", stderr="")
            return subprocess.CompletedProcess(cmd, 0, stdout="\n".join(running) + "\n", stderr="")
        if sub == "inspect":
            cid = cmd[-1]
            started, cap = inspect_map[cid]
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{started}\t{cap}\n", stderr="")
        if sub == "rm":
            removed_calls.append((cmd[-1], "-f" in cmd))
            return subprocess.CompletedProcess(cmd, 0, stdout=cmd[-1] + "\n", stderr="")
        raise AssertionError(f"unexpected docker subcommand: {sub}")

    return fake_run


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
            shm_size="8g",
            gpus="all",
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
    assert "--shm-size" in command
    assert command[command.index("--shm-size") + 1] == "8g"
    assert "--gpus" in command
    assert command[command.index("--gpus") + 1] == "all"
    assert "--cap-drop" in command
    assert command[command.index("--cap-drop") + 1] == "ALL"
    assert "--security-opt" in command
    assert command[command.index("--security-opt") + 1] == "no-new-privileges"
    assert "--read-only" in command
    assert "--tmpfs" in command
    assert command[command.index("--tmpfs") + 1] == "/tmp:rw,nosuid,nodev,size=512m"
    assert "--workdir" in command
    assert command[command.index("--workdir") + 1] == "/workspace"
    # HOME/TMPDIR redirect ~ and temp files onto the writable, disk-backed /workspace
    # mount (rootfs is read-only; /tmp is a small RAM-backed tmpfs).
    assert "--env" in command
    assert "HOME=/workspace" in command
    assert "TMPDIR=/workspace/.tmp" in command
    assert (tmp_path / "workspace" / ".tmp").is_dir()
    assert "ultra-agent-sandbox:test" in command
    assert command[-3:] == ["bash", "-lc", "python analysis.py"]


def test_docker_sandbox_can_disable_no_new_privileges_for_snap_docker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("ULTRA_DEEPAGENTS_SANDBOX_NO_NEW_PRIVILEGES", "false")
    backend = DockerSandboxBackend(
        workspace_dir=tmp_path / "workspace",
        config=DockerSandboxConfig(image="ultra-agent-sandbox:test"),
    )

    command = backend.build_docker_command("python analysis.py")

    assert "--security-opt" not in command
    assert "--cap-drop" in command
    assert command[command.index("--cap-drop") + 1] == "ALL"
    assert "--read-only" in command
    assert "--network" in command
    assert command[command.index("--network") + 1] == "none"


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
    # shm/gpu are opt-in: a --gpus flag on a non-GPU daemon makes `docker run` fail, so it
    # must never appear unless explicitly configured.
    assert "--shm-size" not in command
    assert "--gpus" not in command
    # HOME/TMPDIR are always set (rootfs robustness), independent of resource caps.
    assert "HOME=/workspace" in command
    assert "TMPDIR=/workspace/.tmp" in command
    assert command[-3:] == ["bash", "-lc", "python train.py"]


def test_docker_sandbox_ignores_tool_timeout_when_admin_timeout_is_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    captured: dict[str, object] = {}

    def fake_run(
        command: list[str],
        *,
        timeout: int | None,
        source_command: str,
        progress_callback,
    ) -> subprocess.CompletedProcess[str]:
        captured["timeout"] = timeout
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(docker, "_run_command_with_progress", fake_run)
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
        timeout: int | None,
        source_command: str,
        progress_callback,
    ) -> subprocess.CompletedProcess[str]:
        captured["timeout"] = timeout
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(docker, "_run_command_with_progress", fake_run)
    backend = DockerSandboxBackend(
        workspace_dir=tmp_path / "workspace",
        config=DockerSandboxConfig(image="ultra-agent-sandbox:test", timeout_seconds=1800),
    )

    result = backend.execute("python train.py", timeout=7)

    assert captured["timeout"] == 1800
    assert result.exit_code == 0


def test_sandbox_concurrency_cap_returns_busy_when_saturated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("ULTRA_DEEPAGENTS_SANDBOX_MAX_CONCURRENCY", "1")
    monkeypatch.setenv("ULTRA_DEEPAGENTS_SANDBOX_QUEUE_TIMEOUT_SECONDS", "0.05")
    docker._reset_sandbox_semaphore_for_tests()
    semaphore = docker._get_sandbox_semaphore()
    assert semaphore is not None
    assert semaphore.acquire(timeout=1)  # occupy the only slot

    def _must_not_run(*_args, **_kwargs):
        raise AssertionError("subprocess.run must not be called when the sandbox is saturated")

    monkeypatch.setattr(docker.subprocess, "run", _must_not_run)
    backend = DockerSandboxBackend(
        workspace_dir=tmp_path / "workspace",
        config=DockerSandboxConfig(image="ultra-agent-sandbox:test"),
    )
    try:
        result = backend.execute("python train.py")
    finally:
        semaphore.release()
        docker._reset_sandbox_semaphore_for_tests()

    assert result.exit_code == docker._SANDBOX_BUSY_EXIT_CODE
    assert "capacity" in result.output.lower()


def test_sandbox_concurrency_cap_releases_slot_between_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("ULTRA_DEEPAGENTS_SANDBOX_MAX_CONCURRENCY", "1")
    monkeypatch.setenv("ULTRA_DEEPAGENTS_SANDBOX_QUEUE_TIMEOUT_SECONDS", "1")
    docker._reset_sandbox_semaphore_for_tests()

    def fake_run(command, *, timeout, source_command, progress_callback):
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(docker, "_run_command_with_progress", fake_run)
    backend = DockerSandboxBackend(
        workspace_dir=tmp_path / "workspace",
        config=DockerSandboxConfig(image="ultra-agent-sandbox:test"),
    )
    try:
        # A cap of 1 must still allow back-to-back runs: the slot is released after each.
        first = backend.execute("python a.py")
        second = backend.execute("python b.py")
    finally:
        docker._reset_sandbox_semaphore_for_tests()

    assert first.exit_code == 0
    assert second.exit_code == 0


def test_docker_sandbox_streams_throttled_execute_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    progress_events: list[dict[str, object]] = []

    class FakeProcess:
        stdout = io.StringIO("condition 1 ok\ncondition 2 ok\ncondition 3 ok\n")
        stderr = io.StringIO("")
        returncode = 0

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

        def kill(self):
            raise AssertionError("completed fake process must not be killed")

    def fake_popen(command, *, stdout, stderr, text, bufsize):
        assert stdout is subprocess.PIPE
        assert stderr is subprocess.PIPE
        assert text is True
        assert bufsize == 1
        return FakeProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setenv("ULTRA_DEEPAGENTS_SANDBOX_PROGRESS_INTERVAL_SECONDS", "999")
    backend = DockerSandboxBackend(
        workspace_dir=tmp_path / "workspace",
        config=DockerSandboxConfig(image="ultra-agent-sandbox:test", timeout_seconds=60),
        progress_callback=progress_events.append,
    )

    result = backend.execute("python run_full_experiment.py")

    assert result.exit_code == 0
    assert result.output == "condition 1 ok\ncondition 2 ok\ncondition 3 ok\n"
    assert len(progress_events) == 2
    assert progress_events[0]["text"] == "condition 1 ok"
    assert progress_events[0]["stream"] == "stdout"
    assert progress_events[0]["progress_index"] == 1
    assert progress_events[1]["text"] == "condition 3 ok"
    assert progress_events[1]["suppressed_line_count"] == 1
    assert progress_events[1]["progress_index"] == 2


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


def _encoded_glob_command(path: str, pattern: str) -> str:
    """Mirror the deepagents ls/glob backend tool, which base64-encodes its path and
    pattern into the generated python so the literals never appear in the command."""
    enc_path = base64.b64encode(path.encode()).decode()
    enc_pattern = base64.b64encode(pattern.encode()).decode()
    return (
        'python3 -c "'
        "import glob, os, base64; "
        f"path = base64.b64decode('{enc_path}').decode('utf-8'); "
        f"pattern = base64.b64decode('{enc_pattern}').decode('utf-8'); "
        "os.chdir(path); "
        'print(sorted(glob.glob(pattern, recursive=True)))"'
    )


def test_docker_sandbox_rejects_base64_encoded_root_globs_before_launch(tmp_path: Path):
    """The two prod hangs were `glob('**/*', path='/')` issued by the deepagents tool,
    which base64-encodes its args — so the literal `/**` regex never saw it. The guard
    must decode the literals and reject the root-anchored recursive walk before launch."""
    backend = DockerSandboxBackend(
        workspace_dir=tmp_path / "workspace",
        config=DockerSandboxConfig(image="ultra-agent-sandbox:test"),
    )

    for path, pattern in (("/", "**/*"), ("/", "**/detections.csv"), ("/opt", "**/*.pt")):
        unsafe = _encoded_glob_command(path, pattern)
        violation = docker.validate_sandbox_command(unsafe)
        result = backend.execute(unsafe)
        assert violation is not None, (path, pattern)
        assert result.exit_code == 126
        assert "Recursive searches must stay under /workspace" in result.output

    # A root-anchored recursive pattern in a single literal ("/**/*") is caught too.
    rooted = _encoded_glob_command("/workspace", "/**/*.csv")
    assert docker.validate_sandbox_command(rooted) is not None


def test_docker_sandbox_allows_base64_encoded_workspace_globs():
    # The same tool scoped to the run's own mounts must still pass unblocked.
    assert docker.validate_sandbox_command(_encoded_glob_command("/workspace", "**/*.csv")) is None
    assert docker.validate_sandbox_command(_encoded_glob_command("/outputs", "**/*.png")) is None
    assert (
        docker.validate_sandbox_command(_encoded_glob_command("/workspace/run", "**/*.json"))
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


def test_armed_admin_default_bounds_a_hung_execute(monkeypatch, tmp_path: Path):
    """The execute() hang fix: the admin ceiling now DEFAULTS non-zero, so a sandbox call
    that never returns is bounded (exit 124, recoverable) instead of awaiting forever — while
    the model's per-call timeout stays IGNORED (admin precedence is a security boundary)."""
    from ultra_deepagents.config import RuntimeSettings

    # the default admin ceiling is armed (was 0 = unbounded -> a latent infinite hang)
    assert RuntimeSettings(openai_base_url="x", openai_model="m").sandbox_timeout_seconds == 21600

    backend = DockerSandboxBackend(
        workspace_dir=tmp_path / "ws",
        config=DockerSandboxConfig(image="x:test", timeout_seconds=1800),
    )
    seen = {}

    def _fake_run(cmd, **kw):
        seen["timeout"] = kw.get("timeout")
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kw.get("timeout"), output=b"", stderr=b"")

    monkeypatch.setattr(docker, "_run_command_with_progress", _fake_run)

    # the model passes timeout=7, but the admin ceiling (1800) governs AND bounds the hang
    resp = backend.execute("python train.py", timeout=7)
    assert seen["timeout"] == 1800   # admin precedence preserved (model's 7 ignored)
    assert resp.exit_code == 124     # the hung call is surfaced as a recoverable error


def test_docker_sandbox_command_stamps_reaper_labels(tmp_path: Path):
    backend = DockerSandboxBackend(
        workspace_dir=tmp_path / "workspace",
        config=DockerSandboxConfig(
            image="img:test", timeout_seconds=21600, worker_id="w@host:1", run_id="run-7"
        ),
    )
    command = backend.build_docker_command("python x.py")

    assert "--rm" in command  # primary cleanup (daemon-side AutoRemove) is preserved
    assert "ultra.sandbox=1" in command  # the reaper's multi-tenant safety filter
    assert "ultra.sandbox.cap=21600" in command  # per-container age threshold
    assert "ultra.sandbox.worker=w@host:1" in command
    assert "ultra.sandbox.run=run-7" in command


def test_docker_sandbox_command_omits_diagnostic_labels_when_unset(tmp_path: Path):
    command = DockerSandboxBackend(
        workspace_dir=tmp_path / "ws2",
        config=DockerSandboxConfig(image="img:test"),
    ).build_docker_command("python x.py")

    # Ownership + cap labels are always present; worker/run diagnostics only when set.
    assert "ultra.sandbox=1" in command
    assert any(str(x).startswith("ultra.sandbox.cap=") for x in command)
    assert not any(str(x).startswith("ultra.sandbox.worker=") for x in command)
    assert not any(str(x).startswith("ultra.sandbox.run=") for x in command)


def test_reap_removes_stopped_and_aged_running_orphans(monkeypatch):
    now = 1_000_000.0
    cap, grace = 3600, 600
    inspect_map = {
        # healthy: started well within the cap → must NOT be touched
        "healthy1": (_rfc3339(now - 100), str(cap)),
        # orphan: older than cap+grace → must be force-removed
        "orphan1": (_rfc3339(now - (cap + grace + 50)), str(cap)),
    }
    removed_calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(cleanup.shutil, "which", lambda _exe: "/usr/bin/docker")
    monkeypatch.setattr(
        cleanup.subprocess,
        "run",
        _fake_docker(["stop1", "stop2"], ["healthy1", "orphan1"], inspect_map, removed_calls),
    )

    removed = reap_orphaned_sandbox_containers(
        default_cap_seconds=cap, grace_seconds=grace, max_per_sweep=200, now_seconds=now
    )

    assert ("stop1", False) in removed_calls  # stopped → plain rm
    assert ("stop2", False) in removed_calls
    assert ("orphan1", True) in removed_calls  # aged running → rm -f
    assert all(cid != "healthy1" for cid, _ in removed_calls)  # in-flight job untouched
    assert set(removed) == {"stop1", "stop2", "orphan1"}


def test_reap_skips_running_when_cap_disabled(monkeypatch):
    now = 1_000_000.0
    # very old, but cap label is 0 (disabled) → no defensible orphan threshold → leave it
    inspect_map = {"x1": (_rfc3339(now - 999_999), "0")}
    removed_calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(cleanup.shutil, "which", lambda _exe: "/usr/bin/docker")
    monkeypatch.setattr(
        cleanup.subprocess, "run", _fake_docker([], ["x1"], inspect_map, removed_calls)
    )

    removed = reap_orphaned_sandbox_containers(
        default_cap_seconds=0, grace_seconds=600, now_seconds=now
    )

    assert removed_calls == []
    assert removed == []


def test_reap_is_a_noop_without_docker(monkeypatch):
    monkeypatch.setattr(cleanup.shutil, "which", lambda _exe: None)

    def _boom(*_a, **_k):  # docker must never be invoked when the binary is absent
        raise AssertionError("docker should not be called")

    monkeypatch.setattr(cleanup.subprocess, "run", _boom)
    assert reap_orphaned_sandbox_containers(default_cap_seconds=3600) == []


def test_reap_swallows_docker_errors(monkeypatch):
    monkeypatch.setattr(cleanup.shutil, "which", lambda _exe: "/usr/bin/docker")

    def _raise(*_a, **_k):
        raise OSError("daemon down")

    monkeypatch.setattr(cleanup.subprocess, "run", _raise)
    # never raises; a down daemon / missing binary degrades to an empty sweep
    assert reap_orphaned_sandbox_containers(default_cap_seconds=3600) == []


def test_reap_respects_max_per_sweep(monkeypatch):
    stopped = [f"s{i}" for i in range(10)]
    removed_calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(cleanup.shutil, "which", lambda _exe: "/usr/bin/docker")
    monkeypatch.setattr(
        cleanup.subprocess, "run", _fake_docker(stopped, [], {}, removed_calls)
    )

    removed = reap_orphaned_sandbox_containers(
        default_cap_seconds=3600, max_per_sweep=3, now_seconds=1.0
    )

    assert len(removed) == 3  # bounded work, even with a backlog of 10


def test_parse_rfc3339_handles_nanoseconds_and_sentinels():
    epoch = cleanup._parse_rfc3339("2026-06-21T00:51:01.204848139Z")
    expected = datetime(2026, 6, 21, 0, 51, 1, 204848, tzinfo=timezone.utc).timestamp()
    assert epoch is not None and abs(epoch - expected) < 1e-3
    # never-started sentinel and blank → None (caller skips, not "infinitely old")
    assert cleanup._parse_rfc3339("0001-01-01T00:00:00Z") is None
    assert cleanup._parse_rfc3339("") is None
