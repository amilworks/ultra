from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from ultra_deepagents.agent import _should_register_git_tools, git_staging_config
from ultra_deepagents.code_execution.git_staging import (
    GitStageError,
    GitStagingConfig,
    clone_repo_to_dir,
    repo_slug,
    validate_commit,
    validate_git_repo_url,
    validate_ref,
)
from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.context import AgentRunContext

ALLOWED = ("github.com",)


# --------------------------------------------------------------------------- #
# URL validation — the trust boundary. Must reject everything but allowlisted   #
# https hosts.                                                                  #
# --------------------------------------------------------------------------- #
def test_validate_git_repo_url_accepts_allowlisted_https_and_normalizes():
    assert (
        validate_git_repo_url("https://github.com/owner/repo.git", allowed_hosts=ALLOWED)
        == "https://github.com/owner/repo.git"
    )
    # Query/fragment dropped; trailing slash trimmed; host lowercased.
    assert (
        validate_git_repo_url("https://GitHub.com/owner/repo/?x=1#frag", allowed_hosts=ALLOWED)
        == "https://github.com/owner/repo"
    )


@pytest.mark.parametrize(
    "url, code",
    [
        ("", "empty_url"),
        ("-oProxyCommand=evil", "invalid_url"),
        ("http://github.com/o/r.git", "scheme_not_allowed"),
        ("git://github.com/o/r.git", "scheme_not_allowed"),
        ("ssh://git@github.com/o/r.git", "scheme_not_allowed"),
        ("file:///etc/passwd", "scheme_not_allowed"),
        ("ext::sh -c whoami", "scheme_not_allowed"),
        ("https://user:token@github.com/o/r.git", "credentials_in_url"),
        ("https://gitlab.com/o/r.git", "host_not_allowed"),
        ("https://github.com.evil.com/o/r.git", "host_not_allowed"),
        ("https://github.com:8080/o/r.git", "port_not_allowed"),
        ("https://github.com", "invalid_url"),
    ],
)
def test_validate_git_repo_url_rejects(url, code):
    with pytest.raises(GitStageError) as excinfo:
        validate_git_repo_url(url, allowed_hosts=ALLOWED)
    assert excinfo.value.code == code


def test_validate_git_repo_url_respects_configured_allowlist():
    assert validate_git_repo_url(
        "https://gitlab.example.edu/lab/pipeline.git",
        allowed_hosts=("gitlab.example.edu",),
    ) == "https://gitlab.example.edu/lab/pipeline.git"


def test_validate_ref_and_commit():
    assert validate_ref("main") == "main"
    assert validate_ref("release/v1.2") == "release/v1.2"
    assert validate_ref("") == ""
    for bad in ("-x", "a b", "a;b", "$(x)"):
        with pytest.raises(GitStageError):
            validate_ref(bad)
    assert validate_commit("abc1234") == "abc1234"
    assert validate_commit("") == ""
    for bad in ("xyz", "g" * 12, "12345"):
        with pytest.raises(GitStageError):
            validate_commit(bad)


def test_repo_slug():
    assert repo_slug("https://github.com/owner/repo.git") == "owner_repo"
    assert repo_slug("https://github.com/Org/Sub.Project") == "Org_Sub.Project"


# --------------------------------------------------------------------------- #
# Clone mechanics — against a local repo (offline). Production keeps https-only; #
# tests widen allowed_protocols to exercise the real subprocess path.           #
# --------------------------------------------------------------------------- #
def _make_local_repo(path: Path, files: dict[str, str]) -> str:
    path.mkdir(parents=True, exist_ok=True)
    env = {
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
        "HOME": str(path / ".home"), "PATH": __import__("os").environ.get("PATH", ""),
    }
    (path / ".home").mkdir(exist_ok=True)
    subprocess.run(["git", "init", "-b", "main", str(path)], env=env, check=True, capture_output=True)
    for name, content in files.items():
        (path / name).write_text(content)
    subprocess.run(["git", "-C", str(path), "add", "-A"], env=env, check=True, capture_output=True)
    subprocess.run(["git", "-C", str(path), "commit", "-m", "init"], env=env, check=True, capture_output=True)
    out = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], env=env, check=True, capture_output=True, text=True
    )
    return out.stdout.strip()


def test_clone_repo_to_dir_clones_local_repo_and_records_commit(tmp_path: Path):
    head = _make_local_repo(tmp_path / "src", {"run.py": "print('hi')\n", "data.csv": "a,b\n1,2\n"})
    dest = tmp_path / "ws" / "staged_repos" / "src"
    info = clone_repo_to_dir(
        f"file://{(tmp_path / 'src').resolve()}",
        dest,
        config=GitStagingConfig(),
        allowed_protocols="https:file",
    )
    assert info["resolved_commit"] == head
    assert info["file_count"] == 2
    assert (dest / "run.py").read_text() == "print('hi')\n"


def test_clone_repo_to_dir_default_protocol_blocks_file_url(tmp_path: Path):
    # SECURITY REGRESSION: the production default (https only) must refuse file://.
    _make_local_repo(tmp_path / "src", {"x.py": "1\n"})
    with pytest.raises(GitStageError) as excinfo:
        clone_repo_to_dir(
            f"file://{(tmp_path / 'src').resolve()}",
            tmp_path / "ws" / "repo",
            config=GitStagingConfig(),
        )
    assert excinfo.value.code == "git_clone_failed"
    assert not (tmp_path / "ws" / "repo").exists()


def test_clone_repo_to_dir_enforces_size_cap_and_cleans_up(tmp_path: Path):
    _make_local_repo(tmp_path / "src", {"big.txt": "x" * 5000})
    dest = tmp_path / "ws" / "repo"
    with pytest.raises(GitStageError) as excinfo:
        clone_repo_to_dir(
            f"file://{(tmp_path / 'src').resolve()}",
            dest,
            config=GitStagingConfig(max_bytes=100),
            allowed_protocols="https:file",
        )
    assert excinfo.value.code == "repo_too_large"
    assert not dest.exists()


def test_clone_repo_to_dir_refuses_on_low_free_disk(tmp_path: Path, monkeypatch):
    import ultra_deepagents.code_execution.git_staging as gs

    class _Usage:
        free = 10  # far below 2 * max_bytes

    monkeypatch.setattr(gs.shutil, "disk_usage", lambda _p: _Usage)
    with pytest.raises(GitStageError) as excinfo:
        clone_repo_to_dir(
            "https://github.com/o/r.git",
            tmp_path / "ws" / "repo",
            config=GitStagingConfig(max_bytes=1_000_000),
            allowed_protocols="https:file",
        )
    assert excinfo.value.code == "insufficient_disk"
    # The clone never started (no dest left behind).
    assert not (tmp_path / "ws" / "repo").exists()


def test_clone_repo_to_dir_caps_concurrent_clones(tmp_path: Path, monkeypatch):
    import threading

    import ultra_deepagents.code_execution.git_staging as gs

    # Saturate a single-slot semaphore so the next clone is refused, not blocked.
    busy = threading.BoundedSemaphore(1)
    busy.acquire()
    monkeypatch.setattr(gs, "_CLONE_SEMAPHORE", busy)
    with pytest.raises(GitStageError) as excinfo:
        clone_repo_to_dir(
            "https://github.com/o/r.git",
            tmp_path / "ws" / "repo",
            config=GitStagingConfig(timeout_seconds=1),
        )
    assert excinfo.value.code == "staging_busy"


def test_stage_git_repo_redacts_host_path_from_failure_message(tmp_path: Path, monkeypatch):
    import ultra_deepagents.context_tools as ct

    ctx = _context(tmp_path)
    ws_root = str(Path(ctx.workspace_root).expanduser().resolve())

    def boom(url, dest, *, ref="", commit="", config, allowed_protocols="https"):
        raise GitStageError(
            "git_clone_failed", f"fatal: could not write to {ws_root}/staged_repos/owner_repo/x"
        )

    monkeypatch.setattr(ct, "clone_repo_to_dir", boom)
    result = ct.stage_git_repo(ctx, repo_url="https://github.com/owner/repo.git", config=GitStagingConfig())
    assert result["ok"] is False
    assert ws_root not in result["message"]
    assert "/workspace/staged_repos/owner_repo/x" in result["message"]


def test_clone_repo_to_dir_checks_out_pinned_commit(tmp_path: Path):
    head = _make_local_repo(tmp_path / "src", {"a.py": "1\n"})
    dest = tmp_path / "ws" / "repo"
    info = clone_repo_to_dir(
        f"file://{(tmp_path / 'src').resolve()}",
        dest,
        commit=head,
        config=GitStagingConfig(),
        allowed_protocols="https:file",
    )
    assert info["resolved_commit"] == head


# --------------------------------------------------------------------------- #
# stage_git_repo orchestration + model-visible redaction.                       #
# --------------------------------------------------------------------------- #
def _context(tmp_path: Path) -> AgentRunContext:
    ws = tmp_path / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    return AgentRunContext(
        assistant_id="a", org_id="o", user_id="u", project_id="p",
        thread_id="t", run_id="r", goal="g", workspace_root=str(ws),
    )


def test_stage_git_repo_disabled_returns_error(tmp_path: Path):
    from ultra_deepagents.context_tools import stage_git_repo

    result = stage_git_repo(
        _context(tmp_path),
        repo_url="https://github.com/o/r.git",
        config=GitStagingConfig(enabled=False),
    )
    assert result == {"ok": False, "error": "git_staging_disabled"}


def test_stage_git_repo_rejects_bad_url_before_any_clone(tmp_path: Path):
    from ultra_deepagents.context_tools import stage_git_repo

    result = stage_git_repo(
        _context(tmp_path),
        repo_url="http://github.com/o/r.git",
        config=GitStagingConfig(),
    )
    assert result["ok"] is False
    assert result["error"] == "scheme_not_allowed"


def test_stage_git_repo_for_analysis_text_redacts_host_paths(tmp_path: Path, monkeypatch):
    import json

    import ultra_deepagents.context_tools as ct

    def fake_clone(url, dest, *, ref="", commit="", config, allowed_protocols="https"):
        Path(dest).mkdir(parents=True, exist_ok=True)
        (Path(dest) / "run.py").write_text("1\n")
        return {"resolved_commit": "deadbeef", "total_bytes": 3, "file_count": 1}

    monkeypatch.setattr(ct, "clone_repo_to_dir", fake_clone)
    payload = ct.stage_git_repo_for_analysis_text(
        _context(tmp_path),
        repo_url="https://github.com/owner/repo.git",
        config=GitStagingConfig(),
    )
    data = json.loads(payload)
    assert data["ok"] is True
    assert data["sandbox_path"] == "/workspace/staged_repos/owner_repo"
    assert data["resolved_commit"] == "deadbeef"
    # Host filesystem path must be redacted to the sandbox path, never leaked.
    assert data["staged_path"] == "/workspace/staged_repos/owner_repo"
    assert str(tmp_path) not in payload


# --------------------------------------------------------------------------- #
# Registration gating + config plumbing.                                        #
# --------------------------------------------------------------------------- #
def _settings(**kw) -> RuntimeSettings:
    return RuntimeSettings(openai_base_url="http://x/v1", openai_model="deepseek_v4", **kw)


@pytest.mark.parametrize(
    "goal, expected",
    [
        ("Clone https://github.com/owner/repo and run it on my data", True),
        ("git clone the analysis pipeline", True),
        ("run the code from my repo.git on these images", True),
        ("Summarize the dataset statistics", False),
        ("plot a histogram of the values", False),
    ],
)
def test_should_register_git_tools_gates_on_goal(goal, expected, tmp_path):
    ctx = AgentRunContext(
        assistant_id="a", org_id="o", user_id="u", project_id="p",
        thread_id="t", run_id="r", goal=goal, workspace_root=str(tmp_path),
    )
    assert _should_register_git_tools(ctx, _settings()) is expected


def test_should_register_git_tools_respects_disable_flag(tmp_path):
    ctx = AgentRunContext(
        assistant_id="a", org_id="o", user_id="u", project_id="p",
        thread_id="t", run_id="r", goal="clone https://github.com/o/r",
        workspace_root=str(tmp_path),
    )
    assert _should_register_git_tools(ctx, _settings(git_staging_enabled=False)) is False


def test_git_staging_config_from_settings():
    cfg = git_staging_config(_settings(
        git_staging_allowed_hosts=("github.com", "gitlab.com"),
        git_staging_max_bytes=123,
        git_staging_timeout_seconds=45,
        git_staging_depth=2,
    ))
    assert cfg.enabled is True
    assert cfg.allowed_hosts == ("github.com", "gitlab.com")
    assert cfg.max_bytes == 123
    assert cfg.timeout_seconds == 45
    assert cfg.depth == 2
