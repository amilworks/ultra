"""Host-side Git repository staging for the Deep Agents sandbox.

A repository is cloned in the *worker* process (which has controlled network
egress), validated against a host allowlist, and materialized into the per-run
``/workspace`` mount. The model then runs that code via ``execute()`` in the
unchanged ``--network none`` sandbox — so fetched third-party code is contained
exactly like model-authored code, and only because sandbox egress stays off.

The trust boundary lives entirely here, before any container runs:
HTTPS-only, host allowlist, no embedded credentials, no credential helper, a
clean environment, a protocol allowlist (blocks ``file://``/``ext::`` submodule
smuggling), no submodule recursion, depth/size/time caps, and a reproducible
resolved commit SHA recorded for every clone.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


def _max_concurrent_clones_from_env() -> int:
    try:
        return max(1, int(os.getenv("ULTRA_DEEPAGENTS_GIT_STAGING_MAX_CONCURRENT_CLONES", "4")))
    except ValueError:
        return 4


# Process-wide cap on simultaneous clones. The per-clone byte cap multiplies by
# worker concurrency (up to 64 runs); without a shared limit a clone flood could
# fill a shared disk that co-hosts the control plane. One semaphore per worker
# process bounds the aggregate in-flight footprint.
_CLONE_SEMAPHORE = threading.BoundedSemaphore(_max_concurrent_clones_from_env())

# A commit is a 7-40 char hex SHA; a ref is a conservative branch/tag charset.
_SHA_RE = re.compile(r"\A[0-9a-fA-F]{7,40}\Z")
_REF_RE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._/+-]{0,199}\Z")
_HOST_RE = re.compile(r"\A[A-Za-z0-9.-]+\Z")

# Default transport policy: https only. ``protocol.*`` config + GIT_ALLOW_PROTOCOL
# pin the transport so a malicious .gitmodules / redirect cannot reach file://,
# ext::, ssh://; credential.helper="" disables host credential helpers.
_DEFAULT_ALLOWED_PROTOCOLS = "https"


def _git_hardening(allowed_protocols: str) -> list[str]:
    parts = [
        "-c", "credential.helper=",
        "-c", "core.askPass=",
        "-c", "protocol.allow=never",
        "-c", "submodule.recurse=false",
    ]
    for proto in allowed_protocols.split(":"):
        proto = proto.strip()
        if proto:
            parts += ["-c", f"protocol.{proto}.allow=always"]
    return parts


def _git_env(home: Path, allowed_protocols: str) -> dict[str, str]:
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(home),
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_ASKPASS": "/bin/true",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_ALLOW_PROTOCOL": allowed_protocols,
        "GIT_LFS_SKIP_SMUDGE": "1",
    }


class GitStageError(Exception):
    """Structured staging failure carrying a stable ``code`` for the tool result."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class GitStagingConfig:
    enabled: bool = True
    allowed_hosts: tuple[str, ...] = ("github.com",)
    max_bytes: int = 2 * 1024**3
    timeout_seconds: int = 600
    depth: int = 1


def _normalize_host(host: str | None) -> str:
    return str(host or "").strip().lower().rstrip(".")


def validate_git_repo_url(url: str, *, allowed_hosts: tuple[str, ...]) -> str:
    """Return a clean ``https://<host><path>`` URL or raise :class:`GitStageError`.

    HTTPS-only, host on the allowlist, default port, no embedded credentials,
    no leading ``-`` (option injection). Query and fragment are dropped.
    """
    raw = str(url or "").strip()
    if not raw:
        raise GitStageError("empty_url", "repo_url is required.")
    if raw.startswith("-"):
        raise GitStageError("invalid_url", "repo_url must not start with '-'.")
    parsed = urlparse(raw)
    if parsed.scheme.lower() != "https":
        raise GitStageError(
            "scheme_not_allowed",
            "Only https:// Git URLs are allowed (no ssh://, git://, file://, http://).",
        )
    if parsed.username or parsed.password:
        raise GitStageError("credentials_in_url", "repo_url must not embed credentials.")
    try:
        port = parsed.port
    except ValueError as exc:
        raise GitStageError("invalid_url", "repo_url has an invalid port.") from exc
    if port not in (None, 443):
        raise GitStageError("port_not_allowed", "Only the default https port (443) is allowed.")
    host = _normalize_host(parsed.hostname)
    if not host or not _HOST_RE.match(host):
        raise GitStageError("invalid_host", "repo_url has no valid host.")
    allowed = {_normalize_host(h) for h in allowed_hosts if str(h or "").strip()}
    if host not in allowed:
        raise GitStageError(
            "host_not_allowed",
            f"Host '{host}' is not on the Git clone allowlist {sorted(allowed)}.",
        )
    path = parsed.path.rstrip("/")
    if not path or path == "":
        raise GitStageError("invalid_url", "repo_url must include a repository path.")
    return f"https://{host}{path}"


def validate_ref(ref: str) -> str:
    value = str(ref or "").strip()
    if not value:
        return ""
    if not _REF_RE.match(value):
        raise GitStageError(
            "invalid_ref",
            "ref must be a simple branch/tag name (no spaces or leading '-').",
        )
    return value


def validate_commit(commit: str) -> str:
    value = str(commit or "").strip()
    if not value:
        return ""
    if not _SHA_RE.match(value):
        raise GitStageError("invalid_commit", "commit must be a 7-40 character hex SHA.")
    return value


def repo_slug(url: str) -> str:
    path = urlparse(url).path.strip("/")
    if path.endswith(".git"):
        path = path[:-4]
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", path.replace("/", "_")).strip("._")
    return slug or "repo"


def clone_repo_to_dir(
    url: str,
    dest: str | Path,
    *,
    ref: str = "",
    commit: str = "",
    config: GitStagingConfig,
    allowed_protocols: str = _DEFAULT_ALLOWED_PROTOCOLS,
) -> dict[str, object]:
    """Shallow-clone ``url`` into ``dest`` with hardened git, return clone facts.

    Optionally checks out an exact ``commit`` SHA; always records the resolved
    HEAD SHA. Enforces ``config.max_bytes`` and removes ``dest`` on overflow.
    ``allowed_protocols`` is a colon-separated transport allowlist; production
    keeps the ``"https"`` default and only tests widen it (e.g. ``"https:file"``
    to exercise clone mechanics against a local repo).
    """
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    depth = max(1, int(config.depth or 1))
    timeout = config.timeout_seconds if config.timeout_seconds and config.timeout_seconds > 0 else None

    # Bound concurrent clones process-wide, then refuse to start on a near-full
    # disk. The post-clone byte cap is a cleanup; these are the guards that keep a
    # clone flood or a giant tip from ENOSPC-ing a shared partition. The slot wait
    # is always finite (even if git ops are configured unbounded) so a saturated
    # semaphore can never deadlock the worker.
    if not _CLONE_SEMAPHORE.acquire(timeout=timeout or 600):
        raise GitStageError("staging_busy", "Too many concurrent repo clones; try again shortly.")
    try:
        _check_free_disk(dest.parent, config.max_bytes)
        return _do_clone(
            url, dest, ref=ref, commit=commit, config=config,
            depth=depth, timeout=timeout, allowed_protocols=allowed_protocols,
        )
    finally:
        try:
            _CLONE_SEMAPHORE.release()
        except ValueError:
            pass


def _check_free_disk(parent: Path, max_bytes: int) -> None:
    if max_bytes <= 0:
        return
    try:
        free = shutil.disk_usage(parent).free
    except OSError:
        return
    # A capped clone can write the working tree plus a similar-sized .git pack, so
    # require headroom for ~2x the cap before starting.
    needed = 2 * max_bytes
    if free < needed:
        raise GitStageError(
            "insufficient_disk",
            f"Refusing to clone: {free} bytes free, need at least {needed}.",
        )


def _do_clone(
    url: str,
    dest: Path,
    *,
    ref: str,
    commit: str,
    config: GitStagingConfig,
    depth: int,
    timeout: int | None,
    allowed_protocols: str,
) -> dict[str, object]:
    hardening = _git_hardening(allowed_protocols)
    # HOME is a throwaway dir on the worker so no ~/.gitconfig or ~/.git-credentials
    # is ever consulted (belt-and-suspenders with GIT_CONFIG_GLOBAL=/dev/null).
    home = Path(tempfile.mkdtemp(prefix="ultra-githome-"))
    env = _git_env(home, allowed_protocols)
    try:
        clone_cmd = [
            "git", *hardening, "clone",
            "--depth", str(depth), "--single-branch", "--no-tags",
            "--no-recurse-submodules",
        ]
        if ref:
            clone_cmd += ["--branch", ref]
        clone_cmd += ["--", url, str(dest)]
        _run_git(clone_cmd, env=env, timeout=timeout, cwd=dest.parent, what="clone")

        if commit:
            try:
                _run_git(
                    ["git", *hardening, "fetch", "--depth", str(depth), "origin", commit],
                    env=env, timeout=timeout, cwd=dest, what="fetch",
                )
            except GitStageError:
                # Shallow fetch can't always reach an arbitrary SHA; deepen once.
                _run_git(
                    ["git", *hardening, "fetch", "origin", commit],
                    env=env, timeout=timeout, cwd=dest, what="fetch",
                )
            _run_git(
                ["git", *hardening, "checkout", "--detach", commit],
                env=env, timeout=timeout, cwd=dest, what="checkout",
            )

        resolved = _run_git(
            ["git", "-C", str(dest), "rev-parse", "HEAD"],
            env=env, timeout=timeout, cwd=dest, what="rev-parse",
        ).strip()
    finally:
        shutil.rmtree(home, ignore_errors=True)

    total_bytes, file_count = _measure_tree(dest)
    if config.max_bytes > 0 and total_bytes > config.max_bytes:
        shutil.rmtree(dest, ignore_errors=True)
        raise GitStageError(
            "repo_too_large",
            f"Cloned repo is {total_bytes} bytes, exceeds the {config.max_bytes}-byte cap.",
        )
    return {"resolved_commit": resolved, "total_bytes": total_bytes, "file_count": file_count}


def _run_git(
    cmd: list[str],
    *,
    env: dict[str, str],
    timeout: int | None,
    cwd: str | Path,
    what: str,
) -> str:
    try:
        proc = subprocess.run(  # noqa: S603 - args are a fixed list, never shell-interpolated
            cmd,
            env=env,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise GitStageError("git_not_found", "git executable not found on the worker.") from exc
    except subprocess.TimeoutExpired as exc:
        raise GitStageError("timeout", f"git {what} timed out after {timeout}s.") from exc
    except (OSError, subprocess.SubprocessError) as exc:
        raise GitStageError("git_launch_failed", f"git {what} failed to launch: {exc}") from exc
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise GitStageError(f"git_{what}_failed", _short(detail) or f"git {what} failed.")
    return proc.stdout


def _measure_tree(root: Path) -> tuple[int, int]:
    """Size and file count of the working tree, excluding the ``.git`` dir.

    The cap targets the materialized content the sandbox will use; for a
    ``--depth 1`` clone ``.git`` is incidental overhead, and excluding it keeps
    the reported ``file_count`` meaningful (source files, not pack internals).
    """
    total = 0
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        if ".git" in dirnames:
            dirnames.remove(".git")
        for name in filenames:
            fp = Path(dirpath) / name
            try:
                if fp.is_symlink():
                    continue
                total += fp.stat().st_size
                count += 1
            except OSError:
                continue
    return total, count


def _short(text: str, limit: int = 600) -> str:
    cleaned = " ".join(str(text or "").split())
    return cleaned[:limit]
