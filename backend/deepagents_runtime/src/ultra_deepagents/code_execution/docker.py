from __future__ import annotations

import base64
import binascii
import os
import re
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

from deepagents.backends.protocol import ExecuteResponse, FileDownloadResponse, FileUploadResponse
from deepagents.backends.sandbox import BaseSandbox

from ultra_deepagents.code_execution.paths import resolve_workspace_file

MATPLOTLIBRC = """\
backend: Agg
figure.dpi: 300
savefig.dpi: 300
"""

# EX_TEMPFAIL: the sandbox is saturated but the request itself is fine — the caller
# (the model, or a retry) may try again shortly. Distinct from a real failure exit.
_SANDBOX_BUSY_EXIT_CODE = 75

# Process-wide admission control for sandbox containers. A single worker runs up to
# worker_max_concurrency (default 64) runs at once, and each run can launch a code-exec
# container; with N worker replicas pointed at ONE Docker daemon the in-flight container
# count is N x that. Without a shared ceiling, a burst of (by default unbounded-memory)
# sandboxes can OOM the host. This semaphore bounds simultaneous `docker run`s per worker
# PROCESS, so it is the guard to set before running multiple workers against one daemon.
# 0 (the default) disables it, leaving single-worker behaviour unchanged. Mirrors the
# git-clone semaphore in git_staging.py.
_sandbox_slot_lock = threading.Lock()
_sandbox_semaphore: threading.BoundedSemaphore | None = None
_sandbox_semaphore_ready = False


def _sandbox_max_concurrency_from_env() -> int:
    try:
        return max(0, int(os.getenv("ULTRA_DEEPAGENTS_SANDBOX_MAX_CONCURRENCY", "0")))
    except ValueError:
        return 0


def _sandbox_queue_timeout_seconds() -> float:
    """How long execute() waits for a free slot before returning a retryable busy
    result. Bounded so a saturated cap can never block a worker thread forever; <=0
    waits indefinitely (only safe when execute runs on a dedicated, not shared, pool)."""
    try:
        return max(0.0, float(os.getenv("ULTRA_DEEPAGENTS_SANDBOX_QUEUE_TIMEOUT_SECONDS", "120")))
    except ValueError:
        return 120.0


def _get_sandbox_semaphore() -> threading.BoundedSemaphore | None:
    """Lazily build the process-wide sandbox semaphore from the env (None = disabled).

    Built once and cached so every per-run DockerSandboxBackend shares one ceiling.
    """
    global _sandbox_semaphore, _sandbox_semaphore_ready
    if not _sandbox_semaphore_ready:
        with _sandbox_slot_lock:
            if not _sandbox_semaphore_ready:
                limit = _sandbox_max_concurrency_from_env()
                _sandbox_semaphore = threading.BoundedSemaphore(limit) if limit > 0 else None
                _sandbox_semaphore_ready = True
    return _sandbox_semaphore


def _reset_sandbox_semaphore_for_tests() -> None:
    """Test-only: drop the cached semaphore so the next call re-reads the env."""
    global _sandbox_semaphore, _sandbox_semaphore_ready
    with _sandbox_slot_lock:
        _sandbox_semaphore = None
        _sandbox_semaphore_ready = False


@dataclass(frozen=True)
class DockerSandboxConfig:
    image: str
    network: str = "none"
    cpus: float = 0.0
    memory: str = ""
    pids_limit: int = 0
    # Size of /dev/shm inside the container. Docker's default is a tiny 64MB, which
    # makes torch DataLoader(num_workers>0), OpenCV, joblib/loky, and any
    # multiprocessing shared-memory array crash with an opaque "Bus error (core
    # dumped)" the moment it fills. Scientific parallelism needs real shared memory,
    # so set this generously (e.g. "8g"). Empty keeps the Docker default.
    shm_size: str = ""
    # GPU passthrough for in-sandbox code (e.g. "all" or "device=0"). Empty = no GPU
    # in the sandbox (the default); GPU inference is otherwise reached via the
    # MegaSeg/RareSpot services. Opt-in per deployment: only set on a node whose
    # Docker daemon has the NVIDIA container runtime, or `docker run` fails outright.
    gpus: str = ""
    timeout_seconds: int = 0
    output_limit_bytes: int = 0
    # Identity stamped onto every container as Docker labels so the reaper
    # (code_execution/cleanup.reap_orphaned_sandbox_containers) can find and remove
    # leftover/orphaned sandbox containers WITHOUT ever touching a non-Ultra container
    # on a shared daemon. worker_id/run_id are diagnostic-only (let an operator trace a
    # leaked container back to its run); the reaper's kill set is gated by the
    # ultra.sandbox=1 label plus the per-container ultra.sandbox.cap (= timeout_seconds).
    worker_id: str = ""
    run_id: str = ""


class DockerSandboxBackend(BaseSandbox):
    """Deep Agents sandbox backend that executes commands in isolated Docker containers."""

    def __init__(
        self,
        *,
        workspace_dir: str | Path,
        config: DockerSandboxConfig,
        outputs_dir: str | Path | None = None,
    ) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.outputs_dir = Path(outputs_dir) if outputs_dir is not None else None
        self.config = config
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        if self.outputs_dir is not None:
            self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_matplotlib_config()
        self._ensure_writable_tmp()
        try:
            os.chmod(self.workspace_dir, 0o777)
            if self.outputs_dir is not None:
                os.chmod(self.outputs_dir, 0o777)
        except OSError:
            pass

    @property
    def id(self) -> str:
        return f"docker:{self.workspace_dir.resolve()}"

    def _ensure_matplotlib_config(self) -> None:
        config_dir = self.workspace_dir / ".cache" / "matplotlib"
        config_dir.mkdir(parents=True, exist_ok=True)
        for path in (self.workspace_dir / "matplotlibrc", config_dir / "matplotlibrc"):
            path.write_text(MATPLOTLIBRC)
        for path in (self.workspace_dir / ".cache", config_dir):
            try:
                os.chmod(path, 0o777)
            except OSError:
                pass

    def _ensure_writable_tmp(self) -> None:
        # The container rootfs is read-only and the in-container /tmp is a small,
        # RAM-backed tmpfs. Point TMPDIR (set in build_docker_command) at a disk-backed
        # dir on the /workspace bind mount so large scientific temp files (Bio-Formats
        # JVM scratch, dcm2niix, tifffile memmaps, intermediate arrays) land on real
        # disk and are not capped at the tmpfs size. Created here, on the host, so the
        # bind-mounted container sees it.
        tmp_dir = self.workspace_dir / ".tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(tmp_dir, 0o777)
        except OSError:
            pass

    def build_docker_command(self, command: str) -> list[str]:
        workspace_mount = f"{self.workspace_dir.resolve()}:/workspace:rw"
        docker_command = [
            "docker",
            "run",
            "--rm",
            # --rm is the primary cleanup (daemon-side AutoRemove fires on exit even if
            # our docker CLI is killed). These labels make the container self-identifying
            # so the reaper can back-stop the residual leak classes (running orphans whose
            # launching execute() timed out; rare stopped residue after a daemon restart)
            # while NEVER touching a non-Ultra container on a shared daemon. ultra.sandbox.cap
            # records THIS container's wall-clock cap so the reaper's age>cap+grace orphan test
            # uses the container's own cap even if the admin default changed since launch.
            "--label",
            "ultra.sandbox=1",
            "--label",
            f"ultra.sandbox.cap={self.config.timeout_seconds}",
            "--network",
            self.config.network,
            "--cap-drop",
            "ALL",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=512m",
            "--volume",
            workspace_mount,
            "--workdir",
            "/workspace",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--env",
            "MPLCONFIGDIR=/workspace/.cache/matplotlib",
            "--env",
            "XDG_CACHE_HOME=/workspace/.cache",
            # HOME defaults to read-only /root on this rootfs, which breaks libraries
            # that write to ~ directly (astropy, numba, fontconfig, torch hub). Point it
            # and TMPDIR at the writable, disk-backed /workspace bind mount.
            "--env",
            "HOME=/workspace",
            "--env",
            "TMPDIR=/workspace/.tmp",
        ]
        # no-new-privileges blocks setuid escalation, but snap-packaged docker (e.g. stitch)
        # denies exec under it ("operation not permitted"). Allow a per-node opt-out via env
        # while keeping --cap-drop ALL / --network none / --read-only / tmpfs intact.
        if os.getenv("ULTRA_DEEPAGENTS_SANDBOX_NO_NEW_PRIVILEGES", "true").strip().lower() not in ("0", "false", "no"):
            docker_command.extend(["--security-opt", "no-new-privileges"])
        if self.outputs_dir is not None:
            docker_command.extend(
                [
                    "--volume",
                    f"{self.outputs_dir.resolve()}:/outputs:rw",
                ]
            )
        if self.config.cpus > 0:
            docker_command.extend(["--cpus", str(self.config.cpus)])
        if self.config.memory.strip():
            docker_command.extend(["--memory", self.config.memory])
        if self.config.pids_limit > 0:
            docker_command.extend(["--pids-limit", str(self.config.pids_limit)])
        if self.config.shm_size.strip():
            docker_command.extend(["--shm-size", self.config.shm_size])
        if self.config.gpus.strip():
            docker_command.extend(["--gpus", self.config.gpus])
        # Diagnostic-only labels (NOT used by the reaper's kill filter) so an operator can
        # trace a leaked container back to the worker/run that created it.
        if self.config.worker_id.strip():
            docker_command.extend(["--label", f"ultra.sandbox.worker={self.config.worker_id.strip()}"])
        if self.config.run_id.strip():
            docker_command.extend(["--label", f"ultra.sandbox.run={self.config.run_id.strip()}"])
        docker_command.extend([self.config.image, "bash", "-lc", command])
        return docker_command

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        violation = validate_sandbox_command(command)
        if violation is not None:
            return ExecuteResponse(output=violation, exit_code=126)

        # The model's per-call timeout is intentionally IGNORED — only the operator's admin
        # ceiling (config.timeout_seconds) governs the sandbox, so the model can neither
        # extend nor shrink its own runtime cap (a security boundary). The hang fix is that
        # this admin ceiling now DEFAULTS non-zero (config.py / env), so a hung or zombie
        # call (a codeexec container that died mid-run) is bounded — subprocess.run kills the
        # docker process on expiry (exit 124, recoverable) — instead of awaiting forever.
        _ = timeout
        timeout_seconds = self.config.timeout_seconds

        # Aggregate, process-wide admission control: when an operator sets a cap
        # (ULTRA_DEEPAGENTS_SANDBOX_MAX_CONCURRENCY > 0) bound how many sandbox
        # containers this worker can have in flight at once. Acquire BEFORE launching,
        # release in finally, and fail fast with a retryable busy result rather than
        # blocking a worker thread forever when the cap is saturated.
        semaphore = _get_sandbox_semaphore()
        acquired = False
        if semaphore is not None:
            queue_timeout = _sandbox_queue_timeout_seconds()
            acquired = semaphore.acquire(timeout=queue_timeout if queue_timeout > 0 else None)
            if not acquired:
                return ExecuteResponse(
                    output=(
                        "Sandbox is at capacity: the per-worker concurrency cap "
                        "(ULTRA_DEEPAGENTS_SANDBOX_MAX_CONCURRENCY) is full. No code was "
                        "run; retry shortly."
                    ),
                    exit_code=_SANDBOX_BUSY_EXIT_CODE,
                )
        try:
            try:
                completed = subprocess.run(
                    self.build_docker_command(command),
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds if timeout_seconds > 0 else None,
                    check=False,
                )
            except FileNotFoundError:
                return ExecuteResponse(output="Docker executable not found.", exit_code=127)
            except subprocess.TimeoutExpired as exc:
                output = _combine_output(exc.stdout, exc.stderr)
                truncated_output, truncated = _truncate_output(
                    f"Command timed out after {timeout_seconds} seconds.\n{output}",
                    self.config.output_limit_bytes,
                )
                return ExecuteResponse(
                    output=truncated_output,
                    exit_code=124,
                    truncated=truncated,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                # A host-level launch failure (PermissionError on the docker binary,
                # ENOMEM/EAGAIN on fork, etc.) must surface as a structured result the
                # model can react to, not an unhandled graph error.
                return ExecuteResponse(
                    output=f"Sandbox launch failed: {exc}",
                    exit_code=127,
                )

            output, truncated = _truncate_output(
                _combine_output(completed.stdout, completed.stderr),
                self.config.output_limit_bytes,
            )
            return ExecuteResponse(
                output=output,
                exit_code=completed.returncode,
                truncated=truncated,
            )
        finally:
            if acquired:
                semaphore.release()

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        responses: list[FileUploadResponse] = []
        for requested_path, content in files:
            try:
                target = resolve_workspace_file(self.workspace_dir, requested_path)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
            except ValueError:
                responses.append(FileUploadResponse(path=requested_path, error="permission_denied"))
            except OSError as exc:
                responses.append(FileUploadResponse(path=requested_path, error=str(exc)))
            else:
                responses.append(FileUploadResponse(path=requested_path, error=None))
        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        responses: list[FileDownloadResponse] = []
        for requested_path in paths:
            try:
                target = resolve_workspace_file(self.workspace_dir, requested_path)
                if not target.exists():
                    responses.append(
                        FileDownloadResponse(path=requested_path, error="file_not_found")
                    )
                    continue
                if not target.is_file():
                    responses.append(FileDownloadResponse(path=requested_path, error="not_a_file"))
                    continue
                content = target.read_bytes()
            except ValueError:
                responses.append(
                    FileDownloadResponse(path=requested_path, error="permission_denied")
                )
            except OSError as exc:
                responses.append(FileDownloadResponse(path=requested_path, error=str(exc)))
            else:
                responses.append(
                    FileDownloadResponse(path=requested_path, content=content, error=None)
                )
        return responses


def validate_sandbox_command(command: str) -> str | None:
    normalized = " ".join(str(command or "").split())
    if not normalized:
        return "Command is empty."
    if _uses_shell_timeout_wrapper(normalized):
        return _shell_timeout_message()
    if _searches_root_with_python_glob(normalized):
        return _root_search_message()
    if re.search(r"\bfind\s+/(?:\s|$)", normalized):
        return _root_search_message()
    if re.search(r"\bfind\s+/(?!workspace(?:/|\s|$))", normalized):
        return _root_search_message()
    if re.search(r"\bos\.walk\(\s*['\"]/\s*['\"]\s*\)", normalized):
        return _root_search_message()
    if re.search(r"Path\(\s*['\"]/\s*['\"]\s*\)\.rglob\(", normalized):
        return _root_search_message()
    if re.search(r"\b(?:grep|rg)\b[^;&|]*\s+/(?:\s|$)", normalized) and "-R" in normalized:
        return _root_search_message()
    return None


def _uses_shell_timeout_wrapper(command: str) -> bool:
    return bool(re.search(r"(?:^|(?:&&|\|\||;|\|)\s*)g?timeout\b", command))


# Recursive filesystem searches are only ever legitimate under the run's own mounts.
# A recursive walk anchored anywhere else (above all, "/") stat-walks the entire 9GB+
# sandbox image (torch + site-packages = hundreds of thousands of files), pegging a core
# and ballooning memory for tens of minutes — the exact failure that hung two prod runs.
_ALLOWED_SEARCH_ROOTS = ("/workspace", "/outputs")


def _decode_base64_string_literals(command: str) -> list[str]:
    """Decode base64 string literals embedded in a sandbox command.

    The deepagents ls/glob/grep backend tools base64-encode their path and pattern
    arguments into the python they generate (so an arbitrary path can never break the
    shell quoting). That same encoding hides "/" and "**/*" from the literal root-search
    regexes below — so a `glob('**/*', path='/')` sails straight past the guard and
    scans the whole image. Decode the literals here so the guard can see through it.
    Best-effort: anything that is not valid base64 / utf-8 is skipped.
    """
    decoded: list[str] = []
    for match in re.finditer(r"b64decode\(\s*['\"]([A-Za-z0-9+/=\s]+)['\"]", command):
        token = "".join(match.group(1).split())
        if not token:
            continue
        try:
            decoded.append(base64.b64decode(token, validate=True).decode("utf-8"))
        except (ValueError, binascii.Error, UnicodeDecodeError):
            continue
    return decoded


def _path_is_within_allowed_roots(path: str) -> bool:
    normalized = "/" + path.strip().strip("/")
    if normalized == "/":
        return False
    return any(
        normalized == root or normalized.startswith(root + "/")
        for root in _ALLOWED_SEARCH_ROOTS
    )


def _searches_root_with_python_glob(command: str) -> bool:
    if "glob" not in command and "rglob" not in command:
        return False
    # Literal form: glob.glob('/**...') / rglob anchored at the filesystem root.
    if re.search(r"['\"]/\*\*", command):
        return True
    # Base64-obfuscated form (the deepagents ls/glob tools encode path + pattern).
    if "recursive=True" not in command and "rglob" not in command:
        return False
    decoded = _decode_base64_string_literals(command)
    # (a) a decoded pattern that is itself a root-anchored recursive glob ("/**...").
    if any(value.startswith("/**") for value in decoded):
        return True
    # (b) a recursive "**" pattern combined with a decoded chdir base that escapes the
    #     allowed roots (the deepagents tool always encodes the absolute search base).
    if not any("**" in value for value in decoded):
        return False
    bases = [v for v in decoded if v.startswith("/") and "**" not in v]
    return any(not _path_is_within_allowed_roots(base) for base in bases)


def _root_search_message() -> str:
    return (
        "Recursive searches must stay under /workspace. Uploaded files are app "
        "storage handles; stage exact IDs with stage_uploaded_files_for_analysis "
        "before reading them, and use /workspace paths only."
    )


def _shell_timeout_message() -> str:
    return (
        "Do not wrap sandbox commands with shell timeout. Long-running analysis is "
        "allowed; run the command directly and let the platform's operator-configured "
        "sandbox policy handle any hard limit."
    )


def _combine_output(stdout: str | bytes | None, stderr: str | bytes | None) -> str:
    parts = []
    for value in (stdout, stderr):
        if isinstance(value, bytes):
            parts.append(value.decode("utf-8", errors="replace"))
        elif value:
            parts.append(value)
    return "".join(parts)


def _truncate_output(output: str, limit_bytes: int) -> tuple[str, bool]:
    if limit_bytes <= 0:
        return output, False
    encoded = output.encode("utf-8", errors="replace")
    if len(encoded) <= limit_bytes:
        return output, False
    truncated = encoded[:limit_bytes].decode("utf-8", errors="replace")
    return f"{truncated}\n...[truncated]", True
