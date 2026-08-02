from __future__ import annotations

import base64
import binascii
import os
import queue
import re
import stat
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from deepagents.backends.protocol import ExecuteResponse, FileDownloadResponse, FileUploadResponse
from deepagents.backends.sandbox import BaseSandbox

from ultra_deepagents.code_execution.paths import resolve_workspace_file
from ultra_deepagents.code_execution.progress import (
    ExecuteProgressEvent,
    current_execute_progress_sink,
)

MATPLOTLIBRC = """\
backend: Agg
figure.dpi: 300
savefig.dpi: 300
"""

# EX_TEMPFAIL: the sandbox is saturated but the request itself is fine — the caller
# (the model, or a retry) may try again shortly. Distinct from a real failure exit.
_SANDBOX_BUSY_EXIT_CODE = 75
_DOCKER_IMAGE_ID_PATTERN = re.compile(r"^sha256:[0-9a-fA-F]{64}$")
_docker_image_id_cache: dict[str, str] = {}
_docker_image_id_cache_lock = threading.Lock()


def resolve_docker_image_id(image_ref: str) -> str:
    """Resolve a local Docker reference to the immutable image configuration ID.

    The returned ID can both pin ``docker run`` and be attached to execution
    trace events. Resolution is cached per worker process so a mutable tag is
    fixed for the lifetime of the worker instead of drifting between runs.
    Empty output means the image is not locally inspectable; callers may retain
    the original reference, but must not claim an immutable attestation.
    """

    reference = str(image_ref or "").strip()
    if not reference:
        return ""
    if _DOCKER_IMAGE_ID_PATTERN.fullmatch(reference):
        return reference.lower()
    # functools.lru_cache does not single-flight concurrent cold misses: two run
    # threads can resolve the same mutable tag to different IDs while an image is
    # being replaced. Keep lookup + inspection under one process lock so the ID
    # used by agent construction and the ID stamped on trace events are identical.
    with _docker_image_id_cache_lock:
        if reference in _docker_image_id_cache:
            return _docker_image_id_cache[reference]
        try:
            result = subprocess.run(
                ["docker", "image", "inspect", "--format={{.Id}}", reference],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            resolved = ""
        else:
            candidate = result.stdout.strip()
            resolved = (
                candidate.lower()
                if result.returncode == 0 and _DOCKER_IMAGE_ID_PATTERN.fullmatch(candidate)
                else ""
            )
        if len(_docker_image_id_cache) >= 64:
            _docker_image_id_cache.pop(next(iter(_docker_image_id_cache)))
        _docker_image_id_cache[reference] = resolved
        return resolved


def _reset_docker_image_id_cache_for_tests() -> None:
    with _docker_image_id_cache_lock:
        _docker_image_id_cache.clear()


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
    # None preserves the generic sandbox's environment-controlled compatibility
    # behavior. Security-sensitive typed primitives clone the backend with True,
    # so a process-level opt-out cannot weaken their container after registration.
    no_new_privileges: bool | None = None
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
        progress_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.outputs_dir = Path(outputs_dir) if outputs_dir is not None else None
        self.config = config
        self._progress_callback = progress_callback
        self._ensure_root_directory(self.workspace_dir, label="workspace")
        if self.outputs_dir is not None:
            self._ensure_root_directory(self.outputs_dir, label="outputs")
        self._ensure_matplotlib_config()
        self._ensure_numba_cache()
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

    @staticmethod
    def _ensure_root_directory(path: Path, *, label: str) -> None:
        if path.is_symlink():
            raise ValueError(f"unsafe {label} directory symlink: {path}")
        path.mkdir(parents=True, exist_ok=True)
        try:
            info = os.lstat(path)
        except OSError as exc:
            raise ValueError(f"could not inspect {label} directory: {path}") from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"unsafe {label} directory type: {path}")

    def _ensure_workspace_subdirectory(self, *parts: str) -> Path:
        current = self.workspace_dir
        for part in parts:
            if not part or part in {".", ".."} or "/" in part or "\\" in part:
                raise ValueError(f"unsafe workspace directory component: {part!r}")
            candidate = current / part
            try:
                info = os.lstat(candidate)
            except FileNotFoundError:
                candidate.mkdir(mode=0o777)
                info = os.lstat(candidate)
            except OSError as exc:
                raise ValueError(
                    f"could not inspect workspace cache directory: {candidate}"
                ) from exc
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise ValueError(f"unsafe workspace cache directory: {candidate}")
            flags = os.O_RDONLY
            flags |= getattr(os, "O_DIRECTORY", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            try:
                descriptor = os.open(candidate, flags)
                try:
                    os.fchmod(descriptor, 0o777)
                finally:
                    os.close(descriptor)
            except OSError as exc:
                raise ValueError(
                    f"could not secure workspace cache directory: {candidate}"
                ) from exc
            current = candidate
        return current

    @staticmethod
    def _write_no_follow(path: Path, content: str) -> None:
        try:
            info = os.lstat(path)
        except FileNotFoundError:
            info = None
        if info is not None and (stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode)):
            raise ValueError(f"unsafe workspace configuration file: {path}")
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags, 0o666)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(content)
        except OSError as exc:
            raise ValueError(f"could not write workspace configuration file: {path}") from exc

    def _ensure_matplotlib_config(self) -> None:
        config_dir = self._ensure_workspace_subdirectory(".cache", "matplotlib")
        for path in (self.workspace_dir / "matplotlibrc", config_dir / "matplotlibrc"):
            self._write_no_follow(path, MATPLOTLIBRC)

    def _ensure_numba_cache(self) -> None:
        # The image rootfs is read-only, while Numba-decorated scientific packages
        # such as scanpy and umap request on-disk caches during import.  Give Numba
        # an explicit directory on the writable workspace bind mount instead of
        # letting it attempt to cache beside files under /usr/local/site-packages.
        self._ensure_workspace_subdirectory(".cache", "numba")

    def _ensure_writable_tmp(self) -> None:
        # The container rootfs is read-only and the in-container /tmp is a small,
        # RAM-backed tmpfs. Point TMPDIR (set in build_docker_command) at a disk-backed
        # dir on the /workspace bind mount so large scientific temp files (Bio-Formats
        # JVM scratch, dcm2niix, tifffile memmaps, intermediate arrays) land on real
        # disk and are not capped at the tmpfs size. Created here, on the host, so the
        # bind-mounted container sees it.
        self._ensure_workspace_subdirectory(".tmp")

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
            "NUMBA_CACHE_DIR=/workspace/.cache/numba",
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
        no_new_privileges = self.config.no_new_privileges
        if no_new_privileges is None:
            no_new_privileges = os.getenv(
                "ULTRA_DEEPAGENTS_SANDBOX_NO_NEW_PRIVILEGES", "true"
            ).strip().lower() not in ("0", "false", "no")
        if no_new_privileges:
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
            docker_command.extend(
                ["--label", f"ultra.sandbox.worker={self.config.worker_id.strip()}"]
            )
        if self.config.run_id.strip():
            docker_command.extend(["--label", f"ultra.sandbox.run={self.config.run_id.strip()}"])
        docker_command.extend([self.config.image, "bash", "-lc", command])
        return docker_command

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        violation = validate_sandbox_command(command)
        if violation is not None:
            return ExecuteResponse(output=violation, exit_code=126)

        # Per-call timeout is a SHRINK-ONLY hint (see resolve_execute_timeout):
        # the operator ceiling stays absolute — extension remains impossible,
        # which is the security boundary the old ignore-both-directions
        # comment protected. Honoring the downward direction is the fix for a
        # traced failure mode: a model-authored infinite loop inside a call
        # that DECLARED timeout=120 held a sandbox at 100% CPU until killed by
        # hand, because only the multi-hour admin ceiling applied
        # (run_10bb3712, 2026-08-01). On expiry subprocess.run kills the
        # docker process (exit 124, recoverable by the model).
        timeout_seconds = resolve_execute_timeout(timeout, self.config.timeout_seconds)
        timeout_hint_applied = timeout_seconds != max(0, int(self.config.timeout_seconds))

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
                completed = _run_command_with_progress(
                    self.build_docker_command(command),
                    timeout=timeout_seconds if timeout_seconds > 0 else None,
                    source_command=command,
                    progress_callback=self._emit_execute_progress,
                )
            except FileNotFoundError:
                return ExecuteResponse(output="Docker executable not found.", exit_code=127)
            except subprocess.TimeoutExpired as exc:
                output = _combine_output(exc.stdout, exc.stderr)
                source_note = (
                    " (the timeout this call requested)" if timeout_hint_applied else ""
                )
                guidance = (
                    "The process was killed. If the work legitimately needs longer, "
                    "raise or omit the timeout parameter; if this duration should "
                    "have been enough, check for an unterminated loop or a process "
                    "waiting on input before retrying."
                )
                truncated_output, truncated = _truncate_output(
                    f"Command timed out after {timeout_seconds} seconds{source_note}. "
                    f"{guidance}\n{output}",
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

    def _emit_execute_progress(self, event: ExecuteProgressEvent) -> None:
        payload = event.to_payload()
        if self._progress_callback is not None:
            try:
                self._progress_callback(payload)
            except Exception:
                pass
        sink = current_execute_progress_sink()
        if sink is not None:
            try:
                sink(event)
            except Exception:
                pass

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


def _run_command_with_progress(
    command: list[str],
    *,
    timeout: int | float | None,
    source_command: str,
    progress_callback: Callable[[ExecuteProgressEvent], None],
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    output_queue: queue.Queue[tuple[str, str | None]] = queue.Queue()
    readers = [
        threading.Thread(
            target=_drain_process_stream,
            args=("stdout", process.stdout, output_queue),
            daemon=True,
        ),
        threading.Thread(
            target=_drain_process_stream,
            args=("stderr", process.stderr, output_queue),
            daemon=True,
        ),
    ]
    for reader in readers:
        reader.start()

    started = time.monotonic()
    deadline = started + float(timeout) if timeout and timeout > 0 else None
    active_readers = len(readers)
    output_size_chars = 0
    progress_index = 0
    last_emit_at = 0.0
    pending_line: tuple[str, str] | None = None
    pending_line_count = 0
    progress_interval = _sandbox_progress_interval_seconds()

    def emit_progress(
        stream: str,
        line: str,
        *,
        suppressed_line_count: int = 0,
    ) -> None:
        nonlocal progress_index, last_emit_at
        progress_index += 1
        last_emit_at = time.monotonic()
        progress_callback(
            ExecuteProgressEvent(
                command=source_command,
                stream=stream,
                text=_progress_line_text(line),
                elapsed_seconds=max(0.0, last_emit_at - started),
                output_size_chars=output_size_chars,
                progress_index=progress_index,
                suppressed_line_count=max(0, suppressed_line_count),
            )
        )

    def flush_pending() -> None:
        nonlocal pending_line, pending_line_count
        if pending_line is None:
            return
        stream, line = pending_line
        emit_progress(
            stream,
            line,
            suppressed_line_count=max(0, pending_line_count - 1),
        )
        pending_line = None
        pending_line_count = 0

    timed_out = False
    while active_readers > 0:
        if deadline is not None and time.monotonic() >= deadline:
            timed_out = True
            break
        queue_timeout = 0.1
        if deadline is not None:
            queue_timeout = max(0.0, min(queue_timeout, deadline - time.monotonic()))
        try:
            stream, line = output_queue.get(timeout=queue_timeout)
        except queue.Empty:
            continue
        if line is None:
            active_readers -= 1
            continue
        if stream == "stdout":
            stdout_parts.append(line)
        else:
            stderr_parts.append(line)
        output_size_chars += len(line)
        if not line.strip():
            continue
        now = time.monotonic()
        if progress_index == 0 or progress_interval <= 0 or now - last_emit_at >= progress_interval:
            flush_pending()
            emit_progress(stream, line)
        else:
            pending_line = (stream, line)
            pending_line_count += 1

    if timed_out:
        try:
            process.kill()
        except OSError:
            pass
        process.wait(timeout=5)
        _join_readers(readers)
        raise subprocess.TimeoutExpired(
            cmd=command,
            timeout=timeout,
            output="".join(stdout_parts),
            stderr="".join(stderr_parts),
        )

    process.wait(timeout=5)
    _join_readers(readers)
    flush_pending()
    return subprocess.CompletedProcess(
        command,
        process.returncode,
        stdout="".join(stdout_parts),
        stderr="".join(stderr_parts),
    )


def _drain_process_stream(
    stream_name: str,
    stream: object,
    output_queue: queue.Queue[tuple[str, str | None]],
) -> None:
    try:
        if stream is None:
            return
        for line in stream:
            output_queue.put((stream_name, str(line)))
    finally:
        output_queue.put((stream_name, None))


def _join_readers(readers: list[threading.Thread]) -> None:
    for reader in readers:
        reader.join(timeout=1)


def _sandbox_progress_interval_seconds() -> float:
    try:
        return max(
            0.0,
            float(os.getenv("ULTRA_DEEPAGENTS_SANDBOX_PROGRESS_INTERVAL_SECONDS", "5")),
        )
    except ValueError:
        return 5.0


def _sandbox_progress_max_line_chars() -> int:
    try:
        return max(
            80,
            int(os.getenv("ULTRA_DEEPAGENTS_SANDBOX_PROGRESS_MAX_LINE_CHARS", "1200")),
        )
    except ValueError:
        return 1200


def _progress_line_text(line: str) -> str:
    text = line.rstrip("\r\n")
    limit = _sandbox_progress_max_line_chars()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 24)] + "... [line truncated]"


# Floor for a model-requested execute timeout: below this, a mistyped hint
# (0/1/2s) would kill legitimate work during cold-container startup
# (interpreter launch + imports) before it begins.
_MIN_REQUESTED_TIMEOUT_SECONDS = 10


def resolve_execute_timeout(requested: object, ceiling: int) -> int:
    """Effective wall-clock bound for one execute call, in seconds (0 = none).

    The model's per-call timeout is a SHRINK-ONLY hint: the operator ceiling
    is absolute (a hint can never extend it), but a positive hint below the
    ceiling is honored, floored at _MIN_REQUESTED_TIMEOUT_SECONDS. Absent or
    unparseable hints fall back to the ceiling alone.
    """
    try:
        hint = int(requested) if requested is not None else 0
    except (TypeError, ValueError):
        hint = 0
    ceiling_seconds = max(0, int(ceiling))
    if hint <= 0:
        return ceiling_seconds
    bounded = max(hint, _MIN_REQUESTED_TIMEOUT_SECONDS)
    if ceiling_seconds > 0:
        return min(ceiling_seconds, bounded)
    return bounded


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
        normalized == root or normalized.startswith(root + "/") for root in _ALLOWED_SEARCH_ROOTS
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
        "Do not wrap sandbox commands with shell timeout. Pass the execute tool's "
        "`timeout` parameter instead — it bounds this one call and can only shorten "
        "the operator's wall-clock cap, never extend it. Long-running analysis "
        "without a timeout remains allowed."
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
