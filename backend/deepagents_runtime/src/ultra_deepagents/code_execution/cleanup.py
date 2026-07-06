from __future__ import annotations

import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

# Coarse ownership label stamped on every sandbox container (code_execution/docker.py).
# It is the multi-tenant safety boundary: the reaper ONLY ever enumerates and removes
# containers carrying this label, so postgres / megaseg / codeexec-service / sibling
# tenants on a shared daemon are invisible to it.
SANDBOX_LABEL = "ultra.sandbox=1"
_DOCKER_CALL_TIMEOUT_SECONDS = 30
_STOPPED_STATUSES = ("exited", "created", "dead")


def cleanup_expired_code_workspaces(
    *,
    root_dir: str | Path,
    retention_seconds: int,
    now_seconds: float | None = None,
) -> list[Path]:
    """Remove per-run scratch workspaces older than the retention window.

    The live layout is flat: each direct child of ``root_dir`` is a per-run
    workspace (``<workspace_root>/<run_id>``) holding scratch files — matplotlib
    output, staged uploads, and staged repos. Durable deliverables live separately
    under the artifact root, so reclaiming an expired run dir never deletes user
    outputs. Best-effort and mtime-based: an active run keeps a recent mtime and
    sits far inside any sane retention window. ``retention_seconds <= 0`` disables
    the sweep.
    """
    if retention_seconds <= 0:
        return []
    root = Path(root_dir)
    if not root.exists():
        return []

    now = time.time() if now_seconds is None else now_seconds
    expires_before = now - retention_seconds
    removed: list[Path] = []
    for run_dir in sorted(root.iterdir()):
        try:
            if not run_dir.is_dir():
                continue
            if run_dir.stat().st_mtime >= expires_before:
                continue
            shutil.rmtree(run_dir)
            removed.append(run_dir)
        except OSError:
            continue
    return removed


def reap_orphaned_sandbox_containers(
    *,
    default_cap_seconds: int,
    grace_seconds: int = 600,
    max_per_sweep: int = 200,
    docker_executable: str = "docker",
    now_seconds: float | None = None,
) -> list[str]:
    """Best-effort backstop that removes leftover/orphaned Ultra sandbox containers.

    The primary cleanup is ``docker run --rm`` (daemon-side AutoRemove, which fires on exit
    even if our docker CLI is killed). This reaper only handles the two residual leak
    classes, and ONLY ever touches containers carrying the ``ultra.sandbox=1`` label — never
    another tenant's container on a shared daemon (postgres, megaseg, codeexec-service, a
    sibling worker). The kill set is derived solely from label-filtered ``docker ps -q`` IDs;
    there is no broad ``rm``, no ``container prune``, and no name/image/status-only match.

    - STOPPED residue (exited/created/dead): removed unconditionally. A stopped
      ``ultra.sandbox`` container is by definition finished (``--rm`` should already have
      removed it; its presence means AutoRemove failed or the daemon restarted). This is the
      cheap path that directly answers the "hundreds of stopped containers" concern.
    - RUNNING orphans: force-removed only when the container's age exceeds its OWN recorded
      wall-clock cap (the ``ultra.sandbox.cap`` label) plus ``grace_seconds``. Such a
      container is provably abandoned: the ``execute()`` that launched it already timed out
      at the cap and killed its docker CLI, detaching the container. A legitimate in-flight
      job is bounded by that same cap, so it is never simultaneously running AND older than
      cap+grace — it is never reaped. This age test is worker-independent, so it also catches
      orphans left by a crashed/restarted worker (whose pid-bearing id no longer matches).
      Containers whose effective cap is <= 0 (cap disabled) are left alone — there is then no
      defensible orphan threshold.

    Never raises and never blocks indefinitely: a missing docker binary, a down daemon, or a
    container that vanished mid-sweep are all swallowed. ``max_per_sweep`` bounds total
    removals so a pathological backlog cannot make one sweep run unbounded. Returns the
    container IDs actually removed.
    """
    if max_per_sweep <= 0:
        return []
    if shutil.which(docker_executable) is None:
        return []
    now = time.time() if now_seconds is None else now_seconds
    removed: list[str] = []

    # Step A: stopped residue. Docker ORs repeated --filter status=, so one call lists all.
    status_filters: list[str] = []
    for status in _STOPPED_STATUSES:
        status_filters += ["--filter", f"status={status}"]
    for cid in _docker_ids(
        docker_executable,
        ["ps", "-a", "--filter", f"label={SANDBOX_LABEL}", *status_filters, "-q"],
    ):
        if len(removed) >= max_per_sweep:
            return removed
        if _docker_rm(docker_executable, cid, force=False):
            removed.append(cid)

    # Step B: running orphans (age > the container's own cap + grace).
    grace = max(0, grace_seconds)
    for cid in _docker_ids(
        docker_executable, ["ps", "--filter", f"label={SANDBOX_LABEL}", "-q"]
    ):
        if len(removed) >= max_per_sweep:
            return removed
        info = _inspect_started_and_cap(docker_executable, cid)
        if info is None:
            continue
        started_at, cap_label = info
        cap = _coerce_int(cap_label, default_cap_seconds)
        if cap <= 0:
            continue
        age = _age_seconds(started_at, now)
        if age is None or age <= cap + grace:
            continue
        if _docker_rm(docker_executable, cid, force=True):
            removed.append(cid)
    return removed


def kill_sandbox_containers_for_run(
    run_id: str,
    *,
    docker_executable: str = "docker",
) -> list[str]:
    """Force-remove any sandbox containers still running for ``run_id`` on THIS host.

    Called when a run's task ends on this worker (completed, cancelled, or redelivered).
    A cancelled/superseded run's ``docker run --rm`` cannot be stopped by the async task
    cancel — ``execute()`` blocks in a worker thread that Python cannot interrupt — so
    without this the container keeps grinding to the 6h ``ultra.sandbox.cap`` and, on a
    redelivered run, duplicates the whole computation on a second worker. On a clean
    completion the container has already exited (``--rm``), so this is a no-op.

    Scoped by BOTH the coarse ``ultra.sandbox=1`` tenant boundary AND the per-run
    ``ultra.sandbox.run`` label, and local to this docker daemon — so it can never touch
    a sibling tenant or another worker's copy of the same run.
    """
    run_id = (run_id or "").strip()
    if not run_id or shutil.which(docker_executable) is None:
        return []
    ids = _docker_ids(
        docker_executable,
        [
            "ps",
            "-q",
            "--filter",
            f"label={SANDBOX_LABEL}",
            "--filter",
            f"label=ultra.sandbox.run={run_id}",
        ],
    )
    removed: list[str] = []
    for cid in ids:
        if _docker_rm(docker_executable, cid, force=True):
            removed.append(cid)
    return removed


def _docker_ids(docker_executable: str, args: list[str]) -> list[str]:
    out = _run_docker(docker_executable, args)
    if out is None:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def _inspect_started_and_cap(docker_executable: str, cid: str) -> tuple[str, str] | None:
    out = _run_docker(
        docker_executable,
        [
            "inspect",
            "-f",
            '{{.State.StartedAt}}\t{{index .Config.Labels "ultra.sandbox.cap"}}',
            cid,
        ],
    )
    if out is None:
        return None
    line = out.strip()
    if not line:
        return None
    started, _, cap = line.partition("\t")
    return started.strip(), cap.strip()


def _docker_rm(docker_executable: str, cid: str, *, force: bool) -> bool:
    args = ["rm", "-f", cid] if force else ["rm", cid]
    return _run_docker(docker_executable, args) is not None


def _run_docker(docker_executable: str, args: list[str]) -> str | None:
    """Run a docker subcommand best-effort; return stdout on success, else None.

    A short explicit timeout + check=False means a wedged daemon cannot hang the sweep,
    and OSError/SubprocessError (missing binary, fork failure) degrade to None.
    """
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, never shell-interpolated
            [docker_executable, *args],
            capture_output=True,
            text=True,
            timeout=_DOCKER_CALL_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def _coerce_int(value: str, default: int) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _age_seconds(started_at: str, now: float) -> float | None:
    epoch = _parse_rfc3339(started_at)
    if epoch is None:
        return None
    return now - epoch


def _parse_rfc3339(value: str) -> float | None:
    """Parse Docker's ``State.StartedAt`` (RFC3339, up to nanosecond precision) to epoch secs.

    ``datetime`` only supports microseconds, so trim the fractional part to 6 digits and
    normalize a trailing ``Z`` to ``+00:00``. A zero/blank value (a created-but-never-started
    container reports ``0001-01-01T00:00:00Z``) or any unparseable string returns None so the
    caller skips that container rather than treating it as infinitely old.
    """
    raw = str(value or "").strip()
    if not raw or raw.startswith("0001-01-01"):
        return None
    text = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    if "." in text:
        head, frac = text.split(".", 1)
        tz = ""
        for marker in ("+", "-"):
            idx = frac.find(marker)
            if idx != -1:
                tz = frac[idx:]
                frac = frac[:idx]
                break
        text = f"{head}.{frac[:6]}{tz}"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()
