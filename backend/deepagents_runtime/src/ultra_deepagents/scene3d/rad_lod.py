"""Build and inventory Spark's paged RAD Gaussian level-of-detail artifact.

Ultra's browser budget is intentionally far below the 14.5 million Gaussians in the
reference estate.  A uniform subset leaves holes because it drops splats without
reconstructing their coverage.  Spark's quality builder instead merges nearby Gaussian
distributions into an LoD tree and lets the renderer choose nodes for the current view.

The converter is an offline worker dependency, never a request-path process.  Its input
is a private byte-for-byte staging copy.  A hard link would be cheaper, but creating and
removing one changes the source inode's ctime and therefore (correctly) trips Ultra's
source-generation fence after the long conversion.  The outer job checks the original
generation again before publication, so a concurrent source change discards this copy
and every derived byte atomically.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ultra_deepagents.imaging.derivative_manifest import TransientDerivativeError

__all__ = [
    "RAD_BUILDER_REVISION",
    "RadArtifact",
    "RadLodResult",
    "build_paged_rad",
    "builder_available",
    "is_rad_artifact_name",
]

RAD_BUILDER_REVISION = "spark-build-lod-2.1.0-f22236f"
RAD_HEADER_NAME = "scene-lod.rad"
_RAD_CHUNK_RE = re.compile(r"^scene-lod-(0|[1-9][0-9]*)\.radc$")
_DEFAULT_EXECUTABLE = "spark-build-lod"
_DEFAULT_TIMEOUT_SECONDS = 60 * 60
_MIN_TIMEOUT_SECONDS = 60
_MAX_TIMEOUT_SECONDS = 4 * 60 * 60
_COPY_BLOCK_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class RadArtifact:
    name: str
    bytes: int


@dataclass(frozen=True)
class RadLodResult:
    header: RadArtifact
    chunks: list[RadArtifact]
    method: str
    builder_revision: str


RunFn = Callable[..., Any]


def _configured_executable() -> str:
    return os.environ.get("ULTRA_SCENE3D_BUILD_LOD", _DEFAULT_EXECUTABLE).strip()


def _resolved_executable() -> str | None:
    configured = _configured_executable()
    if not configured:
        return None
    if os.path.sep in configured:
        try:
            info = os.stat(configured, follow_symlinks=False)
        except OSError:
            return None
        return configured if stat.S_ISREG(info.st_mode) and os.access(configured, os.X_OK) else None
    return shutil.which(configured)


def builder_available() -> bool:
    """Whether this worker can publish the preferred paged Gaussian representation."""

    return _resolved_executable() is not None


def is_rad_artifact_name(name: str) -> bool:
    """True only for the fixed header or a canonical, numeric RAD chunk name."""

    return name == RAD_HEADER_NAME or _RAD_CHUNK_RE.fullmatch(name) is not None


def _timeout_seconds() -> int:
    raw = os.environ.get("ULTRA_SCENE3D_BUILD_LOD_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return _DEFAULT_TIMEOUT_SECONDS
    try:
        parsed = int(raw)
    except ValueError:
        return _DEFAULT_TIMEOUT_SECONDS
    return min(_MAX_TIMEOUT_SECONDS, max(_MIN_TIMEOUT_SECONDS, parsed))


def _regular_artifact(path: Path) -> RadArtifact:
    try:
        info = path.lstat()
    except OSError as exc:
        raise TransientDerivativeError("scene_lod_output_missing") from exc
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode) or info.st_size < 1:
        raise TransientDerivativeError("scene_lod_output_invalid")
    return RadArtifact(name=path.name, bytes=info.st_size)


def _chunk_index(path: Path) -> int:
    match = _RAD_CHUNK_RE.fullmatch(path.name)
    if match is None:  # The caller filters candidates; retain a fail-closed invariant.
        raise TransientDerivativeError("scene_lod_output_invalid")
    return int(match.group(1))


def _copy_regular_source(source: Path, staged: Path) -> None:
    """Copy one no-follow regular source into exclusive private staging.

    Descriptor-based I/O avoids a second pathname resolution after validation, and the
    exclusive destination cannot follow or replace a link even if staging permissions
    are accidentally weakened later.
    """

    source_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    destination_flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    source_descriptor = os.open(source, source_flags)
    try:
        if not stat.S_ISREG(os.fstat(source_descriptor).st_mode):
            raise TransientDerivativeError("scene_lod_source_invalid")
        destination_descriptor = os.open(staged, destination_flags, 0o600)
        try:
            while block := os.read(source_descriptor, _COPY_BLOCK_BYTES):
                remaining = memoryview(block)
                while remaining:
                    written = os.write(destination_descriptor, remaining)
                    if written <= 0:
                        raise OSError("short scene LoD staging write")
                    remaining = remaining[written:]
            os.fsync(destination_descriptor)
        finally:
            os.close(destination_descriptor)
    finally:
        os.close(source_descriptor)


def build_paged_rad(
    source_path: str,
    directory: str,
    *,
    retained_sh_degree: int,
    run_fn: RunFn = subprocess.run,
) -> RadLodResult:
    """Run Spark's quality Bhattacharyya LoD builder and inventory atomic outputs.

    The returned names are the only names the control plane is allowed to serve.  The
    `.rad` header contains relative `.radc` names, so both remain fixed and source-name
    independent.
    """

    executable = _resolved_executable()
    if executable is None:
        raise TransientDerivativeError("scene_lod_builder_unavailable")

    destination = Path(directory)
    source = Path(source_path)
    staged_source = destination / "scene.ply"
    destination.mkdir(parents=True, exist_ok=True)
    if staged_source.exists() or staged_source.is_symlink():
        raise TransientDerivativeError("scene_lod_staging_conflict")

    try:
        _copy_regular_source(source, staged_source)
        command = [
            executable,
            "--quality",
            "--rad-chunked",
            f"--max-sh={max(0, min(3, int(retained_sh_degree)))}",
            staged_source.name,
        ]
        try:
            completed = run_fn(
                command,
                cwd=str(destination),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=_timeout_seconds(),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise TransientDerivativeError("scene_lod_builder_timeout") from exc
        except OSError as exc:
            raise TransientDerivativeError("scene_lod_builder_unavailable") from exc
        if int(getattr(completed, "returncode", 1)) != 0:
            raise TransientDerivativeError("scene_lod_builder_failed")
    finally:
        with suppress(OSError):
            staged_source.unlink()

    header = _regular_artifact(destination / RAD_HEADER_NAME)
    chunk_paths = sorted(
        (
            candidate
            for candidate in destination.iterdir()
            if _RAD_CHUNK_RE.fullmatch(candidate.name) is not None
        ),
        key=_chunk_index,
    )
    if not chunk_paths:
        raise TransientDerivativeError("scene_lod_output_missing")
    chunks = [_regular_artifact(path) for path in chunk_paths]
    return RadLodResult(
        header=header,
        chunks=chunks,
        method="bhatt-lod-quality",
        builder_revision=RAD_BUILDER_REVISION,
    )
