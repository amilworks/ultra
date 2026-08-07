"""Strict, source-bound publication for derived image pyramids.

The stable ``__pyramid.manifest.json`` file is the commit record.  Artifacts are
immutable, digest-named files in the same directory; a reader must never infer a
derivative from the legacy ``__pyramid.tif`` path alone.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import math
import os
import re
import secrets
import stat
import tempfile
import threading
from collections.abc import Callable
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ultra_deepagents.safe_storage import open_directory_chain_no_follow

SCHEMA_V1 = "ultra.image-derived-pyramid-manifest.v1"
SCHEMA_V2 = "ultra.image-derived-pyramid-manifest.v2"
# Reader support lands before writers advance the public contract. Keeping the
# rollout writer on v1 lets independently deployed edge readers consume every
# new artifact; forced-delivery identity lives in a private publication record.
SCHEMA = SCHEMA_V1
SUPPORTED_SCHEMAS = frozenset({SCHEMA_V1, SCHEMA_V2})
CONVERSION_CONTRACT = "ultra.image-pyramid.v1"
FORCE_RECORD_SCHEMA = "ultra.image-pyramid-force-record.v1"
PUBLICATION_JOURNAL_SCHEMA_V1 = "ultra.image-pyramid-publication-journal.v1"
PUBLICATION_JOURNAL_SCHEMA_V2 = "ultra.image-pyramid-publication-journal.v2"
MAX_MANIFEST_BYTES = 1 << 20
MAX_MANIFEST_CHANNELS = 4096
PRODUCER_REVISION = "ultra-deepagents.image-pyramid-publisher.v1"
CONVERTER_REVISION = "libbioimage.imgcnv-pyramid.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FORCE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_PYRAMID_COMPRESSIONS = frozenset({"none", "packbits", "lzw", "jpeg", "zip", "lzma", "jxr"})
_PYRAMID_LAYOUTS = frozenset({"subdirs", "topdirs"})
_PYRAMID_FORMATS = frozenset({"bigtiff", "ome-bigtiff"})
_TRANSIENT_IO_ERRNOS = frozenset(
    error
    for error in (
        errno.EAGAIN,
        errno.EBUSY,
        errno.EIO,
        errno.EMFILE,
        errno.ENFILE,
        errno.ENOMEM,
        errno.ENOSPC,
        getattr(errno, "ESTALE", None),
        getattr(errno, "ETIMEDOUT", None),
    )
    if error is not None
)

ViewerInfoFn = Callable[[str], dict[str, Any]]
ProduceFn = Callable[[str], dict[str, Any]]


class DerivativeJobError(RuntimeError):
    """Sanitized worker-visible failure with a stable, non-sensitive code."""

    def __init__(self, code: str) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", code):
            raise ValueError("invalid derivative failure code")
        self.code = code
        super().__init__(code)


class StaleDerivativeJobError(DerivativeJobError):
    """The queued catalog generation is no longer current; retire without a marker."""


class DeterministicDerivativeError(DerivativeJobError):
    """The same immutable source/spec will fail again; terminate and mark it."""


class TransientDerivativeError(DerivativeJobError):
    """A dependency or temporary runtime condition should be retried."""


def _is_transient_io_error(exc: BaseException) -> bool:
    return isinstance(exc, OSError) and exc.errno in _TRANSIENT_IO_ERRNOS


def _strict_int(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"invalid derivative manifest {field}")
    return int(value)


def _strict_string(value: Any, *, field: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ValueError(f"invalid derivative manifest {field}")
    return value


def _exact_keys(value: Any, expected: set[str], *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"invalid derivative manifest {field}")
    return dict(value)


def _publication_request(value: Any) -> dict[str, str]:
    request = _exact_keys(value, {"force_id"}, field="request")
    force_id = _strict_string(request["force_id"], field="request.force_id", allow_empty=True)
    if force_id and _FORCE_ID_RE.fullmatch(force_id) is None:
        raise ValueError("invalid derivative manifest request.force_id")
    return {"force_id": force_id}


def _sha256_file_with_identity(path: Path) -> tuple[str, dict[str, int]]:
    descriptor, identity = _open_regular_no_follow(path, label="digest input")
    try:
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        if _stat_identity(os.fstat(descriptor)) != identity:
            raise ValueError("digest input changed while hashing")
    finally:
        os.close(descriptor)
    if _source_snapshot(path, label="digest input") != identity:
        raise ValueError("digest input changed after hashing")
    return digest.hexdigest(), identity


def _sha256_file(path: Path) -> str:
    digest, _ = _sha256_file_with_identity(path)
    return digest


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate derivative manifest key {key!r}")
        result[key] = value
    return result


def _regular_file_stat(path: Path, *, label: str) -> os.stat_result:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{label} must be a regular file")
    return info


def _stat_identity(info: os.stat_result) -> dict[str, int]:
    return {
        "device": info.st_dev,
        "inode": info.st_ino,
        "size_bytes": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
    }


def _open_regular_no_follow(path: Path, *, label: str) -> tuple[int, dict[str, int]]:
    before = _regular_file_stat(path, label=label)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        if _is_transient_io_error(exc):
            raise
        raise ValueError(f"{label} must be a regular file") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or _stat_identity(opened) != _stat_identity(before):
            raise ValueError("source generation changed while opening")
        return descriptor, _stat_identity(opened)
    except Exception:
        os.close(descriptor)
        raise


def _open_source_no_follow(path: Path) -> tuple[int, dict[str, int]]:
    return _open_regular_no_follow(path, label="source image")


def _source_snapshot(path: Path, *, label: str = "source image") -> dict[str, int]:
    descriptor, opened = _open_regular_no_follow(path, label=label)
    try:
        after = _regular_file_stat(path, label=label)
        if _stat_identity(after) != opened:
            raise ValueError("source generation changed while inspecting")
        return opened
    finally:
        os.close(descriptor)


def _read_stable_regular_file(path: Path, *, label: str, max_bytes: int) -> bytes:
    descriptor, identity = _open_regular_no_follow(path, label=label)
    try:
        if identity["size_bytes"] <= 0 or identity["size_bytes"] > max_bytes:
            raise ValueError(f"{label} size is invalid")
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        if len(data) != identity["size_bytes"] or _stat_identity(os.fstat(descriptor)) != identity:
            raise ValueError(f"{label} changed while reading")
    finally:
        os.close(descriptor)
    if _source_snapshot(path, label=label) != identity:
        raise ValueError(f"{label} changed after reading")
    return data


def _read_stable_regular_file_at(
    directory: int,
    name: str,
    *,
    label: str,
    max_bytes: int,
) -> bytes:
    """Read one exact child through an already no-follow directory handle."""

    if Path(name).name != name:
        raise ValueError(f"{label} name is invalid")
    before = os.stat(name, dir_fd=directory, follow_symlinks=False)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{label} must be a regular file")
    identity = _stat_identity(before)
    if identity["size_bytes"] <= 0 or identity["size_bytes"] > max_bytes:
        raise ValueError(f"{label} size is invalid")
    descriptor = os.open(
        name,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=directory,
    )
    try:
        if _stat_identity(os.fstat(descriptor)) != identity:
            raise ValueError(f"{label} changed while opening")
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        if len(data) != identity["size_bytes"] or _stat_identity(os.fstat(descriptor)) != identity:
            raise ValueError(f"{label} changed while reading")
    finally:
        os.close(descriptor)
    after = os.stat(name, dir_fd=directory, follow_symlinks=False)
    if _stat_identity(after) != identity:
        raise ValueError(f"{label} changed after reading")
    return data


def _normalize_regular_file_permissions(path: Path, *, label: str) -> dict[str, int]:
    descriptor, _ = _open_regular_no_follow(path, label=label)
    try:
        os.fchmod(descriptor, 0o644)
        identity = _stat_identity(os.fstat(descriptor))
    finally:
        os.close(descriptor)
    if _source_snapshot(path, label=label) != identity:
        raise ValueError(f"{label} changed while setting permissions")
    return identity


def _open_digest_and_sync_regular_file(
    path: Path, *, label: str, expected_identity: dict[str, int]
) -> tuple[int, int, str]:
    descriptor, identity = _open_regular_no_follow(path, label=label)
    try:
        if identity != expected_identity:
            raise ValueError(f"{label} changed before publication")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        if _stat_identity(os.fstat(descriptor)) != identity:
            raise ValueError(f"{label} changed while preparing publication")
        os.fsync(descriptor)
        if _source_snapshot(path, label=label) != identity:
            raise ValueError(f"{label} changed before publication")
        return descriptor, identity["size_bytes"], digest.hexdigest()
    except Exception:
        os.close(descriptor)
        raise


def _same_open_file(descriptor: int, path: Path) -> bool:
    opened = os.fstat(descriptor)
    candidate = path.lstat()
    return stat.S_ISREG(candidate.st_mode) and _stat_identity(candidate) == _stat_identity(opened)


@dataclass
class _ArtifactRecovery:
    artifact_path: Path
    link_path: Path | None
    descriptor: int
    identity: dict[str, int]
    size_bytes: int
    sha256: str
    owns_descriptor: bool = False


@dataclass
class _ManifestSnapshot:
    payload: bytes
    artifact: _ArtifactRecovery


def _descriptor_sha256(descriptor: int, size_bytes: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while offset < size_bytes:
        chunk = os.pread(descriptor, min(1024 * 1024, size_bytes - offset), offset)
        if not chunk:
            break
        digest.update(chunk)
        offset += len(chunk)
    if offset != size_bytes:
        raise ValueError("artifact recovery descriptor changed while hashing")
    return digest.hexdigest()


def _artifact_descriptor_matches(recovery: _ArtifactRecovery) -> bool:
    opened = os.fstat(recovery.descriptor)
    return (
        stat.S_ISREG(opened.st_mode)
        and _stat_identity(opened) == recovery.identity
        and opened.st_size == recovery.size_bytes
        and _descriptor_sha256(recovery.descriptor, recovery.size_bytes) == recovery.sha256
        and _stat_identity(os.fstat(recovery.descriptor)) == recovery.identity
    )


def _artifact_recovery_tracks_target_quick(recovery: _ArtifactRecovery) -> bool:
    """Check the already-hashed generation without re-hashing while fenced."""

    try:
        opened = os.fstat(recovery.descriptor)
        return (
            stat.S_ISREG(opened.st_mode)
            and _stat_identity(opened) == recovery.identity
            and opened.st_size == recovery.size_bytes
            and recovery.link_path is not None
            and _same_open_file(recovery.descriptor, recovery.link_path)
            and _same_open_file(recovery.descriptor, recovery.artifact_path)
        )
    except (FileNotFoundError, ValueError):
        return False


def _recover_artifact_path_quick(recovery: _ArtifactRecovery) -> bool:
    """Restore a descriptor-retained artifact using identity already hashed earlier."""

    if recovery.link_path is None:
        return False
    try:
        opened = os.fstat(recovery.descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or _stat_identity(opened) != recovery.identity
            or opened.st_size != recovery.size_bytes
            or not _same_open_file(recovery.descriptor, recovery.link_path)
        ):
            return False
        try:
            if _same_open_file(recovery.descriptor, recovery.artifact_path):
                return True
        except (FileNotFoundError, ValueError):
            pass
        os.replace(recovery.link_path, recovery.artifact_path)
        recovery.link_path = None
        return _same_open_file(recovery.descriptor, recovery.artifact_path)
    except (FileNotFoundError, ValueError):
        return False


def _retain_artifact_recovery_link_quick(
    recovery: _ArtifactRecovery,
    *,
    link_path: Path | None = None,
) -> None:
    """Retain an already content-verified artifact while the lifecycle lock is held."""

    if recovery.link_path is not None:
        if not _artifact_recovery_tracks_target_quick(recovery):
            raise ValueError("artifact recovery link changed")
        return
    opened = os.fstat(recovery.descriptor)
    if (
        not stat.S_ISREG(opened.st_mode)
        or _stat_identity(opened) != recovery.identity
        or opened.st_size != recovery.size_bytes
        or not _same_open_file(recovery.descriptor, recovery.artifact_path)
    ):
        raise ValueError("artifact changed before recovery retention")
    candidates = (
        [link_path]
        if link_path is not None
        else [
            recovery.artifact_path.parent
            / f".{recovery.artifact_path.name}.recovery-{secrets.token_hex(12)}"
            for _attempt in range(10)
        ]
    )
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            os.link(recovery.artifact_path, candidate, follow_symlinks=False)
        except FileExistsError:
            continue
        recovery.link_path = candidate
        break
    if recovery.link_path is None:
        raise TransientDerivativeError("artifact_publication_unavailable")
    recovery.identity = _stat_identity(os.fstat(recovery.descriptor))
    if not _artifact_recovery_tracks_target_quick(recovery):
        recovery.link_path.unlink(missing_ok=True)
        recovery.link_path = None
        raise ValueError("artifact changed while retaining recovery link")


def _release_artifact_recovery(recovery: _ArtifactRecovery | None) -> None:
    if recovery is None:
        return
    if recovery.link_path is not None:
        with suppress(OSError):
            recovery.link_path.unlink(missing_ok=True)
        recovery.link_path = None
    if recovery.owns_descriptor and recovery.descriptor >= 0:
        os.close(recovery.descriptor)
        recovery.descriptor = -1


def _remove_superseded_artifact_quick(
    recovery: _ArtifactRecovery | None,
    *,
    current_artifact: Path,
) -> None:
    if recovery is None or recovery.artifact_path == current_artifact:
        return
    opened = os.fstat(recovery.descriptor)
    if (
        not stat.S_ISREG(opened.st_mode)
        or _stat_identity(opened) != recovery.identity
        or not _same_open_file(recovery.descriptor, recovery.artifact_path)
    ):
        raise ValueError("superseded derivative artifact changed before reclamation")
    recovery.artifact_path.unlink()
    _fsync_directory(recovery.artifact_path.parent)


def _refresh_retained_recovery_after_owned_replacement(recovery: _ArtifactRecovery) -> None:
    """Account for ctime changes caused by replacing one link we own.

    The descriptor content was fully hashed before the lifecycle fence. Replacing
    its public directory entry can change link count/ctime, but must not change
    its inode, size, or mtime. The private recovery link remains the authority.
    """

    if recovery.link_path is None:
        raise ValueError("artifact recovery link is unavailable")
    opened = os.fstat(recovery.descriptor)
    previous = recovery.identity
    current = _stat_identity(opened)
    for field in ("device", "inode", "size_bytes", "mtime_ns"):
        if current[field] != previous[field]:
            raise ValueError("artifact changed during owned replacement")
    if not stat.S_ISREG(opened.st_mode) or not _same_open_file(
        recovery.descriptor, recovery.link_path
    ):
        raise ValueError("artifact recovery link changed during owned replacement")
    recovery.identity = current


class _PublicationLockEntry:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.references = 0


_PUBLICATION_LOCKS_GUARD = threading.Lock()
_PUBLICATION_LOCKS: dict[str, _PublicationLockEntry] = {}
_DERIVATION_LOCKS_GUARD = threading.Lock()
_DERIVATION_LOCKS: dict[str, _PublicationLockEntry] = {}


def _derivation_lock_path(destination: Path) -> Path:
    return destination.parent.parent / ".locks" / f".{destination.stem}.work.lock"


def _publication_workspace_path(destination: Path) -> Path:
    resource_id = _publication_resource_id(destination)
    return destination.parent.parent / ".staging" / resource_id / "pyramid"


def _force_record_matches(destination: Path, force_id: str, manifest_payload: bytes) -> bool:
    resource_id = _publication_resource_id(destination)
    try:
        with open_directory_chain_no_follow(
            destination.parent.parent,
            (".staging", resource_id, "pyramid"),
        ) as workspace_directory:
            raw = _read_stable_regular_file_at(
                workspace_directory,
                "request.json",
                label="derivative force record",
                max_bytes=4096,
            ).decode("utf-8")
        decoder = json.JSONDecoder(object_pairs_hook=_reject_duplicate_pairs)
        value, end = decoder.raw_decode(raw)
        if raw[end:].strip():
            return False
        record = _exact_keys(
            value,
            {"schema", "force_id", "manifest_sha256"},
            field="force record",
        )
        return (
            record["schema"] == FORCE_RECORD_SCHEMA
            and _publication_request({"force_id": record["force_id"]})["force_id"] == force_id
            and record["manifest_sha256"] == hashlib.sha256(manifest_payload).hexdigest()
        )
    except FileNotFoundError:
        return False
    except OSError as exc:
        if _is_transient_io_error(exc):
            raise TransientDerivativeError("manifest_io_unavailable") from exc
        return False
    except (UnicodeError, ValueError, json.JSONDecodeError):
        return False


def _force_record_payload(force_id: str, manifest_sha256: str) -> bytes:
    if _SHA256_RE.fullmatch(manifest_sha256) is None:
        raise ValueError("invalid derivative force record manifest digest")
    return (
        json.dumps(
            {
                "schema": FORCE_RECORD_SCHEMA,
                "force_id": _publication_request({"force_id": force_id})["force_id"],
                "manifest_sha256": manifest_sha256,
            },
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _write_force_record_at(
    workspace_directory: int,
    force_id: str,
    manifest_sha256: str,
) -> None:
    payload = _force_record_payload(force_id, manifest_sha256)
    descriptor = os.open(
        "request.tmp",
        os.O_CREAT
        | os.O_EXCL
        | os.O_WRONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=workspace_directory,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    os.replace(
        "request.tmp",
        "request.json",
        src_dir_fd=workspace_directory,
        dst_dir_fd=workspace_directory,
    )
    os.fsync(workspace_directory)


def _write_force_record(destination: Path, force_id: str, manifest_payload: bytes) -> None:
    resource_id = _publication_resource_id(destination)
    try:
        with open_directory_chain_no_follow(
            destination.parent.parent,
            (".staging", resource_id, "pyramid"),
        ) as workspace_directory:
            _write_force_record_at(
                workspace_directory,
                force_id,
                hashlib.sha256(manifest_payload).hexdigest(),
            )
    except OSError as exc:
        raise TransientDerivativeError("manifest_io_unavailable") from exc


def _publication_artifact_basename(destination: Path, value: Any) -> str:
    basename = _strict_string(value, field="workspace artifact basename")
    pattern = re.compile(
        rf"^{re.escape(destination.stem)}\.sha256-[0-9a-f]{{64}}{re.escape(destination.suffix)}$"
    )
    if pattern.fullmatch(basename) is None or Path(basename).name != basename:
        raise ValueError("invalid publication workspace artifact basename")
    return basename


def _committed_manifest_state(
    destination: Path,
    derived_directory: int | None,
) -> tuple[str, bytes] | None:
    if derived_directory is None:
        return None
    manifest_name = destination.with_suffix(".manifest.json").name
    try:
        payload = _read_stable_regular_file_at(
            derived_directory,
            manifest_name,
            label="derivative manifest",
            max_bytes=MAX_MANIFEST_BYTES,
        )
    except FileNotFoundError:
        return None
    raw = payload.decode("utf-8")
    decoder = json.JSONDecoder(object_pairs_hook=_reject_duplicate_pairs)
    manifest, end = decoder.raw_decode(raw)
    if raw[end:].strip() or not isinstance(manifest, dict):
        raise ValueError("invalid committed derivative manifest")
    artifact = manifest.get("artifact")
    if not isinstance(artifact, dict):
        raise ValueError("invalid committed derivative manifest artifact")
    return _publication_artifact_basename(destination, artifact.get("basename")), payload


def _cleanup_publication_workspace(destination: Path) -> None:
    """Reclaim one resource's exact crash journal and private staging paths."""

    upload_root = destination.parent.parent
    resource_id = _publication_resource_id(destination)
    try:
        with ExitStack() as stack:
            workspace_directory = stack.enter_context(
                open_directory_chain_no_follow(
                    upload_root,
                    (".staging", resource_id, "pyramid"),
                )
            )
            try:
                derived_directory: int | None = stack.enter_context(
                    open_directory_chain_no_follow(upload_root, ("derived",))
                )
            except FileNotFoundError:
                derived_directory = None

            journal: dict[str, Any] | None = None
            try:
                raw = _read_stable_regular_file_at(
                    workspace_directory,
                    "journal.json",
                    label="derivative publication journal",
                    max_bytes=4096,
                ).decode("utf-8")
                decoder = json.JSONDecoder(object_pairs_hook=_reject_duplicate_pairs)
                value, end = decoder.raw_decode(raw)
                if raw[end:].strip():
                    raise ValueError("publication journal has trailing content")
                if not isinstance(value, dict):
                    raise ValueError("invalid publication journal")
                schema = value.get("schema")
                journal_keys = {
                    "schema",
                    "artifact_basename",
                    "previous_artifact_basename",
                }
                if schema == PUBLICATION_JOURNAL_SCHEMA_V2:
                    journal_keys.update({"force_id", "manifest_sha256"})
                elif schema != PUBLICATION_JOURNAL_SCHEMA_V1:
                    raise ValueError("unsupported publication journal")
                journal = _exact_keys(
                    value,
                    journal_keys,
                    field="publication journal",
                )
                journal["artifact_basename"] = _publication_artifact_basename(
                    destination, journal["artifact_basename"]
                )
                previous = _strict_string(
                    journal["previous_artifact_basename"],
                    field="publication journal previous artifact",
                    allow_empty=True,
                )
                journal["previous_artifact_basename"] = (
                    _publication_artifact_basename(destination, previous) if previous else ""
                )
                if schema == PUBLICATION_JOURNAL_SCHEMA_V2:
                    journal["force_id"] = _publication_request({"force_id": journal["force_id"]})[
                        "force_id"
                    ]
                    manifest_sha256 = _strict_string(
                        journal["manifest_sha256"],
                        field="publication journal manifest digest",
                    )
                    if _SHA256_RE.fullmatch(manifest_sha256) is None:
                        raise ValueError("invalid publication journal manifest digest")
            except FileNotFoundError:
                journal = None

            committed_state: tuple[str, bytes] | None = None
            committed: str | None = None
            removed_public = False
            if journal is not None:
                # Only a durable crash journal authorizes public-artifact
                # recovery. All public operations remain anchored to the
                # verified derived-directory descriptor.
                committed_state = _committed_manifest_state(destination, derived_directory)
                committed = committed_state[0] if committed_state is not None else None
                for key in ("artifact_basename", "previous_artifact_basename"):
                    basename = str(journal[key])
                    if not basename or basename == committed or derived_directory is None:
                        continue
                    try:
                        os.unlink(basename, dir_fd=derived_directory)
                        removed_public = True
                    except FileNotFoundError:
                        pass
            if removed_public and derived_directory is not None:
                os.fsync(derived_directory)

            # A v2 journal durably binds a forced delivery to the public
            # manifest before publication begins. If a process dies after the
            # public manifest commit but before request.json, reconstruct that
            # private replay record before retiring the journal.
            if (
                journal is not None
                and journal["schema"] == PUBLICATION_JOURNAL_SCHEMA_V2
                and bool(journal["force_id"])
                and committed_state is not None
                and committed == journal["artifact_basename"]
                and hashlib.sha256(committed_state[1]).hexdigest() == journal["manifest_sha256"]
            ):
                try:
                    os.unlink("request.tmp", dir_fd=workspace_directory)
                except FileNotFoundError:
                    pass
                _write_force_record_at(
                    workspace_directory,
                    str(journal["force_id"]),
                    str(journal["manifest_sha256"]),
                )

            for name in (
                "artifact.tif",
                "artifact.tif.transcode.ome.tif",
                "manifest.tmp",
                "publish.link",
                "recovery.link",
                "journal.tmp",
                "journal.json",
                "request.tmp",
            ):
                try:
                    os.unlink(name, dir_fd=workspace_directory)
                except FileNotFoundError:
                    pass
            os.fsync(workspace_directory)
    except FileNotFoundError:
        return
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise TransientDerivativeError("publication_recovery_unavailable") from exc


def _write_publication_journal(
    destination: Path,
    workspace: Path,
    artifact_path: Path,
    previous_artifact_path: Path | None,
    *,
    force_id: str,
    manifest_payload: bytes,
) -> None:
    payload = (
        json.dumps(
            {
                "schema": PUBLICATION_JOURNAL_SCHEMA_V2,
                "artifact_basename": _publication_artifact_basename(
                    destination, artifact_path.name
                ),
                "previous_artifact_basename": (
                    _publication_artifact_basename(destination, previous_artifact_path.name)
                    if previous_artifact_path is not None
                    else ""
                ),
                "force_id": _publication_request({"force_id": force_id})["force_id"],
                "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
            },
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    temporary = workspace / "journal.tmp"
    final = workspace / "journal.json"
    descriptor = os.open(
        temporary,
        os.O_CREAT
        | os.O_EXCL
        | os.O_WRONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    os.replace(temporary, final)
    _fsync_directory(workspace)


@contextmanager
def _derivation_singleflight(destination: Path, source: Path):
    """Serialize heavyweight work across processes without blocking deletion."""

    key = os.path.abspath(destination)
    with _DERIVATION_LOCKS_GUARD:
        entry = _DERIVATION_LOCKS.setdefault(key, _PublicationLockEntry())
        entry.references += 1
    entry.lock.acquire()
    descriptor = -1
    try:
        upload_root = destination.parent.parent
        if destination.parent != upload_root / "derived":
            raise TransientDerivativeError("publication_destination_unavailable")
        with open_directory_chain_no_follow(
            upload_root, (".locks",), create=True
        ) as lock_directory:
            lock_name = _derivation_lock_path(destination).name
            while True:
                if _resource_deleted_for_publication(destination):
                    raise StaleDerivativeJobError("resource_deleted")
                if not _source_exists_for_publication(source):
                    raise StaleDerivativeJobError("source_generation_changed")
                try:
                    descriptor = os.open(
                        lock_name,
                        os.O_CREAT
                        | os.O_RDWR
                        | getattr(os, "O_CLOEXEC", 0)
                        | getattr(os, "O_NOFOLLOW", 0),
                        0o600,
                        dir_fd=lock_directory,
                    )
                    fcntl.flock(descriptor, fcntl.LOCK_EX)
                    opened = os.fstat(descriptor)
                    visible = os.stat(lock_name, dir_fd=lock_directory, follow_symlinks=False)
                except FileNotFoundError:
                    if descriptor >= 0:
                        with suppress(OSError):
                            fcntl.flock(descriptor, fcntl.LOCK_UN)
                        os.close(descriptor)
                        descriptor = -1
                    continue
                except OSError as exc:
                    if descriptor >= 0:
                        with suppress(OSError):
                            fcntl.flock(descriptor, fcntl.LOCK_UN)
                        os.close(descriptor)
                        descriptor = -1
                    raise TransientDerivativeError("publication_lock_unavailable") from exc
                if (
                    stat.S_ISREG(opened.st_mode)
                    and stat.S_ISREG(visible.st_mode)
                    and (opened.st_dev, opened.st_ino) == (visible.st_dev, visible.st_ino)
                ):
                    break
                with suppress(OSError):
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)
                descriptor = -1
            if _resource_deleted_for_publication(destination):
                raise StaleDerivativeJobError("resource_deleted")
            if not _source_exists_for_publication(source):
                raise StaleDerivativeJobError("source_generation_changed")
            resource_id = _publication_resource_id(destination)
            with open_directory_chain_no_follow(
                upload_root,
                (".staging", resource_id, "pyramid"),
                create=True,
            ):
                workspace = _publication_workspace_path(destination)
                _cleanup_publication_workspace(destination)
                yield workspace
    except DerivativeJobError:
        raise
    except (OSError, ValueError) as exc:
        raise TransientDerivativeError("publication_lock_unavailable") from exc
    finally:
        if descriptor >= 0:
            with suppress(OSError):
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
        entry.lock.release()
        with _DERIVATION_LOCKS_GUARD:
            entry.references -= 1
            if entry.references == 0 and _DERIVATION_LOCKS.get(key) is entry:
                del _DERIVATION_LOCKS[key]


def _publication_lock_path(destination: Path) -> Path:
    return destination.parent.parent / ".locks" / f".{destination.stem}.lock"


def _publication_tombstone_paths(destination: Path) -> tuple[Path, Path]:
    suffix = "__pyramid"
    if not destination.stem.endswith(suffix):
        raise ValueError("derivative destination does not match resource identity")
    resource_id = destination.stem[: -len(suffix)]
    if not resource_id:
        raise ValueError("derivative destination does not match resource identity")
    tombstone_root = destination.parent.parent / ".tombstones"
    return (
        tombstone_root / "permanent" / resource_id,
        tombstone_root / "deleted" / resource_id,
    )


def _publication_resource_id(destination: Path) -> str:
    suffix = "__pyramid"
    if not destination.stem.endswith(suffix):
        raise ValueError("derivative destination does not match resource identity")
    resource_id = destination.stem[: -len(suffix)]
    if not resource_id or Path(resource_id).name != resource_id:
        raise ValueError("derivative destination does not match resource identity")
    return resource_id


def _resource_deleted_for_publication(destination: Path) -> bool:
    upload_root = destination.parent.parent
    resource_id = _publication_resource_id(destination)
    for marker_kind in ("permanent", "deleted"):
        try:
            with open_directory_chain_no_follow(
                upload_root, (".tombstones", marker_kind)
            ) as marker_directory:
                info = os.stat(resource_id, dir_fd=marker_directory, follow_symlinks=False)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise TransientDerivativeError("publication_tombstone_unavailable") from exc
        if not stat.S_ISREG(info.st_mode):
            raise TransientDerivativeError("publication_tombstone_invalid")
        return True
    return False


def _source_exists_for_publication(source: Path) -> bool:
    try:
        os.lstat(source)
    except FileNotFoundError:
        return False
    return True


@contextmanager
def _publication_lock(destination: Path, source: Path):
    key = os.path.abspath(destination)
    with _PUBLICATION_LOCKS_GUARD:
        entry = _PUBLICATION_LOCKS.setdefault(key, _PublicationLockEntry())
        entry.references += 1
    entry.lock.acquire()
    descriptor = -1
    try:
        lock_path = _publication_lock_path(destination)
        upload_root = destination.parent.parent
        if destination.parent != upload_root / "derived":
            raise TransientDerivativeError("publication_destination_unavailable")
        if _resource_deleted_for_publication(destination):
            raise StaleDerivativeJobError("resource_deleted")
        if not _source_exists_for_publication(source):
            raise StaleDerivativeJobError("source_generation_changed")
        try:
            try:
                with open_directory_chain_no_follow(upload_root, ("derived",)):
                    pass
            except FileNotFoundError:
                # Admission failures must not create the derivative namespace.
                # The producer creates it only after all immutable preflight.
                pass
            with open_directory_chain_no_follow(
                upload_root, (".locks",), create=True
            ) as lock_directory:
                lock_name = lock_path.name
                while True:
                    if _resource_deleted_for_publication(destination):
                        raise StaleDerivativeJobError("resource_deleted")
                    if not _source_exists_for_publication(source):
                        raise StaleDerivativeJobError("source_generation_changed")
                    try:
                        descriptor = os.open(
                            lock_name,
                            os.O_CREAT
                            | os.O_RDWR
                            | getattr(os, "O_CLOEXEC", 0)
                            | getattr(os, "O_NOFOLLOW", 0),
                            0o600,
                            dir_fd=lock_directory,
                        )
                        fcntl.flock(descriptor, fcntl.LOCK_EX)
                        opened = os.fstat(descriptor)
                        visible = os.stat(lock_name, dir_fd=lock_directory, follow_symlinks=False)
                    except FileNotFoundError:
                        if descriptor >= 0:
                            with suppress(OSError):
                                fcntl.flock(descriptor, fcntl.LOCK_UN)
                            os.close(descriptor)
                            descriptor = -1
                        continue
                    except OSError as exc:
                        if descriptor >= 0:
                            with suppress(OSError):
                                fcntl.flock(descriptor, fcntl.LOCK_UN)
                            os.close(descriptor)
                            descriptor = -1
                        raise TransientDerivativeError("publication_lock_unavailable") from exc
                    if (
                        stat.S_ISREG(opened.st_mode)
                        and stat.S_ISREG(visible.st_mode)
                        and (opened.st_dev, opened.st_ino) == (visible.st_dev, visible.st_ino)
                    ):
                        break
                    with suppress(OSError):
                        fcntl.flock(descriptor, fcntl.LOCK_UN)
                    os.close(descriptor)
                    descriptor = -1
                if _resource_deleted_for_publication(destination):
                    raise StaleDerivativeJobError("resource_deleted")
                if not _source_exists_for_publication(source):
                    raise StaleDerivativeJobError("source_generation_changed")
                yield
        except (OSError, ValueError) as exc:
            raise TransientDerivativeError("publication_lock_unavailable") from exc
    finally:
        if descriptor >= 0:
            with suppress(OSError):
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
        entry.lock.release()
        with _PUBLICATION_LOCKS_GUARD:
            entry.references -= 1
            if entry.references == 0 and _PUBLICATION_LOCKS.get(key) is entry:
                del _PUBLICATION_LOCKS[key]


def _verify_catalog_source(
    path: Path, catalog_sha256: str, catalog_size_bytes: int
) -> dict[str, int]:
    if not _SHA256_RE.fullmatch(catalog_sha256):
        raise ValueError("catalog source sha256 must be a lowercase SHA-256 digest")
    if isinstance(catalog_size_bytes, bool) or catalog_size_bytes < 0:
        raise ValueError("catalog source size must be a non-negative integer")
    try:
        descriptor, before = _open_source_no_follow(path)
    except (OSError, ValueError) as exc:
        if _is_transient_io_error(exc):
            raise TransientDerivativeError("source_io_unavailable") from exc
        raise StaleDerivativeJobError("source_generation_changed") from exc
    try:
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        opened_after = _stat_identity(os.fstat(descriptor))
    except OSError as exc:
        if _is_transient_io_error(exc):
            raise TransientDerivativeError("source_io_unavailable") from exc
        raise StaleDerivativeJobError("source_generation_changed") from exc
    finally:
        os.close(descriptor)
    try:
        path_after = _source_snapshot(path)
    except OSError as exc:
        if _is_transient_io_error(exc):
            raise TransientDerivativeError("source_io_unavailable") from exc
        raise StaleDerivativeJobError("source_generation_changed") from exc
    if before != opened_after or before != path_after:
        raise StaleDerivativeJobError("source_generation_changed")
    if path_after["size_bytes"] != catalog_size_bytes:
        raise StaleDerivativeJobError("catalog_source_size_mismatch")
    if digest.hexdigest() != catalog_sha256:
        raise StaleDerivativeJobError("catalog_source_digest_mismatch")
    return path_after


def _require_source_generation(path: Path, expected: dict[str, int]) -> None:
    try:
        current = _source_snapshot(path)
    except OSError as exc:
        if _is_transient_io_error(exc):
            raise TransientDerivativeError("source_io_unavailable") from exc
        raise StaleDerivativeJobError("source_generation_changed") from exc
    except ValueError as exc:
        raise StaleDerivativeJobError("source_generation_changed") from exc
    if current != expected:
        raise StaleDerivativeJobError("source_generation_changed")


def _require_publication_generation(
    destination: Path, source: Path, expected: dict[str, int]
) -> None:
    if _resource_deleted_for_publication(destination):
        raise StaleDerivativeJobError("resource_deleted")
    _require_source_generation(source, expected)


def _read_semantic_viewer_info(
    viewer_info_fn: ViewerInfoFn,
    path: str,
    *,
    role: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        info = dict(viewer_info_fn(path))
    except DerivativeJobError:
        raise
    except Exception as exc:
        raise TransientDerivativeError(f"{role}_viewer_info_unavailable") from exc
    if not info:
        raise TransientDerivativeError(f"{role}_viewer_info_unavailable")
    if info.get("kind") == "unsupported" or info.get("decodable") is False:
        raise DeterministicDerivativeError("unsupported_source")
    try:
        return info, _semantic_fingerprint(info)
    except ValueError as exc:
        raise TransientDerivativeError(f"{role}_viewer_info_invalid") from exc


def _semantic_fingerprint(info: dict[str, Any]) -> dict[str, Any]:
    axis_sizes_raw = _exact_keys(
        info.get("axis_sizes"), {"T", "C", "Z", "Y", "X"}, field="axis_sizes"
    )
    axis_sizes = {
        axis: _strict_int(axis_sizes_raw.get(axis), field=f"axis_sizes.{axis}", minimum=1)
        for axis in ("T", "C", "Z", "Y", "X")
    }
    dims_order = _strict_string(info.get("dims_order"), field="dims_order")
    dtype = _strict_string(info.get("dtype"), field="dtype")

    metadata_raw = info.get("metadata")
    metadata: dict[str, Any] = dict(metadata_raw) if isinstance(metadata_raw, dict) else {}
    scene_count = _strict_int(
        info.get("scene_count", metadata.get("scene_count", 1)),
        field="scene.count",
        minimum=1,
    )
    scene_id = info.get("selected_scene_id", metadata.get("selected_scene_id"))
    if scene_id is not None:
        scene_id = _strict_string(scene_id, field="scene.id", allow_empty=False)
    scene_index = info.get("selected_scene_index", metadata.get("selected_scene_index"))
    if scene_index is None:
        scene_index = 0
    scene_index = _strict_int(scene_index, field="scene.index")
    if scene_index >= scene_count:
        raise ValueError("viewer info selected scene is out of range")

    names = info.get("channel_names")
    if not isinstance(names, list) or len(names) != axis_sizes["C"]:
        raise ValueError("viewer info channel_names must match axis_sizes.C")
    channels = [
        {
            "index": index,
            "name": _strict_string(name, field=f"channels[{index}].name", allow_empty=True),
        }
        for index, name in enumerate(names)
    ]

    spacing_raw = _exact_keys(
        info.get("physical_spacing"), {"x", "y", "z"}, field="physical_spacing"
    )
    units_raw = _exact_keys(
        metadata.get("spacing_units"), {"x", "y", "z"}, field="metadata.spacing_units"
    )
    spacing: dict[str, dict[str, Any]] = {}
    for axis in ("x", "y", "z"):
        value = spacing_raw[axis]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"invalid viewer info physical_spacing.{axis}")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric <= 0:
            raise ValueError(f"invalid viewer info physical_spacing.{axis}")
        spacing[axis] = {
            "value": numeric,
            "unit": _strict_string(units_raw[axis], field=f"metadata.spacing_units.{axis}"),
        }

    viewer_raw = info.get("viewer")
    viewer: dict[str, Any] = dict(viewer_raw) if isinstance(viewer_raw, dict) else {}
    top_level_defaults = info.get("display_defaults")
    viewer_defaults = viewer.get("display_defaults")
    if isinstance(top_level_defaults, dict):
        display_defaults: dict[str, Any] = dict(top_level_defaults)
    elif isinstance(viewer_defaults, dict):
        display_defaults = dict(viewer_defaults)
    else:
        display_defaults = {}
    default_channels_raw = display_defaults.get("channels")
    if default_channels_raw is None:
        default_channels_raw = list(range(min(axis_sizes["C"], 3)))
    if not isinstance(default_channels_raw, list):
        raise ValueError("viewer info display default channels must be an array")
    default_channels: list[int] = []
    for channel in default_channels_raw:
        parsed = _strict_int(channel, field="display.default_channels")
        if parsed >= axis_sizes["C"] or parsed in default_channels:
            raise ValueError("viewer info display default channels are invalid")
        default_channels.append(parsed)

    return {
        "dims_order": dims_order,
        "axis_sizes": axis_sizes,
        "dtype": dtype,
        "scene": {"count": scene_count, "id": scene_id, "index": scene_index},
        "channels": channels,
        "spacing": spacing,
        "display": {
            "render_policy": _strict_string(
                viewer.get("render_policy", "scalar"), field="display.render_policy"
            ),
            "channel_mode": _strict_string(
                viewer.get("channel_mode", "composite" if axis_sizes["C"] > 1 else "single"),
                field="display.channel_mode",
            ),
            "default_channels": default_channels,
        },
    }


def _capabilities(info: dict[str, Any], semantics: dict[str, Any]) -> dict[str, bool]:
    viewer_raw = info.get("viewer")
    viewer: dict[str, Any] = dict(viewer_raw) if isinstance(viewer_raw, dict) else {}
    tile = isinstance(viewer.get("tile_scheme") or info.get("tile_scheme"), dict)
    atlas = semantics["axis_sizes"]["Z"] > 1 and isinstance(viewer.get("atlas_scheme"), dict)
    return {
        "atlas": atlas,
        "atlas_t": atlas,
        "lut": True,
        "ordered_channels": True,
        "slice": True,
        "thumbnail": True,
        # The embedded tile reader only has a proven selector contract for singleton
        # T/Z.  Non-singleton datasets are served via selector-aware slice/atlas paths.
        "tile": tile,
        "tile_t": tile and semantics["axis_sizes"]["T"] == 1,
        "tile_z": tile and semantics["axis_sizes"]["Z"] == 1,
    }


def _artifact_semantics_match(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Compare pixel semantics while keeping source-container scene provenance separate."""

    keys = {"dims_order", "axis_sizes", "dtype", "channels", "spacing"}
    return all(actual.get(key) == expected.get(key) for key in keys)


def _pyramid_spec(value: Any, *, field: str, allow_auto_fmt: bool = False) -> dict[str, Any]:
    spec = _exact_keys(value, {"tile_size", "compression", "layout", "fmt"}, field=field)
    compression = _strict_string(spec["compression"], field=f"{field}.compression")
    layout = _strict_string(spec["layout"], field=f"{field}.layout")
    fmt = _strict_string(spec["fmt"], field=f"{field}.fmt")
    if compression not in _PYRAMID_COMPRESSIONS:
        raise ValueError(f"invalid derivative manifest {field}.compression")
    if layout not in _PYRAMID_LAYOUTS:
        raise ValueError(f"invalid derivative manifest {field}.layout")
    supported_formats = _PYRAMID_FORMATS | ({"auto"} if allow_auto_fmt else set())
    if fmt not in supported_formats:
        raise ValueError(f"invalid derivative manifest {field}.fmt")
    return {
        "tile_size": _strict_int(spec["tile_size"], field=f"{field}.tile_size", minimum=1),
        "compression": compression,
        "layout": layout,
        "fmt": fmt,
    }


def _conversion_spec(value: Any) -> dict[str, Any]:
    spec = _exact_keys(
        value,
        {"requested", "effective", "producer_revision", "converter_revision"},
        field="conversion_spec",
    )
    producer_revision = _strict_string(
        spec["producer_revision"], field="conversion_spec.producer_revision"
    )
    converter_revision = _strict_string(
        spec["converter_revision"], field="conversion_spec.converter_revision"
    )
    if producer_revision != PRODUCER_REVISION or converter_revision != CONVERTER_REVISION:
        raise ValueError("derivative manifest conversion revision is unsupported")
    return {
        "requested": _pyramid_spec(
            spec["requested"], field="conversion_spec.requested", allow_auto_fmt=True
        ),
        "effective": _pyramid_spec(spec["effective"], field="conversion_spec.effective"),
        "producer_revision": producer_revision,
        "converter_revision": converter_revision,
    }


def _producer(value: Any) -> dict[str, Any]:
    producer = _exact_keys(
        value,
        {"reader", "series_count", "series_index", "series_name"},
        field="producer",
    )
    count = _strict_int(producer["series_count"], field="producer.series_count", minimum=1)
    index = _strict_int(producer["series_index"], field="producer.series_index")
    if index >= count:
        raise ValueError("invalid derivative manifest producer series")
    return {
        "reader": _strict_string(producer["reader"], field="producer.reader"),
        "series_count": count,
        "series_index": index,
        "series_name": _strict_string(
            producer["series_name"], field="producer.series_name", allow_empty=True
        ),
    }


def _producer_from_source(
    info: dict[str, Any], semantics: dict[str, Any], reader: str
) -> dict[str, Any]:
    scene = semantics["scene"]
    metadata_raw = info.get("metadata")
    metadata: dict[str, Any] = dict(metadata_raw) if isinstance(metadata_raw, dict) else {}
    return _producer(
        {
            "reader": reader or str(metadata.get("reader") or "unknown"),
            "series_count": scene["count"],
            "series_index": scene["index"],
            "series_name": scene["id"] or "",
        }
    )


def _validate_manifest_shape(
    manifest: Any,
    *,
    destination: Path,
    source_sha256: str,
    source_size_bytes: int,
    semantics: dict[str, Any],
    conversion_spec: dict[str, Any],
    producer: dict[str, Any],
) -> tuple[Path, dict[str, Any], dict[str, int]]:
    if not isinstance(manifest, dict):
        raise ValueError("invalid derivative manifest root")
    schema = manifest.get("schema")
    root_keys = {
        "schema",
        "conversion_contract",
        "conversion_spec",
        "producer",
        "source",
        "semantics",
        "artifact",
        "capabilities",
    }
    if schema == SCHEMA_V2:
        root_keys.add("request")
    elif schema != SCHEMA_V1:
        raise ValueError("unsupported derivative manifest contract")
    root = _exact_keys(manifest, root_keys, field="root")
    if (
        root["schema"] not in SUPPORTED_SCHEMAS
        or root["conversion_contract"] != CONVERSION_CONTRACT
    ):
        raise ValueError("unsupported derivative manifest contract")
    root["request"] = (
        _publication_request(root["request"]) if root["schema"] == SCHEMA_V2 else {"force_id": ""}
    )
    if _conversion_spec(root["conversion_spec"]) != conversion_spec:
        raise ValueError("derivative manifest conversion spec does not match request")
    if _producer(root["producer"]) != producer:
        raise ValueError("derivative manifest producer does not match source selection")

    source = _exact_keys(root["source"], {"sha256", "size_bytes"}, field="source")
    _strict_int(source["size_bytes"], field="source.size_bytes")
    if source["sha256"] != source_sha256 or source["size_bytes"] != source_size_bytes:
        raise ValueError("derivative manifest belongs to a different source generation")
    if root["semantics"] != semantics:
        raise ValueError("derivative manifest semantic fingerprint does not match source")

    artifact = _exact_keys(root["artifact"], {"basename", "size_bytes", "sha256"}, field="artifact")
    basename = _strict_string(artifact["basename"], field="artifact.basename")
    artifact_size = _strict_int(artifact["size_bytes"], field="artifact.size_bytes", minimum=1)
    artifact_sha256 = _strict_string(artifact["sha256"], field="artifact.sha256")
    if not _SHA256_RE.fullmatch(artifact_sha256):
        raise ValueError("invalid derivative manifest artifact.sha256")
    expected_basename = f"{destination.stem}.sha256-{artifact_sha256}{destination.suffix}"
    if basename != expected_basename or basename != os.path.basename(basename):
        raise ValueError("unsafe derivative manifest artifact basename")

    capabilities = _exact_keys(
        root["capabilities"],
        {
            "atlas",
            "atlas_t",
            "lut",
            "ordered_channels",
            "slice",
            "thumbnail",
            "tile",
            "tile_t",
            "tile_z",
        },
        field="capabilities",
    )
    if any(not isinstance(value, bool) for value in capabilities.values()):
        raise ValueError("invalid derivative manifest capabilities")

    artifact_path = destination.parent / basename
    artifact_stat = _regular_file_stat(artifact_path, label="derivative artifact")
    actual_sha256, artifact_identity = _sha256_file_with_identity(artifact_path)
    if (
        _stat_identity(artifact_stat) != artifact_identity
        or artifact_stat.st_size != artifact_size
        or actual_sha256 != artifact_sha256
    ):
        raise ValueError("derivative artifact does not match manifest")
    return artifact_path, root, artifact_identity


def _load_replay(
    manifest_path: Path,
    *,
    destination: Path,
    source_sha256: str,
    source_size_bytes: int,
    semantics: dict[str, Any],
    conversion_spec: dict[str, Any],
    producer: dict[str, Any],
    required_force_id: str | None = None,
) -> tuple[Path, dict[str, Any], dict[str, int]] | None:
    try:
        payload = _read_stable_regular_file(
            manifest_path,
            label="derivative manifest",
            max_bytes=MAX_MANIFEST_BYTES,
        )
        raw = payload.decode("utf-8")
        decoder = json.JSONDecoder(object_pairs_hook=_reject_duplicate_pairs)
        manifest, end = decoder.raw_decode(raw)
        if raw[end:].strip():
            raise ValueError("derivative manifest has trailing content")
        validated = _validate_manifest_shape(
            manifest,
            destination=destination,
            source_sha256=source_sha256,
            source_size_bytes=source_size_bytes,
            semantics=semantics,
            conversion_spec=conversion_spec,
            producer=producer,
        )
        if required_force_id is not None:
            if validated[1]["schema"] == SCHEMA_V2:
                if validated[1]["request"]["force_id"] != required_force_id:
                    return None
            elif not _force_record_matches(destination, required_force_id, payload):
                return None
        return validated
    except OSError as exc:
        if _is_transient_io_error(exc):
            raise TransientDerivativeError("artifact_io_unavailable") from exc
        return None
    except (UnicodeError, ValueError, json.JSONDecodeError):
        return None


def _prior_manifest_snapshot(
    manifest_path: Path,
    *,
    destination: Path,
) -> _ManifestSnapshot | None:
    """Retain a prior commit independently of replay compatibility.

    A source, reader, or conversion revision change must invalidate replay, but
    it must not orphan the artifact referenced by the prior commit. This parser
    trusts only the exact resource-bound digest basename and rehashes its bytes.
    """

    artifact_descriptor: int | None = None
    artifact_recovery: _ArtifactRecovery | None = None
    snapshot: _ManifestSnapshot | None = None
    try:
        payload = _read_stable_regular_file(
            manifest_path,
            label="derivative manifest",
            max_bytes=MAX_MANIFEST_BYTES,
        )
        raw = payload.decode("utf-8")
        decoder = json.JSONDecoder(object_pairs_hook=_reject_duplicate_pairs)
        manifest, end = decoder.raw_decode(raw)
        if raw[end:].strip():
            raise ValueError("derivative manifest has trailing content")
        if not isinstance(manifest, dict):
            raise ValueError("invalid prior derivative manifest root")
        schema = manifest.get("schema")
        root_keys = {
            "schema",
            "conversion_contract",
            "conversion_spec",
            "producer",
            "source",
            "semantics",
            "artifact",
            "capabilities",
        }
        if schema == SCHEMA_V2:
            root_keys.add("request")
        elif schema != SCHEMA_V1:
            raise ValueError("unsupported prior derivative manifest")
        root = _exact_keys(manifest, root_keys, field="prior root")
        if root["conversion_contract"] != CONVERSION_CONTRACT:
            raise ValueError("unsupported prior derivative contract")
        if schema == SCHEMA_V2:
            _publication_request(root["request"])
        source = _exact_keys(root["source"], {"sha256", "size_bytes"}, field="prior source")
        if (
            _SHA256_RE.fullmatch(_strict_string(source["sha256"], field="prior source.sha256"))
            is None
        ):
            raise ValueError("invalid prior source digest")
        _strict_int(source["size_bytes"], field="prior source.size_bytes")
        artifact = _exact_keys(
            root["artifact"],
            {"basename", "size_bytes", "sha256"},
            field="prior artifact",
        )
        artifact_size = _strict_int(
            artifact["size_bytes"], field="prior artifact.size_bytes", minimum=1
        )
        artifact_sha256 = _strict_string(artifact["sha256"], field="prior artifact.sha256")
        if _SHA256_RE.fullmatch(artifact_sha256) is None:
            raise ValueError("invalid prior artifact digest")
        basename = _strict_string(artifact["basename"], field="prior artifact.basename")
        expected_basename = f"{destination.stem}.sha256-{artifact_sha256}{destination.suffix}"
        if basename != expected_basename or Path(basename).name != basename:
            raise ValueError("unsafe prior artifact basename")
        artifact_path = destination.parent / basename
        artifact_identity = _source_snapshot(artifact_path, label="prior derivative artifact")
        artifact_descriptor, artifact_size, artifact_sha256 = _open_digest_and_sync_regular_file(
            artifact_path,
            label="prior derivative artifact",
            expected_identity=artifact_identity,
        )
        if artifact_size != artifact["size_bytes"] or artifact_sha256 != artifact["sha256"]:
            raise ValueError("prior derivative artifact changed while retaining rollback")
        artifact_recovery = _ArtifactRecovery(
            artifact_path=artifact_path,
            link_path=None,
            descriptor=artifact_descriptor,
            identity=_stat_identity(os.fstat(artifact_descriptor)),
            size_bytes=artifact_size,
            sha256=artifact_sha256,
            owns_descriptor=True,
        )
        if not _artifact_descriptor_matches(artifact_recovery) or not _same_open_file(
            artifact_descriptor, artifact_path
        ):
            raise ValueError("prior derivative artifact changed while retaining rollback")
        artifact_descriptor = None
        snapshot = _ManifestSnapshot(payload=payload, artifact=artifact_recovery)
        return snapshot
    except FileNotFoundError:
        return None
    except DerivativeJobError:
        raise
    except OSError as exc:
        if _is_transient_io_error(exc):
            raise TransientDerivativeError("manifest_io_unavailable") from exc
        return None
    except (UnicodeError, ValueError, json.JSONDecodeError):
        return None
    finally:
        if snapshot is None:
            _release_artifact_recovery(artifact_recovery)
            if artifact_descriptor is not None:
                os.close(artifact_descriptor)


def _write_manifest_snapshot(manifest_path: Path, payload: bytes) -> None:
    """Atomically write one already-validated manifest payload."""

    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{manifest_path.name}.rollback-",
        dir=manifest_path.parent,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            os.fchmod(stream.fileno(), 0o644)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, manifest_path)
    finally:
        with suppress(OSError):
            temp_path.unlink(missing_ok=True)


def _restore_manifest_snapshot_quick(
    manifest_path: Path,
    snapshot: _ManifestSnapshot | None,
    *,
    current_artifact: _ArtifactRecovery | None,
) -> None:
    """Rollback a commit using descriptors whose content was verified before fencing."""

    restored_prior = False
    if snapshot is not None:
        try:
            restored_prior = _recover_artifact_path_quick(snapshot.artifact)
            if not restored_prior:
                raise ValueError("prior artifact recovery is no longer valid")
            _write_manifest_snapshot(manifest_path, snapshot.payload)
        except (OSError, ValueError):
            restored_prior = False
    if not restored_prior:
        manifest_path.unlink(missing_ok=True)
    if current_artifact is not None and (
        snapshot is None or current_artifact.artifact_path != snapshot.artifact.artifact_path
    ):
        try:
            if _same_open_file(current_artifact.descriptor, current_artifact.artifact_path):
                current_artifact.artifact_path.unlink(missing_ok=True)
        except (FileNotFoundError, ValueError):
            pass
    _fsync_directory(manifest_path.parent)


def _fsync_directory(path: Path) -> None:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _effective_pyramid_spec(
    requested: dict[str, Any], semantics: dict[str, Any], reader: str
) -> dict[str, Any]:
    effective = dict(requested)
    axes = semantics["axis_sizes"]
    multidimensional = axes["T"] > 1 or axes["Z"] > 1
    bioio_multidimensional = multidimensional or axes["C"] > 1
    if requested["fmt"] == "auto":
        effective["fmt"] = (
            "ome-bigtiff"
            if multidimensional or (reader == "bioio" and axes["C"] > 1)
            else "bigtiff"
        )
    elif requested["fmt"] == "bigtiff" and reader == "bioio" and bioio_multidimensional:
        effective["fmt"] = "ome-bigtiff"
    return effective


def _publication_conversion_spec(
    requested: dict[str, Any], semantics: dict[str, Any], reader: str
) -> dict[str, Any]:
    return {
        "requested": requested,
        "effective": _effective_pyramid_spec(requested, semantics, reader),
        "producer_revision": PRODUCER_REVISION,
        "converter_revision": CONVERTER_REVISION,
    }


def _prospective_manifest_size(
    *,
    destination: Path,
    conversion_spec: dict[str, Any],
    producer: dict[str, Any],
    source_sha256: str,
    source_size_bytes: int,
    semantics: dict[str, Any],
    request: dict[str, str],
) -> int:
    # False is one byte longer than true and max-int64 bounds artifact-size digits,
    # so this is an upper envelope for the final payload before conversion begins.
    capabilities = {
        key: False
        for key in (
            "atlas",
            "atlas_t",
            "lut",
            "ordered_channels",
            "slice",
            "thumbnail",
            "tile",
            "tile_t",
            "tile_z",
        )
    }
    digest = "0" * 64
    prospective = {
        "schema": SCHEMA,
        "conversion_contract": CONVERSION_CONTRACT,
        "conversion_spec": conversion_spec,
        "producer": producer,
        "source": {"sha256": source_sha256, "size_bytes": source_size_bytes},
        "semantics": semantics,
        "artifact": {
            "basename": f"{destination.stem}.sha256-{digest}{destination.suffix}",
            "size_bytes": 9_223_372_036_854_775_807,
            "sha256": digest,
        },
        "capabilities": capabilities,
    }
    if SCHEMA == SCHEMA_V2:
        prospective["request"] = request
    return (
        len(
            json.dumps(
                prospective,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )
        + 1
    )


def _run_strict_publication(
    *,
    resource_id: str,
    src_path: str,
    dst_path: str,
    source_sha256: str,
    source_size_bytes: int,
    viewer_info_fn: ViewerInfoFn,
    source_viewer_info_fn: ViewerInfoFn | None,
    source_reader: str,
    conversion_spec: dict[str, Any],
    produce_fn: ProduceFn,
    workspace: Path,
    force: bool = False,
    force_id: str | None = None,
) -> dict[str, Any]:
    """Produce and atomically publish one source-bound derivative generation."""
    source = Path(src_path)
    destination = Path(dst_path)
    if destination.name != f"{resource_id}__pyramid.tif":
        raise ValueError("derivative destination does not match resource identity")
    manifest_path = destination.with_suffix(".manifest.json")
    request = _publication_request({"force_id": force_id or ""})
    try:
        requested_spec = _pyramid_spec(
            conversion_spec,
            field="conversion_spec.requested",
            allow_auto_fmt=True,
        )
    except ValueError as exc:
        raise DeterministicDerivativeError("invalid_conversion_spec") from exc
    source_stat = _verify_catalog_source(source, source_sha256, source_size_bytes)
    source_info, source_semantics = _read_semantic_viewer_info(
        source_viewer_info_fn or viewer_info_fn,
        str(source),
        role="source",
    )
    _require_publication_generation(destination, source, source_stat)
    producer = _producer_from_source(source_info, source_semantics, source_reader)
    conversion_spec = _publication_conversion_spec(
        requested_spec,
        source_semantics,
        producer["reader"],
    )
    if (
        len(source_semantics["channels"]) > MAX_MANIFEST_CHANNELS
        or _prospective_manifest_size(
            destination=destination,
            conversion_spec=conversion_spec,
            producer=producer,
            source_sha256=source_sha256,
            source_size_bytes=source_size_bytes,
            semantics=source_semantics,
            request=request,
        )
        > MAX_MANIFEST_BYTES
    ):
        raise DeterministicDerivativeError("manifest_too_large")

    replay = None
    if not force or force_id is not None:
        replay = _load_replay(
            manifest_path,
            destination=destination,
            source_sha256=source_sha256,
            source_size_bytes=source_size_bytes,
            semantics=source_semantics,
            conversion_spec=conversion_spec,
            producer=producer,
            required_force_id=(force_id if force else None),
        )
    if replay is not None:
        replay_path, replay_manifest, replay_identity = replay
        try:
            replay_info = dict(viewer_info_fn(str(replay_path)))
            replay_semantics = _semantic_fingerprint(replay_info)
            replay_capabilities = _capabilities(replay_info, replay_semantics)
        except Exception:  # Invalid replay falls through to regeneration.
            replay_semantics = None
            replay_capabilities = None
        _require_publication_generation(destination, source, source_stat)
        try:
            replay_generation_matches = (
                _source_snapshot(replay_path, label="derivative artifact") == replay_identity
            )
        except OSError as exc:
            if _is_transient_io_error(exc):
                raise TransientDerivativeError("artifact_io_unavailable") from exc
            replay_generation_matches = False
        except ValueError:
            replay_generation_matches = False
        if (
            not isinstance(replay_semantics, dict)
            or not _artifact_semantics_match(replay_semantics, source_semantics)
            or replay_capabilities != replay_manifest["capabilities"]
            or not replay_generation_matches
        ):
            replay = None
    if replay is not None:
        replay_path, _, replay_identity = replay
        with _publication_lock(destination, source):
            _require_publication_generation(destination, source, source_stat)
            try:
                replay_current = (
                    _source_snapshot(replay_path, label="derivative artifact") == replay_identity
                )
            except (FileNotFoundError, ValueError):
                replay_current = False
            if replay_current:
                return {
                    "resource_id": resource_id,
                    "derived_path": str(replay_path),
                    "manifest_path": str(manifest_path),
                    "status": "replayed",
                }

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = workspace / "artifact.tif"
    descriptor = os.open(
        temp_path,
        os.O_CREAT
        | os.O_EXCL
        | os.O_WRONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    os.close(descriptor)
    manifest_temp: Path | None = None
    artifact_descriptor: int | None = None
    artifact_recovery: _ArtifactRecovery | None = None
    previous_manifest: _ManifestSnapshot | None = None
    publication_guard: Any | None = None
    publication_locked = False
    artifact_published = False
    publication_succeeded = False
    try:
        result = produce_fn(str(temp_path))
        if result.get("status") == "skipped_native_pyramid_no_manifest":
            with _publication_lock(destination, source):
                _require_publication_generation(destination, source, source_stat)
                return result
        if result.get("source_reader") not in (None, producer["reader"]):
            raise DeterministicDerivativeError("producer_reader_mismatch")
        for result_key, producer_key in (
            ("series_count", "series_count"),
            ("series_index", "series_index"),
            ("series_name", "series_name"),
        ):
            if result_key in result and result[result_key] != producer[producer_key]:
                raise DeterministicDerivativeError("producer_scene_mismatch")
        effective_result = {
            key: result.get(key, conversion_spec["effective"][key])
            for key in ("tile_size", "compression", "layout", "fmt")
        }
        if (
            _pyramid_spec(
                effective_result,
                field="conversion_spec.effective",
            )
            != conversion_spec["effective"]
        ):
            raise DeterministicDerivativeError("effective_conversion_spec_mismatch")

        # Publication is fenced to the exact source generation observed before work.
        _require_publication_generation(destination, source, source_stat)
        if not temp_path.exists():
            raise DeterministicDerivativeError("conversion_artifact_missing")
        try:
            artifact_identity = _normalize_regular_file_permissions(
                temp_path,
                label="temporary derivative artifact",
            )
        except OSError as exc:
            if _is_transient_io_error(exc):
                raise TransientDerivativeError("artifact_io_unavailable") from exc
            raise DeterministicDerivativeError("conversion_artifact_changed") from exc
        derived_info, derived_semantics = _read_semantic_viewer_info(
            viewer_info_fn,
            str(temp_path),
            role="derived",
        )
        _require_publication_generation(destination, source, source_stat)
        try:
            artifact_matches = (
                _source_snapshot(temp_path, label="temporary derivative artifact")
                == artifact_identity
            )
        except OSError as exc:
            if _is_transient_io_error(exc):
                raise TransientDerivativeError("artifact_io_unavailable") from exc
            artifact_matches = False
        if not artifact_matches:
            raise DeterministicDerivativeError("conversion_artifact_changed")
        if not _artifact_semantics_match(derived_semantics, source_semantics):
            raise DeterministicDerivativeError("derived_semantics_mismatch")

        try:
            artifact_descriptor, artifact_size, artifact_sha256 = (
                _open_digest_and_sync_regular_file(
                    temp_path,
                    label="temporary derivative artifact",
                    expected_identity=artifact_identity,
                )
            )
        except OSError as exc:
            if _is_transient_io_error(exc):
                raise TransientDerivativeError("artifact_io_unavailable") from exc
            raise DeterministicDerivativeError("conversion_artifact_changed") from exc
        except ValueError as exc:
            raise DeterministicDerivativeError("conversion_artifact_changed") from exc
        artifact_path = destination.parent / (
            f"{destination.stem}.sha256-{artifact_sha256}{destination.suffix}"
        )
        manifest = {
            "schema": SCHEMA,
            "conversion_contract": CONVERSION_CONTRACT,
            "conversion_spec": conversion_spec,
            "producer": producer,
            "source": {
                "sha256": source_sha256,
                "size_bytes": source_size_bytes,
            },
            "semantics": source_semantics,
            "artifact": {
                "basename": artifact_path.name,
                "size_bytes": artifact_size,
                "sha256": artifact_sha256,
            },
            "capabilities": _capabilities(derived_info, derived_semantics),
        }
        if SCHEMA == SCHEMA_V2:
            manifest["request"] = request
        payload = (
            json.dumps(
                manifest, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
            ).encode("utf-8")
            + b"\n"
        )
        if len(payload) > MAX_MANIFEST_BYTES:
            raise DeterministicDerivativeError("manifest_too_large")
        previous_manifest = _prior_manifest_snapshot(
            manifest_path,
            destination=destination,
        )
        _write_publication_journal(
            destination,
            workspace,
            artifact_path,
            (previous_manifest.artifact.artifact_path if previous_manifest is not None else None),
            force_id=request["force_id"] if force else "",
            manifest_payload=payload,
        )
        manifest_temp = workspace / "manifest.tmp"
        manifest_descriptor = os.open(
            manifest_temp,
            os.O_CREAT
            | os.O_EXCL
            | os.O_WRONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            with os.fdopen(manifest_descriptor, "wb") as stream:
                os.fchmod(stream.fileno(), 0o644)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as exc:
            raise TransientDerivativeError("manifest_io_unavailable") from exc

        publication_guard = _publication_lock(destination, source)
        publication_guard.__enter__()
        publication_locked = True
        _require_publication_generation(destination, source, source_stat)
        if previous_manifest is not None:
            try:
                _retain_artifact_recovery_link_quick(
                    previous_manifest.artifact,
                    link_path=workspace / "recovery.link",
                )
            except OSError as exc:
                if _is_transient_io_error(exc):
                    raise TransientDerivativeError("artifact_io_unavailable") from exc
                raise DeterministicDerivativeError("immutable_artifact_conflict") from exc
            except ValueError as exc:
                raise DeterministicDerivativeError("immutable_artifact_conflict") from exc
        link_candidate = workspace / "publish.link"
        try:
            os.link(temp_path, link_candidate, follow_symlinks=False)
            if not _same_open_file(artifact_descriptor, link_candidate):
                raise DeterministicDerivativeError("conversion_artifact_changed")

            # The staged descriptor was content-hashed before the lifecycle lock.
            # Replace any same-name entry with that exact inode rather than hashing
            # an existing artifact while deletion is waiting on this fence.
            os.replace(link_candidate, artifact_path)
            artifact_published = True
            if (
                previous_manifest is not None
                and previous_manifest.artifact.artifact_path == artifact_path
            ):
                _refresh_retained_recovery_after_owned_replacement(previous_manifest.artifact)
            if not _same_open_file(artifact_descriptor, artifact_path):
                raise DeterministicDerivativeError("immutable_artifact_conflict")
        finally:
            link_candidate.unlink(missing_ok=True)
        try:
            _fsync_directory(destination.parent)
        except OSError as exc:
            raise TransientDerivativeError("artifact_io_unavailable") from exc
        artifact_recovery = _ArtifactRecovery(
            artifact_path=artifact_path,
            link_path=temp_path,
            descriptor=artifact_descriptor,
            identity=_stat_identity(os.fstat(artifact_descriptor)),
            size_bytes=artifact_size,
            sha256=artifact_sha256,
        )
        if not _artifact_recovery_tracks_target_quick(artifact_recovery):
            raise DeterministicDerivativeError("conversion_artifact_changed")

        _require_publication_generation(destination, source, source_stat)
        try:
            artifact_valid = (
                artifact_recovery is not None
                and _artifact_recovery_tracks_target_quick(artifact_recovery)
            )
        except OSError as exc:
            if _is_transient_io_error(exc):
                raise TransientDerivativeError("artifact_io_unavailable") from exc
            artifact_valid = False
        if not artifact_valid:
            raise DeterministicDerivativeError("conversion_artifact_changed")
        try:
            # The manifest is the commit record and therefore the final rename.
            os.replace(manifest_temp, manifest_path)
            manifest_temp = None
            _require_publication_generation(destination, source, source_stat)
            try:
                artifact_valid = (
                    artifact_recovery is not None
                    and _artifact_recovery_tracks_target_quick(artifact_recovery)
                )
            except OSError as exc:
                raise TransientDerivativeError("artifact_io_unavailable") from exc
            if not artifact_valid:
                raise TransientDerivativeError("artifact_publication_changed")
            _fsync_directory(destination.parent)
            # Keep rollout publications on the public v1 contract. The private
            # record is committed only after the manifest is durable, so it can
            # never authorize replay of an older manifest generation.
            _write_force_record(
                destination,
                request["force_id"] if force else "",
                payload,
            )
        except (DerivativeJobError, OSError, ValueError) as exc:
            if isinstance(exc, DerivativeJobError):
                raise
            if isinstance(exc, OSError) and _is_transient_io_error(exc):
                raise TransientDerivativeError("manifest_io_unavailable") from exc
            raise DeterministicDerivativeError("conversion_artifact_changed") from exc

        publication_succeeded = True
        try:
            _remove_superseded_artifact_quick(
                previous_manifest.artifact if previous_manifest is not None else None,
                current_artifact=artifact_path,
            )
        except OSError as exc:
            raise TransientDerivativeError("artifact_reclamation_unavailable") from exc
        except ValueError as exc:
            raise DeterministicDerivativeError("immutable_artifact_conflict") from exc
        if publication_guard is not None:
            publication_guard.__exit__(None, None, None)
            publication_locked = False

        _cleanup_publication_workspace(destination)

        result.update(
            {
                "derived_path": str(artifact_path),
                "manifest_path": str(manifest_path),
                "status": "succeeded",
            }
        )
        return result
    finally:
        rollback_error: BaseException | None = None
        if publication_locked and publication_guard is not None:
            try:
                if artifact_published and not publication_succeeded:
                    _restore_manifest_snapshot_quick(
                        manifest_path,
                        previous_manifest,
                        current_artifact=artifact_recovery,
                    )
            except OSError as exc:
                rollback_error = TransientDerivativeError("manifest_rollback_unavailable")
                rollback_error.__cause__ = exc
            finally:
                publication_guard.__exit__(None, None, None)
                publication_locked = False
        if previous_manifest is not None:
            _release_artifact_recovery(previous_manifest.artifact)
        _release_artifact_recovery(artifact_recovery)
        if artifact_descriptor is not None:
            os.close(artifact_descriptor)
        with suppress(OSError):
            temp_path.unlink(missing_ok=True)
        if manifest_temp is not None:
            with suppress(OSError):
                manifest_temp.unlink(missing_ok=True)
        with suppress(DerivativeJobError, OSError, ValueError):
            _cleanup_publication_workspace(destination)
        if rollback_error is not None:
            raise rollback_error


def run_strict_publication(
    *,
    resource_id: str,
    src_path: str,
    dst_path: str,
    source_sha256: str,
    source_size_bytes: int,
    viewer_info_fn: ViewerInfoFn,
    source_viewer_info_fn: ViewerInfoFn | None,
    source_reader: str,
    conversion_spec: dict[str, Any],
    produce_fn: ProduceFn,
    force: bool = False,
    force_id: str | None = None,
) -> dict[str, Any]:
    """Serialize work cross-process; fence only replay and final publication."""

    destination = Path(dst_path)
    source = Path(src_path)
    with _derivation_singleflight(destination, source) as workspace:
        return _run_strict_publication(
            resource_id=resource_id,
            src_path=src_path,
            dst_path=dst_path,
            source_sha256=source_sha256,
            source_size_bytes=source_size_bytes,
            viewer_info_fn=viewer_info_fn,
            source_viewer_info_fn=source_viewer_info_fn,
            source_reader=source_reader,
            conversion_spec=conversion_spec,
            produce_fn=produce_fn,
            workspace=workspace,
            force=force,
            force_id=force_id,
        )
