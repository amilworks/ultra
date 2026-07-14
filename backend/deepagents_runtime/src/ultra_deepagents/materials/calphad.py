"""Bounded, provenance-preserving CALPHAD database and equilibrium runtime.

This module intentionally exposes a small surface around :mod:`pycalphad`.
Thermodynamic databases remain immutable files; callers must supply provenance
and use authorization, or select an entry from a content-addressed Ultra
manifest.  Database inspection and numerical execution are both fail closed.
"""

from __future__ import annotations

import hashlib
import io
import itertools
import json
import math
import os
import re
import signal
import stat
import threading
import warnings
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

DATABASE_MANIFEST_SCHEMA_VERSION = "1"
EQUILIBRIUM_SCHEMA_VERSION = "ultra.calphad.equilibrium.v2"

# Registry metadata is authoritative only when its complete bytes match a
# release-reviewed embedded manifest. A nearby user-authored manifest must never
# turn caller-written provenance into a verified catalog or bypass resource
# hash/identity binding.
TRUSTED_EMBEDDED_MANIFEST_SHA256S = frozenset(
    {"a5dfb3aac68f119a8fe0ee751255a16f31fe8e8515cfa9739320ba21aa28fb09"}
)

MAX_DATABASE_BYTES = 64 * 1024 * 1024
MAX_DATABASE_ELEMENTS = 256
MAX_DATABASE_SPECIES = 4096
MAX_DATABASE_PHASES = 2048
MAX_DATABASE_PARAMETERS = 1_000_000
MAX_TDB_STATEMENTS = 1_500_000
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_MANIFEST_DATABASES = 128
MAX_INSPECTION_JSON_BYTES = 16 * 1024 * 1024
MAX_REFERENCES = 512
MAX_REFERENCE_TEXT_CHARS = 2048
MAX_SUBLATTICES_PER_PHASE = 64
MAX_CONSTITUENTS_PER_SUBLATTICE = 2048
MAX_MODEL_HINTS_PER_PHASE = 128

MAX_REQUEST_COMPONENTS = 32
MAX_REQUEST_PHASES = 128
MAX_AXIS_VALUES = 256
MAX_GRID_POINTS = 4096
MAX_RESULT_BYTES = 64 * 1024 * 1024
DEFAULT_RESULT_BYTES = 16 * 1024 * 1024
DATABASE_PARSE_WALL_TIME_SECONDS = 15.0
MAX_WALL_TIME_SECONDS = 60.0
DEFAULT_WALL_TIME_SECONDS = 30.0
MIN_WALL_TIME_SECONDS = 0.01
PHASE_FRACTION_TOLERANCE = 1e-6
STABLE_PHASE_THRESHOLD = 1e-12
COMPOSITION_TOLERANCE = 1e-12
# Canonical composition coordinates are rounded three decimal orders more
# finely than the closure tolerance. This removes binary floating-point noise
# introduced while changing the dependent component without changing a
# scientifically distinguishable coordinate under this runtime contract.
COMPOSITION_CANONICAL_DECIMAL_PLACES = 15
BULK_COMPOSITION_TOLERANCE = 1e-8
GIBBS_EULER_ABSOLUTE_TOLERANCE_J_PER_MOL = 1e-3

_FIXTURE_DIRECTORY_MARKERS = {
    "benchmarks",
    "examples",
    "fixtures",
    "test_data",
    "testdata",
    "tests",
    "tutorials",
}
_CALPHAD_DATABASE_SUFFIXES = {".dat", ".tdb"}
_MIN_PRESSURE_PA = 1e-9
_MAX_PRESSURE_PA = 1e12
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_NONFINITE_TDB_TOKEN = re.compile(r"(?i)(?<![A-Z0-9_])(?:NAN|[+-]?INF(?:INITY)?)(?![A-Z0-9_])")
_UNUSABLE_PROVENANCE_VALUES = {
    "n/a",
    "na",
    "none",
    "not applicable",
    "pending",
    "tbd",
    "unknown",
    "unspecified",
}


class CalphadInputError(ValueError):
    """The requested database or thermodynamic domain is not usable."""


class CalphadUnsupportedError(CalphadInputError):
    """The bounded runtime recognizes but cannot execute a database construct."""


class CalphadExecutionError(RuntimeError):
    """The bounded equilibrium calculation did not produce valid evidence."""


class CalphadTimeoutError(CalphadExecutionError):
    """The equilibrium calculation exceeded its declared wall-time limit."""


@dataclass(frozen=True)
class _DatabaseFile:
    path: Path
    data: bytes
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class _ManifestSelection:
    manifest_path: Path
    manifest_sha256: str
    manifest_size_bytes: int
    entry: Mapping[str, Any]
    database_path: Path


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CalphadExecutionError("CALPHAD evidence is not finite JSON") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_regular_file(
    path: str | Path,
    *,
    max_bytes: int,
    kind: str,
) -> _DatabaseFile:
    """Read one bounded regular file without following a final symlink."""

    candidate = Path(path).expanduser()
    try:
        before = candidate.lstat()
    except OSError as exc:
        raise CalphadInputError(f"{kind} does not exist or is unreadable: {candidate}") from exc
    if stat.S_ISLNK(before.st_mode):
        raise CalphadInputError(f"{kind} must not be a symbolic link: {candidate}")
    if not stat.S_ISREG(before.st_mode):
        raise CalphadInputError(f"{kind} must be a regular file: {candidate}")
    if before.st_size <= 0:
        raise CalphadInputError(f"{kind} must not be empty")
    if before.st_size > max_bytes:
        raise CalphadInputError(
            f"{kind} exceeds the {max_bytes}-byte input limit: {before.st_size}"
        )

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(candidate, flags)
    except OSError as exc:
        raise CalphadInputError(f"could not securely open {kind}: {candidate}") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise CalphadInputError(f"{kind} must be a regular file")
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise CalphadInputError(f"{kind} changed while it was being opened")
        if opened.st_size <= 0 or opened.st_size > max_bytes:
            raise CalphadInputError(f"{kind} size changed outside its allowed bounds")
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise CalphadInputError(f"{kind} was truncated while it was read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise CalphadInputError(f"{kind} grew while it was read")
        after = os.fstat(descriptor)
        if (after.st_size, after.st_mtime_ns) != (opened.st_size, opened.st_mtime_ns):
            raise CalphadInputError(f"{kind} changed while it was read")
    finally:
        os.close(descriptor)

    data = b"".join(chunks)
    return _DatabaseFile(
        path=Path(os.path.realpath(candidate)),
        data=data,
        sha256=_sha256_bytes(data),
        size_bytes=len(data),
    )


def _clean_text(
    value: Any,
    *,
    field_name: str,
    required: bool,
    max_chars: int = 4096,
) -> str:
    text = str(value or "").strip()
    if required and not text:
        raise CalphadInputError(f"{field_name} is required")
    if len(text) > max_chars:
        raise CalphadInputError(f"{field_name} exceeds {max_chars} characters")
    if any(ord(character) < 32 or ord(character) == 127 for character in text):
        raise CalphadInputError(f"{field_name} contains control characters")
    return text


def _clean_name(value: Any, *, field_name: str) -> str:
    name = _clean_text(value, field_name=field_name, required=True, max_chars=128).upper()
    if any(character.isspace() for character in name):
        raise CalphadInputError(f"{field_name} must not contain whitespace")
    return name


def _clean_provenance(
    value: Any,
    *,
    field_name: str,
    max_chars: int = 4096,
) -> str:
    text = _clean_text(
        value,
        field_name=field_name,
        required=True,
        max_chars=max_chars,
    )
    if text.casefold() in _UNUSABLE_PROVENANCE_VALUES:
        raise CalphadInputError(f"{field_name} must be an explicit usable declaration")
    return text


def _temperature_limits(
    value: Iterable[float] | None,
    *,
    field_name: str,
) -> tuple[float, float] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes, bytearray)):
        raise CalphadInputError(f"{field_name} must contain exactly two finite values")
    try:
        values = list(itertools.islice(iter(value), 3))
    except TypeError as exc:
        raise CalphadInputError(f"{field_name} must contain exactly two finite values") from exc
    if len(values) != 2:
        raise CalphadInputError(f"{field_name} must contain exactly two finite values")
    normalized: list[float] = []
    for raw in values:
        if isinstance(raw, bool):
            raise CalphadInputError(f"{field_name} must contain finite values")
        try:
            number = float(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise CalphadInputError(f"{field_name} must contain finite values") from exc
        if not math.isfinite(number) or number < 1.0 or number > 10_000.0:
            raise CalphadInputError(f"{field_name} values must be finite and within [1, 10000] K")
        normalized.append(number)
    if normalized[0] >= normalized[1]:
        raise CalphadInputError(f"{field_name} must be strictly increasing")
    return normalized[0], normalized[1]


def _pressure_limits(
    value: Iterable[float] | None,
    *,
    field_name: str,
) -> tuple[float, float] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes, bytearray)):
        raise CalphadInputError(f"{field_name} must contain exactly two finite values")
    try:
        values = list(itertools.islice(iter(value), 3))
    except TypeError as exc:
        raise CalphadInputError(f"{field_name} must contain exactly two finite values") from exc
    if len(values) != 2:
        raise CalphadInputError(f"{field_name} must contain exactly two finite values")
    normalized: list[float] = []
    for raw in values:
        if isinstance(raw, bool):
            raise CalphadInputError(f"{field_name} must contain finite values")
        try:
            number = float(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise CalphadInputError(f"{field_name} must contain finite values") from exc
        if not math.isfinite(number) or number < _MIN_PRESSURE_PA or number > _MAX_PRESSURE_PA:
            raise CalphadInputError(
                f"{field_name} values must be finite and within "
                f"[{_MIN_PRESSURE_PA}, {_MAX_PRESSURE_PA}] Pa"
            )
        normalized.append(number)
    if normalized[0] > normalized[1]:
        raise CalphadInputError(f"{field_name} must be nondecreasing")
    return normalized[0], normalized[1]


def _required_names(
    values: Iterable[str],
    *,
    field_name: str,
    max_count: int,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise CalphadInputError(f"{field_name} must be an explicit list of names")
    names: list[str] = []
    try:
        iterator = iter(values)
    except TypeError as exc:
        raise CalphadInputError(f"{field_name} must be an explicit list of names") from exc
    for index, value in enumerate(iterator):
        if index >= max_count:
            raise CalphadInputError(f"{field_name} exceeds the {max_count}-name limit")
        names.append(_clean_name(value, field_name=f"{field_name}[{index}]"))
    if not names:
        raise CalphadInputError(f"at least one explicit {field_name[:-1]} is required")
    if len(set(names)) != len(names):
        raise CalphadInputError(f"{field_name} contains duplicate names")
    return tuple(sorted(names))


def _path_has_fixture_marker(path: Path) -> bool:
    lowered = {part.lower().replace("-", "_") for part in path.parts}
    return bool(lowered.intersection(_FIXTURE_DIRECTORY_MARKERS))


@lru_cache(maxsize=8)
def _installed_fixture_hashes(package_root_value: str) -> frozenset[str]:
    """Hash bounded bundled pycalphad fixtures so renamed copies stay denied."""

    package_root = Path(package_root_value)
    hashes: set[str] = set()
    visited = 0
    try:
        candidates: Iterator[Path] = package_root.rglob("*")
        for candidate in candidates:
            visited += 1
            if visited > 10_000:
                break
            try:
                info = candidate.lstat()
                if (
                    stat.S_ISREG(info.st_mode)
                    and candidate.suffix.casefold() in _CALPHAD_DATABASE_SUFFIXES
                    and _path_has_fixture_marker(candidate.relative_to(package_root))
                    and 0 < info.st_size <= MAX_DATABASE_BYTES
                ):
                    fixture = _read_regular_file(
                        candidate,
                        max_bytes=MAX_DATABASE_BYTES,
                        kind="pycalphad fixture",
                    )
                    hashes.add(fixture.sha256)
            except (CalphadInputError, OSError, ValueError):
                continue
    except OSError:
        return frozenset()
    return frozenset(hashes)


def _is_fixture_database(path: Path, sha256: str) -> bool:
    import pycalphad

    package_root = Path(pycalphad.__file__).resolve().parent
    try:
        relative = path.resolve().relative_to(package_root)
    except (OSError, ValueError):
        relative = None
    if relative is not None and _path_has_fixture_marker(relative):
        return True
    return sha256 in _installed_fixture_hashes(str(package_root))


def is_pycalphad_test_database(path: str | Path) -> bool:
    """Return true for bundled fixtures or byte-identical renamed copies."""

    candidate = Path(path).expanduser()
    try:
        import pycalphad

        package_root = Path(pycalphad.__file__).resolve().parent
        resolved = candidate.resolve()
        try:
            relative = resolved.relative_to(package_root)
        except ValueError:
            relative = None
        if relative is not None and _path_has_fixture_marker(relative):
            return True
        file_info = _read_regular_file(
            candidate,
            max_bytes=MAX_DATABASE_BYTES,
            kind="CALPHAD database",
        )
    except (CalphadInputError, OSError):
        return False
    return file_info.sha256 in _installed_fixture_hashes(str(package_root))


def _reject_duplicate_json_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CalphadInputError(f"CALPHAD manifest contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise CalphadInputError(f"CALPHAD manifest contains non-finite value {value!r}")


def _load_manifest(path: Path) -> tuple[_DatabaseFile, Mapping[str, Any]]:
    manifest_file = _read_regular_file(
        path,
        max_bytes=MAX_MANIFEST_BYTES,
        kind="CALPHAD manifest",
    )
    try:
        parsed = json.loads(
            manifest_file.data.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CalphadInputError("CALPHAD manifest is not valid UTF-8 JSON") from exc
    if not isinstance(parsed, Mapping):
        raise CalphadInputError("CALPHAD manifest root must be an object")
    if str(parsed.get("schema_version") or "") != DATABASE_MANIFEST_SCHEMA_VERSION:
        raise CalphadInputError("unsupported CALPHAD manifest schema_version")
    entries = parsed.get("databases")
    if not isinstance(entries, list) or not entries:
        raise CalphadInputError("CALPHAD manifest requires a non-empty databases list")
    if len(entries) > MAX_MANIFEST_DATABASES:
        raise CalphadInputError(
            f"CALPHAD manifest exceeds the {MAX_MANIFEST_DATABASES}-database limit"
        )
    if not all(isinstance(entry, Mapping) for entry in entries):
        raise CalphadInputError("CALPHAD manifest database entries must be objects")
    return manifest_file, parsed


def _manifest_selection(
    path: str | Path,
    *,
    database_id: str,
) -> _ManifestSelection | None:
    candidate = Path(path).expanduser()
    try:
        candidate_info = candidate.lstat()
    except OSError:
        candidate_info = None
    if candidate_info is not None and stat.S_ISLNK(candidate_info.st_mode):
        raise CalphadInputError(f"CALPHAD input must not be a symbolic link: {candidate}")
    manifest_path: Path | None = None
    selected_filename: str | None = None
    explicitly_selected_manifest = False
    if candidate.is_dir():
        manifest_path = candidate / "manifest.json"
        explicitly_selected_manifest = True
    elif candidate.name.casefold() == "manifest.json":
        manifest_path = candidate
        explicitly_selected_manifest = True
    elif candidate.suffix.casefold() in _CALPHAD_DATABASE_SUFFIXES:
        sibling = candidate.parent / "manifest.json"
        if sibling.exists():
            manifest_path = sibling
            selected_filename = candidate.name
    if manifest_path is None:
        return None

    manifest_file, parsed = _load_manifest(manifest_path)
    entries = list(parsed["databases"])
    clean_database_id = _clean_text(
        database_id,
        field_name="database_id",
        required=False,
        max_chars=128,
    )
    matching: list[Mapping[str, Any]] = []
    for entry in entries:
        entry_id = str(entry.get("database_id") or "").strip()
        filename = str(entry.get("filename") or "").strip()
        if clean_database_id and entry_id != clean_database_id:
            continue
        if selected_filename and filename != selected_filename:
            continue
        matching.append(entry)
    if not matching:
        if explicitly_selected_manifest or clean_database_id:
            raise CalphadInputError("requested CALPHAD database is absent from the manifest")
        return None
    if len(matching) != 1:
        raise CalphadInputError("database_id is required when a manifest has multiple matches")
    entry = matching[0]
    entry_id = _clean_text(
        entry.get("database_id"),
        field_name="manifest database_id",
        required=True,
        max_chars=128,
    )
    filename = _clean_text(
        entry.get("filename"),
        field_name=f"manifest {entry_id} filename",
        required=True,
        max_chars=255,
    )
    if Path(filename).name != filename or filename in {".", ".."}:
        raise CalphadInputError("CALPHAD manifest filename must be a basename")
    if manifest_file.sha256 not in TRUSTED_EMBEDDED_MANIFEST_SHA256S:
        raise CalphadInputError(
            "CALPHAD manifest is not a release-trusted embedded registry; "
            "stage the TDB as a catalog-bound standalone resource"
        )
    database_path = manifest_file.path.parent / filename
    return _ManifestSelection(
        manifest_path=manifest_file.path,
        manifest_sha256=manifest_file.sha256,
        manifest_size_bytes=manifest_file.size_bytes,
        entry=entry,
        database_path=database_path,
    )


def _manifest_string_list(
    value: Any,
    *,
    field_name: str,
    max_count: int,
    max_chars: int = 512,
) -> list[str]:
    if not isinstance(value, list):
        raise CalphadInputError(f"manifest {field_name} must be a list")
    if len(value) > max_count:
        raise CalphadInputError(f"manifest {field_name} exceeds {max_count} entries")
    return [
        _clean_text(
            item,
            field_name=f"manifest {field_name}[{index}]",
            required=True,
            max_chars=max_chars,
        )
        for index, item in enumerate(value)
    ]


def _verified_manifest_metadata(
    selection: _ManifestSelection,
    database_file: _DatabaseFile,
) -> dict[str, Any]:
    entry = selection.entry
    entry_id = _clean_text(
        entry.get("database_id"),
        field_name="manifest database_id",
        required=True,
        max_chars=128,
    )
    expected_digest = str(entry.get("sha256") or "").strip().lower()
    if not _SHA256_PATTERN.fullmatch(expected_digest):
        raise CalphadInputError(f"manifest {entry_id} has an invalid sha256")
    expected_size = entry.get("size_bytes")
    if isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size <= 0:
        raise CalphadInputError(f"manifest {entry_id} has an invalid size_bytes")
    if database_file.sha256 != expected_digest or database_file.size_bytes != expected_size:
        raise CalphadInputError(
            f"CALPHAD database does not match manifest {entry_id}: hash/size mismatch"
        )
    declared_format = _clean_text(
        entry.get("format"),
        field_name=f"manifest {entry_id} format",
        required=True,
        max_chars=16,
    ).casefold()
    database_format = _database_format(database_file.path.suffix)
    if declared_format != database_format:
        raise CalphadInputError(f"manifest {entry_id} format does not match the database filename")

    source_uri = _clean_provenance(
        entry.get("source_uri"),
        field_name=f"manifest {entry_id} source_uri",
    )
    license_id = _clean_provenance(
        entry.get("license_id"),
        field_name=f"manifest {entry_id} license_id",
        max_chars=256,
    )
    declared_elements = _manifest_string_list(
        entry.get("elements"), field_name="elements", max_count=MAX_DATABASE_ELEMENTS
    )
    declared_phases = _manifest_string_list(
        entry.get("phases"), field_name="phases", max_count=MAX_DATABASE_PHASES
    )
    caveats = _manifest_string_list(
        entry.get("caveats", []), field_name="caveats", max_count=64, max_chars=2048
    )
    parsed_temperature_limits = _temperature_limits(
        entry.get("tdb_temperature_limits_K"),
        field_name="manifest tdb_temperature_limits_K",
    )
    temperature_limits = (
        list(parsed_temperature_limits) if parsed_temperature_limits is not None else None
    )
    parsed_pressure_limits = _pressure_limits(
        entry.get("assessment_pressure_limits_Pa"),
        field_name="manifest assessment_pressure_limits_Pa",
    )
    if parsed_pressure_limits is None:
        raise CalphadInputError(f"manifest {entry_id} requires assessment_pressure_limits_Pa")
    pressure_limits = list(parsed_pressure_limits)

    source_sha256 = _clean_text(
        entry.get("source_sha256"),
        field_name="manifest source_sha256",
        required=False,
        max_chars=64,
    )
    if source_sha256 and not _SHA256_PATTERN.fullmatch(source_sha256.lower()):
        raise CalphadInputError("manifest source_sha256 is invalid")

    return {
        "database_id": entry_id,
        "format": database_format,
        "manifest_path": str(selection.manifest_path),
        "manifest_sha256": selection.manifest_sha256,
        "manifest_size_bytes": selection.manifest_size_bytes,
        "source_uri": source_uri,
        "source_sha256": source_sha256.lower() or None,
        "license_id": license_id,
        "license_uri": _clean_text(
            entry.get("license_uri"),
            field_name="manifest license_uri",
            required=False,
        )
        or None,
        "title": _clean_text(
            entry.get("title"), field_name="manifest title", required=False, max_chars=1024
        )
        or None,
        "publication_doi": _clean_text(
            entry.get("publication_doi"),
            field_name="manifest publication_doi",
            required=False,
            max_chars=512,
        )
        or None,
        "normalization": _clean_text(
            entry.get("normalization"),
            field_name="manifest normalization",
            required=False,
            max_chars=2048,
        )
        or None,
        "assessment_scope": _clean_text(
            entry.get("assessment_scope"),
            field_name="manifest assessment_scope",
            required=False,
            max_chars=4096,
        )
        or None,
        "reference_state": _clean_text(
            entry.get("reference_state"),
            field_name="manifest reference_state",
            required=False,
            max_chars=1024,
        )
        or None,
        "temperature_limits_K": temperature_limits,
        "assessment_pressure_limits_Pa": pressure_limits,
        "declared_elements": sorted(
            _clean_name(value, field_name="manifest element") for value in declared_elements
        ),
        "declared_phases": sorted(
            _clean_name(value, field_name="manifest phase") for value in declared_phases
        ),
        "caveats": caveats,
    }


def _decode_database_text(database_file: _DatabaseFile) -> str:
    try:
        text = database_file.data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise CalphadInputError("CALPHAD database must be valid UTF-8 text") from exc
    if "\x00" in text:
        raise CalphadInputError("CALPHAD database contains NUL bytes")
    if _NONFINITE_TDB_TOKEN.search(text):
        raise CalphadInputError("CALPHAD database contains a non-finite numeric token")
    if text.count("!") > MAX_TDB_STATEMENTS:
        raise CalphadInputError(
            f"CALPHAD database exceeds the {MAX_TDB_STATEMENTS}-statement limit"
        )
    return text


def _database_format(suffix: str) -> str:
    normalized_suffix = suffix.casefold()
    if normalized_suffix not in _CALPHAD_DATABASE_SUFFIXES:
        raise CalphadInputError("CALPHAD database has an unsupported parser format")
    return normalized_suffix.removeprefix(".")


def _parse_database(database_text: str, *, suffix: str) -> Any:
    from pycalphad import Database

    return Database.from_file(io.StringIO(database_text), fmt=_database_format(suffix))


def _warning_messages(captured: Sequence[warnings.WarningMessage]) -> list[str]:
    messages: set[str] = set()
    for item in captured[:128]:
        message = " ".join(str(item.message).split())
        if message:
            messages.add(message[:2048])
    return sorted(messages)[:64]


def _bounded_mapping_strings(value: Any, *, max_items: int) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    if len(value) > max_items:
        raise CalphadInputError("CALPHAD phase model metadata exceeds its item limit")
    result: dict[str, str] = {}
    for raw_key, raw_value in value.items():
        key = " ".join(str(raw_key).split())[:256]
        text = " ".join(str(raw_value).split())[:1024]
        if key:
            result[key] = text
    return dict(sorted(result.items()))


def _reference_manifest(database: Any) -> dict[str, Any]:
    references = getattr(database, "references", None)
    if not isinstance(references, Mapping):
        return {"count": 0, "included_count": 0, "truncated": False, "entries": []}
    ordered = sorted(references.items(), key=lambda item: str(item[0]))
    entries: list[dict[str, str]] = []
    for raw_id, raw_text in ordered[:MAX_REFERENCES]:
        reference_id = " ".join(str(raw_id).split())[:256]
        text = " ".join(str(raw_text).split())[:MAX_REFERENCE_TEXT_CHARS]
        entries.append({"reference_id": reference_id, "text": text})
    return {
        "count": len(ordered),
        "included_count": len(entries),
        "truncated": len(ordered) > len(entries),
        "entries": entries,
    }


def _database_manifest(  # noqa: N803 - scientific unit suffix is part of the API
    database: Any,
    database_file: _DatabaseFile,
    *,
    requested_components: tuple[str, ...],
    requested_phases: tuple[str, ...],
    source: str,
    license_id: str,
    artifact_id: str,
    assessment_scope: str,
    reference_state: str,
    assessment_temperature_limits_K: tuple[float, float] | None,  # noqa: N803
    assessment_pressure_limits_Pa: tuple[float, float] | None,  # noqa: N803
    database_format: str,
    registry: Mapping[str, Any] | None,
    parse_warnings: Sequence[str],
) -> dict[str, Any]:
    import pycalphad

    available_components = tuple(sorted(str(value).strip().upper() for value in database.elements))
    available_species = tuple(sorted(str(value).strip().upper() for value in database.species))
    available_phases = tuple(sorted(str(value).strip().upper() for value in database.phases))
    parameter_count = len(database._parameters)
    if len(available_components) > MAX_DATABASE_ELEMENTS:
        raise CalphadInputError(
            f"CALPHAD database exceeds the {MAX_DATABASE_ELEMENTS}-element limit"
        )
    if len(available_species) > MAX_DATABASE_SPECIES:
        raise CalphadInputError(
            f"CALPHAD database exceeds the {MAX_DATABASE_SPECIES}-species limit"
        )
    if len(available_phases) > MAX_DATABASE_PHASES:
        raise CalphadInputError(f"CALPHAD database exceeds the {MAX_DATABASE_PHASES}-phase limit")
    if parameter_count > MAX_DATABASE_PARAMETERS:
        raise CalphadInputError(
            f"CALPHAD database exceeds the {MAX_DATABASE_PARAMETERS}-parameter limit"
        )

    missing_components = sorted(set(requested_components) - set(available_components))
    missing_phases = sorted(set(requested_phases) - set(available_phases))
    if missing_components or missing_phases:
        raise CalphadInputError(
            "CALPHAD request is outside the supplied database: "
            f"missing_components={missing_components}, missing_phases={missing_phases}"
        )

    phase_models: list[dict[str, Any]] = []
    phase_projection_warnings: list[str] = []
    for phase_name in available_phases:
        phase = database.phases[phase_name]
        site_ratios: list[float] = []
        for raw_ratio in phase.sublattices:
            ratio = float(raw_ratio)
            if not math.isfinite(ratio) or ratio <= 0:
                raise CalphadInputError(f"phase {phase_name} has an invalid sublattice ratio")
            site_ratios.append(ratio)
        if len(site_ratios) > MAX_SUBLATTICES_PER_PHASE:
            raise CalphadInputError(
                f"phase {phase_name} exceeds the {MAX_SUBLATTICES_PER_PHASE}-sublattice limit"
            )
        constituent_groups = tuple(phase.constituents)
        if len(constituent_groups) != len(site_ratios):
            if database_format != "dat" or len(site_ratios) != 1 or not constituent_groups:
                raise CalphadInputError(f"phase {phase_name} has inconsistent sublattice metadata")
            # ChemSage MQMQA phases expose cation/anion constituent groups but one
            # aggregate phase ratio through pycalphad. Preserve every constituent
            # without inventing a fixed ratio for either charge-dependent group.
            constituent_groups = (frozenset().union(*constituent_groups),)
            phase_projection_warnings.append(
                f"ChemSage phase {phase_name} constituent groups were flattened into "
                "one aggregate manifest group because pycalphad does not expose "
                "fixed per-group site ratios."
            )
        sublattices: list[dict[str, Any]] = []
        for index, (ratio, constituents) in enumerate(zip(site_ratios, constituent_groups)):
            names = sorted(str(value).strip().upper() for value in constituents)
            if len(names) > MAX_CONSTITUENTS_PER_SUBLATTICE:
                raise CalphadInputError(
                    f"phase {phase_name} sublattice {index} exceeds its constituent limit"
                )
            sublattices.append({"index": index, "site_ratio": ratio, "constituents": names})
        phase_models.append(
            {
                "name": phase_name,
                "sublattice_site_ratios": site_ratios,
                "sublattices": sublattices,
                "model_hints": _bounded_mapping_strings(
                    getattr(phase, "model_hints", {}), max_items=MAX_MODEL_HINTS_PER_PHASE
                ),
            }
        )

    registry_payload = dict(registry) if registry is not None else None
    if registry_payload is not None:
        parsed_physical_elements = sorted(
            name for name in available_components if name not in {"/-", "ELECTRON_GAS"}
        )
        if registry_payload["declared_elements"] != parsed_physical_elements:
            raise CalphadInputError("manifest elements do not match parsed database elements")
        if registry_payload["declared_phases"] != list(available_phases):
            raise CalphadInputError("manifest phases do not match parsed database phases")

    physical_elements = [name for name in available_components if name not in {"VA", "/-"}]
    vacancy_components = [name for name in available_components if name == "VA"]
    pseudo_elements = [name for name in available_components if name == "/-"]
    manifest: dict[str, Any] = {
        "schema_version": DATABASE_MANIFEST_SCHEMA_VERSION,
        "path": str(database_file.path),
        "name": database_file.path.name,
        "sha256": database_file.sha256,
        "size_bytes": database_file.size_bytes,
        "format": database_format,
        "package_test_database": False,
        "source": source,
        "license_id": license_id,
        "artifact_id": artifact_id or None,
        "assessment_scope": assessment_scope,
        "reference_state": reference_state,
        "requested_components": list(requested_components),
        "requested_phases": list(requested_phases),
        "components": list(available_components),
        "physical_elements": physical_elements,
        "vacancy_components": vacancy_components,
        "pseudo_elements": pseudo_elements,
        "species": list(available_species),
        "phases": list(available_phases),
        "available_components": list(available_components),
        "available_phases": list(available_phases),
        "phase_models": phase_models,
        "parameter_count": parameter_count,
        "references": _reference_manifest(database),
        "pycalphad_version": str(pycalphad.__version__),
        "registry_manifest": registry_payload,
        "assessment_temperature_limits_K": (
            list(assessment_temperature_limits_K)
            if assessment_temperature_limits_K is not None
            else None
        ),
        "assessment_pressure_limits_Pa": (
            list(assessment_pressure_limits_Pa)
            if assessment_pressure_limits_Pa is not None
            else None
        ),
        "warnings": sorted(set(parse_warnings) | set(phase_projection_warnings)),
        "limits": {
            "max_database_bytes": MAX_DATABASE_BYTES,
            "max_elements": MAX_DATABASE_ELEMENTS,
            "max_species": MAX_DATABASE_SPECIES,
            "max_phases": MAX_DATABASE_PHASES,
            "max_parameters": MAX_DATABASE_PARAMETERS,
            "database_parse_wall_time_seconds": DATABASE_PARSE_WALL_TIME_SECONDS,
        },
    }
    encoded = _canonical_json_bytes(manifest)
    if len(encoded) > MAX_INSPECTION_JSON_BYTES:
        raise CalphadInputError(
            f"CALPHAD inspection exceeds the {MAX_INSPECTION_JSON_BYTES}-byte limit"
        )
    manifest["manifest_sha256"] = _sha256_bytes(encoded)
    return manifest


def _load_inspected_database(  # noqa: N803 - scientific unit suffix is part of the API
    path: str | Path,
    *,
    components: Iterable[str] | None,
    phases: Iterable[str] | None,
    source: str,
    license_id: str,
    artifact_id: str,
    assessment_scope: str,
    reference_state: str,
    database_id: str,
    expected_sha256: str,
    expected_size_bytes: int | None,
    assessment_temperature_limits_K: Iterable[float] | None,  # noqa: N803
    assessment_pressure_limits_Pa: Iterable[float] | None,  # noqa: N803
) -> tuple[Any, dict[str, Any]]:
    requested_components = (
        None
        if components is None
        else _required_names(components, field_name="components", max_count=MAX_REQUEST_COMPONENTS)
    )
    requested_phases = (
        None
        if phases is None
        else _required_names(phases, field_name="phases", max_count=MAX_REQUEST_PHASES)
    )
    selection = _manifest_selection(path, database_id=database_id)
    database_path = selection.database_path if selection is not None else Path(path).expanduser()
    database_format = _database_format(database_path.suffix)
    database_file = _read_regular_file(
        database_path,
        max_bytes=MAX_DATABASE_BYTES,
        kind="CALPHAD database",
    )
    normalized_expected_sha = str(expected_sha256 or "").strip().lower()
    if (bool(normalized_expected_sha)) != (expected_size_bytes is not None):
        raise CalphadInputError("expected_sha256 and expected_size_bytes must be supplied together")
    if normalized_expected_sha:
        if not _SHA256_PATTERN.fullmatch(normalized_expected_sha):
            raise CalphadInputError("expected_sha256 must be a lowercase SHA-256 digest")
        if (
            isinstance(expected_size_bytes, bool)
            or not isinstance(expected_size_bytes, int)
            or expected_size_bytes <= 0
        ):
            raise CalphadInputError("expected_size_bytes must be a positive integer")
        if (
            database_file.sha256 != normalized_expected_sha
            or database_file.size_bytes != expected_size_bytes
        ):
            raise CalphadInputError("CALPHAD database does not match its catalog hash/size")
    if _is_fixture_database(database_file.path, database_file.sha256):
        raise CalphadInputError(
            "pycalphad package test databases are fixtures, not production scientific evidence"
        )

    registry: dict[str, Any] | None = None
    if selection is not None:
        registry = _verified_manifest_metadata(selection, database_file)
        manifest_source = registry["source_uri"]
        manifest_license = registry["license_id"]
        supplied_source = _clean_text(
            source, field_name="CALPHAD database source/provenance", required=False
        )
        supplied_license = _clean_text(
            license_id,
            field_name="CALPHAD database license or use authorization",
            required=False,
            max_chars=256,
        )
        if supplied_source and supplied_source != manifest_source:
            raise CalphadInputError("supplied source conflicts with the verified manifest")
        if supplied_license and supplied_license != manifest_license:
            raise CalphadInputError("supplied license conflicts with the verified manifest")
        normalized_source = manifest_source
        normalized_license = manifest_license
        manifest_scope = _clean_provenance(
            registry.get("assessment_scope"),
            field_name="manifest assessment_scope",
        )
        manifest_reference_state = _clean_provenance(
            registry.get("reference_state"),
            field_name="manifest reference_state",
            max_chars=1024,
        )
        supplied_scope = _clean_text(
            assessment_scope,
            field_name="CALPHAD assessment_scope",
            required=False,
        )
        supplied_reference_state = _clean_text(
            reference_state,
            field_name="CALPHAD reference_state",
            required=False,
            max_chars=1024,
        )
        if supplied_scope and supplied_scope != manifest_scope:
            raise CalphadInputError(
                "supplied assessment_scope conflicts with the verified manifest"
            )
        if supplied_reference_state and supplied_reference_state != manifest_reference_state:
            raise CalphadInputError("supplied reference_state conflicts with the verified manifest")
        normalized_scope = manifest_scope
        normalized_reference_state = manifest_reference_state
    else:
        normalized_source = _clean_provenance(
            source,
            field_name="CALPHAD database source/provenance",
        )
        normalized_license = _clean_provenance(
            license_id,
            field_name="CALPHAD database license or use authorization",
            max_chars=256,
        )
        normalized_scope = _clean_provenance(
            assessment_scope,
            field_name="CALPHAD assessment_scope",
        )
        normalized_reference_state = _clean_provenance(
            reference_state,
            field_name="CALPHAD reference_state",
            max_chars=1024,
        )
    normalized_artifact = _clean_text(
        artifact_id, field_name="CALPHAD artifact_id", required=False, max_chars=512
    )
    if selection is None and not normalized_artifact:
        raise CalphadInputError(
            "standalone CALPHAD database requires an artifact_id or resource_id"
        )
    if selection is None and not normalized_expected_sha:
        raise CalphadInputError(
            "standalone CALPHAD database requires catalog-bound expected_sha256 "
            "and expected_size_bytes"
        )
    supplied_temperature_limits = _temperature_limits(
        assessment_temperature_limits_K,
        field_name="assessment_temperature_limits_K",
    )
    if selection is None and supplied_temperature_limits is None:
        raise CalphadInputError(
            "standalone CALPHAD database requires explicit assessment_temperature_limits_K"
        )
    if registry is not None:
        registry_limits = (
            tuple(registry["temperature_limits_K"])
            if registry.get("temperature_limits_K") is not None
            else None
        )
        if (
            supplied_temperature_limits is not None
            and supplied_temperature_limits != registry_limits
        ):
            raise CalphadInputError(
                "supplied assessment temperature limits conflict with the verified manifest"
            )
        effective_temperature_limits = registry_limits
    else:
        effective_temperature_limits = supplied_temperature_limits
    supplied_pressure_limits = _pressure_limits(
        assessment_pressure_limits_Pa,
        field_name="assessment_pressure_limits_Pa",
    )
    if selection is None and supplied_pressure_limits is None:
        raise CalphadInputError(
            "standalone CALPHAD database requires explicit assessment_pressure_limits_Pa"
        )
    if registry is not None:
        registry_pressure_limits = tuple(registry["assessment_pressure_limits_Pa"])
        if (
            supplied_pressure_limits is not None
            and supplied_pressure_limits != registry_pressure_limits
        ):
            raise CalphadInputError(
                "supplied assessment pressure limits conflict with the verified manifest"
            )
        effective_pressure_limits = registry_pressure_limits
    else:
        effective_pressure_limits = supplied_pressure_limits
    database_text = _decode_database_text(database_file)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        try:
            with _wall_time_limit(DATABASE_PARSE_WALL_TIME_SECONDS):
                database = _parse_database(database_text, suffix=database_file.path.suffix)
        except _WallTimeExceededError as exc:
            raise CalphadTimeoutError(
                f"CALPHAD database parse exceeded {DATABASE_PARSE_WALL_TIME_SECONDS} seconds"
            ) from exc
        except CalphadExecutionError:
            raise
        except NotImplementedError as exc:
            raise CalphadUnsupportedError(
                "pycalphad does not support a construct in the CALPHAD database"
            ) from exc
        except Exception as exc:
            raise CalphadInputError(
                f"pycalphad could not parse CALPHAD database {database_file.path.name!r}"
            ) from exc
    if requested_components is None:
        requested_components = tuple(
            sorted(
                str(value).strip().upper()
                for value in database.elements
                if str(value).strip().upper() != "/-"
            )
        )
    if requested_phases is None:
        requested_phases = tuple(sorted(str(value).strip().upper() for value in database.phases))
    manifest = _database_manifest(
        database,
        database_file,
        requested_components=requested_components,
        requested_phases=requested_phases,
        source=normalized_source,
        license_id=normalized_license,
        artifact_id=normalized_artifact,
        assessment_scope=normalized_scope,
        reference_state=normalized_reference_state,
        assessment_temperature_limits_K=effective_temperature_limits,
        assessment_pressure_limits_Pa=effective_pressure_limits,
        database_format=database_format,
        registry=registry,
        parse_warnings=_warning_messages(captured),
    )
    return database, manifest


def inspect_calphad_input(  # noqa: N803 - scientific unit suffix is part of the API
    path: str | Path,
    *,
    components: Iterable[str] | None = None,
    phases: Iterable[str] | None = None,
    source: str = "",
    license_id: str = "",
    artifact_id: str = "",
    assessment_scope: str = "",
    reference_state: str = "",
    database_id: str = "",
    expected_sha256: str = "",
    expected_size_bytes: int | None = None,
    assessment_temperature_limits_K: Iterable[float] | None = None,  # noqa: N803
    assessment_pressure_limits_Pa: Iterable[float] | None = None,  # noqa: N803
) -> dict[str, Any]:
    """Return a bounded, content-addressed manifest for one parsed TDB.

    ``path`` may be a TDB, an Ultra ``manifest.json``, or a directory containing
    that manifest. A verified manifest supplies authoritative source and license
    metadata; standalone/user TDBs require those fields explicitly. Omitting
    ``components`` and/or ``phases`` is permitted only for bounded catalog
    discovery; equilibrium execution still requires both sets explicitly.
    """

    _, manifest = _load_inspected_database(
        path,
        components=components,
        phases=phases,
        source=source,
        license_id=license_id,
        artifact_id=artifact_id,
        assessment_scope=assessment_scope,
        reference_state=reference_state,
        database_id=database_id,
        expected_sha256=expected_sha256,
        expected_size_bytes=expected_size_bytes,
        assessment_temperature_limits_K=assessment_temperature_limits_K,
        assessment_pressure_limits_Pa=assessment_pressure_limits_Pa,
    )
    return manifest


def _finite_axis(
    values: float | Iterable[float],
    *,
    field_name: str,
    minimum: float,
    maximum: float,
) -> tuple[float, ...]:
    if isinstance(values, bool) or isinstance(values, (str, bytes, bytearray)):
        raise CalphadInputError(f"{field_name} must contain finite numeric values")
    if isinstance(values, (int, float)):
        candidates: Iterable[Any] = [values]
    else:
        try:
            candidates = iter(values)
        except TypeError as exc:
            raise CalphadInputError(f"{field_name} must contain finite numeric values") from exc
    normalized: list[float] = []
    for index, raw_value in enumerate(candidates):
        if index >= MAX_AXIS_VALUES:
            raise CalphadInputError(f"{field_name} exceeds the {MAX_AXIS_VALUES}-value axis limit")
        if isinstance(raw_value, bool):
            raise CalphadInputError(f"{field_name} must contain finite numeric values")
        try:
            value = float(raw_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise CalphadInputError(f"{field_name} must contain finite numeric values") from exc
        if not math.isfinite(value):
            raise CalphadInputError(f"{field_name} contains a non-finite value")
        if value < minimum or value > maximum:
            raise CalphadInputError(f"{field_name} values must be within [{minimum}, {maximum}]")
        normalized.append(0.0 if value == 0 else value)
    if not normalized:
        raise CalphadInputError(f"{field_name} must not be empty")
    if len(set(normalized)) != len(normalized):
        raise CalphadInputError(f"{field_name} contains duplicate values")
    return tuple(sorted(normalized))


def _canonical_composition_value(value: float) -> float:
    normalized = round(value, COMPOSITION_CANONICAL_DECIMAL_PLACES)
    if normalized == 0:
        return 0.0
    return min(1.0, max(0.0, normalized))


def _composition_closure_grid(
    compositions: Mapping[str, tuple[float, ...]],
    *,
    dependent_component: str,
) -> list[dict[str, float]]:
    ordered_names = sorted(compositions)
    axes = [compositions[name] for name in ordered_names]
    composition_point_count = math.prod(len(axis) for axis in axes) if axes else 1
    if composition_point_count > MAX_GRID_POINTS:
        raise CalphadInputError(
            "independent composition grid exceeds the global CALPHAD point limit"
        )
    closure_records: list[dict[str, float]] = []
    combinations = itertools.product(*axes) if axes else [()]
    for values in combinations:
        point = dict(zip(ordered_names, values))
        dependent_fraction = 1.0 - math.fsum(values)
        if (
            dependent_fraction < -COMPOSITION_TOLERANCE
            or dependent_fraction > 1 + COMPOSITION_TOLERANCE
        ):
            raise CalphadInputError("independent composition grid violates mole-fraction closure")
        point[dependent_component] = _canonical_composition_value(dependent_fraction)
        closure_records.append(dict(sorted(point.items())))
    return closure_records


def canonicalize_equilibrium_compositions(
    independent_compositions: Mapping[str, float | Iterable[float]],
    *,
    components: Sequence[str],
) -> tuple[dict[str, tuple[float, ...]], str, list[dict[str, float]]]:
    """Return a deterministic Cartesian composition parameterization.

    The alphabetically first physical component is always dependent. Binary
    axes and any higher-dimensional Cartesian point set that remains exactly
    representable after projection are reframed transparently. Coupled grids
    that would gain or lose points under that reparameterization fail closed.
    """

    if not isinstance(independent_compositions, Mapping):
        raise CalphadInputError("independent_compositions must be an explicit mapping")
    nonvacancy_components = tuple(
        sorted(component for component in components if component not in {"VA", "/-"})
    )
    if not nonvacancy_components:
        raise CalphadInputError("at least one non-vacancy component is required")
    expected_count = len(nonvacancy_components) - 1
    if len(independent_compositions) != expected_count:
        raise CalphadInputError(
            "independent_compositions must specify exactly one fewer component "
            "than the non-vacancy component count"
        )
    normalized: dict[str, tuple[float, ...]] = {}
    for raw_component, raw_values in independent_compositions.items():
        component = _clean_name(raw_component, field_name="composition component")
        if component in {"VA", "/-"} or component not in nonvacancy_components:
            raise CalphadInputError(
                f"composition component {component!r} is not an eligible requested component"
            )
        if component in normalized:
            raise CalphadInputError("independent_compositions contains duplicate components")
        axis = tuple(
            _canonical_composition_value(value)
            for value in _finite_axis(
                raw_values,
                field_name=f"X({component})",
                minimum=0.0,
                maximum=1.0,
            )
        )
        if len(set(axis)) != len(axis):
            raise CalphadInputError(
                f"X({component}) contains duplicate values after canonicalization"
            )
        normalized[component] = tuple(sorted(axis))
    supplied_dependent = next(
        component for component in nonvacancy_components if component not in normalized
    )
    supplied_records = _composition_closure_grid(
        normalized,
        dependent_component=supplied_dependent,
    )
    canonical_dependent = nonvacancy_components[0]
    if supplied_dependent == canonical_dependent:
        return dict(sorted(normalized.items())), canonical_dependent, supplied_records

    canonical_independent = tuple(
        component for component in nonvacancy_components if component != canonical_dependent
    )
    canonical_compositions = {
        component: tuple(sorted({point[component] for point in supplied_records}))
        for component in canonical_independent
    }
    canonical_records = _composition_closure_grid(
        canonical_compositions,
        dependent_component=canonical_dependent,
    )
    physical_components = tuple(sorted(nonvacancy_components))
    supplied_point_set = {
        tuple(point[component] for component in physical_components) for point in supplied_records
    }
    canonical_point_set = {
        tuple(point[component] for component in physical_components) for point in canonical_records
    }
    if supplied_point_set != canonical_point_set:
        raise CalphadInputError(
            "independent composition grid cannot be reframed to the canonical dependent "
            "component without changing its Cartesian point set"
        )
    return (
        dict(sorted(canonical_compositions.items())),
        canonical_dependent,
        canonical_records,
    )


def _bounded_positive_int(value: Any, *, field_name: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value > maximum:
        raise CalphadInputError(f"{field_name} must be an integer in [1, {maximum}]")
    return value


def _bounded_wall_seconds(value: Any) -> float:
    if isinstance(value, bool):
        raise CalphadInputError("wall_time_seconds must be a finite number")
    try:
        seconds = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CalphadInputError("wall_time_seconds must be a finite number") from exc
    if (
        not math.isfinite(seconds)
        or seconds < MIN_WALL_TIME_SECONDS
        or seconds > MAX_WALL_TIME_SECONDS
    ):
        raise CalphadInputError(
            "wall_time_seconds must be finite and within "
            f"[{MIN_WALL_TIME_SECONDS}, {MAX_WALL_TIME_SECONDS}]"
        )
    return seconds


class _WallTimeExceededError(Exception):
    pass


@contextmanager
def _wall_time_limit(seconds: float) -> Iterator[None]:
    if threading.current_thread() is not threading.main_thread():
        raise CalphadExecutionError("bounded CALPHAD execution requires the process main thread")
    if not hasattr(signal, "setitimer") or not hasattr(signal, "ITIMER_REAL"):
        raise CalphadExecutionError("bounded CALPHAD execution requires POSIX wall timers")
    previous_delay, previous_interval = signal.getitimer(signal.ITIMER_REAL)
    if previous_delay > 0 or previous_interval > 0:
        raise CalphadExecutionError(
            "cannot establish CALPHAD wall limit while another real-time timer is active"
        )
    previous_handler = signal.getsignal(signal.SIGALRM)

    def _timeout_handler(_signum: int, _frame: Any) -> None:
        raise _WallTimeExceededError

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def _calculate_equilibrium(
    database: Any,
    components: Sequence[str],
    phases: Sequence[str],
    conditions: Mapping[Any, Sequence[float]],
) -> Any:
    from pycalphad import equilibrium

    return equilibrium(
        database,
        list(components),
        list(phases),
        {variable: list(values) for variable, values in conditions.items()},
        output="GM",
        verbose=False,
    )


def _dataset_nbytes(dataset: Any) -> int:
    total = 0
    for variable in dataset.variables.values():
        nbytes = getattr(variable, "nbytes", None)
        if nbytes is None:
            values = variable.values
            nbytes = getattr(values, "nbytes", 0)
        total += int(nbytes)
    return total


def _result_points(
    dataset: Any,
    *,
    condition_axes: Mapping[str, tuple[float, ...]],
    compositions: Mapping[str, tuple[float, ...]],
    dependent_component: str,
    requested_components: tuple[str, ...],
    requested_phases: tuple[str, ...],
) -> list[dict[str, Any]]:
    import numpy as np

    for required in ("GM", "MU", "NP", "Phase", "X"):
        if required not in dataset:
            raise CalphadExecutionError(f"pycalphad result is missing {required}")
    gm = dataset["GM"]
    expected_dimensions = {"N", "P", "T"} | {f"X_{component}" for component in compositions}
    if set(gm.dims) != expected_dimensions:
        raise CalphadExecutionError(f"pycalphad GM dimensions are unexpected: {sorted(gm.dims)}")
    expected_components = tuple(
        component for component in requested_components if component not in {"VA", "/-"}
    )
    if "component" not in dataset.coords:
        raise CalphadExecutionError("pycalphad result lacks coordinate component")
    result_components = tuple(
        str(value).strip().upper() for value in dataset.coords["component"].values.tolist()
    )
    if result_components != expected_components:
        raise CalphadExecutionError(
            "pycalphad component coordinate does not match the requested non-vacancy components"
        )
    expected_variable_dimensions = {
        "MU": expected_dimensions | {"component"},
        "NP": expected_dimensions | {"vertex"},
        "Phase": expected_dimensions | {"vertex"},
        "X": expected_dimensions | {"vertex", "component"},
    }
    for variable_name, dimensions in expected_variable_dimensions.items():
        actual_dimensions = set(dataset[variable_name].dims)
        if actual_dimensions != dimensions:
            raise CalphadExecutionError(
                f"pycalphad {variable_name} dimensions are unexpected: {sorted(actual_dimensions)}"
            )
    expected_axes = {
        "N": condition_axes["N"],
        "P": condition_axes["P"],
        "T": condition_axes["T"],
        **{f"X_{name}": values for name, values in compositions.items()},
    }
    for dimension, expected in expected_axes.items():
        if dimension not in dataset.coords:
            raise CalphadExecutionError(f"pycalphad result lacks coordinate {dimension}")
        actual = tuple(float(value) for value in dataset.coords[dimension].values.tolist())
        if len(actual) != len(expected) or not all(
            math.isclose(left, right, rel_tol=0.0, abs_tol=1e-14)
            for left, right in zip(actual, expected)
        ):
            raise CalphadExecutionError(
                f"pycalphad coordinate {dimension} does not match the request"
            )

    points: list[dict[str, Any]] = []
    requested_phase_set = set(requested_phases)
    for raw_indices in np.ndindex(gm.shape):
        indexers = dict(zip(gm.dims, (int(value) for value in raw_indices)))
        coordinate_values = {
            dimension: float(dataset.coords[dimension].values[index])
            for dimension, index in indexers.items()
        }
        independent_values = {
            component: coordinate_values[f"X_{component}"] for component in compositions
        }
        dependent_fraction = 1.0 - math.fsum(independent_values.values())
        if dependent_fraction < -COMPOSITION_TOLERANCE:
            raise CalphadExecutionError("result composition violates mole-fraction closure")
        all_compositions = dict(independent_values)
        all_compositions[dependent_component] = min(1.0, max(0.0, dependent_fraction))

        gm_value = float(gm.values[raw_indices])
        if not math.isfinite(gm_value):
            raise CalphadExecutionError("pycalphad produced non-finite GM")
        chemical_potential_values = (
            dataset["MU"].isel(indexers).transpose("component").values.reshape(-1)
        )
        if len(chemical_potential_values) != len(result_components):
            raise CalphadExecutionError("pycalphad MU and component shapes disagree")
        chemical_potentials: dict[str, float] = {}
        for component, raw_value in zip(result_components, chemical_potential_values.tolist()):
            try:
                value = float(raw_value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise CalphadExecutionError(
                    f"pycalphad produced a non-numeric MU for component {component}"
                ) from exc
            if not math.isfinite(value):
                raise CalphadExecutionError(
                    f"pycalphad produced non-finite MU for component {component}"
                )
            chemical_potentials[component] = value
        gibbs_from_chemical_potentials = math.fsum(
            all_compositions[component] * chemical_potentials[component]
            for component in result_components
        )
        gibbs_euler_residual = abs(gm_value - gibbs_from_chemical_potentials)
        if gibbs_euler_residual > GIBBS_EULER_ABSOLUTE_TOLERANCE_J_PER_MOL:
            raise CalphadExecutionError(
                "GM is inconsistent with the bulk composition and chemical potentials: "
                f"Euler residual {gibbs_euler_residual} J/mol"
            )
        phase_values = dataset["Phase"].isel(indexers).transpose("vertex").values.reshape(-1)
        amount_values = dataset["NP"].isel(indexers).transpose("vertex").values.reshape(-1)
        if phase_values.shape != amount_values.shape:
            raise CalphadExecutionError("pycalphad Phase and NP shapes disagree")
        phase_composition_values = (
            dataset["X"].isel(indexers).transpose("vertex", "component").values
        )
        expected_phase_composition_shape = (len(phase_values), len(result_components))
        if phase_composition_values.shape != expected_phase_composition_shape:
            raise CalphadExecutionError("pycalphad X, Phase, and component shapes disagree")
        phase_amounts: dict[str, list[float]] = {}
        stable_phase_vertices: list[dict[str, Any]] = []
        for vertex_index, (raw_phase, raw_amount) in enumerate(
            zip(phase_values.tolist(), amount_values.tolist())
        ):
            phase_name = str(raw_phase).strip().upper()
            try:
                amount = float(raw_amount)
            except (TypeError, ValueError, OverflowError) as exc:
                raise CalphadExecutionError("pycalphad produced a non-numeric NP") from exc
            if not phase_name:
                if math.isfinite(amount) and abs(amount) > STABLE_PHASE_THRESHOLD:
                    raise CalphadExecutionError("pycalphad produced an unnamed stable phase")
                continue
            if phase_name not in requested_phase_set:
                raise CalphadExecutionError(f"pycalphad returned unrequested phase {phase_name!r}")
            if not math.isfinite(amount):
                raise CalphadExecutionError(
                    f"pycalphad produced non-finite NP for phase {phase_name}"
                )
            if amount < -PHASE_FRACTION_TOLERANCE or amount > 1 + PHASE_FRACTION_TOLERANCE:
                raise CalphadExecutionError(
                    f"pycalphad produced out-of-range NP for phase {phase_name}"
                )
            if amount > STABLE_PHASE_THRESHOLD:
                normalized_amount = max(0.0, amount)
                raw_phase_composition = phase_composition_values[vertex_index].tolist()
                phase_composition: dict[str, float] = {}
                for component, raw_value in zip(result_components, raw_phase_composition):
                    try:
                        value = float(raw_value)
                    except (TypeError, ValueError, OverflowError) as exc:
                        raise CalphadExecutionError(
                            f"pycalphad produced a non-numeric X for {phase_name}/{component}"
                        ) from exc
                    if not math.isfinite(value):
                        raise CalphadExecutionError(
                            f"pycalphad produced non-finite X for {phase_name}/{component}"
                        )
                    if value < -COMPOSITION_TOLERANCE or value > 1 + COMPOSITION_TOLERANCE:
                        raise CalphadExecutionError(
                            f"pycalphad produced out-of-range X for {phase_name}/{component}"
                        )
                    phase_composition[component] = min(1.0, max(0.0, value))
                phase_composition_sum = math.fsum(phase_composition.values())
                if not math.isclose(
                    phase_composition_sum,
                    1.0,
                    rel_tol=0.0,
                    abs_tol=COMPOSITION_TOLERANCE,
                ):
                    raise CalphadExecutionError(
                        f"phase composition for {phase_name} does not close to one: "
                        f"{phase_composition_sum}"
                    )
                phase_amounts.setdefault(phase_name, []).append(normalized_amount)
                stable_phase_vertices.append(
                    {
                        "vertex_index": vertex_index,
                        "phase": phase_name,
                        "NP_phase_fraction": normalized_amount,
                        "composition_mole_fraction": phase_composition,
                        "composition_sum": phase_composition_sum,
                    }
                )
        stable_phases = [
            {"name": phase_name, "NP_phase_fraction": math.fsum(phase_amounts[phase_name])}
            for phase_name in sorted(phase_amounts)
        ]
        phase_sum = math.fsum(phase["NP_phase_fraction"] for phase in stable_phases)
        if not stable_phases or not math.isclose(
            phase_sum, 1.0, rel_tol=0.0, abs_tol=PHASE_FRACTION_TOLERANCE
        ):
            raise CalphadExecutionError(f"stable phase fractions do not close to one: {phase_sum}")
        reconstructed_composition = {
            component: math.fsum(
                float(vertex["NP_phase_fraction"])
                * float(vertex["composition_mole_fraction"][component])
                for vertex in stable_phase_vertices
            )
            for component in result_components
        }
        bulk_composition_residuals = {
            component: abs(reconstructed_composition[component] - all_compositions[component])
            for component in result_components
        }
        maximum_bulk_composition_residual = max(bulk_composition_residuals.values(), default=0.0)
        if maximum_bulk_composition_residual > BULK_COMPOSITION_TOLERANCE:
            raise CalphadExecutionError(
                "stable phase amounts/compositions do not reconstruct the requested bulk "
                f"composition: maximum residual {maximum_bulk_composition_residual}"
            )
        points.append(
            {
                "conditions": {
                    "T_K": coordinate_values["T"],
                    "P_Pa": coordinate_values["P"],
                    "N_mol": coordinate_values["N"],
                    "composition_mole_fraction": dict(sorted(all_compositions.items())),
                },
                "stable_phases": stable_phases,
                "stable_phase_vertices": stable_phase_vertices,
                "phase_fraction_sum": phase_sum,
                "reconstructed_composition_mole_fraction": reconstructed_composition,
                "bulk_composition_residual_by_component": bulk_composition_residuals,
                "maximum_bulk_composition_residual": maximum_bulk_composition_residual,
                "GM_J_per_mol": gm_value,
                "chemical_potentials_J_per_mol": chemical_potentials,
                "gibbs_from_chemical_potentials_J_per_mol": (gibbs_from_chemical_potentials),
                "gibbs_euler_residual_J_per_mol": gibbs_euler_residual,
            }
        )
    points.sort(
        key=lambda point: (
            point["conditions"]["T_K"],
            point["conditions"]["P_Pa"],
            tuple(point["conditions"]["composition_mole_fraction"].items()),
        )
    )
    return points


def run_calphad_equilibrium(  # noqa: N803 - scientific unit suffix is part of the API
    path: str | Path,
    *,
    components: Iterable[str],
    phases: Iterable[str],
    temperatures_K: float | Iterable[float],  # noqa: N803
    pressures_Pa: float | Iterable[float],  # noqa: N803
    total_amount_mol: float | Iterable[float],
    independent_compositions: Mapping[str, float | Iterable[float]],
    source: str = "",
    license_id: str = "",
    artifact_id: str = "",
    assessment_scope: str = "",
    reference_state: str = "",
    database_id: str = "",
    expected_sha256: str = "",
    expected_size_bytes: int | None = None,
    assessment_temperature_limits_K: Iterable[float] | None = None,  # noqa: N803
    assessment_pressure_limits_Pa: Iterable[float] | None = None,  # noqa: N803
    max_grid_points: int = MAX_GRID_POINTS,
    wall_time_seconds: float = DEFAULT_WALL_TIME_SECONDS,
    max_result_bytes: int = DEFAULT_RESULT_BYTES,
) -> dict[str, Any]:
    """Run a bounded equilibrium grid and return deterministic JSON evidence.

    The API fixes pycalphad's solver surface: callers cannot inject models,
    parameters, custom outputs, or calculation options. ``N`` must be explicitly
    supplied and equal to one mole, which is the normalization supported by the
    pinned pycalphad equilibrium implementation.
    """

    database, database_manifest = _load_inspected_database(
        path,
        components=components,
        phases=phases,
        source=source,
        license_id=license_id,
        artifact_id=artifact_id,
        assessment_scope=assessment_scope,
        reference_state=reference_state,
        database_id=database_id,
        expected_sha256=expected_sha256,
        expected_size_bytes=expected_size_bytes,
        assessment_temperature_limits_K=assessment_temperature_limits_K,
        assessment_pressure_limits_Pa=assessment_pressure_limits_Pa,
    )
    normalized_components = tuple(database_manifest["requested_components"])
    normalized_phases = tuple(database_manifest["requested_phases"])
    temperatures = _finite_axis(
        temperatures_K, field_name="temperatures_K", minimum=1.0, maximum=10_000.0
    )
    assessment_limits = database_manifest["assessment_temperature_limits_K"]
    if assessment_limits is not None and (
        temperatures[0] < assessment_limits[0] or temperatures[-1] > assessment_limits[1]
    ):
        raise CalphadInputError(
            "requested temperature is outside the declared assessment/TDB limits: "
            f"{assessment_limits} K"
        )
    pressures = _finite_axis(
        pressures_Pa,
        field_name="pressures_Pa",
        minimum=_MIN_PRESSURE_PA,
        maximum=_MAX_PRESSURE_PA,
    )
    pressure_limits = database_manifest["assessment_pressure_limits_Pa"]
    if pressure_limits is not None and (
        pressures[0] < pressure_limits[0] or pressures[-1] > pressure_limits[1]
    ):
        raise CalphadInputError(
            f"requested pressure is outside the declared assessment limits: {pressure_limits} Pa"
        )
    amounts = _finite_axis(
        total_amount_mol,
        field_name="total_amount_mol",
        minimum=1e-12,
        maximum=1e12,
    )
    if amounts != (1.0,):
        raise CalphadInputError(
            "pycalphad equilibrium currently requires total_amount_mol to be exactly 1"
        )
    compositions, dependent_component, closure_records = canonicalize_equilibrium_compositions(
        independent_compositions, components=normalized_components
    )
    grid_limit = _bounded_positive_int(
        max_grid_points, field_name="max_grid_points", maximum=MAX_GRID_POINTS
    )
    result_limit = _bounded_positive_int(
        max_result_bytes, field_name="max_result_bytes", maximum=MAX_RESULT_BYTES
    )
    wall_limit = _bounded_wall_seconds(wall_time_seconds)
    grid_points = len(temperatures) * len(pressures) * len(amounts)
    for values in compositions.values():
        grid_points *= len(values)
    if grid_points > grid_limit:
        raise CalphadInputError(
            f"CALPHAD grid has {grid_points} points, exceeding limit {grid_limit}"
        )

    requested_phase_models = [
        phase
        for phase in database_manifest["phase_models"]
        if phase["name"] in set(normalized_phases)
    ]
    internal_dof_bound = max(
        1,
        max(
            (
                sum(len(sublattice["constituents"]) for sublattice in phase["sublattices"])
                for phase in requested_phase_models
            ),
            default=1,
        ),
    )
    vertex_bound = max(1, len(normalized_components), len(normalized_phases))
    estimated_result_bytes = (
        grid_points
        * (
            1
            + 2 * vertex_bound
            + vertex_bound * len(normalized_components)
            + vertex_bound * internal_dof_bound
            + len(normalized_components)
        )
        * 16
    )
    if estimated_result_bytes > result_limit:
        raise CalphadInputError(
            "estimated pycalphad result exceeds max_result_bytes: "
            f"{estimated_result_bytes} > {result_limit}"
        )

    from pycalphad import variables as v

    pycalphad_conditions: dict[Any, Sequence[float]] = {
        v.T: temperatures,
        v.P: pressures,
        v.N: amounts,
    }
    for component, values in compositions.items():
        pycalphad_conditions[v.X(component)] = values
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        try:
            with _wall_time_limit(wall_limit):
                dataset = _calculate_equilibrium(
                    database,
                    normalized_components,
                    normalized_phases,
                    pycalphad_conditions,
                )
        except _WallTimeExceededError as exc:
            raise CalphadTimeoutError(f"CALPHAD equilibrium exceeded {wall_limit} seconds") from exc
        except CalphadExecutionError:
            raise
        except Exception as exc:
            raise CalphadExecutionError("pycalphad equilibrium calculation failed") from exc
    dataset_bytes = _dataset_nbytes(dataset)
    if dataset_bytes > result_limit:
        raise CalphadExecutionError(
            f"pycalphad result exceeds max_result_bytes: {dataset_bytes} > {result_limit}"
        )
    condition_axes = {"T": temperatures, "P": pressures, "N": amounts}
    points = _result_points(
        dataset,
        condition_axes=condition_axes,
        compositions=compositions,
        dependent_component=dependent_component,
        requested_components=normalized_components,
        requested_phases=normalized_phases,
    )
    if len(points) != grid_points:
        raise CalphadExecutionError(
            f"pycalphad returned {len(points)} points for a {grid_points}-point request"
        )

    excluded_database_phases = sorted(set(database_manifest["phases"]) - set(normalized_phases))
    phase_selection_scope = (
        "all_database_phases" if not excluded_database_phases else "restricted_database_phase_set"
    )

    request_record: dict[str, Any] = {
        "components": list(normalized_components),
        "phases": list(normalized_phases),
        "conditions": {
            "T": {"values": list(temperatures), "units": "K"},
            "P": {"values": list(pressures), "units": "Pa"},
            "N": {"values": list(amounts), "units": "mol"},
            "independent_compositions": {
                component: {"values": list(values), "units": "mole_fraction"}
                for component, values in compositions.items()
            },
        },
        "dependent_component": dependent_component,
        "phase_selection": {
            "scope": phase_selection_scope,
            "excluded_database_phases": excluded_database_phases,
            "global_equilibrium_claim_supported": not excluded_database_phases,
        },
        "composition_closure": {
            "grid": closure_records,
            "sum": 1.0,
            "absolute_tolerance": COMPOSITION_TOLERANCE,
            "units": "mole_fraction",
        },
        "grid_points": grid_points,
        "limits": {
            "max_grid_points": grid_limit,
            "wall_time_seconds": wall_limit,
            "max_result_bytes": result_limit,
        },
    }
    result_record: dict[str, Any] = {
        "point_count": len(points),
        "dataset_size_bytes": dataset_bytes,
        "points": points,
        "units": {
            "T": "K",
            "P": "Pa",
            "N": "mol",
            "X": "mole_fraction",
            "phase_X": "mole_fraction",
            "bulk_composition_residual": "mole_fraction",
            "NP": "phase_amount_fraction_at_N_equals_1_mol",
            "GM": "J/mol",
            "MU": "J/mol",
            "gibbs_euler_residual": "J/mol",
        },
    }
    result_warnings = sorted(
        set(database_manifest["warnings"])
        | set(_warning_messages(captured))
        | {
            "A successful numerical solve does not independently validate the database assessment or extrapolation domain.",
            "NP is a phase-amount fraction on the fixed N=1 mol calculation basis.",
        }
        | set((database_manifest.get("registry_manifest") or {}).get("caveats") or [])
    )
    if excluded_database_phases:
        result_warnings.append(
            "The requested phase set excludes database phases; this is a restricted-phase "
            "calculation and must not be presented as global equilibrium."
        )
        result_warnings.sort()
    evidence_payload = {
        "schema_version": EQUILIBRIUM_SCHEMA_VERSION,
        "database_sha256": database_manifest["sha256"],
        "database_manifest_sha256": database_manifest["manifest_sha256"],
        "request": request_record,
        "result": result_record,
        "warnings": result_warnings,
        "pycalphad_version": database_manifest["pycalphad_version"],
    }
    evidence_sha256 = _sha256_bytes(_canonical_json_bytes(evidence_payload))
    response = {
        "schema_version": EQUILIBRIUM_SCHEMA_VERSION,
        "database": database_manifest,
        "request": request_record,
        "result": result_record,
        "warnings": result_warnings,
        "evidence": {
            "sha256": evidence_sha256,
            "algorithm": "sha256",
            "canonicalization": "UTF-8 JSON, sorted keys, compact separators, finite numbers",
            "canonical_serialization": True,
            "solver_replay_determinism_claimed": False,
        },
    }
    encoded = _canonical_json_bytes(response)
    if len(encoded) > result_limit:
        raise CalphadExecutionError(
            f"serialized CALPHAD evidence exceeds max_result_bytes: {len(encoded)} > {result_limit}"
        )
    return response
