"""Closed, provenance-bound inputs for the isolated Kawin runtime."""

from __future__ import annotations

import hashlib
import math
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import KineticsInputError, KineticsUnsupportedError

REQUEST_SCHEMA_VERSION = "ultra.materials.kinetics-request.v1"
RESULT_SCHEMA_VERSION = "ultra.materials.kinetics-result.v1"
MAX_DATABASE_BYTES = 50 * 1024 * 1024
MAX_REQUEST_BYTES = 1024 * 1024
MAX_RESULT_BYTES = 64 * 1024 * 1024
MAX_WALL_TIME_SECONDS = 600.0
MIN_COMPONENT_FRACTION = 1e-10
QUALIFIED_PRESSURE_PA = 101325.0

_TOKEN_RE = re.compile(r"^[A-Z][A-Z0-9_+\-]{0,63}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class InspectedDatabase:
    """A parsed database built from the exact bytes whose digest was checked."""

    database: Any
    manifest: dict[str, Any]
    kinetic_inventory: dict[str, dict[str, Any]]


def require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise KineticsInputError(f"{field} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise KineticsInputError(f"{field} keys must be strings")
    return value


def require_keys(
    value: Mapping[str, Any],
    *,
    field: str,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    missing = sorted(required - set(value))
    extra = sorted(set(value) - required - optional)
    if missing:
        raise KineticsInputError(f"{field} is missing required keys: {', '.join(missing)}")
    if extra:
        raise KineticsInputError(f"{field} has unknown keys: {', '.join(extra)}")


def require_nonempty_string(value: Any, *, field: str, maximum: int = 4096) -> str:
    if not isinstance(value, str):
        raise KineticsInputError(f"{field} must be a string")
    normalized = value.strip()
    if not normalized:
        raise KineticsInputError(f"{field} must not be empty")
    if len(normalized) > maximum:
        raise KineticsInputError(f"{field} exceeds {maximum} characters")
    return normalized


def require_token(value: Any, *, field: str) -> str:
    token = require_nonempty_string(value, field=field, maximum=64).upper()
    if _TOKEN_RE.fullmatch(token) is None:
        raise KineticsInputError(f"{field} is not a valid normalized materials token")
    return token


def require_float(
    value: Any,
    *,
    field: str,
    minimum: float,
    maximum: float,
    include_minimum: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise KineticsInputError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise KineticsInputError(f"{field} must be finite")
    below = result < minimum if include_minimum else result <= minimum
    if below or result > maximum:
        bound = "[" if include_minimum else "("
        raise KineticsInputError(f"{field} must be in {bound}{minimum}, {maximum}]")
    return 0.0 if result == 0 else result


def require_int(value: Any, *, field: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise KineticsInputError(f"{field} must be an integer")
    if value < minimum or value > maximum:
        raise KineticsInputError(f"{field} must be in [{minimum}, {maximum}]")
    return int(value)


def require_bool(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise KineticsInputError(f"{field} must be a boolean")
    return value


def require_numeric_sequence(
    value: Any,
    *,
    field: str,
    minimum_length: int,
    maximum_length: int,
    minimum: float,
    maximum: float,
) -> list[float]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise KineticsInputError(f"{field} must be an array")
    if len(value) < minimum_length or len(value) > maximum_length:
        raise KineticsInputError(f"{field} length must be in [{minimum_length}, {maximum_length}]")
    return [
        require_float(item, field=f"{field}[{index}]", minimum=minimum, maximum=maximum)
        for index, item in enumerate(value)
    ]


def normalize_components(value: Any) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise KineticsInputError("components must be an array")
    components = tuple(require_token(item, field="components[]") for item in value)
    if len(components) < 2 or len(components) > 8:
        raise KineticsInputError("components must contain 2 to 8 physical components")
    if len(set(components)) != len(components):
        raise KineticsInputError("components must be unique")
    if any(component in {"VA", "/-"} for component in components):
        raise KineticsInputError("components must contain physical components only, not VA or /-")
    return components


def normalize_composition(
    value: Any,
    *,
    components: Sequence[str],
) -> tuple[dict[str, float], dict[str, float]]:
    mapping = require_mapping(value, field="independent_composition_mole_fraction")
    independent = tuple(components[1:])
    if set(mapping) != set(independent):
        raise KineticsInputError(
            "independent_composition_mole_fraction must contain exactly every component "
            "after the first reference component"
        )
    normalized = {
        component: require_float(
            mapping[component],
            field=f"independent_composition_mole_fraction.{component}",
            minimum=MIN_COMPONENT_FRACTION,
            maximum=1.0 - MIN_COMPONENT_FRACTION,
        )
        for component in independent
    }
    dependent = 1.0 - math.fsum(normalized.values())
    if dependent < MIN_COMPONENT_FRACTION:
        raise KineticsInputError(
            "independent mole fractions leave the reference component below the minimum"
        )
    full = {components[0]: dependent, **normalized}
    if not math.isclose(math.fsum(full.values()), 1.0, rel_tol=0.0, abs_tol=2e-15):
        raise KineticsInputError("bulk composition does not close to one")
    return normalized, full


def normalize_limits(value: Any) -> dict[str, Any]:
    mapping = require_mapping(value, field="limits")
    require_keys(
        mapping,
        field="limits",
        required={"wall_time_seconds", "max_result_bytes"},
    )
    return {
        "wall_time_seconds": require_float(
            mapping["wall_time_seconds"],
            field="limits.wall_time_seconds",
            minimum=0.1,
            maximum=MAX_WALL_TIME_SECONDS,
        ),
        "max_result_bytes": require_int(
            mapping["max_result_bytes"],
            field="limits.max_result_bytes",
            minimum=1024,
            maximum=MAX_RESULT_BYTES,
        ),
    }


def normalize_temperature_limits(value: Any, *, field: str) -> list[float]:
    limits = require_numeric_sequence(
        value,
        field=field,
        minimum_length=2,
        maximum_length=2,
        minimum=1.0,
        maximum=10_000.0,
    )
    if limits[0] >= limits[1]:
        raise KineticsInputError(f"{field} must be strictly increasing")
    return limits


def normalize_pressure_limits(value: Any, *, field: str) -> list[float]:
    limits = require_numeric_sequence(
        value,
        field=field,
        minimum_length=2,
        maximum_length=2,
        minimum=1e-9,
        maximum=1e12,
    )
    if limits[0] > limits[1]:
        raise KineticsInputError(f"{field} must be non-decreasing")
    if not limits[0] <= QUALIFIED_PRESSURE_PA <= limits[1]:
        raise KineticsInputError(
            f"{field} does not contain the fixed {QUALIFIED_PRESSURE_PA} Pa runtime pressure"
        )
    return limits


def safe_existing_file(raw_path: Any, *, workspace_root: Path, field: str) -> Path:
    text = require_nonempty_string(raw_path, field=field, maximum=4096)
    root = workspace_root.resolve(strict=True)
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        if candidate.is_symlink():
            raise KineticsInputError(f"{field} must not be a symbolic link")
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except FileNotFoundError as exc:
        raise KineticsInputError(f"{field} does not exist") from exc
    except ValueError as exc:
        raise KineticsInputError(f"{field} escapes the workspace root") from exc
    mode = resolved.stat().st_mode
    if not stat.S_ISREG(mode):
        raise KineticsInputError(f"{field} must be a regular file")
    return resolved


def _kinetic_inventory(database: Any, phases: Sequence[str]) -> dict[str, dict[str, Any]]:
    from tinydb import where

    result: dict[str, dict[str, Any]] = {}
    for phase in phases:
        rows = database.search(
            (where("phase_name") == phase)
            & (
                (where("parameter_type") == "MF")
                | (where("parameter_type") == "MQ")
                | (where("parameter_type") == "DF")
                | (where("parameter_type") == "DQ")
            )
        )
        family_species: dict[str, set[str]] = {"mobility": set(), "diffusivity": set()}
        family_types: dict[str, set[str]] = {"mobility": set(), "diffusivity": set()}
        family_counts = {"mobility": 0, "diffusivity": 0}
        for row in rows:
            parameter_type = str(row.get("parameter_type") or "").upper()
            family = "mobility" if parameter_type in {"MF", "MQ"} else "diffusivity"
            family_types[family].add(parameter_type)
            family_counts[family] += 1
            species = row.get("diffusing_species")
            if species is not None:
                family_species[family].add(str(species).upper())
        result[phase] = {
            "mobility": {
                "parameter_types": sorted(family_types["mobility"]),
                "diffusing_species": sorted(family_species["mobility"]),
                "parameter_count": family_counts["mobility"],
            },
            "diffusivity": {
                "parameter_types": sorted(family_types["diffusivity"]),
                "diffusing_species": sorted(family_species["diffusivity"]),
                "parameter_count": family_counts["diffusivity"],
            },
        }
    return result


def inspect_database(
    value: Any,
    *,
    workspace_root: Path,
    components: Sequence[str],
    phases: Sequence[str],
) -> InspectedDatabase:
    mapping = require_mapping(value, field="database")
    require_keys(
        mapping,
        field="database",
        required={
            "path",
            "sha256",
            "size_bytes",
            "artifact_id",
            "source",
            "license_id",
            "assessment_scope",
            "reference_state",
            "assessment_temperature_limits_K",
            "assessment_pressure_limits_Pa",
        },
    )
    path = safe_existing_file(mapping["path"], workspace_root=workspace_root, field="database.path")
    if path.suffix.casefold() != ".tdb":
        raise KineticsInputError("database.path must have a .tdb suffix")
    expected_size = require_int(
        mapping["size_bytes"],
        field="database.size_bytes",
        minimum=1,
        maximum=MAX_DATABASE_BYTES,
    )
    digest = require_nonempty_string(mapping["sha256"], field="database.sha256", maximum=64)
    if _SHA256_RE.fullmatch(digest) is None:
        raise KineticsInputError("database.sha256 must be a lowercase SHA-256 digest")
    payload = path.read_bytes()
    if len(payload) != expected_size:
        raise KineticsInputError("database.size_bytes does not match the staged database")
    actual_digest = hashlib.sha256(payload).hexdigest()
    if actual_digest != digest:
        raise KineticsInputError("database.sha256 does not match the staged database")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise KineticsInputError("database must be a UTF-8 TDB") from exc
    try:
        from pycalphad import Database

        database = Database.from_string(text, fmt="tdb")
    except Exception as exc:
        raise KineticsInputError("database is not a parseable TDB") from exc

    database_elements = {str(element).upper() for element in database.elements}
    missing_components = sorted(set(components) - database_elements)
    if missing_components:
        raise KineticsInputError(
            f"database does not declare requested components: {', '.join(missing_components)}"
        )
    database_phases = {str(phase).upper() for phase in database.phases}
    missing_phases = sorted(set(phases) - database_phases)
    if missing_phases:
        raise KineticsInputError(
            f"database does not declare requested phases: {', '.join(missing_phases)}"
        )

    temperature_limits = normalize_temperature_limits(
        mapping["assessment_temperature_limits_K"],
        field="database.assessment_temperature_limits_K",
    )
    pressure_limits = normalize_pressure_limits(
        mapping["assessment_pressure_limits_Pa"],
        field="database.assessment_pressure_limits_Pa",
    )
    manifest = {
        "artifact_id": require_nonempty_string(
            mapping["artifact_id"], field="database.artifact_id"
        ),
        "source": require_nonempty_string(mapping["source"], field="database.source"),
        "license_id": require_nonempty_string(mapping["license_id"], field="database.license_id"),
        "assessment_scope": require_nonempty_string(
            mapping["assessment_scope"], field="database.assessment_scope"
        ),
        "reference_state": require_nonempty_string(
            mapping["reference_state"], field="database.reference_state"
        ),
        "sha256": actual_digest,
        "size_bytes": len(payload),
        "format": "tdb",
        "assessment_temperature_limits_K": temperature_limits,
        "assessment_pressure_limits_Pa": pressure_limits,
        "requested_components": list(components),
        "requested_phases": list(phases),
    }
    inventory = _kinetic_inventory(database, phases)
    return InspectedDatabase(database=database, manifest=manifest, kinetic_inventory=inventory)


def select_transport_family(
    inventory: Mapping[str, Any],
    *,
    components: Sequence[str],
    binary_solute_only_ok: bool,
) -> str:
    mobility = require_mapping(inventory.get("mobility"), field="kinetic_inventory.mobility")
    diffusivity = require_mapping(
        inventory.get("diffusivity"), field="kinetic_inventory.diffusivity"
    )
    mobility_species = set(mobility.get("diffusing_species") or [])
    diffusivity_species = set(diffusivity.get("diffusing_species") or [])
    if int(mobility.get("parameter_count") or 0) > 0:
        missing = sorted(set(components) - mobility_species)
        if missing:
            raise KineticsUnsupportedError(
                "selected phase has MF/MQ mobility parameters but lacks requested species: "
                + ", ".join(missing)
            )
        return "MF/MQ mobility"
    if int(diffusivity.get("parameter_count") or 0) > 0:
        if len(components) != 2:
            raise KineticsUnsupportedError(
                "multicomponent DF/DQ execution is unsupported because cross diffusion requires MF/MQ"
            )
        required = {components[1]} if binary_solute_only_ok else set(components)
        missing = sorted(required - diffusivity_species)
        if missing:
            raise KineticsUnsupportedError(
                "selected phase lacks DF/DQ parameters for the binary solute: " + ", ".join(missing)
            )
        return "DF/DQ direct diffusivity"
    raise KineticsUnsupportedError("selected phase has no MF/MQ or DF/DQ kinetic parameters")


def require_temperature_in_assessment(temperature: float, manifest: Mapping[str, Any]) -> None:
    lower, upper = manifest["assessment_temperature_limits_K"]
    if not lower <= temperature <= upper:
        raise KineticsInputError("temperature_K is outside the declared assessment limits")
