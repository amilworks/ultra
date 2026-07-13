"""Closed identity bindings for typed materials-tool trace evidence."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

MAX_TYPED_RESULT_BINDING_BYTES = 1_000_000
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

# Keep the tool/operation relation closed. A completed tool event is useful as
# scientific execution evidence only when its returned operation is one that the
# registered tool can actually perform.
MATERIALS_TYPED_SCIENTIFIC_TOOL_OPERATIONS: dict[str, frozenset[str]] = {
    "calphad_run_equilibrium": frozenset({"equilibrium"}),
    "calphad_run_scheil": frozenset({"scheil"}),
    "materials_analyze_crystal_slip": frozenset({"analytical_slip_geometry"}),
    "materials_calculate_diffraction_profile_metrics": frozenset({"diffraction_profile_metrics"}),
    "materials_convert_uniform_corrosion": frozenset({"faraday_uniform_corrosion_conversion"}),
    "materials_evaluate_mode_i_lefm": frozenset({"mode_i_lefm_screen"}),
    "materials_evaluate_norton_arrhenius_creep": frozenset({"norton_arrhenius_secondary_creep"}),
    "materials_evaluate_oxidation_mass_gain": frozenset({"oxidation_areal_mass_gain"}),
    "materials_fit_held_out_rigid_registration": frozenset({"held_out_rigid_registration"}),
    "materials_fit_paris_law": frozenset({"paris_law_fit"}),
    "materials_run_binary_precipitation_kwn": frozenset({"binary_precipitation_kwn"}),
    "materials_run_diffusion_1d": frozenset({"single_phase_diffusion_1d"}),
    "materials_transport_coefficients": frozenset({"transport_coefficients"}),
    "materials_validate_cpfe_contract": frozenset({"cpfe_contract_validation"}),
}


def _result_mapping(value: Any, *, depth: int = 0) -> Mapping[str, Any] | None:
    """Recover one bounded structured tool result without trusting a preview."""

    if depth > 4:
        return None
    if isinstance(value, Mapping):
        if "operation" in value:
            return value
        for key in ("content", "output", "result"):
            if key in value:
                recovered = _result_mapping(value[key], depth=depth + 1)
                if recovered is not None:
                    return recovered
        return None
    if isinstance(value, str):
        encoded = value.encode("utf-8")
        if not encoded or len(encoded) > MAX_TYPED_RESULT_BINDING_BYTES:
            return None
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return None
        return _result_mapping(parsed, depth=depth + 1)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value[:20]:
            recovered = _result_mapping(item, depth=depth + 1)
            if recovered is not None:
                return recovered
        return None
    content = getattr(value, "content", None)
    if content is not None and content is not value:
        return _result_mapping(content, depth=depth + 1)
    return None


def _artifact_sha256(result: Mapping[str, Any], *, operation: str) -> str:
    for key in ("analysis_artifact", f"{operation}_artifact", "artifact", "result_artifact"):
        artifact = result.get(key)
        if not isinstance(artifact, Mapping):
            continue
        digest = str(artifact.get("sha256") or "").strip().lower()
        if _SHA256.fullmatch(digest):
            return digest
    return ""


def typed_materials_result_binding(tool_name: str, output: Any) -> dict[str, Any]:
    """Extract a fail-closed operation/result identity from a typed tool output."""

    allowed_operations = MATERIALS_TYPED_SCIENTIFIC_TOOL_OPERATIONS.get(tool_name)
    if not allowed_operations:
        return {}
    result = _result_mapping(output)
    if result is None or result.get("ok") is not True:
        return {}
    operation = str(result.get("operation") or "").strip().lower()
    if operation not in allowed_operations:
        return {}
    result_sha256 = _artifact_sha256(result, operation=operation)
    if not result_sha256:
        return {}

    binding: dict[str, Any] = {
        "scientific_operation": operation,
        "result_artifact_sha256": result_sha256,
        "scientific_result_ok": True,
    }
    validation_artifact = result.get("materials_validation_artifact")
    if isinstance(validation_artifact, Mapping):
        validation_sha256 = str(validation_artifact.get("sha256") or "").strip().lower()
        if _SHA256.fullmatch(validation_sha256):
            binding["materials_validation_artifact_sha256"] = validation_sha256
    return binding


__all__ = [
    "MATERIALS_TYPED_SCIENTIFIC_TOOL_OPERATIONS",
    "typed_materials_result_binding",
]
