#!/usr/bin/env python3
"""Reviewed semantic repairs for known defects in pinned MatTools validators.

The upstream score remains historical evidence.  This module only computes the
additional fail-closed scientific score used by Ultra's promotion policy.  Its
rules are intentionally narrow, versioned, and tied to the pinned benchmark
snapshot; unknown tasks retain the pre-normalization strict-shadow score.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

REPAIR_SPEC_VERSION = "ultra.mattools.semantic-repairs.v1"

_FULL_TASK_REPAIRS = {
    "test_topography_analyzer": "inverted_type_predicate",
    "test_defect_band_raises": "inverted_type_predicate",
    "test_chgcar_insertion": "zip_without_outer_length_check",
}

# Python bool is a subclass of int.  These exact pinned properties can be
# falsely accepted as numeric when True == 1 or False == 0.
_BOOL_NUMERIC_REPAIRS: dict[str, tuple[tuple[str, float, str], ...]] = {
    "test_HarmonicDefect": (("spin_index", 1, "int"),),
    "test_interstitial_generator": (("number_of_interstitials", 1, "int"),),
    "test_multi": (("Formation_Energy_Diagrams_Count", 1, "int"),),
    "test_vacancy_generators": (("vacancy_count_for_specific_species", 1, "int"),),
    "test_formation_energy_diagram_numerical": (("formation_energy", 1.0, "real"),),
    "test_defect_entry": (("freysoldt_correction", 0.0, "real"),),
    "test_freysoldt": (("freysoldt_correction_energy", 0.0, "real"),),
    "test_kumagai": (("correction_energy_neutral", 0.0, "real"),),
}


def _exact_bool(value: Any, expected: bool) -> bool:
    return type(value) is bool and value is expected


def _exact_int(value: Any, expected: int) -> bool:
    return type(value) is int and value == expected


def _exact_str(value: Any, expected: str) -> bool:
    return type(value) is str and value == expected


def _finite_numeric_array(value: Any) -> np.ndarray[Any, Any] | None:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return None
    if array.dtype.kind not in {"i", "u", "f", "c"} or array.dtype.kind == "b":
        return None
    try:
        if not bool(np.all(np.isfinite(array))):
            return None
    except TypeError:
        return None
    return array


def _allclose(value: Any, expected: Any) -> bool:
    actual_array = _finite_numeric_array(value)
    expected_array = _finite_numeric_array(expected)
    if actual_array is None or expected_array is None or actual_array.shape != expected_array.shape:
        return False
    return bool(np.allclose(actual_array, expected_array))


def _topography_score(generated: dict[str, Any]) -> tuple[int, dict[str, bool]]:
    fields = {
        "dummy_sites_count": _exact_int(generated.get("dummy_sites_count"), 100),
        "value_error_check": _exact_bool(generated.get("value_error_check"), True),
    }
    return sum(fields.values()), fields


def _defect_band_score(generated: dict[str, Any]) -> tuple[int, dict[str, bool]]:
    fields = {
        "defect_band_index_mismatch": _exact_str(
            generated.get("defect_band_index_mismatch"), "Raises ValueError"
        ),
        "defect_spin_index_mismatch": _exact_str(
            generated.get("defect_spin_index_mismatch"), "Raises ValueError"
        ),
    }
    return sum(fields.values()), fields


def _chgcar_score(generated: dict[str, Any]) -> tuple[int, dict[str, bool]]:
    expected_positions = (
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.5), (0.0, 0.5, 0.0), (0.5, 0.0, 0.0)),
        ((0.375, 0.375, 0.375), (0.625, 0.625, 0.625)),
    )
    positions = generated.get("insertion_site_positions")
    positions_valid = isinstance(positions, (list, tuple)) and len(positions) == len(
        expected_positions
    )
    if positions_valid:
        positions_valid = all(
            _allclose(actual, expected)
            for actual, expected in zip(positions, expected_positions, strict=True)
        )
    fields = {
        "average_charge": _allclose(
            generated.get("average_charge"),
            (0.03692438178614583, 0.10068764899215804),
        ),
        "insertion_site_positions": positions_valid,
    }
    return sum(fields.values()), fields


def repair_task_score(
    *,
    task_id: str,
    generated: Any,
    upstream_strict_scientific_pass: int,
    subtask_count: int,
) -> dict[str, Any]:
    """Return a deterministic repaired score for one pinned MatTools task."""

    if (
        not isinstance(upstream_strict_scientific_pass, int)
        or isinstance(upstream_strict_scientific_pass, bool)
        or not isinstance(subtask_count, int)
        or isinstance(subtask_count, bool)
        or not 0 <= upstream_strict_scientific_pass <= subtask_count
        or subtask_count < 1
    ):
        raise ValueError("invalid upstream strict score or subtask denominator")

    repaired = upstream_strict_scientific_pass
    field_verdicts: dict[str, bool] = {}
    defects: list[str] = []
    if task_id in _FULL_TASK_REPAIRS:
        if not isinstance(generated, dict):
            repaired = 0
        elif task_id == "test_topography_analyzer":
            repaired, field_verdicts = _topography_score(generated)
        elif task_id == "test_defect_band_raises":
            repaired, field_verdicts = _defect_band_score(generated)
        else:
            repaired, field_verdicts = _chgcar_score(generated)
        defects.append(_FULL_TASK_REPAIRS[task_id])

    numeric_specs = _BOOL_NUMERIC_REPAIRS.get(task_id, ())
    bool_false_accepts = 0
    if numeric_specs:
        defects.append("bool_is_subclass_of_int")
        for property_name, expected, numeric_kind in numeric_specs:
            value = generated.get(property_name) if isinstance(generated, dict) else None
            if numeric_kind == "int":
                verdict = _exact_int(value, int(expected))
                falsely_accepted = type(value) is bool and int(value) == int(expected)
            else:
                verdict = (
                    type(value) in {int, float}
                    and type(value) is not bool
                    and math.isfinite(float(value))
                    and bool(np.allclose(float(value), expected))
                )
                falsely_accepted = type(value) is bool and bool(np.allclose(value, expected))
            field_verdicts[property_name] = verdict
            bool_false_accepts += int(falsely_accepted)
        repaired = max(0, repaired - bool_false_accepts)

    if not 0 <= repaired <= subtask_count:
        raise ValueError("semantic repair produced an out-of-range score")
    return {
        "repair_spec_version": REPAIR_SPEC_VERSION,
        "repair_applied": bool(defects),
        "validator_defects": defects,
        "field_verdicts": field_verdicts,
        "bool_false_accept_count": bool_false_accepts,
        "upstream_strict_scientific_pass": upstream_strict_scientific_pass,
        "repaired_scientific_pass": repaired,
        "repaired_scientific_fail": subtask_count - repaired,
    }


__all__ = ["REPAIR_SPEC_VERSION", "repair_task_score"]
