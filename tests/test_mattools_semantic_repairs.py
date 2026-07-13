from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "mattools_semantic_repairs.py"
SPEC = importlib.util.spec_from_file_location("mattools_semantic_repairs", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
repairs = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = repairs
SPEC.loader.exec_module(repairs)


def _score(task_id: str, generated: object, passed: int = 2, total: int = 2) -> dict[str, object]:
    return repairs.repair_task_score(
        task_id=task_id,
        generated=generated,
        upstream_strict_scientific_pass=passed,
        subtask_count=total,
    )


def test_repairs_inverted_type_predicates_from_scientific_values() -> None:
    topography = _score(
        "test_topography_analyzer",
        {"dummy_sites_count": 100, "value_error_check": True},
        passed=0,
    )
    defect_band = _score(
        "test_defect_band_raises",
        {
            "defect_band_index_mismatch": "Raises ValueError",
            "defect_spin_index_mismatch": "Raises ValueError",
        },
        passed=0,
    )

    assert topography["repaired_scientific_pass"] == 2
    assert defect_band["repaired_scientific_pass"] == 2
    assert _score(
        "test_topography_analyzer",
        {"dummy_sites_count": 100.0, "value_error_check": 1},
    )["repaired_scientific_pass"] == 0


def test_repairs_zip_length_bug_without_weakening_numerical_tolerance() -> None:
    average = [0.03692438178614583, 0.10068764899215804]
    first_group = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.5, 0.0], [0.5, 0.0, 0.0]]
    second_group = [[0.375, 0.375, 0.375], [0.625, 0.625, 0.625]]

    truncated = _score(
        "test_chgcar_insertion",
        {"average_charge": average, "insertion_site_positions": [first_group]},
    )
    complete = _score(
        "test_chgcar_insertion",
        {"average_charge": average, "insertion_site_positions": [first_group, second_group]},
    )

    assert truncated["repaired_scientific_pass"] == 1
    assert complete["repaired_scientific_pass"] == 2


@pytest.mark.parametrize(
    ("task_id", "property_name", "value"),
    [
        ("test_HarmonicDefect", "spin_index", True),
        ("test_interstitial_generator", "number_of_interstitials", True),
        ("test_multi", "Formation_Energy_Diagrams_Count", True),
        ("test_vacancy_generators", "vacancy_count_for_specific_species", True),
        ("test_formation_energy_diagram_numerical", "formation_energy", True),
        ("test_defect_entry", "freysoldt_correction", False),
        ("test_freysoldt", "freysoldt_correction_energy", False),
        ("test_kumagai", "correction_energy_neutral", False),
    ],
)
def test_bool_cannot_receive_numeric_scientific_credit(
    task_id: str,
    property_name: str,
    value: bool,
) -> None:
    result = _score(task_id, {property_name: value})

    assert result["bool_false_accept_count"] == 1
    assert result["repaired_scientific_pass"] == 1


def test_unaffected_task_preserves_pre_normalization_strict_score() -> None:
    result = _score("test_unaffected", {"anything": "value"}, passed=1, total=3)

    assert result["repair_applied"] is False
    assert result["repaired_scientific_pass"] == 1


def test_invalid_bool_counter_is_rejected() -> None:
    with pytest.raises(ValueError, match="invalid upstream strict score"):
        repairs.repair_task_score(
            task_id="test_unaffected",
            generated={},
            upstream_strict_scientific_pass=True,
            subtask_count=2,
        )
