from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable
from typing import Any

import pytest
import ultra_deepagents.papers.table_evidence as table_evidence
from ultra_deepagents.papers.table_evidence import (
    PAPER_TABLE_EVIDENCE_SCHEMA,
    PAPER_TABLE_EVIDENCE_SCHEMA_V1,
    PROMPT_INJECTION_NEUTRALITY,
    ObservationStatus,
    PaperTableEvidenceValidationError,
    seal_paper_table_evidence,
    validate_paper_table_evidence,
)


def _digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _payload() -> dict[str, Any]:
    born_125 = "125"
    born_180 = "180"
    return {
        "schema": PAPER_TABLE_EVIDENCE_SCHEMA,
        "source": {
            "pdf_sha256": "a" * 64,
            "page": 7,
            "render": {
                "png_sha256": "b" * 64,
                "width_px": 1200,
                "height_px": 1800,
                "zoom": 2,
            },
            "region": {
                "bbox_px": [0, 0, 1200, 1800],
                "png_sha256": "2" * 64,
                "width_px": 1200,
                "height_px": 1800,
            },
        },
        "inference": {
            "model_id": "Qwen/Qwen3.6-27B-Instruct",
            "model_revision": "release-2026-07-01.4f9c2d1",
            "runtime_identity": "sha256:" + "c" * 64,
            "prompt_sha256": "d" * 64,
            "config_sha256": "e" * 64,
            "raw_response_sha256": "f" * 64,
            "deployment_attestation_sha256": "3" * 64,
            "attestation_authority": "materials-platform-ci",
            "response_model_id": "Qwen/Qwen3.6-27B-Instruct",
            "response_system_fingerprint": "fp_mock_immutable",
            "model_input_sha256": "4" * 64,
            "model_input_width_px": 800,
            "model_input_height_px": 1200,
        },
        "extraction_spec": {
            "identity_mode": "specified",
            "scientific_identity_status": "specified",
            "table_id": "table-2",
            "table_label": "Table 2",
            "page": 7,
            "row_bounds": {"minimum": 2, "maximum": 2},
            "column_bounds": {"minimum": 2, "maximum": 2},
            "expected_rows": [
                {"row_id": "alloy-a", "label": "Alloy A"},
                {"row_id": "alloy-b", "label": "Alloy B"},
            ],
            "expected_columns": [
                {"column_id": "yield", "label": "Yield strength", "unit": "MPa"},
                {"column_id": "phase", "label": "Primary phase", "unit": None},
            ],
            "source_region_px": None,
        },
        "table": {
            "table_id": "table-2",
            "rows": [
                {"row_id": "alloy-a", "label": "Alloy A"},
                {"row_id": "alloy-b", "label": "Alloy B"},
            ],
            "columns": [
                {"column_id": "yield", "label": "Yield strength", "unit": "MPa"},
                {"column_id": "phase", "label": "Primary phase", "unit": None},
            ],
            "cells": [
                {
                    "row_id": "alloy-a",
                    "column_id": "yield",
                    "text": "125",
                    "numeric_value": 125,
                    "unit": "MPa",
                    "bbox_px": [100, 200, 260, 240],
                    "observation_status": "cross_checked",
                },
                {
                    "row_id": "alloy-a",
                    "column_id": "phase",
                    "text": "γ′",
                    "numeric_value": None,
                    "unit": None,
                    "bbox_px": [300, 200, 440, 240],
                    "observation_status": "model_observed",
                },
                {
                    "row_id": "alloy-b",
                    "column_id": "yield",
                    "text": "130",
                    "numeric_value": None,
                    "unit": None,
                    "bbox_px": [100, 260, 260, 300],
                    "observation_status": "conflict",
                },
                {
                    "row_id": "alloy-b",
                    "column_id": "phase",
                    "text": None,
                    "numeric_value": None,
                    "unit": None,
                    "bbox_px": [300, 260, 440, 300],
                    "observation_status": "unreadable",
                },
            ],
        },
        "prompt_injection_neutrality": dict(PROMPT_INJECTION_NEUTRALITY),
        "born_digital_cross_check": {
            "extractor_id": "pymupdf-text-blocks",
            "extractor_revision": "1.26.3+pipeline.2",
            "page_text_sha256": "1" * 64,
            "cells": [
                {
                    "row_id": "alloy-a",
                    "column_id": "yield",
                    "text": born_125,
                    "text_sha256": _digest_text(born_125),
                },
                {
                    "row_id": "alloy-b",
                    "column_id": "yield",
                    "text": born_180,
                    "text_sha256": _digest_text(born_180),
                },
            ],
        },
    }


def _expect_error(payload: dict[str, Any], code: str) -> PaperTableEvidenceValidationError:
    with pytest.raises(PaperTableEvidenceValidationError) as caught:
        seal_paper_table_evidence(payload)
    assert caught.value.code == code
    return caught.value


def test_valid_envelope_binds_source_qwen_cells_cross_checks_and_canonical_sha() -> None:
    sealed = seal_paper_table_evidence(_payload())
    validated = validate_paper_table_evidence(sealed)

    assert validated.source.pdf_sha256 == "a" * 64
    assert validated.source.page == 7
    assert validated.source.render.png_sha256 == "b" * 64
    assert validated.source.render.zoom == 2.0
    assert validated.inference.model_id == "Qwen/Qwen3.6-27B-Instruct"
    assert validated.inference.runtime_identity == "sha256:" + "c" * 64
    assert validated.table.cells[0].numeric_value == 125.0
    assert validated.table.cells[0].observation_status is ObservationStatus.CROSS_CHECKED
    assert validated.table.cells[2].observation_status is ObservationStatus.CONFLICT
    assert validated.table.cells[3].bbox_px is not None
    assert validated.prompt_injection_neutrality.validator_enforcement == "metadata_only"

    unsigned = copy.deepcopy(sealed)
    declared = unsigned.pop("evidence_sha256")
    canonical = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    assert declared == hashlib.sha256(canonical).hexdigest()
    assert validated.as_dict() == sealed


def test_cell_and_cross_check_input_order_do_not_change_canonical_evidence() -> None:
    first = _payload()
    second = copy.deepcopy(first)
    second["table"]["cells"].reverse()
    second["born_digital_cross_check"]["cells"].reverse()

    sealed_first = seal_paper_table_evidence(first)
    sealed_second = seal_paper_table_evidence(second)

    assert sealed_first == sealed_second
    assert sealed_first["evidence_sha256"] == sealed_second["evidence_sha256"]


def test_sealed_evidence_rejects_semantic_tampering() -> None:
    sealed = seal_paper_table_evidence(_payload())
    sealed["table"]["cells"][0]["numeric_value"] = 126.0
    sealed["table"]["cells"][0]["text"] = "126"
    sealed["born_digital_cross_check"]["cells"][0]["text"] = "126"
    sealed["born_digital_cross_check"]["cells"][0]["text_sha256"] = _digest_text("126")

    with pytest.raises(PaperTableEvidenceValidationError) as caught:
        validate_paper_table_evidence(sealed)
    assert caught.value.code == "evidence_sha256_mismatch"


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda value: value.update({"extra": True}), "unexpected_keys"),
        (lambda value: value["source"].update({"filename": "paper.pdf"}), "unexpected_keys"),
        (
            lambda value: value["table"]["cells"][0].update({"uncertainty": 2.5}),
            "unsupported_uncertainty",
        ),
    ],
)
def test_every_schema_level_is_closed_and_v1_rejects_invented_uncertainty(
    mutate: Callable[[dict[str, Any]], Any],
    code: str,
) -> None:
    payload = _payload()
    mutate(payload)
    _expect_error(payload, code)


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (
            lambda value: value["source"]["render"].update({"zoom": float("nan")}),
            "nonfinite_number",
        ),
        (
            lambda value: value["table"]["cells"][0].update({"numeric_value": float("inf")}),
            "nonfinite_number",
        ),
        (
            lambda value: value["table"]["cells"][0].update({"bbox_px": [0, 0, float("nan"), 10]}),
            "nonfinite_number",
        ),
        (lambda value: value["source"]["render"].update({"zoom": 0}), "invalid_render_zoom"),
    ],
)
def test_nonfinite_or_nonpositive_geometry_values_fail_closed(
    mutate: Callable[[dict[str, Any]], Any],
    code: str,
) -> None:
    payload = _payload()
    mutate(payload)
    _expect_error(payload, code)


@pytest.mark.parametrize(
    "bbox",
    [
        [-1, 0, 10, 10],
        [10, 0, 10, 10],
        [0, 20, 10, 10],
        [0, 0, 1201, 10],
        [0, 0, 10, 1801],
    ],
)
def test_bbox_must_have_positive_area_inside_exact_render(bbox: list[float]) -> None:
    payload = _payload()
    payload["table"]["cells"][0]["bbox_px"] = bbox
    _expect_error(payload, "bbox_out_of_bounds")


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda value: value["table"]["cells"].pop(), "incomplete_rectangular_grid"),
        (
            lambda value: value["table"]["cells"][3].update(
                {"row_id": "alloy-a", "column_id": "phase"}
            ),
            "duplicate_cell",
        ),
        (
            lambda value: value["table"]["cells"][0].update({"row_id": "missing-row"}),
            "unknown_row_id",
        ),
        (
            lambda value: value["table"]["rows"][1].update({"row_id": "alloy-a"}),
            "duplicate_row_id",
        ),
        (
            lambda value: value["table"]["columns"][1].update({"column_id": "yield"}),
            "duplicate_column_id",
        ),
    ],
)
def test_grid_requires_unique_axes_and_one_cell_per_rectangular_coordinate(
    mutate: Callable[[dict[str, Any]], Any],
    code: str,
) -> None:
    payload = _payload()
    mutate(payload)
    _expect_error(payload, code)


@pytest.mark.parametrize(
    ("field", "replacement", "code"),
    [
        ("text", None, "numeric_without_text"),
        ("bbox_px", None, "numeric_without_bbox"),
        ("unit", None, "numeric_unit_unbound"),
    ],
)
def test_numeric_values_require_observed_text_bbox_and_explicit_units(
    field: str,
    replacement: Any,
    code: str,
) -> None:
    payload = _payload()
    payload["table"]["cells"][0][field] = replacement
    _expect_error(payload, code)


def test_numeric_value_must_equal_the_single_observed_text_literal() -> None:
    payload = _payload()
    payload["table"]["cells"][0]["numeric_value"] = 999

    _expect_error(payload, "numeric_text_mismatch")


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ("125 ± 2", "nonliteral_numeric_text"),
        ("125 +/- 2", "nonliteral_numeric_text"),
        ("<125", "nonliteral_numeric_text"),
        ("approximately 125", "nonliteral_numeric_text"),
        ("about 125", "nonliteral_numeric_text"),
        ("120-130", "ambiguous_numeric_text"),
        ("not measured", "numeric_text_unparseable"),
    ],
)
def test_ambiguous_or_nonexact_text_cannot_expose_a_numeric_value(
    text: str,
    code: str,
) -> None:
    payload = _payload()
    payload["table"]["cells"][0]["text"] = text

    _expect_error(payload, code)


@pytest.mark.parametrize(
    ("text", "numeric_value"),
    [
        ("1.25e2", 125),
        ("1,250.0", 1250),
        ("−1.25e2", -125),
        ("125 MPa", 125),
    ],
)
def test_exact_decimal_scientific_and_grouped_literals_are_bound(
    text: str,
    numeric_value: float,
) -> None:
    payload = _payload()
    payload["table"]["cells"][0]["text"] = text
    payload["table"]["cells"][0]["numeric_value"] = numeric_value
    cross_check = payload["born_digital_cross_check"]["cells"][0]
    cross_check["text"] = text
    cross_check["text_sha256"] = _digest_text(text)

    sealed = seal_paper_table_evidence(payload)
    assert validate_paper_table_evidence(sealed).table.cells[0].numeric_value == numeric_value


def test_dimensionless_numeric_value_must_use_explicit_one_unit() -> None:
    payload = _payload()
    payload["table"]["columns"][0]["unit"] = "1"
    payload["table"]["cells"][0]["unit"] = "1"
    payload["extraction_spec"]["expected_columns"][0]["unit"] = "1"
    sealed = seal_paper_table_evidence(payload)
    assert validate_paper_table_evidence(sealed).table.cells[0].unit == "1"


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda value: value["table"]["cells"][0].update({"unit": "GPa"}), "unit_mismatch"),
        (
            lambda value: value["table"]["columns"][0].update({"unit": "unknown"}),
            "invalid_unit",
        ),
        (
            lambda value: value["table"]["columns"][0].update({"unit": " MPa"}),
            "invalid_unit",
        ),
    ],
)
def test_units_are_explicit_and_exact_not_silently_converted(
    mutate: Callable[[dict[str, Any]], Any],
    code: str,
) -> None:
    payload = _payload()
    mutate(payload)
    _expect_error(payload, code)


def test_cross_checked_text_uses_bounded_unicode_and_whitespace_normalization() -> None:
    payload = _payload()
    payload["table"]["cells"][0]["text"] = "  125\n"
    sealed = seal_paper_table_evidence(payload)
    validated = validate_paper_table_evidence(sealed)
    assert validated.table.cells[0].text == "  125\n"


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (
            lambda value: value["born_digital_cross_check"]["cells"].pop(0),
            "cross_check_missing",
        ),
        (
            lambda value: _replace_born_text(value, 0, "126"),
            "cross_check_disagrees",
        ),
        (
            lambda value: _replace_born_text(value, 1, "130"),
            "false_conflict",
        ),
        (
            lambda value: value["table"]["cells"][2].update({"numeric_value": 130, "unit": "MPa"}),
            "unresolved_conflict_value",
        ),
        (
            lambda value: value["table"]["cells"][3].update({"text": "maybe"}),
            "unreadable_cell_has_value",
        ),
        (
            lambda value: value["table"]["cells"][0].update(
                {"observation_status": "model_observed"}
            ),
            "cross_check_status_mismatch",
        ),
    ],
)
def test_observation_statuses_cannot_overstate_resolution(
    mutate: Callable[[dict[str, Any]], Any],
    code: str,
) -> None:
    payload = _payload()
    mutate(payload)
    _expect_error(payload, code)


def _replace_born_text(payload: dict[str, Any], index: int, text: str) -> None:
    check = payload["born_digital_cross_check"]["cells"][index]
    check["text"] = text
    check["text_sha256"] = _digest_text(text)


def test_born_digital_cell_hash_must_bind_exact_utf8_text() -> None:
    payload = _payload()
    payload["born_digital_cross_check"]["cells"][0]["text_sha256"] = "0" * 64
    _expect_error(payload, "cross_check_text_sha256_mismatch")


def test_born_digital_cross_check_optionally_preserves_exact_page_character_span() -> None:
    payload = _payload()
    check = payload["born_digital_cross_check"]["cells"][0]
    check["start_char"] = 42
    check["end_char"] = 45

    sealed = seal_paper_table_evidence(payload)
    validated = validate_paper_table_evidence(sealed)

    assert validated.born_digital_cross_check is not None
    cell = validated.born_digital_cross_check.cells[0]
    assert cell.text == "125"
    assert cell.start_char == 42
    assert cell.end_char == 45
    assert sealed["born_digital_cross_check"]["cells"][0]["start_char"] == 42


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (
            lambda cell: cell.update({"start_char": 42}),
            "cross_check_span_incomplete",
        ),
        (
            lambda cell: cell.update({"start_char": 42, "end_char": 44}),
            "cross_check_span_text_mismatch",
        ),
        (
            lambda cell: cell.update({"start_char": True, "end_char": 4}),
            "invalid_integer",
        ),
    ],
)
def test_born_digital_cross_check_span_must_be_complete_and_match_exact_text(
    mutate: Callable[[dict[str, Any]], Any],
    code: str,
) -> None:
    payload = _payload()
    mutate(payload["born_digital_cross_check"]["cells"][0])
    _expect_error(payload, code)


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("model_id", "Acme/Generic-VLM", "non_qwen_model"),
        ("model_revision", "latest", "mutable_revision"),
        ("runtime_identity", "qwen-runtime-1", "invalid_runtime_identity"),
        ("raw_response_sha256", "ABC", "invalid_sha256"),
    ],
)
def test_qwen_identity_and_response_provenance_must_be_immutable(
    field: str,
    value: str,
    code: str,
) -> None:
    payload = _payload()
    payload["inference"][field] = value
    _expect_error(payload, code)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("content_treatment", "instructions_and_data"),
        ("instruction_authority", "paper"),
        ("validator_enforcement", "runtime_enforced"),
    ],
)
def test_prompt_injection_metadata_is_data_only_and_never_claims_runtime_enforcement(
    field: str,
    value: str,
) -> None:
    payload = _payload()
    payload["prompt_injection_neutrality"][field] = value
    _expect_error(payload, "invalid_prompt_injection_metadata")


def test_born_digital_cross_check_is_optional_for_model_only_observation() -> None:
    payload = _payload()
    del payload["born_digital_cross_check"]
    payload["table"] = {
        "table_id": "single-cell",
        "rows": [{"row_id": "r1", "label": None}],
        "columns": [{"column_id": "phase", "label": "Phase", "unit": None}],
        "cells": [
            {
                "row_id": "r1",
                "column_id": "phase",
                "text": "L1₂",
                "numeric_value": None,
                "unit": None,
                "bbox_px": [10, 10, 100, 40],
                "observation_status": "model_observed",
            }
        ],
    }
    payload["extraction_spec"].update(
        {
            "identity_mode": "specified",
            "scientific_identity_status": "specified",
            "table_id": "single-cell",
            "table_label": "Single cell",
            "row_bounds": {"minimum": 1, "maximum": 1},
            "column_bounds": {"minimum": 1, "maximum": 1},
            "expected_rows": [{"row_id": "r1", "label": None}],
            "expected_columns": [{"column_id": "phase", "label": "Phase", "unit": None}],
        }
    )

    sealed = seal_paper_table_evidence(payload)
    assert validate_paper_table_evidence(sealed).born_digital_cross_check is None


def test_legacy_v1_evidence_remains_readable_without_v2_claims() -> None:
    payload = _payload()
    payload["schema"] = PAPER_TABLE_EVIDENCE_SCHEMA_V1
    del payload["source"]["region"]
    del payload["extraction_spec"]
    for key in (
        "deployment_attestation_sha256",
        "attestation_authority",
        "response_model_id",
        "response_system_fingerprint",
        "model_input_sha256",
        "model_input_width_px",
        "model_input_height_px",
    ):
        del payload["inference"][key]

    sealed = seal_paper_table_evidence(payload)
    validated = validate_paper_table_evidence(sealed)

    assert validated.schema == PAPER_TABLE_EVIDENCE_SCHEMA_V1
    assert validated.source.region is None
    assert validated.extraction_spec is None


def test_generic_mode_is_only_valid_with_explicitly_unverified_identity() -> None:
    payload = _payload()
    spec = payload["extraction_spec"]
    spec.update(
        {
            "identity_mode": "generic_unverified",
            "scientific_identity_status": "unverified",
            "expected_rows": [],
            "expected_columns": [],
        }
    )

    sealed = seal_paper_table_evidence(payload)
    assert sealed["extraction_spec"]["scientific_identity_status"] == "unverified"

    payload = _payload()
    payload["extraction_spec"]["identity_mode"] = "generic_unverified"
    _expect_error(payload, "invalid_scientific_identity_status")


def test_closed_spec_enforces_expected_header_identity() -> None:
    payload = _payload()
    payload["table"]["columns"][0]["label"] = "0.2% proof stress"
    _expect_error(payload, "expected_column_identity_mismatch")


def test_identical_cell_locations_are_rejected() -> None:
    payload = _payload()
    payload["table"]["cells"][1]["bbox_px"] = list(payload["table"]["cells"][0]["bbox_px"])
    _expect_error(payload, "duplicate_cell_bbox")


def test_implausibly_overlapping_cell_locations_are_rejected() -> None:
    payload = _payload()
    payload["table"]["cells"][1]["bbox_px"] = [105, 202, 255, 238]
    _expect_error(payload, "implausible_cell_overlap")


def test_large_rectangular_grid_avoids_quadratic_exact_overlap_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact_checks = 0
    original = table_evidence._bbox_intersection_over_min_area

    def counted(left: table_evidence.PixelBBox, right: table_evidence.PixelBBox) -> float:
        nonlocal exact_checks
        exact_checks += 1
        return original(left, right)

    monkeypatch.setattr(table_evidence, "_bbox_intersection_over_min_area", counted)
    cells = tuple(
        table_evidence.TableCell(
            row_id=f"r{row}",
            column_id=f"c{column}",
            text="x",
            numeric_value=None,
            unit=None,
            bbox_px=table_evidence.PixelBBox(
                2.0 * column,
                2.0 * row,
                2.0 * column + 1.0,
                2.0 * row + 1.0,
            ),
            observation_status=ObservationStatus.MODEL_OBSERVED,
        )
        for row in range(100)
        for column in range(100)
    )

    table_evidence._reject_implausible_cell_overlaps(cells)

    assert exact_checks == 0


def test_pathological_nonrejecting_geometry_fails_at_candidate_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(table_evidence, "MAX_CELL_GEOMETRY_CANDIDATES", 2)
    cells = tuple(
        table_evidence.TableCell(
            row_id=f"r{index}",
            column_id="c0",
            text="x",
            numeric_value=None,
            unit=None,
            bbox_px=table_evidence.PixelBBox(0.0, index * 0.5, 100.0, index * 0.5 + 1.0),
            observation_status=ObservationStatus.MODEL_OBSERVED,
        )
        for index in range(10)
    )

    with pytest.raises(PaperTableEvidenceValidationError) as exc_info:
        table_evidence._reject_implausible_cell_overlaps(cells)

    assert exc_info.value.code == "cell_geometry_complexity_exceeded"


def test_cell_locations_must_follow_declared_column_order() -> None:
    payload = _payload()
    first = payload["table"]["cells"][0]["bbox_px"]
    second = payload["table"]["cells"][1]["bbox_px"]
    payload["table"]["cells"][0]["bbox_px"] = second
    payload["table"]["cells"][1]["bbox_px"] = first
    _expect_error(payload, "cell_column_order_mismatch")


def test_cell_locations_must_follow_declared_row_order() -> None:
    payload = _payload()
    first = payload["table"]["cells"][0]["bbox_px"]
    second_row = payload["table"]["cells"][2]["bbox_px"]
    payload["table"]["cells"][0]["bbox_px"] = second_row
    payload["table"]["cells"][2]["bbox_px"] = first
    _expect_error(payload, "cell_row_order_mismatch")


def test_cells_must_be_inside_the_exact_observation_crop() -> None:
    payload = _payload()
    payload["source"]["region"].update(
        {
            "bbox_px": [0, 0, 500, 500],
            "width_px": 500,
            "height_px": 500,
        }
    )
    payload["extraction_spec"]["source_region_px"] = [0, 0, 500, 500]
    payload["table"]["cells"][0]["bbox_px"] = [600, 200, 700, 240]
    _expect_error(payload, "bbox_outside_observation_region")


def test_effective_model_input_pixels_are_enforced_per_cell() -> None:
    payload = _payload()
    payload["inference"]["model_input_width_px"] = 67
    payload["inference"]["model_input_height_px"] = 100
    _expect_error(payload, "insufficient_effective_cell_resolution")
