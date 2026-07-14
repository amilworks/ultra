"""Fail-closed evidence envelope for tables observed in rendered paper pages.

``ultra.paper-table-evidence.v2`` binds a rectangular table transcription to the
exact PDF, rendered page, observation-region, model-input, independently pinned
deployment attestation, endpoint-reported model identity, and raw model response.
It records a closed scientific extraction specification, cell-level visual
locations, and optional born-digital text comparisons, then seals the normalized
envelope with a canonical SHA-256. Legacy v1 artifacts remain readable.

This module is validation only.  It does not render PDFs, invoke Qwen, perform OCR,
verify that declared hashes correspond to bytes held elsewhere, or enforce
prompt-injection isolation. It does require every exposed numeric value to equal
the one unambiguous finite decimal/scientific literal in its observed cell text;
that consistency check does not prove that the visual transcription itself is true.
The required prompt-injection metadata is deliberately labelled ``metadata_only``:
the caller must enforce the data/instruction boundary in its prompt and inference
runtime.

The v1 cell schema intentionally has no structured uncertainty field.  Literal
uncertainty text can be retained in ``text``, but an uncertainty value cannot be
added unless a future schema defines how it is observed and provenance-bound.
"""

from __future__ import annotations

import hashlib
import heapq
import hmac
import json
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, NoReturn, cast

PAPER_TABLE_EVIDENCE_SCHEMA_V1 = "ultra.paper-table-evidence.v1"
PAPER_TABLE_EVIDENCE_SCHEMA = "ultra.paper-table-evidence.v2"
CANONICAL_JSON_PROFILE = "ultra.canonical-json.v1"

PROMPT_INJECTION_NEUTRALITY = {
    "content_treatment": "data_only",
    "instruction_authority": "none",
    "validator_enforcement": "metadata_only",
}

MAX_ROWS = 10_000
MAX_COLUMNS = 1_000
MAX_CELLS = 100_000
MAX_RENDER_DIMENSION_PX = 100_000
MAX_RENDER_ZOOM = 16.0
MIN_EFFECTIVE_CELL_WIDTH_PX = 12.0
MIN_EFFECTIVE_CELL_HEIGHT_PX = 8.0
MAX_IMPLAUSIBLE_CELL_OVERLAP = 0.80
# Exact rectangle-overlap candidates are bounded even for adversarial layouts. The
# sweep/interval index is O((n + k) log n), where k is the number of actual 2-D
# intersections, and ordinary rectangular tables have k=0. Highly pathological but
# non-rejecting layouts fail closed instead of consuming quadratic time.
MAX_CELL_GEOMETRY_CANDIDATES = 2_000_000

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RUNTIME_IDENTITY_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_QWEN_MODEL_RE = re.compile(r"(?:^|[/_.:-])qwen(?:$|[0-9/_.:-])", re.IGNORECASE)
_MUTABLE_REVISIONS = {"default", "head", "latest", "main", "master", "unknown", "unspecified"}
_UNBOUND_UNITS = {"n/a", "na", "none", "unknown", "unspecified"}
_NUMERIC_LITERAL_RE = re.compile(
    r"(?<![A-Za-z0-9_.])"
    r"(?P<number>[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)"
    r"(?:[eE][+-]?\d+)?)"
    r"(?![A-Za-z0-9_.])"
)
_NONEXACT_NUMERIC_MARKERS = frozenset("<>≤≥~≈≃∼±")
_NONEXACT_NUMERIC_QUALIFIER_RE = re.compile(
    r"(?:"
    r"\+/-|\bplus(?:\s+or)?\s+minus\b|"
    r"\b(?:about|approximately|around|circa|roughly|near|nearly|between|"
    r"less\s+than|greater\s+than|at\s+least|at\s+most|more\s+than|"
    r"no\s+more\s+than|no\s+less\s+than|up\s+to|below|above|under|over)\b|"
    r"\bapprox\.?|\bca\."
    r")",
    re.IGNORECASE,
)


class ObservationStatus(str, Enum):
    """Trust state for a single visually located table cell."""

    MODEL_OBSERVED = "model_observed"
    CROSS_CHECKED = "cross_checked"
    CONFLICT = "conflict"
    UNREADABLE = "unreadable"


class PaperTableEvidenceValidationError(ValueError):
    """Stable, machine-classifiable paper-table evidence failure."""

    def __init__(self, code: str, path: str, message: str) -> None:
        self.code = code
        self.path = path
        self.message = message
        super().__init__(f"{code} at {path}: {message}")


@dataclass(frozen=True, slots=True)
class PixelBBox:
    x0: float
    y0: float
    x1: float
    y1: float

    def as_list(self) -> list[float]:
        return [self.x0, self.y0, self.x1, self.y1]


@dataclass(frozen=True, slots=True)
class RenderBinding:
    png_sha256: str
    width_px: int
    height_px: int
    zoom: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "png_sha256": self.png_sha256,
            "width_px": self.width_px,
            "height_px": self.height_px,
            "zoom": self.zoom,
        }


@dataclass(frozen=True, slots=True)
class RegionBinding:
    bbox_px: PixelBBox
    png_sha256: str
    width_px: int
    height_px: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "bbox_px": self.bbox_px.as_list(),
            "png_sha256": self.png_sha256,
            "width_px": self.width_px,
            "height_px": self.height_px,
        }


@dataclass(frozen=True, slots=True)
class SourceBinding:
    pdf_sha256: str
    page: int
    render: RenderBinding
    region: RegionBinding | None = None

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "pdf_sha256": self.pdf_sha256,
            "page": self.page,
            "render": self.render.as_dict(),
        }
        if self.region is not None:
            result["region"] = self.region.as_dict()
        return result


@dataclass(frozen=True, slots=True)
class QwenInferenceBinding:
    model_id: str
    model_revision: str
    runtime_identity: str
    prompt_sha256: str
    config_sha256: str
    raw_response_sha256: str
    deployment_attestation_sha256: str | None = None
    attestation_authority: str | None = None
    response_model_id: str | None = None
    response_system_fingerprint: str | None = None
    model_input_sha256: str | None = None
    model_input_width_px: int | None = None
    model_input_height_px: int | None = None

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "runtime_identity": self.runtime_identity,
            "prompt_sha256": self.prompt_sha256,
            "config_sha256": self.config_sha256,
            "raw_response_sha256": self.raw_response_sha256,
        }
        if self.deployment_attestation_sha256 is not None:
            result.update(
                {
                    "deployment_attestation_sha256": self.deployment_attestation_sha256,
                    "attestation_authority": self.attestation_authority,
                    "response_model_id": self.response_model_id,
                    "response_system_fingerprint": self.response_system_fingerprint,
                    "model_input_sha256": self.model_input_sha256,
                    "model_input_width_px": self.model_input_width_px,
                    "model_input_height_px": self.model_input_height_px,
                }
            )
        return result


@dataclass(frozen=True, slots=True)
class AxisBounds:
    minimum: int
    maximum: int

    def as_dict(self) -> dict[str, int]:
        return {"minimum": self.minimum, "maximum": self.maximum}


@dataclass(frozen=True, slots=True)
class ExpectedRowIdentity:
    row_id: str
    label: str | None

    def as_dict(self) -> dict[str, Any]:
        return {"row_id": self.row_id, "label": self.label}


@dataclass(frozen=True, slots=True)
class ExpectedColumnIdentity:
    column_id: str
    label: str | None
    unit: str | None

    def as_dict(self) -> dict[str, Any]:
        return {"column_id": self.column_id, "label": self.label, "unit": self.unit}


@dataclass(frozen=True, slots=True)
class ExtractionSpecBinding:
    identity_mode: str
    scientific_identity_status: str
    table_id: str
    table_label: str | None
    page: int
    row_bounds: AxisBounds
    column_bounds: AxisBounds
    expected_rows: tuple[ExpectedRowIdentity, ...]
    expected_columns: tuple[ExpectedColumnIdentity, ...]
    source_region_px: PixelBBox | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "identity_mode": self.identity_mode,
            "scientific_identity_status": self.scientific_identity_status,
            "table_id": self.table_id,
            "table_label": self.table_label,
            "page": self.page,
            "row_bounds": self.row_bounds.as_dict(),
            "column_bounds": self.column_bounds.as_dict(),
            "expected_rows": [row.as_dict() for row in self.expected_rows],
            "expected_columns": [column.as_dict() for column in self.expected_columns],
            "source_region_px": (
                self.source_region_px.as_list() if self.source_region_px is not None else None
            ),
        }


@dataclass(frozen=True, slots=True)
class TableRow:
    row_id: str
    label: str | None

    def as_dict(self) -> dict[str, Any]:
        return {"row_id": self.row_id, "label": self.label}


@dataclass(frozen=True, slots=True)
class TableColumn:
    column_id: str
    label: str | None
    unit: str | None

    def as_dict(self) -> dict[str, Any]:
        return {"column_id": self.column_id, "label": self.label, "unit": self.unit}


@dataclass(frozen=True, slots=True)
class TableCell:
    row_id: str
    column_id: str
    text: str | None
    numeric_value: float | None
    unit: str | None
    bbox_px: PixelBBox | None
    observation_status: ObservationStatus

    @property
    def coordinate(self) -> tuple[str, str]:
        return (self.row_id, self.column_id)

    def as_dict(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "column_id": self.column_id,
            "text": self.text,
            "numeric_value": self.numeric_value,
            "unit": self.unit,
            "bbox_px": self.bbox_px.as_list() if self.bbox_px is not None else None,
            "observation_status": self.observation_status.value,
        }


@dataclass(frozen=True, slots=True)
class BornDigitalCellCheck:
    row_id: str
    column_id: str
    text: str
    text_sha256: str
    start_char: int | None = None
    end_char: int | None = None

    @property
    def coordinate(self) -> tuple[str, str]:
        return (self.row_id, self.column_id)

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "row_id": self.row_id,
            "column_id": self.column_id,
            "text": self.text,
            "text_sha256": self.text_sha256,
        }
        if self.start_char is not None and self.end_char is not None:
            result["start_char"] = self.start_char
            result["end_char"] = self.end_char
        return result


@dataclass(frozen=True, slots=True)
class BornDigitalCrossCheck:
    extractor_id: str
    extractor_revision: str
    page_text_sha256: str
    cells: tuple[BornDigitalCellCheck, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "extractor_id": self.extractor_id,
            "extractor_revision": self.extractor_revision,
            "page_text_sha256": self.page_text_sha256,
            "cells": [cell.as_dict() for cell in self.cells],
        }


@dataclass(frozen=True, slots=True)
class PromptInjectionNeutrality:
    content_treatment: str
    instruction_authority: str
    validator_enforcement: str

    def as_dict(self) -> dict[str, str]:
        return {
            "content_treatment": self.content_treatment,
            "instruction_authority": self.instruction_authority,
            "validator_enforcement": self.validator_enforcement,
        }


@dataclass(frozen=True, slots=True)
class TableGrid:
    table_id: str
    rows: tuple[TableRow, ...]
    columns: tuple[TableColumn, ...]
    cells: tuple[TableCell, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "table_id": self.table_id,
            "rows": [row.as_dict() for row in self.rows],
            "columns": [column.as_dict() for column in self.columns],
            "cells": [cell.as_dict() for cell in self.cells],
        }


@dataclass(frozen=True, slots=True)
class PaperTableEvidence:
    schema: str
    source: SourceBinding
    inference: QwenInferenceBinding
    extraction_spec: ExtractionSpecBinding | None
    table: TableGrid
    prompt_injection_neutrality: PromptInjectionNeutrality
    born_digital_cross_check: BornDigitalCrossCheck | None
    evidence_sha256: str

    def unsigned_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "source": self.source.as_dict(),
            "inference": self.inference.as_dict(),
            "table": self.table.as_dict(),
            "prompt_injection_neutrality": self.prompt_injection_neutrality.as_dict(),
        }
        if self.extraction_spec is not None:
            payload["extraction_spec"] = self.extraction_spec.as_dict()
        if self.born_digital_cross_check is not None:
            payload["born_digital_cross_check"] = self.born_digital_cross_check.as_dict()
        return payload

    def as_dict(self) -> dict[str, Any]:
        payload = self.unsigned_dict()
        payload["evidence_sha256"] = self.evidence_sha256
        return payload


def _fail(code: str, path: str, message: str) -> NoReturn:
    raise PaperTableEvidenceValidationError(code, path, message)


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail("invalid_object", path, "expected an object")
    return cast(Mapping[str, Any], value)


def _sequence(value: Any, path: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        _fail("invalid_array", path, "expected an array")
    return cast(Sequence[Any], value)


def _require_exact_keys(
    value: Mapping[str, Any],
    path: str,
    *,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    keys = set(value)
    if any(not isinstance(key, str) for key in value):
        _fail("invalid_key", path, "object keys must be strings")
    missing = sorted(required - keys)
    extra = sorted(keys - required - optional)
    if "uncertainty" in extra:
        _fail(
            "unsupported_uncertainty",
            f"{path}.uncertainty",
            "v1 cannot represent a structured uncertainty without observed provenance",
        )
    if missing:
        _fail("missing_keys", path, f"missing required keys: {', '.join(missing)}")
    if extra:
        _fail("unexpected_keys", path, f"unexpected keys: {', '.join(extra)}")


def _sha256(value: Any, path: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        _fail("invalid_sha256", path, "expected 64 lowercase hexadecimal characters")
    return cast(str, value)


def _runtime_identity(value: Any, path: str) -> str:
    if not isinstance(value, str) or not _RUNTIME_IDENTITY_RE.fullmatch(value):
        _fail("invalid_runtime_identity", path, "expected immutable sha256:<64 lowercase hex>")
    return cast(str, value)


def _positive_int(value: Any, path: str, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1 or value > maximum:
        _fail("invalid_integer", path, f"expected an integer in [1, {maximum}]")
    return cast(int, value)


def _nonnegative_int(value: Any, path: str, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > maximum:
        _fail("invalid_integer", path, f"expected an integer in [0, {maximum}]")
    return cast(int, value)


def _finite_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail("invalid_number", path, "expected a JSON number")
    number = float(value)
    if not math.isfinite(number):
        _fail("nonfinite_number", path, "NaN and infinity are forbidden")
    return number


def _identifier(value: Any, path: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        _fail("invalid_identifier", path, "expected a stable ASCII identifier")
    return cast(str, value)


def _text(value: Any, path: str, *, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail("invalid_text", path, "expected non-empty text")
    if len(value) > maximum:
        _fail("invalid_text", path, f"text exceeds {maximum} characters")
    for character in value:
        if ord(character) < 32 and character not in {"\t", "\n", "\r"}:
            _fail("invalid_text", path, "text contains a forbidden control character")
    return value


def _optional_text(value: Any, path: str, *, maximum: int) -> str | None:
    if value is None:
        return None
    return _text(value, path, maximum=maximum)


def _unit(value: Any, path: str) -> str | None:
    if value is None:
        return None
    unit = _text(value, path, maximum=64)
    if unit != unit.strip():
        _fail("invalid_unit", path, "unit must not contain leading or trailing whitespace")
    if unit.casefold() in _UNBOUND_UNITS:
        _fail("invalid_unit", path, "unknown units must be null, not a placeholder string")
    return unit


def _immutable_revision(value: Any, path: str) -> str:
    revision = _text(value, path, maximum=256)
    if revision.casefold() in _MUTABLE_REVISIONS:
        _fail("mutable_revision", path, "revision must be immutable, not an alias")
    return revision


def _bbox(value: Any, path: str, *, width_px: int, height_px: int) -> PixelBBox | None:
    if value is None:
        return None
    raw = _sequence(value, path)
    if len(raw) != 4:
        _fail("invalid_bbox", path, "expected [x0, y0, x1, y1]")
    x0, y0, x1, y1 = (
        _finite_number(raw[0], f"{path}[0]"),
        _finite_number(raw[1], f"{path}[1]"),
        _finite_number(raw[2], f"{path}[2]"),
        _finite_number(raw[3], f"{path}[3]"),
    )
    if x0 < 0 or y0 < 0 or x0 >= x1 or y0 >= y1 or x1 > width_px or y1 > height_px:
        _fail(
            "bbox_out_of_bounds",
            path,
            f"bbox must have positive area inside the {width_px}x{height_px} render",
        )
    return PixelBBox(x0=x0, y0=y0, x1=x1, y1=y1)


def _normalized_comparison_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def parse_exact_numeric_literal(value: str, path: str = "$.text") -> Decimal:
    """Return one exact numeric literal or fail on ambiguity/nonexact qualifiers."""

    normalized = _normalized_comparison_text(value).replace("−", "-")
    if any(marker in normalized for marker in _NONEXACT_NUMERIC_MARKERS) or (
        _NONEXACT_NUMERIC_QUALIFIER_RE.search(normalized) is not None
    ):
        _fail(
            "nonliteral_numeric_text",
            path,
            "inequality, approximation, and plus/minus text cannot expose an exact numeric value",
        )
    matches = list(_NUMERIC_LITERAL_RE.finditer(normalized))
    if not matches:
        _fail(
            "numeric_text_unparseable",
            path,
            "numeric_value requires one decimal or scientific-notation literal in text",
        )
    if len(matches) != 1:
        _fail(
            "ambiguous_numeric_text",
            path,
            "numeric_value requires exactly one numeric literal in text",
        )
    literal = matches[0].group("number").replace(",", "")
    try:
        parsed = Decimal(literal)
    except InvalidOperation:
        _fail("numeric_text_unparseable", path, "numeric text literal is invalid")
    if not parsed.is_finite():
        _fail("numeric_text_unparseable", path, "numeric text literal must be finite")
    return parsed


def _numeric_literal_from_observed_text(value: str, path: str) -> Decimal:
    return parse_exact_numeric_literal(value, path)


def _parse_source(value: Any, *, schema: str) -> SourceBinding:
    source = _mapping(value, "$.source")
    source_keys = {"pdf_sha256", "page", "render"}
    if schema == PAPER_TABLE_EVIDENCE_SCHEMA:
        source_keys.add("region")
    _require_exact_keys(source, "$.source", required=source_keys)
    render_raw = _mapping(source["render"], "$.source.render")
    _require_exact_keys(
        render_raw,
        "$.source.render",
        required={"png_sha256", "width_px", "height_px", "zoom"},
    )
    width = _positive_int(
        render_raw["width_px"],
        "$.source.render.width_px",
        maximum=MAX_RENDER_DIMENSION_PX,
    )
    height = _positive_int(
        render_raw["height_px"],
        "$.source.render.height_px",
        maximum=MAX_RENDER_DIMENSION_PX,
    )
    zoom = _finite_number(render_raw["zoom"], "$.source.render.zoom")
    if zoom <= 0 or zoom > MAX_RENDER_ZOOM:
        _fail(
            "invalid_render_zoom",
            "$.source.render.zoom",
            f"zoom must be within (0, {MAX_RENDER_ZOOM}]",
        )
    render = RenderBinding(
        png_sha256=_sha256(render_raw["png_sha256"], "$.source.render.png_sha256"),
        width_px=width,
        height_px=height,
        zoom=zoom,
    )
    region: RegionBinding | None = None
    if schema == PAPER_TABLE_EVIDENCE_SCHEMA:
        region_raw = _mapping(source["region"], "$.source.region")
        _require_exact_keys(
            region_raw,
            "$.source.region",
            required={"bbox_px", "png_sha256", "width_px", "height_px"},
        )
        region_bbox = _bbox(
            region_raw["bbox_px"],
            "$.source.region.bbox_px",
            width_px=width,
            height_px=height,
        )
        assert region_bbox is not None
        region_width = _positive_int(
            region_raw["width_px"],
            "$.source.region.width_px",
            maximum=MAX_RENDER_DIMENSION_PX,
        )
        region_height = _positive_int(
            region_raw["height_px"],
            "$.source.region.height_px",
            maximum=MAX_RENDER_DIMENSION_PX,
        )
        if region_width != region_bbox.x1 - region_bbox.x0 or (
            region_height != region_bbox.y1 - region_bbox.y0
        ):
            _fail(
                "region_dimension_mismatch",
                "$.source.region",
                "declared region dimensions must exactly equal its full-render bbox",
            )
        region = RegionBinding(
            bbox_px=region_bbox,
            png_sha256=_sha256(region_raw["png_sha256"], "$.source.region.png_sha256"),
            width_px=region_width,
            height_px=region_height,
        )
    return SourceBinding(
        pdf_sha256=_sha256(source["pdf_sha256"], "$.source.pdf_sha256"),
        page=_positive_int(source["page"], "$.source.page", maximum=10_000_000),
        render=render,
        region=region,
    )


def _parse_inference(value: Any, *, schema: str) -> QwenInferenceBinding:
    inference = _mapping(value, "$.inference")
    required = {
        "model_id",
        "model_revision",
        "runtime_identity",
        "prompt_sha256",
        "config_sha256",
        "raw_response_sha256",
    }
    if schema == PAPER_TABLE_EVIDENCE_SCHEMA:
        required.update(
            {
                "deployment_attestation_sha256",
                "attestation_authority",
                "response_model_id",
                "response_system_fingerprint",
                "model_input_sha256",
                "model_input_width_px",
                "model_input_height_px",
            }
        )
    _require_exact_keys(
        inference,
        "$.inference",
        required=required,
    )
    model_id = _text(inference["model_id"], "$.inference.model_id", maximum=256)
    if _QWEN_MODEL_RE.search(model_id) is None:
        _fail("non_qwen_model", "$.inference.model_id", "model ID must identify a Qwen model")
    result = QwenInferenceBinding(
        model_id=model_id,
        model_revision=_immutable_revision(
            inference["model_revision"], "$.inference.model_revision"
        ),
        runtime_identity=_runtime_identity(
            inference["runtime_identity"], "$.inference.runtime_identity"
        ),
        prompt_sha256=_sha256(inference["prompt_sha256"], "$.inference.prompt_sha256"),
        config_sha256=_sha256(inference["config_sha256"], "$.inference.config_sha256"),
        raw_response_sha256=_sha256(
            inference["raw_response_sha256"], "$.inference.raw_response_sha256"
        ),
    )
    if schema == PAPER_TABLE_EVIDENCE_SCHEMA:
        response_model_id = _text(
            inference["response_model_id"], "$.inference.response_model_id", maximum=256
        )
        if response_model_id != model_id:
            _fail(
                "response_model_identity_mismatch",
                "$.inference.response_model_id",
                "endpoint-reported model identity must equal the attested model ID",
            )
        fingerprint = _optional_text(
            inference["response_system_fingerprint"],
            "$.inference.response_system_fingerprint",
            maximum=256,
        )
        result = replace(
            result,
            deployment_attestation_sha256=_sha256(
                inference["deployment_attestation_sha256"],
                "$.inference.deployment_attestation_sha256",
            ),
            attestation_authority=_identifier(
                inference["attestation_authority"], "$.inference.attestation_authority"
            ),
            response_model_id=response_model_id,
            response_system_fingerprint=fingerprint,
            model_input_sha256=_sha256(
                inference["model_input_sha256"], "$.inference.model_input_sha256"
            ),
            model_input_width_px=_positive_int(
                inference["model_input_width_px"],
                "$.inference.model_input_width_px",
                maximum=MAX_RENDER_DIMENSION_PX,
            ),
            model_input_height_px=_positive_int(
                inference["model_input_height_px"],
                "$.inference.model_input_height_px",
                maximum=MAX_RENDER_DIMENSION_PX,
            ),
        )
    return result


def _parse_axis_bounds(value: Any, path: str, *, maximum: int) -> AxisBounds:
    bounds = _mapping(value, path)
    _require_exact_keys(bounds, path, required={"minimum", "maximum"})
    minimum = _positive_int(bounds["minimum"], f"{path}.minimum", maximum=maximum)
    upper = _positive_int(bounds["maximum"], f"{path}.maximum", maximum=maximum)
    if upper < minimum:
        _fail("invalid_axis_bounds", path, "maximum must be greater than or equal to minimum")
    return AxisBounds(minimum=minimum, maximum=upper)


def _parse_expected_rows(value: Any) -> tuple[ExpectedRowIdentity, ...]:
    rows = _sequence(value, "$.extraction_spec.expected_rows")
    if len(rows) > MAX_ROWS:
        _fail("invalid_rows", "$.extraction_spec.expected_rows", "too many expected rows")
    parsed: list[ExpectedRowIdentity] = []
    seen: set[str] = set()
    for index, raw in enumerate(rows):
        path = f"$.extraction_spec.expected_rows[{index}]"
        item = _mapping(raw, path)
        _require_exact_keys(item, path, required={"row_id", "label"})
        row_id = _identifier(item["row_id"], f"{path}.row_id")
        if row_id in seen:
            _fail("duplicate_expected_row_id", f"{path}.row_id", f"duplicate row ID {row_id!r}")
        seen.add(row_id)
        parsed.append(
            ExpectedRowIdentity(
                row_id=row_id,
                label=_optional_text(item["label"], f"{path}.label", maximum=512),
            )
        )
    return tuple(parsed)


def _parse_expected_columns(value: Any) -> tuple[ExpectedColumnIdentity, ...]:
    columns = _sequence(value, "$.extraction_spec.expected_columns")
    if len(columns) > MAX_COLUMNS:
        _fail("invalid_columns", "$.extraction_spec.expected_columns", "too many expected columns")
    parsed: list[ExpectedColumnIdentity] = []
    seen: set[str] = set()
    for index, raw in enumerate(columns):
        path = f"$.extraction_spec.expected_columns[{index}]"
        item = _mapping(raw, path)
        _require_exact_keys(item, path, required={"column_id", "label", "unit"})
        column_id = _identifier(item["column_id"], f"{path}.column_id")
        if column_id in seen:
            _fail(
                "duplicate_expected_column_id",
                f"{path}.column_id",
                f"duplicate column ID {column_id!r}",
            )
        seen.add(column_id)
        parsed.append(
            ExpectedColumnIdentity(
                column_id=column_id,
                label=_optional_text(item["label"], f"{path}.label", maximum=512),
                unit=_unit(item["unit"], f"{path}.unit"),
            )
        )
    return tuple(parsed)


def _parse_extraction_spec(value: Any, *, render: RenderBinding) -> ExtractionSpecBinding:
    spec = _mapping(value, "$.extraction_spec")
    _require_exact_keys(
        spec,
        "$.extraction_spec",
        required={
            "identity_mode",
            "scientific_identity_status",
            "table_id",
            "table_label",
            "page",
            "row_bounds",
            "column_bounds",
            "expected_rows",
            "expected_columns",
            "source_region_px",
        },
    )
    mode = spec["identity_mode"]
    if mode not in {"specified", "generic_unverified"}:
        _fail(
            "invalid_identity_mode",
            "$.extraction_spec.identity_mode",
            "expected 'specified' or explicit 'generic_unverified'",
        )
    expected_status = "specified" if mode == "specified" else "unverified"
    if spec["scientific_identity_status"] != expected_status:
        _fail(
            "invalid_scientific_identity_status",
            "$.extraction_spec.scientific_identity_status",
            f"identity_mode {mode!r} requires status {expected_status!r}",
        )
    table_label = _optional_text(spec["table_label"], "$.extraction_spec.table_label", maximum=512)
    row_bounds = _parse_axis_bounds(
        spec["row_bounds"], "$.extraction_spec.row_bounds", maximum=MAX_ROWS
    )
    column_bounds = _parse_axis_bounds(
        spec["column_bounds"], "$.extraction_spec.column_bounds", maximum=MAX_COLUMNS
    )
    expected_rows = _parse_expected_rows(spec["expected_rows"])
    expected_columns = _parse_expected_columns(spec["expected_columns"])
    if expected_rows and not (row_bounds.minimum <= len(expected_rows) <= row_bounds.maximum):
        _fail(
            "expected_rows_outside_bounds",
            "$.extraction_spec.expected_rows",
            "expected row identities must fit inside the declared row bounds",
        )
    if expected_columns and not (
        column_bounds.minimum <= len(expected_columns) <= column_bounds.maximum
    ):
        _fail(
            "expected_columns_outside_bounds",
            "$.extraction_spec.expected_columns",
            "expected column identities must fit inside the declared column bounds",
        )
    if mode == "specified":
        if table_label is None:
            _fail(
                "specified_table_label_required",
                "$.extraction_spec.table_label",
                "specified mode requires the visible table label/selector",
            )
        if not expected_columns:
            _fail(
                "specified_headers_required",
                "$.extraction_spec.expected_columns",
                "specified mode requires expected column header IDs",
            )
    elif expected_rows or expected_columns:
        _fail(
            "generic_identity_must_be_unclaimed",
            "$.extraction_spec",
            "generic_unverified mode cannot declare expected scientific row/column identities",
        )
    source_region = _bbox(
        spec["source_region_px"],
        "$.extraction_spec.source_region_px",
        width_px=render.width_px,
        height_px=render.height_px,
    )
    return ExtractionSpecBinding(
        identity_mode=cast(str, mode),
        scientific_identity_status=expected_status,
        table_id=_identifier(spec["table_id"], "$.extraction_spec.table_id"),
        table_label=table_label,
        page=_positive_int(spec["page"], "$.extraction_spec.page", maximum=10_000_000),
        row_bounds=row_bounds,
        column_bounds=column_bounds,
        expected_rows=expected_rows,
        expected_columns=expected_columns,
        source_region_px=source_region,
    )


def _parse_prompt_injection_neutrality(value: Any) -> PromptInjectionNeutrality:
    metadata = _mapping(value, "$.prompt_injection_neutrality")
    required = set(PROMPT_INJECTION_NEUTRALITY)
    _require_exact_keys(metadata, "$.prompt_injection_neutrality", required=required)
    for key, expected in PROMPT_INJECTION_NEUTRALITY.items():
        if metadata[key] != expected:
            _fail(
                "invalid_prompt_injection_metadata",
                f"$.prompt_injection_neutrality.{key}",
                f"expected {expected!r}",
            )
    return PromptInjectionNeutrality(**PROMPT_INJECTION_NEUTRALITY)


def _parse_rows(value: Any) -> tuple[TableRow, ...]:
    raw_rows = _sequence(value, "$.table.rows")
    if not raw_rows or len(raw_rows) > MAX_ROWS:
        _fail("invalid_rows", "$.table.rows", f"expected 1..{MAX_ROWS} rows")
    rows: list[TableRow] = []
    seen: set[str] = set()
    for index, raw_row in enumerate(raw_rows):
        path = f"$.table.rows[{index}]"
        row = _mapping(raw_row, path)
        _require_exact_keys(row, path, required={"row_id", "label"})
        row_id = _identifier(row["row_id"], f"{path}.row_id")
        if row_id in seen:
            _fail("duplicate_row_id", f"{path}.row_id", f"duplicate row ID {row_id!r}")
        seen.add(row_id)
        rows.append(
            TableRow(
                row_id=row_id,
                label=_optional_text(row["label"], f"{path}.label", maximum=512),
            )
        )
    return tuple(rows)


def _parse_columns(value: Any) -> tuple[TableColumn, ...]:
    raw_columns = _sequence(value, "$.table.columns")
    if not raw_columns or len(raw_columns) > MAX_COLUMNS:
        _fail("invalid_columns", "$.table.columns", f"expected 1..{MAX_COLUMNS} columns")
    columns: list[TableColumn] = []
    seen: set[str] = set()
    for index, raw_column in enumerate(raw_columns):
        path = f"$.table.columns[{index}]"
        column = _mapping(raw_column, path)
        _require_exact_keys(column, path, required={"column_id", "label", "unit"})
        column_id = _identifier(column["column_id"], f"{path}.column_id")
        if column_id in seen:
            _fail("duplicate_column_id", f"{path}.column_id", f"duplicate column ID {column_id!r}")
        seen.add(column_id)
        columns.append(
            TableColumn(
                column_id=column_id,
                label=_optional_text(column["label"], f"{path}.label", maximum=512),
                unit=_unit(column["unit"], f"{path}.unit"),
            )
        )
    return tuple(columns)


def _parse_cell(
    value: Any,
    path: str,
    *,
    rows: Mapping[str, int],
    columns: Mapping[str, tuple[int, TableColumn]],
    render: RenderBinding,
    region: RegionBinding | None,
) -> TableCell:
    cell = _mapping(value, path)
    _require_exact_keys(
        cell,
        path,
        required={
            "row_id",
            "column_id",
            "text",
            "numeric_value",
            "unit",
            "bbox_px",
            "observation_status",
        },
    )
    row_id = _identifier(cell["row_id"], f"{path}.row_id")
    column_id = _identifier(cell["column_id"], f"{path}.column_id")
    if row_id not in rows:
        _fail("unknown_row_id", f"{path}.row_id", f"unknown row ID {row_id!r}")
    if column_id not in columns:
        _fail("unknown_column_id", f"{path}.column_id", f"unknown column ID {column_id!r}")
    column = columns[column_id][1]
    text_value = _optional_text(cell["text"], f"{path}.text", maximum=4096)
    numeric_raw = cell["numeric_value"]
    numeric_value = (
        None if numeric_raw is None else _finite_number(numeric_raw, f"{path}.numeric_value")
    )
    unit_value = _unit(cell["unit"], f"{path}.unit")
    bbox = _bbox(
        cell["bbox_px"],
        f"{path}.bbox_px",
        width_px=render.width_px,
        height_px=render.height_px,
    )
    if bbox is not None and region is not None:
        rb = region.bbox_px
        if bbox.x0 < rb.x0 or bbox.y0 < rb.y0 or bbox.x1 > rb.x1 or bbox.y1 > rb.y1:
            _fail(
                "bbox_outside_observation_region",
                f"{path}.bbox_px",
                "cell bbox must lie inside the exact model observation region",
            )
    try:
        status = ObservationStatus(cell["observation_status"])
    except (TypeError, ValueError):
        allowed = ", ".join(status.value for status in ObservationStatus)
        _fail("invalid_observation_status", f"{path}.observation_status", f"expected {allowed}")

    if unit_value is not None and unit_value != column.unit:
        _fail(
            "unit_mismatch",
            f"{path}.unit",
            f"cell unit must exactly match column unit {column.unit!r}",
        )
    if numeric_value is not None:
        if text_value is None:
            _fail("numeric_without_text", f"{path}.numeric_value", "numeric value requires text")
        observed_numeric = _numeric_literal_from_observed_text(text_value, f"{path}.text")
        if observed_numeric != Decimal(str(numeric_value)):
            _fail(
                "numeric_text_mismatch",
                f"{path}.numeric_value",
                "numeric_value must equal the unambiguous numeric literal in observed text",
            )
        if bbox is None:
            _fail("numeric_without_bbox", f"{path}.numeric_value", "numeric value requires a bbox")
        if column.unit is None or unit_value is None:
            _fail(
                "numeric_unit_unbound",
                f"{path}.unit",
                "numeric values require an explicit column and cell unit; use '1' if dimensionless",
            )
    if unit_value is not None and text_value is None:
        _fail("unit_without_text", f"{path}.unit", "a cell unit requires observed text")

    if status in {
        ObservationStatus.MODEL_OBSERVED,
        ObservationStatus.CROSS_CHECKED,
        ObservationStatus.CONFLICT,
    }:
        if text_value is None or bbox is None:
            _fail(
                "observed_cell_unlocated",
                path,
                f"{status.value} requires non-null text and bbox_px",
            )
    if status is ObservationStatus.CONFLICT and (
        numeric_value is not None or unit_value is not None
    ):
        _fail(
            "unresolved_conflict_value",
            path,
            "a conflict cannot expose a numeric value or resolved unit",
        )
    if status is ObservationStatus.UNREADABLE and (
        text_value is not None or numeric_value is not None or unit_value is not None
    ):
        _fail(
            "unreadable_cell_has_value",
            path,
            "unreadable requires null text, numeric_value, and unit; bbox may locate the cell",
        )
    return TableCell(
        row_id=row_id,
        column_id=column_id,
        text=text_value,
        numeric_value=numeric_value,
        unit=unit_value,
        bbox_px=bbox,
        observation_status=status,
    )


def _parse_table(
    value: Any,
    *,
    render: RenderBinding,
    region: RegionBinding | None = None,
) -> TableGrid:
    table = _mapping(value, "$.table")
    _require_exact_keys(table, "$.table", required={"table_id", "rows", "columns", "cells"})
    table_id = _identifier(table["table_id"], "$.table.table_id")
    rows = _parse_rows(table["rows"])
    columns = _parse_columns(table["columns"])
    cell_count = len(rows) * len(columns)
    if cell_count > MAX_CELLS:
        _fail(
            "table_too_large",
            "$.table",
            f"rectangular grid contains {cell_count} cells; maximum is {MAX_CELLS}",
        )
    raw_cells = _sequence(table["cells"], "$.table.cells")
    if len(raw_cells) != cell_count:
        _fail(
            "incomplete_rectangular_grid",
            "$.table.cells",
            f"expected exactly {cell_count} cells, found {len(raw_cells)}",
        )
    row_lookup = {row.row_id: index for index, row in enumerate(rows)}
    column_lookup = {column.column_id: (index, column) for index, column in enumerate(columns)}
    cells: dict[tuple[str, str], TableCell] = {}
    for index, raw_cell in enumerate(raw_cells):
        path = f"$.table.cells[{index}]"
        cell = _parse_cell(
            raw_cell,
            path,
            rows=row_lookup,
            columns=column_lookup,
            render=render,
            region=region,
        )
        if cell.coordinate in cells:
            _fail(
                "duplicate_cell",
                path,
                f"duplicate cell coordinate {cell.coordinate!r}",
            )
        cells[cell.coordinate] = cell
    expected = {(row.row_id, column.column_id) for row in rows for column in columns}
    missing = expected - set(cells)
    if missing:
        coordinate = min(missing, key=lambda item: (row_lookup[item[0]], column_lookup[item[1]][0]))
        _fail("incomplete_rectangular_grid", "$.table.cells", f"missing cell {coordinate!r}")
    ordered = tuple(cells[(row.row_id, column.column_id)] for row in rows for column in columns)
    return TableGrid(table_id=table_id, rows=rows, columns=columns, cells=ordered)


def _bbox_intersection_over_min_area(left: PixelBBox, right: PixelBBox) -> float:
    width = max(0.0, min(left.x1, right.x1) - max(left.x0, right.x0))
    height = max(0.0, min(left.y1, right.y1) - max(left.y0, right.y0))
    if width == 0.0 or height == 0.0:
        return 0.0
    left_area = (left.x1 - left.x0) * (left.y1 - left.y0)
    right_area = (right.x1 - right.x0) * (right.y1 - right.y0)
    return width * height / min(left_area, right_area)


@dataclass(slots=True)
class _YIntervalNode:
    """One deterministic treap node for active y-interval intersection queries."""

    key: tuple[float, int]
    y1: float
    cell: TableCell
    priority: int
    max_y1: float
    left: _YIntervalNode | None = None
    right: _YIntervalNode | None = None


def _interval_priority(index: int) -> int:
    """Return a deterministic splitmix64 priority with no adversarial sorted-key shape."""

    value = (index + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return value ^ (value >> 31)


def _refresh_interval_node(node: _YIntervalNode) -> _YIntervalNode:
    node.max_y1 = max(
        node.y1,
        node.left.max_y1 if node.left is not None else -math.inf,
        node.right.max_y1 if node.right is not None else -math.inf,
    )
    return node


def _rotate_interval_right(root: _YIntervalNode) -> _YIntervalNode:
    child = root.left
    assert child is not None
    root.left = child.right
    child.right = _refresh_interval_node(root)
    return _refresh_interval_node(child)


def _rotate_interval_left(root: _YIntervalNode) -> _YIntervalNode:
    child = root.right
    assert child is not None
    root.right = child.left
    child.left = _refresh_interval_node(root)
    return _refresh_interval_node(child)


def _insert_interval(
    root: _YIntervalNode | None,
    node: _YIntervalNode,
) -> _YIntervalNode:
    if root is None:
        return node
    if node.key < root.key:
        root.left = _insert_interval(root.left, node)
        if root.left.priority < root.priority:
            return _rotate_interval_right(root)
    else:
        root.right = _insert_interval(root.right, node)
        if root.right.priority < root.priority:
            return _rotate_interval_left(root)
    return _refresh_interval_node(root)


def _merge_intervals(
    left: _YIntervalNode | None,
    right: _YIntervalNode | None,
) -> _YIntervalNode | None:
    if left is None:
        return right
    if right is None:
        return left
    if left.priority < right.priority:
        left.right = _merge_intervals(left.right, right)
        return _refresh_interval_node(left)
    right.left = _merge_intervals(left, right.left)
    return _refresh_interval_node(right)


def _delete_interval(
    root: _YIntervalNode | None,
    key: tuple[float, int],
) -> _YIntervalNode | None:
    if root is None:
        return None
    if key < root.key:
        root.left = _delete_interval(root.left, key)
        return _refresh_interval_node(root)
    if key > root.key:
        root.right = _delete_interval(root.right, key)
        return _refresh_interval_node(root)
    return _merge_intervals(root.left, root.right)


def _iter_y_intersections(
    root: _YIntervalNode | None,
    *,
    y0: float,
    y1: float,
) -> Sequence[TableCell]:
    """Return active cells whose open y-interval intersects ``(y0, y1)``."""

    if root is None:
        return ()
    matches: list[TableCell] = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node.left is not None and node.left.max_y1 > y0:
            stack.append(node.left)
        if node.key[0] < y1 and node.y1 > y0:
            matches.append(node.cell)
        if node.right is not None and node.key[0] < y1:
            stack.append(node.right)
    return matches


def _reject_implausible_cell_overlaps(located: Sequence[TableCell]) -> None:
    """Reject the same >80% overlaps using a bounded plane sweep."""

    indexed = sorted(
        enumerate(located),
        key=lambda item: (
            cast(PixelBBox, item[1].bbox_px).x0,
            cast(PixelBBox, item[1].bbox_px).x1,
            cast(PixelBBox, item[1].bbox_px).y0,
            cast(PixelBBox, item[1].bbox_px).y1,
            item[0],
        ),
    )
    active_by_x1: list[tuple[float, int, tuple[float, int]]] = []
    active_y: _YIntervalNode | None = None
    candidate_count = 0
    for index, cell in indexed:
        bbox = cast(PixelBBox, cell.bbox_px)
        while active_by_x1 and active_by_x1[0][0] <= bbox.x0:
            _x1, _expired_index, expired_key = heapq.heappop(active_by_x1)
            active_y = _delete_interval(active_y, expired_key)

        for other in _iter_y_intersections(active_y, y0=bbox.y0, y1=bbox.y1):
            other_bbox = cast(PixelBBox, other.bbox_px)
            if _bbox_intersection_over_min_area(other_bbox, bbox) > MAX_IMPLAUSIBLE_CELL_OVERLAP:
                _fail(
                    "implausible_cell_overlap",
                    "$.table.cells",
                    f"cells {other.coordinate!r} and {cell.coordinate!r} substantially overlap",
                )
            candidate_count += 1
            if candidate_count > MAX_CELL_GEOMETRY_CANDIDATES:
                _fail(
                    "cell_geometry_complexity_exceeded",
                    "$.table.cells",
                    "cell-location intersection candidates exceed the bounded geometry budget",
                )

        key = (bbox.y0, index)
        active_y = _insert_interval(
            active_y,
            _YIntervalNode(
                key=key,
                y1=bbox.y1,
                cell=cell,
                priority=_interval_priority(index),
                max_y1=bbox.y1,
            ),
        )
        heapq.heappush(active_by_x1, (bbox.x1, index, key))


def _validate_cell_geometry(
    table: TableGrid,
    *,
    source: SourceBinding,
    inference: QwenInferenceBinding,
) -> None:
    located = [cell for cell in table.cells if cell.bbox_px is not None]
    bbox_owner: dict[tuple[float, float, float, float], tuple[str, str]] = {}
    for cell in located:
        assert cell.bbox_px is not None
        key = tuple(cell.bbox_px.as_list())
        if key in bbox_owner:
            _fail(
                "duplicate_cell_bbox",
                "$.table.cells",
                f"cells {bbox_owner[key]!r} and {cell.coordinate!r} have identical locations",
            )
        bbox_owner[key] = cell.coordinate
    _reject_implausible_cell_overlaps(located)

    by_coordinate = {cell.coordinate: cell for cell in table.cells}
    for row in table.rows:
        previous_center: float | None = None
        for column in table.columns:
            bbox = by_coordinate[(row.row_id, column.column_id)].bbox_px
            if bbox is None:
                continue
            center = (bbox.x0 + bbox.x1) / 2.0
            if previous_center is not None and center <= previous_center:
                _fail(
                    "cell_column_order_mismatch",
                    "$.table.cells",
                    f"cell x-order contradicts declared column order in row {row.row_id!r}",
                )
            previous_center = center
    for column in table.columns:
        previous_center = None
        for row in table.rows:
            bbox = by_coordinate[(row.row_id, column.column_id)].bbox_px
            if bbox is None:
                continue
            center = (bbox.y0 + bbox.y1) / 2.0
            if previous_center is not None and center <= previous_center:
                _fail(
                    "cell_row_order_mismatch",
                    "$.table.cells",
                    f"cell y-order contradicts declared row order in column {column.column_id!r}",
                )
            previous_center = center

    if source.region is None or inference.model_input_width_px is None:
        return
    scale_x = inference.model_input_width_px / source.region.width_px
    assert inference.model_input_height_px is not None
    scale_y = inference.model_input_height_px / source.region.height_px
    for cell in located:
        assert cell.bbox_px is not None
        effective_width = (cell.bbox_px.x1 - cell.bbox_px.x0) * scale_x
        effective_height = (cell.bbox_px.y1 - cell.bbox_px.y0) * scale_y
        if (
            effective_width < MIN_EFFECTIVE_CELL_WIDTH_PX
            or effective_height < MIN_EFFECTIVE_CELL_HEIGHT_PX
        ):
            _fail(
                "insufficient_effective_cell_resolution",
                "$.table.cells",
                f"cell {cell.coordinate!r} is only {effective_width:.1f}x"
                f"{effective_height:.1f} effective model-input pixels",
            )


def _validate_extraction_semantics(
    spec: ExtractionSpecBinding,
    *,
    source: SourceBinding,
    inference: QwenInferenceBinding,
    table: TableGrid,
) -> None:
    if spec.page != source.page:
        _fail(
            "extraction_page_mismatch",
            "$.extraction_spec.page",
            "requested page must equal the provenance-bound rendered page",
        )
    if spec.table_id != table.table_id:
        _fail(
            "extraction_table_id_mismatch",
            "$.table.table_id",
            "model output table ID must equal the requested stable table ID",
        )
    if not (spec.row_bounds.minimum <= len(table.rows) <= spec.row_bounds.maximum):
        _fail("row_count_out_of_bounds", "$.table.rows", "model row count violates request bounds")
    if not (spec.column_bounds.minimum <= len(table.columns) <= spec.column_bounds.maximum):
        _fail(
            "column_count_out_of_bounds",
            "$.table.columns",
            "model column count violates request bounds",
        )
    if spec.expected_rows:
        observed_rows = tuple((row.row_id, row.label) for row in table.rows)
        expected_rows = tuple((row.row_id, row.label) for row in spec.expected_rows)
        if observed_rows != expected_rows:
            _fail(
                "expected_row_identity_mismatch",
                "$.table.rows",
                "model row IDs/labels do not exactly match the closed extraction request",
            )
    if spec.expected_columns:
        observed_columns = tuple(
            (column.column_id, column.label, column.unit) for column in table.columns
        )
        expected_columns = tuple(
            (column.column_id, column.label, column.unit) for column in spec.expected_columns
        )
        if observed_columns != expected_columns:
            _fail(
                "expected_column_identity_mismatch",
                "$.table.columns",
                "model header IDs/labels/units do not exactly match the closed extraction request",
            )
    assert source.region is not None
    requested_region = spec.source_region_px or PixelBBox(
        x0=0.0,
        y0=0.0,
        x1=float(source.render.width_px),
        y1=float(source.render.height_px),
    )
    if source.region.bbox_px != requested_region:
        _fail(
            "observation_region_mismatch",
            "$.source.region.bbox_px",
            "observation region must exactly equal the requested full-page region or crop",
        )
    assert inference.model_input_width_px is not None
    assert inference.model_input_height_px is not None
    if (
        inference.model_input_width_px > source.region.width_px
        or inference.model_input_height_px > source.region.height_px
    ):
        _fail(
            "model_input_upsampled",
            "$.inference",
            "durable table evidence forbids unqualified model-input upsampling",
        )
    source_aspect = source.region.width_px / source.region.height_px
    model_aspect = inference.model_input_width_px / inference.model_input_height_px
    if not math.isclose(source_aspect, model_aspect, rel_tol=0.01, abs_tol=0.01):
        _fail(
            "model_input_aspect_mismatch",
            "$.inference",
            "model input must preserve the exact observation region aspect ratio",
        )


def _parse_born_digital_cross_check(
    value: Any,
    *,
    table: TableGrid,
) -> BornDigitalCrossCheck:
    cross_check = _mapping(value, "$.born_digital_cross_check")
    _require_exact_keys(
        cross_check,
        "$.born_digital_cross_check",
        required={"extractor_id", "extractor_revision", "page_text_sha256", "cells"},
    )
    coordinates = {cell.coordinate for cell in table.cells}
    row_order = {row.row_id: index for index, row in enumerate(table.rows)}
    column_order = {column.column_id: index for index, column in enumerate(table.columns)}
    raw_cells = _sequence(cross_check["cells"], "$.born_digital_cross_check.cells")
    checks: dict[tuple[str, str], BornDigitalCellCheck] = {}
    for index, raw_check in enumerate(raw_cells):
        path = f"$.born_digital_cross_check.cells[{index}]"
        check = _mapping(raw_check, path)
        _require_exact_keys(
            check,
            path,
            required={"row_id", "column_id", "text", "text_sha256"},
            optional={"start_char", "end_char"},
        )
        row_id = _identifier(check["row_id"], f"{path}.row_id")
        column_id = _identifier(check["column_id"], f"{path}.column_id")
        coordinate = (row_id, column_id)
        if coordinate not in coordinates:
            _fail("unknown_cross_check_cell", path, f"unknown cell coordinate {coordinate!r}")
        if coordinate in checks:
            _fail("duplicate_cross_check_cell", path, f"duplicate coordinate {coordinate!r}")
        text_value = _text(check["text"], f"{path}.text", maximum=4096)
        text_sha256 = _sha256(check["text_sha256"], f"{path}.text_sha256")
        observed_digest = hashlib.sha256(text_value.encode("utf-8")).hexdigest()
        if not hmac.compare_digest(text_sha256, observed_digest):
            _fail(
                "cross_check_text_sha256_mismatch",
                f"{path}.text_sha256",
                "digest does not match the exact UTF-8 born-digital cell text",
            )
        has_start = "start_char" in check
        has_end = "end_char" in check
        if has_start != has_end:
            _fail(
                "cross_check_span_incomplete",
                path,
                "start_char and end_char must either both be present or both be omitted",
            )
        start_char: int | None = None
        end_char: int | None = None
        if has_start and has_end:
            start_char = _nonnegative_int(
                check["start_char"],
                f"{path}.start_char",
                maximum=100_000_000,
            )
            end_char = _positive_int(
                check["end_char"],
                f"{path}.end_char",
                maximum=100_000_000,
            )
            if end_char <= start_char or end_char - start_char != len(text_value):
                _fail(
                    "cross_check_span_text_mismatch",
                    path,
                    "the exclusive character span length must equal the exact cell text length",
                )
        checks[coordinate] = BornDigitalCellCheck(
            row_id=row_id,
            column_id=column_id,
            text=text_value,
            text_sha256=text_sha256,
            start_char=start_char,
            end_char=end_char,
        )
    ordered = tuple(
        checks[coordinate]
        for coordinate in sorted(
            checks,
            key=lambda item: (row_order[item[0]], column_order[item[1]]),
        )
    )
    return BornDigitalCrossCheck(
        extractor_id=_text(
            cross_check["extractor_id"],
            "$.born_digital_cross_check.extractor_id",
            maximum=256,
        ),
        extractor_revision=_immutable_revision(
            cross_check["extractor_revision"],
            "$.born_digital_cross_check.extractor_revision",
        ),
        page_text_sha256=_sha256(
            cross_check["page_text_sha256"],
            "$.born_digital_cross_check.page_text_sha256",
        ),
        cells=ordered,
    )


def _validate_cross_check_semantics(
    table: TableGrid,
    cross_check: BornDigitalCrossCheck | None,
) -> None:
    checks = {} if cross_check is None else {cell.coordinate: cell for cell in cross_check.cells}
    for cell in table.cells:
        path = f"$.table.cells[{cell.row_id!r},{cell.column_id!r}]"
        check = checks.get(cell.coordinate)
        if cell.observation_status is ObservationStatus.CROSS_CHECKED:
            if check is None:
                _fail("cross_check_missing", path, "cross_checked requires born-digital cell text")
            assert cell.text is not None
            if _normalized_comparison_text(cell.text) != _normalized_comparison_text(check.text):
                _fail(
                    "cross_check_disagrees",
                    path,
                    "cross_checked text must match born-digital text after NFKC/whitespace normalization",
                )
        elif cell.observation_status is ObservationStatus.CONFLICT:
            if check is None:
                _fail("conflict_evidence_missing", path, "conflict requires born-digital cell text")
            assert cell.text is not None
            if _normalized_comparison_text(cell.text) == _normalized_comparison_text(check.text):
                _fail(
                    "false_conflict",
                    path,
                    "conflict texts match after NFKC/whitespace normalization",
                )
        elif check is not None:
            _fail(
                "cross_check_status_mismatch",
                path,
                "born-digital cell evidence requires cross_checked or conflict status",
            )


def canonical_json_bytes(value: Any) -> bytes:
    """Encode the repository's explicit deterministic JSON profile.

    This is sorted-key, compact, UTF-8 JSON with non-finite numbers forbidden;
    it is named ``ultra.canonical-json.v1`` and is not a claim of RFC 8785.
    """

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        _fail(
            "noncanonical_json",
            "$",
            f"value cannot be encoded as {CANONICAL_JSON_PROFILE}: {exc}",
        )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _parse_payload(
    payload: Mapping[str, Any],
    *,
    sealed: bool,
) -> tuple[PaperTableEvidence, str | None]:
    schema = payload.get("schema")
    if schema not in {PAPER_TABLE_EVIDENCE_SCHEMA_V1, PAPER_TABLE_EVIDENCE_SCHEMA}:
        _fail(
            "unsupported_schema",
            "$.schema",
            f"expected {PAPER_TABLE_EVIDENCE_SCHEMA_V1!r} or {PAPER_TABLE_EVIDENCE_SCHEMA!r}",
        )
    required = {
        "schema",
        "source",
        "inference",
        "table",
        "prompt_injection_neutrality",
    }
    if schema == PAPER_TABLE_EVIDENCE_SCHEMA:
        required.add("extraction_spec")
    if sealed:
        required.add("evidence_sha256")
    _require_exact_keys(
        payload,
        "$",
        required=required,
        optional={"born_digital_cross_check"},
    )
    source = _parse_source(payload["source"], schema=cast(str, schema))
    inference = _parse_inference(payload["inference"], schema=cast(str, schema))
    extraction_spec = (
        _parse_extraction_spec(payload["extraction_spec"], render=source.render)
        if schema == PAPER_TABLE_EVIDENCE_SCHEMA
        else None
    )
    table = _parse_table(payload["table"], render=source.render, region=source.region)
    prompt_metadata = _parse_prompt_injection_neutrality(payload["prompt_injection_neutrality"])
    cross_check = (
        _parse_born_digital_cross_check(payload["born_digital_cross_check"], table=table)
        if "born_digital_cross_check" in payload
        else None
    )
    _validate_cross_check_semantics(table, cross_check)
    if extraction_spec is not None:
        _validate_extraction_semantics(
            extraction_spec,
            source=source,
            inference=inference,
            table=table,
        )
        _validate_cell_geometry(table, source=source, inference=inference)
    declared_digest = _sha256(payload["evidence_sha256"], "$.evidence_sha256") if sealed else None
    provisional = PaperTableEvidence(
        schema=cast(str, schema),
        source=source,
        inference=inference,
        extraction_spec=extraction_spec,
        table=table,
        prompt_injection_neutrality=prompt_metadata,
        born_digital_cross_check=cross_check,
        evidence_sha256="",
    )
    computed_digest = _canonical_sha256(provisional.unsigned_dict())
    return replace(provisional, evidence_sha256=computed_digest), declared_digest


def seal_paper_table_evidence(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an unsealed closed-schema payload and append its canonical digest."""

    evidence, _ = _parse_payload(_mapping(payload, "$"), sealed=False)
    return evidence.as_dict()


def validate_paper_table_evidence(payload: Mapping[str, Any]) -> PaperTableEvidence:
    """Validate a sealed v1 payload and return an immutable normalized representation."""

    evidence, declared_digest = _parse_payload(_mapping(payload, "$"), sealed=True)
    assert declared_digest is not None
    if not hmac.compare_digest(declared_digest, evidence.evidence_sha256):
        _fail(
            "evidence_sha256_mismatch",
            "$.evidence_sha256",
            "digest does not match the normalized unsigned evidence envelope",
        )
    return evidence
