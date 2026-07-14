"""Bounded typed tools for degradation and characterization data reduction.

This module exposes only the deterministic kernels already qualified under
``ultra_deepagents.materials``.  Inputs are scientific values and closed
provenance declarations; there is no code, command, filesystem-path, solver,
fitting plugin, or arbitrary-expression surface.  Caller-supplied digests are
validated structurally but are not treated as byte-bound unless an independent
resolver has retrieved and re-hashed their sources.  Successful results are
finite, content-addressed JSON.  Invalid or out-of-domain requests fail without
partial scientific output.

The mechanics tools are deliberately small models: an LEFM applicability
screen, a Paris-regime interpolation fit, a secondary-creep rate evaluation,
two isothermal oxidation mass-gain laws, and a Faraday uniform-corrosion
conversion.  They do not predict component life.  The characterization tools
calculate profile residuals and a held-out proper rigid registration; they do
not perform Rietveld refinement, indexing, segmentation, reconstruction, or
feature matching.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from typing import Any, TypeVar, cast

import numpy as np
from langchain.tools import tool
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    ValidationError,
)

from ultra_deepagents.materials.advanced_characterization import (
    CharacterizationInputError,
    DataProvenance,
    ReflectionRegistrationError,
    calculate_diffraction_profile_metrics,
    fit_rigid_registration,
)
from ultra_deepagents.materials.degradation import (
    CalibrationDomainError,
    ClosedInterval,
    CorrosionPenetrationInputs,
    DegradationInputError,
    EvidenceProvenance,
    GeometryFactorCalibration,
    NortonArrheniusCreepModel,
    OxidationKineticsModel,
    ParisTestConditions,
    convert_corrosion_current_to_uniform_penetration,
    evaluate_mode_i_lefm,
    evaluate_norton_arrhenius_creep_rate,
    evaluate_oxidation_mass_gain,
    fit_paris_law,
)
from ultra_deepagents.materials.processing_kinetics import processing_method_support
from ultra_deepagents.materials.validation import (
    EvidenceArtifact,
    ValidationCheck,
    ValidationOutcome,
    assess_scientific_status,
    canonical_record_json,
    parse_assessment_record,
)

TOOL_RESULT_SCHEMA_VERSION = "ultra.materials.bounded-analysis-tool-result.v1"
MAX_TYPED_INPUT_BYTES = 2 * 1024 * 1024
MAX_TYPED_OUTPUT_BYTES = 1024 * 1024
MAX_TYPED_PARIS_OBSERVATIONS = 20_000
MAX_TYPED_REGISTRATION_POINTS = 10_000
_WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")


class BoundedMaterialsToolError(RuntimeError):
    """A typed request failed before producing a supported result."""

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(message or code)
        self.code = code
        self.public_message = " ".join((message or code).split())[:1000]


class _ClosedScientificInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


class DegradationProvenanceInput(_ClosedScientificInput):
    """Caller declaration; digest syntax is checked but source bytes are not re-hashed."""

    artifact_id: str = Field(min_length=1, max_length=256)
    sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    locator: str = Field(min_length=1, max_length=2048)
    citation: str = Field(min_length=1, max_length=2048)


class CharacterizationProvenanceInput(_ClosedScientificInput):
    """Caller declaration; digest syntax is checked but source bytes are not re-hashed."""

    artifact_id: str = Field(min_length=1, max_length=256)
    sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    locator: str = Field(min_length=1, max_length=2048)
    processing_history_id: str | None = Field(default=None, min_length=1, max_length=256)


class ClosedIntervalInput(_ClosedScientificInput):
    quantity: str = Field(min_length=1, max_length=128)
    unit: str = Field(min_length=1, max_length=64)
    lower: float
    upper: float


class GeometryFactorInput(_ClosedScientificInput):
    geometry_id: str = Field(min_length=1, max_length=256)
    crack_length_definition: str = Field(min_length=1, max_length=512)
    nominal_stress_definition: str = Field(min_length=1, max_length=512)
    geometry_factor: float
    domain: ClosedIntervalInput
    evaluated_parameter: float
    provenance: DegradationProvenanceInput


class ParisConditionsInput(_ClosedScientificInput):
    material_state_id: str = Field(min_length=1, max_length=1024)
    environment_id: str = Field(min_length=1, max_length=1024)
    load_ratio: float
    temperature_k: float
    cycle_frequency_hz: float
    waveform_id: str = Field(min_length=1, max_length=1024)
    specimen_thickness_m: float
    specimen_geometry_id: str = Field(min_length=1, max_length=1024)
    delta_k_definition_id: str = Field(min_length=1, max_length=1024)
    crack_growth_rate_method_id: str = Field(min_length=1, max_length=1024)


_InputT = TypeVar("_InputT", bound=_ClosedScientificInput)


def _validated_input(
    value: _InputT | Mapping[str, Any],
    model_type: type[_InputT],
    *,
    field_name: str,
) -> _InputT:
    if isinstance(value, model_type):
        return value
    if not isinstance(value, Mapping):
        raise BoundedMaterialsToolError(
            "invalid_typed_input", f"{field_name} must be a closed scientific record"
        )
    try:
        return cast(_InputT, model_type.model_validate(dict(value), strict=True))
    except ValidationError as exc:
        errors = exc.errors(include_url=False, include_context=False)
        detail = errors[0] if errors else {"msg": "invalid record"}
        location = ".".join(str(item) for item in detail.get("loc", ()))
        message = str(detail.get("msg") or "invalid record")
        raise BoundedMaterialsToolError(
            "invalid_typed_input",
            f"{field_name}{'.' + location if location else ''}: {message}",
        ) from exc


def _degradation_provenance(
    value: DegradationProvenanceInput | Mapping[str, Any],
    *,
    field_name: str,
) -> EvidenceProvenance:
    record = _validated_input(value, DegradationProvenanceInput, field_name=field_name)
    _reject_filesystem_locator(record.locator, field_name=f"{field_name}.locator")
    return EvidenceProvenance(**record.model_dump())


def _characterization_provenance(
    value: CharacterizationProvenanceInput | Mapping[str, Any],
    *,
    field_name: str,
) -> DataProvenance:
    record = _validated_input(value, CharacterizationProvenanceInput, field_name=field_name)
    _reject_filesystem_locator(record.locator, field_name=f"{field_name}.locator")
    return DataProvenance(**record.model_dump())


def _reject_filesystem_locator(value: str, *, field_name: str) -> None:
    normalized = value.strip()
    if (
        normalized.startswith(("/", "\\", "~/", "~\\"))
        or _WINDOWS_ABSOLUTE_PATH.match(normalized)
        or normalized.casefold().startswith("file:")
    ):
        raise BoundedMaterialsToolError(
            "invalid_typed_input",
            f"{field_name} must be a content/source locator, not a filesystem path",
        )


def _interval(
    value: ClosedIntervalInput | Mapping[str, Any],
    *,
    field_name: str,
) -> ClosedInterval:
    record = _validated_input(value, ClosedIntervalInput, field_name=field_name)
    return ClosedInterval(**record.model_dump())


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _jsonable(value.model_dump())
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value


def _canonical_json_bytes(value: Any, *, label: str, limit: int) -> bytes:
    try:
        payload = json.dumps(
            _jsonable(value),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise BoundedMaterialsToolError(
            "invalid_typed_input", f"{label} must be finite JSON"
        ) from exc
    if not payload or len(payload) > limit:
        raise BoundedMaterialsToolError(
            "typed_input_too_large" if limit == MAX_TYPED_INPUT_BYTES else "typed_output_too_large",
            f"{label} exceeds the {limit}-byte limit",
        )
    return payload


def _input_evidence(payload: bytes) -> dict[str, Any]:
    return {
        "algorithm": "sha256",
        "canonicalization": "UTF-8 JSON, sorted keys, compact separators, finite numbers",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _json_artifact(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = _canonical_json_bytes(value, label="analysis artifact", limit=MAX_TYPED_OUTPUT_BYTES)
    return {
        "media_type": "application/json",
        "canonical_json": payload.decode("utf-8"),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _validation_artifact(
    *,
    operation: str,
    analysis_artifact: Mapping[str, Any],
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
    validation_passed: bool,
    provenance_declarations_present: bool,
) -> dict[str, Any]:
    evidence = EvidenceArtifact(
        name=f"typed {operation} analysis record",
        sha256=str(analysis_artifact["sha256"]),
        artifact_id=f"typed-{operation}:{analysis_artifact['sha256']}",
        size_bytes=int(analysis_artifact["size_bytes"]),
    )
    calculation_check = ValidationCheck(
        validator_id=f"materials.bounded_tool.{operation}",
        outcome=(ValidationOutcome.PASS if validation_passed else ValidationOutcome.FAIL),
        observed=_jsonable(observed),
        expected=_jsonable(expected),
        units="operation-specific SI or explicitly declared units",
        tolerance_rationale=(
            "The qualified kernel rejects non-finite inputs/results, domain extrapolation, "
            "unit/frame ambiguity, and incomplete held-out partitions before this pass is emitted."
        ),
        required=True,
        critical=True,
        library_versions={"numpy": np.__version__},
        evidence=(evidence,),
        message=(
            "Bounded deterministic data-reduction kernel completed inside its declared scope."
            if validation_passed
            else "The bounded calculation completed, but a required applicability check failed."
        ),
    )
    checks = [calculation_check]
    if provenance_declarations_present:
        checks.append(
            ValidationCheck(
                validator_id="materials.bounded_tool.provenance_bytes_bound",
                outcome=ValidationOutcome.SKIP,
                observed={
                    "caller_declarations_present": True,
                    "digest_syntax_validated": True,
                    "source_bytes_resolved": False,
                    "source_bytes_rehashed": False,
                    "digest_match_independently_verified": False,
                },
                expected={
                    "source_bytes_resolved": True,
                    "source_bytes_rehashed": True,
                    "digest_match_independently_verified": True,
                },
                units="1",
                tolerance_rationale=(
                    "Provenance binding requires exact source-byte retrieval and SHA-256 replay; "
                    "a syntactically valid caller-supplied digest has zero evidentiary tolerance."
                ),
                required=True,
                critical=True,
                library_versions={"hash_algorithm": "sha256"},
                evidence=(evidence,),
                message=(
                    "Caller-declared digest syntax is valid, but this typed surface did not "
                    "resolve or re-hash the referenced source bytes."
                ),
            )
        )
    assessment = assess_scientific_status(
        run_status="succeeded",
        checks=tuple(checks),
        required_validator_ids=tuple(check.validator_id for check in checks),
        capability_supported=True,
    )
    canonical = canonical_record_json(assessment)
    parse_assessment_record(json.loads(canonical))
    payload = canonical.encode("utf-8")
    return {
        "media_type": "application/json",
        "canonical_json": canonical,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "validation_scope": (
            "bounded_typed_tool_operation_with_declaration_only_provenance"
            if provenance_declarations_present
            else "bounded_typed_tool_operation"
        ),
        "scientific_status": assessment.scientific_status.value,
        "verified": assessment.verified,
    }


def _bounded_response(value: Mapping[str, Any]) -> dict[str, Any]:
    _canonical_json_bytes(value, label="typed result", limit=MAX_TYPED_OUTPUT_BYTES)
    return dict(value)


def _public_error(exc: Exception, *, operation: str) -> dict[str, Any]:
    if isinstance(exc, BoundedMaterialsToolError):
        code = exc.code
        message = exc.public_message
    elif isinstance(exc, ReflectionRegistrationError):
        code = "improper_rotation_required"
        message = " ".join(str(exc).split())[:1000]
    elif isinstance(exc, CalibrationDomainError):
        code = "outside_calibration_domain"
        message = " ".join(str(exc).split())[:1000]
    elif isinstance(exc, CharacterizationInputError):
        code = "invalid_characterization_input"
        message = " ".join(str(exc).split())[:1000]
    else:
        code = "invalid_degradation_input"
        message = " ".join(str(exc).split())[:1000]
    return _bounded_response(
        {
            "schema_version": TOOL_RESULT_SCHEMA_VERSION,
            "ok": False,
            "operation": operation,
            "error": code,
            "message": message,
            "partial_results_returned": False,
            "scientific_result_returned": False,
        }
    )


def _successful_result(
    *,
    operation: str,
    request_payload: bytes,
    result: Mapping[str, Any],
    capability_boundary: Mapping[str, Any],
    validation_observed: Mapping[str, Any],
    validation_expected: Mapping[str, Any],
    validation_passed: bool = True,
    provenance_declarations_present: bool = True,
) -> dict[str, Any]:
    response: dict[str, Any] = {
        "schema_version": TOOL_RESULT_SCHEMA_VERSION,
        "ok": True,
        "operation": operation,
        "input_evidence": _input_evidence(request_payload),
        "result": _jsonable(result),
        "capability_boundary": _jsonable(capability_boundary),
        "partial_results_returned": False,
    }
    if provenance_declarations_present:
        response["provenance_binding"] = {
            "status": "unverified",
            "mode": "caller_declared_digest_only",
            "digest_syntax_validated": True,
            "source_bytes_resolved": False,
            "source_bytes_rehashed": False,
            "digest_match_independently_verified": False,
        }
    analysis_artifact = _json_artifact(response)
    response["analysis_artifact"] = analysis_artifact
    response["materials_validation_artifact"] = _validation_artifact(
        operation=operation,
        analysis_artifact=analysis_artifact,
        observed=validation_observed,
        expected=validation_expected,
        validation_passed=validation_passed,
        provenance_declarations_present=provenance_declarations_present,
    )
    return _bounded_response(response)


def _partition_summary(values: Sequence[int], *, label: str) -> dict[str, Any]:
    payload = _canonical_json_bytes(list(values), label=label, limit=MAX_TYPED_INPUT_BYTES)
    return {
        "count": len(values),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "canonicalization": "ordered integer JSON array",
    }


def evaluate_mode_i_lefm_typed(
    *,
    nominal_tensile_stress_pa: float,
    crack_length_m: float,
    remaining_ligament_m: float,
    thickness_m: float,
    yield_strength_pa: float,
    constraint_state: str,
    minimum_dimension_to_plastic_zone_ratio: float,
    geometry: GeometryFactorInput | Mapping[str, Any],
    criterion_provenance: DegradationProvenanceInput | Mapping[str, Any],
) -> dict[str, Any]:
    """Run the bounded Mode-I LEFM algebra/applicability screen."""

    operation = "mode_i_lefm_screen"
    raw_request = {
        "nominal_tensile_stress_pa": nominal_tensile_stress_pa,
        "crack_length_m": crack_length_m,
        "remaining_ligament_m": remaining_ligament_m,
        "thickness_m": thickness_m,
        "yield_strength_pa": yield_strength_pa,
        "constraint_state": constraint_state,
        "minimum_dimension_to_plastic_zone_ratio": minimum_dimension_to_plastic_zone_ratio,
        "geometry": _jsonable(geometry),
        "criterion_provenance": _jsonable(criterion_provenance),
    }
    try:
        request_payload = _canonical_json_bytes(
            raw_request, label="Mode-I LEFM request", limit=MAX_TYPED_INPUT_BYTES
        )
        geometry_input = _validated_input(geometry, GeometryFactorInput, field_name="geometry")
        geometry_value = GeometryFactorCalibration(
            geometry_id=geometry_input.geometry_id,
            crack_length_definition=geometry_input.crack_length_definition,
            nominal_stress_definition=geometry_input.nominal_stress_definition,
            geometry_factor=geometry_input.geometry_factor,
            domain=_interval(geometry_input.domain, field_name="geometry.domain"),
            evaluated_parameter=geometry_input.evaluated_parameter,
            provenance=_degradation_provenance(
                geometry_input.provenance, field_name="geometry.provenance"
            ),
        )
        evaluated = evaluate_mode_i_lefm(
            nominal_tensile_stress_pa=nominal_tensile_stress_pa,
            crack_length_m=crack_length_m,
            remaining_ligament_m=remaining_ligament_m,
            thickness_m=thickness_m,
            yield_strength_pa=yield_strength_pa,
            constraint_state=constraint_state,
            minimum_dimension_to_plastic_zone_ratio=minimum_dimension_to_plastic_zone_ratio,
            geometry=geometry_value,
            criterion_provenance=_degradation_provenance(
                criterion_provenance, field_name="criterion_provenance"
            ),
        )
    except (BoundedMaterialsToolError, DegradationInputError) as exc:
        return _public_error(exc, operation=operation)
    return _successful_result(
        operation=operation,
        request_payload=request_payload,
        result=_jsonable(evaluated),
        capability_boundary={
            "mode_i_stress_intensity_calculated": True,
            "declared_small_scale_yielding_screened": True,
            "fracture_toughness_measured": False,
            "residual_stress_analyzed": False,
            "failure_or_life_predicted": False,
            "astm_e399_compliance_claimed": False,
        },
        validation_observed={
            "applicability_passed": evaluated.applicability_passed,
            "check_count": len(evaluated.applicability_checks),
            "geometry_domain_bound": True,
        },
        validation_expected={
            "all_inputs_finite": True,
            "geometry_coordinate_matches_specimen_dimensions": True,
            "criterion_provenance_declaration_is_explicit": True,
        },
        validation_passed=evaluated.applicability_passed,
    )


def fit_paris_law_typed(
    *,
    delta_k_mpa_sqrt_m: Sequence[float],
    crack_growth_rate_m_per_cycle: Sequence[float],
    calibration_indices: Sequence[int],
    held_out_indices: Sequence[int],
    conditions: ParisConditionsInput | Mapping[str, Any],
    observations_provenance: DegradationProvenanceInput | Mapping[str, Any],
    prediction_delta_k_mpa_sqrt_m: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Fit a leakage-resistant Paris relation and optionally interpolate."""

    operation = "paris_law_fit"
    raw_request = {
        "delta_k_mpa_sqrt_m": _jsonable(delta_k_mpa_sqrt_m),
        "crack_growth_rate_m_per_cycle": _jsonable(crack_growth_rate_m_per_cycle),
        "calibration_indices": _jsonable(calibration_indices),
        "held_out_indices": _jsonable(held_out_indices),
        "conditions": _jsonable(conditions),
        "observations_provenance": _jsonable(observations_provenance),
        "prediction_delta_k_mpa_sqrt_m": _jsonable(prediction_delta_k_mpa_sqrt_m),
    }
    try:
        request_payload = _canonical_json_bytes(
            raw_request, label="Paris-law request", limit=MAX_TYPED_INPUT_BYTES
        )
        try:
            observation_count = len(delta_k_mpa_sqrt_m)
            growth_rate_count = len(crack_growth_rate_m_per_cycle)
        except TypeError as exc:
            raise BoundedMaterialsToolError(
                "invalid_typed_input",
                "Paris delta-K and growth-rate inputs must be bounded sequences",
            ) from exc
        if (
            observation_count > MAX_TYPED_PARIS_OBSERVATIONS
            or growth_rate_count > MAX_TYPED_PARIS_OBSERVATIONS
        ):
            raise BoundedMaterialsToolError(
                "typed_input_too_large",
                "Paris fitting exceeds the "
                f"{MAX_TYPED_PARIS_OBSERVATIONS}-observation typed-tool limit",
            )
        conditions_input = _validated_input(
            conditions, ParisConditionsInput, field_name="conditions"
        )
        conditions_value = ParisTestConditions(**conditions_input.model_dump())
        fitted = fit_paris_law(
            delta_k_mpa_sqrt_m,
            crack_growth_rate_m_per_cycle,
            calibration_indices=calibration_indices,
            held_out_indices=held_out_indices,
            conditions=conditions_value,
            observations_provenance=_degradation_provenance(
                observations_provenance, field_name="observations_provenance"
            ),
        )
        predictions: list[float] | None = None
        if prediction_delta_k_mpa_sqrt_m is not None:
            predictions = fitted.predict_growth_rate_m_per_cycle(
                prediction_delta_k_mpa_sqrt_m,
                conditions=conditions_value,
            ).tolist()
    except (BoundedMaterialsToolError, DegradationInputError) as exc:
        return _public_error(exc, operation=operation)

    result = {
        "schema_version": fitted.schema_version,
        "coefficient_c": fitted.coefficient_c,
        "exponent_m": fitted.exponent_m,
        "coefficient_unit": fitted.coefficient_unit,
        "delta_k_domain_mpa_sqrt_m": _jsonable(fitted.delta_k_domain_mpa_sqrt_m),
        "conditions": _jsonable(fitted.conditions),
        "calibration_indices": list(fitted.calibration_indices),
        "held_out_indices": list(fitted.held_out_indices),
        "calibration_partition": _partition_summary(
            fitted.calibration_indices, label="Paris calibration indices"
        ),
        "held_out_partition": _partition_summary(
            fitted.held_out_indices, label="Paris held-out indices"
        ),
        "calibration_residuals": _jsonable(fitted.calibration_residuals),
        "held_out_residuals": _jsonable(fitted.held_out_residuals),
        "observations_provenance": _jsonable(fitted.observations_provenance),
        "regression_space": fitted.regression_space,
        "weighting_scheme": fitted.weighting_scheme,
        "prediction_delta_k_mpa_sqrt_m": (
            None if prediction_delta_k_mpa_sqrt_m is None else list(prediction_delta_k_mpa_sqrt_m)
        ),
        "predicted_growth_rate_m_per_cycle": predictions,
        "method_reference_url": fitted.method_reference_url,
        "standard_compliance_claimed": fitted.standard_compliance_claimed,
        "validation_only": fitted.validation_only,
        "limitation": fitted.limitation,
    }
    return _successful_result(
        operation=operation,
        request_payload=request_payload,
        result=result,
        capability_boundary={
            "paris_regime_fit_calculated": True,
            "disjoint_complete_holdout_scored": True,
            "prediction_requested": prediction_delta_k_mpa_sqrt_m is not None,
            "prediction_is_interpolation_only": predictions is not None,
            "variable_amplitude_life_predicted": False,
            "component_failure_predicted": False,
            "astm_e647_compliance_claimed": False,
        },
        validation_observed={
            "calibration_count": fitted.calibration_residuals.count,
            "held_out_count": fitted.held_out_residuals.count,
            "held_out_inside_calibration_domain": True,
        },
        validation_expected={
            "calibration_and_holdout_disjoint": True,
            "calibration_and_holdout_cover_all_rows": True,
            "no_extrapolation": True,
        },
    )


def evaluate_norton_arrhenius_creep_typed(
    *,
    pre_exponential_per_s: float,
    reference_stress_pa: float,
    stress_exponent: float,
    activation_energy_j_per_mol: float,
    stress_domain_pa: ClosedIntervalInput | Mapping[str, Any],
    temperature_domain_k: ClosedIntervalInput | Mapping[str, Any],
    material_state_id: str,
    environment_id: str,
    stress_measure_id: str,
    model_provenance: DegradationProvenanceInput | Mapping[str, Any],
    stress_pa: float,
    temperature_k: float,
) -> dict[str, Any]:
    """Evaluate a calibrated secondary-creep scalar inside both domains."""

    operation = "norton_arrhenius_secondary_creep"
    raw_request = {
        name: _jsonable(value) for name, value in locals().copy().items() if name != "operation"
    }
    try:
        request_payload = _canonical_json_bytes(
            raw_request, label="Norton-Arrhenius request", limit=MAX_TYPED_INPUT_BYTES
        )
        model = NortonArrheniusCreepModel(
            pre_exponential_per_s=pre_exponential_per_s,
            reference_stress_pa=reference_stress_pa,
            stress_exponent=stress_exponent,
            activation_energy_j_per_mol=activation_energy_j_per_mol,
            stress_domain_pa=_interval(stress_domain_pa, field_name="stress_domain_pa"),
            temperature_domain_k=_interval(temperature_domain_k, field_name="temperature_domain_k"),
            material_state_id=material_state_id,
            environment_id=environment_id,
            stress_measure_id=stress_measure_id,
            provenance=_degradation_provenance(model_provenance, field_name="model_provenance"),
        )
        evaluated = evaluate_norton_arrhenius_creep_rate(
            model,
            stress_pa=stress_pa,
            temperature_k=temperature_k,
            material_state_id=material_state_id,
            environment_id=environment_id,
        )
    except (BoundedMaterialsToolError, DegradationInputError) as exc:
        return _public_error(exc, operation=operation)
    return _successful_result(
        operation=operation,
        request_payload=request_payload,
        result=_jsonable(evaluated),
        capability_boundary={
            "secondary_steady_state_rate_calculated": True,
            "primary_or_tertiary_creep_calculated": False,
            "rupture_or_damage_calculated": False,
            "remaining_life_predicted": False,
        },
        validation_observed={"stress_in_domain": True, "temperature_in_domain": True},
        validation_expected={
            "material_and_environment_match_calibration": True,
            "rate_finite_and_positive": True,
        },
    )


def evaluate_oxidation_mass_gain_typed(
    *,
    law: str,
    rate_constant: float,
    rate_constant_unit: str,
    initial_areal_mass_gain_kg_per_m2: float,
    time_domain_s: ClosedIntervalInput | Mapping[str, Any],
    temperature_domain_k: ClosedIntervalInput | Mapping[str, Any],
    material_state_id: str,
    environment_id: str,
    area_basis_id: str,
    model_provenance: DegradationProvenanceInput | Mapping[str, Any],
    exposure_time_s: float,
    temperature_k: float,
) -> dict[str, Any]:
    """Evaluate one calibrated linear/parabolic isothermal mass-gain law."""

    operation = "oxidation_areal_mass_gain"
    raw_request = {
        name: _jsonable(value) for name, value in locals().copy().items() if name != "operation"
    }
    try:
        request_payload = _canonical_json_bytes(
            raw_request, label="oxidation request", limit=MAX_TYPED_INPUT_BYTES
        )
        model = OxidationKineticsModel(
            law=law,
            rate_constant=rate_constant,
            rate_constant_unit=rate_constant_unit,
            initial_areal_mass_gain_kg_per_m2=initial_areal_mass_gain_kg_per_m2,
            time_domain_s=_interval(time_domain_s, field_name="time_domain_s"),
            temperature_domain_k=_interval(temperature_domain_k, field_name="temperature_domain_k"),
            material_state_id=material_state_id,
            environment_id=environment_id,
            area_basis_id=area_basis_id,
            provenance=_degradation_provenance(model_provenance, field_name="model_provenance"),
        )
        evaluated = evaluate_oxidation_mass_gain(
            model,
            exposure_time_s=exposure_time_s,
            temperature_k=temperature_k,
            material_state_id=material_state_id,
            environment_id=environment_id,
        )
    except (BoundedMaterialsToolError, DegradationInputError) as exc:
        return _public_error(exc, operation=operation)
    return _successful_result(
        operation=operation,
        request_payload=request_payload,
        result=_jsonable(evaluated),
        capability_boundary={
            "isothermal_areal_mass_gain_calculated": True,
            "law": evaluated.model.law,
            "calibrated_isothermal_temperature_k": evaluated.temperature_k,
            "temperature_dependence_modeled": False,
            "oxide_thickness_calculated": False,
            "metal_loss_calculated": False,
            "spallation_or_breakaway_modeled": False,
            "component_life_predicted": False,
        },
        validation_observed={
            "time_in_domain": True,
            "evaluation_temperature_equals_single_calibration_temperature": True,
            "rate_constant_unit": evaluated.model.rate_constant_unit,
        },
        validation_expected={
            "material_environment_and_area_basis_bound": True,
            "singleton_isothermal_temperature_required": True,
            "law_consistent_rate_constant_unit_required": True,
            "mass_gain_finite_and_nonnegative": True,
        },
    )


def convert_uniform_corrosion_typed(
    *,
    corrosion_current_density_a_per_m2: float,
    equivalent_mass_kg_per_mol_electron: float,
    density_kg_per_m3: float,
    current_efficiency: float,
    duration_s: float,
    material_state_id: str,
    environment_id: str,
    current_density_area_basis_id: str,
    current_density_provenance: DegradationProvenanceInput | Mapping[str, Any],
    equivalent_mass_provenance: DegradationProvenanceInput | Mapping[str, Any],
    density_provenance: DegradationProvenanceInput | Mapping[str, Any],
    efficiency_provenance: DegradationProvenanceInput | Mapping[str, Any],
) -> dict[str, Any]:
    """Convert current density to average uniform penetration using Faraday's law."""

    operation = "faraday_uniform_corrosion_conversion"
    raw_request = {
        name: _jsonable(value) for name, value in locals().copy().items() if name != "operation"
    }
    try:
        request_payload = _canonical_json_bytes(
            raw_request, label="uniform-corrosion request", limit=MAX_TYPED_INPUT_BYTES
        )
        inputs = CorrosionPenetrationInputs(
            corrosion_current_density_a_per_m2=corrosion_current_density_a_per_m2,
            equivalent_mass_kg_per_mol_electron=equivalent_mass_kg_per_mol_electron,
            density_kg_per_m3=density_kg_per_m3,
            current_efficiency=current_efficiency,
            duration_s=duration_s,
            material_state_id=material_state_id,
            environment_id=environment_id,
            current_density_area_basis_id=current_density_area_basis_id,
            current_density_provenance=_degradation_provenance(
                current_density_provenance, field_name="current_density_provenance"
            ),
            equivalent_mass_provenance=_degradation_provenance(
                equivalent_mass_provenance, field_name="equivalent_mass_provenance"
            ),
            density_provenance=_degradation_provenance(
                density_provenance, field_name="density_provenance"
            ),
            efficiency_provenance=_degradation_provenance(
                efficiency_provenance, field_name="efficiency_provenance"
            ),
        )
        evaluated = convert_corrosion_current_to_uniform_penetration(inputs)
    except (BoundedMaterialsToolError, DegradationInputError) as exc:
        return _public_error(exc, operation=operation)
    return _successful_result(
        operation=operation,
        request_payload=request_payload,
        result=_jsonable(evaluated),
        capability_boundary={
            "faraday_uniform_conversion_calculated": True,
            "pitting_or_crevice_depth_calculated": False,
            "passivation_or_transport_modeled": False,
            "component_life_predicted": False,
            "astm_g102_compliance_claimed": False,
        },
        validation_observed={
            "current_efficiency_bounded_zero_to_one": True,
            "result_finite_and_nonnegative": True,
        },
        validation_expected={
            "equivalent_mass_density_efficiency_and_area_basis_declared": True,
            "uniform_dissolution_assumption_explicit": True,
        },
    )


def calculate_diffraction_profile_metrics_typed(
    *,
    coordinate: Sequence[float],
    observed_intensity: Sequence[float],
    calculated_intensity: Sequence[float],
    coordinate_unit: str,
    observed_intensity_unit: str,
    calculated_intensity_unit: str,
    observed_provenance: CharacterizationProvenanceInput | Mapping[str, Any],
    calculated_provenance: CharacterizationProvenanceInput | Mapping[str, Any],
    included_mask: Sequence[bool] | None = None,
    uncertainties: Sequence[float] | None = None,
    uncertainty_semantics: str | None = None,
    refined_parameter_count: int | None = None,
    independent_constraint_count: int = 0,
) -> dict[str, Any]:
    """Calculate convention-explicit Rp/Rwp and statistically valid metrics."""

    operation = "diffraction_profile_metrics"
    raw_request = {
        name: _jsonable(value) for name, value in locals().copy().items() if name != "operation"
    }
    try:
        request_payload = _canonical_json_bytes(
            raw_request, label="diffraction-profile request", limit=MAX_TYPED_INPUT_BYTES
        )
        evaluated = calculate_diffraction_profile_metrics(
            coordinate,
            observed_intensity,
            calculated_intensity,
            coordinate_unit=coordinate_unit,
            observed_intensity_unit=observed_intensity_unit,
            calculated_intensity_unit=calculated_intensity_unit,
            observed_provenance=_characterization_provenance(
                observed_provenance, field_name="observed_provenance"
            ),
            calculated_provenance=_characterization_provenance(
                calculated_provenance, field_name="calculated_provenance"
            ),
            included_mask=included_mask,
            uncertainties=uncertainties,
            uncertainty_semantics=uncertainty_semantics,
            refined_parameter_count=refined_parameter_count,
            independent_constraint_count=independent_constraint_count,
        )
    except (BoundedMaterialsToolError, CharacterizationInputError) as exc:
        return _public_error(exc, operation=operation)
    return _successful_result(
        operation=operation,
        request_payload=request_payload,
        result=_jsonable(evaluated),
        capability_boundary={
            "profile_residual_metrics_calculated": True,
            "statistical_goodness_of_fit_calculated": evaluated.goodness_of_fit is not None,
            "rietveld_refinement_performed": False,
            "phase_identification_performed": False,
            "instrument_calibration_performed": False,
            "model_uniqueness_established": False,
        },
        validation_observed={
            "total_point_count": evaluated.total_point_count,
            "included_point_count": evaluated.included_point_count,
            "weighting_scheme": evaluated.weighting_scheme,
        },
        validation_expected={
            "strictly_increasing_coordinate": True,
            "matching_intensity_units": True,
            "uncertainty_semantics_explicit_when_statistical": True,
        },
    )


def fit_rigid_registration_typed(
    *,
    source_points: Sequence[Sequence[float]],
    target_points: Sequence[Sequence[float]],
    source_frame_id: str,
    target_frame_id: str,
    source_coordinate_unit: str,
    target_coordinate_unit: str,
    source_provenance: CharacterizationProvenanceInput | Mapping[str, Any],
    target_provenance: CharacterizationProvenanceInput | Mapping[str, Any],
    calibration_indices: Sequence[int],
    held_out_indices: Sequence[int],
) -> dict[str, Any]:
    """Fit a proper rigid transform and score a complete held-out partition."""

    operation = "held_out_rigid_registration"
    raw_request = {
        name: _jsonable(value) for name, value in locals().copy().items() if name != "operation"
    }
    try:
        request_payload = _canonical_json_bytes(
            raw_request, label="rigid-registration request", limit=MAX_TYPED_INPUT_BYTES
        )
        try:
            source_count = len(source_points)
            target_count = len(target_points)
        except TypeError as exc:
            raise BoundedMaterialsToolError(
                "invalid_typed_input",
                "source_points and target_points must be bounded point sequences",
            ) from exc
        if (
            source_count > MAX_TYPED_REGISTRATION_POINTS
            or target_count > MAX_TYPED_REGISTRATION_POINTS
        ):
            raise BoundedMaterialsToolError(
                "typed_input_too_large",
                "rigid registration exceeds the "
                f"{MAX_TYPED_REGISTRATION_POINTS}-correspondence typed-tool limit",
            )
        evaluated = fit_rigid_registration(
            source_points,
            target_points,
            source_frame_id=source_frame_id,
            target_frame_id=target_frame_id,
            source_coordinate_unit=source_coordinate_unit,
            target_coordinate_unit=target_coordinate_unit,
            source_provenance=_characterization_provenance(
                source_provenance, field_name="source_provenance"
            ),
            target_provenance=_characterization_provenance(
                target_provenance, field_name="target_provenance"
            ),
            calibration_indices=calibration_indices,
            held_out_indices=held_out_indices,
        )
    except (BoundedMaterialsToolError, CharacterizationInputError) as exc:
        return _public_error(exc, operation=operation)

    result = {
        "schema_version": evaluated.schema_version,
        "source_frame_id": evaluated.source_frame_id,
        "target_frame_id": evaluated.target_frame_id,
        "coordinate_unit": evaluated.coordinate_unit,
        "source_provenance": _jsonable(evaluated.source_provenance),
        "target_provenance": _jsonable(evaluated.target_provenance),
        "calibration_indices": list(evaluated.calibration_indices),
        "held_out_indices": list(evaluated.held_out_indices),
        "calibration_partition": _partition_summary(
            evaluated.calibration_indices, label="registration calibration indices"
        ),
        "held_out_partition": _partition_summary(
            evaluated.held_out_indices, label="registration held-out indices"
        ),
        "rotation_source_to_target": evaluated.rotation_source_to_target.tolist(),
        "translation_source_to_target": evaluated.translation_source_to_target.tolist(),
        "calibration_residual_norms": evaluated.calibration_residual_norms.tolist(),
        "held_out_residual_norms": evaluated.held_out_residual_norms.tolist(),
        "calibration_statistics": _jsonable(evaluated.calibration_statistics),
        "held_out_statistics": _jsonable(evaluated.held_out_statistics),
        "calibration_cross_covariance_singular_values": (
            evaluated.calibration_cross_covariance_singular_values.tolist()
        ),
        "rotation_determinant": evaluated.rotation_determinant,
        "method_reference_doi": evaluated.method_reference_doi,
        "proper_rotation_enforced": evaluated.proper_rotation_enforced,
        "validation_only": evaluated.validation_only,
        "limitation": evaluated.limitation,
    }
    return _successful_result(
        operation=operation,
        request_payload=request_payload,
        result=result,
        capability_boundary={
            "proper_rigid_transform_fitted": True,
            "held_out_correspondences_scored": True,
            "partitions_cover_every_correspondence": True,
            "feature_identity_established": False,
            "non_rigid_registration_performed": False,
            "segmentation_or_indexing_validated": False,
        },
        validation_observed={
            "calibration_count": evaluated.calibration_statistics.count,
            "held_out_count": evaluated.held_out_statistics.count,
            "rotation_determinant": evaluated.rotation_determinant,
        },
        validation_expected={
            "proper_rotation": True,
            "full_dimensional_calibration": True,
            "disjoint_complete_holdout": True,
            "coordinate_units_match": True,
        },
    )


def processing_method_support_typed() -> dict[str, Any]:
    """Return the honest Scheil/Kawin/phase-field capability boundary."""

    operation = "processing_method_support"
    request_payload = _canonical_json_bytes(
        {}, label="processing support request", limit=MAX_TYPED_INPUT_BYTES
    )
    support = processing_method_support()
    return _successful_result(
        operation=operation,
        request_payload=request_payload,
        result={
            "methods": support,
            "limitation": (
                "This is static support discovery only. It does not prove that a governed "
                "database, required physical parameters, immutable runtime, external HPC "
                "adapter, or complete run contract is present for the current request."
            ),
        },
        capability_boundary={
            "support_discovery_only": True,
            "scheil_execution_performed": False,
            "kawin_execution_performed": False,
            "phase_field_execution_performed": False,
            "toy_solver_substitution_permitted": False,
        },
        validation_observed={
            "method_count": len(support),
            "phase_field_status": support["phase_field"]["status"],
        },
        validation_expected={
            "qualified_methods_named": True,
            "unsupported_external_solver_boundary_explicit": True,
        },
        provenance_declarations_present=False,
    )


def _json_response(value: Mapping[str, Any]) -> str:
    return json.dumps(value, allow_nan=False, indent=2, sort_keys=True)


def build_degradation_characterization_tools(
    *,
    include_degradation: bool = True,
    include_characterization: bool = True,
    include_processing_support: bool = True,
) -> list[Any]:
    """Build only the requested bounded scientific tool groups.

    Agent routing uses these switches to avoid carrying seven irrelevant
    numerical schemas when the user asks only for the zero-argument processing
    support matrix, or carrying degradation schemas during a registration task.
    The default remains the complete surface for direct qualification.
    """

    if not all(
        isinstance(value, bool)
        for value in (
            include_degradation,
            include_characterization,
            include_processing_support,
        )
    ):
        raise TypeError("tool-group selectors must be booleans")

    @tool
    def materials_evaluate_mode_i_lefm(
        nominal_tensile_stress_pa: StrictFloat,
        crack_length_m: StrictFloat,
        remaining_ligament_m: StrictFloat,
        thickness_m: StrictFloat,
        yield_strength_pa: StrictFloat,
        constraint_state: str,
        minimum_dimension_to_plastic_zone_ratio: StrictFloat,
        geometry: GeometryFactorInput,
        criterion_provenance: DegradationProvenanceInput,
    ) -> str:
        """Calculate K_I and a caller-cited small-scale-yielding applicability screen for one Mode-I geometry. Geometry must declare Y for crack_length/(crack_length+remaining_ligament), its closed domain, definitions, and source digest. Digest syntax is checked, but source bytes are not resolved or re-hashed and provenance binding remains unverified. This is not fracture-toughness measurement, ASTM E399 compliance, residual-stress analysis, failure prediction, or life prediction. Returns exact content-addressed analysis and materials-validation JSON strings; copy them directly rather than reconstructing verdicts. Accepts no code, commands, filesystem paths, or solver options."""

        return _json_response(
            evaluate_mode_i_lefm_typed(
                nominal_tensile_stress_pa=nominal_tensile_stress_pa,
                crack_length_m=crack_length_m,
                remaining_ligament_m=remaining_ligament_m,
                thickness_m=thickness_m,
                yield_strength_pa=yield_strength_pa,
                constraint_state=constraint_state,
                minimum_dimension_to_plastic_zone_ratio=(minimum_dimension_to_plastic_zone_ratio),
                geometry=geometry,
                criterion_provenance=criterion_provenance,
            )
        )

    @tool
    def materials_fit_paris_law(
        delta_k_mpa_sqrt_m: list[StrictFloat],
        crack_growth_rate_m_per_cycle: list[StrictFloat],
        calibration_indices: list[StrictInt],
        held_out_indices: list[StrictInt],
        conditions: ParisConditionsInput,
        observations_provenance: DegradationProvenanceInput,
        prediction_delta_k_mpa_sqrt_m: list[StrictFloat] | None = None,
    ) -> str:
        """Fit the classical unweighted log-space Paris relation on calibration rows, score disjoint held-out interpolation rows, and optionally predict only inside the calibration Delta-K interval under identical conditions. The two partitions must cover every row. Caller-declared digest syntax is checked, but source bytes are not resolved or re-hashed and provenance binding remains unverified. This is not threshold/overload/short-crack/closure modeling, ASTM E647 compliance, variable-amplitude life, or component failure prediction. Accepts no code, commands, filesystem paths, or arbitrary models."""

        return _json_response(
            fit_paris_law_typed(
                delta_k_mpa_sqrt_m=delta_k_mpa_sqrt_m,
                crack_growth_rate_m_per_cycle=crack_growth_rate_m_per_cycle,
                calibration_indices=calibration_indices,
                held_out_indices=held_out_indices,
                conditions=conditions,
                observations_provenance=observations_provenance,
                prediction_delta_k_mpa_sqrt_m=prediction_delta_k_mpa_sqrt_m,
            )
        )

    @tool
    def materials_evaluate_norton_arrhenius_creep(
        pre_exponential_per_s: StrictFloat,
        reference_stress_pa: StrictFloat,
        stress_exponent: StrictFloat,
        activation_energy_j_per_mol: StrictFloat,
        stress_domain_pa: ClosedIntervalInput,
        temperature_domain_k: ClosedIntervalInput,
        material_state_id: str,
        environment_id: str,
        stress_measure_id: str,
        model_provenance: DegradationProvenanceInput,
        stress_pa: StrictFloat,
        temperature_k: StrictFloat,
    ) -> str:
        """Evaluate A*(stress/reference_stress)^n*exp(-Q/RT) only inside the caller-declared stress/temperature/material/environment calibration domain. Digest declarations are syntax-checked but source bytes are not resolved or re-hashed, so provenance binding remains unverified. Reports effective secondary steady-state creep rate only; it does not calculate primary or tertiary creep, multiaxial flow, damage, rupture, oxidation coupling, or remaining life. Accepts no code, commands, or filesystem paths."""

        return _json_response(
            evaluate_norton_arrhenius_creep_typed(
                pre_exponential_per_s=pre_exponential_per_s,
                reference_stress_pa=reference_stress_pa,
                stress_exponent=stress_exponent,
                activation_energy_j_per_mol=activation_energy_j_per_mol,
                stress_domain_pa=stress_domain_pa,
                temperature_domain_k=temperature_domain_k,
                material_state_id=material_state_id,
                environment_id=environment_id,
                stress_measure_id=stress_measure_id,
                model_provenance=model_provenance,
                stress_pa=stress_pa,
                temperature_k=temperature_k,
            )
        )

    @tool
    def materials_evaluate_oxidation_mass_gain(
        law: str,
        rate_constant: StrictFloat,
        rate_constant_unit: str,
        initial_areal_mass_gain_kg_per_m2: StrictFloat,
        time_domain_s: ClosedIntervalInput,
        temperature_domain_k: ClosedIntervalInput,
        material_state_id: str,
        environment_id: str,
        area_basis_id: str,
        model_provenance: DegradationProvenanceInput,
        exposure_time_s: StrictFloat,
        temperature_k: StrictFloat,
    ) -> str:
        """Evaluate a caller-declared linear or parabolic areal mass-gain law only at one exact calibrated isothermal temperature. rate_constant_unit is required and must be exactly kg*m^-2*s^-1 for linear or kg^2*m^-4*s^-1 for parabolic; aliases and law/unit mismatches fail closed. A multi-temperature domain is rejected because the constant has no Arrhenius term. Digest declarations are syntax-checked but source bytes are not resolved or re-hashed. Areal mass gain is not oxide thickness or metal loss. The tool excludes transient, breakaway, spallation, volatilization, cyclic, transport-limited, multiphase, and life modeling. Accepts no code, commands, or filesystem paths."""

        return _json_response(
            evaluate_oxidation_mass_gain_typed(
                law=law,
                rate_constant=rate_constant,
                rate_constant_unit=rate_constant_unit,
                initial_areal_mass_gain_kg_per_m2=initial_areal_mass_gain_kg_per_m2,
                time_domain_s=time_domain_s,
                temperature_domain_k=temperature_domain_k,
                material_state_id=material_state_id,
                environment_id=environment_id,
                area_basis_id=area_basis_id,
                model_provenance=model_provenance,
                exposure_time_s=exposure_time_s,
                temperature_k=temperature_k,
            )
        )

    @tool
    def materials_convert_uniform_corrosion(
        corrosion_current_density_a_per_m2: StrictFloat,
        equivalent_mass_kg_per_mol_electron: StrictFloat,
        density_kg_per_m3: StrictFloat,
        current_efficiency: StrictFloat,
        duration_s: StrictFloat,
        material_state_id: str,
        environment_id: str,
        current_density_area_basis_id: str,
        current_density_provenance: DegradationProvenanceInput,
        equivalent_mass_provenance: DegradationProvenanceInput,
        density_provenance: DegradationProvenanceInput,
        efficiency_provenance: DegradationProvenanceInput,
    ) -> str:
        """Convert corrosion current density to average uniform mass-loss flux, penetration rate, and penetration using Faraday's law with explicit equivalent mass, density, efficiency, area basis, and caller-declared source digests. Digest syntax is checked, but source bytes are not resolved or re-hashed and provenance binding remains unverified. This assumes constant spatially uniform dissolution; it does not predict pitting, crevice/galvanic attack, passivation, transport limits, localized depth, ASTM G102 compliance, or component life. Accepts no code, commands, or filesystem paths."""

        return _json_response(
            convert_uniform_corrosion_typed(
                corrosion_current_density_a_per_m2=corrosion_current_density_a_per_m2,
                equivalent_mass_kg_per_mol_electron=equivalent_mass_kg_per_mol_electron,
                density_kg_per_m3=density_kg_per_m3,
                current_efficiency=current_efficiency,
                duration_s=duration_s,
                material_state_id=material_state_id,
                environment_id=environment_id,
                current_density_area_basis_id=current_density_area_basis_id,
                current_density_provenance=current_density_provenance,
                equivalent_mass_provenance=equivalent_mass_provenance,
                density_provenance=density_provenance,
                efficiency_provenance=efficiency_provenance,
            )
        )

    @tool
    def materials_calculate_diffraction_profile_metrics(
        coordinate: list[StrictFloat],
        observed_intensity: list[StrictFloat],
        calculated_intensity: list[StrictFloat],
        coordinate_unit: str,
        observed_intensity_unit: str,
        calculated_intensity_unit: str,
        observed_provenance: CharacterizationProvenanceInput,
        calculated_provenance: CharacterizationProvenanceInput,
        included_mask: list[StrictBool] | None = None,
        uncertainties: list[StrictFloat] | None = None,
        uncertainty_semantics: str | None = None,
        refined_parameter_count: StrictInt | None = None,
        independent_constraint_count: StrictInt = 0,
    ) -> str:
        """Calculate Rp/Rwp for measured-versus-calculated diffraction profiles and report Rexp, chi-square, reduced chi-square, and goodness of fit only when independent absolute one-sigma uncertainties and positive N-P+C are explicit. Caller-declared digest syntax is checked, but source bytes are not resolved or re-hashed and provenance binding remains unverified. This validates supplied arrays only; it does not run or validate Rietveld refinement, identify phases, calibrate the instrument, or establish model uniqueness. Accepts no code, commands, or filesystem paths."""

        return _json_response(
            calculate_diffraction_profile_metrics_typed(
                coordinate=coordinate,
                observed_intensity=observed_intensity,
                calculated_intensity=calculated_intensity,
                coordinate_unit=coordinate_unit,
                observed_intensity_unit=observed_intensity_unit,
                calculated_intensity_unit=calculated_intensity_unit,
                observed_provenance=observed_provenance,
                calculated_provenance=calculated_provenance,
                included_mask=included_mask,
                uncertainties=uncertainties,
                uncertainty_semantics=uncertainty_semantics,
                refined_parameter_count=refined_parameter_count,
                independent_constraint_count=independent_constraint_count,
            )
        )

    @tool
    def materials_fit_held_out_rigid_registration(
        source_points: list[list[StrictFloat]],
        target_points: list[list[StrictFloat]],
        source_frame_id: str,
        target_frame_id: str,
        source_coordinate_unit: str,
        target_coordinate_unit: str,
        source_provenance: CharacterizationProvenanceInput,
        target_provenance: CharacterizationProvenanceInput,
        calibration_indices: list[StrictInt],
        held_out_indices: list[StrictInt],
    ) -> str:
        """Fit a proper Kabsch rigid transform from source to target using only full-dimensional calibration correspondences and score a disjoint held-out partition that must cover every remaining correspondence. Caller-declared digest syntax is checked, but source bytes are not resolved or re-hashed and provenance binding remains unverified. Reflections, unit/frame ambiguity, rank-deficient calibration, overlap, and omitted points fail closed. This does not establish feature identity, perform non-rigid registration, or validate segmentation, EBSD indexing, TEM/4D-STEM reconstruction, or APT chemistry. Accepts no code, commands, or filesystem paths."""

        return _json_response(
            fit_rigid_registration_typed(
                source_points=source_points,
                target_points=target_points,
                source_frame_id=source_frame_id,
                target_frame_id=target_frame_id,
                source_coordinate_unit=source_coordinate_unit,
                target_coordinate_unit=target_coordinate_unit,
                source_provenance=source_provenance,
                target_provenance=target_provenance,
                calibration_indices=calibration_indices,
                held_out_indices=held_out_indices,
            )
        )

    @tool
    def materials_processing_method_support() -> str:
        """Return the exact qualified processing/kinetics support matrix without running a solver. It distinguishes classic Scheil-Gulliver, post-solidification single-phase 1-D back diffusion, bounded mobility/diffusion, binary isothermal spherical KWN precipitation, and external-solver-only phase field. It never executes or fabricates Scheil, Kawin, phase-field, moving-interface, or coupled solidification results."""

        return _json_response(processing_method_support_typed())

    tools: list[Any] = []
    if include_degradation:
        tools.extend(
            (
                materials_evaluate_mode_i_lefm,
                materials_fit_paris_law,
                materials_evaluate_norton_arrhenius_creep,
                materials_evaluate_oxidation_mass_gain,
                materials_convert_uniform_corrosion,
            )
        )
    if include_characterization:
        tools.extend(
            (
                materials_calculate_diffraction_profile_metrics,
                materials_fit_held_out_rigid_registration,
            )
        )
    if include_processing_support:
        tools.append(materials_processing_method_support)
    return tools


__all__ = [
    "BoundedMaterialsToolError",
    "CharacterizationProvenanceInput",
    "ClosedIntervalInput",
    "DegradationProvenanceInput",
    "GeometryFactorInput",
    "ParisConditionsInput",
    "TOOL_RESULT_SCHEMA_VERSION",
    "build_degradation_characterization_tools",
    "calculate_diffraction_profile_metrics_typed",
    "convert_uniform_corrosion_typed",
    "evaluate_mode_i_lefm_typed",
    "evaluate_norton_arrhenius_creep_typed",
    "evaluate_oxidation_mass_gain_typed",
    "fit_paris_law_typed",
    "fit_rigid_registration_typed",
    "processing_method_support_typed",
]
