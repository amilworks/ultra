"""Bounded crystal-plasticity geometry and input-contract validation.

The module deliberately stops before constitutive integration or a finite-element
solve.  It provides convention-explicit orientation/stress validation, canonical
slip geometry, resolved shear stress, and an auditable CPFE input contract.

Conventions
-----------
``rotation_crystal_to_sample`` is an active proper rotation ``R_sc`` such that
``v_sample = R_sc @ v_crystal``.  Stress is a symmetric Cauchy-stress tensor in
the sample frame.  For slip direction ``d`` and plane normal ``n``, resolved
shear is ``tau = d_sample.T @ sigma_sample @ n_sample``.  Direction and plane
signs are crystallographic conventions, so use ``abs(tau)`` for activation.

The built-in deformation-system tables are a deterministic transcription of
DAMASK 3.1.0 ``damask.Crystal.kinematics('slip')``.  When DAMASK is installed,
``cross_validate_slip_systems_with_damask`` compares the complete Schmid-tensor
sets without depending on arbitrary system ordering or sign.
"""

from __future__ import annotations

import importlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from types import MappingProxyType
from typing import Any, cast

import numpy as np

CPFE_CONTRACT_SCHEMA_VERSION = "1"
CPFE_EXECUTION_SUPPORTED = False
DAMASK_REFERENCE_VERSION = "3.1.0"
DAMASK_CRYSTAL_SOURCE_URL = "https://damask-multiphysics.org/_modules/damask/_crystal.html"
MAX_GRAINS = 1_000_000
MAX_GRAIN_SYSTEM_VALUES = 24_000_000
BATCH_INTERMEDIATE_BYTES = 64 * 1024 * 1024
MAX_HARDENING_PARAMETERS = 256

FCC_111_110 = "fcc-{111}<110>"
FCC_110_110 = "fcc-{110}<110>"
BCC_110_111 = "bcc-{110}<111>"
BCC_112_111 = "bcc-{112}<111>"
BCC_123_111 = "bcc-{123}<111>"
HCP_BASAL_A = "hcp-basal-{0001}<11-20>"
HCP_PRISMATIC_A = "hcp-prismatic-{10-10}<11-20>"
HCP_PYRAMIDAL_A = "hcp-pyramidal-{10-11}<11-20>"
HCP_PYRAMIDAL_CA = "hcp-pyramidal-{10-11}<11-23>"
HCP_PYRAMIDAL2_CA = "hcp-pyramidal-{11-22}<11-23>"

_STRUCTURE_FAMILIES = {
    "fcc": (FCC_111_110, FCC_110_110),
    "bcc": (BCC_110_111, BCC_112_111, BCC_123_111),
    "hcp": (
        HCP_BASAL_A,
        HCP_PRISMATIC_A,
        HCP_PYRAMIDAL_A,
        HCP_PYRAMIDAL_CA,
        HCP_PYRAMIDAL2_CA,
    ),
}
_STRUCTURE_SYMMETRY = {"fcc": "m-3m", "bcc": "m-3m", "hcp": "6/mmm"}
_DAMASK_LATTICE = {"fcc": "cF", "bcc": "cI", "hcp": "hP"}
_STRESS_UNITS = frozenset({"Pa", "kPa", "MPa", "GPa"})
_SOURCE_TYPES = frozenset(
    {
        "database",
        "experimental_calibration",
        "fitted",
        "publication",
        "user_declared",
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class CrystalPlasticityInputError(ValueError):
    """An input is ambiguous, malformed, non-finite, or inconsistent."""


class CrystalPlasticityUnsupportedError(RuntimeError):
    """A recognized operation lacks a qualified execution backend."""


class DamaskReferenceUnavailableError(CrystalPlasticityUnsupportedError):
    """The exact optional DAMASK reference backend is not available."""


@dataclass(frozen=True)
class SlipSystem:
    """One sign-convention-specific slip direction and plane normal."""

    system_id: str
    crystal_structure: str
    family: str
    slip_direction_crystal: tuple[float, float, float]
    plane_normal_crystal: tuple[float, float, float]
    direction_indices: tuple[int, ...]
    plane_indices: tuple[int, ...]
    reference_backend: str = f"DAMASK-{DAMASK_REFERENCE_VERSION}-Crystal.kinematics(slip)"

    def __post_init__(self) -> None:
        direction = np.asarray(self.slip_direction_crystal, dtype=float)
        normal = np.asarray(self.plane_normal_crystal, dtype=float)
        if direction.shape != (3,) or normal.shape != (3,):
            raise CrystalPlasticityInputError("slip directions and normals must be 3-vectors")
        if not np.all(np.isfinite(direction)) or not np.all(np.isfinite(normal)):
            raise CrystalPlasticityInputError("slip directions and normals must be finite")
        if not math.isclose(float(np.linalg.norm(direction)), 1.0, abs_tol=1e-12):
            raise CrystalPlasticityInputError("slip direction must be normalized")
        if not math.isclose(float(np.linalg.norm(normal)), 1.0, abs_tol=1e-12):
            raise CrystalPlasticityInputError("slip-plane normal must be normalized")
        if abs(float(direction @ normal)) > 1e-12:
            raise CrystalPlasticityInputError("slip direction must lie in its slip plane")


@dataclass(frozen=True)
class ResolvedShearResult:
    """Resolved shear for one grain; numerical values retain ``stress_unit``."""

    system_ids: tuple[str, ...]
    stress_unit: str
    resolved_shear_stress: np.ndarray
    normalized_resolved_shear: np.ndarray | None
    reference_stress: float | None

    def __post_init__(self) -> None:
        _freeze_array(self.resolved_shear_stress)
        if self.normalized_resolved_shear is not None:
            _freeze_array(self.normalized_resolved_shear)


@dataclass(frozen=True)
class GrainBatchAnalysis:
    """Batched slip-system response for shared or per-grain sample stresses."""

    phase_id: str
    grain_ids: tuple[str, ...]
    system_ids: tuple[str, ...]
    stress_unit: str
    resolved_shear_stress: np.ndarray
    normalized_resolved_shear: np.ndarray | None
    reference_stress: np.ndarray | None
    max_abs_system_index: np.ndarray

    def __post_init__(self) -> None:
        _freeze_array(self.resolved_shear_stress)
        _freeze_array(self.max_abs_system_index)
        if self.normalized_resolved_shear is not None:
            _freeze_array(self.normalized_resolved_shear)
        if self.reference_stress is not None:
            _freeze_array(self.reference_stress)


@dataclass(frozen=True)
class SourceProvenance:
    """Caller-declared origin and digest of a phase or parameter set.

    Contract validation checks this declaration's schema and digest syntax.  It
    does not resolve source bytes or prove that ``sha256`` matches them.
    """

    source_id: str
    source_type: str
    citation: str
    sha256: str


@dataclass(frozen=True)
class CPFEInputContract:
    """Validated, SI-bound CPFE input metadata; not an executable CPFE model."""

    schema_version: str
    phase_id: str
    crystal_structure: str
    symmetry: str
    c_over_a: float | None
    phase_provenance: SourceProvenance
    orientations_crystal_to_sample: np.ndarray
    slip_families: tuple[str, ...]
    crss_pa: Mapping[str, float]
    crss_provenance: SourceProvenance
    hardening_model_id: str
    hardening_parameters: Mapping[str, float]
    hardening_parameter_units: Mapping[str, str]
    hardening_provenance: SourceProvenance
    execution_supported: bool = False
    unsupported_reason: str = (
        "No qualified constitutive integrator or finite-element/spectral solver backend is "
        "bound; this contract validates inputs only."
    )

    def __post_init__(self) -> None:
        _freeze_array(self.orientations_crystal_to_sample)


@dataclass(frozen=True)
class DamaskCrossValidationResult:
    """Comparison of built-in and DAMASK Schmid-tensor sets."""

    damask_version: str
    crystal_structure: str
    families: tuple[str, ...]
    system_count: int
    minimum_bidirectional_tensor_overlap: float
    passed: bool


def _freeze_array(value: np.ndarray) -> None:
    value.setflags(write=False)


def _normalized(vector: np.ndarray, *, field_name: str) -> tuple[float, float, float]:
    magnitude = float(np.linalg.norm(vector))
    if not math.isfinite(magnitude) or magnitude <= 0.0:
        raise CrystalPlasticityInputError(f"{field_name} must have finite nonzero magnitude")
    normalized = vector / magnitude
    return (float(normalized[0]), float(normalized[1]), float(normalized[2]))


# Each row is (direction indices, plane indices), transcribed from the official
# DAMASK 3.1.0 Crystal kinematics tables.  Cubic indices are [uvw]/(hkl), and
# HCP indices are [uvtw]/(hkil).  Keeping family boundaries is essential because
# activation and CRSS are family-specific material assumptions.
_RAW_SYSTEMS: dict[str, tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]] = {
    FCC_111_110: (
        ((0, 1, -1), (1, 1, 1)),
        ((-1, 0, 1), (1, 1, 1)),
        ((1, -1, 0), (1, 1, 1)),
        ((0, -1, -1), (-1, -1, 1)),
        ((1, 0, 1), (-1, -1, 1)),
        ((-1, 1, 0), (-1, -1, 1)),
        ((0, -1, 1), (1, -1, -1)),
        ((-1, 0, -1), (1, -1, -1)),
        ((1, 1, 0), (1, -1, -1)),
        ((0, 1, 1), (-1, 1, -1)),
        ((1, 0, -1), (-1, 1, -1)),
        ((-1, -1, 0), (-1, 1, -1)),
    ),
    FCC_110_110: (
        ((1, 1, 0), (1, -1, 0)),
        ((1, -1, 0), (1, 1, 0)),
        ((1, 0, 1), (1, 0, -1)),
        ((1, 0, -1), (1, 0, 1)),
        ((0, 1, 1), (0, 1, -1)),
        ((0, 1, -1), (0, 1, 1)),
    ),
    BCC_110_111: (
        ((1, -1, 1), (0, 1, 1)),
        ((1, -1, 1), (1, 0, -1)),
        ((1, -1, 1), (-1, -1, 0)),
        ((-1, -1, 1), (0, -1, -1)),
        ((-1, -1, 1), (1, 0, 1)),
        ((-1, -1, 1), (-1, 1, 0)),
        ((1, 1, 1), (0, 1, -1)),
        ((1, 1, 1), (-1, 0, 1)),
        ((1, 1, 1), (1, -1, 0)),
        ((-1, 1, 1), (0, -1, 1)),
        ((-1, 1, 1), (-1, 0, -1)),
        ((-1, 1, 1), (1, 1, 0)),
    ),
    BCC_112_111: (
        ((1, -1, 1), (2, 1, -1)),
        ((1, -1, 1), (-1, 1, 2)),
        ((1, -1, 1), (1, 2, 1)),
        ((-1, -1, 1), (2, -1, 1)),
        ((-1, -1, 1), (1, 1, 2)),
        ((-1, -1, 1), (-1, 2, 1)),
        ((1, 1, 1), (1, 1, -2)),
        ((1, 1, 1), (1, -2, 1)),
        ((1, 1, 1), (-2, 1, 1)),
        ((-1, 1, 1), (1, -1, 2)),
        ((-1, 1, 1), (1, 2, -1)),
        ((-1, 1, 1), (2, 1, 1)),
    ),
    BCC_123_111: (
        ((1, -1, 1), (-1, 2, 3)),
        ((1, -1, 1), (1, 3, 2)),
        ((1, -1, 1), (-2, 1, 3)),
        ((1, -1, 1), (2, 3, 1)),
        ((1, -1, 1), (3, 1, -2)),
        ((1, -1, 1), (3, 2, -1)),
        ((-1, -1, 1), (1, 2, 3)),
        ((-1, -1, 1), (-1, 3, 2)),
        ((-1, -1, 1), (2, 1, 3)),
        ((-1, -1, 1), (-2, 3, 1)),
        ((-1, -1, 1), (3, -1, 2)),
        ((-1, -1, 1), (3, -2, 1)),
        ((1, 1, 1), (1, 2, -3)),
        ((1, 1, 1), (1, -3, 2)),
        ((1, 1, 1), (2, 1, -3)),
        ((1, 1, 1), (2, -3, 1)),
        ((1, 1, 1), (-3, 1, 2)),
        ((1, 1, 1), (-3, 2, 1)),
        ((-1, 1, 1), (1, -2, 3)),
        ((-1, 1, 1), (1, 3, -2)),
        ((-1, 1, 1), (2, -1, 3)),
        ((-1, 1, 1), (2, 3, -1)),
        ((-1, 1, 1), (3, 1, 2)),
        ((-1, 1, 1), (3, 2, 1)),
    ),
    HCP_BASAL_A: (
        ((2, -1, -1, 0), (0, 0, 0, 1)),
        ((-1, 2, -1, 0), (0, 0, 0, 1)),
        ((-1, -1, 2, 0), (0, 0, 0, 1)),
    ),
    HCP_PRISMATIC_A: (
        ((2, -1, -1, 0), (0, 1, -1, 0)),
        ((-1, 2, -1, 0), (-1, 0, 1, 0)),
        ((-1, -1, 2, 0), (1, -1, 0, 0)),
    ),
    HCP_PYRAMIDAL_A: (
        ((-1, 2, -1, 0), (1, 0, -1, 1)),
        ((-2, 1, 1, 0), (0, 1, -1, 1)),
        ((-1, -1, 2, 0), (-1, 1, 0, 1)),
        ((1, -2, 1, 0), (-1, 0, 1, 1)),
        ((2, -1, -1, 0), (0, -1, 1, 1)),
        ((1, 1, -2, 0), (1, -1, 0, 1)),
    ),
    HCP_PYRAMIDAL_CA: (
        ((-2, 1, 1, 3), (1, 0, -1, 1)),
        ((-1, -1, 2, 3), (1, 0, -1, 1)),
        ((-1, -1, 2, 3), (0, 1, -1, 1)),
        ((1, -2, 1, 3), (0, 1, -1, 1)),
        ((1, -2, 1, 3), (-1, 1, 0, 1)),
        ((2, -1, -1, 3), (-1, 1, 0, 1)),
        ((2, -1, -1, 3), (-1, 0, 1, 1)),
        ((1, 1, -2, 3), (-1, 0, 1, 1)),
        ((1, 1, -2, 3), (0, -1, 1, 1)),
        ((-1, 2, -1, 3), (0, -1, 1, 1)),
        ((-1, 2, -1, 3), (1, -1, 0, 1)),
        ((-2, 1, 1, 3), (1, -1, 0, 1)),
    ),
    HCP_PYRAMIDAL2_CA: (
        ((-1, -1, 2, 3), (1, 1, -2, 2)),
        ((1, -2, 1, 3), (-1, 2, -1, 2)),
        ((2, -1, -1, 3), (-2, 1, 1, 2)),
        ((1, 1, -2, 3), (-1, -1, 2, 2)),
        ((-1, 2, -1, 3), (1, -2, 1, 2)),
        ((-2, 1, 1, 3), (2, -1, -1, 2)),
    ),
}


def available_slip_families(crystal_structure: str) -> tuple[str, ...]:
    """Return the canonical catalog families for ``fcc``, ``bcc``, or ``hcp``."""

    structure = str(crystal_structure or "").strip().lower()
    try:
        return _STRUCTURE_FAMILIES[structure]
    except KeyError as exc:
        raise CrystalPlasticityInputError(
            "crystal_structure must be one of 'fcc', 'bcc', or 'hcp'"
        ) from exc


def _hcp_vectors(
    direction_indices: tuple[int, ...],
    plane_indices: tuple[int, ...],
    *,
    c_over_a: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    u, v, t, w = direction_indices
    h, k, i, ell = plane_indices
    if u + v + t != 0 or h + k + i != 0:
        raise CrystalPlasticityInputError("invalid internal Miller-Bravais indices")
    a1 = np.asarray([1.0, 0.0, 0.0])
    a2 = np.asarray([-0.5, math.sqrt(3.0) / 2.0, 0.0])
    a3 = -(a1 + a2)
    c = np.asarray([0.0, 0.0, c_over_a])
    direction = u * a1 + v * a2 + t * a3 + w * c

    direct_basis = np.column_stack([a1, a2, c])
    reciprocal_basis = np.linalg.inv(direct_basis).T
    normal = reciprocal_basis @ np.asarray([h, k, ell], dtype=float)
    return (
        _normalized(direction, field_name="HCP slip direction"),
        _normalized(normal, field_name="HCP plane normal"),
    )


def _finite_positive(value: Any, *, field_name: str, maximum: float | None = None) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CrystalPlasticityInputError(f"{field_name} must be a finite positive number") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise CrystalPlasticityInputError(f"{field_name} must be a finite positive number")
    if maximum is not None and number > maximum:
        raise CrystalPlasticityInputError(f"{field_name} must not exceed {maximum:g}")
    return number


def canonical_slip_systems(
    crystal_structure: str,
    *,
    families: Sequence[str] | None = None,
    c_over_a: float | None = None,
) -> tuple[SlipSystem, ...]:
    """Build canonical slip geometry without assuming which families are active.

    ``families=None`` returns the full DAMASK catalog for the crystal structure.
    Selecting an active family, its CRSS, and its temperature dependence remains
    a material-model decision.  HCP geometry requires the phase's explicit
    lattice ratio because pyramidal directions and normals depend on it.
    """

    structure = str(crystal_structure or "").strip().lower()
    available = available_slip_families(structure)
    selected = available if families is None else tuple(str(item) for item in families)
    if not selected:
        raise CrystalPlasticityInputError("at least one slip family is required")
    if len(set(selected)) != len(selected):
        raise CrystalPlasticityInputError("slip_families must not contain duplicates")
    unknown = [family for family in selected if family not in available]
    if unknown:
        raise CrystalPlasticityInputError(
            f"slip families {unknown!r} are not canonical for {structure}"
        )

    ratio: float | None = None
    if structure == "hcp":
        ratio = _finite_positive(c_over_a, field_name="c_over_a", maximum=10.0)
    elif c_over_a is not None:
        raise CrystalPlasticityInputError("c_over_a is only valid for hcp")

    systems: list[SlipSystem] = []
    for family in selected:
        for index, (direction_indices, plane_indices) in enumerate(_RAW_SYSTEMS[family], start=1):
            if structure == "hcp":
                assert ratio is not None
                direction, normal = _hcp_vectors(
                    direction_indices,
                    plane_indices,
                    c_over_a=ratio,
                )
            else:
                direction = _normalized(
                    np.asarray(direction_indices, dtype=float),
                    field_name="cubic slip direction",
                )
                normal = _normalized(
                    np.asarray(plane_indices, dtype=float),
                    field_name="cubic plane normal",
                )
            systems.append(
                SlipSystem(
                    system_id=f"{family}:{index:02d}",
                    crystal_structure=structure,
                    family=family,
                    slip_direction_crystal=direction,
                    plane_normal_crystal=normal,
                    direction_indices=direction_indices,
                    plane_indices=plane_indices,
                )
            )
    return tuple(systems)


def validate_crystal_to_sample_rotation(
    rotation: Any,
    *,
    orthogonality_tolerance: float = 1e-10,
) -> np.ndarray:
    """Validate and return one active, right-handed crystal-to-sample rotation."""

    tolerance = _finite_positive(
        orthogonality_tolerance,
        field_name="orthogonality_tolerance",
        maximum=1e-3,
    )
    try:
        matrix = np.asarray(rotation, dtype=float)
    except (TypeError, ValueError) as exc:
        raise CrystalPlasticityInputError("rotation must be a numeric 3x3 matrix") from exc
    if matrix.shape != (3, 3):
        raise CrystalPlasticityInputError("rotation must have shape (3, 3)")
    if not np.all(np.isfinite(matrix)):
        raise CrystalPlasticityInputError("rotation must contain only finite values")
    orthogonality_error = float(np.max(np.abs(matrix.T @ matrix - np.eye(3))))
    determinant = float(np.linalg.det(matrix))
    if orthogonality_error > tolerance:
        raise CrystalPlasticityInputError(
            "rotation is not orthonormal within the declared tolerance"
        )
    if abs(determinant - 1.0) > tolerance:
        raise CrystalPlasticityInputError(
            "rotation must be a proper right-handed rotation with determinant +1"
        )
    result = matrix.copy()
    _freeze_array(result)
    return cast(np.ndarray, result)


def validate_sample_frame_stress(
    stress: Any,
    *,
    symmetry_tolerance: float = 1e-10,
) -> np.ndarray:
    """Validate one finite symmetric sample-frame Cauchy-stress tensor."""

    tolerance = _finite_positive(
        symmetry_tolerance,
        field_name="symmetry_tolerance",
        maximum=1e-3,
    )
    try:
        tensor = np.asarray(stress, dtype=float)
    except (TypeError, ValueError) as exc:
        raise CrystalPlasticityInputError("stress must be a numeric 3x3 matrix") from exc
    if tensor.shape != (3, 3):
        raise CrystalPlasticityInputError("stress must have shape (3, 3)")
    if not np.all(np.isfinite(tensor)):
        raise CrystalPlasticityInputError("stress must contain only finite values")
    scale = max(float(np.max(np.abs(tensor))), 1.0)
    if float(np.max(np.abs(tensor - tensor.T))) > tolerance * scale:
        raise CrystalPlasticityInputError("sample-frame Cauchy stress must be symmetric")
    result = np.asarray((tensor + tensor.T) * 0.5, dtype=float).copy()
    _freeze_array(result)
    return cast(np.ndarray, result)


def _clean_stress_unit(value: str) -> str:
    unit = str(value or "").strip()
    if unit not in _STRESS_UNITS:
        raise CrystalPlasticityInputError(f"stress_unit must be one of {sorted(_STRESS_UNITS)!r}")
    return unit


def _validated_systems(slip_systems: Sequence[SlipSystem]) -> tuple[SlipSystem, ...]:
    systems = tuple(slip_systems)
    if not systems:
        raise CrystalPlasticityInputError("at least one slip system is required")
    if not all(isinstance(system, SlipSystem) for system in systems):
        raise CrystalPlasticityInputError("slip_systems must contain only SlipSystem values")
    if len({system.system_id for system in systems}) != len(systems):
        raise CrystalPlasticityInputError("slip system IDs must be unique")
    return systems


def _sample_vectors(
    rotation_crystal_to_sample: np.ndarray,
    systems: tuple[SlipSystem, ...],
) -> tuple[np.ndarray, np.ndarray]:
    directions_crystal = np.asarray(
        [system.slip_direction_crystal for system in systems], dtype=float
    )
    normals_crystal = np.asarray([system.plane_normal_crystal for system in systems], dtype=float)
    return (
        directions_crystal @ rotation_crystal_to_sample.T,
        normals_crystal @ rotation_crystal_to_sample.T,
    )


def resolved_shear_stresses(
    *,
    stress_sample: Any,
    rotation_crystal_to_sample: Any,
    slip_systems: Sequence[SlipSystem],
    stress_unit: str,
    reference_stress: float | None = None,
) -> ResolvedShearResult:
    """Resolve arbitrary symmetric sample stress onto each crystallographic system.

    ``reference_stress`` is optional and must use ``stress_unit``.  When present,
    ``abs(tau) / reference_stress`` is reported as a generic normalized resolved
    shear, not mislabeled as a classical Schmid factor for multiaxial loading.
    """

    unit = _clean_stress_unit(stress_unit)
    stress = validate_sample_frame_stress(stress_sample)
    rotation = validate_crystal_to_sample_rotation(rotation_crystal_to_sample)
    systems = _validated_systems(slip_systems)
    directions, normals = _sample_vectors(rotation, systems)
    resolved = np.einsum("si,ij,sj->s", directions, stress, normals)
    normalized: np.ndarray | None = None
    reference: float | None = None
    if reference_stress is not None:
        reference = _finite_positive(reference_stress, field_name="reference_stress")
        normalized = np.abs(resolved) / reference
    return ResolvedShearResult(
        system_ids=tuple(system.system_id for system in systems),
        stress_unit=unit,
        resolved_shear_stress=np.asarray(resolved, dtype=float),
        normalized_resolved_shear=normalized,
        reference_stress=reference,
    )


def uniaxial_schmid_factors(
    *,
    load_axis_sample: Any,
    rotation_crystal_to_sample: Any,
    slip_systems: Sequence[SlipSystem],
) -> np.ndarray:
    """Return classical ``|cos(phi) cos(lambda)|`` factors for uniaxial load."""

    try:
        axis = np.asarray(load_axis_sample, dtype=float)
    except (TypeError, ValueError) as exc:
        raise CrystalPlasticityInputError("load_axis_sample must be a numeric 3-vector") from exc
    if axis.shape != (3,) or not np.all(np.isfinite(axis)):
        raise CrystalPlasticityInputError("load_axis_sample must be a finite 3-vector")
    axis = np.asarray(_normalized(axis, field_name="load_axis_sample"))
    rotation = validate_crystal_to_sample_rotation(rotation_crystal_to_sample)
    systems = _validated_systems(slip_systems)
    directions, normals = _sample_vectors(rotation, systems)
    factors = np.abs((directions @ axis) * (normals @ axis))
    if np.any(factors > 0.5 + 1e-12):
        raise CrystalPlasticityInputError("computed Schmid factor violates the uniaxial 0.5 bound")
    factors = np.asarray(np.minimum(factors, 0.5), dtype=float)
    _freeze_array(factors)
    return cast(np.ndarray, factors)


def analyze_grains(
    *,
    phase_id: str,
    stresses_sample: Any,
    rotations_crystal_to_sample: Any,
    slip_systems: Sequence[SlipSystem],
    stress_unit: str,
    grain_ids: Sequence[str] | None = None,
    reference_stress: float | Sequence[float] | None = None,
) -> GrainBatchAnalysis:
    """Analyze a bounded batch with memory-capped chunked intermediates.

    The returned ``(grain, system)`` arrays are necessarily materialized, but
    rotated direction/normal intermediates are limited to
    :data:`BATCH_INTERMEDIATE_BYTES`. The separate grain-system-value cap keeps
    the result itself bounded even when many deformation families are combined.
    """

    unit = _clean_stress_unit(stress_unit)
    if not isinstance(phase_id, str):
        raise CrystalPlasticityInputError(
            "phase_id must name one phase; partition mixed-phase grains before analysis"
        )
    phase = _clean_text(phase_id, field_name="phase_id", max_chars=256)
    try:
        rotations_raw = np.asarray(rotations_crystal_to_sample, dtype=float)
    except (TypeError, ValueError) as exc:
        raise CrystalPlasticityInputError("rotations must be a numeric (N,3,3) array") from exc
    if rotations_raw.ndim != 3 or rotations_raw.shape[1:] != (3, 3):
        raise CrystalPlasticityInputError("rotations must have shape (N, 3, 3)")
    grain_count = int(rotations_raw.shape[0])
    if grain_count <= 0 or grain_count > MAX_GRAINS:
        raise CrystalPlasticityInputError(f"grain count must be in [1, {MAX_GRAINS}]")

    systems = _validated_systems(slip_systems)
    system_count = len(systems)
    if grain_count * system_count > MAX_GRAIN_SYSTEM_VALUES:
        raise CrystalPlasticityInputError(
            "grain/system result exceeds the bounded output cap: "
            f"{grain_count}*{system_count} > {MAX_GRAIN_SYSTEM_VALUES}"
        )

    try:
        stresses_raw = np.asarray(stresses_sample, dtype=float)
    except (TypeError, ValueError) as exc:
        raise CrystalPlasticityInputError("stresses must be numeric sample-frame tensors") from exc
    shared_stress: np.ndarray | None = None
    if stresses_raw.shape == (3, 3):
        shared_stress = validate_sample_frame_stress(stresses_raw)
    elif stresses_raw.shape != (grain_count, 3, 3):
        raise CrystalPlasticityInputError(
            "stresses must have shape (3, 3) or match the grain batch as (N, 3, 3)"
        )

    if grain_ids is None:
        ids = tuple(f"grain-{index}" for index in range(grain_count))
    else:
        ids = tuple(str(value or "").strip() for value in grain_ids)
        if len(ids) != grain_count:
            raise CrystalPlasticityInputError("grain_ids length must match the orientation batch")
        if any(not value for value in ids) or len(set(ids)) != len(ids):
            raise CrystalPlasticityInputError("grain_ids must be nonblank and unique")

    references: np.ndarray | None = None
    if reference_stress is not None:
        raw_reference = np.asarray(reference_stress, dtype=float)
        if raw_reference.ndim == 0:
            references = np.full(
                grain_count, _finite_positive(raw_reference.item(), field_name="reference_stress")
            )
        elif raw_reference.shape == (grain_count,):
            references = np.asarray(
                [
                    _finite_positive(item, field_name=f"reference_stress[{index}]")
                    for index, item in enumerate(raw_reference)
                ],
                dtype=float,
            )
        else:
            raise CrystalPlasticityInputError(
                "reference_stress must be scalar or have one value per grain"
            )

    directions = np.asarray([system.slip_direction_crystal for system in systems], dtype=float)
    normals = np.asarray([system.plane_normal_crystal for system in systems], dtype=float)
    resolved = np.empty((grain_count, system_count), dtype=float)
    normalized = (
        np.empty((grain_count, system_count), dtype=float) if references is not None else None
    )
    max_indices = np.empty(grain_count, dtype=np.int64)

    intermediate_bytes_per_grain = max(1, system_count * 3 * 8 * 2 + system_count * 8)
    chunk_size = max(
        1,
        min(grain_count, BATCH_INTERMEDIATE_BYTES // intermediate_bytes_per_grain),
    )
    identity = np.eye(3, dtype=float)
    for start in range(0, grain_count, chunk_size):
        stop = min(grain_count, start + chunk_size)
        rotations = rotations_raw[start:stop]
        if not np.all(np.isfinite(rotations)):
            raise CrystalPlasticityInputError("rotations must contain only finite values")
        gram = np.matmul(np.swapaxes(rotations, 1, 2), rotations)
        orthogonality_error = np.max(np.abs(gram - identity), axis=(1, 2))
        if np.any(orthogonality_error > 1e-10):
            raise CrystalPlasticityInputError(
                "rotation batch contains a matrix that is not orthonormal"
            )
        determinants = np.linalg.det(rotations)
        if np.any(np.abs(determinants - 1.0) > 1e-10):
            raise CrystalPlasticityInputError(
                "rotation batch contains a matrix without determinant +1"
            )

        if shared_stress is not None:
            stresses = np.broadcast_to(shared_stress, (stop - start, 3, 3))
        else:
            stress_chunk = stresses_raw[start:stop]
            if not np.all(np.isfinite(stress_chunk)):
                raise CrystalPlasticityInputError("stresses must contain only finite values")
            scales = np.maximum(np.max(np.abs(stress_chunk), axis=(1, 2)), 1.0)
            symmetry_error = np.max(
                np.abs(stress_chunk - np.swapaxes(stress_chunk, 1, 2)), axis=(1, 2)
            )
            if np.any(symmetry_error > 1e-10 * scales):
                raise CrystalPlasticityInputError("sample-frame Cauchy stresses must be symmetric")
            stresses = np.asarray(
                (stress_chunk + np.swapaxes(stress_chunk, 1, 2)) * 0.5,
                dtype=float,
            )

        directions_sample = np.einsum("gij,sj->gsi", rotations, directions)
        normals_sample = np.einsum("gij,sj->gsi", rotations, normals)
        chunk_resolved = np.einsum("gsi,gij,gsj->gs", directions_sample, stresses, normals_sample)
        resolved[start:stop] = chunk_resolved
        max_indices[start:stop] = np.argmax(np.abs(chunk_resolved), axis=1)
        if normalized is not None and references is not None:
            normalized[start:stop] = np.abs(chunk_resolved) / references[start:stop, np.newaxis]

    return GrainBatchAnalysis(
        phase_id=phase,
        grain_ids=ids,
        system_ids=tuple(system.system_id for system in systems),
        stress_unit=unit,
        resolved_shear_stress=resolved,
        normalized_resolved_shear=normalized,
        reference_stress=references,
        max_abs_system_index=max_indices,
    )


def _closed_mapping(
    value: Any,
    *,
    field_name: str,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CrystalPlasticityInputError(f"{field_name} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise CrystalPlasticityInputError(f"{field_name} keys must be strings")
    keys = set(value)
    missing = sorted(required - keys)
    unknown = sorted(keys - required - optional)
    if missing:
        raise CrystalPlasticityInputError(f"{field_name} is missing required fields {missing!r}")
    if unknown:
        raise CrystalPlasticityInputError(f"{field_name} has unknown fields {unknown!r}")
    return value


def _clean_text(value: Any, *, field_name: str, max_chars: int = 4096) -> str:
    text = str(value or "").strip()
    if not text:
        raise CrystalPlasticityInputError(f"{field_name} is required")
    if len(text) > max_chars or any(ord(character) < 32 for character in text):
        raise CrystalPlasticityInputError(f"{field_name} is invalid")
    return text


def _provenance(value: Any, *, field_name: str) -> SourceProvenance:
    raw = _closed_mapping(
        value,
        field_name=field_name,
        required=frozenset({"source_id", "source_type", "citation", "sha256"}),
    )
    source_type = _clean_text(raw["source_type"], field_name=f"{field_name}.source_type")
    if source_type not in _SOURCE_TYPES:
        raise CrystalPlasticityInputError(
            f"{field_name}.source_type must be one of {sorted(_SOURCE_TYPES)!r}"
        )
    digest = str(raw["sha256"] or "").strip().lower()
    if not _SHA256.fullmatch(digest):
        raise CrystalPlasticityInputError(f"{field_name}.sha256 must be a lowercase SHA-256")
    return SourceProvenance(
        source_id=_clean_text(raw["source_id"], field_name=f"{field_name}.source_id"),
        source_type=source_type,
        citation=_clean_text(raw["citation"], field_name=f"{field_name}.citation"),
        sha256=digest,
    )


def _finite_scalar(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise CrystalPlasticityInputError(f"{field_name} must be a finite scalar")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CrystalPlasticityInputError(f"{field_name} must be a finite scalar") from exc
    if not math.isfinite(number):
        raise CrystalPlasticityInputError(f"{field_name} must be a finite scalar")
    return number


def validate_cpfe_input_contract(payload: Mapping[str, Any]) -> CPFEInputContract:
    """Validate a closed, SI-unit CPFE input contract without executing a model.

    The validator establishes phase/symmetry consistency, frame conventions,
    proper orientations, selected family geometry, positive CRSS values, and
    structurally valid provenance declarations.  It does not resolve or re-hash
    the declared source bytes.  Hardening parameter names and semantics are
    deliberately engine-owned; this layer only guarantees finite scalars,
    explicit units, and declaration structure.
    """

    root = _closed_mapping(
        payload,
        field_name="contract",
        required=frozenset(
            {
                "schema_version",
                "phase",
                "frames",
                "units",
                "orientations",
                "slip_families",
                "crss",
                "hardening",
            }
        ),
    )
    if root["schema_version"] != CPFE_CONTRACT_SCHEMA_VERSION:
        raise CrystalPlasticityInputError(
            f"schema_version must be {CPFE_CONTRACT_SCHEMA_VERSION!r}"
        )

    phase = _closed_mapping(
        root["phase"],
        field_name="phase",
        required=frozenset({"phase_id", "crystal_structure", "symmetry", "provenance"}),
        optional=frozenset({"c_over_a"}),
    )
    structure = str(phase["crystal_structure"] or "").strip().lower()
    available = available_slip_families(structure)
    symmetry = str(phase["symmetry"] or "").strip()
    expected_symmetry = _STRUCTURE_SYMMETRY[structure]
    if symmetry != expected_symmetry:
        raise CrystalPlasticityInputError(
            f"phase.symmetry must be {expected_symmetry!r} for {structure}"
        )
    ratio: float | None = None
    if structure == "hcp":
        if "c_over_a" not in phase:
            raise CrystalPlasticityInputError("phase.c_over_a is required for hcp")
        ratio = _finite_positive(phase["c_over_a"], field_name="phase.c_over_a", maximum=10.0)
    elif "c_over_a" in phase:
        raise CrystalPlasticityInputError("phase.c_over_a is only valid for hcp")

    frames = _closed_mapping(
        root["frames"],
        field_name="frames",
        required=frozenset({"orientation", "stress"}),
    )
    if frames["orientation"] != "crystal_to_sample":
        raise CrystalPlasticityInputError(
            "frames.orientation must be the active 'crystal_to_sample' convention"
        )
    if frames["stress"] != "sample":
        raise CrystalPlasticityInputError("frames.stress must be 'sample'")

    units = _closed_mapping(
        root["units"],
        field_name="units",
        required=frozenset({"stress", "length", "time"}),
    )
    expected_units = {"stress": "Pa", "length": "m", "time": "s"}
    for unit_name, expected in expected_units.items():
        if units[unit_name] != expected:
            raise CrystalPlasticityInputError(f"units.{unit_name} must be {expected!r}")

    orientations_value = root["orientations"]
    if not isinstance(orientations_value, Sequence) or isinstance(
        orientations_value, (str, bytes, bytearray)
    ):
        raise CrystalPlasticityInputError("orientations must be a list of 3x3 matrices")
    if not orientations_value or len(orientations_value) > MAX_GRAINS:
        raise CrystalPlasticityInputError(f"orientation count must be in [1, {MAX_GRAINS}]")
    orientations = np.stack(
        [validate_crystal_to_sample_rotation(item) for item in orientations_value],
        axis=0,
    )

    families_value = root["slip_families"]
    if not isinstance(families_value, Sequence) or isinstance(
        families_value, (str, bytes, bytearray)
    ):
        raise CrystalPlasticityInputError("slip_families must be a list")
    families = tuple(str(item) for item in families_value)
    if not families or len(set(families)) != len(families):
        raise CrystalPlasticityInputError("slip_families must be nonempty and unique")
    wrong = [family for family in families if family not in available]
    if wrong:
        raise CrystalPlasticityInputError(
            f"slip families {wrong!r} are not canonical for {structure}"
        )
    canonical_slip_systems(structure, families=families, c_over_a=ratio)

    crss = _closed_mapping(
        root["crss"],
        field_name="crss",
        required=frozenset({"unit", "values", "provenance"}),
    )
    if crss["unit"] != "Pa":
        raise CrystalPlasticityInputError("crss.unit must be 'Pa'")
    values = _closed_mapping(
        crss["values"],
        field_name="crss.values",
        required=frozenset(families),
    )
    crss_pa = {
        family: _finite_positive(
            values[family],
            field_name=f"crss.values.{family}",
            maximum=1e14,
        )
        for family in families
    }

    hardening = _closed_mapping(
        root["hardening"],
        field_name="hardening",
        required=frozenset({"model_id", "parameters", "parameter_units", "provenance"}),
    )
    parameters_raw = hardening["parameters"]
    if not isinstance(parameters_raw, Mapping) or not parameters_raw:
        raise CrystalPlasticityInputError("hardening.parameters must be a nonempty object")
    if len(parameters_raw) > MAX_HARDENING_PARAMETERS:
        raise CrystalPlasticityInputError(
            f"hardening.parameters exceeds {MAX_HARDENING_PARAMETERS} entries"
        )
    if not all(isinstance(key, str) and key.strip() for key in parameters_raw):
        raise CrystalPlasticityInputError("hardening parameter names must be nonblank strings")
    parameter_names = frozenset(parameters_raw)
    parameter_units_raw = _closed_mapping(
        hardening["parameter_units"],
        field_name="hardening.parameter_units",
        required=parameter_names,
    )
    parameter_units: dict[str, str] = {}
    parameters: dict[str, float] = {}
    for name in sorted(parameter_names):
        parameters[name] = _finite_scalar(
            parameters_raw[name], field_name=f"hardening.parameters.{name}"
        )
        parameter_units[name] = _clean_text(
            parameter_units_raw[name],
            field_name=f"hardening.parameter_units.{name}.unit",
            max_chars=64,
        )

    return CPFEInputContract(
        schema_version=CPFE_CONTRACT_SCHEMA_VERSION,
        phase_id=_clean_text(phase["phase_id"], field_name="phase.phase_id", max_chars=128),
        crystal_structure=structure,
        symmetry=symmetry,
        c_over_a=ratio,
        phase_provenance=_provenance(phase["provenance"], field_name="phase.provenance"),
        orientations_crystal_to_sample=orientations,
        slip_families=families,
        crss_pa=MappingProxyType(crss_pa),
        crss_provenance=_provenance(crss["provenance"], field_name="crss.provenance"),
        hardening_model_id=_clean_text(
            hardening["model_id"], field_name="hardening.model_id", max_chars=128
        ),
        hardening_parameters=MappingProxyType(parameters),
        hardening_parameter_units=MappingProxyType(parameter_units),
        hardening_provenance=_provenance(
            hardening["provenance"], field_name="hardening.provenance"
        ),
    )


def execute_cpfe(contract: CPFEInputContract, /, **_: Any) -> None:
    """Fail closed: no constitutive or FE/spectral solver is bundled here."""

    if not isinstance(contract, CPFEInputContract):
        raise CrystalPlasticityInputError("execute_cpfe requires a validated CPFEInputContract")
    raise CrystalPlasticityUnsupportedError(
        "CPFE solver execution is unsupported: bind and qualify a real constitutive "
        "integrator plus finite-element or spectral solver backend first"
    )


def cross_validate_slip_systems_with_damask(
    crystal_structure: str,
    *,
    families: Sequence[str] | None = None,
    c_over_a: float | None = None,
    required_version: str = DAMASK_REFERENCE_VERSION,
    overlap_tolerance: float = 1e-12,
) -> DamaskCrossValidationResult:
    """Compare built-in Schmid tensors with an exact optional DAMASK release."""

    structure = str(crystal_structure or "").strip().lower()
    available = available_slip_families(structure)
    selected = available if families is None else tuple(str(item) for item in families)
    systems = canonical_slip_systems(structure, families=selected, c_over_a=c_over_a)
    try:
        installed_version = version("damask")
        damask = importlib.import_module("damask")
    except (PackageNotFoundError, ImportError) as exc:
        raise DamaskReferenceUnavailableError(
            "DAMASK is not installed; optional reference cross-validation cannot run"
        ) from exc
    if installed_version != required_version:
        raise DamaskReferenceUnavailableError(
            f"DAMASK {required_version} is required for reference validation; "
            f"found {installed_version}"
        )

    lattice = _DAMASK_LATTICE[structure]
    if structure == "hcp":
        ratio = _finite_positive(c_over_a, field_name="c_over_a", maximum=10.0)
        crystal = damask.Crystal(lattice=lattice, a=1.0, c=ratio)
    else:
        crystal = damask.Crystal(lattice=lattice)
    active_counts = [len(_RAW_SYSTEMS[family]) if family in selected else 0 for family in available]
    reference = np.asarray(crystal.Schmid(N_slip=active_counts), dtype=float)
    if reference.ndim != 3 or reference.shape[1:] != (3, 3):
        raise DamaskReferenceUnavailableError("DAMASK returned an unexpected Schmid-tensor shape")

    built_in = np.asarray(
        [
            np.outer(system.slip_direction_crystal, system.plane_normal_crystal)
            for system in systems
        ],
        dtype=float,
    )
    if reference.shape[0] != built_in.shape[0]:
        minimum_overlap = 0.0
        passed = False
    else:
        built_flat = built_in.reshape((built_in.shape[0], 9))
        reference_flat = reference.reshape((reference.shape[0], 9))
        built_flat /= np.linalg.norm(built_flat, axis=1, keepdims=True)
        reference_flat /= np.linalg.norm(reference_flat, axis=1, keepdims=True)
        overlaps = np.abs(built_flat @ reference_flat.T)
        minimum_overlap = float(
            min(float(np.min(np.max(overlaps, axis=1))), float(np.min(np.max(overlaps, axis=0))))
        )
        passed = minimum_overlap >= 1.0 - overlap_tolerance
    return DamaskCrossValidationResult(
        damask_version=installed_version,
        crystal_structure=structure,
        families=selected,
        system_count=len(systems),
        minimum_bidirectional_tensor_overlap=minimum_overlap,
        passed=passed,
    )
