"""Analytical and fail-closed invariants for crystal-plasticity foundations."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest
import ultra_deepagents.materials.crystal_plasticity as crystal_plasticity
from ultra_deepagents.materials.crystal_plasticity import (
    BCC_110_111,
    BCC_112_111,
    BCC_123_111,
    FCC_110_110,
    FCC_111_110,
    HCP_BASAL_A,
    HCP_PRISMATIC_A,
    HCP_PYRAMIDAL2_CA,
    HCP_PYRAMIDAL_A,
    HCP_PYRAMIDAL_CA,
    CrystalPlasticityInputError,
    CrystalPlasticityUnsupportedError,
    analyze_grains,
    canonical_slip_systems,
    cross_validate_slip_systems_with_damask,
    execute_cpfe,
    resolved_shear_stresses,
    uniaxial_schmid_factors,
    validate_cpfe_input_contract,
    validate_crystal_to_sample_rotation,
    validate_sample_frame_stress,
)

IDEAL_HCP_C_OVER_A = math.sqrt(8.0 / 3.0)


@pytest.mark.parametrize(
    ("structure", "family", "expected_count", "c_over_a"),
    [
        ("fcc", FCC_111_110, 12, None),
        ("fcc", FCC_110_110, 6, None),
        ("bcc", BCC_110_111, 12, None),
        ("bcc", BCC_112_111, 12, None),
        ("bcc", BCC_123_111, 24, None),
        ("hcp", HCP_BASAL_A, 3, IDEAL_HCP_C_OVER_A),
        ("hcp", HCP_PRISMATIC_A, 3, IDEAL_HCP_C_OVER_A),
        ("hcp", HCP_PYRAMIDAL_A, 6, IDEAL_HCP_C_OVER_A),
        ("hcp", HCP_PYRAMIDAL_CA, 12, IDEAL_HCP_C_OVER_A),
        ("hcp", HCP_PYRAMIDAL2_CA, 6, IDEAL_HCP_C_OVER_A),
    ],
)
def test_canonical_slip_families_have_expected_counts_and_geometry(
    structure: str,
    family: str,
    expected_count: int,
    c_over_a: float | None,
) -> None:
    systems = canonical_slip_systems(structure, families=[family], c_over_a=c_over_a)

    assert len(systems) == expected_count
    assert len({system.system_id for system in systems}) == expected_count
    for system in systems:
        direction = np.asarray(system.slip_direction_crystal)
        normal = np.asarray(system.plane_normal_crystal)
        assert np.linalg.norm(direction) == pytest.approx(1.0, abs=1e-12)
        assert np.linalg.norm(normal) == pytest.approx(1.0, abs=1e-12)
        assert np.dot(direction, normal) == pytest.approx(0.0, abs=1e-12)


def test_hcp_geometry_requires_explicit_lattice_ratio() -> None:
    with pytest.raises(CrystalPlasticityInputError, match="c_over_a"):
        canonical_slip_systems("hcp", families=[HCP_PYRAMIDAL_CA])


def test_fcc_001_uniaxial_schmid_maximum_is_inverse_sqrt_six() -> None:
    systems = canonical_slip_systems("fcc", families=[FCC_111_110])
    factors = uniaxial_schmid_factors(
        load_axis_sample=[0.0, 0.0, 1.0],
        rotation_crystal_to_sample=np.eye(3),
        slip_systems=systems,
    )

    assert factors.shape == (12,)
    assert np.max(factors) == pytest.approx(1.0 / math.sqrt(6.0), abs=1e-12)
    assert np.all((factors >= 0.0) & (factors <= 0.5 + 1e-12))


def test_arbitrary_symmetric_stress_uses_tensor_contraction_and_is_rotation_covariant() -> None:
    systems = canonical_slip_systems("bcc", families=[BCC_110_111])
    stress = np.asarray(
        [
            [180.0, 35.0, -12.0],
            [35.0, -40.0, 21.0],
            [-12.0, 21.0, 75.0],
        ]
    )
    theta = math.radians(37.0)
    rotation = np.asarray(
        [
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    identity_result = resolved_shear_stresses(
        stress_sample=stress,
        rotation_crystal_to_sample=np.eye(3),
        slip_systems=systems,
        stress_unit="MPa",
        reference_stress=250.0,
    )
    first = systems[0]
    expected_first = (
        np.asarray(first.slip_direction_crystal) @ stress @ np.asarray(first.plane_normal_crystal)
    )
    assert identity_result.resolved_shear_stress[0] == pytest.approx(expected_first)
    assert identity_result.normalized_resolved_shear[0] == pytest.approx(
        abs(expected_first) / 250.0
    )

    rotated_result = resolved_shear_stresses(
        stress_sample=rotation @ stress @ rotation.T,
        rotation_crystal_to_sample=rotation,
        slip_systems=systems,
        stress_unit="MPa",
    )
    assert rotated_result.resolved_shear_stress == pytest.approx(
        identity_result.resolved_shear_stress,
        abs=1e-12,
    )

    hydrostatic = resolved_shear_stresses(
        stress_sample=123.0 * np.eye(3),
        rotation_crystal_to_sample=rotation,
        slip_systems=systems,
        stress_unit="MPa",
    )
    assert hydrostatic.resolved_shear_stress == pytest.approx(np.zeros(12), abs=1e-12)


def test_batch_grain_analysis_broadcasts_stress_and_matches_scalar_results() -> None:
    systems = canonical_slip_systems("fcc", families=[FCC_111_110])
    rotation_90_z = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    orientations = np.stack([np.eye(3), rotation_90_z])
    stress = np.diag([0.0, 0.0, 100.0])

    batch = analyze_grains(
        phase_id="gamma",
        stresses_sample=stress,
        rotations_crystal_to_sample=orientations,
        slip_systems=systems,
        stress_unit="MPa",
        grain_ids=["grain-7", "grain-9"],
        reference_stress=100.0,
    )
    scalar = resolved_shear_stresses(
        stress_sample=stress,
        rotation_crystal_to_sample=np.eye(3),
        slip_systems=systems,
        stress_unit="MPa",
        reference_stress=100.0,
    )

    assert batch.phase_id == "gamma"
    assert batch.grain_ids == ("grain-7", "grain-9")
    assert batch.resolved_shear_stress.shape == (2, 12)
    assert batch.normalized_resolved_shear is not None
    assert batch.max_abs_system_index.shape == (2,)
    assert batch.resolved_shear_stress[0] == pytest.approx(scalar.resolved_shear_stress)


def test_chunked_grain_analysis_matches_the_same_unchunked_contractions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    systems = canonical_slip_systems("fcc", families=[FCC_111_110])
    angles = np.linspace(0.0, 2.0 * math.pi, 17, endpoint=False)
    orientations = np.asarray(
        [
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
            for angle in angles
        ]
    )
    stresses = np.asarray(
        [
            [[100.0 + index, 2.0, 0.0], [2.0, 50.0, 1.0], [0.0, 1.0, 75.0]]
            for index in range(len(angles))
        ]
    )
    references = np.linspace(100.0, 200.0, len(angles))

    baseline = analyze_grains(
        phase_id="gamma",
        stresses_sample=stresses,
        rotations_crystal_to_sample=orientations,
        slip_systems=systems,
        stress_unit="MPa",
        reference_stress=references,
    )
    monkeypatch.setattr(crystal_plasticity, "BATCH_INTERMEDIATE_BYTES", 1)
    chunked = analyze_grains(
        phase_id="gamma",
        stresses_sample=stresses,
        rotations_crystal_to_sample=orientations,
        slip_systems=systems,
        stress_unit="MPa",
        reference_stress=references,
    )

    np.testing.assert_allclose(
        chunked.resolved_shear_stress,
        baseline.resolved_shear_stress,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        chunked.normalized_resolved_shear,
        baseline.normalized_resolved_shear,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_array_equal(chunked.max_abs_system_index, baseline.max_abs_system_index)


def test_grain_batch_rejects_a_result_larger_than_the_memory_derived_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    systems = canonical_slip_systems("fcc", families=[FCC_111_110])
    monkeypatch.setattr(crystal_plasticity, "MAX_GRAIN_SYSTEM_VALUES", 20)

    with pytest.raises(CrystalPlasticityInputError, match="bounded output cap"):
        analyze_grains(
            phase_id="gamma",
            stresses_sample=np.eye(3),
            rotations_crystal_to_sample=np.stack([np.eye(3), np.eye(3)]),
            slip_systems=systems,
            stress_unit="MPa",
        )


def test_grain_batch_requires_one_explicit_phase_partition() -> None:
    systems = canonical_slip_systems("fcc", families=[FCC_111_110])

    with pytest.raises(CrystalPlasticityInputError, match="partition mixed-phase grains"):
        analyze_grains(
            phase_id=["gamma", "alpha"],  # type: ignore[arg-type]
            stresses_sample=np.eye(3),
            rotations_crystal_to_sample=np.stack([np.eye(3), np.eye(3)]),
            slip_systems=systems,
            stress_unit="MPa",
        )


@pytest.mark.parametrize(
    "rotation",
    [
        [[1.0, 0.0], [0.0, 1.0]],
        [[1.0, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [[math.nan, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    ],
)
def test_orientation_validation_rejects_wrong_shape_nonrotation_reflection_and_nonfinite(
    rotation: list[list[float]],
) -> None:
    with pytest.raises(CrystalPlasticityInputError):
        validate_crystal_to_sample_rotation(rotation)


def test_stress_validation_rejects_wrong_frame_shape_and_asymmetry() -> None:
    with pytest.raises(CrystalPlasticityInputError, match="shape"):
        validate_sample_frame_stress(np.zeros((6,)))
    with pytest.raises(CrystalPlasticityInputError, match="symmetric"):
        validate_sample_frame_stress([[1.0, 2.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


def _provenance(digest_character: str) -> dict[str, str]:
    return {
        "source_id": "doi:10.0000/example",
        "source_type": "publication",
        "citation": "Example et al., reviewed source",
        "sha256": digest_character * 64,
    }


def _valid_cpfe_contract() -> dict[str, object]:
    return {
        "schema_version": "1",
        "phase": {
            "phase_id": "gamma",
            "crystal_structure": "fcc",
            "symmetry": "m-3m",
            "provenance": _provenance("a"),
        },
        "frames": {
            "orientation": "crystal_to_sample",
            "stress": "sample",
        },
        "units": {"stress": "Pa", "length": "m", "time": "s"},
        "orientations": [np.eye(3).tolist()],
        "slip_families": [FCC_111_110],
        "crss": {
            "unit": "Pa",
            "values": {FCC_111_110: 45.0e6},
            "provenance": _provenance("b"),
        },
        "hardening": {
            "model_id": "voce-isotropic-v1",
            "parameters": {
                "saturation_crss": 150.0e6,
                "initial_hardening_modulus": 800.0e6,
            },
            "parameter_units": {
                "saturation_crss": "Pa",
                "initial_hardening_modulus": "Pa",
            },
            "provenance": _provenance("c"),
        },
    }


def test_cpfe_contract_validates_phase_frames_units_crss_and_provenance() -> None:
    contract = validate_cpfe_input_contract(_valid_cpfe_contract())

    assert contract.phase_id == "gamma"
    assert contract.crystal_structure == "fcc"
    assert contract.symmetry == "m-3m"
    assert contract.slip_families == (FCC_111_110,)
    assert contract.crss_pa[FCC_111_110] == pytest.approx(45.0e6)
    assert contract.orientations_crystal_to_sample.shape == (1, 3, 3)
    assert contract.execution_supported is False
    assert "constitutive" in contract.unsupported_reason.lower()


@pytest.mark.parametrize(
    ("path", "bad_value", "message"),
    [
        (("frames", "orientation"), "sample_to_crystal", "crystal_to_sample"),
        (("frames", "stress"), "crystal", "sample"),
        (("units", "stress"), "MPa", "Pa"),
        (("phase", "symmetry"), "6/mmm", "symmetry"),
        (("crss", "unit"), "psi", "Pa"),
        (("crss", "values", FCC_111_110), -1.0, "positive"),
        (("crss", "values", FCC_111_110), math.nan, "positive"),
        (
            ("hardening", "parameters", "initial_hardening_modulus"),
            math.inf,
            "finite",
        ),
        (("hardening", "parameter_units", "saturation_crss"), "", "unit"),
        (("hardening", "provenance", "sha256"), "not-a-digest", "sha256"),
    ],
)
def test_cpfe_contract_fails_closed_on_adversarial_convention_inputs(
    path: tuple[str, ...],
    bad_value: object,
    message: str,
) -> None:
    payload = copy.deepcopy(_valid_cpfe_contract())
    target = payload
    for key in path[:-1]:
        target = target[key]  # type: ignore[index,assignment]
    target[path[-1]] = bad_value  # type: ignore[index]

    with pytest.raises(CrystalPlasticityInputError, match=message):
        validate_cpfe_input_contract(payload)


def test_cpfe_contract_rejects_family_from_another_crystal_structure() -> None:
    payload = _valid_cpfe_contract()
    payload["slip_families"] = [BCC_110_111]
    payload["crss"]["values"] = {BCC_110_111: 45.0e6}  # type: ignore[index]
    with pytest.raises(CrystalPlasticityInputError, match="fcc"):
        validate_cpfe_input_contract(payload)


def test_cpfe_contract_rejects_missing_provenance_and_unknown_fields() -> None:
    missing = _valid_cpfe_contract()
    del missing["crss"]["provenance"]  # type: ignore[index]
    with pytest.raises(CrystalPlasticityInputError, match="provenance"):
        validate_cpfe_input_contract(missing)

    unknown = _valid_cpfe_contract()
    unknown["frames"]["handedness"] = "right"  # type: ignore[index]
    with pytest.raises(CrystalPlasticityInputError, match="unknown"):
        validate_cpfe_input_contract(unknown)


def test_cpfe_execution_is_explicitly_unsupported_without_a_solver_backend() -> None:
    contract = validate_cpfe_input_contract(_valid_cpfe_contract())
    with pytest.raises(CrystalPlasticityUnsupportedError, match="solver"):
        execute_cpfe(contract)


@pytest.mark.parametrize(
    ("structure", "family", "c_over_a"),
    [
        ("fcc", FCC_111_110, None),
        ("fcc", FCC_110_110, None),
        ("bcc", BCC_110_111, None),
        ("bcc", BCC_112_111, None),
        ("bcc", BCC_123_111, None),
        ("hcp", HCP_BASAL_A, IDEAL_HCP_C_OVER_A),
        ("hcp", HCP_PRISMATIC_A, IDEAL_HCP_C_OVER_A),
        ("hcp", HCP_PYRAMIDAL_A, IDEAL_HCP_C_OVER_A),
        ("hcp", HCP_PYRAMIDAL_CA, IDEAL_HCP_C_OVER_A),
        ("hcp", HCP_PYRAMIDAL2_CA, IDEAL_HCP_C_OVER_A),
    ],
)
def test_each_builtin_slip_family_matches_optional_damask_reference(
    structure: str,
    family: str,
    c_over_a: float | None,
) -> None:
    pytest.importorskip("damask")
    result = cross_validate_slip_systems_with_damask(
        structure,
        families=[family],
        c_over_a=c_over_a,
        required_version="3.1.0",
    )
    assert result.passed
    assert result.system_count > 0
    assert result.minimum_bidirectional_tensor_overlap >= 1.0 - 1e-12
