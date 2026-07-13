"""Materials domain-correctness invariants.

Each check fails for the hand-rolled shortcut the computational-materials skill
warns against. Lean tests skip absent scientific libraries. The promotion target
sets ``ULTRA_FAIL_ON_DOMAIN_SKIP=1`` so a missing dependency fails closed.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import os
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pytest


def _require(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        if os.getenv("ULTRA_FAIL_ON_DOMAIN_SKIP") == "1":
            pytest.fail(f"required promotion dependency {module_name!r} is missing: {exc}")
        pytest.skip(f"scientific dependency {module_name!r} is not installed")


def _close(rgb, target, tol=0.15):
    return all(abs(a - b) <= tol for a, b in zip(rgb, target))


def _versions(*distribution_names: str) -> dict[str, str]:
    return {name: version(name) for name in distribution_names}


def _embedded_calphad_root() -> Path:
    configured = os.getenv("ULTRA_CALPHAD_DATABASE_ROOT")
    if configured:
        return Path(configured)
    installed = Path("/opt/ultra-calphad")
    if installed.is_dir():
        return installed
    return Path(__file__).resolve().parents[2] / "materials_data" / "calphad"


def test_ipf_cubic_key_corners_are_rgb(materials_evidence):
    """Cubic IPF-Z key: 001->red, 101->green, 111->blue.

    The naive 'sort xyz then R=x,G=y,B=z' mapping paints the triangle blue/cyan
    and FAILS this — which is exactly the bug this guards against.
    """
    _require("orix")
    from orix.plot import IPFColorKeyTSL
    from orix.quaternion import Orientation, symmetry
    from orix.vector import Vector3d

    ckey = IPFColorKeyTSL(symmetry.Oh, direction=Vector3d.zvector())
    # Orientations whose sample-Z (ND) sits at each triangle corner:
    #   Cube (0,0,0) -> ND=[001];  Goss (0,45,0) -> ND=[101];  (30,54.7,45) -> ND=[111]
    euler = np.array([[0.0, 0.0, 0.0], [0.0, 45.0, 0.0], [30.0, 54.7, 45.0]])
    ori = Orientation.from_euler(euler, symmetry=symmetry.Oh, degrees=True)
    rgb = np.asarray(ckey.orientation2color(ori))

    expected_colors = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    corner_errors = np.max(np.abs(rgb - expected_colors), axis=1)
    passed = bool(np.all(corner_errors <= 0.15))
    materials_evidence(
        validator_id="materials.ebsd.ipf_cubic_tsl_corners.v1",
        observed={
            "rgb_001": rgb[0].tolist(),
            "rgb_101": rgb[1].tolist(),
            "rgb_111": rgb[2].tolist(),
            "max_channel_error_by_corner": corner_errors.tolist(),
        },
        expected={
            "rgb_001": [1.0, 0.0, 0.0],
            "rgb_101": [0.0, 1.0, 0.0],
            "rgb_111": [0.0, 0.0, 1.0],
            "max_absolute_channel_error": 0.15,
        },
        tolerance_rationale=(
            "The TSL cubic IPF key fixes the three fundamental-sector corners; "
            "0.15 allows rendering interpolation without accepting a channel swap."
        ),
        units="normalized RGB channel",
        convention="TSL/EDAX cubic m-3m IPF-Z: 001=red, 101=green, 111=blue",
        library_versions=_versions("orix", "numpy"),
        passed=passed,
    )

    assert _close(rgb[0], (1, 0, 0)), f"001/Cube should be RED, got {tuple(rgb[0])}"
    assert _close(rgb[1], (0, 1, 0)), f"101/Goss should be GREEN, got {tuple(rgb[1])}"
    assert _close(rgb[2], (0, 0, 1)), f"111 should be BLUE, got {tuple(rgb[2])}"


def test_ipf_not_blue_everywhere(materials_evidence):
    """Guard the specific failure signature: the whole key is NOT blue-dominant.

    The hand-rolled mapping makes every direction blue-dominant (B is the largest
    channel almost everywhere). The correct key spans red+green+blue.
    """
    _require("orix")
    from orix.plot import IPFColorKeyTSL
    from orix.quaternion import Orientation, symmetry
    from orix.vector import Vector3d

    ckey = IPFColorKeyTSL(symmetry.Oh, direction=Vector3d.zvector())
    rng = np.random.default_rng(0)
    ori = Orientation.from_euler(rng.uniform(0, 360, (200, 3)), symmetry=symmetry.Oh, degrees=True)
    rgb = np.asarray(ckey.orientation2color(ori))
    dominant = np.argmax(rgb, axis=1)  # 0=R,1=G,2=B
    frac_blue = float(np.mean(dominant == 2))
    dominant_fractions = {
        "red": float(np.mean(dominant == 0)),
        "green": float(np.mean(dominant == 1)),
        "blue": frac_blue,
    }
    passed = frac_blue < 0.6 and all(value > 0 for value in dominant_fractions.values())
    materials_evidence(
        validator_id="materials.ebsd.ipf_cubic_color_coverage.v1",
        observed={
            "seed": 0,
            "sample_size": 200,
            "dominant_channel_fractions": dominant_fractions,
        },
        expected={
            "blue_dominant_fraction_max": 0.6,
            "each_primary_channel_observed": True,
        },
        tolerance_rationale=(
            "A correct cubic TSL key spans all three primaries; the broad 0.60 blue cap "
            "detects the known hand-rolled all-blue mapping without treating this "
            "nonuniform Euler sample as a texture measurement."
        ),
        units="fraction of sampled orientations",
        convention="Seeded Bunge Euler angles colored with the cubic m-3m TSL IPF-Z key",
        library_versions=_versions("orix", "numpy"),
        passed=passed,
    )
    # A correct key over random orientations is nowhere near all-blue.
    assert frac_blue < 0.6, f"IPF looks blue-dominant ({frac_blue:.0%}) — likely hand-rolled"
    assert all(value > 0 for value in dominant_fractions.values()), dominant_fractions


def test_random_misorientation_matches_mackenzie(materials_evidence):
    """Random cubic misorientations obey the Mackenzie distribution.

    Disorientation angle is capped at ~62.8° for cubic (m-3m) and the
    distribution has stable reference quantiles and CDF values. A hand-rolled
    misorientation that forgets symmetry reduction runs to 180° and fails these
    distributional checks.
    """
    _require("orix")
    from orix.quaternion import Misorientation, Orientation, symmetry

    pg = symmetry.Oh
    seed = 1729
    sample_size = 20_000
    random_state = np.random.get_state()
    try:
        # orix 0.14 draws through NumPy's legacy global RNG, so explicitly seed and
        # restore that state until orix exposes a Generator parameter.
        np.random.seed(seed)
        a = Orientation.random(sample_size)
        a.symmetry = pg
        b = Orientation.random(sample_size)
        b.symmetry = pg
    finally:
        np.random.set_state(random_state)
    mis = Misorientation(~a * b)
    mis.symmetry = (pg, pg)
    ang = np.rad2deg(mis.reduce().angle)

    quantile_probabilities = np.asarray([0.05, 0.25, 0.50, 0.75, 0.95])
    quantiles = np.quantile(ang, quantile_probabilities)
    quantile_ranges = np.asarray(
        [
            [18.5, 20.5],
            [32.5, 34.5],
            [41.5, 43.2],
            [48.5, 50.0],
            [56.2, 57.8],
        ]
    )
    cdf_angles = (30.0, 45.0, 55.0)
    empirical_cdf = [float(np.mean(ang <= threshold)) for threshold in cdf_angles]
    cdf_ranges = ((0.16, 0.20), (0.58, 0.62), (0.895, 0.925))
    maximum = float(np.max(ang))
    quantiles_pass = bool(
        np.all(quantiles >= quantile_ranges[:, 0]) and np.all(quantiles <= quantile_ranges[:, 1])
    )
    cdf_pass = all(
        lower <= observed <= upper
        for observed, (lower, upper) in zip(empirical_cdf, cdf_ranges, strict=True)
    )
    passed = maximum < 62.9 and quantiles_pass and cdf_pass
    materials_evidence(
        validator_id="materials.ebsd.mackenzie_cubic_distribution.v1",
        observed={
            "seed": seed,
            "sample_size": sample_size,
            "maximum_disorientation_deg": maximum,
            "quantiles_deg": {
                f"q{int(probability * 100):02d}": float(value)
                for probability, value in zip(quantile_probabilities, quantiles, strict=True)
            },
            "empirical_cdf": {
                f"le_{int(threshold)}_deg": value
                for threshold, value in zip(cdf_angles, empirical_cdf, strict=True)
            },
        },
        expected={
            "maximum_disorientation_deg_exclusive": 62.9,
            "quantile_ranges_deg": {
                f"q{int(probability * 100):02d}": bounds.tolist()
                for probability, bounds in zip(quantile_probabilities, quantile_ranges, strict=True)
            },
            "empirical_cdf_ranges": {
                f"le_{int(threshold)}_deg": list(bounds)
                for threshold, bounds in zip(cdf_angles, cdf_ranges, strict=True)
            },
        },
        tolerance_rationale=(
            "Ranges bracket the Mackenzie random-cubic reference distribution with "
            "margin wider than the binomial/quantile sampling uncertainty for 20,000 "
            "uniform orientation pairs, while rejecting unreduced or biased angles."
        ),
        units="degree disorientation and empirical probability",
        convention="Mackenzie distribution for independent uniform cubic m-3m orientations",
        library_versions=_versions("orix", "numpy"),
        passed=passed,
    )
    assert maximum < 62.9, f"cubic disorientation must cap at 62.8°, got {maximum:.2f}"
    assert quantiles_pass, {"quantiles": quantiles, "ranges": quantile_ranges}
    assert cdf_pass, {"empirical_cdf": empirical_cdf, "ranges": cdf_ranges}


def _fcc_structure(*, species: str = "Ni", lattice_constant: float = 3.52):
    from pymatgen.core import Lattice, Structure

    return Structure(
        Lattice.cubic(lattice_constant),
        [species] * 4,
        [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]],
    )


def test_ordering_changes_l12_space_group_across_declared_tolerances(materials_evidence):
    """Ordered Ni3Al L1_2 is Pm-3m (221); elemental FCC is Fm-3m (225)."""

    _require("pymatgen")
    _require("spglib")
    from pymatgen.core import Lattice, Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    lattice = Lattice.cubic(3.57)
    sites = [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]
    ordered = Structure(lattice, ["Al", "Ni", "Ni", "Ni"], sites)
    elemental_fcc = Structure(lattice, ["Ni"] * 4, sites)

    symprec_values = (0.001, 0.01, 0.1)
    ordered_numbers: list[int] = []
    fcc_numbers: list[int] = []
    for symprec in symprec_values:
        ordered_number = SpacegroupAnalyzer(
            ordered, symprec=symprec, angle_tolerance=5
        ).get_space_group_number()
        fcc_number = SpacegroupAnalyzer(
            elemental_fcc, symprec=symprec, angle_tolerance=5
        ).get_space_group_number()
        ordered_numbers.append(int(ordered_number))
        fcc_numbers.append(int(fcc_number))

    passed = ordered_numbers == [221, 221, 221] and fcc_numbers == [225, 225, 225]
    materials_evidence(
        validator_id="materials.structure.ordering_sensitive_space_group.v1",
        observed={
            "symprec_angstrom": list(symprec_values),
            "angle_tolerance_deg": 5.0,
            "ordered_ni3al_space_group_numbers": ordered_numbers,
            "elemental_fcc_ni_space_group_numbers": fcc_numbers,
        },
        expected={
            "ordered_ni3al_space_group_number": 221,
            "elemental_fcc_ni_space_group_number": 225,
            "stable_across_all_declared_tolerances": True,
        },
        tolerance_rationale=(
            "The exact ideal structures must retain their composition-sensitive space "
            "groups across a 0.001–0.1 Å regression sweep; this is an API/control range, "
            "not a universal tolerance for measured structures."
        ),
        units="space-group number; symprec in angstrom; angle tolerance in degree",
        convention="Conventional cubic cells, International Tables space-group numbers",
        library_versions=_versions("pymatgen", "spglib", "numpy"),
        passed=passed,
    )
    for symprec, ordered_number, fcc_number in zip(
        symprec_values, ordered_numbers, fcc_numbers, strict=True
    ):
        assert ordered_number == 221, (symprec, ordered_number)
        assert fcc_number == 225, (symprec, fcc_number)


def test_fcc_ni_cuka_xrd_first_peak_is_111_near_44_6_degrees(materials_evidence):
    """Guard radiation, degree/radian, and forbidden-reflection mistakes."""

    _require("pymatgen")
    from pymatgen.analysis.diffraction.xrd import XRDCalculator

    calculator = XRDCalculator(wavelength="CuKa")
    pattern = calculator.get_pattern(_fcc_structure(), two_theta_range=(10, 100))
    first_peak = float(pattern.x[0])
    first_hkls = {tuple(entry["hkl"]) for entry in pattern.hkls[0]}
    all_hkls = {
        tuple(entry["hkl"])
        for equivalent_reflections in pattern.hkls
        for entry in equivalent_reflections
    }

    forbidden_present = sorted({hkl for hkl in ((1, 0, 0), (1, 1, 0)) if hkl in all_hkls})
    passed = abs(first_peak - 44.59) < 0.35 and (1, 1, 1) in first_hkls and not forbidden_present
    materials_evidence(
        validator_id="materials.xrd.fcc_ni_cuka_peak_and_extinctions.v1",
        observed={
            "first_two_theta_deg": first_peak,
            "first_peak_hkls": [list(hkl) for hkl in sorted(first_hkls)],
            "forbidden_hkls_present": [list(hkl) for hkl in forbidden_present],
            "wavelength_angstrom": float(calculator.wavelength),
        },
        expected={
            "first_two_theta_deg_center": 44.59,
            "first_two_theta_absolute_tolerance_deg": 0.35,
            "first_peak_contains_hkl": [1, 1, 1],
            "forbidden_hkls_absent": [[1, 0, 0], [1, 1, 0]],
        },
        tolerance_rationale=(
            "The 0.35° window covers stated lattice/wavelength rounding but not a "
            "radiation or degree/radian error; FCC systematic absences are exact."
        ),
        units="degree 2theta and angstrom wavelength",
        convention="Ideal kinematic powder XRD, conventional FCC Ni, pymatgen CuKa average",
        library_versions=_versions("pymatgen", "numpy"),
        passed=passed,
    )

    assert abs(first_peak - 44.59) < 0.35, first_peak
    assert (1, 1, 1) in first_hkls, first_hkls
    assert (1, 0, 0) not in all_hkls
    assert (1, 1, 0) not in all_hkls


def test_magpie_ni3al_features_are_finite_and_schema_stable(materials_evidence):
    _require("matminer")
    from matminer.featurizers.composition import ElementProperty
    from pymatgen.core import Composition

    featurizer = ElementProperty.from_preset("magpie")
    labels = featurizer.feature_labels()
    values = np.asarray(featurizer.featurize(Composition("Ni3Al")), dtype=float)

    statistics_present = {
        statistic: any(statistic in label.lower() for label in labels)
        for statistic in ("avg_dev", "range", "mode")
    }
    finite_count = int(np.isfinite(values).sum())
    passed = (
        len(labels) == values.size == 132
        and finite_count == values.size
        and all(statistics_present.values())
    )
    materials_evidence(
        validator_id="materials.informatics.magpie_ni3al_schema.v1",
        observed={
            "feature_count": int(values.size),
            "label_count": len(labels),
            "finite_feature_count": finite_count,
            "required_statistics_present": statistics_present,
        },
        expected={
            "feature_count": 132,
            "label_count": 132,
            "all_features_finite": True,
            "required_statistics": ["avg_dev", "range", "mode"],
        },
        tolerance_rationale=(
            "This pinned Magpie preset has a 132-column schema; every descriptor must "
            "be finite and retain the named distribution statistics."
        ),
        units="heterogeneous named Magpie descriptor units",
        convention="matminer ElementProperty.from_preset('magpie') for Composition('Ni3Al')",
        library_versions=_versions("matminer", "pymatgen", "numpy"),
        passed=passed,
    )

    assert len(labels) == values.size == 132
    assert np.isfinite(values).all()
    for statistic in ("avg_dev", "range", "mode"):
        assert any(statistic in label.lower() for label in labels), statistic


def test_anisotropic_voxel_stereology_returns_physical_volume(materials_evidence):
    """Voxel counts without spacing are not valid physical volumes."""

    _require("skimage")
    from skimage.measure import label, regionprops_table

    spacing = np.asarray([1.0, 0.5, 0.5])
    semiaxes = np.asarray([20.0, 15.0, 10.0])
    shape = (51, 81, 61)
    coords = np.indices(shape, dtype=float)
    center = (np.asarray(shape) - 1)[:, None, None, None] / 2
    physical = (coords - center) * spacing[:, None, None, None]
    ellipsoid = np.sum((physical / semiaxes[:, None, None, None]) ** 2, axis=0) <= 1

    props = regionprops_table(label(ellipsoid), spacing=tuple(spacing), properties=("area",))
    measured_volume = float(props["area"][0])
    analytic_volume = float(4 * math.pi * np.prod(semiaxes) / 3)
    relative_error = abs(measured_volume - analytic_volume) / analytic_volume
    passed = relative_error < 0.03
    materials_evidence(
        validator_id="materials.microstructure.anisotropic_stereology_volume.v1",
        observed={
            "spacing_zyx": spacing.tolist(),
            "semiaxes_zyx": semiaxes.tolist(),
            "measured_volume": measured_volume,
            "analytic_volume": analytic_volume,
            "relative_error": relative_error,
        },
        expected={"relative_error_exclusive_max": 0.03},
        tolerance_rationale=(
            "A 3% bound accommodates voxelized ellipsoid surface error at this grid "
            "resolution while rejecting raw-voxel-count or isotropic-spacing mistakes."
        ),
        units="arbitrary physical length cubed; spacing and semiaxes in same length unit",
        convention="NumPy z-y-x array order with scikit-image spacing=(dz,dy,dx)",
        library_versions=_versions("scikit-image", "numpy"),
        passed=passed,
    )
    assert relative_error < 0.03, relative_error


def test_porespy_true_phase_is_measured_as_void_and_local_radius(materials_evidence):
    """PoreSpy convention: True is pore/void for these metrics and filters."""

    _require("porespy")
    import porespy as ps

    shape = (41, 41, 41)
    center = (np.asarray(shape) - 1)[:, None, None, None] / 2
    coords = np.indices(shape, dtype=float)
    radius = 8.0
    void = np.sum((coords - center) ** 2, axis=0) <= radius**2
    porosity = float(ps.metrics.porosity(void))
    local_thickness = np.asarray(ps.filters.local_thickness(void))

    maximum_local_radius = float(np.nanmax(local_thickness))
    radius_error = abs(maximum_local_radius - radius)
    passed = 0 < porosity < 0.5 and radius_error < 1.0
    materials_evidence(
        validator_id="materials.porosity.porespy_true_void_local_radius.v1",
        observed={
            "porosity": porosity,
            "sphere_radius_voxel": radius,
            "maximum_local_thickness_value": maximum_local_radius,
            "absolute_radius_error_voxel": radius_error,
        },
        expected={
            "porosity_range_exclusive": [0.0, 0.5],
            "maximum_local_thickness_absolute_error_voxel_exclusive": 1.0,
        },
        tolerance_rationale=(
            "The centered digital sphere is a minority phase and its local-thickness "
            "maximum should recover the constructed radius within one voxel."
        ),
        units="void fraction and voxel radius",
        convention="PoreSpy boolean image with True=pore/void",
        library_versions=_versions("porespy", "numpy"),
        passed=passed,
    )

    assert 0 < porosity < 0.5, porosity
    assert radius_error < 1.0


def test_ase_emt_copper_eos_has_plausible_grid_minimum(materials_evidence):
    """A unit/convention smoke test, not a claim of ab-initio Cu accuracy."""

    _require("ase")
    from ase.build import bulk
    from ase.calculators.emt import EMT

    lattice_constants = np.linspace(3.45, 3.75, 7)
    energies: list[float] = []
    for lattice_constant in lattice_constants:
        atoms = bulk("Cu", "fcc", a=float(lattice_constant), cubic=True)
        atoms.calc = EMT()
        energies.append(float(atoms.get_potential_energy() / len(atoms)))

    best = float(lattice_constants[int(np.argmin(energies))])
    passed = 3.5 <= best <= 3.7 and bool(np.isfinite(energies).all())
    materials_evidence(
        validator_id="materials.atomistics.ase_emt_cu_eos_smoke.v1",
        observed={
            "lattice_constants_angstrom": lattice_constants.tolist(),
            "energies_ev_per_atom": energies,
            "grid_minimum_lattice_constant_angstrom": best,
            "all_energies_finite": bool(np.isfinite(energies).all()),
        },
        expected={"grid_minimum_lattice_constant_angstrom_inclusive": [3.5, 3.7]},
        tolerance_rationale=(
            "The coarse 0.05 Å EMT grid need only place its Cu minimum in a broad "
            "plausible region; this is a units/API smoke test, not ab-initio accuracy."
        ),
        units="angstrom and electronvolt per atom",
        convention="ASE conventional cubic FCC Cu with the explicitly toy EMT calculator",
        library_versions=_versions("ase", "numpy"),
        passed=passed,
    )
    assert 3.5 <= best <= 3.7, best
    assert np.isfinite(energies).all(), energies


def test_point_defect_generators_preserve_stoichiometric_deltas(materials_evidence):
    """Validate defect generator, multiplicity, and supercell-count semantics."""

    _require("pymatgen.analysis.defects")
    from pymatgen.analysis.defects.generators import (
        AntiSiteGenerator,
        SubstitutionGenerator,
        VacancyGenerator,
        VoronoiInterstitialGenerator,
    )
    from pymatgen.core import Lattice, Structure

    bulk = Structure.from_spacegroup(
        "Fm-3m",
        Lattice.cubic(5.64),
        ["Na", "Cl"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    vacancies = list(VacancyGenerator(symprec=0.01).generate(bulk))
    substitutions = list(SubstitutionGenerator().generate(bulk, {"Na": "K"}))
    antisites = list(AntiSiteGenerator().generate(bulk))
    interstitials = list(VoronoiInterstitialGenerator(min_dist=1.0).generate(bulk, {"Li"}))

    vacancy_names = sorted(defect.name for defect in vacancies)
    vacancy_multiplicities = sorted({int(defect.multiplicity) for defect in vacancies})
    vacancy_site_counts = [len(defect.defect_structure) for defect in vacancies]
    vacancy_supercell_counts = [
        len(defect.get_supercell_structure(sc_mat=np.diag([2, 2, 2]))) for defect in vacancies
    ]
    substitution_names = [defect.name for defect in substitutions]
    substitution = substitutions[0] if len(substitutions) == 1 else None
    substitution_multiplicity = int(substitution.multiplicity) if substitution is not None else None
    substitution_site_count = (
        len(substitution.defect_structure) if substitution is not None else None
    )
    antisite_names = sorted(defect.name for defect in antisites)
    interstitial_site_counts = [len(defect.defect_structure) for defect in interstitials]
    passed = (
        vacancy_names == ["v_Cl", "v_Na"]
        and vacancy_multiplicities == [4]
        and vacancy_site_counts == [len(bulk) - 1] * 2
        and vacancy_supercell_counts == [len(bulk) * 8 - 1] * 2
        and substitution_names == ["K_Na"]
        and substitution_multiplicity == 4
        and substitution_site_count == len(bulk)
        and antisite_names == ["Cl_Na", "Na_Cl"]
        and bool(interstitials)
        and all(count == len(bulk) + 1 for count in interstitial_site_counts)
    )
    materials_evidence(
        validator_id="materials.defects.nacl_generator_stoichiometry.v1",
        observed={
            "bulk_site_count": len(bulk),
            "vacancy_names": vacancy_names,
            "vacancy_multiplicities": vacancy_multiplicities,
            "vacancy_defect_site_counts": vacancy_site_counts,
            "vacancy_2x2x2_supercell_site_counts": vacancy_supercell_counts,
            "substitution_names": substitution_names,
            "substitution_multiplicity": substitution_multiplicity,
            "substitution_site_count": substitution_site_count,
            "antisite_names": antisite_names,
            "interstitial_candidate_count": len(interstitials),
            "interstitial_site_counts": interstitial_site_counts,
        },
        expected={
            "vacancy_names": ["v_Cl", "v_Na"],
            "vacancy_multiplicity": 4,
            "vacancy_unit_cell_site_count": 7,
            "vacancy_2x2x2_supercell_site_count": 63,
            "substitution_name": "K_Na",
            "substitution_multiplicity": 4,
            "substitution_site_count": 8,
            "antisite_names": ["Cl_Na", "Na_Cl"],
            "interstitial_candidate_count_min": 1,
            "interstitial_site_count": 9,
        },
        tolerance_rationale=(
            "Site-count and multiplicity relations are exact for the ideal conventional "
            "NaCl control; only the number of symmetry-distinct Voronoi candidates is "
            "bounded below because generator details may add valid candidates."
        ),
        units="site count and crystallographic multiplicity",
        convention="Conventional Fm-3m Na4Cl4 cell; 2x2x2 diagonal supercell",
        library_versions=_versions("pymatgen", "pymatgen-analysis-defects", "spglib", "numpy"),
        passed=passed,
    )
    assert {defect.name for defect in vacancies} == {"v_Na", "v_Cl"}
    assert {defect.multiplicity for defect in vacancies} == {4}
    assert all(len(defect.defect_structure) == len(bulk) - 1 for defect in vacancies)
    assert all(
        len(defect.get_supercell_structure(sc_mat=np.diag([2, 2, 2]))) == len(bulk) * 8 - 1
        for defect in vacancies
    )
    assert len(substitutions) == 1
    assert substitutions[0].name == "K_Na"
    assert substitutions[0].multiplicity == 4
    assert len(substitutions[0].defect_structure) == len(bulk)
    assert {defect.name for defect in antisites} == {"Cl_Na", "Na_Cl"}
    assert interstitials
    assert all(len(defect.defect_structure) == len(bulk) + 1 for defect in interstitials)


def test_dream3d_prefers_complete_geometry_and_preserves_real_zero_grain(
    tmp_path, materials_evidence
):
    """Exercise the production DREAM.3D selector, sentinel, and phase provenance."""

    h5py = _require("h5py")
    from ultra_deepagents.imaging.hdf5 import materials_dashboard

    path = tmp_path / "promotion-layout.dream3d"
    with h5py.File(path, "w") as handle:
        containers = handle.create_group("DataContainers")
        stats = containers.create_group("StatsGeneratorDataContainer")
        stats.create_group("_SIMPL_GEOMETRY")
        stats_ensemble = stats.create_group("CellEnsembleData")
        stats_ensemble.create_dataset(
            "PhaseName", data=np.array([b"Unknown Phase", b"Gamma"], dtype="S16")
        )

        synthetic = containers.create_group("SyntheticVolumeDataContainer")
        geometry = synthetic.create_group("_SIMPL_GEOMETRY")
        geometry.create_dataset("DIMENSIONS", data=np.array([6, 5, 4], dtype="i8"))
        geometry.create_dataset("SPACING", data=np.array([0.5, 0.5, 1.0], dtype="f4"))
        geometry.create_dataset("ORIGIN", data=np.array([1.0, 2.0, 3.0], dtype="f4"))
        cell = synthetic.create_group("CellData")
        feature_ids = np.resize(np.array([0, 1, 2, 3], dtype="i4"), 4 * 5 * 6).reshape(4, 5, 6, 1)
        cell.create_dataset("FeatureIds", data=feature_ids)
        grain = synthetic.create_group("Grain Data")
        # Row zero is reserved; row one is a real zero and must remain represented.
        grain.create_dataset("Volumes", data=np.array([[0.0], [0.0], [1.0], [2.0]], dtype="f4"))
        grain.create_dataset(
            "EulerAngles",
            data=np.array(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.1, 0.2, 0.3], [0.2, 0.3, 0.4]],
                dtype="f4",
            ),
        )
        ensemble = synthetic.create_group("CellEnsembleData")
        ensemble.create_dataset(
            "PhaseName", data=np.array([b"Unknown Phase", b"Gamma"], dtype="S16")
        )

    dashboard = materials_dashboard(str(path))
    overview = dashboard["overview"]
    volume_chart = next(
        chart
        for chart in dashboard["grain_charts"]
        if chart["source_paths"][0].endswith("/Grain Data/Volumes")
    )
    occupied_labels = [float(row["label"]) for row in volume_chart["data"] if int(row["count"]) > 0]
    selected_geometry = overview["geometry"]
    geometry_complete = all(
        selected_geometry.get(field) is not None for field in ("dimensions", "spacing", "origin")
    )
    cell_data_consistent = list(reversed(feature_ids.shape[:3])) == selected_geometry["dimensions"]
    observed_feature_ids = sorted(int(value) for value in np.unique(feature_ids))
    reserved_feature_zero_established = (
        observed_feature_ids == [0, 1, 2, 3] and overview["grain_count"] == 3
    )
    histogram_count = sum(row["count"] for row in volume_chart["data"])
    real_zero_preserved = min(occupied_labels) < 0.2
    phase_provenance = str(overview["phase_names_provenance"])
    phase_identification_algorithm_run = (
        "no phase-identification algorithm" not in phase_provenance.lower()
    )
    passed = (
        selected_geometry["path"].endswith("/SyntheticVolumeDataContainer/_SIMPL_GEOMETRY")
        and geometry_complete
        and selected_geometry["dimensions"] == [6, 5, 4]
        and selected_geometry["spacing"] == [0.5, 0.5, 1.0]
        and cell_data_consistent
        and reserved_feature_zero_established
        and overview["phase_names"] == ["Gamma"]
        and overview["phase_names_source"] == "stored_metadata"
        and not phase_identification_algorithm_run
        and histogram_count == 3
        and real_zero_preserved
    )
    materials_evidence(
        validator_id="materials.dream3d.geometry_feature_sentinel.v1",
        observed={
            "selected_geometry_path": selected_geometry["path"],
            "dimensions_xyz": selected_geometry["dimensions"],
            "spacing_xyz": selected_geometry["spacing"],
            "origin_xyz": selected_geometry["origin"],
            "geometry_complete": geometry_complete,
            "cell_data_shape_zyxc": list(feature_ids.shape),
            "cell_data_consistent": cell_data_consistent,
            "feature_ids_observed": observed_feature_ids,
            "reserved_feature_zero_established": reserved_feature_zero_established,
            "grain_count": overview["grain_count"],
            "phase_names": overview["phase_names"],
            "phase_names_source": overview["phase_names_source"],
            "phase_names_provenance": phase_provenance,
            "phase_identification_algorithm_run": phase_identification_algorithm_run,
            "grain_histogram_sample_count": histogram_count,
            "real_zero_measurement_preserved": real_zero_preserved,
        },
        expected={
            "selected_container": "SyntheticVolumeDataContainer",
            "dimensions_xyz": [6, 5, 4],
            "spacing_xyz": [0.5, 0.5, 1.0],
            "geometry_complete": True,
            "cell_data_consistent": True,
            "feature_ids_observed": [0, 1, 2, 3],
            "reserved_feature_zero_established": True,
            "grain_count": 3,
            "phase_names": ["Gamma"],
            "phase_names_source": "stored_metadata",
            "phase_identification_algorithm_run": False,
            "grain_histogram_sample_count": 3,
            "real_zero_measurement_preserved": True,
        },
        tolerance_rationale=(
            "Geometry and feature counts are exact for the synthetic DREAM.3D layout; "
            "a histogram center below 0.2 proves the legitimate zero-valued real row "
            "was retained after excluding only the schema-established feature zero."
        ),
        units="stored DREAM.3D geometry units; feature and histogram counts",
        convention="DIMENSIONS/SPACING/ORIGIN in x-y-z; CellData array in z-y-x-component",
        library_versions=_versions("h5py", "numpy"),
        passed=passed,
    )
    assert overview["geometry"]["path"].endswith("/SyntheticVolumeDataContainer/_SIMPL_GEOMETRY")
    assert overview["geometry"]["dimensions"] == [6, 5, 4]
    assert overview["geometry"]["spacing"] == [0.5, 0.5, 1.0]
    assert overview["grain_count"] == 3
    assert overview["phase_names"] == ["Gamma"]
    assert overview["phase_names_source"] == "stored_metadata"
    assert "no phase-identification algorithm" in overview["phase_names_provenance"].lower()
    assert sum(row["count"] for row in volume_chart["data"]) == 3
    assert min(occupied_labels) < 0.2


def test_calphad_provenance_axes_and_test_database_rejection(
    tmp_path, monkeypatch, materials_evidence
):
    """Require a hashed TDB, explicit domain, physical axes, and no package fixture."""

    _require("pycalphad")
    import pycalphad
    from pycalphad import Database, equilibrium
    from pycalphad import variables as v
    from ultra_deepagents.materials import (
        CalphadInputError,
        inspect_calphad_input,
        is_pycalphad_test_database,
    )

    tdb_text = """$ Synthetic Cu-Ni ideal-solution regression input
ELEMENT VA VACUUM 0 0 0 !
ELEMENT CU FCC_A1 63.546 0 0 !
ELEMENT NI FCC_A1 58.6934 0 0 !
TYPE_DEFINITION % SEQ * !
PHASE FCC_A1 % 1 1 !
CONSTITUENT FCC_A1 :CU,NI,VA: !
PARAMETER G(FCC_A1,CU;0) 298.15 0; 6000 N !
PARAMETER G(FCC_A1,NI;0) 298.15 1000; 6000 N !
PARAMETER G(FCC_A1,VA;0) 298.15 100000; 6000 N !
"""
    database_path = tmp_path / "synthetic-cu-ni.tdb"
    database_path.write_text(tdb_text, encoding="utf-8")
    standalone_contract = {
        "artifact_id": "resource-test-synthetic-cu-ni",
        "expected_sha256": hashlib.sha256(database_path.read_bytes()).hexdigest(),
        "expected_size_bytes": database_path.stat().st_size,
        "assessment_temperature_limits_K": [298.15, 6000.0],
        "assessment_pressure_limits_Pa": [101325.0, 101325.0],
        "assessment_scope": (
            "Synthetic Cu-Ni ideal-solution runtime regression only; not an assessed database"
        ),
        "reference_state": (
            "Synthetic zero-reference regression convention; not a physical SER assessment"
        ),
    }
    missing_source_rejected = False
    missing_license_rejected = False
    try:
        inspect_calphad_input(
            database_path,
            components=["CU", "NI", "VA"],
            phases=["FCC_A1"],
            license_id="test-only",
            **standalone_contract,
        )
    except CalphadInputError as exc:
        missing_source_rejected = "source/provenance is required" in str(exc)
    try:
        inspect_calphad_input(
            database_path,
            components=["CU", "NI", "VA"],
            phases=["FCC_A1"],
            source="Ultra synthetic regression authored in this test",
            **standalone_contract,
        )
    except CalphadInputError as exc:
        missing_license_rejected = "license or use authorization is required" in str(exc)
    provenance = inspect_calphad_input(
        database_path,
        components=["CU", "NI", "VA"],
        phases=["FCC_A1"],
        source="Ultra synthetic regression authored in this test",
        license_id="test-only synthetic input; not production evidence",
        **standalone_contract,
    )
    database = Database(str(database_path))
    axes = {"T": "K", "P": "Pa", "X_NI": "mole fraction"}
    result = equilibrium(
        database,
        ["CU", "NI", "VA"],
        ["FCC_A1"],
        {v.T: [1000.0, 1200.0], v.P: 101325.0, v.X("NI"): [0.25, 0.75]},
    )
    fake_package = tmp_path / "fake_pycalphad"
    fake_init = fake_package / "__init__.py"
    fake_fixture = fake_package / "tests" / "databases" / "fixture.tdb"
    fake_fixture.parent.mkdir(parents=True)
    fake_init.write_text("", encoding="utf-8")
    fake_fixture.write_text(tdb_text, encoding="utf-8")
    monkeypatch.setattr(pycalphad, "__file__", str(fake_init))
    fixture_detected = is_pycalphad_test_database(fake_fixture)
    copied_fixture = tmp_path / "user-looking-database.tdb"
    copied_fixture.write_bytes(fake_fixture.read_bytes())
    copied_fixture_detected = is_pycalphad_test_database(copied_fixture)
    fixture_rejected = False
    copied_fixture_rejected = False
    rejection_message = ""
    copied_rejection_message = ""
    try:
        inspect_calphad_input(
            fake_fixture,
            components=["CU", "NI", "VA"],
            phases=["FCC_A1"],
            source="bundled fake fixture",
            license_id="test-only",
        )
    except CalphadInputError as exc:
        rejection_message = str(exc)
        fixture_rejected = "test databases are fixtures" in rejection_message
    try:
        inspect_calphad_input(
            copied_fixture,
            components=["CU", "NI", "VA"],
            phases=["FCC_A1"],
            source="copied file with misleading path",
            license_id="test-only",
        )
    except CalphadInputError as exc:
        copied_rejection_message = str(exc)
        copied_fixture_rejected = "test databases are fixtures" in copied_rejection_message

    gm_all_finite = bool(result.GM.notnull().all())
    passed = (
        len(provenance["sha256"]) == 64
        and provenance["requested_components"] == ["CU", "NI", "VA"]
        and provenance["requested_phases"] == ["FCC_A1"]
        and provenance["package_test_database"] is False
        and missing_source_rejected
        and missing_license_rejected
        and result.sizes["T"] == 2
        and result.sizes["X_NI"] == 2
        and gm_all_finite
        and fixture_detected
        and copied_fixture_detected
        and fixture_rejected
        and copied_fixture_rejected
    )
    materials_evidence(
        validator_id="materials.calphad.input_domain_axes_fixture_rejection.v1",
        observed={
            "database_sha256": provenance["sha256"],
            "requested_components": provenance["requested_components"],
            "requested_phases": provenance["requested_phases"],
            "axes": axes,
            "temperature_axis_size": int(result.sizes["T"]),
            "composition_axis_size": int(result.sizes["X_NI"]),
            "molar_gibbs_energy_all_finite": gm_all_finite,
            "missing_source_rejected": missing_source_rejected,
            "missing_license_rejected": missing_license_rejected,
            "package_fixture_detected": fixture_detected,
            "copied_fixture_detected": copied_fixture_detected,
            "package_fixture_rejected": fixture_rejected,
            "copied_fixture_rejected": copied_fixture_rejected,
            "fixture_rejection_message": rejection_message,
            "copied_fixture_rejection_message": copied_rejection_message,
        },
        expected={
            "sha256_hex_length": 64,
            "requested_components": ["CU", "NI", "VA"],
            "requested_phases": ["FCC_A1"],
            "axes": {"T": "K", "P": "Pa", "X_NI": "mole fraction"},
            "temperature_axis_size": 2,
            "composition_axis_size": 2,
            "molar_gibbs_energy_all_finite": True,
            "missing_source_rejected": True,
            "missing_license_rejected": True,
            "package_fixture_detected": True,
            "copied_fixture_detected": True,
            "package_fixture_rejected": True,
            "copied_fixture_rejected": True,
        },
        tolerance_rationale=(
            "Hashes, requested thermodynamic domain, grid sizes, and fixture rejection "
            "are exact; finite GM is a runnable toy-equilibrium control, not validation "
            "of an assessed Cu-Ni database."
        ),
        units="kelvin, pascal, mole fraction, and joule per mole for GM",
        convention="pycalphad equilibrium with components CU-NI-VA and phase FCC_A1",
        library_versions=_versions("pycalphad", "numpy"),
        passed=passed,
    )
    assert len(provenance["sha256"]) == 64
    assert provenance["requested_components"] == ["CU", "NI", "VA"]
    assert provenance["requested_phases"] == ["FCC_A1"]
    assert provenance["package_test_database"] is False
    assert axes == {"T": "K", "P": "Pa", "X_NI": "mole fraction"}
    assert result.sizes["T"] == 2 and result.sizes["X_NI"] == 2
    assert bool(result.GM.notnull().all())
    assert fixture_detected is True
    assert fixture_rejected, rejection_message
    assert copied_fixture_detected is True
    assert copied_fixture_rejected, copied_rejection_message


def test_calphad_nist_al_co_w_phase_field_checkpoints(materials_evidence):
    """Exercise the assessed NIST catalog against published phase fields.

    Pure-element endmembers can be degenerate between an ordered phase and its
    disordered parent. The checkpoint therefore validates the appropriate phase
    family for FCC Al/Co and BCC W, while liquid and HCP checkpoints remain exact.

    Two non-unary points are independently read from the assessment paper's
    phase-diagram figures: an Al-W single-phase field and an Al-Co-W three-phase
    triangle. They are cross-engine reproduction checks (Thermo-Calc publication
    versus pycalphad execution), not independent experimental validation.
    """

    _require("pycalphad")
    from ultra_deepagents.materials import (
        inspect_calphad_input,
        run_calphad_equilibrium,
    )

    catalog_root = _embedded_calphad_root()
    database_id = "nist-al-co-w-wang-2017"
    expected_database_sha256 = "107c7330f0326a334742632f7494c7beadf53370edbc188df1a030853ceab5a8"
    expected_source_sha256 = "da3ede1592b8691abe85a1c15d8649364dc0553e4480bb29da4733ec9cef3d96"
    expected_source = "https://materialsdata.nist.gov/handle/11256/948"
    publication_doi = "10.1016/j.calphad.2017.09.007"
    publication_uri = "https://www.sciencedirect.com/science/article/pii/S0364591617301062"
    publication_page_range = "112-130"
    phase_fraction_tolerance = 1e-6
    phase_composition_tolerance = 1e-10
    bulk_mass_balance_tolerance = 1e-8
    gibbs_euler_tolerance_j_per_mol = 1e-3
    expected_phase_families = {
        ("AL", 900.0): {"FCC_A1", "L12_FCC"},
        ("AL", 950.0): {"LIQUID"},
        ("CO", 600.0): {"HCP_A3"},
        ("CO", 800.0): {"FCC_A1", "L12_FCC"},
        ("CO", 1800.0): {"LIQUID"},
        ("W", 3000.0): {"BCC_A2", "BCC_B2"},
        ("W", 3700.0): {"LIQUID"},
    }
    temperatures_by_element = {
        "AL": [900.0, 950.0],
        "CO": [600.0, 800.0, 1800.0],
        "W": [3000.0, 3700.0],
    }
    published_phase_expectations = [
        {
            "checkpoint_id": "wang2017_fig8b_al_w_al4w_single_phase",
            "source_locator": "Wang et al. (2017), Fig. 8(b)",
            "figure_asset_uri": (
                "https://ars.els-cdn.com/content/image/1-s2.0-S0364591617301062-gr8_lrg.jpg"
            ),
            "figure_asset_sha256": (
                "ce4ba92e8861bd56cc37ef5b997477780bde1e9edd784797a555f26c063793ad"
            ),
            "figure_asset_size_bytes": 857601,
            "evidence_basis": (
                "The benchmark point is inside the published present-work Al-W "
                "Al4W single-phase field, away from the annotated invariant lines."
            ),
            "temperature_K": 1000.0,
            "components": ["AL", "W", "VA"],
            "independent_compositions": {"W": 0.2},
            "composition_mole_fraction": {"AL": 0.8, "W": 0.2},
            "published_phase_labels": ["Al4W"],
            "runtime_phase_names": ["AL4W"],
        },
        {
            "checkpoint_id": "wang2017_fig12a_alco_al4w_al5co2_three_phase",
            "source_locator": "Wang et al. (2017), Fig. 12(a)",
            "figure_asset_uri": (
                "https://ars.els-cdn.com/content/image/1-s2.0-S0364591617301062-gr12_lrg.jpg"
            ),
            "figure_asset_sha256": (
                "2cbd5bda9493d138442133a4796b4dbd944cdf106dcda197341baa9f935e59e6"
            ),
            "figure_asset_size_bytes": 822036,
            "evidence_basis": (
                "The benchmark point is near the interior of the published 1173 K "
                "AlCo-Al4W-Al5Co2 three-phase triangle; it is not a digitized boundary."
            ),
            "temperature_K": 1173.0,
            "components": ["AL", "CO", "W", "VA"],
            "independent_compositions": {"CO": 0.26, "W": 0.065},
            "composition_mole_fraction": {"AL": 0.675, "CO": 0.26, "W": 0.065},
            "published_phase_labels": ["Al4W", "Al5Co2", "AlCo"],
            "runtime_phase_names": ["AL4W", "AL5CO2", "BCC_B2"],
        },
    ]

    manifest = inspect_calphad_input(
        catalog_root,
        database_id=database_id,
        components=["AL", "VA"],
        phases=["LIQUID"],
    )
    declared_phases = manifest["available_phases"]
    checkpoints: list[dict[str, object]] = []
    published_checkpoints: list[dict[str, object]] = []
    every_request_used_all_phases = True
    every_request_used_v2_global_equilibrium = True
    every_checkpoint_matches = True
    every_published_checkpoint_matches = True
    all_gm_finite = True
    all_mu_finite = True
    all_vertex_values_finite = True
    maximum_closure_error = 0.0
    maximum_phase_composition_closure_error = 0.0
    maximum_bulk_mass_balance_residual = 0.0
    maximum_gibbs_euler_residual_j_per_mol = 0.0
    caveat_propagated = False

    def _numerical_validity(point: dict[str, object]) -> dict[str, object]:
        bulk = point["conditions"]["composition_mole_fraction"]
        chemical_potentials = point["chemical_potentials_J_per_mol"]
        vertices = point["stable_phase_vertices"]
        bulk_components = set(bulk)
        mu_components_match = set(chemical_potentials) == bulk_components
        chemical_potentials_finite = mu_components_match and all(
            math.isfinite(float(value)) for value in chemical_potentials.values()
        )
        vertex_components_match = bool(vertices) and all(
            set(vertex["composition_mole_fraction"]) == bulk_components for vertex in vertices
        )
        vertex_values_finite = vertex_components_match and all(
            math.isfinite(float(vertex["NP_phase_fraction"]))
            and float(vertex["NP_phase_fraction"]) > 0.0
            and all(
                math.isfinite(float(value))
                for value in vertex["composition_mole_fraction"].values()
            )
            for vertex in vertices
        )
        phase_composition_errors = [
            abs(float(vertex["composition_sum"]) - 1.0) for vertex in vertices
        ]
        max_phase_composition_error = max(phase_composition_errors, default=math.inf)
        reconstructed_bulk = {
            component: math.fsum(
                float(vertex["NP_phase_fraction"])
                * float(vertex["composition_mole_fraction"][component])
                for vertex in vertices
            )
            for component in sorted(bulk_components)
        }
        bulk_residuals = {
            component: abs(reconstructed_bulk[component] - float(bulk[component]))
            for component in sorted(bulk_components)
        }
        max_bulk_residual = max(bulk_residuals.values(), default=math.inf)
        gibbs_from_chemical_potentials = math.fsum(
            float(bulk[component]) * float(chemical_potentials[component])
            for component in sorted(bulk_components)
        )
        computed_gibbs_euler_residual = abs(
            float(point["GM_J_per_mol"]) - gibbs_from_chemical_potentials
        )
        reported_gibbs_euler_residual = float(point["gibbs_euler_residual_J_per_mol"])
        gibbs_euler_residual_matches = math.isclose(
            reported_gibbs_euler_residual,
            computed_gibbs_euler_residual,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        valid = all(
            (
                chemical_potentials_finite,
                vertex_values_finite,
                max_phase_composition_error <= phase_composition_tolerance,
                max_bulk_residual <= bulk_mass_balance_tolerance,
                math.isfinite(computed_gibbs_euler_residual),
                gibbs_euler_residual_matches,
                computed_gibbs_euler_residual <= gibbs_euler_tolerance_j_per_mol,
            )
        )
        return {
            "chemical_potential_components_match_bulk": mu_components_match,
            "all_chemical_potentials_finite": chemical_potentials_finite,
            "stable_vertex_count": len(vertices),
            "vertex_components_match_bulk": vertex_components_match,
            "all_vertex_values_finite": vertex_values_finite,
            "maximum_phase_composition_closure_error": max_phase_composition_error,
            "reconstructed_bulk_composition_mole_fraction": reconstructed_bulk,
            "bulk_mass_balance_residual_by_component": bulk_residuals,
            "maximum_bulk_mass_balance_residual": max_bulk_residual,
            "gibbs_from_chemical_potentials_J_per_mol": gibbs_from_chemical_potentials,
            "computed_gibbs_euler_residual_J_per_mol": computed_gibbs_euler_residual,
            "reported_gibbs_euler_residual_J_per_mol": reported_gibbs_euler_residual,
            "gibbs_euler_residual_matches": gibbs_euler_residual_matches,
            "valid": valid,
        }

    for element, temperatures in temperatures_by_element.items():
        result = run_calphad_equilibrium(
            catalog_root,
            database_id=database_id,
            components=[element, "VA"],
            phases=declared_phases,
            temperatures_K=temperatures,
            pressures_Pa=101325.0,
            total_amount_mol=1.0,
            independent_compositions={},
            max_grid_points=len(temperatures),
            wall_time_seconds=20.0,
            max_result_bytes=8 * 1024 * 1024,
        )
        every_request_used_all_phases &= result["request"]["phases"] == declared_phases
        every_request_used_v2_global_equilibrium &= all(
            (
                result["schema_version"] == "ultra.calphad.equilibrium.v2",
                result["request"]["phase_selection"]["scope"] == "all_database_phases",
                result["request"]["phase_selection"]["excluded_database_phases"] == [],
                result["request"]["phase_selection"]["global_equilibrium_claim_supported"] is True,
            )
        )
        caveat_propagated |= any("metastable" in warning.lower() for warning in result["warnings"])
        for point in result["result"]["points"]:
            temperature = float(point["conditions"]["T_K"])
            stable_phases = [phase["name"] for phase in point["stable_phases"]]
            phase_fraction_sum = float(point["phase_fraction_sum"])
            closure_error = abs(phase_fraction_sum - 1.0)
            gm = float(point["GM_J_per_mol"])
            expected_family = expected_phase_families[(element, temperature)]
            phase_family_match = bool(stable_phases) and set(stable_phases).issubset(
                expected_family
            )
            numerical_validity = _numerical_validity(point)
            every_checkpoint_matches &= phase_family_match
            all_gm_finite &= math.isfinite(gm)
            all_mu_finite &= bool(numerical_validity["all_chemical_potentials_finite"])
            all_vertex_values_finite &= bool(numerical_validity["all_vertex_values_finite"])
            maximum_closure_error = max(maximum_closure_error, closure_error)
            maximum_phase_composition_closure_error = max(
                maximum_phase_composition_closure_error,
                float(numerical_validity["maximum_phase_composition_closure_error"]),
            )
            maximum_bulk_mass_balance_residual = max(
                maximum_bulk_mass_balance_residual,
                float(numerical_validity["maximum_bulk_mass_balance_residual"]),
            )
            maximum_gibbs_euler_residual_j_per_mol = max(
                maximum_gibbs_euler_residual_j_per_mol,
                float(numerical_validity["computed_gibbs_euler_residual_J_per_mol"]),
            )
            checkpoints.append(
                {
                    "element": element,
                    "temperature_K": temperature,
                    "stable_phases": stable_phases,
                    "expected_phase_family": sorted(expected_family),
                    "phase_family_match": phase_family_match,
                    "phase_fraction_sum": phase_fraction_sum,
                    "closure_error": closure_error,
                    "GM_J_per_mol": gm,
                    "chemical_potentials_J_per_mol": point["chemical_potentials_J_per_mol"],
                    "stable_phase_vertices": point["stable_phase_vertices"],
                    "numerical_validity": numerical_validity,
                    "runtime_evidence_sha256": result["evidence"]["sha256"],
                }
            )

    for expectation in published_phase_expectations:
        result = run_calphad_equilibrium(
            catalog_root,
            database_id=database_id,
            components=expectation["components"],
            phases=declared_phases,
            temperatures_K=expectation["temperature_K"],
            pressures_Pa=101325.0,
            total_amount_mol=1.0,
            independent_compositions=expectation["independent_compositions"],
            max_grid_points=1,
            wall_time_seconds=20.0,
            max_result_bytes=8 * 1024 * 1024,
        )
        every_request_used_all_phases &= result["request"]["phases"] == declared_phases
        global_equilibrium_request = all(
            (
                result["schema_version"] == "ultra.calphad.equilibrium.v2",
                result["request"]["phase_selection"]["scope"] == "all_database_phases",
                result["request"]["phase_selection"]["excluded_database_phases"] == [],
                result["request"]["phase_selection"]["global_equilibrium_claim_supported"] is True,
            )
        )
        every_request_used_v2_global_equilibrium &= global_equilibrium_request
        assert result["result"]["point_count"] == 1
        point = result["result"]["points"][0]
        stable_phase_names = [phase["name"] for phase in point["stable_phases"]]
        phase_identity_match = stable_phase_names == expectation["runtime_phase_names"]
        composition_match = set(point["conditions"]["composition_mole_fraction"]) == set(
            expectation["composition_mole_fraction"]
        ) and all(
            math.isclose(
                float(point["conditions"]["composition_mole_fraction"][component]),
                float(expected_value),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for component, expected_value in expectation["composition_mole_fraction"].items()
        )
        phase_fraction_sum = float(point["phase_fraction_sum"])
        closure_error = abs(phase_fraction_sum - 1.0)
        gm = float(point["GM_J_per_mol"])
        numerical_validity = _numerical_validity(point)
        checkpoint_matches = all(
            (
                phase_identity_match,
                composition_match,
                global_equilibrium_request,
                math.isfinite(gm),
                closure_error <= phase_fraction_tolerance,
                bool(numerical_validity["valid"]),
            )
        )
        every_published_checkpoint_matches &= checkpoint_matches
        all_gm_finite &= math.isfinite(gm)
        all_mu_finite &= bool(numerical_validity["all_chemical_potentials_finite"])
        all_vertex_values_finite &= bool(numerical_validity["all_vertex_values_finite"])
        maximum_closure_error = max(maximum_closure_error, closure_error)
        maximum_phase_composition_closure_error = max(
            maximum_phase_composition_closure_error,
            float(numerical_validity["maximum_phase_composition_closure_error"]),
        )
        maximum_bulk_mass_balance_residual = max(
            maximum_bulk_mass_balance_residual,
            float(numerical_validity["maximum_bulk_mass_balance_residual"]),
        )
        maximum_gibbs_euler_residual_j_per_mol = max(
            maximum_gibbs_euler_residual_j_per_mol,
            float(numerical_validity["computed_gibbs_euler_residual_J_per_mol"]),
        )
        published_checkpoints.append(
            {
                "checkpoint_id": expectation["checkpoint_id"],
                "source_locator": expectation["source_locator"],
                "figure_asset_uri": expectation["figure_asset_uri"],
                "figure_asset_sha256": expectation["figure_asset_sha256"],
                "figure_asset_size_bytes": expectation["figure_asset_size_bytes"],
                "evidence_basis": expectation["evidence_basis"],
                "temperature_K": point["conditions"]["T_K"],
                "composition_mole_fraction": point["conditions"]["composition_mole_fraction"],
                "published_phase_labels": expectation["published_phase_labels"],
                "expected_runtime_phase_names": expectation["runtime_phase_names"],
                "stable_phases": point["stable_phases"],
                "stable_phase_vertices": point["stable_phase_vertices"],
                "phase_identity_match": phase_identity_match,
                "composition_match": composition_match,
                "global_equilibrium_request": global_equilibrium_request,
                "phase_fraction_sum": phase_fraction_sum,
                "closure_error": closure_error,
                "GM_J_per_mol": gm,
                "chemical_potentials_J_per_mol": point["chemical_potentials_J_per_mol"],
                "numerical_validity": numerical_validity,
                "runtime_evidence_sha256": result["evidence"]["sha256"],
                "matches_published_phase_field": checkpoint_matches,
            }
        )

    registry = manifest["registry_manifest"]
    provenance_valid = all(
        (
            manifest["sha256"] == expected_database_sha256,
            manifest["size_bytes"] == 21274,
            manifest["source"] == expected_source,
            manifest["license_id"] == "CC0-1.0",
            manifest["physical_elements"] == ["AL", "CO", "W"],
            manifest["vacancy_components"] == ["VA"],
            manifest["pseudo_elements"] == ["/-"],
            manifest["assessment_temperature_limits_K"] == [298.14, 6000.0],
            manifest["assessment_pressure_limits_Pa"] == [101325.0, 101325.0],
            registry["database_id"] == database_id,
            registry["source_sha256"] == expected_source_sha256,
            registry["publication_doi"] == publication_doi,
            len(registry["manifest_sha256"]) == 64,
            len(manifest["manifest_sha256"]) == 64,
            len(declared_phases) == 18,
            manifest["parameter_count"] == 174,
        )
    )
    passed = all(
        (
            provenance_valid,
            len(checkpoints) == len(expected_phase_families),
            len(published_checkpoints) == len(published_phase_expectations),
            every_request_used_all_phases,
            every_request_used_v2_global_equilibrium,
            every_checkpoint_matches,
            every_published_checkpoint_matches,
            all_gm_finite,
            all_mu_finite,
            all_vertex_values_finite,
            maximum_closure_error <= phase_fraction_tolerance,
            maximum_phase_composition_closure_error <= phase_composition_tolerance,
            maximum_bulk_mass_balance_residual <= bulk_mass_balance_tolerance,
            maximum_gibbs_euler_residual_j_per_mol <= gibbs_euler_tolerance_j_per_mol,
            caveat_propagated,
        )
    )
    checkpoints.sort(key=lambda item: (str(item["element"]), float(item["temperature_K"])))
    materials_evidence(
        validator_id="materials.calphad.nist_al_co_w_phase_field_checkpoints.v2",
        observed={
            "database_id": registry["database_id"],
            "database_sha256": manifest["sha256"],
            "database_size_bytes": manifest["size_bytes"],
            "source": manifest["source"],
            "source_sha256": registry["source_sha256"],
            "license_id": manifest["license_id"],
            "publication_doi": registry["publication_doi"],
            "publication_uri": publication_uri,
            "publication_page_range": publication_page_range,
            "manifest_sha256": registry["manifest_sha256"],
            "physical_elements": manifest["physical_elements"],
            "vacancy_components": manifest["vacancy_components"],
            "pseudo_elements": manifest["pseudo_elements"],
            "assessment_temperature_limits_K": manifest["assessment_temperature_limits_K"],
            "assessment_pressure_limits_Pa": manifest["assessment_pressure_limits_Pa"],
            "declared_phase_count": len(declared_phases),
            "declared_phases": declared_phases,
            "parameter_count": manifest["parameter_count"],
            "all_declared_phases_requested": every_request_used_all_phases,
            "runtime_schema_version": "ultra.calphad.equilibrium.v2",
            "global_equilibrium_over_all_database_phases": (
                every_request_used_v2_global_equilibrium
            ),
            "all_requests_used_v2_global_equilibrium": (every_request_used_v2_global_equilibrium),
            "checkpoint_count": len(checkpoints),
            "checkpoints": checkpoints,
            "published_phase_checkpoint_count": len(published_checkpoints),
            "published_phase_checkpoints": published_checkpoints,
            "all_GM_finite": all_gm_finite,
            "all_MU_finite": all_mu_finite,
            "all_phase_vertex_values_finite": all_vertex_values_finite,
            "maximum_phase_fraction_closure_error": maximum_closure_error,
            "maximum_phase_composition_closure_error": (maximum_phase_composition_closure_error),
            "maximum_bulk_mass_balance_residual": (maximum_bulk_mass_balance_residual),
            "maximum_gibbs_euler_residual_J_per_mol": (maximum_gibbs_euler_residual_j_per_mol),
            "assessment_caveat_propagated": caveat_propagated,
            "qualification_scope": (
                "Cross-engine reproduction of phase fields published with the same "
                "assessment; not independent experimental validation and not a "
                "digitized phase-boundary benchmark."
            ),
        },
        expected={
            "database_sha256": expected_database_sha256,
            "database_size_bytes": 21274,
            "source": expected_source,
            "source_sha256": expected_source_sha256,
            "license_id": "CC0-1.0",
            "publication_doi": publication_doi,
            "publication_uri": publication_uri,
            "publication_page_range": publication_page_range,
            "physical_elements": ["AL", "CO", "W"],
            "vacancy_components": ["VA"],
            "pseudo_elements": ["/-"],
            "assessment_temperature_limits_K": [298.14, 6000.0],
            "assessment_pressure_limits_Pa": [101325.0, 101325.0],
            "declared_phase_count": 18,
            "parameter_count": 174,
            "all_declared_phases_requested": True,
            "runtime_schema_version": "ultra.calphad.equilibrium.v2",
            "global_equilibrium_over_all_database_phases": True,
            "checkpoint_phase_families": [
                {
                    "element": element,
                    "temperature_K": temperature,
                    "allowed_stable_phases": sorted(phase_family),
                }
                for (element, temperature), phase_family in sorted(expected_phase_families.items())
            ],
            "published_phase_expectations": [
                {
                    key: expectation[key]
                    for key in (
                        "checkpoint_id",
                        "source_locator",
                        "figure_asset_uri",
                        "figure_asset_sha256",
                        "figure_asset_size_bytes",
                        "evidence_basis",
                        "temperature_K",
                        "composition_mole_fraction",
                        "published_phase_labels",
                        "runtime_phase_names",
                    )
                }
                for expectation in published_phase_expectations
            ],
            "all_GM_finite": True,
            "all_MU_finite": True,
            "all_phase_vertex_values_finite": True,
            "maximum_phase_fraction_closure_error": phase_fraction_tolerance,
            "maximum_phase_composition_closure_error": (phase_composition_tolerance),
            "maximum_bulk_mass_balance_residual": bulk_mass_balance_tolerance,
            "maximum_gibbs_euler_residual_J_per_mol": gibbs_euler_tolerance_j_per_mol,
            "assessment_caveat_propagated": True,
            "qualification_scope": (
                "Same-assessment cross-engine phase-field reproduction only; an "
                "independent experimental boundary/tie-line dataset remains required."
            ),
        },
        tolerance_rationale=(
            "Catalog hash, license, inventory, and unary checkpoint temperatures are exact. "
            "The temperatures bracket the TDB's Al, Co, and W unary transitions; pure "
            "endmembers may select the ordered or disordered name within the declared FCC/BCC "
            "phase family. Fig. 8(b) supplies the Al4W single-phase expectation and Fig. 12(a) "
            "supplies the AlCo-Al4W-Al5Co2 three-phase expectation; the selected compositions "
            "are interior points, so no phase-boundary value or solver-derived fraction is used "
            "as an oracle. Phase fractions close within 1e-6, phase compositions within 1e-10, "
            "vertex-weighted bulk mass balance within 1e-8, and the Gibbs-Euler residual "
            "within 1e-3 J/mol; GM and every MU/X/NP value are finite."
        ),
        units="kelvin, pascal, mole, mole fraction, joule per mole, bytes",
        convention=(
            "NIST Al-Co-W Wang 2017 TDB; global equilibrium over all 18 declared phases; "
            "N=1 mol at 101325 Pa; pure-element ordered/disordered aliases treated as one family; "
            "published phase labels AlCo/Al4W/Al5Co2 map to BCC_B2/AL4W/AL5CO2"
        ),
        library_versions=_versions("pycalphad", "numpy"),
        passed=passed,
    )

    assert provenance_valid
    assert len(checkpoints) == 7
    assert len(published_checkpoints) == 2
    assert every_request_used_all_phases
    assert every_request_used_v2_global_equilibrium
    assert every_checkpoint_matches
    assert every_published_checkpoint_matches
    assert all_gm_finite
    assert all_mu_finite
    assert all_vertex_values_finite
    assert maximum_closure_error <= phase_fraction_tolerance
    assert maximum_phase_composition_closure_error <= phase_composition_tolerance
    assert maximum_bulk_mass_balance_residual <= bulk_mass_balance_tolerance
    assert maximum_gibbs_euler_residual_j_per_mol <= gibbs_euler_tolerance_j_per_mol
    assert caveat_propagated
