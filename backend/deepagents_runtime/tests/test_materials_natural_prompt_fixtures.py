from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import zipfile
from pathlib import Path
from types import ModuleType

import fitz
import numpy as np
import pytest
import zarr
from ultra_deepagents.context import AgentRunContext
from ultra_deepagents.materials.degradation import (
    ClosedInterval,
    CorrosionPenetrationInputs,
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
from ultra_deepagents.papers.tools import (
    PaperTextEvidenceError,
    bind_paper_text_literal_from_cache,
    ingest_pdf_file,
)
from ultra_deepagents.sensors import build_min_max_envelope, open_sensor_series

_BUILDER_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "materials_natural_prompts" / "build_fixtures.py"
)


def _load_builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "materials_natural_prompt_fixtures", _BUILDER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fixture_builder_is_reproducible_and_oracles_match_strict_readers(tmp_path: Path) -> None:
    builder = _load_builder()
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"

    first = builder.build(first_root)
    second = builder.build(second_root)

    for name in (
        builder.SENSOR_ARCHIVE_NAME,
        builder.PAPER_NAME,
        builder.GOLD_NAME,
    ):
        assert _sha256(first_root / name) == _sha256(second_root / name)
    assert first == second

    encoded_gold = json.loads((first_root / builder.GOLD_NAME).read_text(encoding="utf-8"))
    assert encoded_gold == first
    sensor_gold = first["sensor"]
    extraction_root = tmp_path / "extracted"
    with zipfile.ZipFile(first_root / sensor_gold["archive"]) as archive:
        archive.extractall(extraction_root)
    sensor_root = extraction_root / sensor_gold["directory_name"]

    series = open_sensor_series(
        sensor_root,
        expected_tree_manifest_sha256=sensor_gold["tree_manifest_sha256"],
    )
    group = zarr.open_group(str(sensor_root), mode="r")
    values = np.asarray(group["signals/ae-1"][:]).tolist()
    validity = np.asarray(group["quality/ae-1-valid"][:]).tolist()
    saturation = np.asarray(group["quality/ae-1-saturated"][:]).tolist()
    envelope = build_min_max_envelope(
        values,
        max_buckets=5,
        validity=validity,
        saturation=saturation,
    )

    assert series.lineage.status == "tree_verified"
    assert series.values_validated
    assert series.channels[0].invalid_count == 1
    assert series.channels[0].saturation_count == 1
    assert envelope.factor == 5
    assert len(envelope.buckets) == 5
    assert min(bucket.minimum for bucket in envelope.buckets if bucket.minimum is not None) == -800
    assert max(bucket.maximum for bucket in envelope.buckets if bucket.maximum is not None) == 1000
    assert sum(bucket.invalid_count for bucket in envelope.buckets) == 1
    assert sum(bucket.saturation_count for bucket in envelope.buckets) == 1

    paper_gold = first["paper"]
    paper_path = first_root / paper_gold["file"]
    assert _sha256(paper_path) == paper_gold["pdf_sha256"]
    document = fitz.open(paper_path)
    try:
        assert document.page_count == 2
        assert "Tomaszewska" in document.load_page(0).get_text()
        assert document.load_page(1).get_text().strip() == ""
        rendered = document.load_page(1).get_pixmap(
            matrix=fitz.Matrix(2.0, 2.0),
            alpha=False,
        )
        rendered_sha = hashlib.sha256(rendered.tobytes(output="png")).hexdigest()
    finally:
        document.close()
    assert rendered_sha == paper_gold["page_2_render_png_sha256"]

    paper_context = AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="fixture-org",
        user_id="fixture-user",
        project_id="fixture-project",
        thread_id="fixture-thread",
        run_id="fixture-run",
        workspace_root=str(tmp_path / "paper-workspace"),
        artifact_root=str(tmp_path / "paper-artifacts"),
    )
    ingest_pdf_file(
        paper_context,
        paper_path,
        paper_id="synthetic-calphad-tables",
    )
    table_1_binding = bind_paper_text_literal_from_cache(
        paper_context,
        paper_id="synthetic-calphad-tables",
        page=1,
        exact_text="9.1",
        numeric_binding=True,
        row_id="tomaszewska",
        column_id="al-at-pct",
    )
    assert table_1_binding["exact_substring"] == "9.1"
    assert table_1_binding["numeric_value_decimal"] == "9.1"
    assert table_1_binding["page_text_sha256"]
    assert table_1_binding["extractor_revision"].startswith("pymupdf-")
    with pytest.raises(PaperTextEvidenceError, match="paper_text_literal_ambiguous"):
        bind_paper_text_literal_from_cache(
            paper_context,
            paper_id="synthetic-calphad-tables",
            page=1,
            exact_text="9.0",
            numeric_binding=True,
        )

    analytical = first["analytical_oracles"]
    assert analytical["fcc_001_schmid_maximum"] == 1 / math.sqrt(6)
    assert analytical["profile_metrics"]["goodness_of_fit"] == math.sqrt(1.125)

    degradation = analytical["degradation"]
    provenance = EvidenceProvenance(
        artifact_id="synthetic-acceptance-input",
        sha256="a" * 64,
        locator="fixture://acceptance-gold",
        citation="Synthetic acceptance control; not materials evidence",
    )
    lefm = evaluate_mode_i_lefm(
        nominal_tensile_stress_pa=100.0e6,
        crack_length_m=0.01,
        remaining_ligament_m=0.09,
        thickness_m=0.02,
        yield_strength_pa=500.0e6,
        constraint_state="plane_strain",
        minimum_dimension_to_plastic_zone_ratio=20.0,
        geometry=GeometryFactorCalibration(
            geometry_id="synthetic-centered-crack",
            crack_length_definition="half crack length from centerline to one tip",
            nominal_stress_definition="remote gross-section tensile stress",
            geometry_factor=1.12,
            domain=ClosedInterval(
                "crack_length_over_crack_plus_remaining_ligament", "1", 0.01, 0.6
            ),
            evaluated_parameter=0.1,
            provenance=provenance,
        ),
        criterion_provenance=provenance,
    )
    assert lefm.stress_intensity_pa_sqrt_m == pytest.approx(
        degradation["mode_i_lefm"]["stress_intensity_pa_sqrt_m"]
    )
    assert lefm.plastic_zone_radius_m == pytest.approx(
        degradation["mode_i_lefm"]["plane_strain_plastic_zone_radius_m"]
    )
    assert lefm.applicability_passed is True

    paris_gold = degradation["paris"]
    conditions = ParisTestConditions(
        material_state_id="synthetic-state",
        environment_id="synthetic-dry-air",
        load_ratio=0.1,
        temperature_k=298.15,
        cycle_frequency_hz=10.0,
        waveform_id="constant-amplitude-sine",
        specimen_thickness_m=0.012,
        specimen_geometry_id="synthetic-compact-tension",
        delta_k_definition_id="applied-linear-elastic-Kmax-minus-Kmin",
        crack_growth_rate_method_id="incremental-polynomial-reduction-v1",
    )
    paris = fit_paris_law(
        paris_gold["delta_k_mpa_sqrt_m"],
        paris_gold["growth_rate_m_per_cycle"],
        calibration_indices=paris_gold["calibration_indices"],
        held_out_indices=paris_gold["held_out_indices"],
        conditions=conditions,
        observations_provenance=provenance,
    )
    assert paris.coefficient_c == pytest.approx(paris_gold["coefficient_c"], rel=1.0e-12)
    assert paris.exponent_m == pytest.approx(paris_gold["exponent_m"], rel=1.0e-12)
    assert (
        paris.held_out_residuals.maximum_absolute_log_error
        < paris_gold["maximum_exact_data_log_residual"]
    )

    creep = evaluate_norton_arrhenius_creep_rate(
        NortonArrheniusCreepModel(
            pre_exponential_per_s=1.0e-4,
            reference_stress_pa=100.0e6,
            stress_exponent=4.0,
            activation_energy_j_per_mol=200_000.0,
            stress_domain_pa=ClosedInterval("stress", "Pa", 50.0e6, 300.0e6),
            temperature_domain_k=ClosedInterval("temperature", "K", 900.0, 1200.0),
            material_state_id="synthetic-state",
            environment_id="synthetic-argon",
            stress_measure_id="von-Mises-effective-stress",
            provenance=provenance,
        ),
        stress_pa=200.0e6,
        temperature_k=1000.0,
        material_state_id="synthetic-state",
        environment_id="synthetic-argon",
    )
    assert creep.effective_secondary_creep_rate_per_s == pytest.approx(
        degradation["creep"]["secondary_rate_per_s"]
    )

    linear_oxidation = evaluate_oxidation_mass_gain(
        OxidationKineticsModel(
            law="linear",
            rate_constant=1.0e-3,
            rate_constant_unit=degradation["oxidation"]["linear_rate_constant_unit"],
            initial_areal_mass_gain_kg_per_m2=1.0e-2,
            time_domain_s=ClosedInterval("time", "s", 0.0, 100.0),
            temperature_domain_k=ClosedInterval("temperature", "K", 1073.0, 1073.0),
            material_state_id="synthetic-state",
            environment_id="synthetic-dry-air",
            area_basis_id="initial-total-geometric-exposed-area",
            provenance=provenance,
        ),
        exposure_time_s=10.0,
        temperature_k=1073.0,
        material_state_id="synthetic-state",
        environment_id="synthetic-dry-air",
    )
    assert linear_oxidation.areal_mass_gain_kg_per_m2 == pytest.approx(
        degradation["oxidation"]["linear_areal_mass_gain_kg_per_m2_at_10_s"]
    )
    parabolic_oxidation = evaluate_oxidation_mass_gain(
        OxidationKineticsModel(
            law="parabolic",
            rate_constant=4.0e-4,
            rate_constant_unit=degradation["oxidation"]["parabolic_rate_constant_unit"],
            initial_areal_mass_gain_kg_per_m2=3.0e-2,
            time_domain_s=ClosedInterval("time", "s", 0.0, 100.0),
            temperature_domain_k=ClosedInterval("temperature", "K", 1073.0, 1073.0),
            material_state_id="synthetic-state",
            environment_id="synthetic-dry-air",
            area_basis_id="initial-total-geometric-exposed-area",
            provenance=provenance,
        ),
        exposure_time_s=4.0,
        temperature_k=1073.0,
        material_state_id="synthetic-state",
        environment_id="synthetic-dry-air",
    )
    assert parabolic_oxidation.areal_mass_gain_kg_per_m2 == pytest.approx(
        degradation["oxidation"]["parabolic_areal_mass_gain_kg_per_m2_at_4_s"]
    )
    assert linear_oxidation.model.rate_constant_unit == "kg*m^-2*s^-1"
    assert parabolic_oxidation.model.rate_constant_unit == "kg^2*m^-4*s^-1"

    corrosion = convert_corrosion_current_to_uniform_penetration(
        CorrosionPenetrationInputs(
            corrosion_current_density_a_per_m2=1.0,
            equivalent_mass_kg_per_mol_electron=0.055845 / 2.0,
            density_kg_per_m3=7874.0,
            current_efficiency=0.8,
            duration_s=365.25 * 24.0 * 3600.0,
            material_state_id="synthetic-state",
            environment_id="synthetic-cell",
            current_density_area_basis_id="geometric-electrode-area-before-exposure",
            current_density_provenance=provenance,
            equivalent_mass_provenance=provenance,
            density_provenance=provenance,
            efficiency_provenance=provenance,
        )
    )
    assert corrosion.average_uniform_penetration_m == pytest.approx(
        degradation["corrosion"]["average_uniform_penetration_m_at_one_year"]
    )
