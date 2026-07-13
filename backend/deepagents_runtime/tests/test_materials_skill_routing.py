import json
import re
from pathlib import Path

import pytest
from ultra_deepagents.agent import (
    build_run_context_brief,
    build_runtime_prompt_suffix,
    build_subagents,
    looks_materials_computational_goal,
    looks_quantitative_rigor_goal,
    looks_scoped_delegation_goal,
    sandbox_compute_resources,
)
from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.context import AgentRunContext
from ultra_deepagents.materials.validation import ValidationCheck, parse_assessment_record

_RUNTIME_ROOT = Path(__file__).resolve().parents[1]
_SKILLS_ROOT = _RUNTIME_ROOT / "skills"


def _context(
    goal: str,
    *,
    suggested_domain: str = "",
    pro: bool = False,
) -> AgentRunContext:
    return AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="local-org",
        user_id="user-1",
        project_id="local-project",
        thread_id="thread-1",
        run_id="run-materials-routing",
        goal=goal,
        selection_context=({"suggested_domain": suggested_domain} if suggested_domain else {}),
        workflow_hint={"id": "pro_mode"} if pro else {},
    )


@pytest.mark.parametrize(
    "goal",
    [
        "Identify this CIF's space group.",
        "Calculate the Cu K-alpha XRD pattern for this structure.",
        "Build a CALPHAD phase diagram from the supplied TDB.",
        "Construct an octahedral self-interstitial point defect in GaN.",
        "Calculate vacancy multiplicities for this crystal structure.",
        "Determine the defect formation energy inputs and report unsupported terms.",
        "Create a Ga_N antisite in GaN.",
        "Construct a Frenkel pair in silicon.",
        "Substitute Mg onto a Ga site in GaN.",
        "Find all symmetry-inequivalent vacancy sites in this structure.",
        "Enumerate antisite defects in the supplied crystal.",
        "Run a Scheil solidification path for the selected alloy and TDB.",
        "Calculate resolved shear stress on every FCC slip system for these grain orientations.",
        "Generate the HCP first-order pyramidal c+a slip-system geometry without assuming c/a.",
        "Fit a rigid registration and score held-out EBSD-to-APT fiducials.",
        "Compute weighted profile residuals for this measured XRD Rietveld fit.",
        "Analyze this calibrated 4D-STEM diffraction dataset.",
        "Estimate fatigue and creep metrics for these stress-strain histories.",
        "Quantify oxidation and corrosion rates from the measured mass-change series.",
        "Analyze this acoustic-emission waveform without losing transient peaks.",
        "Align the calibrated thermocouple and photodiode process telemetry with the thermal video.",
        "Validate the clocks, units, uncertainty, and saturation flags in this sensor data Zarr.",
    ],
)
def test_canonical_materials_requests_register_computational_delegation(goal: str):
    assert looks_materials_computational_goal(goal) is True
    assert looks_scoped_delegation_goal(goal) is True
    assert looks_quantitative_rigor_goal(goal) is False
    subagents = build_subagents(context=_context(goal), skills_sources=["/skills/"])
    assert [subagent["name"] for subagent in subagents] == ["code-runner"]
    assert subagents[0]["skills"] == ["/skills/"]


def test_materials_selection_hint_registers_computational_delegation():
    context = _context("Inspect the selected file.", suggested_domain="materials")

    subagents = build_subagents(context=context, skills_sources=["/skills/"])

    assert [subagent["name"] for subagent in subagents] == ["code-runner"]


def test_materials_classifier_uses_token_boundaries_and_requires_an_action():
    assert looks_materials_computational_goal("Identify the specific code path.") is False
    assert looks_materials_computational_goal("What does XRD mean?") is False
    assert looks_materials_computational_goal("Analyze the supplied EDS spectrum.") is True


@pytest.mark.parametrize(
    "goal",
    [
        # The flagship phrasing: plural method tokens must match.
        "Compute the equilibrium phase fractions of a Co-9Al-9W at.% alloy at 900 C.",
        "Identify the space groups of these CIF files.",
        "Calculate the vacancies formed during quenching of this alloy.",
        # Canonical materials verbs that previously carried no action token.
        "Measure the misorientation distribution across grain boundaries in this EBSD map.",
        "Estimate grain size from this optical micrograph using stereology.",
        "Segment the grains in this EBSD map and report misorientation statistics.",
    ],
)
def test_materials_classifier_recognizes_plurals_and_measurement_verbs(goal: str):
    assert looks_materials_computational_goal(goal) is True


@pytest.mark.parametrize(
    "goal",
    [
        # Measurement verbs alone must not pull unrelated requests in.
        "Estimate the marketing budget for next year.",
        "Measure how long the test suite takes to run.",
        "Segment the customer list by region.",
    ],
)
def test_measurement_verbs_without_materials_methods_stay_out(goal: str):
    assert looks_materials_computational_goal(goal) is False


def test_calphad_shaped_run_implies_materials_context():
    """A run that registers the typed CALPHAD tools must also carry the
    materials routing (skill brief, and on Pro the materials contract);
    the two gates previously disagreed on 'equilibrium phase fractions'."""
    from ultra_deepagents.agent import _should_register_calphad_tools, is_materials_context

    flagship = _context(
        "Compute the equilibrium phase fractions of a Co-9Al-9W at.% alloy at 900 C."
    )
    assert _should_register_calphad_tools(flagship) is True
    assert is_materials_context(flagship) is True

    selected_tdb = AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="local-org",
        user_id="user-1",
        project_id="local-project",
        thread_id="thread-1",
        run_id="run-materials-routing",
        goal="Run equilibrium for my alloy.",
        resource_descriptors=(
            {"type": "selected_resource", "original_name": "mydb.tdb", "content_type": ""},
        ),
    )
    assert _should_register_calphad_tools(selected_tdb) is True
    assert is_materials_context(selected_tdb) is True

    plain = _context("Summarize this paper about turbine blades.")
    assert _should_register_calphad_tools(plain) is False
    assert is_materials_context(plain) is False


@pytest.mark.parametrize(
    "goal",
    [
        "space group CIF",
        "phase XRD",
        "grains DREAM.3D",
        "What is the space group in this CIF?",
        "Which phases are present in this XRD pattern?",
        "How many grains are in this DREAM.3D file?",
        "Space-group determination, CIF attached.",
        "Phase identification from X-ray diffraction.",
        "Grain count from DREAM3D.",
    ],
)
def test_materials_classifier_recognizes_question_and_acronym_forms(goal: str):
    assert looks_materials_computational_goal(goal) is True
    assert looks_scoped_delegation_goal(goal) is True
    assert looks_quantitative_rigor_goal(goal) is False


@pytest.mark.parametrize(
    "goal",
    [
        "DFT silicon",
        "Can DFT predict the silicon band structure?",
        "Run density-functional theory for Si.",
        "Simulate MD for silicon.",
        "Molecular-dynamics simulation of Si at 300 K.",
        "Run a LAMMPS molecular dynamics trajectory for silicon.",
    ],
)
def test_heavy_atomistics_shorthand_routes_to_explicit_materials_boundary(goal: str):
    assert looks_materials_computational_goal(goal) is True
    assert looks_scoped_delegation_goal(goal) is True
    assert looks_quantitative_rigor_goal(goal) is False

    suffix = build_runtime_prompt_suffix(_context(goal, pro=True), elapsed_seconds=5)
    assert "Materials results contract" in suffix
    assert "no production DFT or molecular-dynamics engine" in suffix
    assert "unsupported" in suffix
    assert "at least 3 seeds" not in suffix


@pytest.mark.parametrize(
    "goal",
    [
        "What does XRD mean?",
        "What is a CIF file?",
        "Convert this MD file to PDF.",
        "Send the space-group report to the team.",
    ],
)
def test_materials_classifier_adversarial_noncomputational_forms_stay_out(goal: str):
    assert looks_materials_computational_goal(goal) is False


def test_unavailable_heavy_atomistics_request_routes_to_capability_boundary():
    goal = "Run a VASP DFT calculation for this POSCAR."

    assert looks_materials_computational_goal(goal) is True
    assert looks_scoped_delegation_goal(goal) is True
    assert looks_quantitative_rigor_goal(goal) is False


def test_materials_selection_hint_is_visible_as_skill_routing_evidence():
    context = _context(
        "Inspect the selected HDF5 file.",
        suggested_domain=" MaTeRiAlS ",
    )

    brief = build_run_context_brief(context)

    assert "suggested_domain: materials" in brief
    assert "/skills/computational-materials/SKILL.md" in brief
    assert "/skills/materials-structure-thermo/SKILL.md" in brief
    assert "/skills/materials-characterization/SKILL.md" in brief
    assert "/skills/materials-processing-kinetics/SKILL.md" in brief
    assert "/skills/materials-characterization-advanced/SKILL.md" in brief
    assert "/skills/materials-sensor-data/SKILL.md" in brief
    assert "/skills/materials-crystal-plasticity/SKILL.md" not in brief
    assert "/skills/materials-mechanics-degradation/SKILL.md" not in brief


def test_prompt_inferred_materials_context_also_requires_skill_routing():
    brief = build_run_context_brief(
        _context("Compute a CALPHAD equilibrium with the embedded Al-Co-W TDB.")
    )

    assert "suggested_domain: materials" in brief
    assert "/skills/materials-structure-thermo/SKILL.md" in brief
    assert "/skills/materials-processing-kinetics/SKILL.md" in brief


def test_deterministic_materials_pro_uses_materials_contract_not_dynamics_contract():
    goal = "Simulate the Cu K-alpha XRD spectrum for ordered Ni3Al."
    context = _context(
        goal,
        suggested_domain="materials",
        pro=True,
    )

    suffix = build_runtime_prompt_suffix(context, elapsed_seconds=5)

    # runner.py uses this shared classifier for its completion re-prompt gate.
    assert looks_quantitative_rigor_goal(goal) is False
    assert "Materials results contract" in suffix
    assert "radiation source and wavelength" in suffix
    assert "/outputs/materials_validation.json" in suffix
    assert "parse_assessment_record" in suffix
    assert "at least 3 seeds or initial conditions" not in suffix
    assert "at least 2 observation durations" not in suffix
    assert "borderline parameter values" not in suffix

    inferred_suffix = build_runtime_prompt_suffix(
        _context(goal, pro=True),
        elapsed_seconds=5,
    )
    assert "Materials results contract" in inferred_suffix
    assert "at least 2 observation durations" not in inferred_suffix


def test_hcp_slip_system_hyphen_routes_to_materials_contract():
    goal = (
        "Generate the HCP first-order pyramidal c+a family for a phase whose c/a was not "
        "supplied. Call materials_analyze_crystal_slip and report whether a numerical "
        "slip-system calculation is supportable."
    )

    assert looks_materials_computational_goal(goal) is True
    assert looks_quantitative_rigor_goal(goal) is False
    suffix = build_runtime_prompt_suffix(_context(goal, pro=True), elapsed_seconds=5)
    assert "Materials results contract" in suffix
    assert "at least 3 seeds" not in suffix


def test_nonmaterials_pro_keeps_general_quantitative_contract():
    context = _context(
        "Run a simulation and classify chaotic regimes from Lyapunov metrics.",
        pro=True,
    )

    suffix = build_runtime_prompt_suffix(context, elapsed_seconds=5)

    assert "Results contract" in suffix
    assert "at least 3 seeds" in suffix
    assert "initial conditions" in suffix
    assert "Materials results contract" not in suffix


def test_materials_skill_family_is_scoped_and_declares_engine_boundary():
    microstructure = (_SKILLS_ROOT / "computational-materials" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    structure_thermo = (_SKILLS_ROOT / "materials-structure-thermo" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    characterization = (_SKILLS_ROOT / "materials-characterization" / "SKILL.md").read_text(
        encoding="utf-8"
    )

    assert "compatibility" in microstructure.lower()
    assert "EBSD" in microstructure
    assert "TriBeam" in microstructure
    assert "materials-structure-thermo" in structure_thermo
    assert "test fixture" in structure_thermo.lower()
    assert "no production atomistic engine" in structure_thermo.lower()
    assert "canonical dependent-component convention" in structure_thermo
    assert "self-consistency alone" in structure_thermo
    assert "generic code" in structure_thermo
    assert "runner to manufacture" in structure_thermo
    assert "automatically retains inspected `VA`" in structure_thermo
    assert "parse_assessment_record(json.loads(encoded))" in structure_thermo
    assert "Never instantiate `ScientificAssessment`" in structure_thermo
    assert "materials-characterization" in characterization
    assert "XRD" in characterization
    assert "Raman" in characterization
    assert "XPS" in characterization
    for skill_text in (microstructure, structure_thermo, characterization):
        assert "/outputs/materials_validation.json" in skill_text
        assert "validator_id" in skill_text
        assert '"outcome"' in skill_text
        assert '"expected"' in skill_text
        assert "tolerance_rationale" in skill_text
        assert '"evidence"' in skill_text
        assert "scientific_status" in skill_text
        assert "capability_supported" in skill_text
        assert "contradiction_failures" in skill_text
        assert "separat" in skill_text.lower()


def test_sandbox_manifest_advertises_the_installed_materials_stack():
    resources = sandbox_compute_resources(
        RuntimeSettings(
            openai_base_url="http://127.0.0.1:8003/v1",
            openai_model="deepseek_v4",
        )
    )

    assert {
        "pymatgen",
        "pymatgen-analysis-defects",
        "ase",
        "spglib",
        "pycalphad",
        "scheil",
        "damask",
        "matminer",
        "orix",
        "kikuchipy",
        "diffsims",
        "defdap",
        "porespy",
    }.issubset(resources["preinstalled_packages"])


def test_skill_json_examples_match_the_strict_canonical_schema():
    skill_paths = [
        _SKILLS_ROOT / "computational-materials" / "SKILL.md",
        _SKILLS_ROOT / "materials-structure-thermo" / "SKILL.md",
        _SKILLS_ROOT / "materials-characterization" / "SKILL.md",
    ]
    payloads = []
    for path in skill_paths:
        text = path.read_text(encoding="utf-8")
        blocks = re.findall(r"```json\s*(.*?)\s*```", text, flags=re.DOTALL)
        assert len(blocks) == 1, path
        payloads.append(json.loads(blocks[0]))

    # The compatibility skill illustrates one check; the two focused skills
    # illustrate complete assessment records.
    ValidationCheck.from_mapping(payloads[0])
    assert parse_assessment_record(payloads[1]).verified is True
    assert parse_assessment_record(payloads[2]).verified is True
