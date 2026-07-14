from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import shutil
import time
from pathlib import Path

import pytest
import ultra_deepagents.materials.calphad as calphad
from ultra_deepagents.materials import (
    CalphadExecutionError,
    CalphadInputError,
    CalphadTimeoutError,
    inspect_calphad_input,
    is_pycalphad_test_database,
    run_calphad_equilibrium,
)

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pycalphad") is None,
    reason="CALPHAD runtime tests execute in the pinned materials environment",
)

_TEST_ONLY_TDB = """$ TEST-ONLY synthetic ideal solution; not assessed scientific evidence
ELEMENT VA VACUUM 0 0 0 !
ELEMENT AL FCC_A1 26.9815 0 0 !
ELEMENT CU FCC_A1 63.546 0 0 !
ELEMENT NI FCC_A1 58.6934 0 0 !
TYPE_DEFINITION % SEQ * !
PHASE FCC_A1 % 1 1 !
CONSTITUENT FCC_A1 :AL,CU,NI,VA: !
PARAMETER G(FCC_A1,AL;0) 298.15 500; 6000 N !
PARAMETER G(FCC_A1,CU;0) 298.15 0; 6000 N !
PARAMETER G(FCC_A1,NI;0) 298.15 1000; 6000 N !
PARAMETER G(FCC_A1,VA;0) 298.15 100000; 6000 N !
"""

_SOURCE = "Ultra test-only synthetic TDB authored in test_calphad_runtime.py"
_LICENSE = "test-only synthetic input; not production scientific evidence"
_ASSESSMENT_SCOPE = "test-only ideal FCC solution over the declared temperature interval"
_REFERENCE_STATE = "test-only arbitrary elemental reference energies"


def _write_test_tdb(tmp_path: Path, *, name: str = "synthetic.tdb") -> Path:
    path = tmp_path / name
    path.write_text(_TEST_ONLY_TDB, encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inspect(path: Path, **kwargs):
    if path.is_file():
        kwargs.setdefault("artifact_id", "resource-test-synthetic-tdb")
        if "expected_sha256" not in kwargs and "expected_size_bytes" not in kwargs:
            kwargs["expected_sha256"] = _sha256(path)
            kwargs["expected_size_bytes"] = path.stat().st_size
        kwargs.setdefault("assessment_temperature_limits_K", [298.15, 6000.0])
        kwargs.setdefault("assessment_pressure_limits_Pa", [101325.0, 101325.0])
        kwargs.setdefault("assessment_scope", _ASSESSMENT_SCOPE)
        kwargs.setdefault("reference_state", _REFERENCE_STATE)
    return inspect_calphad_input(
        path,
        components=kwargs.pop("components", ["CU", "NI", "VA"]),
        phases=kwargs.pop("phases", ["FCC_A1"]),
        source=kwargs.pop("source", _SOURCE),
        license_id=kwargs.pop("license_id", _LICENSE),
        **kwargs,
    )


def _run(path: Path, **kwargs):
    kwargs.setdefault("artifact_id", "resource-test-synthetic-tdb")
    if "expected_sha256" not in kwargs and "expected_size_bytes" not in kwargs:
        kwargs["expected_sha256"] = _sha256(path)
        kwargs["expected_size_bytes"] = path.stat().st_size
    kwargs.setdefault("assessment_temperature_limits_K", [298.15, 6000.0])
    kwargs.setdefault("assessment_pressure_limits_Pa", [101325.0, 101325.0])
    kwargs.setdefault("assessment_scope", _ASSESSMENT_SCOPE)
    kwargs.setdefault("reference_state", _REFERENCE_STATE)
    return run_calphad_equilibrium(
        path,
        components=kwargs.pop("components", ["CU", "NI", "VA"]),
        phases=kwargs.pop("phases", ["FCC_A1"]),
        temperatures_K=kwargs.pop("temperatures_K", [1200.0, 1000.0]),
        pressures_Pa=kwargs.pop("pressures_Pa", 101325.0),
        total_amount_mol=kwargs.pop("total_amount_mol", 1.0),
        independent_compositions=kwargs.pop("independent_compositions", {"NI": [0.75, 0.25]}),
        source=kwargs.pop("source", _SOURCE),
        license_id=kwargs.pop("license_id", _LICENSE),
        **kwargs,
    )


def test_parser_uses_the_validated_database_format(monkeypatch: pytest.MonkeyPatch):
    from pycalphad import Database

    observed: list[tuple[str, str]] = []

    def fake_from_file(handle, *, fmt):
        observed.append((handle.read(), fmt))
        return "parsed"

    monkeypatch.setattr(Database, "from_file", staticmethod(fake_from_file))

    assert calphad._parse_database("TDB", suffix=".tdb") == "parsed"
    assert calphad._parse_database("CHEMSAGE", suffix=".dat") == "parsed"
    assert observed == [("TDB", "tdb"), ("CHEMSAGE", "dat")]
    with pytest.raises(CalphadInputError, match="unsupported parser format"):
        calphad._parse_database("UNKNOWN", suffix=".db")


def test_database_input_rejects_unregistered_db_suffix(tmp_path: Path):
    path = _write_test_tdb(tmp_path, name="synthetic.db")

    with pytest.raises(CalphadInputError, match="unsupported parser format"):
        _inspect(path)


def test_pinned_pycalphad_database_corpus_parses_all_registered_text_formats():
    import pycalphad

    database_root = Path(pycalphad.__file__).resolve().parent / "tests" / "databases"
    database_paths = sorted(
        path for path in database_root.iterdir() if path.suffix.casefold() in {".tdb", ".dat"}
    )
    if not database_paths:
        pytest.skip("the installed pycalphad wheel does not include its parser corpus")
    assert len(database_paths) == 45

    failures: list[str] = []
    for path in database_paths:
        try:
            text = path.read_text(encoding="utf-8-sig")
            calphad._parse_database(text, suffix=path.suffix)
        except Exception as exc:  # pragma: no cover - retained as a compact corpus diagnostic
            failures.append(f"{path.name}: {type(exc).__name__}: {exc}")
    assert failures == []


def test_dat_inspection_records_the_actual_parser_format(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pycalphad

    database_root = Path(pycalphad.__file__).resolve().parent / "tests" / "databases"
    source_paths = sorted(database_root.glob("*.dat"))
    if not source_paths:
        pytest.skip("the installed pycalphad wheel does not include a ChemSage fixture")
    assert len(source_paths) == 7
    monkeypatch.setattr(calphad, "_is_fixture_database", lambda *_args: False)

    for source_path in source_paths:
        path = tmp_path / source_path.name
        shutil.copyfile(source_path, path)
        manifest = inspect_calphad_input(
            path,
            source="test-only pycalphad ChemSage parser fixture",
            license_id="test-only parser fixture",
            artifact_id=f"resource-test-chemsage-{source_path.stem}",
            expected_sha256=_sha256(path),
            expected_size_bytes=path.stat().st_size,
            assessment_scope="test-only parser-format regression",
            reference_state="fixture-defined reference state",
            assessment_temperature_limits_K=[1.0, 10_000.0],
            assessment_pressure_limits_Pa=[101325.0, 101325.0],
        )

        assert manifest["format"] == "dat"


def test_inspection_emits_bounded_database_and_phase_model_manifest(tmp_path: Path):
    path = _write_test_tdb(tmp_path)

    manifest = _inspect(path)

    assert manifest["sha256"] == _sha256(path)
    assert manifest["size_bytes"] == path.stat().st_size
    assert manifest["source"] == _SOURCE
    assert manifest["license_id"] == _LICENSE
    assert manifest["assessment_scope"] == _ASSESSMENT_SCOPE
    assert manifest["reference_state"] == _REFERENCE_STATE
    assert manifest["components"] == ["AL", "CU", "NI", "VA"]
    assert manifest["physical_elements"] == ["AL", "CU", "NI"]
    assert manifest["vacancy_components"] == ["VA"]
    assert manifest["pseudo_elements"] == []
    assert manifest["species"] == ["AL", "CU", "NI", "VA"]
    assert manifest["phases"] == ["FCC_A1"]
    assert manifest["parameter_count"] == 4
    assert manifest["pycalphad_version"]
    assert manifest["assessment_temperature_limits_K"] == [298.15, 6000.0]
    assert manifest["assessment_pressure_limits_Pa"] == [101325.0, 101325.0]
    assert manifest["format"] == "tdb"
    assert len(manifest["manifest_sha256"]) == 64
    phase = manifest["phase_models"][0]
    assert phase == {
        "name": "FCC_A1",
        "sublattice_site_ratios": [1.0],
        "sublattices": [
            {
                "index": 0,
                "site_ratio": 1.0,
                "constituents": ["AL", "CU", "NI", "VA"],
            }
        ],
        "model_hints": {},
    }
    json.dumps(manifest, allow_nan=False, sort_keys=True)


def test_catalog_inspection_discovers_components_and_phases_without_solver_request(
    tmp_path: Path,
):
    path = _write_test_tdb(tmp_path)

    manifest = inspect_calphad_input(
        path,
        source=_SOURCE,
        license_id=_LICENSE,
        artifact_id="resource-discovery-test",
        expected_sha256=_sha256(path),
        expected_size_bytes=path.stat().st_size,
        assessment_temperature_limits_K=[298.15, 6000.0],
        assessment_pressure_limits_Pa=[101325.0, 101325.0],
        assessment_scope=_ASSESSMENT_SCOPE,
        reference_state=_REFERENCE_STATE,
    )

    assert manifest["requested_components"] == ["AL", "CU", "NI", "VA"]
    assert manifest["requested_phases"] == ["FCC_A1"]
    assert manifest["components"] == ["AL", "CU", "NI", "VA"]


def test_embedded_nist_manifest_directory_json_and_tdb_are_verified():
    root = Path(__file__).resolve().parents[1] / "materials_data" / "calphad"
    expected = "107c7330f0326a334742632f7494c7beadf53370edbc188df1a030853ceab5a8"

    directory_manifest = inspect_calphad_input(
        root,
        database_id="nist-al-co-w-wang-2017",
        components=["AL", "CO", "W", "VA"],
        phases=["FCC_A1", "LIQUID"],
    )
    json_manifest = inspect_calphad_input(
        root / "manifest.json",
        components=["AL", "CO", "W", "VA"],
        phases=["FCC_A1", "LIQUID"],
    )
    tdb_manifest = inspect_calphad_input(
        root / "alcow_CALPHAD-2017-Wang.tdb",
        components=["AL", "CO", "W", "VA"],
        phases=["FCC_A1", "LIQUID"],
    )

    for manifest in (directory_manifest, json_manifest, tdb_manifest):
        assert manifest["sha256"] == expected
        assert manifest["size_bytes"] == 21_274
        assert manifest["source"] == "https://materialsdata.nist.gov/handle/11256/948"
        assert manifest["license_id"] == "CC0-1.0"
        assert manifest["physical_elements"] == ["AL", "CO", "W"]
        assert manifest["vacancy_components"] == ["VA"]
        assert manifest["pseudo_elements"] == ["/-"]
        assert len(manifest["phases"]) == 18
        assert manifest["parameter_count"] == 174
        assert manifest["assessment_temperature_limits_K"] == [298.14, 6000.0]
        assert manifest["assessment_pressure_limits_Pa"] == [101325.0, 101325.0]
        assert manifest["registry_manifest"]["assessment_pressure_limits_Pa"] == [
            101325.0,
            101325.0,
        ]
        assert manifest["assessment_scope"].startswith("Critically reviewed Al-Co-W")
        assert manifest["reference_state"] == "Standard Element Reference (SER/GHSER)"
        assert manifest["registry_manifest"]["source_sha256"].startswith("da3ede")


def test_manifest_hash_size_and_path_traversal_fail_closed(tmp_path: Path):
    embedded = Path(__file__).resolve().parents[1] / "materials_data" / "calphad"
    copied = tmp_path / "catalog"
    shutil.copytree(embedded, copied)
    database = copied / "alcow_CALPHAD-2017-Wang.tdb"
    database.write_bytes(database.read_bytes() + b"\n")

    with pytest.raises(CalphadInputError, match="hash/size mismatch"):
        inspect_calphad_input(
            copied,
            components=["AL", "CO", "W", "VA"],
            phases=["LIQUID"],
        )

    manifest_path = copied / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["databases"][0]["filename"] = "../escape.tdb"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(CalphadInputError, match="filename must be a basename"):
        inspect_calphad_input(
            copied,
            components=["AL", "CO", "W", "VA"],
            phases=["LIQUID"],
        )


def test_self_authored_sibling_manifest_cannot_mint_verified_provenance(
    tmp_path: Path,
):
    path = _write_test_tdb(tmp_path)
    manifest = {
        "schema_version": "1",
        "databases": [
            {
                "database_id": "attacker-claimed-assessment",
                "filename": path.name,
                "format": "tdb",
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
                "source_uri": "https://attacker.example.test/fake-provenance",
                "license_id": "CC0-1.0",
                "assessment_scope": "attacker-written scope",
                "reference_state": "attacker-written reference state",
                "elements": ["AL", "CU", "NI", "VA"],
                "phases": ["FCC_A1"],
                "tdb_temperature_limits_K": [298.15, 6000.0],
                "assessment_pressure_limits_Pa": [101325.0, 101325.0],
                "caveats": [],
            }
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(CalphadInputError, match="not a release-trusted embedded registry"):
        inspect_calphad_input(
            path,
            components=["CU", "NI", "VA"],
            phases=["FCC_A1"],
        )


def test_source_license_and_catalog_binding_are_fail_closed(tmp_path: Path):
    path = _write_test_tdb(tmp_path)
    digest = _sha256(path)
    binding = {
        "artifact_id": "resource-123",
        "expected_sha256": digest,
        "expected_size_bytes": path.stat().st_size,
        "assessment_temperature_limits_K": [298.15, 6000.0],
        "assessment_pressure_limits_Pa": [101325.0, 101325.0],
        "assessment_scope": _ASSESSMENT_SCOPE,
        "reference_state": _REFERENCE_STATE,
    }

    with pytest.raises(CalphadInputError, match="source/provenance is required"):
        inspect_calphad_input(
            path,
            components=["CU", "NI", "VA"],
            phases=["FCC_A1"],
            license_id=_LICENSE,
            **binding,
        )
    with pytest.raises(CalphadInputError, match="license or use authorization is required"):
        inspect_calphad_input(
            path,
            components=["CU", "NI", "VA"],
            phases=["FCC_A1"],
            source=_SOURCE,
            **binding,
        )
    with pytest.raises(CalphadInputError, match="must be supplied together"):
        _inspect(path, artifact_id="resource-123", expected_sha256=digest)
    with pytest.raises(CalphadInputError, match="requires an artifact_id"):
        inspect_calphad_input(
            path,
            components=["CU", "NI", "VA"],
            phases=["FCC_A1"],
            source=_SOURCE,
            license_id=_LICENSE,
            expected_sha256=digest,
            expected_size_bytes=path.stat().st_size,
            assessment_scope=_ASSESSMENT_SCOPE,
            reference_state=_REFERENCE_STATE,
        )
    with pytest.raises(CalphadInputError, match="requires catalog-bound"):
        inspect_calphad_input(
            path,
            components=["CU", "NI", "VA"],
            phases=["FCC_A1"],
            source=_SOURCE,
            license_id=_LICENSE,
            artifact_id="resource-123",
            assessment_temperature_limits_K=[298.15, 6000.0],
            assessment_pressure_limits_Pa=[101325.0, 101325.0],
            assessment_scope=_ASSESSMENT_SCOPE,
            reference_state=_REFERENCE_STATE,
        )

    with pytest.raises(CalphadInputError, match="explicit usable declaration"):
        inspect_calphad_input(
            path,
            components=["CU", "NI", "VA"],
            phases=["FCC_A1"],
            source="unknown",
            license_id=_LICENSE,
            **binding,
        )
    with pytest.raises(CalphadInputError, match="assessment_temperature_limits_K"):
        inspect_calphad_input(
            path,
            components=["CU", "NI", "VA"],
            phases=["FCC_A1"],
            source=_SOURCE,
            license_id=_LICENSE,
            artifact_id="resource-123",
            expected_sha256=digest,
            expected_size_bytes=path.stat().st_size,
            assessment_scope=_ASSESSMENT_SCOPE,
            reference_state=_REFERENCE_STATE,
        )
    with pytest.raises(CalphadInputError, match="assessment_pressure_limits_Pa"):
        inspect_calphad_input(
            path,
            components=["CU", "NI", "VA"],
            phases=["FCC_A1"],
            source=_SOURCE,
            license_id=_LICENSE,
            artifact_id="resource-123",
            expected_sha256=digest,
            expected_size_bytes=path.stat().st_size,
            assessment_temperature_limits_K=[298.15, 6000.0],
            assessment_scope=_ASSESSMENT_SCOPE,
            reference_state=_REFERENCE_STATE,
        )

    with pytest.raises(CalphadInputError, match="assessment_scope is required"):
        inspect_calphad_input(
            path,
            components=["CU", "NI", "VA"],
            phases=["FCC_A1"],
            source=_SOURCE,
            license_id=_LICENSE,
            artifact_id="resource-123",
            expected_sha256=digest,
            expected_size_bytes=path.stat().st_size,
            assessment_temperature_limits_K=[298.15, 6000.0],
            assessment_pressure_limits_Pa=[101325.0, 101325.0],
            reference_state=_REFERENCE_STATE,
        )
    with pytest.raises(CalphadInputError, match="reference_state is required"):
        inspect_calphad_input(
            path,
            components=["CU", "NI", "VA"],
            phases=["FCC_A1"],
            source=_SOURCE,
            license_id=_LICENSE,
            artifact_id="resource-123",
            expected_sha256=digest,
            expected_size_bytes=path.stat().st_size,
            assessment_temperature_limits_K=[298.15, 6000.0],
            assessment_pressure_limits_Pa=[101325.0, 101325.0],
            assessment_scope=_ASSESSMENT_SCOPE,
        )
    with pytest.raises(CalphadInputError, match="catalog hash/size"):
        _inspect(
            path,
            artifact_id="resource-123",
            expected_sha256="0" * 64,
            expected_size_bytes=path.stat().st_size,
        )

    manifest = _inspect(
        path,
        artifact_id="resource-123",
        expected_sha256=digest,
        expected_size_bytes=path.stat().st_size,
    )
    assert manifest["artifact_id"] == "resource-123"


@pytest.mark.parametrize(
    "limits",
    [
        [101326.0, 101325.0],
        [0.0, 101325.0],
        [101325.0, 1e12 + 1.0],
        [101325.0, float("nan")],
    ],
)
def test_assessment_pressure_limits_are_finite_bounded_and_nondecreasing(
    tmp_path: Path,
    limits: list[float],
) -> None:
    path = _write_test_tdb(tmp_path)

    with pytest.raises(CalphadInputError, match="assessment_pressure_limits_Pa"):
        _inspect(path, assessment_pressure_limits_Pa=limits)


def test_package_fixture_and_byte_identical_copy_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import pycalphad

    fake_package = tmp_path / "fake_pycalphad"
    fake_init = fake_package / "__init__.py"
    fixture = fake_package / "tests" / "databases" / "fixture.tdb"
    fixture.parent.mkdir(parents=True)
    fake_init.write_text("", encoding="utf-8")
    fixture.write_text(_TEST_ONLY_TDB, encoding="utf-8")
    copied = tmp_path / "renamed-assessed-looking.tdb"
    copied.write_bytes(fixture.read_bytes())
    monkeypatch.setattr(pycalphad, "__file__", str(fake_init))
    calphad._installed_fixture_hashes.cache_clear()

    assert is_pycalphad_test_database(fixture)
    assert is_pycalphad_test_database(copied)
    with pytest.raises(CalphadInputError, match="test databases are fixtures"):
        _inspect(copied)


def test_symlink_nonregular_oversize_and_nonfinite_tdb_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = _write_test_tdb(tmp_path)
    symlink = tmp_path / "linked.tdb"
    symlink.symlink_to(path)
    with pytest.raises(CalphadInputError, match="symbolic link"):
        _inspect(symlink)
    with pytest.raises(CalphadInputError, match="manifest"):
        _inspect(tmp_path)

    monkeypatch.setattr(calphad, "MAX_DATABASE_BYTES", path.stat().st_size - 1)
    with pytest.raises(CalphadInputError, match="input limit"):
        _inspect(path)
    monkeypatch.setattr(calphad, "MAX_DATABASE_BYTES", 64 * 1024 * 1024)

    nonfinite = tmp_path / "nonfinite.tdb"
    nonfinite.write_text(_TEST_ONLY_TDB.replace("1000", "NAN", 1), encoding="utf-8")
    with pytest.raises(CalphadInputError, match="non-finite numeric token"):
        _inspect(nonfinite)


@pytest.mark.parametrize(
    ("limit_name", "limit_value", "message"),
    [
        ("MAX_DATABASE_ELEMENTS", 3, "element limit"),
        ("MAX_DATABASE_SPECIES", 3, "species limit"),
        ("MAX_DATABASE_PHASES", 0, "phase limit"),
        ("MAX_DATABASE_PARAMETERS", 3, "parameter limit"),
    ],
)
def test_database_domain_count_limits_are_enforced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    limit_name: str,
    limit_value: int,
    message: str,
):
    path = _write_test_tdb(tmp_path)
    monkeypatch.setattr(calphad, limit_name, limit_value)
    with pytest.raises(CalphadInputError, match=message):
        _inspect(path)


def test_equilibrium_returns_phase_compositions_mu_np_gm_and_canonical_evidence(
    tmp_path: Path,
):
    path = _write_test_tdb(tmp_path)

    first = _run(path, assessment_temperature_limits_K=[298.15, 6000.0])
    second = _run(path, assessment_temperature_limits_K=[298.15, 6000.0])
    reframed_binary = _run(
        path,
        assessment_temperature_limits_K=[298.15, 6000.0],
        temperatures_K=[1000.0],
        independent_compositions={"CU": [0.2, 0.6]},
    )

    assert first["schema_version"] == "ultra.calphad.equilibrium.v2"
    assert first["request"]["grid_points"] == 4
    assert first["request"]["dependent_component"] == "CU"
    assert reframed_binary["request"]["dependent_component"] == "CU"
    assert reframed_binary["request"]["conditions"]["independent_compositions"] == {
        "NI": {"values": [0.4, 0.8], "units": "mole_fraction"}
    }
    assert first["request"]["phase_selection"] == {
        "scope": "all_database_phases",
        "excluded_database_phases": [],
        "global_equilibrium_claim_supported": True,
    }
    assert first["result"]["point_count"] == 4
    assert first["result"]["units"] == {
        "T": "K",
        "P": "Pa",
        "N": "mol",
        "X": "mole_fraction",
        "phase_X": "mole_fraction",
        "bulk_composition_residual": "mole_fraction",
        "NP": "phase_amount_fraction_at_N_equals_1_mol",
        "GM": "J/mol",
        "MU": "J/mol",
        "gibbs_euler_residual": "J/mol",
    }
    assert first["evidence"]["canonical_serialization"] is True
    assert first["evidence"]["solver_replay_determinism_claimed"] is False
    assert first["evidence"]["sha256"] == second["evidence"]["sha256"]
    assert [point["conditions"]["T_K"] for point in first["result"]["points"]] == [
        1000.0,
        1000.0,
        1200.0,
        1200.0,
    ]
    for point in first["result"]["points"]:
        assert math.isfinite(point["GM_J_per_mol"])
        assert point["stable_phases"] == [
            {"name": "FCC_A1", "NP_phase_fraction": pytest.approx(1.0)}
        ]
        assert point["stable_phase_vertices"] == [
            {
                "vertex_index": 0,
                "phase": "FCC_A1",
                "NP_phase_fraction": pytest.approx(1.0),
                "composition_mole_fraction": {
                    "CU": pytest.approx(point["conditions"]["composition_mole_fraction"]["CU"]),
                    "NI": pytest.approx(point["conditions"]["composition_mole_fraction"]["NI"]),
                },
                "composition_sum": pytest.approx(1.0),
            }
        ]
        assert set(point["chemical_potentials_J_per_mol"]) == {"CU", "NI"}
        assert all(
            math.isfinite(value) for value in point["chemical_potentials_J_per_mol"].values()
        )
        assert point["gibbs_from_chemical_potentials_J_per_mol"] == pytest.approx(
            point["GM_J_per_mol"], abs=calphad.GIBBS_EULER_ABSOLUTE_TOLERANCE_J_PER_MOL
        )
        assert (
            point["gibbs_euler_residual_J_per_mol"]
            <= calphad.GIBBS_EULER_ABSOLUTE_TOLERANCE_J_PER_MOL
        )
        assert point["phase_fraction_sum"] == pytest.approx(1.0)
        assert sum(point["conditions"]["composition_mole_fraction"].values()) == pytest.approx(1.0)
        assert point["reconstructed_composition_mole_fraction"] == pytest.approx(
            point["conditions"]["composition_mole_fraction"]
        )
        assert point["maximum_bulk_composition_residual"] <= calphad.BULK_COMPOSITION_TOLERANCE
        assert set(point["bulk_composition_residual_by_component"]) == {"CU", "NI"}
    json.dumps(first, allow_nan=False, sort_keys=True)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperatures_K", float("nan")),
        ("pressures_Pa", float("inf")),
        ("total_amount_mol", float("-inf")),
        ("independent_compositions", {"NI": [float("nan")]}),
        ("wall_time_seconds", float("nan")),
    ],
)
def test_equilibrium_rejects_nonfinite_request_values(tmp_path: Path, field: str, value):
    path = _write_test_tdb(tmp_path)
    with pytest.raises(CalphadInputError, match="finite|non-finite"):
        _run(path, **{field: value})


def test_composition_closure_domain_subset_grid_and_temperature_bounds(
    tmp_path: Path,
):
    path = _write_test_tdb(tmp_path)
    with pytest.raises(CalphadInputError, match="exactly one fewer"):
        _run(path, independent_compositions={})
    with pytest.raises(CalphadInputError, match="mole-fraction closure"):
        _run(
            path,
            components=["AL", "CU", "NI", "VA"],
            independent_compositions={"AL": [0.8], "NI": [0.4]},
        )
    with pytest.raises(CalphadInputError, match="outside the supplied database"):
        _run(path, phases=["NOT_A_PHASE"])
    with pytest.raises(CalphadInputError, match="exceeding limit 3"):
        _run(path, max_grid_points=3)
    with pytest.raises(CalphadInputError, match="outside the declared"):
        _run(
            path,
            temperatures_K=[1000.0],
            assessment_temperature_limits_K=[1200.0, 2000.0],
        )
    with pytest.raises(CalphadInputError, match="pressure is outside the declared"):
        _run(
            path,
            pressures_Pa=[101324.0],
            assessment_pressure_limits_Pa=[101325.0, 101325.0],
        )
    with pytest.raises(CalphadInputError, match="exactly 1"):
        _run(path, total_amount_mol=2.0)


def test_result_size_and_wall_time_are_hard_limits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = _write_test_tdb(tmp_path)
    with pytest.raises(CalphadInputError, match="estimated pycalphad result"):
        _run(path, max_result_bytes=100)

    def _slow(*_args, **_kwargs):
        time.sleep(0.2)
        raise AssertionError("wall timer did not interrupt the solver")

    monkeypatch.setattr(calphad, "_calculate_equilibrium", _slow)
    with pytest.raises(CalphadTimeoutError, match="exceeded"):
        _run(path, wall_time_seconds=0.05)


def test_database_parse_has_an_independent_wall_time_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = _write_test_tdb(tmp_path)

    def _slow_parse(_text: str, *, suffix: str):
        assert suffix == ".tdb"
        time.sleep(0.2)
        raise AssertionError("database parse wall timer did not interrupt")

    monkeypatch.setattr(calphad, "DATABASE_PARSE_WALL_TIME_SECONDS", 0.05)
    monkeypatch.setattr(calphad, "_parse_database", _slow_parse)
    with pytest.raises(CalphadTimeoutError, match="database parse exceeded 0.05 seconds"):
        _inspect(path)


def test_nonfinite_solver_output_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = _write_test_tdb(tmp_path)
    real_calculate = calphad._calculate_equilibrium

    def _nonfinite(*args, **kwargs):
        dataset = real_calculate(*args, **kwargs)
        dataset["GM"].values.flat[0] = float("nan")
        return dataset

    monkeypatch.setattr(calphad, "_calculate_equilibrium", _nonfinite)
    with pytest.raises(CalphadExecutionError, match="non-finite GM"):
        _run(path)


@pytest.mark.parametrize(
    ("variable", "message"),
    [("MU", "non-finite MU"), ("X", "non-finite X")],
)
def test_nonfinite_thermodynamic_output_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
    message: str,
):
    path = _write_test_tdb(tmp_path)
    real_calculate = calphad._calculate_equilibrium

    def _nonfinite(*args, **kwargs):
        dataset = real_calculate(*args, **kwargs)
        dataset[variable].values.flat[0] = float("nan")
        return dataset

    monkeypatch.setattr(calphad, "_calculate_equilibrium", _nonfinite)
    with pytest.raises(CalphadExecutionError, match=message):
        _run(path)


def test_solver_component_coordinate_must_match_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = _write_test_tdb(tmp_path)
    real_calculate = calphad._calculate_equilibrium

    def _wrong_components(*args, **kwargs):
        dataset = real_calculate(*args, **kwargs)
        return dataset.assign_coords(component=["CU", "AL"])

    monkeypatch.setattr(calphad, "_calculate_equilibrium", _wrong_components)
    with pytest.raises(CalphadExecutionError, match="component coordinate"):
        _run(path)


def test_solver_phase_vertices_must_reconstruct_requested_bulk_composition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = _write_test_tdb(tmp_path)
    real_calculate = calphad._calculate_equilibrium

    def _wrong_phase_composition(*args, **kwargs):
        dataset = real_calculate(*args, **kwargs)
        dataset["X"].values[..., 0, :] = [1.0, 0.0]
        return dataset

    monkeypatch.setattr(calphad, "_calculate_equilibrium", _wrong_phase_composition)
    with pytest.raises(CalphadExecutionError, match="reconstruct the requested bulk"):
        _run(path)


def test_solver_chemical_potentials_must_satisfy_gibbs_euler_relation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = _write_test_tdb(tmp_path)
    real_calculate = calphad._calculate_equilibrium

    def _wrong_chemical_potential(*args, **kwargs):
        dataset = real_calculate(*args, **kwargs)
        dataset["MU"].values[..., 0] += 100.0
        return dataset

    monkeypatch.setattr(calphad, "_calculate_equilibrium", _wrong_chemical_potential)
    with pytest.raises(CalphadExecutionError, match="Euler residual"):
        _run(path)


def test_off_main_thread_execution_fails_closed(tmp_path: Path):
    path = _write_test_tdb(tmp_path)
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run, path)
        with pytest.raises(CalphadExecutionError, match="main thread"):
            future.result()
