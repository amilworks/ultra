from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import mattools_runner_wrapper as runner_wrapper  # noqa: E402
import mattools_safe_parser as safe_parser  # noqa: E402


@dataclass(frozen=True)
class FakeElement:
    symbol: str

    def __post_init__(self) -> None:
        if self.symbol not in {"H", "Mg", "Ga", "N", "O"}:
            raise ValueError("unknown element")


def parser() -> safe_parser.SafeComplexDictParser:
    return safe_parser.SafeComplexDictParser(element_type=FakeElement)


def test_safe_parser_accepts_reviewed_materials_value_shapes() -> None:
    value = parser().parse(
        """{
            'energy': -1.25e-4,
            'matrix': array([[1.0, 2.0], [3.0, 4.0]], dtype=float128),
            'element': Element Mg,
            'changes': {Element('Ga'): -1, Element('O'): 1},
            'coordinates': {(0.0, 0.5, 1.0)},
            'empty': set(),
        }"""
    )

    assert value is not None
    assert value["element"] == FakeElement("Mg")
    assert value["changes"] == {FakeElement("Ga"): -1, FakeElement("O"): 1}
    assert value["coordinates"] == {(0.0, 0.5, 1.0)}
    assert value["empty"] == set()
    assert value["matrix"].dtype == np.dtype(np.float64)
    assert np.array_equal(value["matrix"], np.array([[1.0, 2.0], [3.0, 4.0]]))


@pytest.mark.parametrize(
    "candidate_output",
    [
        "{'probe': array(__import__('builtins').sum([19, 23]))}",
        "{'probe': array((1).__class__.__mro__)}",
        "{'probe': array([value for value in [1, 2]])}",
        "{'probe': (lambda: 42)()}",
        "{'probe': eval('40 + 2')}",
        "{'probe': np.load('/tmp/input.npy')}",
        "{'probe': array([1], dtype=object)}",
        "{'probe': array([[1], [2, 3]])}",
        "{'probe': 1e999}",
        "{'probe': float('nan')}",
        "{'probe': 1, 'probe': 2}",
        "{1: 'non-string top-level key'}",
    ],
)
def test_safe_parser_rejects_code_execution_ambiguous_and_nonfinite_values(
    candidate_output: str,
) -> None:
    assert parser().parse(candidate_output) is None


def test_safe_parser_rejects_side_effect_payload_without_touching_host(tmp_path: Path) -> None:
    sentinel = tmp_path / "must-not-exist"
    payload = (
        "{'probe': array(__import__('pathlib').Path("
        + repr(str(sentinel))
        + ").write_text('unsafe'))}"
    )

    assert parser().parse(payload) is None
    assert not sentinel.exists()


def test_safe_parser_rejects_bounded_resource_abuse(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(safe_parser, "MAX_INPUT_BYTES", 32)
    assert parser().parse("{'value': '" + ("x" * 40) + "'}") is None

    monkeypatch.setattr(safe_parser, "MAX_INPUT_BYTES", 4 * 1024 * 1024)
    deeply_nested = "{'value': " + ("[" * 70) + "0" + ("]" * 70) + "}"
    assert parser().parse(deeply_nested) is None


def test_runner_wrapper_uses_synthetic_safe_utils_and_never_imports_snapshot_utils(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "src"
    snapshot.mkdir()
    runner = snapshot / "result_analysis.py"
    upstream_utils = snapshot / "utils.py"
    runner.write_text("raise AssertionError('runner must not execute in this unit test')\n")
    upstream_utils.write_text("raise AssertionError('unsafe utils imported')\n")
    monkeypatch.setattr(runner_wrapper, "OFFICIAL_RUNNER_SHA256", runner_wrapper._sha256(runner))
    monkeypatch.setattr(
        runner_wrapper,
        "OFFICIAL_UNSAFE_UTILS_SHA256",
        runner_wrapper._sha256(upstream_utils),
    )
    monkeypatch.delitem(sys.modules, "utils", raising=False)

    try:
        resolved_snapshot, resolved_runner = runner_wrapper.install_safe_utils(
            snapshot,
            expected_runner_sha256=runner_wrapper._sha256(runner),
            expected_utils_sha256=runner_wrapper._sha256(upstream_utils),
        )
        synthetic = sys.modules["utils"]

        assert resolved_snapshot == snapshot.resolve()
        assert resolved_runner == runner.resolve()
        assert synthetic.ULTRA_SAFE_PARSER is True
        assert synthetic.ComplexDictParser is safe_parser.SafeComplexDictParser
        assert synthetic.__file__.endswith("#synthetic-safe-utils")
    finally:
        sys.modules.pop("utils", None)


def test_runner_wrapper_rejects_preimported_utils(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "src"
    snapshot.mkdir()
    runner = snapshot / "result_analysis.py"
    upstream_utils = snapshot / "utils.py"
    runner.write_text("pass\n")
    upstream_utils.write_text("pass\n")
    monkeypatch.setattr(runner_wrapper, "OFFICIAL_RUNNER_SHA256", runner_wrapper._sha256(runner))
    monkeypatch.setattr(
        runner_wrapper,
        "OFFICIAL_UNSAFE_UTILS_SHA256",
        runner_wrapper._sha256(upstream_utils),
    )
    spec = importlib.util.spec_from_loader("utils", loader=None)
    assert spec is not None
    prior = importlib.util.module_from_spec(spec)
    sys.modules["utils"] = prior
    try:
        with pytest.raises(runner_wrapper.RunnerWrapperError, match="already-imported"):
            runner_wrapper.install_safe_utils(
                snapshot,
                expected_runner_sha256=runner_wrapper._sha256(runner),
                expected_utils_sha256=runner_wrapper._sha256(upstream_utils),
            )
    finally:
        sys.modules.pop("utils", None)
