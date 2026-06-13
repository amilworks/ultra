"""The benchmark harness runs and reports for every op (validated with the stub)."""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

pytest.importorskip("PIL")

from ultra_deepagents.imaging.benchmark import default_ops, format_table, run_benchmark
from ultra_deepagents.imaging.engine import StubEngine


def test_benchmark_runs_all_ops():
    engine = StubEngine()
    report = run_benchmark(engine, "a.czi", iterations=3)
    assert report["engine"] == "StubEngine"
    ops = set(default_ops(engine, "a.czi"))
    measured = {r["op"] for r in report["results"]}
    assert measured == ops
    for r in report["results"]:
        assert r["ok"] is True
        assert r["iterations"] == 3
        assert not math.isnan(r["ms_median"])


def test_format_table_is_text():
    report = run_benchmark(StubEngine(), "a.czi", iterations=2)
    table = format_table(report)
    assert "median_ms" in table
    assert "thumb_zscrub" in table
