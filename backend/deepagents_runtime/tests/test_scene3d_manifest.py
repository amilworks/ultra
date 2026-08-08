"""The ``ultra.scene3d.v1`` document: exact schema, and honest limitation sentences."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from ultra_deepagents.scene3d import manifest


def _layer():
    return manifest.build_layer(
        layer_type="splats",
        encoding="usx-v1",
        total=128340,
        chunks=[
            {
                "index": 0,
                "count": 128340,
                "bytes": 4106944,
                "origin": [0.0, 0.0, 0.0],
                "bbox": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            }
        ],
        tiers=[[0]],
        activation_domain="post",
        source_frame="source",
        quantization={
            "center": "f32-exact",
            "scale": "half-log",
            "rotation": "oct-10-10-12",
            "color": "half-display-referred",
            "out_of_range_color_fraction": 0.0031,
        },
    )


def test_manifest_has_exactly_the_contract_keys():
    document = manifest.build_manifest(
        resource_id="file-9",
        scene_kind="splat",
        source_format="ply",
        writer="postshot",
        vertex_count=14469103,
        source_bytes=3414709820,
        declared_sh_degree=3,
        measured_sh_degree=0,
        stride_bytes=236,
        bbox=[0.0, 0.0, 0.0, 110.8, 25.7, 111.5],
        up_axis="y",
        up_axis_basis="heuristic",
        layers=[_layer()],
        limitations=["something honest"],
    )

    assert set(document) == {
        "schema",
        "generator_revision",
        "scene_kind",
        "source",
        "world",
        "layers",
        "limitations",
        "service_urls",
    }
    assert set(document["source"]) == {
        "format",
        "writer",
        "vertex_count",
        "bytes",
        "declared_sh_degree",
        "measured_sh_degree",
        "stride_bytes",
    }
    assert set(document["world"]) == {
        "units",
        "up_axis",
        "up_axis_basis",
        "frame",
        "bbox",
        "bbox_robust",
    }
    assert set(document["layers"][0]) == {
        "type",
        "encoding",
        "total",
        "chunks",
        "tiers",
        "activation_domain",
        "source_frame",
        "quantization",
    }
    # Serializable with no NaN/Infinity, which JSON.parse would choke on.
    json.dumps(document, allow_nan=False)


def test_units_are_never_silently_called_meters():
    document = manifest.build_manifest(
        resource_id="f",
        scene_kind="splat",
        source_format="ply",
        writer=None,
        vertex_count=1,
        source_bytes=1,
        declared_sh_degree=0,
        measured_sh_degree=0,
        stride_bytes=236,
        bbox=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        up_axis="unknown",
        up_axis_basis="unknown",
        layers=[_layer()],
        limitations=[],
    )

    assert document["world"]["units"] == "arbitrary"
    assert document["world"]["frame"] == "source"
    assert document["source"]["writer"] is None  # not guessed


@pytest.mark.parametrize(
    ("bbox", "expected"),
    [
        # The measured drone scene: 110.8 x 25.7 x 111.5 -> y is clearly the thin axis.
        ([0, 0, 0, 110.8, 25.7, 111.5], ("y", "heuristic")),
        # The measured corridor: 449.7 x 112.6 x 1119.2 -> y again.
        ([0, 0, 0, 449.7, 112.6, 1119.2], ("y", "heuristic")),
        # A z-up scan.
        ([0, 0, 0, 100.0, 90.0, 4.0], ("z", "heuristic")),
        # Roughly cubic: no evidence, so no claim.
        ([0, 0, 0, 10.0, 9.0, 11.0], ("unknown", "unknown")),
        # x-thin is not a convention any exporter uses, so it is not asserted.
        ([0, 0, 0, 2.0, 40.0, 40.0], ("unknown", "unknown")),
        # Degenerate (planar) input must not divide by zero.
        ([0, 0, 0, 0.0, 0.0, 0.0], ("unknown", "unknown")),
    ],
)
def test_up_axis_is_a_heuristic_that_declines_to_guess(bbox, expected):
    assert manifest.infer_up_axis(bbox) == expected


def test_limitations_name_the_sample_size_behind_the_measured_degree():
    sentences = manifest.splat_limitations(
        declared_sh_degree=3,
        measured_sh_degree=0,
        sh_sample=200000,
        out_of_range_color_fraction=0.0031,
    )

    joined = " ".join(sentences)
    assert "declares spherical-harmonic degree 3" in joined
    assert "200,000 sampled splats" in joined
    assert "0.31% of degree-0 colour components" in joined
    # The colour sentence must state the display-referred convention, because a reader
    # who assumes linear will double-convert and render the whole scene too dark.
    assert "display-referred" in joined
    assert "No transfer function is applied" in joined


def test_a_tiny_but_non_zero_out_of_range_fraction_never_reads_as_zero_percent():
    """Rounding an honest 0.004% to '0.00%' would turn a disclosure into a denial."""
    sentences = manifest.splat_limitations(
        declared_sh_degree=0,
        measured_sh_degree=0,
        sh_sample=10,
        out_of_range_color_fraction=4e-5,
    )

    assert "<0.01% of degree-0 colour components" in " ".join(sentences)


def test_no_sh_sentence_when_nothing_was_over_declared_or_dropped():
    sentences = manifest.splat_limitations(
        declared_sh_degree=0,
        measured_sh_degree=0,
        sh_sample=200000,
        out_of_range_color_fraction=0.0,
    )

    joined = " ".join(sentences)
    assert "spherical-harmonic degree" not in joined
    assert "view-dependent" not in joined
    assert "0% of degree-0 colour components" in joined


def test_wide_position_source_reports_conversion_instead_of_claiming_exact_float32():
    sentences = manifest.splat_limitations(
        declared_sh_degree=0,
        measured_sh_degree=0,
        sh_sample=10,
        out_of_range_color_fraction=0.0,
        position_error=0.125,
    )

    joined = " ".join(sentences)
    assert "wider than float32" in joined
    assert "0.125 source units" in joined
    assert "centres are exact float32" not in joined
