"""Unit tests for the GoldGate freeze-input tooling (no dataset required)."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest
from PIL import Image

from ultra_deepagents.training.gt2_census import select_gt_layer
from ultra_deepagents.training.phash import PHASH_KERNEL, hamming, phash64
from ultra_deepagents.training.recovered_inventory import _quantiles, _tile_stem


def _gradient_image(seed: int) -> Image.Image:
    image = Image.new("RGB", (64, 64))
    image.putdata([((x * seed) % 256, (y * 7) % 256, ((x + y) * 3) % 256) for y in range(64) for x in range(64)])
    return image


def test_phash_is_deterministic_and_kernel_pinned() -> None:
    image = _gradient_image(5)
    first, second = phash64(image), phash64(image)
    assert first == second
    assert len(first) == 16
    int(first, 16)  # valid hex
    # The kernel id is part of the cross-time comparison contract: changing the
    # algorithm without minting a new id corrupts every stored hash comparison.
    assert PHASH_KERNEL == "phash64/goldgate-dct@1/gray-lanczos32-dct8x8-median-ex-dc"


def _stripes(vertical: bool) -> Image.Image:
    image = Image.new("L", (64, 64))
    image.putdata([255 if ((x if vertical else y) // 8) % 2 else 0 for y in range(64) for x in range(64)])
    return image


def test_phash_distinguishes_structurally_different_images() -> None:
    vertical, horizontal = _stripes(True), _stripes(False)
    assert hamming(phash64(vertical), phash64(vertical)) == 0
    # 90-degree structural difference: far beyond the defense-3 near-dup
    # thresholds (Hamming < 6 whole-frame, < 8 tile-level).
    assert hamming(phash64(vertical), phash64(horizontal)) > 16


def test_tile_stem_parsing() -> None:
    assert _tile_stem("EnrNE_Day2_Run1__0001_10_9.jpg") == "EnrNE_Day2_Run1__0001"
    assert _tile_stem("tile_106624_12768_0_11.jpg") == "tile_106624_12768"
    assert _tile_stem("odd-name.jpg") == "odd-name"


def test_quantiles_shape() -> None:
    quantiles = _quantiles([float(v) for v in range(1, 101)])
    assert quantiles["count"] == 100
    assert quantiles["min"] == 1.0
    assert quantiles["max"] == 100.0
    assert quantiles["q25"] < quantiles["median"] < quantiles["q75"]
    assert _quantiles([]) == {}


def _image_xml(layers: list[str]) -> ET.Element:
    inner = "".join(f'<gobject name="{name}"><gobject name="burrow"/></gobject>' for name in layers)
    return ET.fromstring(f"<image>{inner}</image>")


def test_gt_layer_selection_trims_and_prioritizes() -> None:
    name, _ = select_gt_layer(_image_xml([" test_today", " New Ground Truth", "gt2"]))
    assert name == "gt2"
    name, _ = select_gt_layer(_image_xml([" New Ground Truth"]))
    assert name == "New Ground Truth"


def test_gt_layer_selection_fails_closed() -> None:
    with pytest.raises(ValueError, match="no GT layer matches"):
        select_gt_layer(_image_xml([" test_today"]))
    with pytest.raises(ValueError, match="multiple GT layers"):
        select_gt_layer(_image_xml(["gt2", "gt2 "]))
