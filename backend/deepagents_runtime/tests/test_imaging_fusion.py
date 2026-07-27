from __future__ import annotations

import numpy as np
import pytest
from ultra_deepagents.imaging import fusion


def test_parse_hex_color():
    assert fusion.parse_hex_color("#1e90ff") == pytest.approx((30 / 255, 144 / 255, 255 / 255))
    assert fusion.parse_hex_color("00ff66") == pytest.approx((0.0, 1.0, 102 / 255))
    assert fusion.parse_hex_color("#0f0") == pytest.approx((0.0, 1.0, 0.0))
    assert fusion.parse_hex_color("nope") is None
    assert fusion.parse_hex_color("") is None


def test_composite_channels_additive_blend():
    # Channel 0 bright on the left half, channel 1 bright on the right half;
    # they overlap in the middle column.
    arr = np.zeros((2, 1, 3), dtype="float32")
    arr[0, 0, 0] = 100.0  # ch0 only -> blue
    arr[0, 0, 1] = 100.0  # both     -> blue + green = cyan
    arr[1, 0, 1] = 100.0  # both
    arr[1, 0, 2] = 100.0  # ch1 only -> green
    blue = fusion.parse_hex_color("#0000ff")
    green = fusion.parse_hex_color("#00ff00")
    rgb = fusion.composite_channels(arr, [blue, green], np=np)
    assert rgb.shape == (1, 3, 3)
    assert rgb.dtype == np.uint8
    # left pixel: ch0 max only -> pure blue
    assert tuple(rgb[0, 0]) == (0, 0, 255)
    # middle pixel: ch0 and ch1 both max -> additive cyan
    assert tuple(rgb[0, 1]) == (0, 255, 255)
    # right pixel: ch1 max only -> pure green
    assert tuple(rgb[0, 2]) == (0, 255, 0)


def test_composite_channels_windows_each_channel_independently():
    # A dim channel (values ~10) and a bright channel (values ~1000). Per-channel
    # percentile windowing must make the dim channel visible, not crushed by the
    # bright channel's range.
    dim = np.linspace(0, 10, 100, dtype="float32").reshape(1, 10, 10)
    bright = np.linspace(0, 1000, 100, dtype="float32").reshape(1, 10, 10)
    arr = np.concatenate([dim, bright], axis=0)
    red = fusion.parse_hex_color("#ff0000")
    green = fusion.parse_hex_color("#00ff00")
    rgb = fusion.composite_channels(arr, [red, green], np=np)
    # The dim channel's brightest voxel should light its red channel strongly
    # (per-channel window), not stay near-black.
    assert rgb[..., 0].max() > 200
    assert rgb[..., 1].max() > 200


def test_composite_channels_explicit_window_overrides_percentile():
    arr = (np.ones((1, 2, 2), dtype="float32") * 50.0)
    white = (1.0, 1.0, 1.0)
    # Window [0,100]: value 50 -> 0.5 -> ~128.
    rgb = fusion.composite_channels(arr, [white], np=np, windows=[(0.0, 100.0)])
    assert abs(int(rgb[0, 0, 0]) - 128) <= 1


def test_convention_channel_colors_distinct():
    colors = [fusion.convention_channel_color(i) for i in range(3)]
    assert len(set(colors)) == 3
    # First channel (DAPI-like) leans blue.
    assert colors[0][2] > colors[0][0]


def test_parse_fusion_request_gates_and_aligns():
    from ultra_deepagents.imaging.service import _parse_fusion_request

    # Single channel -> no fusion (fast native path), channel still remapped (1-based).
    remap, colors = _parse_fusion_request("0", "#ff0000,#00ff00,#0000ff")
    assert remap == [1]
    assert colors is None

    # Colors already arrive in selected-channel order; sparse absolute channel
    # indices must not index into the compact color list a second time.
    remap, colors = _parse_fusion_request(
        "1,3,5",
        "#ff0000,#00ff00,#0000ff",
    )
    assert remap == [2, 4, 6]
    assert colors == [
        fusion.parse_hex_color("#ff0000"),
        fusion.parse_hex_color("#00ff00"),
        fusion.parse_hex_color("#0000ff"),
    ]

    # Colors absent -> no fusion even with multiple channels.
    remap, colors = _parse_fusion_request("0,1", None)
    assert remap == [1, 2]
    assert colors is None


@pytest.mark.parametrize(
    "channel_colors",
    [
        "#ff0000,#00ff00",
        "#ff0000,#00ff00,#0000ff,#ffffff",
    ],
)
def test_parse_fusion_request_rejects_mismatched_selected_color_count(channel_colors):
    from ultra_deepagents.imaging.service import _parse_fusion_request

    with pytest.raises(ValueError, match="channel colors.*selected channel count"):
        _parse_fusion_request("1,3,5", channel_colors)


def test_service_threads_channels_and_colors():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from ultra_deepagents.imaging.pool import InlineRunner
    from ultra_deepagents.imaging.service import create_app

    app = create_app(InlineRunner(prefer_real=False))
    client = TestClient(app)
    resp = client.get(
        "/slice",
        params={"path": "stub://multichannel", "z": 0, "channels": "0,1", "channel_colors": "#0000ff,#00ff00"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.content[:8] == b"\x89PNG\r\n\x1a\n"

    mismatch = client.get(
        "/slice",
        params={
            "path": "stub://multichannel",
            "z": 0,
            "channels": "0,1",
            "channel_colors": "#0000ff",
        },
    )
    assert mismatch.status_code == 422
