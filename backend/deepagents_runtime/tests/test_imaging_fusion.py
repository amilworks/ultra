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
    arr = np.ones((1, 2, 2), dtype="float32") * 50.0
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

    # An explicit single-channel LUT is scientific identity and must not be discarded.
    remap, colors = _parse_fusion_request("0", "#ff0000")
    assert remap == [1]
    assert colors == [fusion.parse_hex_color("#ff0000")]

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


def test_parse_fusion_request_bounds_composite_channel_count():
    from ultra_deepagents.imaging.service import (
        MAX_COMPOSITE_CHANNELS,
        _parse_fusion_request,
    )

    sparse_indices = [0, 7, 42, 63, 128, 191, 258, 259]
    channels = ",".join(str(index) for index in sparse_indices)
    colors = ",".join("#ffffff" for _index in range(MAX_COMPOSITE_CHANNELS))
    remap, selected_colors = _parse_fusion_request(channels, colors)
    assert remap == [index + 1 for index in sparse_indices]
    assert selected_colors is not None
    assert len(selected_colors) == MAX_COMPOSITE_CHANNELS

    too_many_channels = f"{channels},200"
    too_many_colors = f"{colors},#ffffff"
    for requested_colors in (None, too_many_colors):
        with pytest.raises(
            ValueError,
            match=rf"channel selection supports at most {MAX_COMPOSITE_CHANNELS}",
        ):
            _parse_fusion_request(too_many_channels, requested_colors)


@pytest.mark.parametrize("channels", ["", ",", "0,", ",0", "-1", "0,0", "abc", "0,1.5"])
def test_parse_fusion_request_rejects_malformed_explicit_selection(channels):
    from ultra_deepagents.imaging.service import _parse_fusion_request

    with pytest.raises(ValueError, match="channel selection"):
        _parse_fusion_request(channels, None)


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


@pytest.mark.parametrize(
    "channel_colors",
    ["", "nope,#00ff00", "#fff,#00ff00", "#ff0000,", "#ff0000"],
)
def test_service_rejects_invalid_channel_colors_before_localization(monkeypatch, channel_colors):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from ultra_deepagents.imaging import service as imaging_service
    from ultra_deepagents.imaging.pool import InlineRunner

    localization_calls: list[str] = []

    async def record_localization(path: str) -> str:
        localization_calls.append(path)
        return path

    monkeypatch.setattr(imaging_service, "_localize_pyramid_async", record_localization)
    client = TestClient(imaging_service.create_app(InlineRunner(prefer_real=False)))

    response = client.get(
        "/slice",
        params={
            "path": "stub://multichannel-2",
            "channels": "0,1",
            "channel_colors": channel_colors,
        },
    )

    assert response.status_code == 422
    assert localization_calls == []


def test_service_threads_channels_and_colors():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from ultra_deepagents.imaging.pool import InlineRunner
    from ultra_deepagents.imaging.service import create_app

    app = create_app(InlineRunner(prefer_real=False))
    client = TestClient(app)
    resp = client.get(
        "/slice",
        params={
            "path": "stub://multichannel-2",
            "z": 0,
            "channels": "0,1",
            "channel_colors": "#0000ff,#00ff00",
        },
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.content[:8] == b"\x89PNG\r\n\x1a\n"

    mismatch = client.get(
        "/slice",
        params={
            "path": "stub://multichannel-2",
            "z": 0,
            "channels": "0,1",
            "channel_colors": "#0000ff",
        },
    )
    assert mismatch.status_code == 422


@pytest.mark.parametrize("route", ["/slice", "/tile"])
@pytest.mark.parametrize("include_colors", [False, True])
def test_service_rejects_over_limit_before_localization(monkeypatch, route, include_colors):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from ultra_deepagents.imaging import service as imaging_service
    from ultra_deepagents.imaging.pool import InlineRunner

    localization_calls: list[str] = []

    async def record_localization(path: str) -> str:
        localization_calls.append(path)
        return path

    monkeypatch.setattr(imaging_service, "_localize_pyramid_async", record_localization)
    app = imaging_service.create_app(InlineRunner(prefer_real=False))
    client = TestClient(app)
    params = {
        "path": "stub://multichannel-2",
        "z": 0,
        "channels": ",".join(str(index) for index in range(9)),
    }
    if include_colors:
        params["channel_colors"] = ",".join("#ffffff" for _index in range(9))

    over_limit = client.get(route, params=params)
    assert over_limit.status_code == 422
    assert "at most 8 channels" in over_limit.json()["detail"]
    assert localization_calls == []


@pytest.mark.parametrize("route", ["/slice", "/tile"])
@pytest.mark.parametrize("channels", ["", ",", "-1", "0,0", "bad"])
def test_service_rejects_invalid_selection_before_localization_or_runner(
    monkeypatch, route, channels
):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from ultra_deepagents.imaging import service as imaging_service

    calls: list[tuple[str, tuple, dict]] = []

    class RecordingRunner:
        workers = 1

        async def call(self, operation, *args, **kwargs):
            calls.append((operation, args, kwargs))
            raise AssertionError("runner must not be called for an invalid channel selection")

    localization_calls: list[str] = []

    async def record_localization(path: str) -> str:
        localization_calls.append(path)
        return path

    monkeypatch.setattr(imaging_service, "_localize_pyramid_async", record_localization)
    client = TestClient(imaging_service.create_app(RecordingRunner()))

    response = client.get(route, params={"path": "stub://multichannel-2", "channels": channels})

    assert response.status_code == 422
    assert localization_calls == []
    assert calls == []


@pytest.mark.parametrize(
    ("route", "selector"),
    [
        ("/slice", "channels"),
        ("/slice", "t"),
        ("/tile", "z"),
        ("/tile", "channel_colors"),
        ("/atlas", "t"),
    ],
)
def test_service_rejects_repeated_scientific_selectors_before_localization(
    monkeypatch, route, selector
):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from ultra_deepagents.imaging import service as imaging_service
    from ultra_deepagents.imaging.pool import InlineRunner

    localized: list[str] = []

    async def record_localization(path: str) -> str:
        localized.append(path)
        return path

    monkeypatch.setattr(imaging_service, "_localize_pyramid_async", record_localization)
    client = TestClient(imaging_service.create_app(InlineRunner(prefer_real=False)))
    params = [("path", "stub://multichannel-2"), (selector, "0"), (selector, "1")]

    response = client.get(route, params=params)

    assert response.status_code == 422
    assert localized == []


@pytest.mark.parametrize("workers", [1, 2])
def test_service_threads_atlas_time_through_sequential_and_parallel_paths(monkeypatch, workers):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from ultra_deepagents.imaging import atlas as atlas_module
    from ultra_deepagents.imaging import service as imaging_service

    calls: list[tuple[str, dict]] = []

    class RecordingRunner:
        def __init__(self):
            self.workers = workers

        async def call(self, operation, *_args, **kwargs):
            calls.append((operation, kwargs))
            if operation == "meta":
                return {"image_num_c": 8}
            if operation == "atlas":
                return b"\x89PNG\r\n\x1a\n"
            raise AssertionError(operation)

    async def fake_assemble(_runner, _path, **kwargs):
        calls.append(("assemble_atlas", kwargs))
        return b"\x89PNG\r\n\x1a\n"

    monkeypatch.setattr(atlas_module, "assemble_atlas", fake_assemble)
    client = TestClient(imaging_service.create_app(RecordingRunner()))

    response = client.get(
        "/atlas",
        params={
            "path": "/source.tif",
            "t": "3",
            "channels": "5",
            "channel_colors": "#00ff00",
        },
    )

    assert response.status_code == 200
    operation = "atlas" if workers == 1 else "assemble_atlas"
    forwarded = next(kwargs for name, kwargs in calls if name == operation)
    assert forwarded["t"] == 3
    assert forwarded["channels"] == [6]
    assert forwarded["colors"] == [fusion.parse_hex_color("#00ff00")]


def test_service_threads_native_tile_time_and_z_identity():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from ultra_deepagents.imaging import service as imaging_service

    calls: list[tuple[str, dict]] = []

    class RecordingRunner:
        workers = 1

        async def call(self, operation, *_args, **kwargs):
            calls.append((operation, kwargs))
            if operation == "meta":
                return {"image_num_c": 8}
            if operation == "tile":
                return b"\x89PNG\r\n\x1a\n"
            raise AssertionError(operation)

    client = TestClient(imaging_service.create_app(RecordingRunner()))
    response = client.get(
        "/tile",
        params={
            "path": "/source.tif",
            "t": "2",
            "z": "4",
            "channels": "5",
            "channel_colors": "#0000ff",
        },
    )

    assert response.status_code == 200
    forwarded = next(kwargs for name, kwargs in calls if name == "tile")
    assert forwarded["t"] == 2
    assert forwarded["z"] == 4
    assert forwarded["channels"] == [6]
    assert forwarded["colors"] == [fusion.parse_hex_color("#0000ff")]


def test_decoder_default_channel_policy_is_bounded_and_preserves_native_rgb():
    from ultra_deepagents.imaging.bioio_engine import _zero_based
    from ultra_deepagents.imaging.engine import LibBioImageEngine

    assert _zero_based(None, 260) == list(range(260))
    with pytest.raises(ValueError, match="at most 8 channels"):
        _zero_based(list(range(1, 10)), 260)

    class FakeBinding:
        metadata = {
            "image_num_c": 260,
            "image_pixel_depth": 16,
            "image_pixel_format": "unsigned integer",
        }

        @classmethod
        def meta(cls, _path, _cache):
            return cls.metadata

    engine = LibBioImageEngine.__new__(LibBioImageEngine)
    engine._bim = FakeBinding
    engine._cache = object()
    assert engine._bounded_default_channels("hyperspectral.tif", None) == [1]

    FakeBinding.metadata = {
        "image_num_c": 1,
        "image_pixel_depth": 16,
        "image_pixel_format": "unsigned integer",
    }
    assert engine._bounded_default_channels("single-channel.tif", None) is None

    FakeBinding.metadata = {
        "image_num_c": 3,
        "image_pixel_depth": 8,
        "image_pixel_format": "unsigned integer",
        "image_mode": "RGB",
    }
    assert engine._bounded_default_channels("photo.tif", None) is None


@pytest.mark.parametrize("size", [0, -1, 1025])
def test_libbio_service_rejects_unbounded_tile_size_before_localization(monkeypatch, size):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from ultra_deepagents.imaging import service as imaging_service
    from ultra_deepagents.imaging.pool import InlineRunner

    localization_calls: list[str] = []

    async def record_localization(path: str) -> str:
        localization_calls.append(path)
        return path

    monkeypatch.setattr(imaging_service, "_localize_pyramid_async", record_localization)
    client = TestClient(imaging_service.create_app(InlineRunner(prefer_real=False)))

    response = client.get("/tile", params={"path": "stub://image", "size": size})

    assert response.status_code == 422
    assert localization_calls == []


def test_libbio_service_range_checks_explicit_channels_before_pixel_operation(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from ultra_deepagents.imaging import service as imaging_service

    operations: list[str] = []

    class RecordingRunner:
        workers = 1

        async def call(self, operation, *_args, **_kwargs):
            operations.append(operation)
            if operation == "meta":
                return {"image_num_c": 2}
            raise AssertionError("pixel operation must not run for an out-of-range channel")

    localized: list[str] = []

    async def record_localization(path: str) -> str:
        localized.append(path)
        return "/localized/source.tif"

    monkeypatch.setattr(imaging_service, "_localize_pyramid_async", record_localization)
    client = TestClient(imaging_service.create_app(RecordingRunner()))

    response = client.get(
        "/slice",
        params={"path": "/source.tif", "channels": "259"},
    )

    assert response.status_code == 422
    assert localized == ["/source.tif"]
    assert operations == ["meta"]
