"""Adversarial resource-bound and fidelity tests for the OME-NGFF viewer."""

from __future__ import annotations

import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    import numpy as np
    import zarr
    from PIL import Image

    _HAVE_DEPS = True
except Exception:
    _HAVE_DEPS = False

if "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST"):
    import pytest

    pytestmark = pytest.mark.skipif(not _HAVE_DEPS, reason="zarr/numpy/Pillow not installed")


def _yx_multiscales() -> list[dict[str, object]]:
    return [
        {
            "version": "0.4",
            "name": "image",
            "axes": [
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"},
            ],
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0]}],
                }
            ],
        }
    ]


def _write_lazy_yx(
    path: str,
    height: int,
    width: int,
    *,
    chunks: tuple[int, int] = (256, 256),
    dtype: object = "uint16",
) -> None:
    group = zarr.open_group(path, mode="w", zarr_format=2)
    group.create_array("0", shape=(height, width), chunks=chunks, dtype=dtype, fill_value=0)
    group.attrs["multiscales"] = _yx_multiscales()


def _write_uncompressed_yx(path: str, height: int, width: int) -> None:
    group = zarr.open_group(path, mode="w", zarr_format=2)
    array = group.create_array(
        "0",
        shape=(height, width),
        chunks=(height, width),
        dtype="uint8",
        compressors=None,
    )
    array[:] = 7
    group.attrs["multiscales"] = _yx_multiscales()


def _only_chunk_path(store: str) -> str:
    candidates: list[str] = []
    for root, _directories, files in os.walk(os.path.join(store, "0")):
        for name in files:
            if name not in (".zarray", ".zattrs", "zarr.json"):
                candidates.append(os.path.join(root, name))
    assert len(candidates) == 1
    return candidates[0]


@pytest.mark.parametrize("read_kind", ["plane", "region"])
def test_symlinked_chunk_is_refused_before_pixel_decode(tmp_path, read_kind):
    from ultra_deepagents.ngff.reader import NgffError, open_ngff

    secret = tmp_path / "outside.bin"
    secret.write_bytes(b"\xab" * 64)
    store = str(tmp_path / "attack.ome.zarr")
    _write_uncompressed_yx(store, 8, 8)
    chunk = _only_chunk_path(store)
    os.remove(chunk)
    try:
        os.symlink(secret, chunk)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    image = open_ngff(store)
    with pytest.raises(NgffError, match="symbolic link"):
        if read_kind == "plane":
            image.read_plane(level=0)
        else:
            image.read_region(level=0, y0=0, y1=8, x0=0, x1=8)


def test_chunk_dependency_enumeration_is_bounded(tmp_path, monkeypatch):
    from ultra_deepagents.ngff import reader as reader_module
    from ultra_deepagents.ngff.reader import NgffError, open_ngff

    monkeypatch.setattr(reader_module, "_MAX_CHUNK_FILES_PER_READ", 4)
    store = str(tmp_path / "many-chunks.ome.zarr")
    _write_lazy_yx(store, 8, 8, chunks=(1, 1), dtype="uint8")

    with pytest.raises(NgffError, match="chunk-file budget"):
        open_ngff(store).read_plane(level=0)


def test_full_plane_read_is_bounded_but_tile_region_remains_available(tmp_path, monkeypatch):
    from ultra_deepagents.ngff import reader as reader_module
    from ultra_deepagents.ngff.reader import NgffError, open_ngff

    monkeypatch.setattr(reader_module, "_MAX_PLANE_READ_ELEMENTS", 1_000_000)
    store = str(tmp_path / "large.ome.zarr")
    _write_lazy_yx(store, 2048, 2048)
    image = open_ngff(store)

    with pytest.raises(NgffError, match="full-plane read budget"):
        image.read_plane(level=0)
    region = image.read_region(level=0, y0=0, y1=256, x0=0, x1=256)
    assert region.shape == (256, 256)


def test_full_plane_read_has_an_independent_source_byte_budget(tmp_path, monkeypatch):
    from ultra_deepagents.ngff import reader as reader_module
    from ultra_deepagents.ngff.reader import NgffError, open_ngff

    monkeypatch.setattr(reader_module, "_MAX_PLANE_READ_ELEMENTS", 1_000_000)
    monkeypatch.setattr(reader_module, "_MAX_PLANE_READ_BYTES", 1024)
    store = str(tmp_path / "wide-dtype.ome.zarr")
    _write_lazy_yx(store, 32, 32, chunks=(16, 16), dtype="float64")

    with pytest.raises(NgffError, match="source-byte budget"):
        open_ngff(store).read_plane(level=0)


def test_multichannel_full_plane_render_has_a_working_set_budget(tmp_path, monkeypatch):
    from ultra_deepagents.ngff import render as render_module
    from ultra_deepagents.ngff.reader import NgffError, open_ngff

    store = str(tmp_path / "channels.ome.zarr")
    group = zarr.open_group(store, mode="w", zarr_format=2)
    group.create_array("0", shape=(3, 64, 64), chunks=(1, 32, 32), dtype="uint16")
    group.attrs["multiscales"] = [
        {
            "version": "0.4",
            "name": "channels",
            "axes": [
                {"name": "c", "type": "channel"},
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"},
            ],
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0, 1.0]}],
                }
            ],
        }
    ]
    monkeypatch.setattr(render_module, "_MAX_FULL_RENDER_WORKING_BYTES", 1024)

    with pytest.raises(NgffError, match="render working-set budget"):
        render_module.render_slice_png(open_ngff(store), channels=[0, 1, 2])


def test_chunk_decode_footprint_is_validated_at_open(tmp_path, monkeypatch):
    from ultra_deepagents.ngff import reader as reader_module
    from ultra_deepagents.ngff.reader import NgffError, open_ngff

    monkeypatch.setattr(reader_module, "_MAX_CHUNK_DECODED_BYTES", 1024)
    store = str(tmp_path / "oversized-chunk.ome.zarr")
    _write_lazy_yx(store, 64, 64, chunks=(64, 64), dtype="uint16")

    with pytest.raises(NgffError, match="decoded chunk budget"):
        open_ngff(store)


def test_direct_tile_renderer_rejects_oversized_tiles_before_storage_read(tmp_path):
    from ultra_deepagents.ngff.reader import NgffError, open_ngff
    from ultra_deepagents.ngff.render import render_tile_png

    store = str(tmp_path / "tile.ome.zarr")
    _write_lazy_yx(store, 4096, 4096)

    with pytest.raises(NgffError, match="tile size"):
        render_tile_png(open_ngff(store), level=0, col=0, row=0, tile_size=1_000_000)


def test_root_attributes_symbolic_link_is_rejected(tmp_path):
    from ultra_deepagents.ngff.reader import NgffError, open_ngff

    store = tmp_path / "linked-metadata.ome.zarr"
    _write_lazy_yx(str(store), 8, 8, chunks=(8, 8))
    attributes = store / ".zattrs"
    outside = tmp_path / "outside.json"
    outside.write_bytes(attributes.read_bytes())
    attributes.unlink()
    try:
        attributes.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    with pytest.raises(NgffError, match=r"metadata.*symbolic link"):
        open_ngff(str(store))


def test_deeply_nested_root_metadata_is_rejected_as_ngff_error(tmp_path):
    from ultra_deepagents.ngff.reader import NgffError, open_ngff

    store = tmp_path / "deep.ome.zarr"
    _write_lazy_yx(str(store), 8, 8, chunks=(8, 8))
    prefix = (
        '{"multiscales":[{"version":"0.4","name":"image","axes":['
        '{"name":"y","type":"space"},{"name":"x","type":"space"}],'
        '"datasets":[{"path":"0","coordinateTransformations":['
        '{"type":"scale","scale":[1.0,1.0]}]}]}],"nested":'
    )
    depth = 1000
    (store / ".zattrs").write_text(prefix + '{"value":' * depth + "0" + "}" * depth + "}")

    with pytest.raises(NgffError, match="nesting"):
        open_ngff(str(store))


def test_oversized_root_metadata_is_rejected_before_json_decode(tmp_path, monkeypatch):
    from ultra_deepagents.ngff import reader as reader_module
    from ultra_deepagents.ngff.reader import NgffError, open_ngff

    store = tmp_path / "oversized.ome.zarr"
    _write_lazy_yx(str(store), 8, 8, chunks=(8, 8))
    monkeypatch.setattr(reader_module, "_MAX_METADATA_JSON_BYTES", 32)

    with pytest.raises(NgffError, match="metadata byte budget"):
        open_ngff(str(store))


@pytest.mark.parametrize(
    "dtype",
    ["complex64", np.dtype([("real", "float32"), ("imag", "float32")])],
)
def test_non_scalar_pixel_dtypes_are_rejected_before_render(tmp_path, dtype):
    from ultra_deepagents.ngff.reader import NgffError, open_ngff

    store = str(tmp_path / f"unsupported-{np.dtype(dtype).kind}.ome.zarr")
    _write_lazy_yx(store, 8, 8, chunks=(8, 8), dtype=dtype)

    with pytest.raises(NgffError, match="unsupported pixel dtype"):
        open_ngff(store)


def test_valid_bounded_store_still_renders(tmp_path):
    from ultra_deepagents.ngff.reader import open_ngff
    from ultra_deepagents.ngff.render import render_slice_png, render_tile_png

    store = str(tmp_path / "valid.ome.zarr")
    _write_uncompressed_yx(store, 16, 16)
    image = open_ngff(store)

    assert image.read_plane().shape == (16, 16)
    assert Image.open(io.BytesIO(render_slice_png(image))).size == (16, 16)
    assert Image.open(io.BytesIO(render_tile_png(image, level=0, col=0, row=0))).size == (
        16,
        16,
    )
