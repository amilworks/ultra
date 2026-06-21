"""bioio transcode fallback for formats libbioimage cannot decode.

libbioimage registers many readers that are non-functional in our build (e.g. the
Leica ``.lif`` reader — in the format list, but ``imgcnv -meta`` returns "Input
format is not supported"). When the engine reports no geometry for a source, this
module converts it to an OME-TIFF via ``bioio`` (pure-Python plugins such as
``bioio-lif``/``readlif`` — no Java) so the normal convert-once-to-pyramid path can
serve it.

bioio is used ONLY here, in the batch convert lane, off the serving hot path: the
derived pyramid (not bioio) serves every tile/slice/thumbnail afterwards. This keeps
libbioimage the single serving spine while extending ingest coverage.

Multi-series sources (a LIF often bundles overviews + the real acquisitions across
many series) convert the LARGEST series by pixel count; the full series roster is
returned so callers can surface "series N of M".
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from typing import Any

__all__ = ["TranscodeResult", "TranscodeError", "transcode_to_ome_tiff"]

# Extensions bioio plugins (and libbioimage) commonly handle; used to recover from
# series/page-suffixed upload names (e.g. "scan.lif_15") that defeat bioio's
# extension-based plugin detection. Covers the installed bioio reader plugins
# (czi/dv/lif/nd2/sldy/...) plus other microscopy containers libbioimage knows.
_BIOIO_EXTS = frozenset(
    {
        "lif", "czi", "nd2", "dv", "r3d", "mrc", "sldy", "zarr",
        "lsm", "oib", "oif", "vsi", "scn", "svs", "ndpi", "ims", "zvi", "sld", "ipl",
    }
)


def _clean_extension_alias(src: str) -> str | None:
    """If ``src`` has a series/page-suffixed name (``…\\.lif_15``) whose base extension
    is a known bioio format, return a temp symlink path with the clean extension so
    bioio's extension-based plugin detection works. The caller removes the temp dir.
    Returns None when no such recovery applies."""
    base = os.path.basename(src).lower()
    match = re.search(r"\.([a-z0-9]+)_\d+$", base)  # ".lif_15" -> "lif"
    if not match or match.group(1) not in _BIOIO_EXTS:
        return None
    ext = match.group(1)
    tmpdir = tempfile.mkdtemp(prefix="ultra_transcode_")
    link = os.path.join(tmpdir, f"input.{ext}")
    target = os.path.abspath(src)
    try:
        os.symlink(target, link)
    except OSError:
        try:
            os.link(target, link)  # symlinks unsupported (e.g. some mounts) -> hardlink
        except OSError:
            shutil.rmtree(tmpdir, ignore_errors=True)
            return None
    return link


class TranscodeError(RuntimeError):
    """A source could not be read/transcoded by bioio (no plugin, unreadable, empty)."""


@dataclass
class TranscodeResult:
    path: str
    series_count: int
    series_index: int
    series_name: str
    num_c: int
    num_z: int
    dtype: str
    series_names: list[str]

    @property
    def is_multichannel_or_volume(self) -> bool:
        """Whether the derived pyramid must be OME-BigTIFF (to keep channels + z
        planes) rather than plain BigTIFF (which would flatten them to one plane)."""
        return self.num_c > 1 or self.num_z > 1


def _largest_scene_index(img: Any) -> int:
    """Pick the scene/series with the most pixels — the data of interest in a
    multi-series acquisition. Falls back to 0 if shapes can't be read."""
    import numpy as np

    best_index, best_pixels = 0, -1
    for index in range(len(img.scenes)):
        try:
            img.set_scene(index)
            pixels = int(np.prod([int(n) for n in img.dims.shape]))
        except Exception:  # noqa: BLE001 - a flaky scene must not abort selection
            continue
        if pixels > best_pixels:
            best_index, best_pixels = index, pixels
    return best_index


def transcode_to_ome_tiff(src: str, dst: str, *, prefer: str = "largest") -> TranscodeResult:
    """Read ``src`` with bioio and write its (largest) series to ``dst`` as OME-TIFF.

    Raises :class:`TranscodeError` if bioio is unavailable or cannot read the source —
    the caller then records a permanent derivation failure and the viewer degrades to
    the "preview unavailable" card.
    """
    try:
        from bioio import BioImage
    except Exception as exc:  # noqa: BLE001 - missing dep is a transcode failure, not a crash
        raise TranscodeError(f"bioio is not installed: {exc!r}") from exc
    import tifffile

    alias_dir: str | None = None
    try:
        img = BioImage(src)
    except Exception as first_exc:  # noqa: BLE001 - unreadable / no matching plugin
        # bioio detects the plugin by extension; a series-suffixed name ("scan.lif_15")
        # has no recognized extension, so retry via a clean-extension temp symlink.
        alias = _clean_extension_alias(src)
        if alias is None:
            raise TranscodeError(f"bioio could not open {src!r}: {first_exc!r}") from first_exc
        alias_dir = os.path.dirname(alias)
        try:
            img = BioImage(alias)
        except Exception as exc:  # noqa: BLE001
            shutil.rmtree(alias_dir, ignore_errors=True)
            raise TranscodeError(f"bioio could not open {src!r} (via {alias!r}): {exc!r}") from exc

    try:
        try:
            series_names = [str(name) for name in img.scenes]
        except Exception:  # noqa: BLE001
            series_names = []
        count = len(series_names) or 1
        index = _largest_scene_index(img) if (prefer == "largest" and count > 1) else 0
        try:
            img.set_scene(index)
            name = str(img.current_scene)
            data = img.get_image_data("TCZYX")  # always 5D, contiguous
        except Exception as exc:  # noqa: BLE001
            raise TranscodeError(f"bioio could not read series {index} of {src!r}: {exc!r}") from exc

        if data is None or getattr(data, "size", 0) == 0:
            raise TranscodeError(f"bioio returned an empty array for {src!r} series {index}")

        num_c = int(data.shape[1])
        num_z = int(data.shape[2])

        metadata: dict[str, Any] = {"axes": "TCZYX"}
        try:
            spacing = img.physical_pixel_sizes  # (Z, Y, X) in microns, any may be None
            if spacing is not None:
                if getattr(spacing, "X", None):
                    metadata["PhysicalSizeX"] = float(spacing.X)
                    metadata["PhysicalSizeXUnit"] = "µm"
                if getattr(spacing, "Y", None):
                    metadata["PhysicalSizeY"] = float(spacing.Y)
                    metadata["PhysicalSizeYUnit"] = "µm"
                if getattr(spacing, "Z", None):
                    metadata["PhysicalSizeZ"] = float(spacing.Z)
                    metadata["PhysicalSizeZUnit"] = "µm"
        except Exception:  # noqa: BLE001 - spacing is best-effort metadata
            pass
        try:
            names = img.channel_names
            if names:
                metadata["Channel"] = {"Name": [str(n) for n in names]}
        except Exception:  # noqa: BLE001 - channel names are best-effort metadata
            pass

        try:
            tifffile.imwrite(dst, data, ome=True, bigtiff=True, metadata=metadata)
        except Exception as exc:  # noqa: BLE001
            raise TranscodeError(f"failed writing OME-TIFF {dst!r}: {exc!r}") from exc

        return TranscodeResult(
            path=dst,
            series_count=count,
            series_index=index,
            series_name=name,
            num_c=num_c,
            num_z=num_z,
            dtype=str(getattr(data, "dtype", "")),
            series_names=series_names,
        )
    finally:
        if alias_dir:
            shutil.rmtree(alias_dir, ignore_errors=True)
