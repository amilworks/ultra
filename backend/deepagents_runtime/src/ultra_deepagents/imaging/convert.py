"""Derived-artifact conversion: source image -> tiled pyramidal OME/BigTIFF.

The 50GB serving story is "convert once, read bounded": a non-tiled source is
converted a single time into a tiled, pyramidal artifact that the image service
then serves via bounded tile/region reads (see the engine-understanding doc).
This module builds and runs that conversion with the ``imgcnv`` CLI — the
canonical way to write a tiled TIFF pyramid (``-options "... tiles N pyramid
subdirs"``).

The command builders are pure and unit-tested; the runner shells out to the
engine CLI and is exercised where ``imgcnv`` is installed. The eventual
``image.derive_pyramid`` NATS job calls :func:`derive_pyramid`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass

__all__ = [
    "PyramidSpec",
    "ConvertResult",
    "pyramid_options",
    "convert_command",
    "derive_pyramid",
    "DEFAULT_IMGCNV",
]

DEFAULT_IMGCNV = "imgcnv"

_COMPRESSIONS = frozenset({"none", "packbits", "lzw", "jpeg", "zip", "lzma", "jxr"})
_LAYOUTS = frozenset({"subdirs", "topdirs"})


@dataclass(frozen=True)
class PyramidSpec:
    """How to write the derived tiled pyramid.

    ``layout='topdirs'`` writes each pyramid level as its own top-level TIFF page,
    which keeps the native (level 0) tiles addressable via ``-tile SZ,XID,YID,0``.
    The ``subdirs`` SubIFD layout (COG style) leaves the base level unreadable via
    the embedded-tile reader in this engine build, so ``topdirs`` is the serving default.
    """

    tile_size: int = 512
    compression: str = "lzw"
    layout: str = "topdirs"
    fmt: str = "bigtiff"

    def options(self) -> str:
        return pyramid_options(self.tile_size, self.compression, self.layout)


@dataclass
class ConvertResult:
    src: str
    dst: str
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def pyramid_options(tile_size: int = 512, compression: str = "lzw", layout: str = "topdirs") -> str:
    """Build the ``imgcnv -options`` string for a tiled pyramid write."""
    if not isinstance(tile_size, int) or isinstance(tile_size, bool) or tile_size <= 0:
        raise ValueError(f"tile_size must be a positive int, got {tile_size!r}")
    if compression not in _COMPRESSIONS:
        raise ValueError(f"unknown compression {compression!r}; expected one of {sorted(_COMPRESSIONS)}")
    if layout not in _LAYOUTS:
        raise ValueError(f"layout must be one of {sorted(_LAYOUTS)}, got {layout!r}")
    return f"compression {compression} tiles {tile_size} pyramid {layout}"


def convert_command(src, dst, *, spec: PyramidSpec | None = None, imgcnv_bin: str = DEFAULT_IMGCNV) -> list[str]:
    """Build the ``imgcnv`` argv that converts ``src`` to a tiled pyramid at ``dst``."""
    spec = spec or PyramidSpec()
    return [imgcnv_bin, "-i", str(src), "-o", str(dst), "-t", spec.fmt, "-options", spec.options()]


def derive_pyramid(
    src,
    dst,
    *,
    spec: PyramidSpec | None = None,
    imgcnv_bin: str = DEFAULT_IMGCNV,
    timeout: float | None = None,
) -> ConvertResult:
    """Convert ``src`` into a tiled pyramidal artifact at ``dst``.

    Raises :class:`RuntimeError` if the conversion exits non-zero. This is the
    one-time, memory-heavy step; run it as a background job, not in the serving
    path.
    """
    resolved = shutil.which(imgcnv_bin) or imgcnv_bin
    # Ensure the destination directory exists — imgcnv won't create it and fails
    # with "Cannot write into" otherwise (e.g. a fresh derived/ on first upload).
    parent = os.path.dirname(str(dst))
    if parent:
        os.makedirs(parent, exist_ok=True)
    cmd = convert_command(src, dst, spec=spec, imgcnv_bin=resolved)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    result = ConvertResult(str(src), str(dst), proc.returncode, proc.stdout, proc.stderr)
    if not result.ok:
        raise RuntimeError(
            f"imgcnv conversion failed (exit {proc.returncode}): {proc.stderr.strip()[:500]}"
        )
    return result
