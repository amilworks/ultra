"""Shared scientific-imaging constants and owned-artifact identities."""

from __future__ import annotations

import os
import re

MAX_COMPOSITE_CHANNELS = 8
MAX_ATLAS_GRID_EDGE = 4096
MAX_ATLAS_CELLS = 65536
MAX_TILE_EDGE = 1024
MAX_VIEWERINFO_SIGNAL_CHANNELS = 8

_OWNED_PYRAMID_NAME = re.compile(r"^.+__pyramid(?:\.sha256-[0-9a-f]{64})?\.tif$")


def is_ultra_owned_pyramid(path: object) -> bool:
    """Whether *path* has an exact Ultra-owned derivative pyramid identity.

    The parent-directory check prevents arbitrary TIFFs whose names merely end
    in ``__pyramid.tif`` from bypassing the semantic TIFF decoder.  Both the
    historical stable name and strict content-addressed publication name are
    accepted because either can still be present during rolling upgrades.
    """

    if not isinstance(path, (str, os.PathLike)):
        return False
    normalized = os.path.normpath(os.fspath(path))
    return (
        os.path.basename(os.path.dirname(normalized)) == "derived"
        and _OWNED_PYRAMID_NAME.fullmatch(os.path.basename(normalized)) is not None
    )
