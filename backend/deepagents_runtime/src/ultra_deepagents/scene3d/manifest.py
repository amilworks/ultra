"""The ``ultra.scene3d.v1`` manifest (contract §6).

``limitations`` is the CIFTI honesty field: plain sentences, rendered verbatim in the
provenance panel, stating what the viewer is *not* doing. Everything the derive throws
away — empty SH bands, view-dependent bands it cannot carry, clamped colour, quantized
rotation, the poster's fake render — earns a sentence here. A limitation that is only
true sometimes is emitted only when it is true; a permanently-true one is unconditional.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "SCHEMA",
    "build_layer",
    "build_manifest",
    "camera_limitations",
    "infer_up_axis",
    "splat_limitations",
]

SCHEMA = "ultra.scene3d.v1"
_AXIS_NAMES = ("x", "y", "z")
# An axis is only called "up" when it is clearly the thin one. Aerial and corridor scans
# are wide-wide-thin; a roughly cubic scene has no such evidence and stays "unknown"
# rather than nudging the camera into a wrong-looking default.
_UP_AXIS_RATIO = 0.5


def infer_up_axis(bbox: list[float]) -> tuple[str, str]:
    """``(up_axis, up_axis_basis)`` from the world bounding box.

    A view hint for the camera controls only. It is never baked into geometry: rotating
    a node holding SH-bearing splats rotates the geometry but not the coefficients, and
    every odd band sign-flips (contract §2).
    """
    extents = [float(bbox[axis + 3] - bbox[axis]) for axis in range(3)]
    thin = int(np.argmin(extents))
    others = sorted(extents[axis] for axis in range(3) if axis != thin)
    if others[0] <= 0.0 or extents[thin] > _UP_AXIS_RATIO * others[0]:
        return "unknown", "unknown"
    if _AXIS_NAMES[thin] == "x":
        return "unknown", "unknown"  # x-up is not a convention any exporter uses
    return _AXIS_NAMES[thin], "heuristic"


def build_layer(
    *,
    layer_type: str,
    encoding: str,
    total: int,
    chunks: list[dict[str, Any]],
    tiers: list[list[int]],
    activation_domain: str,
    source_frame: str,
    quantization: dict[str, Any],
) -> dict[str, Any]:
    """One ``layers[]`` entry, field-for-field as the contract spells it."""
    return {
        "type": layer_type,
        "encoding": encoding,
        "total": int(total),
        "chunks": chunks,
        "tiers": [[int(index) for index in tier] for tier in tiers],
        "activation_domain": activation_domain,
        "source_frame": source_frame,
        "quantization": quantization,
    }


def _percent(fraction: float) -> str:
    """A percentage that never reads as 0% when something non-zero happened."""
    if fraction <= 0.0:
        return "0%"
    if fraction < 0.0001:
        return "<0.01%"
    return f"{fraction * 100.0:.2f}%"


def splat_limitations(
    *,
    declared_sh_degree: int,
    measured_sh_degree: int,
    sh_sample: int,
    clamped_color_fraction: float,
) -> list[str]:
    """The splat-specific honesty sentences."""
    sentences: list[str] = []
    if measured_sh_degree < declared_sh_degree:
        sentences.append(
            f"The source declares spherical-harmonic degree {declared_sh_degree}, but every "
            f"f_rest coefficient above degree {measured_sh_degree} is exactly zero in "
            f"{sh_sample:,} sampled splats, so this scene is served at its measured degree "
            f"{measured_sh_degree} and the empty bands are not transmitted."
        )
    if measured_sh_degree > 0:
        sentences.append(
            f"Only the degree-0 (DC) colour is transmitted. The {measured_sh_degree} "
            "view-dependent spherical-harmonic band(s) present in the source are dropped, so "
            "highlights do not shift as you orbit."
        )
    sentences.append(
        "Splat colour is served display-referred, as 0.5 + C0*f_dc, exactly as the reference "
        "3D Gaussian Splatting renderer stores it. No transfer function is applied in either "
        "the derive or the renderer."
    )
    sentences.append(
        f"{_percent(clamped_color_fraction)} of degree-0 colour components fell outside [0,1] "
        "after 0.5 + C0*f_dc and were clamped; that colour detail is not recoverable."
    )
    sentences.append(
        "Splat centres are exact float32. Scale and colour are stored as float16 (~0.05% "
        "relative) and rotation as a 10-10-12 octahedral quaternion (~0.1 degree)."
    )
    return sentences


def camera_limitations(
    *,
    camera_count: int,
    drawn_count: int,
    distorted_count: int,
    distorted_models: tuple[str, ...] = (),
    has_points: bool,
    has_rig_metadata: bool = False,
) -> list[str]:
    """The COLMAP camera-layer honesty sentences.

    ``camera_count`` is every registered image; ``drawn_count`` is how many of them the
    layer actually carries (an image whose ``camera_id`` has no calibration is dropped,
    never guessed at). ``distorted_count`` counts frusta drawn from a camera whose
    distortion coefficients are non-zero — the frustum is the pinhole part of the model
    only, so those edges are straight where the real lens curves them.
    """
    sentences: list[str] = []
    sentences.append(
        "Camera poses are served exactly as COLMAP stores them: a world-to-camera rotation as "
        "a (w, x, y, z) quaternion and its translation, in COLMAP's right-down-forward frame. "
        "The derive neither inverts nor re-orders them; the renderer performs the single "
        "inversion to a camera centre and the one right-up-back flip."
    )
    if distorted_count:
        models = ", ".join(distorted_models)
        detail = f" ({models})" if models else ""
        sentences.append(
            f"{distorted_count:,} of {drawn_count:,} camera frusta come from a model carrying "
            f"non-zero distortion coefficients{detail}, which the frusta do not apply. Each "
            "frustum is drawn from the pinhole part of its model — focal length and principal "
            "point — so its edges are straight where the real lens curves them."
        )
    if drawn_count < camera_count:
        sentences.append(
            f"{camera_count - drawn_count:,} of {camera_count:,} registered image(s) reference a "
            "camera_id the model does not calibrate, so no frustum is drawn for them."
        )
    if not has_points:
        sentences.append(
            "This model registers cameras but holds no 3D points, so the scene is the camera "
            "poses alone. Nothing else is drawn, and the bounds come from the camera centres."
        )
    sentences.append(
        "Per-image 2D feature observations and per-point tracks are read past and dropped. Only "
        "the 3D points, their colours, and the camera poses reach the viewer."
    )
    if has_rig_metadata:
        sentences.append(
            "This model carries rig/frame metadata (rigs, frames) that the viewer does not use; "
            "every pose is read from the images table."
        )
    return sentences


def build_manifest(
    *,
    resource_id: str,
    scene_kind: str,
    source_format: str,
    writer: str | None,
    vertex_count: int,
    source_bytes: int,
    declared_sh_degree: int,
    measured_sh_degree: int,
    stride_bytes: int,
    bbox: list[float],
    bbox_robust: list[float] | None = None,
    up_axis: str,
    up_axis_basis: str,
    layers: list[dict[str, Any]],
    limitations: list[str],
    units: str = "arbitrary",
) -> dict[str, Any]:
    """Assemble the §6 manifest.

    ``units`` stays ``"arbitrary"`` unless a user supplies a reference: a 3DGS scene's
    scale comes from COLMAP's arbitrary reconstruction scale, and labelling that "meters"
    would make every measurement drawn on top of it wrong by an unknown factor.
    """
    return {
        "schema": SCHEMA,
        "scene_kind": scene_kind,
        "source": {
            "format": source_format,
            "writer": writer,
            "vertex_count": int(vertex_count),
            "bytes": int(source_bytes),
            "declared_sh_degree": int(declared_sh_degree),
            "measured_sh_degree": int(measured_sh_degree),
            "stride_bytes": int(stride_bytes),
        },
        "world": {
            "units": units,
            "up_axis": up_axis,
            "up_axis_basis": up_axis_basis,
            "frame": "source",
            "bbox": [float(value) for value in bbox],
            # What a camera should frame on: outliers routinely make `bbox` two orders of
            # magnitude larger than the scene a viewer needs to show. Falls back to `bbox`
            # so consumers can read one field unconditionally.
            "bbox_robust": [float(value) for value in (bbox_robust or bbox)],
        },
        "layers": layers,
        "limitations": list(limitations),
        "service_urls": {
            "chunk": f"/v2/uploads/{resource_id}/scene3d/chunk/{{index}}",
            "download": f"/v2/resources/{resource_id}/download",
        },
    }
