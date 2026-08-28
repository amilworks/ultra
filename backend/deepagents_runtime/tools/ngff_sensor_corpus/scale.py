"""Scale probes: stores that are enormous in declared shape but cheap on disk.

Most of these are ``lazy_fill`` — the arrays are declared at gigapixel / long-timeseries /
deep-volume sizes but no chunks are written, so on disk they are only metadata. That lets
the harness ask the honest question — *does one request try to materialise the whole
thing?* — without needing terabytes of scratch. The reader sees the true (huge) geometry.
"""

from __future__ import annotations

from .specs import StoreSpec

__all__ = ["scale_probes"]


def scale_probes() -> list[StoreSpec]:
    return [
        StoreSpec(
            domain="scale",
            modality="gigapixel_pyramid",
            title="Gigapixel whole-slide pyramid (lazy)",
            instrument="synthetic (declared 16k x 16k, 6-level pyramid)",
            axes=(("y", "micrometer"), ("x", "micrometer")),
            dtype="uint16",
            base={"y": 16384, "x": 16384},
            signal="gradient",
            scale={"y": 0.25, "x": 0.25},
            levels=6,
            value_range=(0, 65535),
            emit_omero=False,
            lazy_fill=True,
            notes="Probes bounded /tile reads + tile_scheme on a 268-megapixel base plane.",
        ),
        StoreSpec(
            domain="scale",
            modality="gigapixel_single_level",
            title="Gigapixel single-level plane (lazy, no pyramid)",
            instrument="synthetic (declared 12k x 12k, one level)",
            axes=(("y", "micrometer"), ("x", "micrometer")),
            dtype="uint16",
            base={"y": 12000, "x": 12000},
            signal="gradient",
            scale={"y": 0.25, "x": 0.25},
            levels=1,
            value_range=(0, 65535),
            emit_omero=False,
            lazy_fill=True,
            notes="No pyramid: probes whether a full-plane /slice materialises 288 MB.",
        ),
        StoreSpec(
            domain="scale",
            modality="long_timeseries",
            title="Very long time series (lazy)",
            instrument="synthetic (t=5000)",
            axes=(("t", "second"), ("y", "micrometer"), ("x", "micrometer")),
            dtype="uint16",
            base={"t": 5000, "y": 128, "x": 128},
            signal="gradient",
            scale={"t": 0.1, "y": 1.0, "x": 1.0},
            levels=1,
            value_range=(0, 65535),
            emit_omero=False,
            lazy_fill=True,
            notes="Probes is_timeseries + t-index validation + metadata size at t=5000.",
        ),
        StoreSpec(
            domain="scale",
            modality="deep_zstack",
            title="Deep z-volume (lazy)",
            instrument="synthetic (z=1024)",
            axes=(("z", "micrometer"), ("y", "micrometer"), ("x", "micrometer")),
            dtype="uint16",
            base={"z": 1024, "y": 256, "x": 256},
            signal="gradient",
            scale={"z": 1.0, "y": 1.0, "x": 1.0},
            levels=1,
            value_range=(0, 65535),
            emit_omero=False,
            lazy_fill=True,
            notes="Probes z_index default (z//2=512) + volume_mode + thumbnail level pick.",
        ),
        StoreSpec(
            domain="scale",
            modality="many_channels",
            title="256-channel spectral stack",
            instrument="synthetic (c=256)",
            axes=(("c", "nanometer"), ("y", "micrometer"), ("x", "micrometer")),
            dtype="uint16",
            base={"c": 256, "y": 64, "x": 64},
            signal="spectral_bands",
            scale={"c": 1.0, "y": 1.0, "x": 1.0},
            levels=1,
            value_range=(0, 4000),
            emit_omero=False,
            notes="Probes channel_names length + compositing cost at c=256 (real pixels).",
        ),
        StoreSpec(
            domain="scale",
            modality="many_levels",
            title="Deep multiscale pyramid (10 levels)",
            instrument="synthetic (4096 base, 10 levels)",
            axes=(("y", "micrometer"), ("x", "micrometer")),
            dtype="uint8",
            base={"y": 4096, "x": 4096},
            signal="gradient",
            scale={"y": 1.0, "x": 1.0},
            levels=10,
            value_range=(0, 255),
            emit_omero=False,
            lazy_fill=True,
            notes="Probes level-cap handling + thumbnail_level selection across 10 levels.",
        ),
    ]
