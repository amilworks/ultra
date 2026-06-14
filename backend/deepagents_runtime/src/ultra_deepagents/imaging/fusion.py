"""Additive multi-channel LUT compositing for fluorescence microscopy.

The display pipeline maps a multi-channel image to RGB by taking the first three
channels as literal R/G/B, which is wrong for fluorescence: a DAPI/GFP/RFP stack
is not red/green/blue, channels past three are dropped, and each channel is
windowed against the whole-image range so dim channels wash out.

``composite_channels`` instead reads each requested channel's raw values, windows
it independently (per-channel robust percentile, or an explicit window), tints it
with its lookup-table color, and sums the tinted channels — the standard additive
fluorescence overlay. Kept as a pure numpy function so it is unit-testable apart
from the libbioimage decode path.
"""
from __future__ import annotations

from typing import Any


def parse_hex_color(value: str) -> tuple[float, float, float] | None:
    """Parse ``#rrggbb``/``rrggbb`` (or ``#rgb``) to an (r, g, b) triple in 0..1."""
    text = str(value or "").strip().lstrip("#")
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    if len(text) != 6:
        return None
    try:
        r = int(text[0:2], 16)
        g = int(text[2:4], 16)
        b = int(text[4:6], 16)
    except ValueError:
        return None
    return (r / 255.0, g / 255.0, b / 255.0)


def composite_channels(
    arr: Any,
    colors: list[tuple[float, float, float]],
    *,
    np: Any,
    windows: list[tuple[float, float] | None] | None = None,
    percentile: tuple[float, float] = (1.0, 99.0),
) -> Any:
    """Additively composite a (C, H, W) array into an (H, W, 3) uint8 RGB image.

    Each channel ``c`` is windowed to [0, 1] — using ``windows[c]`` when given,
    otherwise its own ``percentile`` range so a dim channel stays visible and a
    hot pixel doesn't blow it out — then tinted by ``colors[c]`` (r,g,b in 0..1)
    and summed. Saturating sums clip to white, matching additive fluorescence.
    """
    a = arr
    if a.ndim == 2:
        a = a[np.newaxis, ...]
    if a.ndim != 3:
        raise ValueError(f"composite_channels expects (C,H,W), got ndim={a.ndim}")
    channel_count, height, width = a.shape
    out = np.zeros((height, width, 3), dtype="float32")
    usable = min(channel_count, len(colors))
    for c in range(usable):
        color = colors[c]
        if color is None:
            continue
        plane = a[c].astype("float32")
        window = windows[c] if windows is not None and c < len(windows) else None
        if window is not None:
            lo, hi = float(window[0]), float(window[1])
        else:
            finite = plane[np.isfinite(plane)]
            if finite.size == 0:
                continue
            lo = float(np.percentile(finite, percentile[0]))
            hi = float(np.percentile(finite, percentile[1]))
        span = hi - lo
        if span <= 0:
            span = 1.0
        norm = np.clip((plane - lo) / span, 0.0, 1.0)
        out[..., 0] += norm * float(color[0])
        out[..., 1] += norm * float(color[1])
        out[..., 2] += norm * float(color[2])
    return (np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype("uint8")


# Convention default LUT colors for fluorescence channels lacking an explicit
# color (first channel is typically a nuclear/DAPI stain shown blue).
CONVENTION_CHANNEL_HEX = ["#1e90ff", "#00ff66", "#ff3b3b", "#ff00ff", "#ffd400", "#00e5ff"]


def convention_channel_color(index: int) -> tuple[float, float, float]:
    hex_value = CONVENTION_CHANNEL_HEX[index % len(CONVENTION_CHANNEL_HEX)]
    parsed = parse_hex_color(hex_value)
    return parsed if parsed is not None else (1.0, 1.0, 1.0)
