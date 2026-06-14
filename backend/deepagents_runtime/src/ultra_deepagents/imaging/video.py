"""Video poster + probe via ffmpeg/ffprobe.

libbioimage parses video *containers* but its FFmpeg decoder is not functional in
our build, so video frames come from the system ``ffmpeg``. ffmpeg runs as a
bounded subprocess (``asyncio.create_subprocess_exec``) — never through the
libbioimage process pool — so a long extraction can't starve image decodes.

The command builders are pure and unit-tested. The async runners shell out and
are exercised where ffmpeg is installed (the image-service container).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

__all__ = [
    "FFMPEG",
    "FFPROBE",
    "poster_command",
    "probe_command",
    "poster_seek_seconds",
    "extract_poster",
    "probe_info",
    "VideoError",
]

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"


class VideoError(RuntimeError):
    """ffmpeg/ffprobe failed (non-zero exit, timeout, or empty output)."""


def poster_seek_seconds(duration: float, requested: float = 1.0) -> float:
    """Pick a poster timestamp that works for both long and very short clips.

    Long videos seek to ``requested`` (default 1s — a fast `-ss` keyframe seek,
    past any black intro frame). Clips shorter than ``requested`` seek to ~10% in
    so we still land on a real frame rather than the end.
    """
    req = max(0.0, float(requested))
    dur = float(duration) if duration and duration > 0 else 0.0
    if dur and req >= dur:
        return max(0.0, round(dur * 0.1, 3))
    return req


def poster_command(
    path: str, *, time_seconds: float = 1.0, max_size: int = 512, ffmpeg: str = FFMPEG
) -> list[str]:
    """ffmpeg argv: one frame at ``time_seconds``, scaled to fit ``max_size`` (keep
    aspect ratio, never upscale), encoded as PNG to stdout.

    ``-ss`` *before* ``-i`` is an input seek — O(1)-ish for long videos (jumps to
    the nearest keyframe) instead of decoding from the start.
    """
    if not isinstance(max_size, int) or isinstance(max_size, bool) or max_size <= 0:
        raise ValueError(f"max_size must be a positive int, got {max_size!r}")
    t = max(0.0, float(time_seconds))
    scale = (
        f"scale='min({max_size},iw)':'min({max_size},ih)':"
        "force_original_aspect_ratio=decrease"
    )
    return [
        ffmpeg, "-nostdin", "-loglevel", "error",
        "-ss", f"{t:.3f}", "-i", str(path),
        "-frames:v", "1", "-vf", scale,
        "-f", "image2", "-c:v", "png", "pipe:1",
    ]


def probe_command(path: str, *, ffprobe: str = FFPROBE) -> list[str]:
    """ffprobe argv: container/stream info as JSON on stdout."""
    return [
        ffprobe, "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]


async def _run(argv: list[str], *, timeout: float) -> tuple[int, bytes, bytes]:
    proc = await asyncio.create_subprocess_exec(
        *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except (asyncio.TimeoutError, TimeoutError) as exc:
        try:
            proc.kill()
        finally:
            await proc.wait()
        raise VideoError(f"{argv[0]} timed out after {timeout}s") from exc
    return proc.returncode or 0, out, err


def _parse_probe(stdout: bytes) -> dict[str, Any]:
    data = json.loads(stdout or b"{}")
    fmt = data.get("format") or {}
    streams = data.get("streams") or []
    vstream = next((s for s in streams if s.get("codec_type") == "video"), {})
    try:
        duration = float(fmt.get("duration") or vstream.get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    return {
        "duration": duration,
        "width": int(vstream.get("width") or 0),
        "height": int(vstream.get("height") or 0),
        "codec": vstream.get("codec_name"),
        "format": fmt.get("format_name"),
        "has_video": bool(vstream),
    }


async def probe_info(path: str, *, timeout: float = 10.0, ffprobe: str = FFPROBE) -> dict[str, Any]:
    code, out, err = await _run(probe_command(path, ffprobe=ffprobe), timeout=timeout)
    if code != 0:
        raise VideoError(f"ffprobe failed ({code}): {err.decode('utf-8', 'replace')[:200]}")
    return _parse_probe(out)


async def extract_poster(
    path: str,
    *,
    time_seconds: float = 1.0,
    max_size: int = 512,
    timeout: float = 20.0,
    ffmpeg: str = FFMPEG,
    ffprobe: str = FFPROBE,
) -> bytes:
    """Return PNG bytes of a poster frame, clamping the seek for short clips."""
    seek = time_seconds
    try:
        info = await probe_info(path, timeout=min(timeout, 10.0), ffprobe=ffprobe)
        seek = poster_seek_seconds(info.get("duration", 0.0), time_seconds)
    except VideoError:
        seek = 0.0  # probe failed; grab the first frame
    code, out, err = await _run(
        poster_command(path, time_seconds=seek, max_size=max_size, ffmpeg=ffmpeg), timeout=timeout
    )
    if code != 0 or not out:
        raise VideoError(f"ffmpeg poster failed ({code}): {err.decode('utf-8', 'replace')[:200]}")
    return out
