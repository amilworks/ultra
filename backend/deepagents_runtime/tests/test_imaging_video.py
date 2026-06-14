"""Unit tests for the video ffmpeg command builders + poster-seek logic (pure)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from ultra_deepagents.imaging import video


def test_poster_command_uses_input_seek_and_bounded_scale():
    cmd = video.poster_command("/data/clip.mp4", time_seconds=1.5, max_size=512, ffmpeg="ffmpeg")
    # -ss before -i is the fast input seek (jumps to a keyframe, not decode-from-start).
    assert cmd.index("-ss") < cmd.index("-i")
    assert cmd[cmd.index("-ss") + 1] == "1.500"
    assert cmd[cmd.index("-i") + 1] == "/data/clip.mp4"
    assert cmd[-3:] == ["-c:v", "png", "pipe:1"]
    vf = cmd[cmd.index("-vf") + 1]
    assert "force_original_aspect_ratio=decrease" in vf and "min(512,iw)" in vf
    assert "-frames:v" in cmd and cmd[cmd.index("-frames:v") + 1] == "1"


def test_poster_command_rejects_bad_max_size():
    for bad in (0, -10, True):
        with pytest.raises(ValueError):
            video.poster_command("/a.mp4", max_size=bad)


def test_probe_command_is_json_streams_and_format():
    cmd = video.probe_command("/data/clip.mov", ffprobe="ffprobe")
    assert cmd[0] == "ffprobe"
    assert "-print_format" in cmd and cmd[cmd.index("-print_format") + 1] == "json"
    assert "-show_streams" in cmd and "-show_format" in cmd
    assert cmd[-1] == "/data/clip.mov"


def test_poster_seek_clamps_for_short_clips():
    # Long video -> the requested 1s (fast keyframe seek).
    assert video.poster_seek_seconds(39.8, 1.0) == 1.0
    assert video.poster_seek_seconds(3600.0, 1.0) == 1.0
    # Clip shorter than the request -> ~10% in, never the end.
    assert video.poster_seek_seconds(0.5, 1.0) == 0.05
    assert video.poster_seek_seconds(0.0, 1.0) == 1.0  # unknown duration -> trust request
    assert video.poster_seek_seconds(2.0, 5.0) == 0.2


def test_parse_probe_extracts_video_stream():
    raw = b"""{"format": {"duration": "12.5", "format_name": "mov,mp4"},
               "streams": [{"codec_type": "audio"},
                           {"codec_type": "video", "width": 1920, "height": 1080, "codec_name": "h264"}]}"""
    info = video._parse_probe(raw)
    assert info == {
        "duration": 12.5, "width": 1920, "height": 1080,
        "codec": "h264", "format": "mov,mp4", "has_video": True,
    }


def test_parse_probe_handles_no_video_stream():
    info = video._parse_probe(b'{"format": {}, "streams": [{"codec_type": "audio"}]}')
    assert info["has_video"] is False and info["width"] == 0 and info["duration"] == 0.0
