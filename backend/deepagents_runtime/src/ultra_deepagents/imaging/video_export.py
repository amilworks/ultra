"""Strict, source-bound scientific slice video export.

The encoder never interprets source pixels itself.  Every PNG frame is fetched from
the control plane's worker-only alias of the same selector-aware ``/slice`` endpoint
used by Lens.  This keeps channel fusion, time selection, medical windowing, and
nearest-neighbour mask thresholding identical between the interactive view and MP4.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import stat
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import suppress
from pathlib import Path
from typing import Any

from ultra_deepagents.imaging.derivative_manifest import (
    DeterministicDerivativeError,
    StaleDerivativeJobError,
    TransientDerivativeError,
    _publication_lock,
)

VIDEO_MANIFEST_SCHEMA = "ultra.image-video-export-manifest.v1"
VIDEO_RECIPE_SCHEMA = "ultra.image-video-export-recipe.v1"
VIDEO_RENDERER_REVISION = "ultra.slice-video-renderer.v1"
VIDEO_STATUS_SCHEMA = "ultra.image-video-export-status.v1"
MAX_FRAME_BYTES = 64 << 20
MAX_MARKER_BYTES = 1 << 20
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RESOURCE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_HEX_COLOR_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")


def _strict_int(value: Any, *, minimum: int = 0, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DeterministicDerivativeError("invalid_video_recipe")
    return int(value)


def _strict_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise DeterministicDerivativeError("invalid_video_recipe")
    return value


def _exact_dict(value: Any, expected: set[str], *, code: str = "invalid_video_recipe") -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise DeterministicDerivativeError(code)
    return dict(value)


def _recipe_in_canonical_order(recipe: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "schema",
        "renderer_revision",
        "resource_id",
        "source",
        "axes",
        "mode",
        "profile",
        "fps",
        "source_frame_count",
        "frame_indices",
        "fixed_z",
        "fixed_t",
        "strict_scalar_slice",
        "channels",
        "channel_colors",
        "enhancement",
        "negative",
        "scalar_render_mode",
        "scalar_threshold_value",
        "scalar_threshold_foreground",
        "full_resolution",
        "max_frame_edge",
    }
    recipe = _exact_dict(recipe, expected)
    source = _exact_dict(recipe["source"], {"sha256", "size_bytes"})
    axes = _exact_dict(recipe["axes"], {"T", "C", "Z"})
    return {
        "schema": recipe["schema"],
        "renderer_revision": recipe["renderer_revision"],
        "resource_id": recipe["resource_id"],
        "source": {"sha256": source["sha256"], "size_bytes": source["size_bytes"]},
        "axes": {"T": axes["T"], "C": axes["C"], "Z": axes["Z"]},
        "mode": recipe["mode"],
        "profile": recipe["profile"],
        "fps": recipe["fps"],
        "source_frame_count": recipe["source_frame_count"],
        "frame_indices": recipe["frame_indices"],
        "fixed_z": recipe["fixed_z"],
        "fixed_t": recipe["fixed_t"],
        "strict_scalar_slice": recipe["strict_scalar_slice"],
        "channels": recipe["channels"],
        "channel_colors": recipe["channel_colors"],
        "enhancement": recipe["enhancement"],
        "negative": recipe["negative"],
        "scalar_render_mode": recipe["scalar_render_mode"],
        "scalar_threshold_value": recipe["scalar_threshold_value"],
        "scalar_threshold_foreground": recipe["scalar_threshold_foreground"],
        "full_resolution": recipe["full_resolution"],
        "max_frame_edge": recipe["max_frame_edge"],
    }


def _validate_recipe(recipe: dict[str, Any], render_id: str, resource_id: str) -> dict[str, Any]:
    ordered = _recipe_in_canonical_order(recipe)
    if ordered["schema"] != VIDEO_RECIPE_SCHEMA or ordered["renderer_revision"] != VIDEO_RENDERER_REVISION:
        raise DeterministicDerivativeError("unsupported_video_recipe")
    if ordered["resource_id"] != resource_id:
        raise DeterministicDerivativeError("invalid_video_recipe")
    source = ordered["source"]
    if not isinstance(source["sha256"], str) or _SHA256_RE.fullmatch(source["sha256"]) is None:
        raise DeterministicDerivativeError("invalid_video_recipe")
    _strict_int(source["size_bytes"], field="source.size_bytes")
    axes = ordered["axes"]
    for axis in ("T", "C", "Z"):
        _strict_int(axes[axis], minimum=1, field=f"axes.{axis}")
    if ordered["mode"] not in {"z_sweep", "time_series"} or ordered["profile"] not in {"preview", "complete"}:
        raise DeterministicDerivativeError("invalid_video_recipe")
    if _strict_int(ordered["fps"], minimum=1, field="fps") != 24:
        raise DeterministicDerivativeError("unsupported_video_recipe")
    source_count = _strict_int(ordered["source_frame_count"], minimum=2, field="source_frame_count")
    expected_source_count = axes["Z"] if ordered["mode"] == "z_sweep" else axes["T"]
    if source_count != expected_source_count:
        raise DeterministicDerivativeError("invalid_video_recipe")
    raw_indices = ordered["frame_indices"]
    if not isinstance(raw_indices, list) or not raw_indices or len(raw_indices) > 1200:
        raise DeterministicDerivativeError("invalid_video_recipe")
    indices = [_strict_int(value, field="frame_indices") for value in raw_indices]
    if indices != sorted(set(indices)) or indices[0] != 0 or indices[-1] != source_count - 1:
        raise DeterministicDerivativeError("invalid_video_recipe")
    if ordered["profile"] == "complete" and indices != list(range(source_count)):
        raise DeterministicDerivativeError("invalid_video_recipe")
    fixed_z = _strict_int(ordered["fixed_z"], field="fixed_z")
    fixed_t = _strict_int(ordered["fixed_t"], field="fixed_t")
    if fixed_z >= axes["Z"] or fixed_t >= axes["T"]:
        raise DeterministicDerivativeError("invalid_video_recipe")
    channels = ordered["channels"]
    if not isinstance(channels, list) or not (1 <= len(channels) <= 8):
        raise DeterministicDerivativeError("invalid_video_recipe")
    normalized_channels = [_strict_int(value, field="channels") for value in channels]
    if len(set(normalized_channels)) != len(normalized_channels) or any(value >= axes["C"] for value in normalized_channels):
        raise DeterministicDerivativeError("invalid_video_recipe")
    colors = ordered["channel_colors"]
    if not isinstance(colors, list) or (colors and len(colors) != len(normalized_channels)):
        raise DeterministicDerivativeError("invalid_video_recipe")
    if any(not isinstance(value, str) or _HEX_COLOR_RE.fullmatch(value) is None for value in colors):
        raise DeterministicDerivativeError("invalid_video_recipe")
    if not isinstance(ordered["strict_scalar_slice"], bool):
        raise DeterministicDerivativeError("invalid_video_recipe")
    if ordered["strict_scalar_slice"] and len(normalized_channels) != 1:
        raise DeterministicDerivativeError("invalid_video_recipe")
    if not isinstance(ordered["enhancement"], str) or len(ordered["enhancement"]) > 128:
        raise DeterministicDerivativeError("invalid_video_recipe")
    if not isinstance(ordered["negative"], bool) or ordered["full_resolution"] is not False:
        raise DeterministicDerivativeError("invalid_video_recipe")
    if _strict_int(ordered["max_frame_edge"], minimum=1, field="max_frame_edge") != 1024:
        raise DeterministicDerivativeError("unsupported_video_recipe")
    mode = ordered["scalar_render_mode"]
    if mode not in {"intensity", "mask"}:
        raise DeterministicDerivativeError("invalid_video_recipe")
    threshold = ordered["scalar_threshold_value"]
    foreground = ordered["scalar_threshold_foreground"]
    if mode == "mask":
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not float("-inf") < float(threshold) < float("inf"):
            raise DeterministicDerivativeError("invalid_video_recipe")
        if foreground != "above" or ordered["strict_scalar_slice"]:
            raise DeterministicDerivativeError("invalid_video_recipe")
    elif threshold is not None or foreground != "":
        raise DeterministicDerivativeError("invalid_video_recipe")
    canonical = json.dumps(ordered, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != render_id:
        raise DeterministicDerivativeError("video_recipe_identity_mismatch")
    return ordered


def _validate_job(job: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "resource_id",
        "src_path",
        "source_sha256",
        "source_size_bytes",
        "render_id",
        "recipe",
        "output_path",
        "manifest_path",
        "queued_path",
        "progress_path",
        "failed_path",
        "owner_role",
        "owner_user_id",
        "owner_org_id",
        "force_id",
    }
    if not isinstance(job, dict) or not set(job).issubset(expected) or not expected.difference({"force_id"}).issubset(job):
        raise DeterministicDerivativeError("invalid_video_job")
    resource_id = _strict_string(job["resource_id"], field="resource_id")
    render_id = _strict_string(job["render_id"], field="render_id")
    if _RESOURCE_RE.fullmatch(resource_id) is None or _SHA256_RE.fullmatch(render_id) is None:
        raise DeterministicDerivativeError("invalid_video_job")
    source_sha = _strict_string(job["source_sha256"], field="source_sha256")
    source_size = _strict_int(job["source_size_bytes"], field="source_size_bytes")
    if _SHA256_RE.fullmatch(source_sha) is None:
        raise DeterministicDerivativeError("invalid_video_job")
    recipe = _validate_recipe(job["recipe"], render_id, resource_id)
    if recipe["source"] != {"sha256": source_sha, "size_bytes": source_size}:
        raise DeterministicDerivativeError("invalid_video_job")
    source = Path(_strict_string(job["src_path"], field="src_path"))
    output = Path(_strict_string(job["output_path"], field="output_path"))
    manifest = Path(_strict_string(job["manifest_path"], field="manifest_path"))
    queued = Path(_strict_string(job["queued_path"], field="queued_path"))
    progress = Path(_strict_string(job["progress_path"], field="progress_path"))
    failed = Path(_strict_string(job["failed_path"], field="failed_path"))
    base = f"{resource_id}__video.{render_id}"
    expected_names = {
        output: base + ".mp4",
        manifest: base + ".manifest.json",
        queued: base + ".queued.json",
        progress: base + ".progress.json",
        failed: base + ".failed.json",
    }
    if any(path.name != name for path, name in expected_names.items()):
        raise DeterministicDerivativeError("invalid_video_job")
    if len({path.parent for path in expected_names}) != 1 or output.parent.name != "derived":
        raise DeterministicDerivativeError("invalid_video_job")
    owner_user_id = _strict_string(job["owner_user_id"], field="owner_user_id")
    owner_org_id = _strict_string(job["owner_org_id"], field="owner_org_id")
    owner_role = _strict_string(job["owner_role"], field="owner_role")
    return {
        **job,
        "resource_id": resource_id,
        "render_id": render_id,
        "source_sha256": source_sha,
        "source_size_bytes": source_size,
        "recipe": recipe,
        "src_path": source,
        "output_path": output,
        "manifest_path": manifest,
        "queued_path": queued,
        "progress_path": progress,
        "failed_path": failed,
        "owner_user_id": owner_user_id,
        "owner_org_id": owner_org_id,
        "owner_role": owner_role,
    }


def _regular_source_snapshot(path: Path) -> tuple[int, int, int, int, int]:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise StaleDerivativeJobError("source_generation_changed") from exc
    except OSError as exc:
        raise TransientDerivativeError("source_unavailable") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise StaleDerivativeJobError("source_generation_changed")
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _verify_source(path: Path, expected_sha256: str, expected_size: int) -> tuple[int, int, int, int, int]:
    before = _regular_source_snapshot(path)
    if before[2] != expected_size:
        raise StaleDerivativeJobError("catalog_source_size_mismatch")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except FileNotFoundError as exc:
        raise StaleDerivativeJobError("source_generation_changed") from exc
    except OSError as exc:
        raise TransientDerivativeError("source_unavailable") from exc
    after = _regular_source_snapshot(path)
    if before != after:
        raise StaleDerivativeJobError("source_generation_changed")
    if digest.hexdigest() != expected_sha256:
        raise StaleDerivativeJobError("catalog_source_digest_mismatch")
    return after


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{secrets.token_hex(8)}"
    data = json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8") + b"\n"
    try:
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except OSError as exc:
        raise TransientDerivativeError("video_status_publication_failed") from exc
    finally:
        with suppress(OSError):
            temporary.unlink()


def _png_dimensions(data: bytes) -> tuple[int, int]:
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        raise DeterministicDerivativeError("invalid_video_frame")
    width = int.from_bytes(data[16:20], "big")
    height = int.from_bytes(data[20:24], "big")
    if width < 1 or height < 1 or width > 16384 or height > 16384:
        raise DeterministicDerivativeError("invalid_video_frame")
    return width, height


def _frame_url(control_base_url: str, resource_id: str, recipe: dict[str, Any], source_index: int, render_id: str) -> str:
    query: dict[str, str] = {"axis": "z", "full_resolution": "false", "cache_key": f"video-v1:{render_id}"}
    if recipe["mode"] == "z_sweep":
        query.update({"z": str(source_index), "t": str(recipe["fixed_t"])})
    else:
        query.update({"z": str(recipe["fixed_z"]), "t": str(source_index)})
    if recipe["strict_scalar_slice"]:
        query["c"] = str(recipe["channels"][0])
        if recipe["enhancement"]:
            query["enhancement"] = recipe["enhancement"]
        query["negative"] = "true" if recipe["negative"] else "false"
    else:
        query["channels"] = ",".join(str(value) for value in recipe["channels"])
        if recipe["channel_colors"]:
            query["channel_colors"] = ",".join(recipe["channel_colors"])
        query["scalar_render_mode"] = recipe["scalar_render_mode"]
        if recipe["scalar_render_mode"] == "mask":
            query["scalar_threshold_value"] = format(float(recipe["scalar_threshold_value"]), ".17g")
            query["scalar_threshold_foreground"] = "above"
    return (
        control_base_url.rstrip("/")
        + "/v2/internal/uploads/"
        + urllib.parse.quote(resource_id, safe="")
        + "/render-frame?"
        + urllib.parse.urlencode(query)
    )


def _fetch_frame(url: str, *, worker_token: str, owner_user_id: str, owner_org_id: str, owner_role: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "image/png",
            "X-Ultra-Worker-Token": worker_token,
            "X-Ultra-User-Id": owner_user_id,
            "X-Ultra-Org-Id": owner_org_id,
            "X-Ultra-Role": owner_role,
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            content_type = response.headers.get_content_type()
            data = response.read(MAX_FRAME_BYTES + 1)
    except urllib.error.HTTPError as exc:
        with suppress(Exception):
            exc.read(4096)
        if exc.code == 404:
            raise StaleDerivativeJobError("source_generation_changed") from exc
        if exc.code in {400, 415, 422}:
            raise DeterministicDerivativeError("video_frame_request_rejected") from exc
        raise TransientDerivativeError("video_frame_service_unavailable") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise TransientDerivativeError("video_frame_service_unavailable") from exc
    if content_type != "image/png" or len(data) == 0 or len(data) > MAX_FRAME_BYTES:
        raise DeterministicDerivativeError("invalid_video_frame")
    return data


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise TransientDerivativeError("video_artifact_unavailable") from exc
    return digest.hexdigest()


def _manifest_ready(job: dict[str, Any]) -> bool:
    manifest_path: Path = job["manifest_path"]
    try:
        info = manifest_path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or not (2 <= info.st_size <= MAX_MARKER_BYTES):
            return False
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict) or payload.get("schema") != VIDEO_MANIFEST_SCHEMA or payload.get("render_id") != job["render_id"]:
        return False
    source = payload.get("source")
    artifact = payload.get("artifact")
    if (
        set(payload) != {"schema", "render_id", "created_at", "source", "recipe", "artifact"}
        or source != {"sha256": job["source_sha256"], "size_bytes": job["source_size_bytes"]}
        or payload.get("recipe") != job["recipe"]
        or not isinstance(artifact, dict)
        or set(artifact)
        != {"basename", "sha256", "size_bytes", "media_type", "width", "height", "frame_count", "fps"}
        or artifact.get("basename") != job["output_path"].name
        or artifact.get("media_type") != "video/mp4"
        or artifact.get("frame_count") != len(job["recipe"]["frame_indices"])
        or artifact.get("fps") != job["recipe"]["fps"]
        or not isinstance(artifact.get("size_bytes"), int)
        or artifact["size_bytes"] <= 0
        or not isinstance(artifact.get("width"), int)
        or artifact["width"] <= 0
        or not isinstance(artifact.get("height"), int)
        or artifact["height"] <= 0
        or not isinstance(artifact.get("sha256"), str)
        or _SHA256_RE.fullmatch(artifact["sha256"]) is None
    ):
        return False
    try:
        output_info = job["output_path"].lstat()
        if (
            stat.S_ISLNK(output_info.st_mode)
            or not stat.S_ISREG(output_info.st_mode)
            or output_info.st_size != artifact["size_bytes"]
        ):
            return False
        return _sha256_file(job["output_path"]) == artifact["sha256"]
    except (OSError, TransientDerivativeError):
        return False


def run_video_export_job(raw_job: dict[str, Any]) -> dict[str, Any]:
    job = _validate_job(raw_job)
    control_base_url = os.environ.get("ULTRA_CONTROL_BASE_URL", "").strip()
    worker_token = os.environ.get("ULTRA_CONTROL_WORKER_TOKEN", "").strip()
    parsed_base = urllib.parse.urlparse(control_base_url)
    if parsed_base.scheme not in {"http", "https"} or not parsed_base.netloc or not worker_token:
        raise TransientDerivativeError("video_frame_service_not_configured")
    source: Path = job["src_path"]
    output: Path = job["output_path"]
    recipe: dict[str, Any] = job["recipe"]
    _verify_source(source, job["source_sha256"], job["source_size_bytes"])
    # Reuse the resource-wide lifecycle lock used by strict pyramid publication.
    # A dummy __pyramid destination intentionally maps to the same resource lock;
    # no pyramid artifact is created here.
    lifecycle_destination = output.parent / f"{job['resource_id']}__pyramid.tif"
    with _publication_lock(lifecycle_destination, source):
        _verify_source(source, job["source_sha256"], job["source_size_bytes"])
        if _manifest_ready(job):
            return {"resource_id": job["resource_id"], "render_id": job["render_id"], "status": "ready"}
        output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        for marker in (job["queued_path"], job["progress_path"], job["failed_path"]):
            with suppress(OSError):
                marker.unlink()
        progress = {
            "schema": VIDEO_STATUS_SCHEMA,
            "render_id": job["render_id"],
            "resource_id": job["resource_id"],
            "source": recipe["source"],
            "mode": recipe["mode"],
            "profile": recipe["profile"],
            "source_frame_count": recipe["source_frame_count"],
            "frames_total": len(recipe["frame_indices"]),
            "frames_completed": 0,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _atomic_json(job["progress_path"], progress)
        temporary = output.parent / f".{output.stem}.tmp-{secrets.token_hex(8)}.mp4"
        stderr_file = None
        process: subprocess.Popen[bytes] | None = None
        width = height = 0
        try:
            stderr_file = open(os.devnull, "wb")
            command = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-f",
                "image2pipe",
                "-vcodec",
                "png",
                "-framerate",
                str(recipe["fps"]),
                "-i",
                "pipe:0",
                "-an",
                "-vf",
                (
                    "pad=ceil(iw/2)*2:ceil(ih/2)*2,"
                    "scale=in_range=pc:out_range=tv:out_color_matrix=bt709,"
                    "format=yuv420p"
                ),
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-color_range",
                "tv",
                "-colorspace",
                "bt709",
                "-color_primaries",
                "bt709",
                "-color_trc",
                "iec61966-2-1",
                "-movflags",
                "+faststart",
                "-map_metadata",
                "-1",
                "-threads",
                "2",
                "-y",
                str(temporary),
            ]
            try:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=stderr_file,
                )
            except (FileNotFoundError, OSError) as exc:
                raise TransientDerivativeError("video_encoder_unavailable") from exc
            assert process.stdin is not None
            last_progress = time.monotonic()
            for completed, source_index in enumerate(recipe["frame_indices"], start=1):
                frame = _fetch_frame(
                    _frame_url(control_base_url, job["resource_id"], recipe, source_index, job["render_id"]),
                    worker_token=worker_token,
                    owner_user_id=job["owner_user_id"],
                    owner_org_id=job["owner_org_id"],
                    owner_role=job["owner_role"],
                )
                dimensions = _png_dimensions(frame)
                if completed == 1:
                    width, height = dimensions
                elif dimensions != (width, height):
                    raise DeterministicDerivativeError("video_frame_geometry_changed")
                try:
                    process.stdin.write(frame)
                except (BrokenPipeError, OSError) as exc:
                    raise TransientDerivativeError("video_encoding_failed") from exc
                if completed == len(recipe["frame_indices"]) or time.monotonic() - last_progress >= 1:
                    progress["frames_completed"] = completed
                    progress["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    _atomic_json(job["progress_path"], progress)
                    last_progress = time.monotonic()
            process.stdin.close()
            return_code = process.wait(timeout=300)
            if return_code != 0:
                raise TransientDerivativeError("video_encoding_failed")
            process = None
            temporary_info = temporary.lstat()
            if stat.S_ISLNK(temporary_info.st_mode) or not stat.S_ISREG(temporary_info.st_mode) or temporary_info.st_size <= 0:
                raise TransientDerivativeError("video_encoding_failed")
            artifact_sha256 = _sha256_file(temporary)
            _verify_source(source, job["source_sha256"], job["source_size_bytes"])
            os.replace(temporary, output)
            with output.open("rb") as artifact_stream:
                os.fsync(artifact_stream.fileno())
            directory = os.open(output.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            output_info = output.stat()
            manifest = {
                "schema": VIDEO_MANIFEST_SCHEMA,
                "render_id": job["render_id"],
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "source": recipe["source"],
                "recipe": recipe,
                "artifact": {
                    "basename": output.name,
                    "sha256": artifact_sha256,
                    "size_bytes": output_info.st_size,
                    "media_type": "video/mp4",
                    "width": width + (width % 2),
                    "height": height + (height % 2),
                    "frame_count": len(recipe["frame_indices"]),
                    "fps": recipe["fps"],
                },
            }
            _atomic_json(job["manifest_path"], manifest)
            for marker in (job["queued_path"], job["progress_path"], job["failed_path"]):
                with suppress(OSError):
                    marker.unlink()
            return {"resource_id": job["resource_id"], "render_id": job["render_id"], "status": "ready"}
        finally:
            if process is not None:
                with suppress(Exception):
                    if process.stdin is not None:
                        process.stdin.close()
                with suppress(Exception):
                    process.kill()
                with suppress(Exception):
                    process.wait(timeout=10)
            if stderr_file is not None:
                stderr_file.close()
            with suppress(OSError):
                temporary.unlink()


def write_video_failure_marker(raw_job: dict[str, Any] | None, code: str) -> str:
    """Publish a sanitized terminal status for a validated video job."""

    try:
        if raw_job is None:
            return "not_applicable"
        job = _validate_job(raw_job)
        marker = {
            "schema": VIDEO_STATUS_SCHEMA,
            "render_id": job["render_id"],
            "resource_id": job["resource_id"],
            "source": job["recipe"]["source"],
            "mode": job["recipe"]["mode"],
            "profile": job["recipe"]["profile"],
            "source_frame_count": job["recipe"]["source_frame_count"],
            "frames_total": len(job["recipe"]["frame_indices"]),
            "frames_completed": 0,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "code": code if re.fullmatch(r"[a-z][a-z0-9_]{0,63}", code) else "video_export_failed",
        }
        lifecycle_destination = job["output_path"].parent / f"{job['resource_id']}__pyramid.tif"
        with _publication_lock(lifecycle_destination, job["src_path"]):
            _atomic_json(job["failed_path"], marker)
            for stale in (job["queued_path"], job["progress_path"]):
                with suppress(OSError):
                    stale.unlink()
        return "written"
    except StaleDerivativeJobError:
        return "stale"
    except (TransientDerivativeError, OSError):
        return "retryable"
    except Exception:
        return "not_applicable"
