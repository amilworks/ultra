"""RareSpot (yolov5_rarespot) ModelAdapter — plan sections 4.3-4.4, 5, 6.2, 7.3.

The five hooks:

- ``materialize_dataset`` — BOTH purposes ('sync' and 'gold_cut'): EXIF step 1a
  BEFORE tiling, manifest-priority GT-layer selection, taxonomy reconcile with
  gray-fill ignore masking for ``prairie_dog_in_burrow``, production tiling
  (512 / 0.25 via ``rarespot.tiling.build_sliding_tiles``), YOLO-txt labels,
  sha256 + pHash identity, standard items.
- ``extra_leakage_checks`` — the manifest-declared ``aerial_geospatial_overlap``
  defense plus identity exclusion against the recovered-train-v0 inventory
  (sha-verified, kernel-pinned; fail-closed refusals raise).
- ``evaluate`` — the two-pass eval kernel SHAPE (``yolov5_two_pass/v1``):
  val.py at conf 0.001 for curve metrics, the production detect path at
  conf 0.25 for operating-point metrics; pure ``assemble_detection_metrics``
  emits the detection.v1 blob with the five mandatory conventions.
- ``smoke_test`` — names/nc assertion + one real detect pass (plan 6.2).
- ``train`` — the M3 command builder (``build_finetune_command``) + BN-freeze
  patch source; actual GPU execution lands on the lambda worker deployment.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import logging
import math
import os
import shutil
import ssl
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from PIL import Image

from ultra_deepagents.rarespot.config import RareSpotConfig
from ultra_deepagents.rarespot.geospatial import haversine_m, read_exif_gps
from ultra_deepagents.rarespot.inference import run_yolov5_detect
from ultra_deepagents.rarespot.tiling import Tile, build_sliding_tiles
from ultra_deepagents.training.adapters import (
    MaterializeContext,
    ModelAdapter,
    TrainContext,
    register_adapter,
)
from ultra_deepagents.training.gt2_census import layer_boxes, select_gt_layer
from ultra_deepagents.training.manifests import RARESPOT_MANIFEST
from ultra_deepagents.training.phash import PHASH_KERNEL, hamming, phash64_file

logger = logging.getLogger(__name__)

MODEL_KEY = RARESPOT_MANIFEST.model_key
CLASS_NAMES = ["prairie_dog", "burrow"]
CLASS_ID_BY_NAME = {"prairie_dog": 0, "burrow": 1}
IGNORE_CLASSES = frozenset({"prairie_dog_in_burrow"})
# YOLOv5's letterbox gray: removes the region from BOTH loss and eval with zero
# patches to train.py/val.py (plan 5b).
GRAY_FILL_RGB = (114, 114, 114)
TILE_SIZE = 512
TILE_OVERLAP = 0.25
JPEG_QUALITY = 95

DETECTION_SCHEMA = "detection.v1"
DETECTION_KERNEL = "yolov5_two_pass/v1"
REQUIRED_SLICES = ("prior_train", "held_out_test")

# Pass-1 regime: val.py's only valid confidence (val.py itself warns above 0.001).
CURVE_CONF = 0.001
# Pass-2 operating point: the shipped production path (rarespot/config.py).
OP_CONF = 0.25
OP_IOU = 0.45
OP_MATCH_IOU = 0.5

HELD_OUT_PHASH_HAMMING = 6
DEFAULT_GEO_RADIUS_M = 200.0

HYP_PATH = Path(__file__).with_name("hyp.rarespot.finetune.yaml")
IGNORE_SIDECAR_NAME = "ignore_boxes.json"


class LeakageCheckRefused(RuntimeError):  # noqa: N818 - "refusal" is the plan-5e term
    """Fail-closed refusal: the exclusion inventory cannot be trusted (sha or
    pHash-kernel mismatch), so the freeze must not proceed on it."""


# --------------------------------------------------------------------- source


class FrameSource(Protocol):
    """The source seam for materialization: list frames, fetch one."""

    def list_frames(self) -> list[str]: ...

    def fetch(self, frame: str) -> tuple[Path, Path]:
        """Return ``(image_path, gobject_xml_path)`` for a frame stem."""
        ...


class LocalDirectoryFrameSource:
    """Directory of ``<stem>.JPG`` + ``<stem>.JPG.xml`` pairs (e.g. test-prairie/)."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    def list_frames(self) -> list[str]:
        stems: list[str] = []
        for xml_path in sorted(self._root.glob("*.xml")):
            name = xml_path.name
            if not name.lower().endswith(".jpg.xml"):
                continue
            stems.append(name[: -len(".JPG.xml")])
        return stems

    def fetch(self, frame: str) -> tuple[Path, Path]:
        for suffix in (".JPG", ".jpg", ".JPEG", ".jpeg"):
            image_path = self._root / f"{frame}{suffix}"
            if image_path.is_file():
                xml_path = self._root / f"{image_path.name}.xml"
                if xml_path.is_file():
                    return image_path, xml_path
        raise FileNotFoundError(
            f"frame {frame!r} (image + GObject XML) not found under {self._root}"
        )


class BisqueFrameSource:
    """Pull reviewed frames + GT XML from the BisQue ``Prairie_Dog_Active_Learning``
    pool into a cache dir shaped exactly like ``LocalDirectoryFrameSource`` input
    (``<stem>.JPG`` + ``<stem>.JPG.xml`` with ``<image>`` as the XML root).

    Wire shapes mirror the Go BisqueService (httpapi/bisque.go) against bisque2:
    ``/data_service/dataset/?query=&wpublic=&limit=&offset=`` (paginated to
    exhaustion) finds the pool, ``?view=deep`` on a resource URI resolves dataset
    members and an image's stacked GObject layers, ``/blob_service/{uniq}``
    serves the original frame bytes; every request carries HTTP Basic auth. A
    frame is "reviewed" when its deep XML has a layer matching the manifest
    ``gt_layer_priority`` (gt2 first). An explicit ``tag_query`` swaps the
    dataset walk for a paginated ``/data_service/image/`` search.
    """

    DEFAULT_DATASET = "Prairie_Dog_Active_Learning"
    PAGE_LIMIT = 100

    def __init__(
        self,
        *,
        root_url: str | None = None,
        username: str | None = None,
        password: str | None = None,
        dataset: str | None = None,
        tag_query: str | None = None,
        cache_dir: str | Path | None = None,
        timeout: float = 60.0,
    ) -> None:
        # Fail fast at CONSTRUCTION: _resolve_frame_source routes kind=='bisque'
        # here, and a mid-sync auth surprise is worse than a refused dispatch.
        self._root_url = _config_value(
            root_url, "ULTRA_TRAINING_BISQUE_ROOT_URL", "ULTRA_CONTROL_BISQUE_ROOT_URL"
        ).rstrip("/")
        username = _config_value(
            username, "ULTRA_TRAINING_BISQUE_USERNAME", "ULTRA_CONTROL_BISQUE_USERNAME"
        )
        password = _config_value(
            password, "ULTRA_TRAINING_BISQUE_PASSWORD", "ULTRA_CONTROL_BISQUE_PASSWORD"
        )
        if not self._root_url:
            raise ValueError(
                "BisQue frame source needs a root URL: pass root_url= or set "
                "ULTRA_TRAINING_BISQUE_ROOT_URL (or ULTRA_CONTROL_BISQUE_ROOT_URL)"
            )
        if not username or not password:
            raise ValueError(
                "BisQue frame source needs credentials: pass username=/password= or set "
                "ULTRA_TRAINING_BISQUE_USERNAME/PASSWORD (or ULTRA_CONTROL_BISQUE_USERNAME/PASSWORD)"
            )
        self._dataset = _config_value(dataset, "ULTRA_TRAINING_BISQUE_DATASET") or self.DEFAULT_DATASET
        self._tag_query = _config_value(tag_query, "ULTRA_TRAINING_BISQUE_TAG_QUERY")
        cache = _config_value(cache_dir, "ULTRA_TRAINING_BISQUE_CACHE_DIR")
        self._cache_dir = Path(cache) if cache else Path(tempfile.mkdtemp(prefix="goldgate-bisque-"))
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        token = base64.b64encode(f"{username}:{password}".encode()).decode("ascii")
        self._headers = {"Accept": "text/xml", "Authorization": f"Basic {token}"}
        self._timeout = max(1.0, float(timeout))
        self._frames: dict[str, dict[str, Any]] | None = None

    def list_frames(self) -> list[str]:
        frames = self._frame_index()
        return sorted(stem for stem, entry in frames.items() if entry["reviewed"])

    def fetch(self, frame: str) -> tuple[Path, Path]:
        entry = self._frame_index().get(frame)
        if entry is None:
            raise FileNotFoundError(
                f"frame {frame!r} not found in BisQue pool {self._pool_label()}"
            )
        uniq = str(entry["resource_uniq"] or "")
        if not uniq:
            raise RuntimeError(f"BisQue frame {frame!r} has no resource_uniq for blob download")
        image_path = self._cache_dir / f"{frame}.JPG"
        # A re-upload of identical pixels mints a NEW resource_uniq (plan 5c),
        # so the uniq marker is what makes stem-named cache reuse safe.
        marker = self._cache_dir / f"{frame}.JPG.uniq"
        cached = (
            image_path.is_file()
            and image_path.stat().st_size > 0
            and marker.is_file()
            and marker.read_text(encoding="utf-8").strip() == uniq
        )
        if not cached:
            blob_url = f"{self._root_url}/blob_service/{urllib_parse.quote(uniq, safe='')}"
            self._download(blob_url, image_path)
            marker.write_text(uniq, encoding="utf-8")
        return image_path, entry["xml_path"]

    # ------------------------------------------------------------- pool walk

    def _frame_index(self) -> dict[str, dict[str, Any]]:
        if self._frames is not None:
            return self._frames
        frames: dict[str, dict[str, Any]] = {}
        unreviewed = 0
        for member_uri in self._member_uris():
            element = self._fetch_deep(member_uri)
            image = element if element.tag == "image" else element.find(".//image")
            if image is None:
                logger.warning("BisQue pool member %s is not an image; skipped", member_uri)
                continue
            stem = _frame_stem(str(image.get("name") or "").strip())
            if not stem:
                logger.warning("BisQue pool member %s has no name; skipped", member_uri)
                continue
            if stem in frames:
                raise ValueError(
                    f"duplicate frame stem {stem!r} in BisQue pool {self._pool_label()}"
                )
            try:
                select_gt_layer(image)
                reviewed = True
            except ValueError as exc:
                reviewed = False
                unreviewed += 1
                logger.info("BisQue frame %s excluded from the reviewed pool: %s", stem, exc)
            # Written as <stem>.JPG.xml with <image> as root so downstream
            # (extract_frame_exif, select_gt_layer, tiling) works IDENTICALLY
            # to the local-directory path.
            xml_path = self._cache_dir / f"{stem}.JPG.xml"
            xml_path.write_bytes(ET.tostring(image, encoding="utf-8"))
            frames[stem] = {
                "uri": member_uri,
                "resource_uniq": str(image.get("resource_uniq") or "").strip(),
                "xml_path": xml_path,
                "reviewed": reviewed,
            }
        logger.info(
            "BisQue pool %s: %d frames, %d reviewed, %d without a GT layer in %s",
            self._pool_label(),
            len(frames),
            len(frames) - unreviewed,
            unreviewed,
            list(RARESPOT_MANIFEST.gt_layer_priority),
        )
        self._frames = frames
        return frames

    def _member_uris(self) -> list[str]:
        if self._tag_query:
            uris: list[str] = []
            for element in self._paged_resources("image", [("tag_query", self._tag_query)]):
                uri = str(element.get("uri") or "").strip()
                if not uri:
                    continue
                resolved = urllib_parse.urljoin(self._root_url + "/", uri)
                if resolved not in uris:
                    uris.append(resolved)
            return uris
        dataset_element = self._fetch_deep(self._dataset_uri())
        dataset = (
            dataset_element if dataset_element.tag == "dataset" else dataset_element.find(".//dataset")
        )
        if dataset is None:
            raise RuntimeError(
                f"BisQue dataset {self._dataset!r} deep view carried no <dataset> element"
            )
        members: list[str] = []
        for value in dataset.findall("value"):
            text = str(value.text or "").strip()
            if text:
                members.append(urllib_parse.urljoin(self._root_url + "/", text))
        if not members:
            raise RuntimeError(f"BisQue dataset {self._dataset!r} has no members")
        return members

    def _dataset_uri(self) -> str:
        matches: list[str] = []
        for element in self._paged_resources("dataset", [("query", self._dataset)]):
            if str(element.get("name") or "").strip() != self._dataset:
                continue
            uri = str(element.get("uri") or "").strip()
            if not uri:
                continue
            resolved = urllib_parse.urljoin(self._root_url + "/", uri)
            if resolved not in matches:
                matches.append(resolved)
        if not matches:
            raise RuntimeError(f"BisQue dataset {self._dataset!r} not found at {self._root_url}")
        if len(matches) > 1:
            raise RuntimeError(
                f"BisQue dataset name {self._dataset!r} is ambiguous ({len(matches)} matches: "
                f"{matches})"
            )
        return matches[0]

    def _paged_resources(
        self, resource_type: str, params: list[tuple[str, str]]
    ) -> Iterator[ET.Element]:
        """data_service listing paginated to exhaustion (mirrors bisque.go
        searchURL/countAllSearchResults) — never a silent truncation."""
        offset = 0
        while True:
            page = list(params)
            page.append(("wpublic", "owner,shared"))
            page.append(("limit", str(self.PAGE_LIMIT)))
            if offset:
                page.append(("offset", str(offset)))
            url = (
                f"{self._root_url}/data_service/{resource_type}/?{urllib_parse.urlencode(page)}"
            )
            root = ET.fromstring(self._get(url))
            elements = [root] if root.tag == resource_type else root.findall(resource_type)
            yield from elements
            if len(elements) < self.PAGE_LIMIT:
                return
            offset += self.PAGE_LIMIT

    def _pool_label(self) -> str:
        return f"tag_query={self._tag_query!r}" if self._tag_query else repr(self._dataset)

    # ------------------------------------------------------------------ wire

    def _fetch_deep(self, resource_uri: str) -> ET.Element:
        parsed = urllib_parse.urlsplit(resource_uri)
        query = [(key, value) for key, value in urllib_parse.parse_qsl(parsed.query) if key != "view"]
        query.append(("view", "deep"))
        deep_url = urllib_parse.urlunsplit(
            (parsed.scheme, parsed.netloc, parsed.path, urllib_parse.urlencode(query), "")
        )
        return ET.fromstring(self._get(deep_url))

    def _open(self, url: str) -> Any:
        request = urllib_request.Request(url, headers=dict(self._headers))
        try:
            return urllib_request.urlopen(
                request, timeout=self._timeout, context=_bisque_ssl_context(url)
            )
        except urllib_error.HTTPError as exc:
            if exc.code in (401, 403):
                raise RuntimeError(
                    f"BisQue rejected the configured credentials (HTTP {exc.code}); check "
                    "ULTRA_TRAINING_BISQUE_USERNAME/PASSWORD (or the ULTRA_CONTROL_ pair)"
                ) from exc
            raise RuntimeError(f"BisQue request failed with HTTP {exc.code}: {url}") from exc

    def _get(self, url: str) -> bytes:
        with self._open(url) as response:
            return response.read()

    def _download(self, url: str, destination: Path) -> None:
        partial = destination.with_name(destination.name + ".part")
        with self._open(url) as response, partial.open("wb") as handle:
            shutil.copyfileobj(response, handle, 1024 * 1024)
        partial.replace(destination)


def _bisque_ssl_context(url: str) -> ssl.SSLContext | None:
    """Python's bundled OpenSSL trust store rejects bisque2's chain (verified
    live: CERTIFICATE_VERIFY_FAILED on macOS + python.org builds where the
    system curl succeeds) — certifi's bundle validates it. HTTP URLs need none."""
    if not url.lower().startswith("https:"):
        return None
    import certifi

    return ssl.create_default_context(cafile=certifi.where())


def _config_value(value: Any, *env_names: str) -> str:
    text = str(value).strip() if value is not None else ""
    if text:
        return text
    for name in env_names:
        text = str(os.getenv(name) or "").strip()
        if text:
            return text
    return ""


def _frame_stem(name: str) -> str:
    for suffix in (".JPG", ".jpg", ".JPEG", ".jpeg"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


# ----------------------------------------------------------------------- exif

_EXIF_MAKE = 271
_EXIF_MODEL = 272
_EXIF_DATETIME = 306
_EXIF_IFD = 0x8769
_EXIF_DATETIME_ORIGINAL = 36867
_EXIF_FOCAL_LENGTH = 37386


def extract_frame_exif(image_path: Path) -> tuple[bool, dict[str, Any] | None]:
    """Step 1a — runs BEFORE tiling, for BOTH purposes; tiles are EXIF-stripped
    downstream, so any geometry not captured here is gone (plan 5 step 1a).

    Returns ``(exif_present, footprint_geom_v1 | None)`` with the v1 fields
    ``{center_lat, center_lon, alt_msl, gps_ts, camera, focal_mm, sensor_wh_mm,
    footprint_polygon: None, agl_m: None}``.
    """
    gps = read_exif_gps(image_path)
    camera: str | None = None
    focal_mm: float | None = None
    gps_ts: str | None = None
    try:
        with Image.open(image_path) as image:
            exif = image.getexif()
            if exif:
                make = str(exif.get(_EXIF_MAKE) or "").strip()
                model = str(exif.get(_EXIF_MODEL) or "").strip()
                camera = " ".join(part for part in (make, model) if part) or None
                gps_ts = str(exif.get(_EXIF_DATETIME) or "").strip() or None
                ifd: Any = {}
                with contextlib.suppress(Exception):
                    ifd = exif.get_ifd(_EXIF_IFD)
                if ifd:
                    gps_ts = str(ifd.get(_EXIF_DATETIME_ORIGINAL) or gps_ts or "").strip() or None
                    raw_focal = ifd.get(_EXIF_FOCAL_LENGTH)
                    if raw_focal is not None:
                        with contextlib.suppress(TypeError, ValueError):
                            focal_mm = float(raw_focal)
    except Exception:
        logger.warning("EXIF read failed for %s", image_path, exc_info=True)
    if gps is None:
        return False, None
    footprint = {
        "center_lat": float(gps["lat"]),
        "center_lon": float(gps["lon"]),
        "alt_msl": float(gps["alt"]) if "alt" in gps else None,
        "gps_ts": gps_ts,
        "camera": camera,
        "focal_mm": focal_mm,
        "sensor_wh_mm": None,
        "footprint_polygon": None,
        "agl_m": None,
    }
    return True, footprint


# ------------------------------------------------------------------- taxonomy


def reconcile_class(raw_name: str) -> tuple[str, str]:
    """Trim (BisQue preserves leading whitespace) and map a reviewer class name
    to its disposition: ``map`` (model class), ``ignore`` (gray-fill region),
    or ``dropped``. No non-model class ever feeds a positive label."""
    name = str(raw_name or "").strip()
    if name in CLASS_ID_BY_NAME:
        return name, "map"
    if name in IGNORE_CLASSES:
        return name, "ignore"
    return name, "dropped"


# -------------------------------------------------------------------- adapter


class RareSpotAdapter(ModelAdapter):
    model_key = MODEL_KEY
    implemented_leakage_checks = ("aerial_geospatial_overlap",)

    # -------------------------------------------------------- materialization

    def materialize_dataset(self, ctx: MaterializeContext) -> dict[str, Any]:
        source = self._resolve_frame_source(ctx)
        params = dict(ctx.params or {})
        frames = [str(stem) for stem in (params.get("frame_stems") or source.list_frames())]
        workdir = Path(ctx.workdir)
        items: list[dict[str, Any]] = []
        unsupported: dict[str, dict[str, Any]] = {}
        for stem in frames:
            item, frame_unsupported = self._materialize_frame(
                source=source, stem=stem, workdir=workdir, site_id=str(params.get("site_id") or "")
            )
            items.append(item)
            for name, row in frame_unsupported.items():
                bucket = unsupported.setdefault(
                    name, {"count": 0, "disposition": row["disposition"]}
                )
                bucket["count"] += row["count"]
        return {
            "model_key": self.model_key,
            "purpose": ctx.purpose,
            "tile_spec": {
                "tile_size": TILE_SIZE,
                "overlap": TILE_OVERLAP,
                "stride": max(1, round(TILE_SIZE * (1.0 - TILE_OVERLAP))),
            },
            "masking_spec": {
                "fill_rgb": list(GRAY_FILL_RGB),
                "box_to_pixel_expansion": "round-out",
                "classes": sorted(IGNORE_CLASSES),
            },
            "gt_layer_priority": list(RARESPOT_MANIFEST.gt_layer_priority),
            "items": items,
            "unsupported_class_counts": unsupported,
        }

    def _resolve_frame_source(self, ctx: MaterializeContext) -> FrameSource:
        # Job params override; the env pair is DEPLOYMENT config — the sync
        # dispatch is deliberately parameterless (the UI never chooses a data
        # source), so where frames come from is a per-worker install fact,
        # exactly like the fleet BisQue credentials will be.
        params = dict(ctx.params or {})
        kind = str(
            params.get("frame_source") or os.getenv("ULTRA_TRAINING_FRAME_SOURCE") or ""
        ).strip().lower()
        root = str(
            params.get("frame_source_dir") or os.getenv("ULTRA_TRAINING_FRAME_SOURCE_DIR") or ""
        ).strip()
        if kind in ("", "local_directory") and root:
            return LocalDirectoryFrameSource(root)
        if kind == "bisque":
            return BisqueFrameSource(
                dataset=str(params.get("bisque_dataset") or "").strip() or None,
                tag_query=str(params.get("bisque_tag_query") or "").strip() or None,
                cache_dir=root or None,
            )
        raise ValueError(
            "no frame source configured: set params.frame_source_dir / "
            "ULTRA_TRAINING_FRAME_SOURCE_DIR (local_directory) or "
            "frame_source='bisque'"
        )

    def _materialize_frame(
        self,
        *,
        source: FrameSource,
        stem: str,
        workdir: Path,
        site_id: str = "",
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        image_path, xml_path = source.fetch(stem)

        # Step 1a: EXIF BEFORE tiling — tiles are stripped downstream.
        exif_present, footprint_geom = extract_frame_exif(image_path)

        # GT layer selection per manifest gt_layer_priority (reused census
        # logic; gt2_census.GT_LAYER_PRIORITY == manifest.gt_layer_priority,
        # names trimmed, duplicate same-priority layers fail).
        image_element = ET.parse(xml_path).getroot()
        resource_uniq = str(image_element.get("resource_uniq") or "").strip() or None
        layer_name, layer = select_gt_layer(image_element)

        mapped_boxes: list[dict[str, Any]] = []
        ignore_boxes: list[dict[str, Any]] = []
        unsupported: dict[str, dict[str, Any]] = {}
        for box in layer_boxes(layer):
            name, disposition = reconcile_class(box["class"])
            if disposition == "map":
                mapped_boxes.append({**box, "class": name, "class_id": CLASS_ID_BY_NAME[name]})
                continue
            bucket = unsupported.setdefault(name, {"count": 0, "disposition": disposition})
            bucket["count"] += 1
            if disposition == "ignore":
                ignore_boxes.append(
                    {
                        "class": name,
                        "x0": round(box["cx"] - box["w"] / 2.0, 2),
                        "y0": round(box["cy"] - box["h"] / 2.0, 2),
                        "x1": round(box["cx"] + box["w"] / 2.0, 2),
                        "y1": round(box["cy"] + box["h"] / 2.0, 2),
                    }
                )

        frames_dir = workdir / "frames" / stem
        labels_dir = workdir / "labels" / stem
        frames_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        label_texts: list[tuple[str, str]] = []
        with Image.open(image_path) as image:
            width, height = image.size
            rgb = image.convert("RGB")
            grid = build_sliding_tiles(
                width=width, height=height, tile_size=TILE_SIZE, overlap=TILE_OVERLAP
            )
            for tile in grid.tiles:
                crop = rgb.crop((tile.x0, tile.y0, tile.x1, tile.y1))
                _gray_fill_ignore_regions(crop, tile, ignore_boxes)
                tile_name = f"{stem}_{tile.grid_y}_{tile.grid_x}"
                crop.save(frames_dir / f"{tile_name}.jpg", format="JPEG", quality=JPEG_QUALITY)
                rows = _tile_label_rows(tile, mapped_boxes)
                if rows:
                    text = "\n".join(rows) + "\n"
                    (labels_dir / f"{tile_name}.txt").write_text(text, encoding="utf-8")
                    label_texts.append((tile_name, text))

        # Ignore sidecar folded into gt_label_sha256: editing ignore boxes
        # changes content_hash and invalidates cached benchmarks (plan 4.5).
        ignore_json = json.dumps(ignore_boxes, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        (labels_dir / IGNORE_SIDECAR_NAME).write_bytes(ignore_json)
        label_bytes = b"".join(
            f"{tile_name}\n".encode() + text.encode("utf-8")
            for tile_name, text in sorted(label_texts)
        )
        gt_label_sha256 = hashlib.sha256(label_bytes + b"\0" + ignore_json).hexdigest()

        per_class_count = {name: 0 for name in CLASS_NAMES}
        for box in mapped_boxes:
            per_class_count[box["class"]] += 1
        box_count = len(mapped_boxes)

        item = {
            "item_id": stem,
            "source_ref": {
                "bisque_image_id": resource_uniq,
                "frame_stem": stem,
                "source": type(source).__name__,
                "path": str(image_path),
            },
            "content_sha256": _sha256_file(image_path),
            "phash": phash64_file(image_path),
            "label_kind": "boxes",
            "gt_label_sha256": gt_label_sha256,
            "label_uri": str(labels_dir),
            "width": int(width),
            "height": int(height),
            "exif_present": exif_present,
            "footprint_geom": footprint_geom,
            "label_stats": {
                "tile_count": len(grid.tiles),
                "labeled_tile_count": len(label_texts),
                "box_count": box_count,
                "per_class_box_count": per_class_count,
            },
            "strata_tags": {
                "class_presence": _class_presence(per_class_count),
                "box_density": _box_density(box_count),
                "site_id": site_id or None,
            },
            "metadata": {
                "gt_layer": layer_name,
                "ignore_boxes": ignore_boxes,
                "unsupported_class_counts": unsupported,
                "frames_uri": str(frames_dir),
            },
        }
        return item, unsupported

    # -------------------------------------------------------- leakage extras

    def extra_leakage_checks(
        self,
        train_pool: list[dict[str, Any]],
        gold_items: list[dict[str, Any]],
        *,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Identity exclusion vs the recovered-train-v0 inventory + the scoped
        geographic guard (``aerial_geospatial_overlap``). Held-out only for the
        geographic/near-dup/identity rules — ``prior_train`` gold measures
        trained-on geography BY DESIGN and is exempt (plan 4.4-5 scope)."""
        params = dict(params or {})
        inventory = _load_verified_inventory(params)

        inventory_stems: set[str] = set()
        for run_stems in (inventory.get("frame_stems") or {}).values():
            inventory_stems.update(str(stem) for stem in run_stems or [])
        tiles = list(inventory.get("tiles") or [])
        for row in tiles:
            stem = str(row.get("stem") or "").strip()
            if stem:
                inventory_stems.add(stem)
        inventory_shas = {
            str(row.get("content_sha256")) for row in tiles if row.get("content_sha256")
        }
        inventory_phashes = [str(row["phash"]) for row in tiles if row.get("phash")]
        inventory_centers = [
            center
            for center in (
                _center_from(entry) for entry in inventory.get("exif_frame_centers") or []
            )
            if center
        ]

        radius_m = float(params.get("geo_radius_m") or DEFAULT_GEO_RADIUS_M)
        prior_centers = [
            center
            for center in (
                _item_center(item) for item in gold_items if item.get("slice") == "prior_train"
            )
            if center
        ]
        train_centers = [
            center for center in (_item_center(row) for row in train_pool or []) if center
        ]

        violations: list[dict[str, Any]] = []
        for item in gold_items:
            if str(item.get("slice") or "") != "held_out_test":
                continue  # prior_train is exempt from every rule below
            item_id = str(item.get("item_id") or "")
            stem = _item_stem(item)

            if stem and stem in inventory_stems:
                violations.append(
                    _violation(
                        "identity_stem",
                        item_id,
                        f"held_out_test frame stem {stem!r} appears in the trained-on inventory",
                    )
                )
            content_sha = str(item.get("content_sha256") or "")
            if content_sha and content_sha in inventory_shas:
                violations.append(
                    _violation(
                        "identity_content_sha256",
                        item_id,
                        "held_out_test item content_sha256 matches a trained-on inventory tile",
                    )
                )
            item_phash = str(item.get("phash") or "")
            if item_phash:
                near = _nearest_hamming(item_phash, inventory_phashes)
                if near is not None and near < HELD_OUT_PHASH_HAMMING:
                    violations.append(
                        _violation(
                            "phash_near_dup",
                            item_id,
                            f"held_out_test item pHash within Hamming {near} (< {HELD_OUT_PHASH_HAMMING}) "
                            "of a trained-on inventory tile",
                        )
                    )
            center = _item_center(item)
            if center is not None:
                for other in (*inventory_centers, *prior_centers, *train_centers):
                    distance = haversine_m(center[0], center[1], other[0], other[1])
                    if distance <= radius_m:
                        violations.append(
                            _violation(
                                "aerial_geospatial_overlap",
                                item_id,
                                f"held_out_test frame center within {distance:.1f}m "
                                f"(radius {radius_m:.0f}m) of prior/trained-on geography",
                            )
                        )
                        break
        return violations

    # -------------------------------------------------------------- two-pass

    def evaluate(
        self,
        weights_uri: str,
        gold_manifest: dict[str, Any],
        slice: str | None = None,  # noqa: A002 - plan-3.5 contract name
    ) -> dict[str, Any]:
        """Two-pass kernel (``yolov5_two_pass/v1``): val.py at conf 0.001 per
        populated slice (curve metrics), the production detect path at
        conf 0.25 over the gold tiles (operating-point metrics)."""
        started = time.monotonic()
        manifest = dict(gold_manifest or {})
        slices_spec = {
            str(name): dict(spec or {}) for name, spec in (manifest.get("slices") or {}).items()
        }
        eval_params = dict(manifest.get("params") or {})
        requested = REQUIRED_SLICES if slice in (None, "", "all") else (str(slice),)
        yolov5_path = self._resolve_yolov5_path(eval_params)
        workdir = Path(eval_params.get("workdir") or tempfile.mkdtemp(prefix="goldgate-eval-"))

        pass1_by_slice: dict[str, dict[str, Any]] = {}
        passes = 0
        for name in requested:
            spec = slices_spec.get(name) or {}
            if int(spec.get("label_count") or 0) <= 0 or not spec.get("data_yaml"):
                continue  # empty slice: still emitted by the assembler (null metrics)
            stdout = self._run_val_pass(
                weights_uri=weights_uri,
                data_yaml=str(spec["data_yaml"]),
                yolov5_path=yolov5_path,
                workdir=workdir / f"val_{name}",
                python=str(eval_params.get("python") or sys.executable),
            )
            pass1_by_slice[name] = parse_val_output(stdout, class_names=CLASS_NAMES)
            passes += 1

        op_metrics = self._run_operating_point_pass(
            weights_uri=weights_uri,
            slices_spec=slices_spec,
            requested=requested,
            yolov5_path=yolov5_path,
            workdir=workdir,
        )
        if op_metrics is not None:
            passes += 1

        return assemble_detection_metrics(
            class_names=CLASS_NAMES,
            slices=slices_spec,
            pass1_by_slice=pass1_by_slice,
            op_metrics=op_metrics,
            operating_point={"conf": OP_CONF, "iou": OP_IOU},
            passes=passes,
            wall_clock_s=time.monotonic() - started,
        )

    def _resolve_yolov5_path(self, params: dict[str, Any]) -> Path:
        configured = str(params.get("yolov5_path") or "").strip()
        if configured:
            return Path(configured)
        try:
            return RareSpotConfig.from_env().yolov5_path
        except Exception:
            return Path("/opt/rarespot/yolov5")

    def _run_val_pass(
        self,
        *,
        weights_uri: str,
        data_yaml: str,
        yolov5_path: Path,
        workdir: Path,
        python: str,
    ) -> str:
        """Pass 1: val.py at conf 0.001 — its only valid regime for mAP.
        NEVER pass conf 0.25 to val.py (plan 7.3)."""
        workdir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
        env["YOLOv5_AUTOINSTALL"] = "false"
        env["YOLOV5_CONFIG_DIR"] = str((workdir / ".yolov5-config").resolve())
        env["PYTHONPATH"] = f"{yolov5_path}{os.pathsep}{env.get('PYTHONPATH', '')}"
        command = [
            python,
            str(yolov5_path / "val.py"),
            "--weights",
            str(weights_uri),
            "--data",
            data_yaml,
            "--imgsz",
            str(TILE_SIZE),
            "--conf-thres",
            str(CURVE_CONF),
            "--iou-thres",
            "0.6",
            "--task",
            "val",
            "--verbose",
            "--project",
            str(workdir),
            "--name",
            "val",
            "--exist-ok",
        ]
        result = subprocess.run(
            command,
            cwd=str(yolov5_path),
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        (workdir / "val.stdout.log").write_text(result.stdout or "", encoding="utf-8")
        (workdir / "val.stderr.log").write_text(result.stderr or "", encoding="utf-8")
        if result.returncode != 0:
            tail = (result.stderr or result.stdout or "").strip()[-4000:]
            raise RuntimeError(f"yolov5 val.py pass failed: {tail}")
        return f"{result.stdout or ''}\n{result.stderr or ''}"

    def _run_operating_point_pass(
        self,
        *,
        weights_uri: str,
        slices_spec: dict[str, dict[str, Any]],
        requested: tuple[str, ...],
        yolov5_path: Path,
        workdir: Path,
    ) -> dict[str, Any] | None:
        """Pass 2: the actual production path (run_yolov5_detect + classwise NMS
        at conf 0.25) plus a fresh IoU matcher — the numbers that measure the
        shipped model."""
        gt_by_tile: dict[str, list[dict[str, Any]]] = {}
        pred_by_tile: dict[str, list[dict[str, Any]]] = {}
        ran_any = False
        for name in requested:
            spec = slices_spec.get(name) or {}
            tiles_dir = str(spec.get("tiles_dir") or "")
            if int(spec.get("label_count") or 0) <= 0 or not tiles_dir:
                continue
            labels_dir = Path(str(spec.get("labels_dir") or ""))
            out_dir = workdir / f"op_{name}"
            out_dir.mkdir(parents=True, exist_ok=True)
            config = RareSpotConfig.from_paths(
                weights_path=weights_uri,
                yolov5_path=yolov5_path,
                artifact_root=out_dir,
                tile_size=TILE_SIZE,
                conf=OP_CONF,
                iou=OP_IOU,
                spectral=False,
                stability=False,
            )
            detect_labels = run_yolov5_detect(
                source_dir=Path(tiles_dir), output_dir=out_dir, config=config
            )
            ran_any = True
            for tile_path in sorted(Path(tiles_dir).glob("*.jpg")):
                key = f"{name}/{tile_path.stem}"
                gt_by_tile[key] = _read_yolo_rows(
                    labels_dir / f"{tile_path.stem}.txt", with_conf=False
                )
                pred_by_tile[key] = _read_yolo_rows(
                    detect_labels / f"{tile_path.stem}.txt", with_conf=True
                )
        if not ran_any:
            return None
        return match_operating_point(
            gt_by_tile=gt_by_tile,
            pred_by_tile=pred_by_tile,
            class_names=CLASS_NAMES,
            iou_threshold=OP_MATCH_IOU,
        )

    # -------------------------------------------------------------- smoke/train

    def smoke_test(self, weights_uri: str, gold_sample: Any) -> None:
        """Plan 6.2: assert names/nc on the checkpoint, then one real detect
        pass over a sample tile dir asserting parseable YOLO-txt labels."""
        from ultra_deepagents.rarespot.tiling import yolo_line_to_xyxy
        from ultra_deepagents.training.checkpoint_opt import inspect_checkpoint

        sample = (
            dict(gold_sample) if isinstance(gold_sample, dict) else {"tiles_dir": str(gold_sample)}
        )
        tiles_dir = Path(str(sample.get("tiles_dir") or ""))
        if not tiles_dir.is_dir():
            raise RuntimeError(f"smoke test needs a gold sample tile dir; got {tiles_dir}")
        yolov5_path = self._resolve_yolov5_path(sample)

        report = inspect_checkpoint(Path(weights_uri), yolov5_path)
        names = _normalize_names(report.get("model_names"))
        nc = report.get("model_nc")
        expected = {0: "prairie_dog", 1: "burrow"}
        if names != expected or (nc is not None and int(nc) != 2):
            raise RuntimeError(
                f"candidate rejected: checkpoint names/nc mismatch (names={names!r}, nc={nc!r}; "
                f"expected names={expected!r}, nc=2)"
            )

        out_dir = Path(str(sample.get("workdir") or tempfile.mkdtemp(prefix="goldgate-smoke-")))
        config = RareSpotConfig.from_paths(
            weights_path=weights_uri,
            yolov5_path=yolov5_path,
            artifact_root=out_dir,
            tile_size=TILE_SIZE,
            conf=OP_CONF,
            iou=OP_IOU,
            spectral=False,
            stability=False,
        )
        labels_dir = run_yolov5_detect(source_dir=tiles_dir, output_dir=out_dir, config=config)
        label_files = sorted(labels_dir.glob("*.txt"))
        if not label_files:
            raise RuntimeError("smoke test produced no detection labels on the gold sample")
        parsed_any = False
        for label_file in label_files:
            for line in label_file.read_text(encoding="utf-8", errors="ignore").splitlines():
                if yolo_line_to_xyxy(
                    line=line, width=TILE_SIZE, height=TILE_SIZE, class_names=CLASS_NAMES
                ):
                    parsed_any = True
                    break
            if parsed_any:
                break
        if not parsed_any:
            raise RuntimeError(
                "smoke test labels are not parseable YOLO-txt (format drift would silently "
                "break detect.py/overlays/GObject-XML parsing)"
            )

    def train(self, ctx: TrainContext) -> dict[str, Any]:
        """M3 builder: validate + build the exact plan-6.2 command and the
        BN-freeze patch now; GPU execution is the lambda worker deployment."""
        build_finetune_command(dict(ctx.params or {}))
        bn_freeze_patch_source()
        raise NotImplementedError("GPU execution lands on the lambda worker deployment")


# ------------------------------------------------------------- tiling helpers


def _gray_fill_ignore_regions(
    crop: Image.Image, tile: Tile, ignore_boxes: list[dict[str, Any]]
) -> None:
    """Fill each ignore box's pixels with letterbox gray on the tile BEFORE it
    is saved — train tiles AND gold eval tiles identically (plan 5b). Box→pixel
    expansion rule: round-out (floor mins, ceil maxes)."""
    for box in ignore_boxes:
        x0 = max(math.floor(float(box["x0"])), tile.x0)
        y0 = max(math.floor(float(box["y0"])), tile.y0)
        x1 = min(math.ceil(float(box["x1"])), tile.x1)
        y1 = min(math.ceil(float(box["y1"])), tile.y1)
        if x1 <= x0 or y1 <= y0:
            continue
        crop.paste(GRAY_FILL_RGB, (x0 - tile.x0, y0 - tile.y0, x1 - tile.x0, y1 - tile.y0))


def _tile_label_rows(tile: Tile, mapped_boxes: list[dict[str, Any]]) -> list[str]:
    """YOLO-txt rows for boxes whose CENTER falls inside the tile, clipped to
    the tile and normalized. A center in the overlap band labels both tiles —
    every tile is independently labeled, exactly what the trainer sees."""
    rows: list[str] = []
    for box in mapped_boxes:
        cx, cy = float(box["cx"]), float(box["cy"])
        if not (tile.x0 <= cx < tile.x1 and tile.y0 <= cy < tile.y1):
            continue
        bx0 = max(cx - float(box["w"]) / 2.0, float(tile.x0))
        by0 = max(cy - float(box["h"]) / 2.0, float(tile.y0))
        bx1 = min(cx + float(box["w"]) / 2.0, float(tile.x1))
        by1 = min(cy + float(box["h"]) / 2.0, float(tile.y1))
        if (bx1 - bx0) < 2.0 or (by1 - by0) < 2.0:
            continue
        tw, th = float(tile.width), float(tile.height)
        norm_cx = ((bx0 + bx1) / 2.0 - tile.x0) / tw
        norm_cy = ((by0 + by1) / 2.0 - tile.y0) / th
        norm_w = (bx1 - bx0) / tw
        norm_h = (by1 - by0) / th
        rows.append(f"{int(box['class_id'])} {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f}")
    return rows


def _class_presence(per_class_count: dict[str, int]) -> str:
    has_dog = per_class_count.get("prairie_dog", 0) > 0
    has_burrow = per_class_count.get("burrow", 0) > 0
    if has_dog and has_burrow:
        return "both"
    if has_dog:
        return "prairie_dog-only"
    if has_burrow:
        return "burrow-only"
    return "empty"


def _box_density(box_count: int) -> str:
    if box_count <= 0:
        return "empty"
    if box_count <= 3:
        return "sparse"
    if box_count <= 15:
        return "medium"
    return "dense"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ------------------------------------------------------------ leakage helpers


def _load_verified_inventory(params: dict[str, Any]) -> dict[str, Any]:
    """Verify sha BEFORE consuming (WORM is a mount convention, not an
    integrity check) and refuse cross-kernel pHash comparison (plan 5e)."""
    uri = str(params.get("exclusion_inventory_uri") or "").strip()
    expected_sha = str(params.get("inventory_sha256") or "").strip().lower()
    if not uri or not expected_sha:
        raise LeakageCheckRefused(
            "exclusion inventory not configured (params.exclusion_inventory_uri + "
            "params.inventory_sha256 are required) - refusing to freeze without the "
            "trained-on identity list"
        )
    raw = Path(uri).read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != expected_sha:
        raise LeakageCheckRefused(
            f"exclusion inventory sha mismatch: expected {expected_sha}, file is {actual_sha}"
        )
    inventory = json.loads(raw.decode("utf-8"))
    kernel = str(inventory.get("phash_kernel") or "")
    if kernel != PHASH_KERNEL:
        raise LeakageCheckRefused(
            f"pHash kernel mismatch: inventory={kernel!r} local={PHASH_KERNEL!r} - "
            "cross-kernel Hamming comparisons are meaningless"
        )
    return inventory


def _violation(check: str, item_id: str, reason: str) -> dict[str, Any]:
    return {
        "check": check,
        "defense": "aerial_geospatial_overlap",
        "item_id": item_id,
        "reason": reason,
    }


def _item_stem(item: dict[str, Any]) -> str:
    source_ref = item.get("source_ref") or {}
    return str(source_ref.get("frame_stem") or item.get("item_id") or "").strip()


def _item_center(item: dict[str, Any]) -> tuple[float, float] | None:
    geom = item.get("footprint_geom") or (item.get("metadata") or {}).get("footprint_geom") or {}
    return _center_from(geom)


def _center_from(entry: Any) -> tuple[float, float] | None:
    if not isinstance(entry, dict):
        return None
    lat = entry.get("center_lat", entry.get("lat"))
    lon = entry.get("center_lon", entry.get("lon"))
    if lat is None or lon is None:
        return None
    try:
        return float(lat), float(lon)
    except (TypeError, ValueError):
        return None


def _nearest_hamming(phash: str, candidates: list[str]) -> int | None:
    best: int | None = None
    for candidate in candidates:
        try:
            distance = hamming(phash, candidate)
        except ValueError:
            continue
        if best is None or distance < best:
            best = distance
            if best == 0:
                break
    return best


# ------------------------------------------------------- two-pass pure kernel


def parse_val_output(text: str, *, class_names: list[str]) -> dict[str, Any]:
    """Parse the vendored val.py summary table (pf = '%22s' + '%11i'*2 +
    '%11.3g'*4: Class, Images, Labels, P, R, mAP@.5, mAP@.5:.95). The P/R
    columns are at val.py's floating max-F1 operating point — recorded under
    explicit *_max_f1 names so they are never mistaken for conf-0.25 numbers."""
    known = {"all", *class_names}
    rows: dict[str, dict[str, float]] = {}
    for line in text.splitlines():
        tokens = line.split()
        if len(tokens) < 7:
            continue
        name = " ".join(tokens[:-6])
        if name not in known:
            continue
        try:
            parsed = [float(token) for token in tokens[-6:]]
        except ValueError:
            continue
        rows[name] = {
            "images": parsed[0],
            "labels": parsed[1],
            "precision_max_f1": parsed[2],
            "recall_max_f1": parsed[3],
            "ap50": parsed[4],
            "ap50_95": parsed[5],
        }
    if "all" not in rows:
        raise ValueError("val.py output carried no 'all' summary row - refusing to trust this pass")
    summary = rows["all"]
    per_class: dict[str, dict[str, Any]] = {}
    for name in class_names:
        row = rows.get(name)
        if row is None:
            continue
        per_class[name] = {
            "ap50": row["ap50"],
            "ap50_95": row["ap50_95"],
            "label_count": int(row["labels"]),
            "precision_max_f1": row["precision_max_f1"],
            "recall_max_f1": row["recall_max_f1"],
        }
    return {
        "map50": summary["ap50"],
        "map50_95": summary["ap50_95"],
        "images": int(summary["images"]),
        "labels": int(summary["labels"]),
        "per_class": per_class,
    }


def _read_yolo_rows(path: Path, *, with_conf: bool) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            row = {
                "class_id": int(float(parts[0])),
                "cx": float(parts[1]),
                "cy": float(parts[2]),
                "w": float(parts[3]),
                "h": float(parts[4]),
                "confidence": float(parts[5]) if with_conf and len(parts) > 5 else None,
            }
        except ValueError:
            continue
        rows.append(row)
    return rows


def _iou_cxcywh(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax0, ay0 = a["cx"] - a["w"] / 2.0, a["cy"] - a["h"] / 2.0
    ax1, ay1 = a["cx"] + a["w"] / 2.0, a["cy"] + a["h"] / 2.0
    bx0, by0 = b["cx"] - b["w"] / 2.0, b["cy"] - b["h"] / 2.0
    bx1, by1 = b["cx"] + b["w"] / 2.0, b["cy"] + b["h"] / 2.0
    inter_w = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    inter_h = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    union = (
        max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
        + max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
        - inter
    )
    return inter / union if union > 0.0 else 0.0


def match_operating_point(
    *,
    gt_by_tile: dict[str, list[dict[str, Any]]],
    pred_by_tile: dict[str, list[dict[str, Any]]],
    class_names: list[str],
    iou_threshold: float = OP_MATCH_IOU,
) -> dict[str, Any]:
    """Fresh greedy IoU matcher for pass-2 operating-point metrics: per-class
    tp/fp/fn at the deployed conf, plus FP-per-empty-frame (which val.py cannot
    produce — its fp is back-derived at max-F1 and empty frames append nothing)."""
    id_to_name = {index: name for index, name in enumerate(class_names)}
    per_class = {
        name: {"tp": 0, "fp": 0, "fn": 0, "predicted_count": 0, "label_count": 0}
        for name in class_names
    }
    empty_frame_count = 0
    fp_on_empty_frames = 0
    for tile, gt_rows in gt_by_tile.items():
        preds = sorted(
            pred_by_tile.get(tile, []),
            key=lambda row: float(row.get("confidence") or 0.0),
            reverse=True,
        )
        known_preds = [row for row in preds if row["class_id"] in id_to_name]
        if not gt_rows:
            empty_frame_count += 1
            for row in known_preds:
                stats = per_class[id_to_name[row["class_id"]]]
                stats["predicted_count"] += 1
                stats["fp"] += 1
                fp_on_empty_frames += 1
            continue
        matched: set[int] = set()
        for row in known_preds:
            stats = per_class[id_to_name[row["class_id"]]]
            stats["predicted_count"] += 1
            best_iou = 0.0
            best_index: int | None = None
            for index, gt_row in enumerate(gt_rows):
                if index in matched or gt_row["class_id"] != row["class_id"]:
                    continue
                iou = _iou_cxcywh(row, gt_row)
                if iou >= iou_threshold and iou > best_iou:
                    best_iou, best_index = iou, index
            if best_index is not None:
                matched.add(best_index)
                stats["tp"] += 1
            else:
                stats["fp"] += 1
        for index, gt_row in enumerate(gt_rows):
            name = id_to_name.get(gt_row["class_id"])
            if name is None:
                continue
            per_class[name]["label_count"] += 1
            if index not in matched:
                per_class[name]["fn"] += 1

    for stats in per_class.values():
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        stats["precision"] = round(tp / (tp + fp), 6) if (tp + fp) else None
        stats["recall"] = round(tp / (tp + fn), 6) if (tp + fn) else None
    total_tp = sum(stats["tp"] for stats in per_class.values())
    total_fp = sum(stats["fp"] for stats in per_class.values())
    total_fn = sum(stats["fn"] for stats in per_class.values())
    return {
        "precision": round(total_tp / (total_tp + total_fp), 6) if (total_tp + total_fp) else None,
        "recall": round(total_tp / (total_tp + total_fn), 6) if (total_tp + total_fn) else None,
        "fp_per_empty_frame": (
            round(fp_on_empty_frames / empty_frame_count, 6) if empty_frame_count else None
        ),
        "empty_frame_count": empty_frame_count,
        "fp_on_empty_frames": fp_on_empty_frames,
        "per_class": per_class,
    }


def assemble_detection_metrics(
    *,
    class_names: list[str],
    slices: dict[str, dict[str, Any]],
    pass1_by_slice: dict[str, dict[str, Any]],
    op_metrics: dict[str, Any] | None,
    operating_point: dict[str, Any],
    kernel: str = DETECTION_KERNEL,
    passes: int | None = None,
    wall_clock_s: float = 0.0,
) -> dict[str, Any]:
    """Pure aggregation over parsed pass outputs into the detection.v1 blob.

    The five mandatory conventions (plan 7.4): (1) per_slice always present
    with both slice names — an EMPTY held_out_test emits label_count 0 with
    null metric values, never an omitted key; (2) per_class always present
    keyed by manifest class names; (3) predicted_count/label_count always
    present per class under exactly those key names; (4) per-slice label_count
    present for the min-support rule; (5) null metric values only where the
    enclosing scope's label_count == 0.
    """
    per_slice: dict[str, dict[str, Any]] = {}
    populated: list[tuple[str, int, dict[str, Any]]] = []
    for name in REQUIRED_SLICES:
        spec = dict(slices.get(name) or {})
        label_count = max(0, int(spec.get("label_count") or 0))
        spec_per_class = {
            class_name: int((spec.get("per_class_label_count") or {}).get(class_name) or 0)
            for class_name in class_names
        }
        pass1 = pass1_by_slice.get(name)
        if label_count > 0 and pass1:
            populated.append((name, label_count, pass1))
            slice_classes: dict[str, dict[str, Any]] = {}
            for class_name in class_names:
                row = dict((pass1.get("per_class") or {}).get(class_name) or {})
                class_labels = int(row.get("label_count") or spec_per_class[class_name])
                slice_classes[class_name] = {
                    "ap50": row.get("ap50") if class_labels > 0 else None,
                    "ap50_95": row.get("ap50_95") if class_labels > 0 else None,
                    "label_count": class_labels,
                }
            per_slice[name] = {
                "map50": pass1.get("map50"),
                "map50_95": pass1.get("map50_95"),
                "label_count": label_count,
                "per_class": slice_classes,
            }
        else:
            per_slice[name] = {
                "map50": None,
                "map50_95": None,
                "label_count": label_count,
                "per_class": {
                    class_name: {
                        "ap50": None,
                        "ap50_95": None,
                        "label_count": spec_per_class[class_name],
                    }
                    for class_name in class_names
                },
            }

    op_per_class = dict((op_metrics or {}).get("per_class") or {})
    per_class: dict[str, dict[str, Any]] = {}
    for class_name in class_names:
        weight = 0
        ap50_acc = 0.0
        ap95_acc = 0.0
        for _, _, pass1 in populated:
            row = (pass1.get("per_class") or {}).get(class_name) or {}
            class_labels = int(row.get("label_count") or 0)
            if class_labels > 0 and row.get("ap50") is not None:
                ap50_acc += float(row["ap50"]) * class_labels
                ap95_acc += float(row.get("ap50_95") or 0.0) * class_labels
                weight += class_labels
        op_row = dict(op_per_class.get(class_name) or {})
        spec_total = sum(
            int(((slices.get(name) or {}).get("per_class_label_count") or {}).get(class_name) or 0)
            for name in REQUIRED_SLICES
        )
        per_class[class_name] = {
            "ap50": round(ap50_acc / weight, 6) if weight else None,
            "ap50_95": round(ap95_acc / weight, 6) if weight else None,
            "recall_at_op": op_row.get("recall"),
            "precision_at_op": op_row.get("precision"),
            "predicted_count": int(op_row.get("predicted_count") or 0),
            "label_count": int(op_row.get("label_count") or 0) or spec_total or weight,
        }

    total_weight = sum(label_count for _, label_count, _ in populated)
    if total_weight:
        aggregate_map50 = round(
            sum(
                float(pass1.get("map50") or 0.0) * label_count
                for _, label_count, pass1 in populated
            )
            / total_weight,
            6,
        )
        aggregate_map50_95 = round(
            sum(
                float(pass1.get("map50_95") or 0.0) * label_count
                for _, label_count, pass1 in populated
            )
            / total_weight,
            6,
        )
    else:
        aggregate_map50 = None
        aggregate_map50_95 = None

    return {
        "schema": DETECTION_SCHEMA,
        "aggregate": {
            "map50": aggregate_map50,
            "map50_95": aggregate_map50_95,
            "precision_at_op": (op_metrics or {}).get("precision"),
            "recall_at_op": (op_metrics or {}).get("recall"),
            "fp_per_empty_frame": (op_metrics or {}).get("fp_per_empty_frame"),
            "operating_point": dict(operating_point),
        },
        "per_class": per_class,
        "per_slice": per_slice,
        "eval": {
            "passes": passes if passes is not None else len(populated) + (1 if op_metrics else 0),
            "kernel": kernel,
            "wall_clock_s": round(float(wall_clock_s), 3),
        },
    }


# ------------------------------------------------------------- finetune (M3)


def build_finetune_command(params: dict[str, Any]) -> list[str]:
    """The exact plan-6.2 command: warm-start weights (NEVER ''), assembled
    data yaml, the self-authored checked-in hyp, imgsz 512, 40 epochs (60 above
    1000 new tiles), batch 16, freeze 10, patience 15, seed 0, save-period 5.
    Deliberately absent: --close-mosaic (the vendored train.py has no such arg)
    and --image-weights (destabilizes 2-class)."""
    weights = str(params.get("weights_uri") or "").strip()
    if not weights:
        raise ValueError("finetune requires warm-start weights_uri - never train from scratch ('')")
    data_yaml = str(params.get("data_yaml") or "").strip()
    if not data_yaml:
        raise ValueError("finetune requires the assembled data_yaml")
    run_dir = str(params.get("run_dir") or "").strip()
    if not run_dir:
        raise ValueError("finetune requires run_dir")
    yolov5_path = Path(str(params.get("yolov5_path") or "/opt/rarespot/yolov5"))
    hyp_path = str(params.get("hyp_path") or HYP_PATH)
    new_tile_count = int(params.get("new_tile_count") or 0)
    epochs = 60 if new_tile_count > 1000 else 40
    batch_size = int(params.get("batch_size") or 16)
    return [
        str(params.get("python") or sys.executable),
        str(yolov5_path / "train.py"),
        "--weights",
        weights,
        "--data",
        data_yaml,
        "--hyp",
        hyp_path,
        "--imgsz",
        str(TILE_SIZE),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--freeze",
        "10",
        "--patience",
        "15",
        "--seed",
        "0",
        "--project",
        run_dir,
        "--name",
        "finetune",
        "--exist-ok",
        "--save-period",
        "5",
    ]


_BN_FREEZE_PATCH = """\
# GoldGate BN-freeze patch (plan 6.2): --freeze 10 sets requires_grad=False on
# layers 0-9 but frozen BatchNorm keeps updating running_mean/var toward the
# new-site batch on every forward pass -> silent forgetting on old sites.
# Apply AFTER train.py's freeze loop, before the epoch loop:
for _name, _module in model.named_modules():
    if any(_name == f"model.{_i}" or _name.startswith(f"model.{_i}.") for _i in range(10)):
        if isinstance(_module, torch.nn.modules.batchnorm._BatchNorm):
            _module.eval()
            _module.track_running_stats = False
"""


def bn_freeze_patch_source() -> str:
    """The ~5-line monkeypatch injected around the vendored train.py freeze
    loop: put frozen-layer BatchNorm into eval() with track_running_stats off."""
    return _BN_FREEZE_PATCH


def _normalize_names(raw: Any) -> dict[int, str] | None:
    if isinstance(raw, dict):
        try:
            return {int(key): str(value) for key, value in raw.items()}
        except (TypeError, ValueError):
            return None
    if isinstance(raw, (list, tuple)):
        return dict(enumerate(str(value) for value in raw))
    return None


register_adapter(RareSpotAdapter())
