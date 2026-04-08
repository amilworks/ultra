#!/usr/bin/env python3
"""Upload one prairie tile to BisQue and attach YOLO boxes as gobjects.

This follows the BQAPI image-upload path used by legacy BisQue modules:
- authenticate with BQSession
- upload the file via save_blob()/postblob() with an explicit <image> resource
- update the created image XML with top-level display rectangles plus nested gt2 boxes
- verify the saved XML and image-service fetches
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests
from lxml import etree
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
BQAPI_PATH = REPO_ROOT / "source" / "bqapi"
SCRIPT_DIR = Path(__file__).resolve().parent
if str(BQAPI_PATH) not in sys.path:
    sys.path.insert(0, str(BQAPI_PATH))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from bqapi import BQSession  # noqa: E402
from bqapi.util import save_blob  # noqa: E402
from test_auth_browser_flow import (  # noqa: E402
    _extract_keycloak_form,
    _follow_redirects,
    _location,
    _request,
)


YOLO_CLASS_NAMES = {"0": "prairie_dog", "1": "burrow"}
DISPLAY_COLORS = {"prairie_dog": "#FF3B30", "burrow": "#00A3FF"}


@dataclass(frozen=True)
class YoloBox:
    label: str
    x1: float
    y1: float
    x2: float
    y2: float


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upload a prairie tile to BisQue")
    parser.add_argument("--root", default="http://localhost:8080", help="BisQue root URL")
    parser.add_argument("--image", required=True, help="Path to the source JPG")
    parser.add_argument("--label", required=True, help="Path to the YOLO .txt label file")
    parser.add_argument("--user", default="admin", help="BisQue username")
    parser.add_argument("--password", default="admin", help="BisQue password")
    parser.add_argument("--token", default=os.environ.get("BISQUE_BEARER_TOKEN", ""), help="Bearer token")
    parser.add_argument(
        "--auth-mode",
        choices=["auto", "cookie", "token", "local"],
        default="auto",
        help="Authentication mode. auto prefers an OIDC browser-style cookie session.",
    )
    parser.add_argument(
        "--token-script",
        default=str(REPO_ROOT / "scripts" / "dev" / "oidc_get_token.py"),
        help="Path to the OIDC helper used when auth-mode=auto",
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/prairie_upload_verify",
        help="Directory for verification artifacts",
    )
    return parser


def _normalize_root(root: str) -> str:
    return root.rstrip("/")


def _resolve_token(args: argparse.Namespace) -> str:
    token = str(args.token or "").strip()
    if token:
        return token
    token_script = Path(str(args.token_script or "")).expanduser()
    if not token_script.exists():
        raise RuntimeError(f"Token script not found: {token_script}")
    command = [
        sys.executable,
        str(token_script),
        "--user",
        str(args.user),
        "--password",
        str(args.password),
        "--print-access-token",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    token = completed.stdout.strip()
    if not token:
        raise RuntimeError("OIDC token helper returned an empty access token")
    return token


def _build_cookie_session(args: argparse.Namespace) -> BQSession:
    root = _normalize_root(str(args.root))
    browser = requests.Session()
    browser.headers.update({"User-Agent": "bisque-prairie-upload/1.0"})

    login_url = f"{root}/auth_service/login"
    login_response = _request(browser, "GET", login_url, timeout=15, allow_redirects=False)
    if login_response.status_code not in {301, 302, 303, 307, 308}:
        raise RuntimeError(f"Expected OIDC redirect from /auth_service/login, got {login_response.status_code}")
    oidc_entry_url = _location(login_response, login_url)
    oidc_entry = _request(browser, "GET", oidc_entry_url, timeout=15, allow_redirects=False)
    if oidc_entry.status_code not in {301, 302, 303, 307, 308}:
        raise RuntimeError(f"Expected redirect from /auth_service/oidc_login, got {oidc_entry.status_code}")
    keycloak_url = _location(oidc_entry, oidc_entry_url)
    provider_page = _request(browser, "GET", keycloak_url, timeout=15, allow_redirects=False)
    if provider_page.status_code != 200:
        raise RuntimeError(f"Expected Keycloak login page, got {provider_page.status_code}")

    action_url, payload = _extract_keycloak_form(provider_page.text, provider_page.url)
    payload["username"] = str(args.user)
    payload["password"] = str(args.password)
    submit = _request(browser, "POST", action_url, timeout=15, allow_redirects=False, data=payload)
    final_response, _ = _follow_redirects(browser, submit, timeout=15)
    if final_response.status_code != 200:
        raise RuntimeError(f"OIDC login did not complete successfully: {final_response.status_code}")

    session = BQSession()
    session.bisque_root = root
    session.c.root = root
    session.c.headers.update(browser.headers)
    session.c.cookies.update(browser.cookies)
    session._load_services()
    session._check_session()
    return session


def _build_session(args: argparse.Namespace) -> BQSession:
    root = _normalize_root(str(args.root))
    auth_mode = str(args.auth_mode)
    if auth_mode in {"auto", "cookie"}:
        try:
            return _build_cookie_session(args)
        except Exception:
            if auth_mode == "cookie":
                raise
    if auth_mode in {"auto", "token"}:
        try:
            token = _resolve_token(args)
            return BQSession().init_token(token, bisque_root=root, create_mex=False)
        except Exception:
            if auth_mode == "token":
                raise
    return BQSession().init_local(str(args.user), str(args.password), bisque_root=root, create_mex=False)


def _load_yolo_boxes(label_path: Path, *, width: int, height: int) -> list[YoloBox]:
    boxes: list[YoloBox] = []
    for line in label_path.read_text().splitlines():
        row = line.strip()
        if not row:
            continue
        parts = row.split()
        if len(parts) != 5:
            raise RuntimeError(f"Invalid YOLO row in {label_path}: {row}")
        class_id, x_center, y_center, box_w, box_h = parts
        if class_id not in YOLO_CLASS_NAMES:
            raise RuntimeError(f"Unsupported YOLO class id {class_id!r} in {label_path}")
        cx = float(x_center) * width
        cy = float(y_center) * height
        bw = float(box_w) * width
        bh = float(box_h) * height
        x1 = max(0.0, cx - bw / 2.0)
        y1 = max(0.0, cy - bh / 2.0)
        x2 = min(float(width), cx + bw / 2.0)
        y2 = min(float(height), cy + bh / 2.0)
        boxes.append(YoloBox(label=YOLO_CLASS_NAMES[class_id], x1=x1, y1=y1, x2=x2, y2=y2))
    if not boxes:
        raise RuntimeError(f"No YOLO boxes found in {label_path}")
    return boxes


def _append_rectangle(parent: etree._Element, box: YoloBox, *, name: str | None = None, add_tags: bool = False) -> None:
    rectangle = etree.SubElement(parent, "rectangle")
    if name:
        rectangle.set("name", name)
    if add_tags:
        etree.SubElement(rectangle, "tag", name="label", value=box.label)
        etree.SubElement(rectangle, "tag", name="annotation_layer", value="gt2_display")
        etree.SubElement(
            rectangle,
            "tag",
            name="color",
            type="color",
            value=DISPLAY_COLORS.get(box.label, "#FF3B30"),
        )
    etree.SubElement(rectangle, "vertex", index="0", x=f"{box.x1:.3f}", y=f"{box.y1:.3f}", z="0.0", t="0.0")
    etree.SubElement(rectangle, "vertex", index="1", x=f"{box.x2:.3f}", y=f"{box.y2:.3f}", z="0.0", t="0.0")


def _attach_annotations(image_xml: etree._Element, boxes: Iterable[YoloBox]) -> etree._Element:
    boxes = list(boxes)
    for index, box in enumerate(boxes):
        _append_rectangle(image_xml, box, name=f"{box.label}_{index:03d}", add_tags=True)

    gt2 = etree.SubElement(image_xml, "gobject", name="gt2")
    by_label: dict[str, list[YoloBox]] = {"burrow": [], "prairie_dog": []}
    for box in boxes:
        by_label.setdefault(box.label, []).append(box)
    for label in ("burrow", "prairie_dog"):
        class_node = etree.SubElement(gt2, "gobject", name=label)
        for box in by_label.get(label, []):
            _append_rectangle(class_node, box)
    return image_xml


def _count_nested_gt2(root: etree._Element) -> dict[str, int]:
    counts = {"burrow": 0, "prairie_dog": 0}
    gt2 = root.find("./gobject[@name='gt2']")
    if gt2 is None:
        return counts
    for class_node in gt2.findall("./gobject"):
        label = str(class_node.get("name") or "").strip()
        if label in counts:
            counts[label] = len(class_node.findall("./rectangle"))
    return counts


def _verify_render(session: BQSession, resource_uri: str, output_dir: Path) -> tuple[dict[str, object], dict[str, object]]:
    image = session.load(resource_uri)
    if image is None:
        raise RuntimeError(f"Failed to load uploaded image resource: {resource_uri}")
    meta_raw = image.pixels().meta().fetch()
    meta_xml = etree.fromstring(meta_raw)
    def _meta_value(name: str) -> str | None:
        node = meta_xml.find(f".//tag[@name='{name}']")
        return node.get("value") if node is not None else None
    meta_summary = {
        "image_num_x": _meta_value("image_num_x"),
        "image_num_y": _meta_value("image_num_y"),
        "image_num_z": _meta_value("image_num_z"),
        "image_num_t": _meta_value("image_num_t"),
        "image_num_c": _meta_value("image_num_c"),
    }
    preview_path = output_dir / f"{image.resource_uniq}_preview.jpg"
    image.pixels().format("jpeg").fetch(str(preview_path))
    thumbnail_path = output_dir / f"{image.resource_uniq}_thumb.jpg"
    image.pixels().command("thumbnail", "256,256,BC,,jpg").fetch(str(thumbnail_path))
    with Image.open(preview_path) as preview:
        preview_info = {
            "preview_path": str(preview_path),
            "preview_size": list(preview.size),
            "thumbnail_path": str(thumbnail_path),
            "resource_uniq": image.resource_uniq,
        }
    return meta_summary, preview_info


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    label_path = Path(args.label).expanduser().resolve()
    if not image_path.exists():
        parser.error(f"Image file does not exist: {image_path}")
    if not label_path.exists():
        parser.error(f"Label file does not exist: {label_path}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as image_obj:
        width, height = image_obj.size
    boxes = _load_yolo_boxes(label_path, width=width, height=height)

    session = _build_session(args)
    try:
        upload_resource = etree.Element("image", name=image_path.name)
        uploaded = save_blob(session, localfile=str(image_path), resource=upload_resource)
        if uploaded is None or uploaded.get("uri") is None:
            raise RuntimeError("BisQue upload did not return an image URI")
        resource_uri = str(uploaded.get("uri"))

        image_update = etree.Element("image", uri=resource_uri)
        _attach_annotations(image_update, boxes)
        session.postxml(resource_uri, image_update, method="POST")

        verify_xml = session.fetchxml(resource_uri, view="deep")
        xml_path = output_dir / f"{Path(image_path.name).stem}.xml"
        xml_path.write_bytes(etree.tostring(verify_xml, pretty_print=True))

        top_level_rectangles = verify_xml.findall("./rectangle")
        nested_counts = _count_nested_gt2(verify_xml)
        meta_summary, preview_info = _verify_render(session, resource_uri, output_dir)

        summary = {
            "resource_uri": resource_uri,
            "resource_uniq": verify_xml.get("resource_uniq"),
            "resource_name": verify_xml.get("name"),
            "image_path": str(image_path),
            "label_path": str(label_path),
            "image_size": [width, height],
            "box_count": len(boxes),
            "top_level_rectangles": len(top_level_rectangles),
            "nested_gt2_counts": nested_counts,
            "xml_path": str(xml_path),
            "meta_summary": meta_summary,
            "preview": preview_info,
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    finally:
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
