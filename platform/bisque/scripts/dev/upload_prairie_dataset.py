#!/usr/bin/env python3
"""Bulk-upload the reviewed prairie dog dataset into BisQue.

The script uploads each labeled tile as a real BisQue image resource, attaches
top-level display rectangles plus the nested ``gt2`` layer expected by the
prairie continuous-learning sync, checkpoints progress to disk, and finally
creates a managed BisQue dataset from the uploaded members.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from lxml import etree
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from upload_prairie_tile import (  # noqa: E402
    _attach_annotations,
    _build_session,
    _count_nested_gt2,
    _load_yolo_boxes,
    _normalize_root,
)
from bqapi.util import save_blob  # noqa: E402


@dataclass(frozen=True)
class PrairiePair:
    key: str
    image_path: str
    label_path: str
    split: str


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upload the prairie reviewed dataset to BisQue")
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Local root containing images/ and labels/ for the prairie dataset.",
    )
    parser.add_argument("--root", default="http://localhost:8080", help="BisQue root URL")
    parser.add_argument(
        "--dataset-name",
        default="Prairie_Dog_Active_Learning",
        help="BisQue dataset name to create from the uploaded reviewed images.",
    )
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
        default="/tmp/prairie_dataset_upload",
        help="Directory for progress checkpoints and final summary JSON.",
    )
    parser.add_argument(
        "--state-file",
        default="state.json",
        help="Checkpoint filename under output-dir.",
    )
    parser.add_argument(
        "--summary-file",
        default="summary.json",
        help="Final summary filename under output-dir.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on the number of reviewed images to upload (0 = all labeled images).",
    )
    parser.add_argument(
        "--replace-dataset",
        action="store_true",
        help="Delete any existing BisQue datasets with the same name before creating the new one.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="Write progress state every N successful uploads.",
    )
    return parser


def _scan_reviewed_pairs(dataset_root: Path, *, limit: int = 0) -> list[PrairiePair]:
    labels_root = dataset_root / "labels"
    images_root = dataset_root / "images"
    if not labels_root.exists():
        raise RuntimeError(f"Labels directory not found: {labels_root}")
    if not images_root.exists():
        raise RuntimeError(f"Images directory not found: {images_root}")

    pairs: list[PrairiePair] = []
    for label_path in sorted(labels_root.rglob("*.txt")):
        rel = label_path.relative_to(labels_root)
        image_path = images_root / rel.with_suffix(".jpg")
        if not image_path.exists():
            raise RuntimeError(f"Missing image for label {label_path}: expected {image_path}")
        key = rel.with_suffix(".jpg").as_posix()
        split = rel.parts[0] if rel.parts else "unknown"
        pairs.append(
            PrairiePair(
                key=key,
                image_path=str(image_path),
                label_path=str(label_path),
                split=split,
            )
        )
        if limit > 0 and len(pairs) >= limit:
            break
    if not pairs:
        raise RuntimeError(f"No reviewed prairie labels found under {labels_root}")
    return pairs


def _load_state(path: Path, *, dataset_name: str, dataset_root: Path, total_pairs: int) -> dict[str, Any]:
    if path.exists():
        try:
            payload = json.loads(path.read_text())
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
    return {
        "dataset_name": dataset_name,
        "dataset_root": str(dataset_root),
        "pair_count": total_pairs,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "uploaded": {},
        "failed": {},
        "dataset": None,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _find_matching_datasets(session: Any, root: str, dataset_name: str) -> list[etree._Element]:
    response = session.fetchxml(f"{root}/data_service/dataset", view="deep")
    if response is None:
        return []
    if getattr(response, "tag", None) == "dataset":
        nodes = [response]
    else:
        nodes = list(response.findall(".//dataset"))
    wanted = dataset_name.strip()
    return [node for node in nodes if str(node.get("name") or "").strip() == wanted]


def _delete_matching_datasets(session: Any, root: str, dataset_name: str) -> list[str]:
    removed: list[str] = []
    for node in _find_matching_datasets(session, root, dataset_name):
        uri = str(node.get("uri") or "").strip()
        if not uri:
            continue
        session.deletexml(uri)
        removed.append(uri)
    return removed


def _create_dataset(session: Any, root: str, dataset_name: str, resource_uris: list[str]) -> dict[str, Any]:
    dataset = etree.Element("dataset", name=dataset_name)
    etree.SubElement(dataset, "tag", name="prairie_active_learning", value="true")
    etree.SubElement(dataset, "tag", name="annotation_layer", value="gt2")
    etree.SubElement(dataset, "tag", name="managed_by", value="upload_prairie_dataset.py")
    for index, resource_uri in enumerate(resource_uris):
        value = etree.SubElement(dataset, "value", type="object", index=str(index))
        value.text = resource_uri
    created = session.postxml(f"{root}/data_service/dataset", dataset)
    if created is None:
        raise RuntimeError("BisQue dataset creation returned no XML response")
    dataset_uri = str(created.get("uri") or "").strip()
    if not dataset_uri and len(created):
        dataset_uri = str(created[0].get("uri") or "").strip()
    if not dataset_uri:
        raise RuntimeError("BisQue dataset creation did not return a dataset URI")
    verify = session.fetchxml(dataset_uri, view="deep")
    dataset_node = verify if verify is not None else created
    member_count = len(verify.findall("./value")) if verify is not None else 0
    return {
        "dataset_uri": dataset_uri,
        "dataset_uniq": str(dataset_node.get("resource_uniq") or "").strip() or None,
        "member_count": member_count,
    }


def _upload_pair(session: Any, pair: PrairiePair) -> dict[str, Any]:
    image_path = Path(pair.image_path)
    label_path = Path(pair.label_path)
    with Image.open(image_path) as image_obj:
        width, height = image_obj.size
    boxes = _load_yolo_boxes(label_path, width=width, height=height)
    expected_counts = {"burrow": 0, "prairie_dog": 0}
    for box in boxes:
        expected_counts[box.label] = expected_counts.get(box.label, 0) + 1

    upload_resource = etree.Element("image", name=image_path.name)
    etree.SubElement(upload_resource, "tag", name="prairie_dataset_split", value=pair.split)
    etree.SubElement(upload_resource, "tag", name="prairie_source_key", value=pair.key)
    uploaded = save_blob(session, localfile=str(image_path), resource=upload_resource)
    if uploaded is None or uploaded.get("uri") is None:
        raise RuntimeError(f"BisQue upload did not return an image URI for {image_path}")
    resource_uri = str(uploaded.get("uri"))

    image_update = etree.Element("image", uri=resource_uri)
    _attach_annotations(image_update, boxes)
    session.postxml(resource_uri, image_update, method="POST")

    verify_xml = session.fetchxml(resource_uri, view="deep")
    if verify_xml is None:
        raise RuntimeError(f"Unable to verify uploaded BisQue image {resource_uri}")
    nested_counts = _count_nested_gt2(verify_xml)
    top_level_rectangles = len(verify_xml.findall("./rectangle"))
    if nested_counts != expected_counts:
        raise RuntimeError(
            f"Annotation verification mismatch for {pair.key}: expected {expected_counts}, got {nested_counts}"
        )
    if top_level_rectangles != len(boxes):
        raise RuntimeError(
            f"Display rectangle mismatch for {pair.key}: expected {len(boxes)}, got {top_level_rectangles}"
        )
    return {
        "key": pair.key,
        "split": pair.split,
        "image_path": str(image_path),
        "label_path": str(label_path),
        "resource_uri": resource_uri,
        "resource_uniq": str(verify_xml.get("resource_uniq") or "").strip(),
        "resource_name": str(verify_xml.get("name") or "").strip(),
        "image_size": [width, height],
        "box_count": len(boxes),
        "counts": expected_counts,
    }


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not dataset_root.exists():
        parser.error(f"Dataset root does not exist: {dataset_root}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / str(args.state_file)
    summary_path = output_dir / str(args.summary_file)

    pairs = _scan_reviewed_pairs(dataset_root, limit=max(0, int(args.limit or 0)))
    state = _load_state(
        state_path,
        dataset_name=str(args.dataset_name),
        dataset_root=dataset_root,
        total_pairs=len(pairs),
    )

    session = _build_session(args)
    bisque_root = _normalize_root(str(args.root))

    uploaded = state.setdefault("uploaded", {})
    failed = state.setdefault("failed", {})
    successes_since_checkpoint = 0

    for index, pair in enumerate(pairs, start=1):
        if pair.key in uploaded:
            continue
        try:
            result = _upload_pair(session, pair)
            uploaded[pair.key] = result
            failed.pop(pair.key, None)
            successes_since_checkpoint += 1
        except Exception as exc:
            failed[pair.key] = {
                "image_path": pair.image_path,
                "label_path": pair.label_path,
                "error": str(exc),
            }
        if successes_since_checkpoint >= max(1, int(args.checkpoint_every or 1)):
            state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            _write_json(state_path, state)
            successes_since_checkpoint = 0
            print(
                json.dumps(
                    {
                        "progress": {
                            "processed": index,
                            "total": len(pairs),
                            "uploaded": len(uploaded),
                            "failed": len(failed),
                        }
                    }
                ),
                flush=True,
            )

    if successes_since_checkpoint or not state_path.exists():
        state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _write_json(state_path, state)

    resource_uris = [str(uploaded[key]["resource_uri"]) for key in sorted(uploaded)]
    removed_datasets: list[str] = []
    if args.replace_dataset:
        removed_datasets = _delete_matching_datasets(session, bisque_root, str(args.dataset_name))
    dataset_info = _create_dataset(session, bisque_root, str(args.dataset_name), resource_uris)
    state["dataset"] = dataset_info
    state["removed_datasets"] = removed_datasets
    state["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write_json(state_path, state)

    class_counts = {"burrow": 0, "prairie_dog": 0}
    for row in uploaded.values():
        counts = row.get("counts") if isinstance(row, dict) else {}
        if not isinstance(counts, dict):
            continue
        class_counts["burrow"] += int(counts.get("burrow") or 0)
        class_counts["prairie_dog"] += int(counts.get("prairie_dog") or 0)

    summary = {
        "dataset_root": str(dataset_root),
        "dataset_name": str(args.dataset_name),
        "reviewed_pairs": len(pairs),
        "uploaded_images": len(uploaded),
        "failed_images": len(failed),
        "class_counts": class_counts,
        "dataset": dataset_info,
        "removed_datasets": removed_datasets,
        "state_path": str(state_path),
    }
    _write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
