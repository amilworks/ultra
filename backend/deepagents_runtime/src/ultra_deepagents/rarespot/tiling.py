from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Tile:
    x0: int
    y0: int
    x1: int
    y1: int
    width: int
    height: int
    grid_x: int
    grid_y: int


@dataclass(frozen=True)
class TileGrid:
    tiles: list[Tile]
    stride: int
    tile_size: int
    overlap: float


def clamp_overlap(value: float, *, default: float = 0.25) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = default
    return max(0.0, min(0.95, parsed))


def tile_start_positions(length: int, tile_size: int, stride: int) -> list[int]:
    size = max(1, int(length))
    tile = max(1, int(tile_size))
    step = max(1, int(stride))
    if size <= tile:
        return [0]
    starts = list(range(0, max(1, size - tile + 1), step))
    last = max(0, size - tile)
    if not starts or starts[-1] != last:
        starts.append(last)
    return starts


def build_sliding_tiles(
    *,
    width: int,
    height: int,
    tile_size: int,
    overlap: float = 0.25,
) -> TileGrid:
    tile = max(8, int(tile_size))
    overlap_clamped = clamp_overlap(overlap, default=0.25)
    stride = max(1, round(tile * (1.0 - overlap_clamped)))
    x_starts = tile_start_positions(width, tile, stride)
    y_starts = tile_start_positions(height, tile, stride)
    tiles: list[Tile] = []
    for grid_y, y0 in enumerate(y_starts):
        for grid_x, x0 in enumerate(x_starts):
            x1 = min(int(width), int(x0 + tile))
            y1 = min(int(height), int(y0 + tile))
            tiles.append(
                Tile(
                    x0=int(x0),
                    y0=int(y0),
                    x1=int(x1),
                    y1=int(y1),
                    width=int(max(1, x1 - x0)),
                    height=int(max(1, y1 - y0)),
                    grid_x=int(grid_x),
                    grid_y=int(grid_y),
                )
            )
    return TileGrid(tiles=tiles, stride=stride, tile_size=tile, overlap=overlap_clamped)


def remap_tile_box(
    xyxy: list[float] | tuple[float, float, float, float],
    *,
    tile_x0: int,
    tile_y0: int,
    image_width: int,
    image_height: int,
) -> list[float]:
    x1, y1, x2, y2 = [float(value) for value in xyxy[:4]]
    return [
        max(0.0, min(float(image_width), x1 + float(tile_x0))),
        max(0.0, min(float(image_height), y1 + float(tile_y0))),
        max(0.0, min(float(image_width), x2 + float(tile_x0))),
        max(0.0, min(float(image_height), y2 + float(tile_y0))),
    ]


def yolo_line_to_xyxy(
    *,
    line: str,
    width: int,
    height: int,
    class_names: list[str],
    min_pixels: float = 1.0,
) -> dict[str, Any] | None:
    tokens = [token for token in str(line or "").strip().split() if token]
    if len(tokens) < 5:
        return None
    try:
        class_id = int(float(tokens[0]))
        cx = float(tokens[1])
        cy = float(tokens[2])
        bw = float(tokens[3])
        bh = float(tokens[4])
        confidence = float(tokens[5]) if len(tokens) > 5 else None
    except Exception:
        return None
    x1 = (cx - bw / 2.0) * float(width)
    y1 = (cy - bh / 2.0) * float(height)
    x2 = (cx + bw / 2.0) * float(width)
    y2 = (cy + bh / 2.0) * float(height)
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if (x2 - x1) < float(min_pixels) or (y2 - y1) < float(min_pixels):
        return None
    class_name = class_names[class_id] if 0 <= class_id < len(class_names) else str(class_id)
    return {
        "class_id": class_id,
        "class_name": class_name,
        "confidence": confidence,
        "xyxy": [float(x1), float(y1), float(x2), float(y2)],
    }


def xyxy_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in a]
    bx1, by1, bx2, by2 = [float(value) for value in b]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter_area
    if denom <= 0.0:
        return 0.0
    return float(inter_area / denom)


def classwise_nms(rows: list[dict[str, Any]], *, iou_threshold: float) -> list[dict[str, Any]]:
    if not rows:
        return []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get("class_name") or row.get("class_id") or "").strip()
        if key:
            grouped.setdefault(key, []).append(row)
    threshold = max(0.01, min(0.99, float(iou_threshold)))
    kept_rows: list[dict[str, Any]] = []
    for group in grouped.values():
        ordered = sorted(group, key=lambda item: float(item.get("confidence") or 0.0), reverse=True)
        while ordered:
            candidate = ordered.pop(0)
            kept_rows.append(candidate)
            candidate_box = candidate.get("xyxy")
            if not isinstance(candidate_box, list) or len(candidate_box) != 4:
                continue
            survivors: list[dict[str, Any]] = []
            for row in ordered:
                row_box = row.get("xyxy")
                if not isinstance(row_box, list) or len(row_box) != 4:
                    continue
                if xyxy_iou([float(value) for value in candidate_box], [float(value) for value in row_box]) <= threshold:
                    survivors.append(row)
            ordered = survivors
    kept_rows.sort(key=lambda item: (-float(item.get("confidence") or 0.0), str(item.get("class_name") or "")))
    return kept_rows
