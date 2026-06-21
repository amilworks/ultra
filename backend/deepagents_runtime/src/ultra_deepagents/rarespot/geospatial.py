"""Geospatial survey map for multi-image RareSpot runs.

Survey images usually carry EXIF GPS (camera lat/lon). When a run covers a collection, this
plots each image at its real coordinate sized by detection count and coloured by reliability,
and computes real-world survey metrics (extent, point spacing via haversine, density,
prairie-dog:burrow ratio). This turns per-image counts into a spatial picture of where the
colony activity is + how trustworthy each survey point is.

Per-DETECTION geographic placement (offsetting each box within the image footprint) needs the
ground sample distance / altitude + heading, which a bare EXIF lat/lon does not provide; that
is a follow-up. Here each image is one georeferenced survey point.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

EARTH_RADIUS_M = 6_371_000.0


def _coord_from_dms(dms: Any, ref: Any) -> float | None:
    if not dms or ref is None:
        return None
    try:
        degrees = float(dms[0])
        minutes = float(dms[1])
        seconds = float(dms[2])
    except (TypeError, ValueError, IndexError):
        return None
    decimal = degrees + minutes / 60.0 + seconds / 3600.0
    reference = ref.decode() if isinstance(ref, bytes) else str(ref)
    if reference.strip().upper() in {"S", "W"}:
        decimal = -decimal
    return decimal


def read_exif_gps(image_path: Path) -> dict[str, float] | None:
    """Return {'lat', 'lon', 'alt'?} in decimal degrees / metres, or None if absent."""
    try:
        from PIL import Image
        from PIL.ExifTags import GPSTAGS

        with Image.open(image_path) as image:
            exif = image.getexif()
            if not exif:
                return None
            gps_ifd = exif.get_ifd(0x8825)  # GPSInfo
    except Exception:
        return None
    if not gps_ifd:
        return None
    gps = {GPSTAGS.get(key, key): value for key, value in gps_ifd.items()}
    lat = _coord_from_dms(gps.get("GPSLatitude"), gps.get("GPSLatitudeRef"))
    lon = _coord_from_dms(gps.get("GPSLongitude"), gps.get("GPSLongitudeRef"))
    if lat is None or lon is None:
        return None
    result: dict[str, float] = {"lat": lat, "lon": lon}
    if "GPSAltitude" in gps:
        try:
            altitude = float(gps["GPSAltitude"])
            if gps.get("GPSAltitudeRef") in (1, b"\x01"):
                altitude = -altitude
            result["alt"] = altitude
        except (TypeError, ValueError):
            pass
    return result


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


def _survey_points(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for prediction in predictions:
        image_path = Path(str(prediction.get("input_path") or ""))
        gps = read_exif_gps(image_path)
        if gps is None:
            continue
        boxes = list(prediction.get("boxes") or [])
        label_counts = {"trusted": 0, "borderline": 0, "unstable": 0}
        for box in boxes:
            label = str(box.get("stability_label") or "")
            if label in label_counts:
                label_counts[label] += 1
        total = len(boxes)
        trusted_fraction = (label_counts["trusted"] / total) if total else 0.0
        points.append(
            {
                "file_name": image_path.name,
                "lat": float(gps["lat"]),
                "lon": float(gps["lon"]),
                "alt": gps.get("alt"),
                "counts_by_class": dict(prediction.get("class_counts") or {}),
                "total_detections": total,
                "stability_label_counts": label_counts,
                "trusted_fraction": round(trusted_fraction, 4),
            }
        )
    return points


def _distance_metrics(points: list[dict[str, Any]]) -> dict[str, Any]:
    if len(points) < 2:
        return {"point_count": len(points)}
    nearest: list[float] = []
    max_pairwise = 0.0
    for i, a in enumerate(points):
        best = math.inf
        for j, b in enumerate(points):
            if i == j:
                continue
            distance = haversine_m(a["lat"], a["lon"], b["lat"], b["lon"])
            best = min(best, distance)
            max_pairwise = max(max_pairwise, distance)
        if best != math.inf:
            nearest.append(best)
    return {
        "point_count": len(points),
        "survey_extent_m": round(max_pairwise, 2),
        "nearest_neighbor_m": {
            "min": round(min(nearest), 2),
            "mean": round(sum(nearest) / len(nearest), 2),
            "max": round(max(nearest), 2),
        },
    }


def build_geospatial_summary(
    *,
    predictions: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any] | None:
    """Build the survey map + metrics across all georeferenced images in a run. Returns None
    when fewer than one image carries EXIF GPS (nothing meaningful to plot)."""
    points = _survey_points(predictions)
    if not points:
        return None

    totals_by_class: dict[str, int] = {}
    label_totals = {"trusted": 0, "borderline": 0, "unstable": 0}
    for point in points:
        for class_name, count in (point["counts_by_class"] or {}).items():
            totals_by_class[str(class_name)] = totals_by_class.get(str(class_name), 0) + int(count)
        for label, count in point["stability_label_counts"].items():
            label_totals[label] += int(count)
    prairie = int(totals_by_class.get("prairie_dog", 0))
    burrow = int(totals_by_class.get("burrow", 0))
    metrics = _distance_metrics(points)
    metrics["totals_by_class"] = totals_by_class
    metrics["dog_per_burrow"] = round(prairie / burrow, 3) if burrow else None
    metrics["reliability_totals"] = label_totals

    try:
        map_path = _render_survey_map(points, output_dir)
    except Exception:
        # A plotting failure (e.g. matplotlib unavailable) must not lose the metrics.
        map_path = None
    return {
        "georeferenced_image_count": len(points),
        "points": points,
        "metrics": metrics,
        "map_path": str(map_path) if map_path else None,
    }


def _render_survey_map(points: list[dict[str, Any]], output_dir: Path) -> Path | None:
    if not points:
        return None
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    lons = [point["lon"] for point in points]
    lats = [point["lat"] for point in points]
    totals = [int(point["total_detections"]) for point in points]
    reliability = [float(point["trusted_fraction"]) for point in points]
    sizes = [60.0 + 28.0 * total for total in totals]

    fig, ax = plt.subplots(figsize=(9.5, 8.0))
    scatter = ax.scatter(
        lons,
        lats,
        s=sizes,
        c=reliability,
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        edgecolors="#1f2937",
        linewidths=0.8,
        alpha=0.9,
    )
    for point in points:
        ax.annotate(
            str(point["total_detections"]),
            (point["lon"], point["lat"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
            color="#111827",
        )
    # Roughly equalise the aspect ratio for the survey's latitude so spacing looks true.
    mean_lat = sum(lats) / len(lats)
    ax.set_aspect(1.0 / max(0.1, math.cos(math.radians(mean_lat))))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"RareSpot survey map — {len(points)} georeferenced image(s)\nmarker size = detections, colour = trusted fraction")
    ax.grid(linestyle="--", alpha=0.25)
    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
    colorbar.set_label("Trusted fraction (per-detection stability)")
    fig.tight_layout()
    map_path = output_dir / "survey_detection_map.png"
    fig.savefig(map_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return map_path
