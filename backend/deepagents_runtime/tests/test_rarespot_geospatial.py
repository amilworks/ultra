"""Tests for the EXIF-GPS survey-map math + summary (torch-free; the map render degrades to
metrics-only when matplotlib is absent)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import ultra_deepagents.rarespot.geospatial as geo


def test_haversine_one_degree_latitude():
    # one degree of latitude is ~111.2 km anywhere
    assert abs(geo.haversine_m(0.0, 0.0, 1.0, 0.0) - 111_195) < 600
    assert geo.haversine_m(34.0, -119.0, 34.0, -119.0) == 0.0


def test_dms_to_decimal_and_hemisphere_sign():
    assert abs(geo._coord_from_dms((34, 25, 12), "N") - (34 + 25 / 60 + 12 / 3600)) < 1e-9
    assert geo._coord_from_dms((34, 25, 12), "S") < 0
    assert geo._coord_from_dms((119, 42, 0), "W") < 0
    assert geo._coord_from_dms(None, "N") is None


def test_distance_metrics():
    points = [{"lat": 34.0, "lon": -119.0}, {"lat": 34.001, "lon": -119.0}, {"lat": 34.0, "lon": -119.002}]
    metrics = geo._distance_metrics(points)
    assert metrics["point_count"] == 3
    assert metrics["survey_extent_m"] > 0
    assert metrics["nearest_neighbor_m"]["min"] > 0


def test_build_geospatial_summary(monkeypatch):
    coords = {"a.jpg": {"lat": 34.0, "lon": -119.0}, "b.jpg": {"lat": 34.001, "lon": -119.0}}
    monkeypatch.setattr(geo, "read_exif_gps", lambda path: coords.get(Path(path).name))
    predictions = [
        {"input_path": "a.jpg", "class_counts": {"prairie_dog": 3, "burrow": 5},
         "boxes": [{"class_name": "prairie_dog", "stability_label": "trusted"}] * 3
                  + [{"class_name": "burrow", "stability_label": "trusted"}] * 5},
        {"input_path": "b.jpg", "class_counts": {"prairie_dog": 2, "burrow": 1},
         "boxes": [{"class_name": "prairie_dog", "stability_label": "unstable"}] * 2
                  + [{"class_name": "burrow", "stability_label": "borderline"}]},
        {"input_path": "nogps.jpg", "class_counts": {"prairie_dog": 1},
         "boxes": [{"class_name": "prairie_dog", "stability_label": "trusted"}]},
    ]
    summary = geo.build_geospatial_summary(predictions=predictions, output_dir=Path(tempfile.mkdtemp()))
    assert summary is not None
    assert summary["georeferenced_image_count"] == 2  # the GPS-less image is excluded
    assert summary["metrics"]["totals_by_class"] == {"prairie_dog": 5, "burrow": 6}
    assert summary["metrics"]["dog_per_burrow"] == round(5 / 6, 3)


def test_build_geospatial_summary_none_without_gps(monkeypatch):
    monkeypatch.setattr(geo, "read_exif_gps", lambda path: None)
    assert geo.build_geospatial_summary(
        predictions=[{"input_path": "x.jpg", "boxes": []}], output_dir=Path(tempfile.mkdtemp())
    ) is None
