import pytest

torch = pytest.importorskip("torch")

from ultra_deepagents.rarespot.spectral import SpectralInstabilityConfig, SpectralInstabilityScorer


def test_spectral_filter_preserves_shape_and_reports_retained_energy():
    scorer = SpectralInstabilityScorer(
        SpectralInstabilityConfig(imgsz=32, preservation_ratio=0.5)
    )
    feature_map = torch.rand((2, 4, 16, 16), dtype=torch.float32)

    filtered, diagnostics = scorer.apply_adaptive_spectral_filter(feature_map)

    assert filtered.shape == feature_map.shape
    assert len(diagnostics) == 2
    assert all(0.0 <= item["retained_energy_fraction"] <= 1.0 for item in diagnostics)
    assert all(item["dominant_channel_index"] >= 0 for item in diagnostics)


def test_discrepancy_breakdown_scores_lost_class_spatial_and_confidence_jitter():
    scorer = SpectralInstabilityScorer(SpectralInstabilityConfig())
    original = torch.tensor(
        [
            [0.0, 0.0, 20.0, 20.0, 0.90, 0.0, 5.0, -5.0],
            [40.0, 40.0, 60.0, 60.0, 0.80, 1.0, -5.0, 5.0],
            [80.0, 80.0, 100.0, 100.0, 0.70, 0.0, 5.0, -5.0],
        ]
    )
    filtered = torch.tensor(
        [
            [1.0, 1.0, 21.0, 21.0, 0.50, 0.0, 5.0, -5.0],
            [40.0, 40.0, 60.0, 60.0, 0.80, 0.0, 5.0, -5.0],
        ]
    )

    breakdown = scorer.discrepancy_breakdown(original, filtered)

    assert breakdown["lost"] == 1.0
    assert breakdown["class_jitter"] == 1.0
    assert breakdown["confidence_jitter"] > 0.0
    assert breakdown["matched"] == 2.0
    assert breakdown["score"] > 0.0
