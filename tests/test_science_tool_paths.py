from pathlib import Path
from types import SimpleNamespace

from src import tools


def test_models_root_prefers_explicit_yolo_model_root(monkeypatch, tmp_path):
    root = tmp_path / "models" / "yolo"
    monkeypatch.setenv("YOLO_MODEL_ROOT", str(root))
    monkeypatch.delenv("YOLO_DEFAULT_MODEL", raising=False)
    monkeypatch.setattr(
        tools,
        "get_settings",
        lambda: SimpleNamespace(prairie_rarespot_weights_path=""),
    )

    resolved = Path(tools._models_root())

    assert resolved == root
    assert resolved.is_dir()


def test_models_root_uses_default_model_parent(monkeypatch, tmp_path):
    model_path = tmp_path / "models" / "yolo" / "yolo26x.pt"
    monkeypatch.delenv("YOLO_MODEL_ROOT", raising=False)
    monkeypatch.setenv("YOLO_DEFAULT_MODEL", str(model_path))
    monkeypatch.setattr(
        tools,
        "get_settings",
        lambda: SimpleNamespace(prairie_rarespot_weights_path=""),
    )

    resolved = Path(tools._models_root())

    assert resolved == model_path.parent
    assert resolved.is_dir()


def test_models_root_falls_back_to_rarespot_parent(monkeypatch, tmp_path):
    weights_path = tmp_path / "models" / "yolo" / "RareSpotWeights.pt"
    monkeypatch.delenv("YOLO_MODEL_ROOT", raising=False)
    monkeypatch.delenv("YOLO_DEFAULT_MODEL", raising=False)
    monkeypatch.setattr(
        tools,
        "get_settings",
        lambda: SimpleNamespace(prairie_rarespot_weights_path=str(weights_path)),
    )

    resolved = Path(tools._models_root())

    assert resolved == weights_path.parent
    assert resolved.is_dir()
