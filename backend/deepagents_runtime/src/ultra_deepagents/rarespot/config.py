from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ultra_deepagents.config import RuntimeSettings


@dataclass(frozen=True)
class RareSpotConfig:
    weights_path: Path
    yolov5_path: Path
    artifact_root: Path
    allowed_input_roots: tuple[Path, ...]
    upload_roots: tuple[Path, ...] = ()
    upload_database_url: str = ""
    tile_size: int = 512
    tile_overlap: float = 0.25
    conf: float = 0.25
    iou: float = 0.45
    spectral: bool = True
    # Per-detection stability (perturbation consensus): re-detect under blur/brightness/JPEG
    # perturbations and score each box by how often it survives. The ecologist-facing trust
    # signal; costs one extra detect pass over the tiles.
    stability: bool = True
    stability_match_iou: float = 0.5

    @property
    def stride(self) -> int:
        return max(1, round(int(self.tile_size) * (1.0 - float(self.tile_overlap))))

    @classmethod
    def from_settings(cls, settings: RuntimeSettings | None = None) -> RareSpotConfig:
        settings = settings or RuntimeSettings.from_env()
        allowed_roots = tuple(
            Path(value).expanduser().resolve()
            for value in settings.rarespot_allowed_input_roots
        )
        upload_roots = tuple(
            _resolve_repo_relative(value)
            for value in settings.rarespot_upload_roots
        )
        return cls(
            weights_path=_resolve_repo_relative(settings.rarespot_weights_path),
            yolov5_path=_resolve_repo_relative(settings.rarespot_yolov5_path),
            artifact_root=_resolve_repo_relative(settings.rarespot_artifact_root),
            allowed_input_roots=allowed_roots,
            upload_roots=upload_roots,
            upload_database_url=settings.rarespot_upload_database_url,
            tile_size=int(settings.rarespot_imgsz),
            tile_overlap=float(settings.rarespot_tile_overlap),
            conf=float(settings.rarespot_conf_threshold),
            iou=float(settings.rarespot_iou_threshold),
            spectral=_env_bool("ULTRA_RARESPOT_SPECTRAL", True),
            stability=_env_bool("ULTRA_RARESPOT_STABILITY", True),
            stability_match_iou=float(os.getenv("ULTRA_RARESPOT_STABILITY_MATCH_IOU", "0.5") or "0.5"),
        )

    @classmethod
    def from_env(cls) -> RareSpotConfig:
        return cls.from_settings(RuntimeSettings.from_env())

    @classmethod
    def from_paths(
        cls,
        *,
        weights_path: str | Path,
        yolov5_path: str | Path,
        artifact_root: str | Path,
        allowed_input_roots: tuple[str | Path, ...] = (),
        tile_size: int = 512,
        tile_overlap: float = 0.25,
        conf: float = 0.25,
        iou: float = 0.45,
        spectral: bool = True,
        stability: bool = True,
        stability_match_iou: float = 0.5,
    ) -> RareSpotConfig:
        """Construct a config from explicit absolute paths, for environments where
        the repo-relative resolution (``_repo_root`` -> ``parents[5]``) does not
        hold — e.g. the prairie-dog-detection Skill running inside the code sandbox
        with the model/runtime baked at ``/opt/rarespot``."""
        return cls(
            weights_path=Path(weights_path).expanduser().resolve(),
            yolov5_path=Path(yolov5_path).expanduser().resolve(),
            artifact_root=Path(artifact_root).expanduser().resolve(),
            allowed_input_roots=tuple(
                Path(value).expanduser().resolve() for value in allowed_input_roots
            ),
            tile_size=int(tile_size),
            tile_overlap=float(tile_overlap),
            conf=float(conf),
            iou=float(iou),
            spectral=bool(spectral),
            stability=bool(stability),
            stability_match_iou=float(stability_match_iou),
        )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _resolve_repo_relative(value: str) -> Path:
    path = Path(str(value or "").strip()).expanduser()
    if not path.is_absolute():
        path = _repo_root() / path
    return path.resolve()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    token = value.strip().lower()
    if token in {"1", "true", "yes", "y", "on"}:
        return True
    if token in {"0", "false", "no", "n", "off"}:
        return False
    return default
