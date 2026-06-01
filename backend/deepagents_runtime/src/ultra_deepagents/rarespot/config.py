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
    bisque_base_url: str = ""
    bisque_token: str = ""
    bisque_username: str = ""
    bisque_password: str = ""

    @property
    def stride(self) -> int:
        return max(1, round(int(self.tile_size) * (1.0 - float(self.tile_overlap))))

    @classmethod
    def from_settings(cls, settings: RuntimeSettings | None = None) -> "RareSpotConfig":
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
            bisque_base_url=os.getenv("BISQUE_BASE_URL", ""),
            bisque_token=os.getenv("BISQUE_SERVICE_TOKEN", ""),
            bisque_username=os.getenv("BISQUE_SERVICE_USERNAME", ""),
            bisque_password=os.getenv("BISQUE_SERVICE_PASSWORD", ""),
        )

    @classmethod
    def from_env(cls) -> "RareSpotConfig":
        return cls.from_settings(RuntimeSettings.from_env())


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
