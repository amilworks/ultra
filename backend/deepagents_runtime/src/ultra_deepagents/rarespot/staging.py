from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from xml.etree import ElementTree as ET

from ultra_deepagents.rarespot.artifacts import slug_token
from ultra_deepagents.rarespot.config import RareSpotConfig

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def collect_local_images(paths: list[str], *, config: RareSpotConfig) -> list[Path]:
    images: list[Path] = []
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        ensure_allowed_path(path, config.allowed_input_roots)
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES:
                    ensure_allowed_path(child.resolve(), config.allowed_input_roots)
                    images.append(child.resolve())
        elif path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            images.append(path)
    return dedupe_paths(images)


def ensure_allowed_path(path: Path, allowed_roots: tuple[Path, ...]) -> None:
    if not allowed_roots:
        return
    resolved = path.resolve()
    for root in allowed_roots:
        try:
            resolved.relative_to(root.resolve())
            return
        except ValueError:
            continue
    raise ValueError(f"Input path is outside configured allowed roots: {resolved}")


def dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    deduped: list[Path] = []
    for path in paths:
        token = str(path)
        if token in seen:
            continue
        seen.add(token)
        deduped.append(path)
    return deduped


async def stage_inputs(
    *,
    local_paths: list[str],
    resource_uris: list[str],
    dataset_uris: list[str],
    stage_dir: Path,
    config: RareSpotConfig,
) -> dict[str, Any]:
    stage_dir.mkdir(parents=True, exist_ok=True)
    images = collect_local_images(local_paths, config=config)
    downloaded = await stage_bisque_inputs(
        resource_uris=resource_uris,
        dataset_uris=dataset_uris,
        stage_dir=stage_dir,
        config=config,
    )
    images.extend(downloaded)
    images = dedupe_paths(images)
    return {
        "image_paths": images,
        "local_image_count": len(images) - len(downloaded),
        "downloaded_image_count": len(downloaded),
    }


async def stage_bisque_inputs(
    *,
    resource_uris: list[str],
    dataset_uris: list[str],
    stage_dir: Path,
    config: RareSpotConfig,
) -> list[Path]:
    if not resource_uris and not dataset_uris:
        return []
    import httpx

    if not config.bisque_base_url and any(uri.startswith("bisque://") for uri in resource_uris + dataset_uris):
        raise ValueError("BisQue staging requires BISQUE_BASE_URL for bisque:// URIs.")
    headers = {"Authorization": f"Bearer {config.bisque_token}"} if config.bisque_token else {}
    auth = None
    if config.bisque_username and config.bisque_password:
        auth = (config.bisque_username, config.bisque_password)
    staged: list[Path] = []
    async with httpx.AsyncClient(headers=headers, auth=auth, timeout=120.0) as client:
        expanded_resource_uris = list(resource_uris)
        for dataset_uri in dataset_uris:
            expanded_resource_uris.extend(await list_dataset_resources(client, dataset_uri, config=config))
        for resource_uri in expanded_resource_uris:
            staged.append(await download_resource(client, resource_uri, stage_dir=stage_dir, config=config))
    return staged


async def list_dataset_resources(client: Any, dataset_uri: str, *, config: RareSpotConfig) -> list[str]:
    url = normalize_bisque_url(dataset_uri, config=config)
    response = await client.get(url)
    response.raise_for_status()
    root = ET.fromstring(response.text)
    uris: list[str] = []
    for node in root.iter():
        value = node.attrib.get("uri") or node.attrib.get("value")
        if value and looks_like_resource_uri(value):
            uris.append(value)
    return uris


async def download_resource(client: Any, resource_uri: str, *, stage_dir: Path, config: RareSpotConfig) -> Path:
    url = normalize_bisque_url(resource_uri, config=config)
    response = await client.get(url)
    response.raise_for_status()
    suffix = suffix_from_response(resource_uri, response.headers.get("content-type", ""))
    output_path = stage_dir / f"{slug_token(resource_uri)}{suffix}"
    output_path.write_bytes(response.content)
    if suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"Downloaded BisQue resource is not an image: {resource_uri}")
    return output_path


def normalize_bisque_url(uri: str, *, config: RareSpotConfig) -> str:
    raw = str(uri or "").strip()
    if raw.startswith("http://") or raw.startswith("https://"):
        return raw
    if raw.startswith("bisque://"):
        path = raw.removeprefix("bisque://")
        base = config.bisque_base_url.rstrip("/")
        return f"{base}/{path.lstrip('/')}"
    return raw


def looks_like_resource_uri(value: str) -> bool:
    raw = str(value or "").strip()
    return bool(raw.startswith("http") or raw.startswith("bisque://") or "/data_service/" in raw)


def suffix_from_response(resource_uri: str, content_type: str) -> str:
    parsed = urlparse(resource_uri)
    suffix = Path(parsed.path).suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return suffix
    content_type = content_type.lower()
    if "png" in content_type:
        return ".png"
    if "tiff" in content_type or "tif" in content_type:
        return ".tif"
    return ".jpg"


def copy_or_link_images(image_paths: list[Path], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for index, image_path in enumerate(image_paths):
        target = output_dir / f"{index:04d}-{slug_token(image_path.stem)}{image_path.suffix.lower()}"
        if image_path.resolve() == target.resolve():
            copied.append(target)
            continue
        shutil.copy2(image_path, target)
        copied.append(target)
    return copied
