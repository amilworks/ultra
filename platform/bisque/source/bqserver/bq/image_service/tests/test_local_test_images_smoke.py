import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def _resolve_images_dir() -> Path:
    from_env = os.environ.get("BISQUE_TEST_IMAGES_DIR")
    if from_env:
        return Path(from_env).expanduser().resolve()
    return Path("source/tests/test_images").resolve()


def test_local_test_images_dir_exists():
    images_dir = _resolve_images_dir()
    assert images_dir.exists(), (
        f"Missing test image directory: {images_dir}. "
        "Set BISQUE_TEST_IMAGES_DIR or run scripts/dev/init_test_config.sh."
    )
    assert images_dir.is_dir(), f"Expected directory, got: {images_dir}"


def test_local_test_images_has_supported_files():
    images_dir = _resolve_images_dir()
    if not images_dir.exists():
        pytest.skip("test image directory is not available")

    files = [p for p in images_dir.iterdir() if p.is_file()]
    assert files, f"No files found in test image directory: {images_dir}"

    supported = {".tif", ".tiff", ".ome.tif", ".ome.tiff", ".nii", ".gz"}
    matched = []
    for p in files:
        name = p.name.lower()
        if any(name.endswith(ext) for ext in supported):
            matched.append(p)

    assert matched, (
        f"No supported image files found in {images_dir}. "
        f"Found: {[p.name for p in files]}"
    )
