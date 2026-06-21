import asyncio
from pathlib import Path

from ultra_deepagents.rarespot.uploads import resolve_uploaded_file_ids


def test_resolve_uploaded_file_ids_from_upload_root(tmp_path: Path):
    upload_root = tmp_path / "uploads"
    upload_root.mkdir()
    uploaded_image = upload_root / "file-1__abc123__EnrNE_Day2_Run1__0242.JPG"
    uploaded_image.write_bytes(b"jpeg bytes")

    resolution = asyncio.run(
        resolve_uploaded_file_ids(
            ["file-1"],
            upload_roots=(upload_root,),
            allowed_roots=(tmp_path,),
        )
    )

    assert resolution.image_paths == [uploaded_image.resolve()]
    assert resolution.missing_file_ids == []
