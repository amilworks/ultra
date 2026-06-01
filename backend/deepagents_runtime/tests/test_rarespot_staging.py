import asyncio
from pathlib import Path

from ultra_deepagents.rarespot.config import RareSpotConfig
from ultra_deepagents.rarespot.staging import download_resource, list_dataset_resources


class FakeResponse:
    def __init__(self, *, text: str = "", content: bytes = b"", content_type: str = "image/png") -> None:
        self.text = text
        self.content = content
        self.headers = {"content-type": content_type}

    def raise_for_status(self) -> None:
        return None


class FakeClient:
    async def get(self, url: str) -> FakeResponse:
        if url.endswith("/dataset/1"):
            return FakeResponse(
                text='<dataset><value uri="bisque://resource/1" /><value uri="bisque://resource/2" /></dataset>'
            )
        return FakeResponse(content=b"png-bytes", content_type="image/png")


def test_mock_bisque_dataset_and_resource_staging(tmp_path: Path):
    config = RareSpotConfig(
        weights_path=tmp_path / "weights.pt",
        yolov5_path=tmp_path / "yolov5",
        artifact_root=tmp_path / "artifacts",
        allowed_input_roots=(),
        bisque_base_url="https://bisque.example",
    )

    async def run():
        client = FakeClient()
        resources = await list_dataset_resources(client, "bisque://dataset/1", config=config)
        path = await download_resource(client, resources[0], stage_dir=tmp_path, config=config)
        return resources, path

    resources, path = asyncio.run(run())

    assert resources == ["bisque://resource/1", "bisque://resource/2"]
    assert path.suffix == ".png"
    assert path.read_bytes() == b"png-bytes"
