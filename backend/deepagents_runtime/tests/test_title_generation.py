from __future__ import annotations

import asyncio

from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.title_generation import (
    fallback_conversation_title,
    generate_conversation_title,
)


class FakeTitleResponse:
    content = '{"title": "RareSpot Prairie Dog Analysis"}'


class FakeTitleModel:
    model_name = "deepseek_v4"

    async def ainvoke(self, messages):
        joined = "\n".join(str(message.get("content", "")) for message in messages)
        assert "Run RareSpot on this prairie dog image" in joined
        assert "Detected prairie dogs and burrows" in joined
        return FakeTitleResponse()


class FailingTitleModel:
    model_name = "deepseek_v4"

    async def ainvoke(self, _messages):
        raise RuntimeError("title model unavailable")


def _settings(tmp_path):
    return RuntimeSettings(
        openai_base_url="http://example.test/v1",
        openai_model="deepseek_v4",
        workspace_root=str(tmp_path / "workspaces"),
        artifact_root=str(tmp_path / "artifacts"),
        title_generation_enabled=True,
    )


def test_generate_conversation_title_uses_title_model_context(tmp_path):
    async def scenario():
        return await generate_conversation_title(
            settings=_settings(tmp_path),
            goal="Run RareSpot on this prairie dog image and summarize the ecology signals.",
            messages=[
                {
                    "role": "user",
                    "content": "Run RareSpot on this prairie dog image and discuss burrow detections.",
                }
            ],
            response_text="Detected prairie dogs and burrows, then saved overlays and a CSV.",
            artifact_events=[
                {
                    "payload": {
                        "kind": "image",
                        "title": "RareSpot detection overlay",
                        "path": "outputs/overlay.png",
                    }
                }
            ],
            model_factory=lambda _settings: FakeTitleModel(),
        )

    result = asyncio.run(scenario())

    assert result.title == "RareSpot Prairie Dog Analysis"
    assert result.strategy == "llm"
    assert result.model == "deepseek_v4"


def test_generate_conversation_title_falls_back_without_blocking_run_completion(tmp_path):
    async def scenario():
        return await generate_conversation_title(
            settings=_settings(tmp_path),
            goal="Please train a compact UNet segmentation model on the uploaded microscopy masks, save the weights, and plot training curves.",
            messages=[
                {
                    "role": "user",
                    "content": "Please train a compact UNet segmentation model on the uploaded microscopy masks, save the weights, and plot training curves.",
                }
            ],
            response_text="Trained a UNet, saved model weights, and generated training curves.",
            artifact_events=[],
            model_factory=lambda _settings: FailingTitleModel(),
        )

    result = asyncio.run(scenario())

    assert result.title == "UNet Segmentation Training"
    assert result.strategy == "fallback"
    assert "title model unavailable" in result.reason


def test_fallback_conversation_title_varies_across_scientific_prompt_shapes():
    cases = [
        (
            "Run RareSpot on the prairie dog block, quantify burrows, and upload outputs back to BisQue.",
            "RareSpot Burrow Quantification",
        ),
        (
            "Compare these two uploaded arXiv papers on attention mechanisms and create a limitations table.",
            "Attention Paper Comparison",
        ),
        (
            "Create a matplotlib y = x^2 and y = x^3 plot with the source code attached.",
            "Matplotlib Function Plot",
        ),
        (
            "Inspect this OME-TIFF time series for channel drift and summarize the image metadata.",
            "OME-TIFF Channel Drift",
        ),
        (
            "Write code and visualize how bubble sort works with a small animated trace.",
            "Bubble Sort Visualization",
        ),
    ]

    titles = [fallback_conversation_title(prompt) for prompt, _expected in cases]

    assert titles == [expected for _prompt, expected in cases]
    assert len(set(titles)) == len(cases)
    assert all(len(title) <= 52 for title in titles)
