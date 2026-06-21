from __future__ import annotations

import asyncio

from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.title_generation import (
    _build_title_model,
    fallback_conversation_title,
    generate_conversation_title,
    is_initial_conversation_turn,
    resolve_conversation_title_task,
    start_conversation_title_task,
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


class SlowTitleModel:
    model_name = "deepseek_v4"

    async def ainvoke(self, _messages):
        await asyncio.sleep(30)
        return FakeTitleResponse()


class BindRecordingModel:
    model_name = "deepseek_v4"

    def __init__(self):
        self.bind_kwargs = None

    def bind(self, **kwargs):
        self.bind_kwargs = kwargs
        return self

    async def ainvoke(self, _messages):
        return FakeTitleResponse()


def test_is_initial_conversation_turn_detects_prior_assistant_messages():
    assert is_initial_conversation_turn([{"role": "user", "content": "hi"}])
    assert is_initial_conversation_turn([])
    assert is_initial_conversation_turn(None)
    assert not is_initial_conversation_turn(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "more"},
        ]
    )


def test_start_conversation_title_task_skips_followup_turns(tmp_path):
    async def scenario():
        task = start_conversation_title_task(
            settings=_settings(tmp_path),
            goal="Follow-up question",
            messages=[
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "Follow-up question"},
            ],
            model_factory=lambda _settings: FakeTitleModel(),
        )
        assert task is None
        result = await resolve_conversation_title_task(
            task,
            settings=_settings(tmp_path),
            goal="Compare these two uploaded arXiv papers on attention mechanisms and create a limitations table.",
            messages=[{"role": "user", "content": "Compare the papers."}],
            response_text="",
            artifact_events=[],
        )
        return result

    result = asyncio.run(scenario())

    assert result.strategy == "fallback"
    assert result.reason == "thread_already_titled"
    assert result.title == "Attention Paper Comparison"


def test_resolve_conversation_title_task_uses_early_llm_result(tmp_path):
    async def scenario():
        task = start_conversation_title_task(
            settings=_settings(tmp_path),
            goal="Run RareSpot on this prairie dog image and discuss burrow detections.",
            messages=[
                {
                    "role": "user",
                    "content": "Run RareSpot on this prairie dog image and discuss burrow detections.",
                }
            ],
            model_factory=lambda _settings: EarlyFakeTitleModel(),
        )
        assert task is not None
        return await resolve_conversation_title_task(
            task,
            settings=_settings(tmp_path),
            goal="Run RareSpot on this prairie dog image and discuss burrow detections.",
            messages=[{"role": "user", "content": "Run RareSpot."}],
            response_text="Detected prairie dogs and burrows.",
            artifact_events=[],
        )

    result = asyncio.run(scenario())

    assert result.strategy == "llm"
    assert result.title == "RareSpot Prairie Dog Analysis"


class EarlyFakeTitleResponse:
    content = '{"title": "RareSpot Prairie Dog Analysis"}'


class EarlyFakeTitleModel:
    model_name = "deepseek_v4"

    async def ainvoke(self, messages):
        joined = "\n".join(str(message.get("content", "")) for message in messages)
        # The early call runs before the run finishes: no response context yet.
        assert "Detected prairie dogs" not in joined
        return EarlyFakeTitleResponse()


class _Resp:
    def __init__(self, content):
        self.content = content


class SmartFakeModel:
    """Generic title when called request-only; specific title once the answer is
    in the prompt (the response-aware upgrade includes an 'Assistant result')."""

    model_name = "deepseek_v4"

    async def ainvoke(self, messages):
        joined = "\n".join(str(message.get("content", "")) for message in messages)
        if "Assistant result" in joined:
            return _Resp('{"title": "CT Head Scan Analysis"}')
        return _Resp('{"title": "Image Content Analysis"}')


def test_resolve_upgrades_generic_request_only_title_with_response(tmp_path):
    async def scenario():
        factory = lambda _settings: SmartFakeModel()  # noqa: E731
        task = start_conversation_title_task(
            settings=_settings(tmp_path),
            goal="Describe this image.",
            messages=[{"role": "user", "content": "Describe this image."}],
            model_factory=factory,
        )
        assert task is not None
        return await resolve_conversation_title_task(
            task,
            settings=_settings(tmp_path),
            goal="Describe this image.",
            messages=[{"role": "user", "content": "Describe this image."}],
            response_text="This is a head CT showing enlarged ventricles consistent with NPH.",
            artifact_events=[],
            model_factory=factory,
        )

    result = asyncio.run(scenario())

    assert result.title == "CT Head Scan Analysis"
    assert result.strategy == "llm"
    assert result.reason == "response_aware_upgrade"


def test_resolve_keeps_specific_request_only_title_without_extra_call(tmp_path):
    calls = {"n": 0}

    class CountingModel:
        model_name = "deepseek_v4"

        async def ainvoke(self, _messages):
            calls["n"] += 1
            return _Resp('{"title": "RareSpot Prairie Dog Analysis"}')

    async def scenario():
        factory = lambda _settings: CountingModel()  # noqa: E731
        task = start_conversation_title_task(
            settings=_settings(tmp_path),
            goal="Run RareSpot on this prairie dog image.",
            messages=[{"role": "user", "content": "Run RareSpot on this prairie dog image."}],
            model_factory=factory,
        )
        return await resolve_conversation_title_task(
            task,
            settings=_settings(tmp_path),
            goal="Run RareSpot on this prairie dog image.",
            messages=[{"role": "user", "content": "Run RareSpot on this prairie dog image."}],
            response_text="Detected prairie dogs and burrows.",
            artifact_events=[],
            model_factory=factory,
        )

    result = asyncio.run(scenario())

    assert result.title == "RareSpot Prairie Dog Analysis"
    assert result.strategy == "llm"
    # Specific title -> no response-aware upgrade -> only the early call ran.
    assert calls["n"] == 1


def test_resolve_conversation_title_task_falls_back_when_unresolved(tmp_path):
    async def scenario():
        task = start_conversation_title_task(
            settings=_settings(tmp_path),
            goal="Please train a compact UNet segmentation model and plot training curves.",
            messages=[{"role": "user", "content": "Train the UNet."}],
            model_factory=lambda _settings: SlowTitleModel(),
        )
        assert task is not None
        result = await resolve_conversation_title_task(
            task,
            settings=_settings(tmp_path),
            goal="Please train a compact UNet segmentation model and plot training curves.",
            messages=[{"role": "user", "content": "Train the UNet."}],
            response_text="Trained a UNet, saved weights, plotted curves.",
            artifact_events=[],
            grace_seconds=0.05,
        )
        assert task.cancelled() or task.done()
        return result

    result = asyncio.run(scenario())

    assert result.strategy == "fallback"
    assert result.reason == "early_title_unresolved"
    assert result.title == "UNet Segmentation Training"


def test_resolve_conversation_title_task_enriches_fallback_with_run_outcome(tmp_path):
    async def scenario():
        task = start_conversation_title_task(
            settings=_settings(tmp_path),
            goal="Process the uploaded microscopy data as discussed.",
            messages=[{"role": "user", "content": "Process the uploaded microscopy data."}],
            model_factory=lambda _settings: FailingTitleModel(),
        )
        assert task is not None
        return await resolve_conversation_title_task(
            task,
            settings=_settings(tmp_path),
            goal="Process the uploaded microscopy data as discussed.",
            messages=[{"role": "user", "content": "Process the uploaded microscopy data."}],
            response_text=(
                "Trained a compact UNet segmentation model on the masks and saved weights."
            ),
            artifact_events=[],
        )

    result = asyncio.run(scenario())

    assert result.strategy == "fallback"
    assert "title model unavailable" in result.reason
    # The deterministic title is rebuilt from the full run outcome, matching
    # what the old inline call would have produced.
    assert result.title == "UNet Segmentation Training"


def test_build_title_model_disables_thinking_and_caps_tokens(tmp_path):
    settings = _settings(tmp_path)
    recording = BindRecordingModel()

    model = _build_title_model(settings, lambda _settings: recording)

    assert model is recording
    assert recording.bind_kwargs == {
        "max_tokens": 96,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }


def test_build_title_model_keeps_model_unbound_when_thinking_allowed(tmp_path):
    settings = RuntimeSettings(
        openai_base_url="http://example.test/v1",
        openai_model="deepseek_v4",
        workspace_root=str(tmp_path / "workspaces"),
        artifact_root=str(tmp_path / "artifacts"),
        title_generation_enabled=True,
        title_thinking_disabled=False,
    )
    recording = BindRecordingModel()

    model = _build_title_model(settings, lambda _settings: recording)

    assert model is recording
    assert recording.bind_kwargs is None


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
