"""ocr-reader subagent: registration, routing tokens, and the transcription
contract. The subagent exists because OCR and visual reasoning are
contradictory prompts — these tests pin the separation."""

from __future__ import annotations

from dataclasses import replace

from ultra_deepagents.agent import (
    OCR_DELEGATION_GUIDANCE,
    OCR_SUBAGENT,
    VISION_SUBAGENT,
    _should_register_vision_subagent,
    build_subagents,
    build_system_prompt,
)
from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.schemas import RunJobEnvelope


class _NamedTool:
    def __init__(self, name: str) -> None:
        self.name = name


def _vision_settings(**overrides) -> RuntimeSettings:
    return RuntimeSettings(
        openai_base_url="http://example.test/v1",
        openai_model="test-model",
        qwen_vlm_enabled=True,
        qwen_vlm_base_url="http://vlm.example.test/v1",
        qwen_vlm_api_key="test-key",
        **overrides,
    )


def _context(goal: str):
    return RunJobEnvelope(
        run_id="run-ocr-test",
        thread_id="thread-1",
        user_id="user-1",
        goal=goal,
    ).to_context(artifact_root="/tmp/art", workspace_root="/tmp/ws")


def test_ocr_reader_registers_alongside_vision_reasoner():
    tools = [_NamedTool("inspect_images"), _NamedTool("screen_images")]
    subagents = build_subagents(vision_tools=tools)
    names = [subagent["name"] for subagent in subagents]
    assert "vision-reasoner" in names
    assert "ocr-reader" in names
    ocr = next(s for s in subagents if s["name"] == "ocr-reader")
    assert [tool.name for tool in ocr["tools"]] == ["inspect_images", "screen_images"]
    # Registration must deep-copy the response format (the templates are shared
    # module constants; one run's mutation must never leak into the next).
    assert ocr["response_format"] is not OCR_SUBAGENT["response_format"]


def test_ocr_reader_absent_without_vision_tools():
    subagents = build_subagents(vision_tools=None)
    assert "ocr-reader" not in [subagent["name"] for subagent in subagents]


def test_ocr_goal_tokens_register_the_vision_stack():
    settings = _vision_settings()
    for goal in (
        "Transcribe the text in this screenshot for me",
        "Run OCR on the scanned lab notebook pages",
        "Extract text from the whiteboard photo",
        "Get the subtitles text out of the recorded talk",
    ):
        assert _should_register_vision_subagent(_context(goal), settings), goal


def test_transcription_contract_pins_the_key_clauses():
    prompt = OCR_SUBAGENT["system_prompt"]
    for marker in (
        "[illegible]",
        "never complete truncated words",
        "tesseract",
        "Agreement between engine and VLM",
        "never silently pick one",
        "/outputs/ocr/",
        "ffmpeg",
        "timeout",
        "paper tools",
    ):
        assert marker in prompt, f"transcription contract lost clause: {marker}"
    # Routing separation stays explicit in both descriptions.
    assert "vision-reasoner" in OCR_SUBAGENT["description"]
    assert "ocr" not in VISION_SUBAGENT["description"].lower().replace("ocr-reader", "")


def test_system_prompt_advertises_ocr_reader_for_ocr_goals():
    settings = _vision_settings()
    prompt = build_system_prompt(settings, _context("OCR the attached scanned invoice"))
    assert "ocr-reader" in prompt
    assert OCR_DELEGATION_GUIDANCE.strip().splitlines()[0] in prompt


def test_generic_text_goals_do_not_pay_for_ocr_guidance():
    settings = replace(_vision_settings(), qwen_vlm_enabled=False)
    prompt = build_system_prompt(settings, _context("Summarize the CSV results"))
    assert "ocr-reader" not in prompt


def test_bare_run_manifest_never_contradicts_map_task():
    """Regression for run_69c0e07d (2026-08-04): a run with zero registered
    specialist subagents shipped a manifest saying available_subagents: []
    while map_task sat in the tool list — the model wasted reasoning chasing
    the contradiction. The manifest must always list the general-purpose
    target that task and map_task can really reach."""
    from ultra_deepagents.context_tools import build_tool_capability_manifest
    from ultra_deepagents.map_task import (
        GENERAL_PURPOSE_SUBAGENT_SPEC,
        build_map_task_tool,
    )

    map_tool = build_map_task_tool([dict(GENERAL_PURPOSE_SUBAGENT_SPEC)], workspace_dir=None)
    manifest = build_tool_capability_manifest(
        [map_tool],
        available_subagents=[dict(GENERAL_PURPOSE_SUBAGENT_SPEC)],
    )
    assert "map_task" in manifest["registered_tools"]
    names = [entry["name"] for entry in manifest["available_subagents"]]
    assert names == ["general-purpose"]
    assert "task" in {tool["name"] for tool in manifest["deepagents_builtin_tools"]}
    # And the tool's own description carries the always-available guarantee.
    assert "ALWAYS available" in map_tool.description
