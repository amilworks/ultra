from __future__ import annotations

from types import SimpleNamespace

import src.llm as llm_module
from src.chat_titles import generate_chat_title
from src.tooling.tool_selection import _select_tool_subset


def test_generate_chat_title_falls_back_in_mock_mode(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.chat_titles.get_settings",
        lambda: SimpleNamespace(llm_mock_mode=True),
    )

    title, strategy = generate_chat_title(
        [{"role": "user", "content": "Quantify nuclei in this image and summarize drift"}],
        max_words=4,
    )

    assert title == "Quantify nuclei in this"
    assert strategy == "fallback"


def test_select_tool_subset_preserves_current_megaseg_upload_behavior() -> None:
    tools = [
        {"function": {"name": "repro_report"}},
        {"function": {"name": "bioio_load_image"}},
        {"function": {"name": "segment_image_megaseg"}},
        {"function": {"name": "quantify_segmentation_masks"}},
        {"function": {"name": "search_bisque_resources"}},
    ]

    subset = _select_tool_subset(
        messages=[{"role": "user", "content": "Run MegaSeg on this uploaded microscopy image"}],
        uploaded_files=["/tmp/cells.ome.tiff"],
        all_tools=tools,
    )

    assert [item["function"]["name"] for item in subset] == [
        "repro_report",
        "segment_image_megaseg",
        "search_bisque_resources",
    ]


def test_llm_module_remains_compatibility_shim() -> None:
    assert llm_module.generate_chat_title is generate_chat_title
    assert llm_module._select_tool_subset is _select_tool_subset
