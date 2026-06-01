from __future__ import annotations

import json
from pathlib import Path

import pytest

from ultra_deepagents.agent import build_research_agent
from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.context import AgentRunContext
from ultra_deepagents.papers.tools import (
    _ingest_pdf_resource,
    build_paper_tools,
    ingest_pdf_file,
    normalize_arxiv_id,
    read_paper_pages_from_cache,
    render_paper_page_from_cache,
    search_paper_cache,
)


def _context(tmp_path: Path) -> AgentRunContext:
    return AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="local-org",
        user_id="user-1",
        project_id="local-project",
        thread_id="thread-1",
        run_id="run-1",
        workspace_root=str(tmp_path / "workspace"),
        artifact_root=str(tmp_path / "artifacts" / "run-1"),
    )


def _write_fixture_pdf(path: Path) -> None:
    import fitz

    doc = fitz.open()
    page1 = doc.new_page(width=400, height=400)
    page1.insert_text(
        (48, 72),
        "Abstract\n"
        "This paper introduces scaled dot-product attention for sequence modeling.\n"
        "The attention scaling equation is softmax(Q K^T / sqrt(d_k)) V.\n",
        fontsize=11,
    )
    page2 = doc.new_page(width=400, height=400)
    page2.insert_text(
        (48, 72),
        "Methods\n"
        "Figure 1 shows the encoder-decoder architecture.\n"
        "The scaling term prevents large dot products from saturating softmax.\n",
        fontsize=11,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)
    doc.close()


def test_normalize_arxiv_id_accepts_valid_forms_and_rejects_non_arxiv_urls() -> None:
    assert normalize_arxiv_id("1706.03762") == "1706.03762"
    assert normalize_arxiv_id("1706.03762v7") == "1706.03762v7"
    assert normalize_arxiv_id("https://arxiv.org/abs/1706.03762") == "1706.03762"
    assert normalize_arxiv_id("https://arxiv.org/pdf/1706.03762.pdf") == "1706.03762"
    assert normalize_arxiv_id("https://export.arxiv.org/pdf/2509.26626v1") == "2509.26626v1"

    with pytest.raises(ValueError, match="Only arXiv"):
        normalize_arxiv_id("https://example.com/paper.pdf")
    with pytest.raises(ValueError, match="Invalid arXiv"):
        normalize_arxiv_id("../1706.03762")


def test_ingest_pdf_file_stores_manifest_pages_and_chunks(tmp_path: Path) -> None:
    source_pdf = tmp_path / "source" / "attention.pdf"
    _write_fixture_pdf(source_pdf)
    context = _context(tmp_path)

    result = ingest_pdf_file(context, source_pdf, paper_id="attention")

    assert result["ok"] is True
    assert result["paper_id"] == "attention"
    assert result["page_count"] == 2
    paper_dir = Path(result["paper_dir"])
    assert (paper_dir / "paper.pdf").is_file()
    assert (paper_dir / "manifest.json").is_file()
    assert (paper_dir / "pages.jsonl").is_file()
    assert (paper_dir / "chunks.jsonl").is_file()
    manifest = json.loads((paper_dir / "manifest.json").read_text())
    assert manifest["sha256"]
    assert manifest["extraction_status"] == "ok"
    assert manifest["page_count"] == 2
    assert "scaled dot-product attention" in (paper_dir / "pages.jsonl").read_text()


def test_search_and_read_paper_cache_are_page_grounded(tmp_path: Path) -> None:
    source_pdf = tmp_path / "source" / "attention.pdf"
    _write_fixture_pdf(source_pdf)
    context = _context(tmp_path)
    ingest_pdf_file(context, source_pdf, paper_id="attention")

    search = search_paper_cache(context, paper_id="attention", query="softmax scaling", max_chunks=3)

    assert search["ok"] is True
    assert search["matches"]
    assert search["matches"][0]["page"] == 1
    assert "sqrt(d_k)" in search["matches"][0]["text"]

    pages = read_paper_pages_from_cache(context, paper_id="attention", pages="1-2")

    assert pages["ok"] is True
    assert [page["page"] for page in pages["pages"]] == [1, 2]
    assert "encoder-decoder" in pages["pages"][1]["text"]

    with pytest.raises(ValueError, match="out of range"):
        read_paper_pages_from_cache(context, paper_id="attention", pages="3")


def test_paper_cache_tools_accept_pdf_suffixed_paper_id_alias(tmp_path: Path) -> None:
    source_pdf = tmp_path / "source" / "attention.pdf"
    _write_fixture_pdf(source_pdf)
    context = _context(tmp_path)
    ingest_pdf_file(context, source_pdf, paper_id="attention")

    search = search_paper_cache(context, paper_id="attention.pdf", query="softmax")
    pages = read_paper_pages_from_cache(context, paper_id="attention.pdf", pages="1")
    rendered = render_paper_page_from_cache(context, paper_id="attention.pdf", page=1)

    assert search["paper_id"] == "attention"
    assert pages["paper_id"] == "attention"
    assert pages["pages"][0]["citation"] == "attention:p1"
    assert rendered["paper_id"] == "attention"
    assert rendered["relative_path"] == "paper_pages/attention_page_001.png"


def test_render_paper_page_writes_downloadable_artifact_candidate(tmp_path: Path) -> None:
    source_pdf = tmp_path / "source" / "attention.pdf"
    _write_fixture_pdf(source_pdf)
    context = _context(tmp_path)
    ingest_pdf_file(context, source_pdf, paper_id="attention")

    rendered = render_paper_page_from_cache(context, paper_id="attention", page=2)

    assert rendered["ok"] is True
    assert rendered["page"] == 2
    assert rendered["relative_path"] == "paper_pages/attention_page_002.png"
    output_path = Path(rendered["path"])
    assert output_path.is_file()
    assert output_path.read_bytes().startswith(b"\x89PNG")


def test_ingest_pdf_resource_can_index_selected_uploaded_pdf(tmp_path: Path) -> None:
    upload_root = tmp_path / "uploads"
    source_pdf = upload_root / "file-paper__file_abcd1234__attention.pdf"
    _write_fixture_pdf(source_pdf)
    context = AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="local-org",
        user_id="user-1",
        project_id="local-project",
        thread_id="thread-1",
        run_id="run-1",
        workspace_root=str(tmp_path / "workspace"),
        artifact_root=str(tmp_path / "artifacts" / "run-1"),
        selected_file_ids=("file-paper",),
    )

    result = _ingest_pdf_resource(context, path_or_uri="", upload_roots=(upload_root,))

    assert result["ok"] is True
    assert result["ingested"][0]["paper_id"] == "attention"
    manifest = json.loads(
        (Path(context.workspace_root) / "papers" / "attention" / "manifest.json").read_text()
    )
    assert manifest["page_count"] == 2


def test_paper_cache_can_be_reused_across_followup_run_contexts(tmp_path: Path) -> None:
    source_pdf = tmp_path / "source" / "attention.pdf"
    _write_fixture_pdf(source_pdf)
    cache_root = tmp_path / "paper-cache"
    first_context = _context(tmp_path)
    second_context = AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id=first_context.org_id,
        user_id=first_context.user_id,
        project_id="local-project",
        thread_id=first_context.thread_id,
        run_id="run-2",
        workspace_root=str(tmp_path / "workspace-2"),
        artifact_root=str(tmp_path / "artifacts" / "run-2"),
    )

    ingest_pdf_file(first_context, source_pdf, paper_id="attention", cache_root=cache_root)
    search = search_paper_cache(
        second_context,
        paper_id="attention",
        query="softmax scaling",
        cache_root=cache_root,
    )

    assert search["ok"] is True
    assert search["matches"][0]["citation"] == "attention:p1"


def test_build_paper_tools_registers_narrow_research_tools() -> None:
    tool_names = {tool.name for tool in build_paper_tools()}

    assert {
        "ingest_arxiv_paper",
        "ingest_pdf_resource",
        "paper_manifest",
        "search_paper",
        "read_paper_pages",
        "read_paper_section",
        "render_paper_page",
    }.issubset(tool_names)


def test_literature_reviewer_subagent_has_narrow_paper_tools(monkeypatch) -> None:
    captured = {}

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return "compiled-agent"

    monkeypatch.setattr("ultra_deepagents.agent.create_deep_agent", fake_create_deep_agent)
    settings = RuntimeSettings(
        openai_base_url="http://127.0.0.1:8003/v1",
        openai_model="deepseek_v4",
    )
    context = AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="local-org",
        user_id="user-1",
        project_id="local-project",
        thread_id="thread-1",
        run_id="run-paper",
        goal="Review the uploaded paper and explain its methods.",
        knowledge_context={
            "ingested_papers": [
                {"paper_id": "attention", "page_count": 2, "extraction_status": "ok"}
            ]
        },
    )

    build_research_agent(settings, model=object(), backend=object(), context=context)

    literature = next(
        subagent for subagent in captured["subagents"] if subagent["name"] == "literature-reviewer"
    )
    tool_names = {tool.name for tool in literature["tools"]}
    assert "search_paper" in tool_names
    assert "rarespot_ecology_inference" not in tool_names
    assert "page-grounded" in literature["system_prompt"].lower()
    assert "render_paper_page" in captured["system_prompt"]
