from __future__ import annotations

import hashlib
import io
import json
import time
from pathlib import Path

import pytest
import ultra_deepagents.papers.tools as paper_tools
from PIL import Image, ImageDraw
from ultra_deepagents.agent import build_research_agent
from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.context import AgentRunContext
from ultra_deepagents.papers.tools import (
    PAPER_PAGE_TEXT_SCHEMA,
    PAPER_TEXT_CHUNK_SCHEMA,
    PAPER_TEXT_LITERAL_BINDING_SCHEMA,
    PaperTextEvidenceError,
    _ingest_pdf_resource,
    bind_paper_text_literal_from_cache,
    build_paper_tools,
    ingest_arxiv_pdf,
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


def _write_image_only_pdf(path: Path) -> None:
    import fitz

    image_bytes = io.BytesIO()
    image = Image.new("RGB", (320, 180), color=(245, 245, 245))
    drawing = ImageDraw.Draw(image)
    drawing.text((12, 18), "Raster-only CALPHAD table", fill=(0, 0, 0))
    drawing.text((12, 62), "Solidus / K: 1720.15", fill=(0, 0, 0))
    drawing.text((12, 96), "Liquidus / K: 1760.15", fill=(0, 0, 0))
    image.save(image_bytes, format="PNG")
    doc = fitz.open()
    page = doc.new_page(width=400, height=300)
    page.insert_image((40, 50, 360, 230), stream=image_bytes.getvalue())
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)
    doc.close()


def _write_born_digital_table_pdf(path: Path) -> None:
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((54, 72), "Synthetic CALPHAD-style table benchmark", fontsize=18)
    page.insert_text((54, 120), "Table 1. Nominal alloy compositions (at.%)", fontsize=13)
    lines = [
        "Source              Al       Co       W",
        "Tomaszewska         9.1      81.7     9.2",
        "Migas               9.0      82.0     9.0",
    ]
    for index, line in enumerate(lines):
        page.insert_text((70, 165 + index * 30), line, fontsize=12, fontname="cour")
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)
    doc.close()


def _write_numeric_ambiguity_pdf(path: Path) -> None:
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    lines = [
        "Exact 125",
        "Inequality <126",
        "Uncertainty 127 ± 2",
        "Range 128-130",
        "Approx approximately 131",
        "Bypass about 132",
        "Partial 133 ± 3",
    ]
    for index, line in enumerate(lines):
        page.insert_text((54, 72 + index * 30), line, fontsize=12)
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
    pages = [json.loads(line) for line in (paper_dir / "pages.jsonl").read_text().splitlines()]
    chunks = [json.loads(line) for line in (paper_dir / "chunks.jsonl").read_text().splitlines()]
    assert manifest["text_extraction"]["page_record_schema"] == PAPER_PAGE_TEXT_SCHEMA
    assert manifest["text_extraction"]["chunk_record_schema"] == PAPER_TEXT_CHUNK_SCHEMA
    for page in pages:
        assert page["schema"] == PAPER_PAGE_TEXT_SCHEMA
        assert page["page_text_sha256"] == hashlib.sha256(page["text"].encode()).hexdigest()
        assert page["char_count"] == len(page["text"])
    page_by_number = {page["page"]: page for page in pages}
    for chunk in chunks:
        page = page_by_number[chunk["page"]]
        assert chunk["schema"] == PAPER_TEXT_CHUNK_SCHEMA
        assert chunk["page_text_sha256"] == page["page_text_sha256"]
        assert chunk["text"] == page["text"][chunk["start_char"] : chunk["end_char"]]


def test_pdf_byte_cap_rejects_before_copy_or_text_extraction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_pdf = tmp_path / "source" / "oversized.pdf"
    source_pdf.parent.mkdir(parents=True)
    with source_pdf.open("wb") as stream:
        stream.truncate(65)
    extraction_called = False

    def unexpected_extraction(*_args: object, **_kwargs: object) -> list[dict[str, object]]:
        nonlocal extraction_called
        extraction_called = True
        return []

    monkeypatch.setattr(paper_tools, "MAX_PDF_BYTES", 64)
    monkeypatch.setattr(paper_tools, "_extract_pdf_pages_from_bytes", unexpected_extraction)
    context = _context(tmp_path)

    with pytest.raises(ValueError, match="64-byte local processing limit"):
        ingest_pdf_file(context, source_pdf, paper_id="oversized")

    assert extraction_called is False
    assert not (paper_tools._paper_dir(context, "oversized") / "paper.pdf").exists()


def test_bounded_pdf_reader_accepts_exact_limit_and_rejects_one_byte_over(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = tmp_path / "exact.pdf"
    oversized = tmp_path / "oversized.pdf"
    exact.write_bytes(b"%PDF1234")
    oversized.write_bytes(b"%PDF12345")
    monkeypatch.setattr(paper_tools, "MAX_PDF_BYTES", 8)

    assert paper_tools._read_bounded_pdf_bytes(exact) == b"%PDF1234"
    with pytest.raises(ValueError, match="8-byte local processing limit"):
        paper_tools._read_bounded_pdf_bytes(oversized)


def test_arxiv_stream_enforces_pdf_cap_and_removes_partial_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    class FakeResponse:
        headers = {"content-type": "application/pdf"}

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def raise_for_status(self) -> None:
            return None

        def iter_bytes(self):
            yield b"%PDF"
            yield b"12345"

    class FakeClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def stream(self, method: str, url: str) -> FakeResponse:
            assert method == "GET"
            assert url.endswith("1706.03762.pdf")
            return FakeResponse()

    monkeypatch.setattr(paper_tools, "MAX_PDF_BYTES", 8)
    monkeypatch.setattr(httpx, "Client", FakeClient)
    context = _context(tmp_path)

    with pytest.raises(ValueError, match="8-byte local processing limit"):
        ingest_arxiv_pdf(context, "1706.03762")

    paper_dir = paper_tools._paper_dir(context, "arxiv_1706.03762")
    assert not (paper_dir / "download.pdf.partial").exists()
    assert not (paper_dir / "download.pdf").exists()


def test_search_and_read_paper_cache_are_page_grounded(tmp_path: Path) -> None:
    source_pdf = tmp_path / "source" / "attention.pdf"
    _write_fixture_pdf(source_pdf)
    context = _context(tmp_path)
    ingest_pdf_file(context, source_pdf, paper_id="attention")

    search = search_paper_cache(
        context, paper_id="attention", query="softmax scaling", max_chunks=3
    )

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
    ingested = ingest_pdf_file(context, source_pdf, paper_id="attention")

    rendered = render_paper_page_from_cache(context, paper_id="attention", page=2)

    assert rendered["ok"] is True
    assert rendered["page"] == 2
    assert rendered["relative_path"] == "paper_pages/attention_page_002.png"
    output_path = Path(rendered["path"])
    assert output_path.is_file()
    assert output_path.read_bytes().startswith(b"\x89PNG")
    cached_pdf = Path(ingested["paper_dir"]) / "paper.pdf"
    assert rendered["source_pdf_sha256"] == hashlib.sha256(cached_pdf.read_bytes()).hexdigest()
    assert rendered["render_zoom"] == 2.0
    assert rendered["render_width_px"] > 0
    assert rendered["render_height_px"] > 0
    assert rendered["rendered_png_sha256"] == hashlib.sha256(output_path.read_bytes()).hexdigest()


def test_render_paper_page_re_render_preserves_source_lineage_and_binds_pixels(
    tmp_path: Path,
) -> None:
    source_pdf = tmp_path / "source" / "attention.pdf"
    _write_fixture_pdf(source_pdf)
    context = _context(tmp_path)
    ingested = ingest_pdf_file(context, source_pdf, paper_id="attention")

    first = render_paper_page_from_cache(context, paper_id="attention", page=1, zoom=1.0)
    second = render_paper_page_from_cache(context, paper_id="attention", page=1, zoom=2.0)
    repeated = render_paper_page_from_cache(context, paper_id="attention", page=1, zoom=2.0)

    manifest = json.loads((Path(ingested["paper_dir"]) / "manifest.json").read_text())
    assert first["source_pdf_sha256"] == manifest["sha256"]
    assert second["source_pdf_sha256"] == manifest["sha256"]
    assert first["page"] == second["page"] == 1
    assert second["render_width_px"] == first["render_width_px"] * 2
    assert second["render_height_px"] == first["render_height_px"] * 2
    assert first["rendered_png_sha256"] != second["rendered_png_sha256"]
    assert repeated["rendered_png_sha256"] == second["rendered_png_sha256"]
    assert repeated["render_width_px"] == second["render_width_px"]
    assert repeated["render_height_px"] == second["render_height_px"]
    assert (
        second["rendered_png_sha256"]
        == hashlib.sha256(Path(second["path"]).read_bytes()).hexdigest()
    )


def test_render_paper_page_rejects_mutated_cached_pdf(tmp_path: Path) -> None:
    source_pdf = tmp_path / "source" / "attention.pdf"
    _write_fixture_pdf(source_pdf)
    context = _context(tmp_path)
    ingested = ingest_pdf_file(context, source_pdf, paper_id="attention")
    cached_pdf = Path(ingested["paper_dir"]) / "paper.pdf"
    cached_pdf.write_bytes(cached_pdf.read_bytes() + b"\n% post-ingestion mutation\n")

    with pytest.raises(ValueError, match="no longer matches its ingested SHA-256"):
        render_paper_page_from_cache(context, paper_id="attention", page=1)


def test_image_only_paper_is_renderable_but_not_represented_as_extracted_text(
    tmp_path: Path,
) -> None:
    source_pdf = tmp_path / "source" / "raster-table.pdf"
    _write_image_only_pdf(source_pdf)
    context = _context(tmp_path)

    ingested = ingest_pdf_file(context, source_pdf, paper_id="raster-table")
    paper_dir = Path(ingested["paper_dir"])
    manifest = json.loads((paper_dir / "manifest.json").read_text())
    pages = [json.loads(line) for line in (paper_dir / "pages.jsonl").read_text().splitlines()]

    assert ingested["extraction_status"] == "no_text"
    assert ingested["chunk_count"] == 0
    assert manifest["pages_with_text"] == 0
    assert len(pages) == 1
    assert pages[0]["schema"] == PAPER_PAGE_TEXT_SCHEMA
    assert pages[0]["page"] == 1
    assert pages[0]["text"] == ""
    assert pages[0]["char_count"] == 0
    assert pages[0]["page_text_sha256"] == hashlib.sha256(b"").hexdigest()
    assert (paper_dir / "chunks.jsonl").read_text() == ""

    rendered = render_paper_page_from_cache(context, paper_id="raster-table", page=1)
    assert Path(rendered["path"]).is_file()
    assert rendered["source_pdf_sha256"] == manifest["sha256"]
    assert (
        rendered["rendered_png_sha256"]
        == hashlib.sha256(Path(rendered["path"]).read_bytes()).hexdigest()
    )


def test_table_1_exact_numeric_literal_binds_to_replayed_page_text_span(
    tmp_path: Path,
) -> None:
    source_pdf = tmp_path / "source" / "synthetic-calphad-tables.pdf"
    _write_born_digital_table_pdf(source_pdf)
    context = _context(tmp_path)
    ingest_pdf_file(context, source_pdf, paper_id="synthetic-calphad-tables")

    binding = bind_paper_text_literal_from_cache(
        context,
        paper_id="synthetic-calphad-tables",
        page=1,
        exact_text="9.1",
        numeric_binding=True,
        row_id="tomaszewska",
        column_id="al-at-pct",
    )

    assert binding["ok"] is True
    assert binding["schema"] == PAPER_TEXT_LITERAL_BINDING_SCHEMA
    assert binding["exact_substring"] == "9.1"
    assert binding["numeric_value_decimal"] == "9.1"
    assert binding["end_char"] - binding["start_char"] == len("9.1")
    page = read_paper_pages_from_cache(
        context,
        paper_id="synthetic-calphad-tables",
        pages="1",
    )["pages"][0]
    assert page["text"][binding["start_char"] : binding["end_char"]] == "9.1"
    assert binding["page_text_sha256"] == page["page_text_sha256"]
    cross_check = binding["born_digital_cross_check"]
    assert cross_check["extractor_id"] == page["extractor_id"]
    assert cross_check["extractor_revision"] == page["extractor_revision"]
    assert cross_check["page_text_sha256"] == page["page_text_sha256"]
    assert cross_check["cells"] == [
        {
            "row_id": "tomaszewska",
            "column_id": "al-at-pct",
            "text": "9.1",
            "text_sha256": hashlib.sha256(b"9.1").hexdigest(),
            "start_char": binding["start_char"],
            "end_char": binding["end_char"],
        }
    ]


def test_repeated_table_literal_requires_exact_disambiguation_anchors(tmp_path: Path) -> None:
    source_pdf = tmp_path / "source" / "synthetic-calphad-tables.pdf"
    _write_born_digital_table_pdf(source_pdf)
    context = _context(tmp_path)
    ingest_pdf_file(context, source_pdf, paper_id="synthetic-calphad-tables")

    with pytest.raises(PaperTextEvidenceError) as ambiguous:
        bind_paper_text_literal_from_cache(
            context,
            paper_id="synthetic-calphad-tables",
            page=1,
            exact_text="9.0",
            numeric_binding=True,
        )
    assert ambiguous.value.code == "paper_text_literal_ambiguous"

    anchored = bind_paper_text_literal_from_cache(
        context,
        paper_id="synthetic-calphad-tables",
        page=1,
        exact_text="9.0",
        numeric_binding=True,
        exact_prefix="Migas               ",
        exact_suffix="      82.0",
    )
    assert anchored["exact_substring"] == "9.0"
    assert anchored["numeric_value_decimal"] == "9.0"
    assert anchored["match_count"] == 1

    with pytest.raises(PaperTextEvidenceError) as missing:
        bind_paper_text_literal_from_cache(
            context,
            paper_id="synthetic-calphad-tables",
            page=1,
            exact_text="99.9",
            numeric_binding=True,
        )
    assert missing.value.code == "paper_text_literal_missing"


def test_exact_literal_replay_and_response_stay_bounded(tmp_path: Path) -> None:
    source_pdf = tmp_path / "source" / "synthetic-calphad-tables.pdf"
    _write_born_digital_table_pdf(source_pdf)
    context = _context(tmp_path)
    ingest_pdf_file(context, source_pdf, paper_id="synthetic-calphad-tables")

    started = time.perf_counter()
    results = [
        bind_paper_text_literal_from_cache(
            context,
            paper_id="synthetic-calphad-tables",
            page=1,
            exact_text="9.1",
            numeric_binding=True,
        )
        for _ in range(25)
    ]
    elapsed = time.perf_counter() - started

    assert all(result["exact_substring"] == "9.1" for result in results)
    assert len(json.dumps(results[0])) < 8_000
    # Regression tripwire only; this is not a production latency SLO.
    assert elapsed < 5.0


def test_literal_binding_reextracts_pdf_and_rejects_forged_page_cache(tmp_path: Path) -> None:
    source_pdf = tmp_path / "source" / "synthetic-calphad-tables.pdf"
    _write_born_digital_table_pdf(source_pdf)
    context = _context(tmp_path)
    ingested = ingest_pdf_file(context, source_pdf, paper_id="synthetic-calphad-tables")
    pages_path = Path(ingested["paper_dir"]) / "pages.jsonl"
    pages = [json.loads(line) for line in pages_path.read_text().splitlines()]
    pages[0]["text"] = pages[0]["text"].replace("9.1", "9.9")
    pages[0]["page_text_sha256"] = hashlib.sha256(pages[0]["text"].encode()).hexdigest()
    pages[0]["char_count"] = len(pages[0]["text"])
    pages_path.write_text("".join(json.dumps(page, sort_keys=True) + "\n" for page in pages))

    with pytest.raises(PaperTextEvidenceError) as caught:
        bind_paper_text_literal_from_cache(
            context,
            paper_id="synthetic-calphad-tables",
            page=1,
            exact_text="9.9",
            numeric_binding=True,
        )
    assert caught.value.code == "cached_page_text_replay_mismatch"


@pytest.mark.parametrize(
    ("exact_text", "exact_prefix", "exact_suffix", "expected_code"),
    [
        ("Inequality <126", "", "", "nonliteral_numeric_text"),
        ("Uncertainty 127 ± 2", "", "", "nonliteral_numeric_text"),
        ("Range 128-130", "", "", "ambiguous_numeric_text"),
        ("Approx approximately 131", "", "", "nonliteral_numeric_text"),
        ("132", "Bypass about ", "", "nonliteral_numeric_context"),
        ("133", "Partial ", " ± 3", "nonliteral_numeric_context"),
    ],
)
def test_numeric_binding_refuses_ranges_inequalities_uncertainty_and_approximation(
    tmp_path: Path,
    exact_text: str,
    exact_prefix: str,
    exact_suffix: str,
    expected_code: str,
) -> None:
    source_pdf = tmp_path / "source" / "numeric-ambiguity.pdf"
    _write_numeric_ambiguity_pdf(source_pdf)
    context = _context(tmp_path)
    ingest_pdf_file(context, source_pdf, paper_id="numeric-ambiguity")

    with pytest.raises(PaperTextEvidenceError) as caught:
        bind_paper_text_literal_from_cache(
            context,
            paper_id="numeric-ambiguity",
            page=1,
            exact_text=exact_text,
            exact_prefix=exact_prefix,
            exact_suffix=exact_suffix,
            numeric_binding=True,
        )

    assert caught.value.code == expected_code


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
        "bind_paper_text_literal",
        "render_paper_page",
    }.issubset(tool_names)

    binding_tool = next(
        tool for tool in build_paper_tools() if tool.name == "bind_paper_text_literal"
    )
    assert set(binding_tool.args) == {
        "paper_id",
        "page",
        "exact_text",
        "numeric_binding",
        "exact_prefix",
        "exact_suffix",
        "row_id",
        "column_id",
    }
    assert not set(binding_tool.args).intersection({"path", "pdf_path", "cache_root", "code"})


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
    assert "bind_paper_text_literal" in tool_names
    assert "rarespot_ecology_inference" not in tool_names
    assert "page-grounded" in literature["system_prompt"].lower()
    assert literature["response_format"]["required"] == [
        "summary",
        "key_findings",
        "artifacts",
        "failures",
        "confidence",
        "confidence_basis",
    ]
    assert "render_paper_page" in captured["system_prompt"]
