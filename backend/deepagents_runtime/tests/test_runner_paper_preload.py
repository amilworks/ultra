from __future__ import annotations

from pathlib import Path

from ultra_deepagents.context import AgentRunContext
from ultra_deepagents.runner import (
    _extract_arxiv_references,
    _preload_arxiv_papers_for_context,
    _preload_uploaded_pdf_papers_for_context,
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


def test_extract_arxiv_references_deduplicates_goal_and_messages() -> None:
    references = _extract_arxiv_references(
        goal="Read https://arxiv.org/pdf/1706.03762.pdf",
        messages=[
            {"role": "user", "content": "Follow up on arXiv:1706.03762 and 2509.26626v1."},
            {"role": "assistant", "content": [{"type": "text", "text": "https://arxiv.org/abs/1706.03762"}]},
        ],
    )

    assert references == ["https://arxiv.org/pdf/1706.03762.pdf", "2509.26626v1"]


def test_preload_arxiv_papers_ingests_links_and_surfaces_runtime_context(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def fake_ingest(context, arxiv_id_or_url, *, cache_root):
        calls.append((context.run_id, arxiv_id_or_url, Path(cache_root)))
        return {
            "ok": True,
            "paper_id": "arxiv_1706.03762",
            "page_count": 15,
            "chunk_count": 40,
            "extraction_status": "ok",
        }

    monkeypatch.setattr("ultra_deepagents.runner.ingest_arxiv_pdf", fake_ingest)
    context = _context(tmp_path)

    updated = _preload_arxiv_papers_for_context(
        context,
        goal="Read https://arxiv.org/pdf/1706.03762.pdf",
        messages=[],
        cache_root=tmp_path / "papers",
    )

    assert calls == [("run-1", "https://arxiv.org/pdf/1706.03762.pdf", tmp_path / "papers")]
    assert updated is not context
    assert updated.knowledge_context["ingested_papers"] == [
        {
            "paper_id": "arxiv_1706.03762",
            "source": "https://arxiv.org/pdf/1706.03762.pdf",
            "page_count": 15,
            "chunk_count": 40,
            "extraction_status": "ok",
        }
    ]


def test_preload_uploaded_pdf_papers_ingests_file_ids_and_surfaces_runtime_context(
    monkeypatch, tmp_path: Path
) -> None:
    staged_pdf = (
        tmp_path
        / "workspace"
        / "staged_uploads"
        / "file-1"
        / "file_abcd1234__attention_is_all_you_need.pdf"
    )
    staged_pdf.parent.mkdir(parents=True, exist_ok=True)
    staged_pdf.write_bytes(b"%PDF fixture")
    calls = []

    def fake_stage(context, *, upload_roots, file_ids=None):
        calls.append(("stage", context.selected_file_ids, tuple(upload_roots), file_ids))
        return {
            "ok": True,
            "staged_files": [
                {
                    "file_id": "file-1",
                    "staged_path": str(staged_pdf),
                }
            ],
            "missing_file_ids": [],
        }

    def fake_ingest(context, source_pdf, *, paper_id, cache_root, title="", source_url="", **_kwargs):
        calls.append(
            (
                "ingest",
                context.run_id,
                Path(source_pdf),
                paper_id,
                Path(cache_root),
                title,
                source_url,
            )
        )
        return {
            "ok": True,
            "paper_id": paper_id,
            "page_count": 15,
            "chunk_count": 40,
            "extraction_status": "ok",
        }

    monkeypatch.setattr("ultra_deepagents.runner.stage_uploaded_files", fake_stage)
    monkeypatch.setattr("ultra_deepagents.runner.ingest_pdf_file", fake_ingest)
    context = AgentRunContext(
        assistant_id="ultra-research-agent",
        org_id="local-org",
        user_id="user-1",
        project_id="local-project",
        thread_id="thread-1",
        run_id="run-1",
        workspace_root=str(tmp_path / "workspace"),
        artifact_root=str(tmp_path / "artifacts" / "run-1"),
        selected_file_ids=("file-1",),
    )

    updated = _preload_uploaded_pdf_papers_for_context(
        context,
        upload_roots=(tmp_path / "uploads",),
        cache_root=tmp_path / "papers",
    )

    assert calls == [
        ("stage", ("file-1",), (tmp_path / "uploads",), None),
        (
            "ingest",
            "run-1",
            staged_pdf,
            "attention_is_all_you_need",
            tmp_path / "papers",
            "file_abcd1234__attention_is_all_you_need.pdf",
            "upload:file-1",
        ),
    ]
    assert updated is not context
    assert updated.knowledge_context["ingested_papers"] == [
        {
            "paper_id": "attention_is_all_you_need",
            "source": "upload:file-1",
            "page_count": 15,
            "chunk_count": 40,
            "extraction_status": "ok",
        }
    ]
