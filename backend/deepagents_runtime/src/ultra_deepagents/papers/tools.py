from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

from langchain.tools import ToolRuntime, tool

from ultra_deepagents.context import AgentRunContext
from ultra_deepagents.context_tools import stage_uploaded_files

ARXIV_ID_RE = re.compile(r"^(?:\d{4}\.\d{4,5}|[A-Za-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?$")
MAX_CHARS_PER_CHUNK = 1400
CHUNK_OVERLAP_CHARS = 180


def normalize_arxiv_id(value: str) -> str:
    """Normalize an arXiv identifier or arXiv abs/pdf URL."""
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("Invalid arXiv identifier: empty value.")

    candidate = raw
    parsed = urlparse(raw)
    if parsed.scheme or parsed.netloc:
        host = parsed.netloc.lower()
        if host not in {"arxiv.org", "www.arxiv.org", "export.arxiv.org"}:
            raise ValueError("Only arXiv URLs are supported for paper ingestion.")
        path = parsed.path.strip("/")
        parts = path.split("/")
        if len(parts) >= 2 and parts[0] in {"abs", "pdf"}:
            candidate = "/".join(parts[1:])
        else:
            raise ValueError("Invalid arXiv URL. Expected /abs/<id> or /pdf/<id>.")

    candidate = candidate.removeprefix("arxiv:").strip().strip("/")
    candidate = re.sub(r"\.pdf$", "", candidate, flags=re.IGNORECASE)
    if not ARXIV_ID_RE.fullmatch(candidate):
        raise ValueError(f"Invalid arXiv identifier: {raw!r}.")
    return candidate


def ingest_pdf_file(
    context: AgentRunContext,
    source_pdf: str | Path,
    *,
    paper_id: str | None = None,
    cache_root: str | Path | None = None,
    source_url: str = "",
    title: str = "",
    authors: Iterable[str] = (),
    abstract: str = "",
) -> dict[str, Any]:
    """Copy and index a PDF into the current run's paper cache."""
    source = Path(source_pdf).expanduser().resolve()
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"PDF not found: {source}")
    if source.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got {source.name!r}.")

    safe_paper_id = _safe_paper_id(paper_id or source.stem)
    paper_dir = _paper_dir(context, safe_paper_id, cache_root=cache_root)
    paper_dir.mkdir(parents=True, exist_ok=True)
    target_pdf = paper_dir / "paper.pdf"
    if source.resolve() != target_pdf.resolve():
        shutil.copy2(source, target_pdf)

    sha256 = _file_sha256(target_pdf)
    pages = _extract_pdf_pages(target_pdf)
    chunks = _chunk_pages(safe_paper_id, pages)
    extraction_status = "ok" if any(page["text"].strip() for page in pages) else "no_text"
    manifest = {
        "paper_id": safe_paper_id,
        "title": title or safe_paper_id,
        "authors": [str(author) for author in authors if str(author).strip()],
        "abstract": abstract,
        "source_url": source_url,
        "pdf_path": str(target_pdf),
        "sha256": sha256,
        "page_count": len(pages),
        "pages_with_text": sum(1 for page in pages if page["text"].strip()),
        "chunk_count": len(chunks),
        "extraction_status": extraction_status,
    }
    _write_json(paper_dir / "manifest.json", manifest)
    _write_jsonl(paper_dir / "pages.jsonl", pages)
    _write_jsonl(paper_dir / "chunks.jsonl", chunks)

    return {
        "ok": True,
        "paper_id": safe_paper_id,
        "paper_dir": str(paper_dir),
        "page_count": len(pages),
        "chunk_count": len(chunks),
        "extraction_status": extraction_status,
        "sha256": sha256,
    }


def ingest_arxiv_pdf(
    context: AgentRunContext,
    arxiv_id_or_url: str,
    *,
    cache_root: str | Path | None = None,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    arxiv_id = normalize_arxiv_id(arxiv_id_or_url)
    paper_id = _safe_paper_id(f"arxiv_{arxiv_id}")
    paper_dir = _paper_dir(context, paper_id, cache_root=cache_root)
    existing_manifest = paper_dir / "manifest.json"
    if existing_manifest.exists():
        manifest = _read_json(existing_manifest)
        return {
            "ok": True,
            "paper_id": paper_id,
            "cached": True,
            "paper_dir": str(paper_dir),
            "page_count": manifest.get("page_count", 0),
            "chunk_count": manifest.get("chunk_count", 0),
            "extraction_status": manifest.get("extraction_status", "unknown"),
        }

    import httpx

    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    paper_dir.mkdir(parents=True, exist_ok=True)
    download_path = paper_dir / "download.pdf"
    with httpx.Client(timeout=timeout_seconds, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "pdf" not in content_type.lower() and not response.content.startswith(b"%PDF"):
            raise ValueError(f"arXiv response for {arxiv_id} was not a PDF.")
        download_path.write_bytes(response.content)

    return ingest_pdf_file(
        context,
        download_path,
        paper_id=paper_id,
        cache_root=cache_root,
        source_url=url,
        title=f"arXiv {arxiv_id}",
    )


def paper_manifest_cache(
    context: AgentRunContext,
    *,
    cache_root: str | Path | None = None,
) -> dict[str, Any]:
    papers = []
    root = _paper_root(context, cache_root=cache_root)
    if root.exists():
        for manifest_path in sorted(root.glob("*/manifest.json")):
            manifest = _read_json(manifest_path)
            papers.append(
                {
                    "paper_id": manifest.get("paper_id"),
                    "title": manifest.get("title"),
                    "source_url": manifest.get("source_url"),
                    "page_count": manifest.get("page_count"),
                    "chunk_count": manifest.get("chunk_count"),
                    "extraction_status": manifest.get("extraction_status"),
                }
            )
    return {"ok": True, "papers": papers}


def search_paper_cache(
    context: AgentRunContext,
    *,
    paper_id: str,
    query: str,
    max_chunks: int = 8,
    cache_root: str | Path | None = None,
) -> dict[str, Any]:
    safe_paper_id, paper_dir = _resolve_cached_paper(context, paper_id, cache_root=cache_root)
    chunks = _load_jsonl(paper_dir / "chunks.jsonl")
    if not chunks:
        raise ValueError(f"Paper {paper_id!r} is not indexed.")
    query_tokens = _tokens(query)
    scored = []
    for chunk in chunks:
        text = str(chunk.get("text") or "")
        text_tokens = _tokens(text)
        if query_tokens:
            overlap = sum(text_tokens.count(token) for token in query_tokens)
            score = overlap / max(len(query_tokens), 1)
        else:
            score = 0.0
        if score > 0 or not query_tokens:
            item = dict(chunk)
            item["score"] = round(float(score), 3)
            item["citation"] = f"{safe_paper_id}:p{item.get('page')}"
            scored.append(item)
    scored.sort(key=lambda item: (-float(item.get("score") or 0), int(item.get("page") or 0)))
    limit = max(1, min(int(max_chunks or 8), 20))
    return {"ok": True, "paper_id": safe_paper_id, "matches": scored[:limit]}


def read_paper_pages_from_cache(
    context: AgentRunContext,
    *,
    paper_id: str,
    pages: str,
    cache_root: str | Path | None = None,
) -> dict[str, Any]:
    safe_paper_id, paper_dir = _resolve_cached_paper(context, paper_id, cache_root=cache_root)
    all_pages = _load_jsonl(paper_dir / "pages.jsonl")
    if not all_pages:
        raise ValueError(f"Paper {paper_id!r} has no extracted pages.")
    page_map = {int(page["page"]): page for page in all_pages}
    requested = _parse_page_spec(pages, max_page=max(page_map))
    return {
        "ok": True,
        "paper_id": safe_paper_id,
        "pages": [
            {
                "page": page_number,
                "citation": f"{safe_paper_id}:p{page_number}",
                "text": str(page_map[page_number].get("text") or ""),
            }
            for page_number in requested
        ],
    }


def read_paper_section_from_cache(
    context: AgentRunContext,
    *,
    paper_id: str,
    section_query: str,
    max_chunks: int = 6,
    cache_root: str | Path | None = None,
) -> dict[str, Any]:
    result = search_paper_cache(
        context,
        paper_id=paper_id,
        query=section_query,
        max_chunks=max_chunks,
        cache_root=cache_root,
    )
    result["section_query"] = section_query
    return result


def render_paper_page_from_cache(
    context: AgentRunContext,
    *,
    paper_id: str,
    page: int,
    zoom: float = 2.0,
    cache_root: str | Path | None = None,
) -> dict[str, Any]:
    safe_paper_id, paper_dir = _resolve_cached_paper(context, paper_id, cache_root=cache_root)
    manifest = _read_json(paper_dir / "manifest.json")
    page_count = int(manifest.get("page_count") or 0)
    page_number = int(page)
    if page_number < 1 or page_number > page_count:
        raise ValueError(f"Page {page_number} out of range for {safe_paper_id} with {page_count} pages.")

    fitz = _import_fitz()
    doc = fitz.open(str(paper_dir / "paper.pdf"))
    try:
        matrix = fitz.Matrix(float(zoom), float(zoom))
        pixmap = doc.load_page(page_number - 1).get_pixmap(matrix=matrix, alpha=False)
        relative_path = f"paper_pages/{safe_paper_id}_page_{page_number:03d}.png"
        output_path = Path(context.artifact_root).expanduser().resolve() / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pixmap.save(str(output_path))
    finally:
        doc.close()

    return {
        "ok": True,
        "paper_id": safe_paper_id,
        "page": page_number,
        "path": str(output_path),
        "relative_path": relative_path,
        "sandbox_path": f"/outputs/{relative_path}",
        "citation": f"{safe_paper_id}:p{page_number}",
    }


def build_paper_tools(
    upload_roots: Iterable[str | Path] = (),
    *,
    cache_root: str | Path | None = None,
) -> list[Any]:
    resolved_upload_roots = tuple(upload_roots)
    resolved_cache_root = Path(cache_root).expanduser().resolve() if cache_root is not None else None

    @tool
    def ingest_arxiv_paper(runtime: ToolRuntime[AgentRunContext], arxiv_id_or_url: str) -> str:
        """Fetch an arXiv PDF, extract page text/chunks, and cache it for page-grounded review."""
        return _tool_result_json(
            lambda: ingest_arxiv_pdf(
                runtime.context,
                arxiv_id_or_url,
                cache_root=resolved_cache_root,
            )
        )

    @tool
    def ingest_pdf_resource(
        runtime: ToolRuntime[AgentRunContext],
        path_or_uri: str = "",
        file_ids: list[str] | str | None = None,
    ) -> str:
        """Index a selected/uploaded/local PDF for paper review. Omit path_or_uri to ingest selected PDF uploads."""
        return _tool_result_json(
            lambda: _ingest_pdf_resource(
                runtime.context,
                path_or_uri=path_or_uri,
                upload_roots=resolved_upload_roots,
                cache_root=resolved_cache_root,
                file_ids=file_ids,
            )
        )

    @tool
    def paper_manifest(runtime: ToolRuntime[AgentRunContext]) -> str:
        """List papers already ingested in this run with paper_id, title, page count, and extraction status."""
        return _tool_result_json(
            lambda: paper_manifest_cache(runtime.context, cache_root=resolved_cache_root)
        )

    @tool
    def search_paper(
        runtime: ToolRuntime[AgentRunContext],
        paper_id: str,
        query: str,
        max_chunks: int = 8,
    ) -> str:
        """Search an ingested paper by lexical query and return compact page-grounded chunks."""
        return _tool_result_json(
            lambda: search_paper_cache(
                runtime.context,
                paper_id=paper_id,
                query=query,
                max_chunks=max_chunks,
                cache_root=resolved_cache_root,
            )
        )

    @tool
    def read_paper_pages(runtime: ToolRuntime[AgentRunContext], paper_id: str, pages: str) -> str:
        """Read exact extracted text for paper pages such as "1", "2-4", or "2,5-6"."""
        return _tool_result_json(
            lambda: read_paper_pages_from_cache(
                runtime.context,
                paper_id=paper_id,
                pages=pages,
                cache_root=resolved_cache_root,
            )
        )

    @tool
    def read_paper_section(
        runtime: ToolRuntime[AgentRunContext],
        paper_id: str,
        section_query: str,
        max_chunks: int = 6,
    ) -> str:
        """Find a paper section or topic and return relevant page-grounded chunks."""
        return _tool_result_json(
            lambda: read_paper_section_from_cache(
                runtime.context,
                paper_id=paper_id,
                section_query=section_query,
                max_chunks=max_chunks,
                cache_root=resolved_cache_root,
            )
        )

    @tool
    def render_paper_page(runtime: ToolRuntime[AgentRunContext], paper_id: str, page: int) -> str:
        """Render one PDF page into /outputs/paper_pages so the UI can show it inline as a figure artifact."""
        return _tool_result_json(
            lambda: render_paper_page_from_cache(
                runtime.context,
                paper_id=paper_id,
                page=page,
                cache_root=resolved_cache_root,
            )
        )

    return [
        ingest_arxiv_paper,
        ingest_pdf_resource,
        paper_manifest,
        search_paper,
        read_paper_pages,
        read_paper_section,
        render_paper_page,
    ]


def _ingest_pdf_resource(
    context: AgentRunContext,
    *,
    path_or_uri: str = "",
    upload_roots: Iterable[str | Path] = (),
    cache_root: str | Path | None = None,
    file_ids: list[str] | str | None = None,
) -> dict[str, Any]:
    requested_path = str(path_or_uri or "").strip()
    if requested_path.startswith("http://") or requested_path.startswith("https://"):
        raise ValueError("Only arXiv links may be fetched from the network; use ingest_arxiv_paper.")

    if not requested_path or requested_path.startswith("file_"):
        requested_file_ids = file_ids if file_ids is not None else requested_path or None
        staged = stage_uploaded_files(
            context,
            upload_roots=upload_roots,
            file_ids=requested_file_ids,
        )
        results = []
        for file_info in staged.get("staged_files") or []:
            staged_path = Path(str(file_info.get("staged_path") or ""))
            if staged_path.suffix.lower() == ".pdf":
                results.append(
                    ingest_pdf_file(
                        context,
                        staged_path,
                        paper_id=paper_id_from_pdf_path(staged_path),
                        cache_root=cache_root,
                        title=staged_path.name,
                    )
                )
        return {
            "ok": bool(results),
            "ingested": results,
            "staged": staged,
            "error": "" if results else "no_selected_pdf_uploads",
        }

    path = _resolve_workspace_path(context, requested_path)
    return ingest_pdf_file(context, path, cache_root=cache_root)


def _resolve_workspace_path(context: AgentRunContext, raw_path: str) -> Path:
    raw = raw_path.removeprefix("file://")
    workspace_root = Path(context.workspace_root).expanduser().resolve()
    artifact_root = Path(context.artifact_root).expanduser().resolve()
    if raw.startswith("/workspace/"):
        candidate = workspace_root / raw.removeprefix("/workspace/")
    elif raw.startswith("/outputs/"):
        candidate = artifact_root / raw.removeprefix("/outputs/")
    else:
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = workspace_root / candidate
    resolved = candidate.resolve()
    if not (_is_under(resolved, workspace_root) or _is_under(resolved, artifact_root)):
        raise ValueError("PDF path is outside the current run workspace/artifact roots.")
    return resolved


def paper_id_from_pdf_path(path: str | Path) -> str:
    stem = Path(path).stem
    while True:
        stripped = re.sub(r"^file_[A-Za-z0-9]+__", "", stem)
        if stripped == stem:
            break
        stem = stripped
    return _safe_paper_id(stem)


def _extract_pdf_pages(pdf_path: Path) -> list[dict[str, Any]]:
    fitz = _import_fitz()
    doc = fitz.open(str(pdf_path))
    pages: list[dict[str, Any]] = []
    try:
        for index in range(doc.page_count):
            text = doc.load_page(index).get_text("text").strip()
            pages.append({"page": index + 1, "text": text})
    finally:
        doc.close()
    return pages


def _chunk_pages(paper_id: str, pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for page in pages:
        page_number = int(page["page"])
        text = re.sub(r"\n{3,}", "\n\n", str(page.get("text") or "")).strip()
        if not text:
            continue
        start = 0
        chunk_index = 1
        while start < len(text):
            end = min(len(text), start + MAX_CHARS_PER_CHUNK)
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    {
                        "paper_id": paper_id,
                        "chunk_id": f"{paper_id}:p{page_number}:c{chunk_index}",
                        "page": page_number,
                        "text": chunk_text,
                    }
                )
            if end >= len(text):
                break
            start = max(end - CHUNK_OVERLAP_CHARS, start + 1)
            chunk_index += 1
    return chunks


def _parse_page_spec(pages: str, *, max_page: int) -> list[int]:
    requested: list[int] = []
    for token in str(pages or "").split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text.strip())
            end = int(end_text.strip())
            if start > end:
                raise ValueError(f"Invalid page range {token!r}.")
            requested.extend(range(start, end + 1))
        else:
            requested.append(int(token))
    if not requested:
        raise ValueError("At least one page must be requested.")
    unique = []
    seen: set[int] = set()
    for page in requested:
        if page < 1 or page > max_page:
            raise ValueError(f"Page {page} out of range for paper with {max_page} pages.")
        if page not in seen:
            seen.add(page)
            unique.append(page)
    return unique


def _paper_root(
    context: AgentRunContext,
    *,
    cache_root: str | Path | None = None,
) -> Path:
    if cache_root is None:
        return Path(context.workspace_root).expanduser().resolve() / "papers"
    root = Path(cache_root).expanduser().resolve()
    return root / _safe_paper_id(context.org_id) / _safe_paper_id(context.user_id)


def _paper_dir(
    context: AgentRunContext,
    paper_id: str,
    *,
    cache_root: str | Path | None = None,
) -> Path:
    return _paper_root(context, cache_root=cache_root) / _safe_paper_id(paper_id)


def _resolve_cached_paper(
    context: AgentRunContext,
    paper_id: str,
    *,
    cache_root: str | Path | None = None,
) -> tuple[str, Path]:
    safe_paper_id = _safe_paper_id(paper_id)
    root = _paper_root(context, cache_root=cache_root)
    candidates = [safe_paper_id]
    without_pdf = re.sub(r"\.pdf$", "", safe_paper_id, flags=re.IGNORECASE)
    if without_pdf != safe_paper_id:
        candidates.append(without_pdf)
    for candidate in candidates:
        paper_dir = root / candidate
        if (paper_dir / "manifest.json").exists():
            return candidate, paper_dir
    return safe_paper_id, root / safe_paper_id


def _safe_paper_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "")).strip("._")
    if not safe:
        raise ValueError("paper_id cannot be empty.")
    return safe


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"Missing paper cache file: {path.name}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected object in {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _tokens(value: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", value.lower())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _tool_json(value: dict[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True)


def _tool_result_json(fn: Any) -> str:
    try:
        return _tool_json(fn())
    except Exception as exc:
        return _tool_json(
            {
                "ok": False,
                "error": str(exc),
                "error_type": type(exc).__name__,
            }
        )


def _import_fitz() -> Any:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF is required for paper PDF parsing. Install the runtime with the paper dependencies."
        ) from exc
    return fitz
