from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import shutil
import stat
from collections.abc import Iterable
from pathlib import Path
from typing import Any, NoReturn
from urllib.parse import urlparse

from langchain.tools import ToolRuntime, tool

from ultra_deepagents.context import AgentRunContext
from ultra_deepagents.context_tools import stage_uploaded_files
from ultra_deepagents.papers.table_evidence import (
    PaperTableEvidenceValidationError,
    parse_exact_numeric_literal,
)

ARXIV_ID_RE = re.compile(r"^(?:\d{4}\.\d{4,5}|[A-Za-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?$")
MAX_CHARS_PER_CHUNK = 1400
CHUNK_OVERLAP_CHARS = 180
MAX_PAGE_TEXT_CHARS = 5_000_000
MAX_EXACT_LITERAL_CHARS = 4_096
MAX_EXACT_ANCHOR_CHARS = 1_024
# A local paper operation reads the complete immutable PDF bytes for hashing and
# PyMuPDF replay. Keep that allocation below a fixed worker-safe ceiling even though
# the general upload service intentionally accepts multi-gigabyte scientific data.
MAX_PDF_BYTES = 128 * 1024 * 1024

PAPER_PAGE_TEXT_SCHEMA = "ultra.paper-page-text.v1"
PAPER_TEXT_CHUNK_SCHEMA = "ultra.paper-text-chunk.v1"
PAPER_TEXT_LITERAL_BINDING_SCHEMA = "ultra.paper-text-literal-binding.v1"
PAPER_TEXT_EXTRACTION_SCHEMA = "ultra.paper-text-extraction.v1"
PAPER_TEXT_EXTRACTOR_ID = "pymupdf-page-get-text"
PAPER_TEXT_PIPELINE_REVISION = "ultra-page-text-v1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_STABLE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_NONEXACT_NUMERIC_PREFIX_RE = re.compile(
    r"(?:"
    r"[<>≤≥~≈≃∼]\s*|"
    r"\b(?:about|approx(?:imately)?\.?|around|circa|ca\.?|roughly|near(?:ly)?|"
    r"less\s+than|greater\s+than|at\s+least|at\s+most|more\s+than|"
    r"no\s+more\s+than|no\s+less\s+than|up\s+to|below|above|under|over|between)\s*|"
    r"\d\s*(?:-|–|—|to)\s*"
    r")$",
    re.IGNORECASE,
)


def _pdf_size_error() -> ValueError:
    return ValueError(f"PDF exceeds the {MAX_PDF_BYTES}-byte local processing limit")


def _require_bounded_pdf_file(path: Path) -> int:
    """Reject non-regular or oversized PDFs before copy/render/model work."""

    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("PDF source must be a regular file")
    if info.st_size > MAX_PDF_BYTES:
        raise _pdf_size_error()
    return int(info.st_size)


def _read_bounded_pdf_bytes(path: Path) -> bytes:
    """Read one PDF under the hard byte ceiling and reject concurrent size changes."""

    _require_bounded_pdf_file(path)
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("PDF source must be a regular file")
        if before.st_size > MAX_PDF_BYTES:
            raise _pdf_size_error()
        payload = stream.read(MAX_PDF_BYTES + 1)
        if len(payload) > MAX_PDF_BYTES:
            raise _pdf_size_error()
        after = os.fstat(stream.fileno())
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or after.st_size != len(payload)
    ):
        raise ValueError("PDF changed while its immutable processing bytes were read")
    return payload


_NONEXACT_NUMERIC_SUFFIX_RE = re.compile(
    r"^\s*(?:"
    r"±|\+/-|plus(?:\s+or)?\s+minus|"
    r"(?:-|–|—|to)\s*[+−-]?(?:\d|\.)|"
    r"\(\s*\d|"
    r"or\s+(?:less|more)|and\s+(?:below|above)|at\s+(?:least|most)"
    r")",
    re.IGNORECASE,
)


class PaperTextEvidenceError(ValueError):
    """Stable fail-closed error for exact born-digital source bindings."""

    def __init__(self, code: str, path: str, message: str) -> None:
        self.code = code
        self.path = path
        self.message = message
        super().__init__(f"{code} at {path}: {message}")


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
    _require_bounded_pdf_file(source)
    if source.resolve() != target_pdf.resolve():
        shutil.copy2(source, target_pdf)

    # Read once, then hash and extract from the same immutable byte string. This avoids
    # binding cached page text to a different PDF revision through a hash/open TOCTOU gap.
    source_pdf_bytes = _read_bounded_pdf_bytes(target_pdf)
    sha256 = hashlib.sha256(source_pdf_bytes).hexdigest()
    pages = _extract_pdf_pages_from_bytes(
        source_pdf_bytes,
        source_pdf_sha256=sha256,
    )
    chunks = _chunk_pages(safe_paper_id, pages)
    extraction_status = "ok" if any(page["text"].strip() for page in pages) else "no_text"
    extractor_id, extractor_revision = _paper_text_extractor_binding()
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
        "text_extraction": {
            "schema": PAPER_TEXT_EXTRACTION_SCHEMA,
            "extractor_id": extractor_id,
            "extractor_revision": extractor_revision,
            "page_record_schema": PAPER_PAGE_TEXT_SCHEMA,
            "chunk_record_schema": PAPER_TEXT_CHUNK_SCHEMA,
            "span_semantics": "zero-based Unicode code-point offsets; end_char is exclusive",
        },
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
        "text_extractor_id": extractor_id,
        "text_extractor_revision": extractor_revision,
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
    partial_path = paper_dir / "download.pdf.partial"
    try:
        with httpx.Client(timeout=timeout_seconds, follow_redirects=True) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                content_type = response.headers.get("content-type", "")
                content_length = response.headers.get("content-length", "").strip()
                if content_length:
                    try:
                        declared_bytes = int(content_length)
                    except ValueError as exc:
                        raise ValueError("arXiv returned an invalid Content-Length") from exc
                    if declared_bytes < 0 or declared_bytes > MAX_PDF_BYTES:
                        raise _pdf_size_error()
                total = 0
                prefix = bytearray()
                with partial_path.open("wb") as stream:
                    for chunk in response.iter_bytes():
                        total += len(chunk)
                        if total > MAX_PDF_BYTES:
                            raise _pdf_size_error()
                        if len(prefix) < 4:
                            prefix.extend(chunk[: 4 - len(prefix)])
                        stream.write(chunk)
                if "pdf" not in content_type.lower() and not bytes(prefix).startswith(b"%PDF"):
                    raise ValueError(f"arXiv response for {arxiv_id} was not a PDF.")
        os.replace(partial_path, download_path)
    finally:
        try:
            partial_path.unlink()
        except FileNotFoundError:
            pass

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
                    "text_extraction": manifest.get("text_extraction"),
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
            _public_page_text_record(
                safe_paper_id,
                page_number,
                page_map[page_number],
            )
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


def bind_paper_text_literal_from_cache(
    context: AgentRunContext,
    *,
    paper_id: str,
    page: int,
    exact_text: str,
    numeric_binding: bool = False,
    exact_prefix: str = "",
    exact_suffix: str = "",
    row_id: str = "",
    column_id: str = "",
    cache_root: str | Path | None = None,
) -> dict[str, Any]:
    """Bind one exact born-digital substring to a verified cached PDF page.

    Matching is case- and whitespace-sensitive. Optional exact prefix/suffix anchors may
    disambiguate a repeated literal, but the returned span always covers ``exact_text`` only.
    Numeric mode accepts one exact decimal/scientific literal and rejects uncertainty,
    inequalities, ranges, and approximation qualifiers rather than selecting one endpoint.
    """

    if isinstance(page, bool) or not isinstance(page, int) or page < 1:
        _text_fail("invalid_page", "$.page", "page must be a positive integer")
    if not isinstance(numeric_binding, bool):
        _text_fail(
            "invalid_numeric_binding",
            "$.numeric_binding",
            "numeric_binding must be true or false",
        )
    literal = _bounded_exact_text(
        exact_text,
        "$.exact_text",
        maximum=MAX_EXACT_LITERAL_CHARS,
        allow_empty=False,
    )
    prefix = _bounded_exact_text(
        exact_prefix,
        "$.exact_prefix",
        maximum=MAX_EXACT_ANCHOR_CHARS,
        allow_empty=True,
    )
    suffix = _bounded_exact_text(
        exact_suffix,
        "$.exact_suffix",
        maximum=MAX_EXACT_ANCHOR_CHARS,
        allow_empty=True,
    )
    row = str(row_id or "").strip()
    column = str(column_id or "").strip()
    if bool(row) != bool(column):
        _text_fail(
            "cross_check_coordinate_incomplete",
            "$.row_id",
            "row_id and column_id must either both be supplied or both be omitted",
        )
    if row and (not _STABLE_ID_RE.fullmatch(row) or not _STABLE_ID_RE.fullmatch(column)):
        _text_fail(
            "invalid_cross_check_coordinate",
            "$.row_id",
            "row_id and column_id must be stable ASCII identifiers",
        )

    safe_paper_id, page_record, source_pdf_sha256 = _verified_cached_page_text(
        context,
        paper_id=paper_id,
        page=page,
        cache_root=cache_root,
    )
    page_text = str(page_record["text"])
    anchor = f"{prefix}{literal}{suffix}"
    starts = _exact_occurrence_starts(page_text, anchor, limit=2)
    if not starts:
        _text_fail(
            "paper_text_literal_missing",
            "$.exact_text",
            "the exact case- and whitespace-sensitive literal/anchor was not found on the page",
        )
    if len(starts) != 1:
        _text_fail(
            "paper_text_literal_ambiguous",
            "$.exact_text",
            "the exact literal/anchor occurs more than once; supply bounded exact prefix/suffix anchors",
        )
    anchor_start = starts[0]
    start_char = anchor_start + len(prefix)
    end_char = start_char + len(literal)
    if page_text[start_char:end_char] != literal:
        _text_fail(
            "paper_text_span_inconsistent",
            "$.exact_text",
            "the resolved character span does not reproduce the exact literal",
        )

    numeric_value_decimal: str | None = None
    if numeric_binding:
        try:
            numeric_value = parse_exact_numeric_literal(literal, "$.exact_text")
        except PaperTableEvidenceValidationError as exc:
            _text_fail(exc.code, exc.path, exc.message)
        _reject_nonexact_adjacent_numeric_context(page_text, start_char, end_char)
        numeric_value_decimal = str(numeric_value)

    text_sha256 = hashlib.sha256(literal.encode("utf-8")).hexdigest()
    extractor_id = str(page_record["extractor_id"])
    extractor_revision = str(page_record["extractor_revision"])
    page_text_sha256 = str(page_record["page_text_sha256"])
    cell_check: dict[str, Any] = {
        "text": literal,
        "text_sha256": text_sha256,
        "start_char": start_char,
        "end_char": end_char,
    }
    result: dict[str, Any] = {
        "ok": True,
        "schema": PAPER_TEXT_LITERAL_BINDING_SCHEMA,
        "paper_id": safe_paper_id,
        "page": page,
        "citation": f"{safe_paper_id}:p{page}",
        "source_pdf_sha256": source_pdf_sha256,
        "page_text_sha256": page_text_sha256,
        "page_text_char_count": len(page_text),
        "extractor_id": extractor_id,
        "extractor_revision": extractor_revision,
        "start_char": start_char,
        "end_char": end_char,
        "exact_substring": page_text[start_char:end_char],
        "text_sha256": text_sha256,
        "span_semantics": "zero-based Unicode code-point offsets; end_char is exclusive",
        "numeric_binding_requested": numeric_binding,
        "numeric_value_decimal": numeric_value_decimal,
        "match_count": 1,
        "disambiguation": {
            "anchor_start_char": anchor_start,
            "anchor_end_char": anchor_start + len(anchor),
            "anchor_sha256": hashlib.sha256(anchor.encode("utf-8")).hexdigest(),
            "prefix_char_count": len(prefix),
            "prefix_sha256": hashlib.sha256(prefix.encode("utf-8")).hexdigest(),
            "suffix_char_count": len(suffix),
            "suffix_sha256": hashlib.sha256(suffix.encode("utf-8")).hexdigest(),
        },
        "born_digital_cell_source_span": cell_check,
    }
    if row and column:
        cell_check = {"row_id": row, "column_id": column, **cell_check}
        result["born_digital_cross_check"] = {
            "extractor_id": extractor_id,
            "extractor_revision": extractor_revision,
            "page_text_sha256": page_text_sha256,
            "cells": [cell_check],
        }
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
        raise ValueError(
            f"Page {page_number} out of range for {safe_paper_id} with {page_count} pages."
        )

    try:
        render_zoom = float(zoom)
    except (TypeError, ValueError) as exc:
        raise ValueError("Paper page render zoom must be a finite positive number.") from exc
    if not math.isfinite(render_zoom) or render_zoom <= 0 or render_zoom > 8:
        raise ValueError("Paper page render zoom must be within (0, 8].")

    # Render the exact bytes whose digest was recorded at ingestion. Opening the
    # path after hashing would leave a hash/render TOCTOU window, while trusting the
    # mutable manifest alone could bind a newly rendered page to stale source
    # provenance. Reading once and opening that byte string closes both gaps.
    paper_path = paper_dir / "paper.pdf"
    source_pdf_bytes = _read_bounded_pdf_bytes(paper_path)
    source_pdf_sha256 = hashlib.sha256(source_pdf_bytes).hexdigest()
    declared_source_sha256 = str(manifest.get("sha256") or "")
    if (
        re.fullmatch(r"[0-9a-f]{64}", declared_source_sha256) is None
        or declared_source_sha256 != source_pdf_sha256
    ):
        raise ValueError("Cached paper PDF no longer matches its ingested SHA-256.")

    fitz = _import_fitz()
    doc = fitz.open(stream=source_pdf_bytes, filetype="pdf")
    try:
        if int(doc.page_count) != page_count:
            raise ValueError("Cached paper PDF page count no longer matches its manifest.")
        matrix = fitz.Matrix(render_zoom, render_zoom)
        pixmap = doc.load_page(page_number - 1).get_pixmap(matrix=matrix, alpha=False)
        render_width_px = int(pixmap.width)
        render_height_px = int(pixmap.height)
        relative_path = f"paper_pages/{safe_paper_id}_page_{page_number:03d}.png"
        output_path = Path(context.artifact_root).expanduser().resolve() / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rendered_png_bytes = pixmap.tobytes(output="png")
        rendered_png_sha256 = hashlib.sha256(rendered_png_bytes).hexdigest()
        output_path.write_bytes(rendered_png_bytes)
    finally:
        doc.close()

    return {
        "ok": True,
        "paper_id": safe_paper_id,
        "page": page_number,
        "source_pdf_sha256": source_pdf_sha256,
        "render_zoom": render_zoom,
        "render_width_px": render_width_px,
        "render_height_px": render_height_px,
        "rendered_png_sha256": rendered_png_sha256,
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
    resolved_cache_root = (
        Path(cache_root).expanduser().resolve() if cache_root is not None else None
    )

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
    def bind_paper_text_literal(
        runtime: ToolRuntime[AgentRunContext],
        paper_id: str,
        page: int,
        exact_text: str,
        numeric_binding: bool = False,
        exact_prefix: str = "",
        exact_suffix: str = "",
        row_id: str = "",
        column_id: str = "",
    ) -> str:
        """Bind one exact born-digital PDF substring to its page-text hash and character span. Matching is case/whitespace-sensitive and fails if missing or ambiguous. Set numeric_binding=true only for one exact numeric literal; ranges, inequalities, uncertainty, and approximations are refused. Prefix/suffix are bounded exact disambiguation anchors. Optional row_id+column_id return a born_digital_cross_check fragment."""

        return _tool_result_json(
            lambda: bind_paper_text_literal_from_cache(
                runtime.context,
                paper_id=paper_id,
                page=page,
                exact_text=exact_text,
                numeric_binding=numeric_binding,
                exact_prefix=exact_prefix,
                exact_suffix=exact_suffix,
                row_id=row_id,
                column_id=column_id,
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
        bind_paper_text_literal,
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
        raise ValueError(
            "Only arXiv links may be fetched from the network; use ingest_arxiv_paper."
        )

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
    source_pdf_bytes = _read_bounded_pdf_bytes(pdf_path)
    return _extract_pdf_pages_from_bytes(
        source_pdf_bytes,
        source_pdf_sha256=hashlib.sha256(source_pdf_bytes).hexdigest(),
    )


def _paper_text_extractor_binding() -> tuple[str, str]:
    fitz = _import_fitz()
    version = str(getattr(fitz, "VersionBind", "") or getattr(fitz, "__version__", "")).strip()
    if not version or not any(character.isdigit() for character in version):
        raise RuntimeError("PyMuPDF did not expose an immutable text-extractor version.")
    return PAPER_TEXT_EXTRACTOR_ID, f"pymupdf-{version}+{PAPER_TEXT_PIPELINE_REVISION}"


def _page_text_record(
    page_object: Any,
    *,
    page_number: int,
    source_pdf_sha256: str,
    extractor_id: str,
    extractor_revision: str,
) -> dict[str, Any]:
    # The normalization is deliberately tiny and versioned: preserve PyMuPDF's exact
    # character order/whitespace inside the page and remove only document-edge whitespace.
    text = str(page_object.get_text("text", sort=False)).strip()
    if len(text) > MAX_PAGE_TEXT_CHARS:
        _text_fail(
            "paper_page_text_budget_exceeded",
            "$.page",
            f"extracted page text exceeds {MAX_PAGE_TEXT_CHARS} Unicode code points",
        )
    return {
        "schema": PAPER_PAGE_TEXT_SCHEMA,
        "page": page_number,
        "source_pdf_sha256": source_pdf_sha256,
        "extractor_id": extractor_id,
        "extractor_revision": extractor_revision,
        "page_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "char_count": len(text),
        "span_semantics": "zero-based Unicode code-point offsets; end_char is exclusive",
        "text": text,
    }


def _extract_pdf_pages_from_bytes(
    source_pdf_bytes: bytes,
    *,
    source_pdf_sha256: str,
) -> list[dict[str, Any]]:
    fitz = _import_fitz()
    extractor_id, extractor_revision = _paper_text_extractor_binding()
    doc = fitz.open(stream=source_pdf_bytes, filetype="pdf")
    pages: list[dict[str, Any]] = []
    try:
        for index in range(doc.page_count):
            pages.append(
                _page_text_record(
                    doc.load_page(index),
                    page_number=index + 1,
                    source_pdf_sha256=source_pdf_sha256,
                    extractor_id=extractor_id,
                    extractor_revision=extractor_revision,
                )
            )
    finally:
        doc.close()
    return pages


def _extract_pdf_page_from_bytes(
    source_pdf_bytes: bytes,
    *,
    source_pdf_sha256: str,
    page: int,
) -> tuple[dict[str, Any], int]:
    fitz = _import_fitz()
    extractor_id, extractor_revision = _paper_text_extractor_binding()
    doc = fitz.open(stream=source_pdf_bytes, filetype="pdf")
    try:
        page_count = int(doc.page_count)
        if page < 1 or page > page_count:
            _text_fail(
                "paper_page_out_of_range",
                "$.page",
                f"page {page} is outside the cached PDF's {page_count} pages",
            )
        record = _page_text_record(
            doc.load_page(page - 1),
            page_number=page,
            source_pdf_sha256=source_pdf_sha256,
            extractor_id=extractor_id,
            extractor_revision=extractor_revision,
        )
    finally:
        doc.close()
    return record, page_count


def _chunk_pages(paper_id: str, pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for page in pages:
        page_number = int(page["page"])
        text = str(page.get("text") or "")
        if not text:
            continue
        start = 0
        chunk_index = 1
        while start < len(text):
            end = min(len(text), start + MAX_CHARS_PER_CHUNK)
            segment = text[start:end]
            start_char = start + (len(segment) - len(segment.lstrip()))
            end_char = end - (len(segment) - len(segment.rstrip()))
            chunk_text = text[start_char:end_char]
            if chunk_text:
                chunks.append(
                    {
                        "schema": PAPER_TEXT_CHUNK_SCHEMA,
                        "paper_id": paper_id,
                        "chunk_id": f"{paper_id}:p{page_number}:c{chunk_index}",
                        "page": page_number,
                        "source_pdf_sha256": page["source_pdf_sha256"],
                        "extractor_id": page["extractor_id"],
                        "extractor_revision": page["extractor_revision"],
                        "page_text_sha256": page["page_text_sha256"],
                        "start_char": start_char,
                        "end_char": end_char,
                        "span_semantics": page["span_semantics"],
                        "text": chunk_text,
                    }
                )
            if end >= len(text):
                break
            start = max(end - CHUNK_OVERLAP_CHARS, start + 1)
            chunk_index += 1
    return chunks


def _public_page_text_record(
    paper_id: str,
    page_number: int,
    page: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "page": page_number,
        "citation": f"{paper_id}:p{page_number}",
        "text": str(page.get("text") or ""),
    }
    for key in (
        "schema",
        "source_pdf_sha256",
        "extractor_id",
        "extractor_revision",
        "page_text_sha256",
        "char_count",
        "span_semantics",
    ):
        if key in page:
            result[key] = page[key]
    return result


def _verified_cached_page_text(
    context: AgentRunContext,
    *,
    paper_id: str,
    page: int,
    cache_root: str | Path | None,
) -> tuple[str, dict[str, Any], str]:
    safe_paper_id, paper_dir = _resolve_cached_paper(context, paper_id, cache_root=cache_root)
    try:
        manifest = _read_json(paper_dir / "manifest.json")
        source_pdf_bytes = _read_bounded_pdf_bytes(paper_dir / "paper.pdf")
        pages = _load_jsonl(paper_dir / "pages.jsonl")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _text_fail(
            "paper_text_cache_unreadable",
            "$.paper_id",
            f"the cached paper text evidence could not be read safely ({type(exc).__name__})",
        )

    declared_pdf_sha256 = manifest.get("sha256")
    if not isinstance(declared_pdf_sha256, str) or not _SHA256_RE.fullmatch(declared_pdf_sha256):
        _text_fail(
            "paper_text_identity_unavailable",
            "$.paper_id",
            "the paper cache has no valid ingested PDF identity; re-ingest the selected PDF",
        )
    observed_pdf_sha256 = hashlib.sha256(source_pdf_bytes).hexdigest()
    if not hmac.compare_digest(declared_pdf_sha256, observed_pdf_sha256):
        _text_fail(
            "cached_pdf_sha256_mismatch",
            "$.paper_id",
            "the cached PDF no longer matches its ingested SHA-256",
        )

    extractor_id, extractor_revision = _paper_text_extractor_binding()
    extraction = manifest.get("text_extraction")
    if not isinstance(extraction, dict) or any(
        extraction.get(key) != expected
        for key, expected in {
            "schema": PAPER_TEXT_EXTRACTION_SCHEMA,
            "extractor_id": extractor_id,
            "extractor_revision": extractor_revision,
            "page_record_schema": PAPER_PAGE_TEXT_SCHEMA,
            "chunk_record_schema": PAPER_TEXT_CHUNK_SCHEMA,
        }.items()
    ):
        _text_fail(
            "paper_text_identity_unavailable",
            "$.paper_id",
            "the paper cache predates this exact text extractor or uses another revision; re-ingest it",
        )

    matching_pages = [
        record
        for record in pages
        if isinstance(record.get("page"), int)
        and not isinstance(record.get("page"), bool)
        and record.get("page") == page
    ]
    if len(matching_pages) != 1:
        code = "paper_page_text_missing" if not matching_pages else "paper_page_text_duplicate"
        _text_fail(
            code,
            "$.page",
            "the cache must contain exactly one extracted text record for the requested page",
        )
    cached_record = matching_pages[0]
    cached_text = cached_record.get("text")
    cached_text_sha256 = cached_record.get("page_text_sha256")
    if (
        cached_record.get("schema") != PAPER_PAGE_TEXT_SCHEMA
        or cached_record.get("source_pdf_sha256") != declared_pdf_sha256
        or cached_record.get("extractor_id") != extractor_id
        or cached_record.get("extractor_revision") != extractor_revision
        or not isinstance(cached_text, str)
        or len(cached_text) > MAX_PAGE_TEXT_CHARS
        or not isinstance(cached_text_sha256, str)
        or not _SHA256_RE.fullmatch(cached_text_sha256)
        or isinstance(cached_record.get("char_count"), bool)
        or cached_record.get("char_count") != len(cached_text)
        or cached_record.get("span_semantics")
        != "zero-based Unicode code-point offsets; end_char is exclusive"
    ):
        _text_fail(
            "paper_page_text_record_invalid",
            "$.page",
            "the cached page-text record is incomplete or internally inconsistent",
        )
    computed_text_sha256 = hashlib.sha256(cached_text.encode("utf-8")).hexdigest()
    if not hmac.compare_digest(cached_text_sha256, computed_text_sha256):
        _text_fail(
            "cached_page_text_sha256_mismatch",
            "$.page",
            "the cached page text no longer matches its recorded SHA-256",
        )

    try:
        reextracted, actual_page_count = _extract_pdf_page_from_bytes(
            source_pdf_bytes,
            source_pdf_sha256=declared_pdf_sha256,
            page=page,
        )
    except PaperTextEvidenceError:
        raise
    except Exception as exc:
        _text_fail(
            "paper_text_replay_failed",
            "$.page",
            f"the exact cached PDF page could not be replayed safely ({type(exc).__name__})",
        )
    declared_page_count = manifest.get("page_count")
    if (
        isinstance(declared_page_count, bool)
        or not isinstance(declared_page_count, int)
        or declared_page_count != actual_page_count
    ):
        _text_fail(
            "cached_pdf_page_count_mismatch",
            "$.paper_id",
            "the cached PDF page count no longer matches its manifest",
        )
    exact_fields = (
        "schema",
        "page",
        "source_pdf_sha256",
        "extractor_id",
        "extractor_revision",
        "page_text_sha256",
        "char_count",
        "span_semantics",
        "text",
    )
    if any(cached_record.get(key) != reextracted.get(key) for key in exact_fields):
        _text_fail(
            "cached_page_text_replay_mismatch",
            "$.page",
            "re-extracting the exact cached PDF bytes did not reproduce the page-text record",
        )
    return safe_paper_id, reextracted, declared_pdf_sha256


def _bounded_exact_text(
    value: Any,
    path: str,
    *,
    maximum: int,
    allow_empty: bool,
) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        _text_fail("invalid_exact_text", path, "expected bounded non-empty exact text")
    if len(value) > maximum:
        _text_fail("exact_text_budget_exceeded", path, f"text exceeds {maximum} characters")
    if any(ord(character) < 32 and character not in {"\t", "\n", "\r"} for character in value):
        _text_fail("invalid_exact_text", path, "text contains a forbidden control character")
    return value


def _exact_occurrence_starts(value: str, needle: str, *, limit: int) -> list[int]:
    starts: list[int] = []
    cursor = 0
    while len(starts) < limit:
        found = value.find(needle, cursor)
        if found < 0:
            break
        starts.append(found)
        cursor = found + 1
    return starts


def _reject_nonexact_adjacent_numeric_context(
    page_text: str,
    start_char: int,
    end_char: int,
) -> None:
    before = page_text[max(0, start_char - 96) : start_char]
    after = page_text[end_char : min(len(page_text), end_char + 96)]
    if _NONEXACT_NUMERIC_PREFIX_RE.search(before) or _NONEXACT_NUMERIC_SUFFIX_RE.match(after):
        _text_fail(
            "nonliteral_numeric_context",
            "$.exact_text",
            "adjacent page text marks the literal as an inequality, range, approximation, or uncertainty",
        )


def _text_fail(code: str, path: str, message: str) -> NoReturn:
    raise PaperTextEvidenceError(code, path, message)


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
    except PaperTextEvidenceError as exc:
        return _tool_json(
            {
                "ok": False,
                "error": exc.message,
                "error_code": exc.code,
                "error_path": exc.path,
                "error_type": type(exc).__name__,
            }
        )
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
