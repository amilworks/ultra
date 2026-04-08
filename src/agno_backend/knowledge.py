from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.agentic.repositories import ScientificNoteRepository
from src.agents.context_packs import KNOWLEDGE_ROOT, load_context_packs


class ScientificKnowledgeScope(BaseModel):
    mode: Literal["default", "project_notebook"] = "project_notebook"
    project_id: str | None = None
    namespaces: list[str] = Field(
        default_factory=lambda: ["curated", "uploads", "session_notes", "project_notes"]
    )
    include_curated_packs: bool = True
    include_uploads: bool = True
    include_project_notes: bool = True


@dataclass(frozen=True)
class KnowledgeHit:
    namespace: str
    title: str
    body: str
    score: float
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScientificKnowledgeContext:
    scope: ScientificKnowledgeScope
    system_messages: list[dict[str, str]] = field(default_factory=list)
    hits: list[KnowledgeHit] = field(default_factory=list)
    namespaces_searched: list[str] = field(default_factory=list)
    project_id: str | None = None
    context_chars: int = 0

    def metadata(self) -> dict[str, Any]:
        return {
            "scope": self.scope.model_dump(mode="json"),
            "project_id": str(self.project_id or "").strip() or None,
            "namespaces": list(self.namespaces_searched),
            "hit_count": len(self.hits),
            "top_hits": [
                {
                    "namespace": hit.namespace,
                    "title": hit.title,
                    "score": round(float(hit.score), 4),
                    "provenance": dict(hit.provenance or {}),
                }
                for hit in self.hits[:6]
            ],
            "context_chars": int(self.context_chars),
        }


class ScientificKnowledgeHub:
    def __init__(
        self,
        *,
        notes: ScientificNoteRepository,
        curated_root: Path | None = None,
    ) -> None:
        self.notes = notes
        self.curated_root = Path(curated_root or KNOWLEDGE_ROOT).resolve()

    @staticmethod
    def normalize_scope(
        raw: dict[str, Any] | ScientificKnowledgeScope | None,
        *,
        default_project_id: str | None = None,
    ) -> ScientificKnowledgeScope:
        if isinstance(raw, ScientificKnowledgeScope):
            scope = raw
        elif isinstance(raw, dict):
            scope = ScientificKnowledgeScope.model_validate(raw)
        else:
            scope = ScientificKnowledgeScope()
        if scope.project_id:
            return scope
        normalized_project_id = str(default_project_id or "").strip() or None
        if normalized_project_id is None:
            return scope
        return scope.model_copy(update={"project_id": normalized_project_id})

    def retrieve_context(
        self,
        *,
        user_id: str | None,
        session_id: str | None,
        query: str,
        scope: ScientificKnowledgeScope,
        domain_id: str,
        workflow_hint: dict[str, Any] | None = None,
        knowledge_context: dict[str, Any] | None = None,
        selection_context: dict[str, Any] | None = None,
        uploaded_files: list[str] | None = None,
    ) -> ScientificKnowledgeContext:
        resolved_user_id = str(user_id or "").strip()
        resolved_session_id = str(session_id or "").strip()
        resolved_project_id = str(
            scope.project_id
            or ((knowledge_context or {}).get("project_id") if isinstance(knowledge_context, dict) else "")
            or ""
        ).strip() or None
        normalized_namespaces = self._normalize_namespaces(scope.namespaces)
        context = ScientificKnowledgeContext(
            scope=scope,
            namespaces_searched=normalized_namespaces,
            project_id=resolved_project_id,
        )

        ordered_hits: list[KnowledgeHit] = []
        if "project_notes" in normalized_namespaces and scope.include_project_notes and resolved_user_id:
            ordered_hits.extend(
                self._note_hits(
                    query=query,
                    user_id=resolved_user_id,
                    session_id=resolved_session_id or None,
                    project_id=resolved_project_id,
                )
            )
        if "session_notes" in normalized_namespaces and resolved_user_id and resolved_session_id:
            ordered_hits.extend(
                self._session_note_hits(
                    query=query,
                    user_id=resolved_user_id,
                    session_id=resolved_session_id,
                )
            )
        if "curated" in normalized_namespaces and scope.include_curated_packs:
            ordered_hits.extend(
                self._curated_hits(
                    query=query,
                    domain_id=domain_id,
                    workflow_hint=workflow_hint,
                    knowledge_context=knowledge_context,
                    project_id=resolved_project_id,
                )
            )
        if "uploads" in normalized_namespaces and scope.include_uploads:
            upload_hit = self._upload_hit(
                uploaded_files=uploaded_files,
                selection_context=selection_context,
            )
            if upload_hit is not None:
                ordered_hits.append(upload_hit)

        if not ordered_hits:
            return context

        context.hits = ordered_hits[:8]
        rendered = self._render_hits(context.hits)
        if rendered:
            context.system_messages.append({"role": "system", "content": rendered})
            context.context_chars = len(rendered)
        return context

    @staticmethod
    def _normalize_namespaces(raw: list[str] | None) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()
        for item in list(raw or []):
            token = str(item or "").strip().lower()
            if not token or token in seen:
                continue
            seen.add(token)
            ordered.append(token)
        return ordered or ["curated", "uploads", "session_notes", "project_notes"]

    def _note_hits(
        self,
        *,
        query: str,
        user_id: str,
        session_id: str | None,
        project_id: str | None,
    ) -> list[KnowledgeHit]:
        note_scope = "project_notes" if project_id else "session_notes"
        rows = self.notes.search_notes(
            query,
            user_id=user_id,
            scopes=[note_scope],
            project_id=project_id,
            session_id=None if project_id else session_id,
            limit=3,
        )
        return [
            KnowledgeHit(
                namespace=note_scope,
                title=str(row.get("title") or "Project note"),
                body=str(row.get("body") or "").strip(),
                score=float(row.get("search_score") or row.get("score") or 0.0),
                provenance=dict(row.get("provenance") or {}),
            )
            for row in rows
            if str(row.get("body") or "").strip()
        ]

    def _session_note_hits(
        self,
        *,
        query: str,
        user_id: str,
        session_id: str,
    ) -> list[KnowledgeHit]:
        rows = self.notes.search_notes(
            query,
            user_id=user_id,
            scopes=["session_notes"],
            session_id=session_id,
            limit=2,
        )
        return [
            KnowledgeHit(
                namespace="session_notes",
                title=str(row.get("title") or "Session note"),
                body=str(row.get("body") or "").strip(),
                score=float(row.get("search_score") or row.get("score") or 0.0),
                provenance=dict(row.get("provenance") or {}),
            )
            for row in rows
            if str(row.get("body") or "").strip()
        ]

    def _curated_hits(
        self,
        *,
        query: str,
        domain_id: str,
        workflow_hint: dict[str, Any] | None,
        knowledge_context: dict[str, Any] | None,
        project_id: str | None,
    ) -> list[KnowledgeHit]:
        query_text = str(query or "").strip().lower()
        workflow_id = str((workflow_hint or {}).get("id") or "").strip().lower()
        explicit_pack_ids = {
            str(item or "").strip().lower()
            for item in list((knowledge_context or {}).get("pack_ids") or [])
            if str(item or "").strip()
        }
        scored: list[tuple[float, KnowledgeHit]] = []
        for pack in load_context_packs(str(self.curated_root)):
            score = 0.0
            reasons: dict[str, Any] = {}
            if pack.always_on:
                score += 2.0
                reasons["always_on"] = True
            if domain_id and domain_id in pack.domain_ids:
                score += 1.5 + float(pack.priority)
                reasons["domain_id"] = domain_id
            if workflow_id and workflow_id in pack.workflow_ids:
                score += 2.5 + float(pack.priority)
                reasons["workflow_id"] = workflow_id
            if project_id and project_id in pack.project_ids:
                score += 3.0 + float(pack.priority)
                reasons["project_id"] = project_id
            if pack.pack_id in explicit_pack_ids:
                score += 5.0 + float(pack.priority)
                reasons["pack_id"] = pack.pack_id
            keyword_hits = [keyword for keyword in pack.keywords if self._keyword_match(query_text, keyword)]
            if keyword_hits:
                score += min(3.0, 0.85 * len(keyword_hits)) + float(pack.priority)
                reasons["keywords"] = keyword_hits[:4]
            if score <= 0.0:
                continue
            excerpt = self._pack_excerpt(pack.path, max_chars=int(pack.max_chars or 2200))
            if not excerpt:
                continue
            scored.append(
                (
                    score,
                    KnowledgeHit(
                        namespace="curated",
                        title=pack.title,
                        body=excerpt,
                        score=score,
                        provenance={
                            "pack_id": pack.pack_id,
                            "path": str(pack.path),
                            **reasons,
                        },
                    ),
                )
            )
        scored.sort(key=lambda item: -float(item[0]))
        return [hit for _score, hit in scored[:3]]

    @staticmethod
    def _keyword_match(query_text: str, keyword: str) -> bool:
        normalized_query = str(query_text or "").strip().lower()
        normalized_keyword = str(keyword or "").strip().lower()
        if not normalized_query or not normalized_keyword:
            return False
        return re.search(rf"\b{re.escape(normalized_keyword)}\b", normalized_query) is not None

    @staticmethod
    def _upload_hit(
        *,
        uploaded_files: list[str] | None,
        selection_context: dict[str, Any] | None,
    ) -> KnowledgeHit | None:
        labels: list[str] = []
        for raw in list(uploaded_files or []):
            token = str(raw or "").strip()
            if token:
                labels.append(Path(token).name)
        raw_selection = dict(selection_context or {})
        for key in ("focused_file_ids", "resource_uris", "dataset_uris"):
            values = raw_selection.get(key)
            if not isinstance(values, list):
                continue
            for raw in values[:8]:
                token = str(raw or "").strip()
                if token:
                    labels.append(token)
        if not labels:
            return None
        preview = ", ".join(labels[:8])
        overflow = max(0, len(labels) - 8)
        body = f"Active upload and resource context: {preview}"
        if overflow:
            body += f" (+{overflow} more)"
        return KnowledgeHit(
            namespace="uploads",
            title="Active uploads",
            body=body,
            score=0.65,
            provenance={"count": len(labels)},
        )

    @staticmethod
    def _pack_excerpt(path: Path, *, max_chars: int) -> str:
        try:
            raw = path.read_text(encoding="utf-8")
        except Exception:
            return ""
        if raw.startswith("+++"):
            parts = raw.split("+++", 2)
            if len(parts) == 3:
                raw = parts[2]
        return str(raw or "").strip()[: max(800, int(max_chars or 800))]

    @staticmethod
    def _render_hits(hits: list[KnowledgeHit]) -> str:
        if not hits:
            return ""
        lines = ["Scientific notebook and reference context:"]
        for hit in hits:
            title = str(hit.title or hit.namespace).strip()
            body = str(hit.body or "").strip()
            if not body:
                continue
            lines.append(f"{title} [{hit.namespace}]")
            lines.append(body[:420].strip())
        return "\n".join(lines).strip()
