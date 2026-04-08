"""File-backed reference context packs for bounded specialist handoffs."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10 test envs
    tomllib = None

from .contracts import ContextSnippet, KnowledgeContext, TurnIntent

KNOWLEDGE_ROOT = Path(__file__).resolve().parent / "knowledge"
_FRONTMATTER_DELIMITER = "+++"


@dataclass(frozen=True)
class ContextPack:
    """A small, curated markdown pack that can be injected into a handoff."""

    pack_id: str
    title: str
    path: Path
    scope: str = "domain"
    domain_ids: tuple[str, ...] = ()
    workflow_ids: tuple[str, ...] = ()
    collaborator_ids: tuple[str, ...] = ()
    project_ids: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    priority: int = 0
    always_on: bool = False
    max_chars: int = 2800


def _normalize_token(value: str | None) -> str:
    return str(value or "").strip().lower()


def _normalize_list(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    ordered: list[str] = []
    seen: set[str] = set()
    for item in value:
        token = _normalize_token(str(item or ""))
        if not token or token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return tuple(ordered)


def _extract_frontmatter(markdown: str) -> tuple[dict[str, object], str]:
    text = str(markdown or "")
    if not text.startswith(_FRONTMATTER_DELIMITER):
        return {}, text.strip()
    lines = text.splitlines()
    if not lines or lines[0].strip() != _FRONTMATTER_DELIMITER:
        return {}, text.strip()
    try:
        end_index = next(
            index
            for index in range(1, len(lines))
            if lines[index].strip() == _FRONTMATTER_DELIMITER
        )
    except StopIteration:
        return {}, text.strip()
    frontmatter = "\n".join(lines[1:end_index]).strip()
    body = "\n".join(lines[end_index + 1 :]).strip()
    parsed = _parse_frontmatter_toml(frontmatter) if frontmatter else {}
    return parsed, body


def _parse_frontmatter_toml(frontmatter: str) -> dict[str, object]:
    if tomllib is not None:
        return tomllib.loads(frontmatter)
    parsed: dict[str, object] = {}
    for line in str(frontmatter or "").splitlines():
        stripped = str(line or "").strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, raw_value = stripped.split("=", 1)
        parsed[str(key or "").strip()] = _parse_simple_toml_value(raw_value.strip())
    return parsed


def _parse_simple_toml_value(raw_value: str) -> object:
    lowered = str(raw_value or "").strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return ast.literal_eval(raw_value)
    except Exception:
        return str(raw_value or "").strip().strip("\"'")


def _context_pack_from_file(path: Path) -> ContextPack | None:
    raw = path.read_text(encoding="utf-8")
    frontmatter, body = _extract_frontmatter(raw)
    pack_id = _normalize_token(str(frontmatter.get("pack_id") or path.stem))
    title = str(frontmatter.get("title") or path.stem.replace("_", " ")).strip()
    if not pack_id or not title or not body:
        return None
    try:
        priority = int(frontmatter.get("priority") or 0)
    except Exception:
        priority = 0
    try:
        max_chars = int(frontmatter.get("max_chars") or 2800)
    except Exception:
        max_chars = 2800
    return ContextPack(
        pack_id=pack_id,
        title=title,
        path=path.resolve(),
        scope=str(frontmatter.get("scope") or "domain").strip() or "domain",
        domain_ids=_normalize_list(frontmatter.get("domains")),
        workflow_ids=_normalize_list(frontmatter.get("workflow_ids")),
        collaborator_ids=_normalize_list(frontmatter.get("collaborators")),
        project_ids=_normalize_list(frontmatter.get("projects")),
        keywords=_normalize_list(frontmatter.get("keywords")),
        priority=priority,
        always_on=bool(frontmatter.get("always_on")),
        max_chars=max_chars,
    )


@lru_cache(maxsize=4)
def load_context_packs(root: str | None = None) -> tuple[ContextPack, ...]:
    knowledge_root = Path(root).resolve() if root else KNOWLEDGE_ROOT.resolve()
    if not knowledge_root.exists():
        return ()
    packs: list[ContextPack] = []
    for path in sorted(knowledge_root.rglob("*.md")):
        if not path.is_file():
            continue
        pack = _context_pack_from_file(path)
        if pack is not None:
            packs.append(pack)
    return tuple(packs)


def select_reference_context(
    *,
    domain_id: str,
    turn_intent: TurnIntent,
    selected_tool_names: Iterable[str] | None = None,
    root: str | None = None,
    limit: int = 3,
) -> list[ContextSnippet]:
    """Return deterministic markdown snippets that match the current specialist turn."""

    normalized_text = _compose_lookup_text(
        turn_intent=turn_intent,
        selected_tool_names=selected_tool_names,
    )
    workflow_id = str(turn_intent.workflow_hint.get("id") or "").strip().lower()
    knowledge_context = (
        turn_intent.knowledge_context
        if isinstance(turn_intent.knowledge_context, KnowledgeContext)
        else KnowledgeContext()
    )
    collaborator_id = _normalize_token(knowledge_context.collaborator_id)
    project_id = _normalize_token(knowledge_context.project_id)
    explicit_pack_ids = {_normalize_token(item) for item in knowledge_context.pack_ids}
    scored: list[tuple[float, str, ContextSnippet]] = []
    for pack in load_context_packs(root):
        if pack.domain_ids and _normalize_token(domain_id) not in pack.domain_ids:
            continue
        keyword_hits = [keyword for keyword in pack.keywords if _keyword_match(normalized_text, keyword)]
        if (
            pack.collaborator_ids
            and (
                (collaborator_id and collaborator_id not in pack.collaborator_ids)
                or (
                    not collaborator_id
                    and not keyword_hits
                    and pack.pack_id not in explicit_pack_ids
                )
            )
            and pack.pack_id not in explicit_pack_ids
        ):
            continue
        if (
            pack.project_ids
            and (
                (project_id and project_id not in pack.project_ids)
                or (
                    not project_id
                    and not keyword_hits
                    and pack.pack_id not in explicit_pack_ids
                )
            )
            and pack.pack_id not in explicit_pack_ids
        ):
            continue

        score = float(pack.priority)
        reasons: list[str] = []
        if _normalize_token(domain_id) in pack.domain_ids:
            score += 1.0
            reasons.append(f"domain:{domain_id}")
        if workflow_id and workflow_id in pack.workflow_ids:
            score += 3.0
            reasons.append(f"workflow:{workflow_id}")
        if pack.always_on:
            score += 2.5
            reasons.append("always_on")
        if pack.pack_id in explicit_pack_ids:
            score += 6.0
            reasons.append(f"pack:{pack.pack_id}")
        if collaborator_id and collaborator_id in pack.collaborator_ids:
            score += 4.0
            reasons.append(f"collaborator:{collaborator_id}")
        if project_id and project_id in pack.project_ids:
            score += 5.0
            reasons.append(f"project:{project_id}")
        if keyword_hits:
            score += min(4.0, 0.75 * len(keyword_hits))
            reasons.append("keywords:" + ",".join(keyword_hits[:4]))
        if score <= 0.0:
            continue

        excerpt = _load_pack_excerpt(pack)
        scored.append(
            (
                score,
                pack.pack_id,
                ContextSnippet(
                    pack_id=pack.pack_id,
                    title=pack.title,
                    source_path=str(pack.path.relative_to(KNOWLEDGE_ROOT.parent)),
                    excerpt=excerpt,
                    match_reason="; ".join(reasons),
                ),
            )
        )
    scored.sort(key=lambda item: (-float(item[0]), item[1]))
    return [snippet for _score, _pack_id, snippet in scored[: max(1, int(limit or 1))]]


def render_reference_context_block(snippets: Sequence[ContextSnippet]) -> str:
    """Render selected snippets as a small markdown block for specialist prompts."""

    cleaned = [snippet for snippet in snippets if str(snippet.excerpt or "").strip()]
    if not cleaned:
        return ""
    lines: list[str] = ["Reference context:"]
    for snippet in cleaned:
        lines.append(f"{snippet.title} [{snippet.pack_id}]")
        if str(snippet.match_reason or "").strip():
            lines.append(f"Matched by: {snippet.match_reason}")
        lines.append(str(snippet.excerpt).strip())
    return "\n".join(lines).strip()


def _compose_lookup_text(*, turn_intent: TurnIntent, selected_tool_names: Iterable[str] | None) -> str:
    parts = [str(turn_intent.original_user_text or "").strip()]
    resolved_context = str(turn_intent.resolved_context_text or "").strip()
    if resolved_context:
        parts.append(resolved_context)
    originating = str(turn_intent.resource_focus.originating_user_text or "").strip()
    if originating:
        parts.append(originating)
    workflow_id = str(turn_intent.workflow_hint.get("id") or "").strip()
    if workflow_id:
        parts.append(workflow_id)
    collaborator_id = str(turn_intent.knowledge_context.collaborator_id or "").strip()
    if collaborator_id:
        parts.append(collaborator_id)
    project_id = str(turn_intent.knowledge_context.project_id or "").strip()
    if project_id:
        parts.append(project_id)
    parts.extend(
        str(item or "").strip()
        for item in turn_intent.knowledge_context.pack_ids
        if str(item or "").strip()
    )
    parts.extend(str(name or "").strip() for name in (selected_tool_names or []) if str(name or "").strip())
    parts.extend(
        str(name or "").strip()
        for name in turn_intent.resource_focus.suggested_tool_names
        if str(name or "").strip()
    )
    parts.extend(
        str(uri or "").strip()
        for uri in turn_intent.resource_focus.resource_uris
        if str(uri or "").strip()
    )
    parts.extend(
        str(uri or "").strip()
        for uri in turn_intent.resource_focus.dataset_uris
        if str(uri or "").strip()
    )
    return "\n".join(part for part in parts if part)


def _load_pack_excerpt(pack: ContextPack) -> str:
    try:
        text = pack.path.read_text(encoding="utf-8")
    except Exception as exc:  # pragma: no cover - only exercised if repo packaging breaks
        return f"Unable to load context pack: {exc}"
    _frontmatter, text = _extract_frontmatter(text)
    normalized = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(normalized) <= pack.max_chars:
        return normalized
    truncated = normalized[: pack.max_chars].rstrip()
    return f"{truncated}..."


def _keyword_match(normalized_text: str, keyword: str) -> bool:
    token = _normalize(keyword)
    if not token or not normalized_text:
        return False
    if " " in token:
        return token in normalized_text
    return bool(re.search(rf"\b{re.escape(token)}\b", normalized_text))


def _normalize(text: str) -> str:
    tokenized = re.sub(r"[^a-z0-9\s\-_/]", " ", str(text or "").strip().lower())
    return re.sub(r"\s+", " ", tokenized).strip()
