from __future__ import annotations

from dataclasses import dataclass

from src.agents.contracts import KnowledgeContext, ResourceFocus, TurnIntent
from src.agents.profiles import DOMAIN_PROFILES
from src.agents.routing import compose_route_decision

WRITE_TOOL_NAMES = {
    "upload_to_bisque",
    "add_tags_to_resource",
    "bisque_add_to_dataset",
    "bisque_create_dataset",
    "bisque_add_gobjects",
    "delete_bisque_resource",
    "run_bisque_module",
}
APPROVAL_REQUIRED_TOOL_NAMES: set[str] = set()
READ_TOOL_NAMES = {
    "bisque_find_assets",
    "search_bisque_resources",
    "bisque_advanced_search",
    "load_bisque_resource",
    "bisque_fetch_xml",
    "bisque_download_resource",
    "bisque_download_dataset",
    "bisque_ping",
    "bioio_load_image",
    "stats_list_curated_tools",
}


@dataclass(frozen=True)
class RoutedScientistTurn:
    turn_intent: TurnIntent
    selected_domains: list[str]
    primary_domain: str
    available_tool_names: list[str]
    read_tool_names: list[str]
    analysis_tool_names: list[str]
    write_tool_names: list[str]
    requires_approval: bool
    approval_tool_names: list[str]
    route_reason: str


def _infer_write_tools(user_text: str) -> list[str]:
    normalized = str(user_text or "").strip().lower()
    inferred: list[str] = []
    if "delete" in normalized or "remove" in normalized:
        inferred.append("delete_bisque_resource")
    if "tag" in normalized:
        inferred.append("add_tags_to_resource")
    if "dataset" in normalized and any(token in normalized for token in ("add", "save", "store")):
        inferred.append("bisque_add_to_dataset")
    if "upload" in normalized or "save to bisque" in normalized:
        inferred.append("upload_to_bisque")
    if "module" in normalized and "bisque" in normalized:
        inferred.append("run_bisque_module")
    return inferred


def route_scientist_turn(
    *,
    user_text: str,
    file_ids: list[str],
    resource_uris: list[str],
    dataset_uris: list[str],
    selected_tool_names: list[str],
    knowledge_context: dict[str, object] | None = None,
    selection_context: dict[str, object] | None = None,
    workflow_hint: dict[str, object] | None = None,
) -> RoutedScientistTurn:
    suggested_tool_names = list(selected_tool_names or [])
    selection_context = dict(selection_context or {})
    suggested_tool_names.extend(
        str(name or "").strip()
        for name in selection_context.get("suggested_tool_names", [])
        if str(name or "").strip()
    )
    resource_focus = ResourceFocus(
        focused_file_ids=list(file_ids or []),
        resource_uris=list(resource_uris or []),
        dataset_uris=list(dataset_uris or []),
        suggested_tool_names=suggested_tool_names,
        context_id=str(selection_context.get("context_id") or "").strip() or None,
        source=str(selection_context.get("source") or "").strip() or None,
        originating_message_id=str(selection_context.get("originating_message_id") or "").strip()
        or None,
        originating_user_text=str(selection_context.get("originating_user_text") or "").strip()
        or None,
        suggested_domain=str(selection_context.get("suggested_domain") or "").strip() or None,
    )
    turn_intent = TurnIntent(
        original_user_text=str(user_text or "").strip(),
        normalized_user_text=str(user_text or "").strip().lower(),
        selected_tool_names=[
            str(name or "").strip() for name in selected_tool_names if str(name or "").strip()
        ],
        workflow_hint=dict(workflow_hint or {}),
        resource_focus=resource_focus,
        knowledge_context=KnowledgeContext.model_validate(knowledge_context or {}),
    )
    route = compose_route_decision(turn_intent)
    selected_domains = [
        str(name or "").strip() for name in route.selected_domains if str(name or "").strip()
    ]
    if not selected_domains:
        selected_domains = ["core"]
    primary_domain = selected_domains[0]
    available_tool_names: list[str] = []
    seen: set[str] = set()
    for domain_id in selected_domains:
        profile = DOMAIN_PROFILES.get(domain_id)
        if profile is None:
            continue
        for tool_name in profile.tool_allowlist:
            normalized = str(tool_name or "").strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            available_tool_names.append(normalized)

    explicit_tools = [
        str(name or "").strip() for name in selected_tool_names if str(name or "").strip()
    ]
    inferred_write_tools = _infer_write_tools(user_text)
    if explicit_tools:
        available_tool_names = [
            name for name in available_tool_names if name in set(explicit_tools)
        ] or explicit_tools
    elif inferred_write_tools:
        for tool_name in inferred_write_tools:
            if tool_name not in available_tool_names:
                available_tool_names.append(tool_name)

    write_intent = bool(set(explicit_tools) & WRITE_TOOL_NAMES) or bool(inferred_write_tools)
    read_tool_names = [name for name in available_tool_names if name in READ_TOOL_NAMES]
    write_tool_names = (
        [name for name in available_tool_names if name in WRITE_TOOL_NAMES] if write_intent else []
    )
    analysis_tool_names = [
        name
        for name in available_tool_names
        if name not in set(read_tool_names) and name not in set(write_tool_names)
    ]
    approval_tool_names = [
        name for name in write_tool_names if name in APPROVAL_REQUIRED_TOOL_NAMES
    ]
    requires_approval = bool(approval_tool_names)
    return RoutedScientistTurn(
        turn_intent=turn_intent,
        selected_domains=selected_domains,
        primary_domain=primary_domain,
        available_tool_names=available_tool_names,
        read_tool_names=read_tool_names,
        analysis_tool_names=analysis_tool_names,
        write_tool_names=write_tool_names,
        requires_approval=requires_approval,
        approval_tool_names=approval_tool_names,
        route_reason=str(route.reason or route.route_source or "heuristic"),
    )
