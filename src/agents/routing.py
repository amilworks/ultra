"""Domain routing helpers for the orchestrator agent."""

from __future__ import annotations

import re
from typing import Iterable

from .contracts import RouteDecision, TurnIntent
from .profiles import DOMAIN_PROFILES, DomainId
from .reasoning import is_mcq_prompt


_DOMAIN_KEYWORDS: dict[DomainId, tuple[str, ...]] = {
    "bio": (
        "cell",
        "microscopy",
        "confocal",
        "light-sheet",
        "lightsheet",
        "multiphoton",
        "two-photon",
        "fluorescence",
        "nucleus",
        "organoid",
        "histology",
        "time-lapse",
        "timelapse",
        "electron microscopy",
        "em",
        "biology",
        "protein",
        "gene",
        "tissue",
        "brain",
        "neuron",
        "neuronal",
        "synapse",
        "axon",
        "dendrite",
        "hippocampus",
        "cortex",
        "calcium imaging",
        "virology",
        "viral",
        "virus",
        "sars-cov-2",
        "sars cov 2",
        "sarscov2",
        "retroviral",
        "retrovirus",
        "qpcr",
        "rt-qpcr",
        "rt qpcr",
        "primer",
        "restriction enzyme",
        "promoter",
        "transcription factor",
        "phosphorylation",
        "kinase",
        "genotype",
        "phenotype",
        "pathway",
        "assay",
        "fold change",
        "dilution",
        "copy number",
        "enrichment",
        "localization",
        "colocalization",
        "co-localization",
    ),
    "ecology": (
        "ecology",
        "ecological",
        "wildlife",
        "conservation",
        "habitat",
        "grassland",
        "rangeland",
        "camera trap",
        "aerial survey",
        "species monitoring",
        "occupancy",
        "keystone species",
        "colony",
        "burrow",
        "burrows",
        "prairie dog",
        "prairie dogs",
        "rarespot",
    ),
    "materials": (
        "material",
        "alloy",
        "polymer",
        "grain",
        "grain size",
        "phase fraction",
        "microstructure",
        "fracture",
        "defect",
        "crystal",
        "nanoparticle",
        "orientation",
        "ebsd",
        "dream3d",
        "sem",
        "tem",
    ),
    "medical": (
        "patient",
        "clinical",
        "radiology",
        "ct",
        "mri",
        "x-ray",
        "ultrasound",
        "lesion",
        "tumor",
        "pathology",
        "medical",
        "nifti",
        "nii",
        "nii.gz",
        "fmri",
        "mra",
        "brain volume",
        "structural volume",
        "voxel",
    ),
    "core": (
        "quantum",
        "eigenvector",
        "eigenvalue",
        "operator",
        "hamiltonian",
        "lagrangian",
        "matrix",
        "spin",
        "pauli",
        "proof",
        "derive",
        "theorem",
        "physics",
        "math",
        "mathematical",
        "mechanism",
        "stoichiometry",
        "symmetry",
        "point group",
        "reaction pathway",
        "dominant-negative",
        "transcription factor",
    ),
}

_THEORY_CUES: tuple[str, ...] = (
    "quantum",
    "eigen",
    "operator",
    "hamiltonian",
    "pauli",
    "spin",
    "proof",
    "derive",
    "theorem",
    "mathemat",
    "linear algebra",
)

_VISION_DETECTION_CUES: tuple[str, ...] = (
    "detect",
    "detection",
    "object detection",
    "bounding box",
    "bbox",
    "yolo",
)

_IMAGE_CONTEXT_CUES: tuple[str, ...] = (
    "image",
    "photo",
    "picture",
    "microscopy",
    "volume",
    "stack",
    "scan",
    "uploaded",
    "file",
    "files",
    "this image",
)

_BISQUE_OPERATION_CUES: tuple[str, ...] = (
    "bisque",
    "data_service",
    "client_service/view",
    "upload",
    "download",
    "dataset",
    "dataset uri",
    "resource uri",
    "module",
)

_MICROSCOPY_MODALITY_CUES: tuple[str, ...] = (
    "microscopy",
    "confocal",
    "fluorescence",
    "light-sheet",
    "lightsheet",
    "two-photon",
    "multiphoton",
    "histology",
    "organoid",
    "nucleus",
    "nuclei",
    "cell",
    "cells",
)

_CLINICAL_MODALITY_CUES: tuple[str, ...] = (
    "ct",
    "mri",
    "fmri",
    "mra",
    "x-ray",
    "ultrasound",
    "radiology",
    "clinical",
    "nifti",
    "nii",
    "nii.gz",
    "dicom",
)

_VOLUME_CUES: tuple[str, ...] = (
    "volume",
    "volumetric",
    "3d",
    "stack",
    "slice",
    "voxel",
)

_MATERIALS_MODALITY_CUES: tuple[str, ...] = (
    "ebsd",
    "dream3d",
    "sem",
    "tem",
    "grain",
    "phase fraction",
    "microstructure",
    "alloy",
    "polymer",
    "fracture",
)

_ECOLOGY_PRIORITY_CUES: tuple[str, ...] = (
    "prairie dog",
    "prairie dogs",
    "burrow",
    "burrows",
    "wildlife",
    "habitat",
    "conservation",
    "keystone species",
    "camera trap",
    "aerial survey",
    "rarespot",
)

_TABLE_MODALITY_CUES: tuple[str, ...] = (
    ".h5",
    ".hdf5",
    "hdf5",
    "table",
    "csv",
    "dataset",
)

_DIAGNOSE_CUES: tuple[str, ...] = (
    "abnormal",
    "abnormality",
    "abnormalities",
    "lesion",
    "tumor",
    "mass",
    "hemorrhage",
    "fracture",
    "pathology",
    "diagnos",
    "clinical interpretation",
)

_SEGMENT_CUES: tuple[str, ...] = ("segment", "segmentation", "mask", "outline")
_COUNT_CUES: tuple[str, ...] = ("count", "enumerate", "how many")
_SEARCH_CUES: tuple[str, ...] = ("search", "find", "browse", "look up", "show me")
_LOAD_CUES: tuple[str, ...] = ("load", "open", "inspect", "view", "preview")
_UPLOAD_CUES: tuple[str, ...] = ("upload", "save", "store", "add to dataset", "send to bisque")
_BISQUE_CATALOG_SEARCH_CUES: tuple[str, ...] = (
    "what files",
    "what file",
    "what images",
    "what image",
    "what resources",
    "what resource",
    "what jpg",
    "what jpeg",
    "what png",
    "what tif",
    "what tiff",
    "which files",
    "which file",
    "which images",
    "which image",
    "recently uploaded",
    "recent uploads",
    "latest uploads",
    "most recent",
)
_BISQUE_FILETYPE_CUES: tuple[str, ...] = (
    "jpg",
    "jpeg",
    "png",
    "tif",
    "tiff",
    "bmp",
    "gif",
    "webp",
    "h5",
    "hdf5",
    "csv",
    "tsv",
)


def _score_keyword_hits(normalized_text: str) -> dict[DomainId, float]:
    scores: dict[DomainId, float] = {}
    for domain_id, keywords in _DOMAIN_KEYWORDS.items():
        score = 0.0
        for keyword in keywords:
            if _keyword_match(normalized_text, keyword):
                score += 1.0
        if score > 0:
            scores[domain_id] = score
    return scores


def _compose_intent_text(turn_intent: TurnIntent) -> str:
    parts = [str(turn_intent.original_user_text or "").strip()]
    resolved_context = str(turn_intent.resolved_context_text or "").strip()
    if resolved_context and resolved_context not in parts:
        parts.append(resolved_context)
    resource_focus = turn_intent.resource_focus
    originating = str(resource_focus.originating_user_text or "").strip()
    if originating and originating not in parts:
        parts.append(originating)
    parts.extend(
        str(item).strip()
        for item in resource_focus.resource_uris
        if str(item or "").strip()
    )
    parts.extend(
        str(item).strip()
        for item in resource_focus.dataset_uris
        if str(item or "").strip()
    )
    return "\n".join(part for part in parts if part)


def classify_artifact_modality(turn_intent: TurnIntent) -> tuple[str, list[str]]:
    normalized = _normalize(_compose_intent_text(turn_intent))
    signals: list[str] = []
    if not normalized:
        return "unknown", signals
    has_clinical = any(_keyword_match(normalized, cue) for cue in _CLINICAL_MODALITY_CUES)
    has_volume = any(_keyword_match(normalized, cue) for cue in _VOLUME_CUES)
    has_microscopy = any(_keyword_match(normalized, cue) for cue in _MICROSCOPY_MODALITY_CUES)
    has_materials = any(_keyword_match(normalized, cue) for cue in _MATERIALS_MODALITY_CUES)
    has_table = any(_keyword_match(normalized, cue) for cue in _TABLE_MODALITY_CUES)
    if has_clinical and has_volume:
        signals.extend(["clinical_modality", "volume_modality"])
        return "clinical_volume", signals
    if has_clinical:
        signals.append("clinical_modality")
        return "clinical_image", signals
    if has_microscopy:
        signals.append("microscopy_modality")
        return "microscopy_image", signals
    if has_materials:
        signals.append("materials_modality")
        return "materials_image", signals
    if has_table:
        signals.append("table_modality")
        return "table", signals
    if any(_keyword_match(normalized, cue) for cue in ("dataset", "collection")):
        signals.append("dataset_modality")
        return "dataset", signals
    if any(_keyword_match(normalized, cue) for cue in ("resource", "file", "uploaded")):
        signals.append("resource_modality")
        return "resource", signals
    return "unknown", signals


def classify_operation_intent(turn_intent: TurnIntent) -> tuple[str, list[str]]:
    normalized = _normalize(_compose_intent_text(turn_intent))
    signals: list[str] = []
    workflow_id = str(turn_intent.workflow_hint.get("id") or "").strip().lower()
    requested_tools = {str(name or "").strip().lower() for name in turn_intent.selected_tool_names}
    requested_tools.update(
        str(name or "").strip().lower() for name in turn_intent.resource_focus.suggested_tool_names
    )
    if any(_keyword_match(normalized, cue) for cue in _DIAGNOSE_CUES):
        signals.append("diagnose_cue")
        return "diagnose", signals
    if any(_keyword_match(normalized, cue) for cue in _SEGMENT_CUES) or workflow_id == "segment_sam3":
        signals.append("segment_cue")
        return "segment", signals
    if (
        any(_keyword_match(normalized, cue) for cue in _VISION_DETECTION_CUES)
        or workflow_id in {"detect_prairie_dog", "detect_yolo"}
        or "yolo_detect" in requested_tools
    ):
        signals.append("detect_cue")
        return "detect", signals
    if any(_keyword_match(normalized, cue) for cue in _COUNT_CUES):
        signals.append("count_cue")
        return "count", signals
    if _has_bisque_catalog_search_intent(normalized):
        signals.append("bisque_catalog_search_cue")
        return "search", signals
    if any(_keyword_match(normalized, cue) for cue in _SEARCH_CUES):
        signals.append("search_cue")
        return "search", signals
    if any(_keyword_match(normalized, cue) for cue in _LOAD_CUES):
        signals.append("load_cue")
        return "load", signals
    if any(_keyword_match(normalized, cue) for cue in _UPLOAD_CUES):
        signals.append("upload_cue")
        return "upload", signals
    return "analyze", signals


def compose_route_decision(turn_intent: TurnIntent) -> RouteDecision:
    intent_text = _compose_intent_text(turn_intent)
    normalized = _normalize(intent_text)
    if not normalized:
        return RouteDecision(
            selected_domains=[],
            primary_domain=None,
            confidence=0.0,
            reason="empty",
            route_source="fallback",
        )

    artifact_modality, modality_signals = classify_artifact_modality(turn_intent)
    operation_intent, operation_signals = classify_operation_intent(turn_intent)
    scores = _score_keyword_hits(normalized)
    evidence_signals: list[str] = [*modality_signals, *operation_signals]

    suggested_domain = str(turn_intent.resource_focus.suggested_domain or "").strip().lower()
    if suggested_domain in DOMAIN_PROFILES:
        scores[suggested_domain] = float(scores.get(suggested_domain, 0.0)) + 2.0
        evidence_signals.append("selection_context_domain")

    has_ecology_priority_cues = any(
        _keyword_match(normalized, cue) for cue in _ECOLOGY_PRIORITY_CUES
    )
    has_bisque_operation = _has_bisque_operation_intent(normalized)
    if has_ecology_priority_cues:
        scores["ecology"] = float(scores.get("ecology", 0.0)) + 3.0
        evidence_signals.append("ecology_signal")
    if has_bisque_operation and operation_intent in {"search", "load", "upload"}:
        scores["core"] = float(scores.get("core", 0.0)) + 2.0
        evidence_signals.append("bisque_operation")

    if artifact_modality == "microscopy_image":
        scores["bio"] = float(scores.get("bio", 0.0)) + 3.5
    elif artifact_modality in {"clinical_image", "clinical_volume"}:
        scores["medical"] = float(scores.get("medical", 0.0)) + 4.5
    elif artifact_modality == "materials_image":
        scores["materials"] = float(scores.get("materials", 0.0)) + 3.5
    elif artifact_modality == "table":
        scores["materials"] = float(scores.get("materials", 0.0)) + 1.5

    if operation_intent in {"detect", "segment", "count"} and artifact_modality == "microscopy_image":
        scores["bio"] = float(scores.get("bio", 0.0)) + 1.5
    if operation_intent in {"detect", "count", "analyze"} and has_ecology_priority_cues:
        scores["ecology"] = float(scores.get("ecology", 0.0)) + 1.5
    if operation_intent == "diagnose":
        scores["medical"] = float(scores.get("medical", 0.0)) + 4.0
        evidence_signals.append("diagnostic_intent")
    if operation_intent in {"search", "load", "upload"} and not scores and suggested_domain in DOMAIN_PROFILES:
        scores[suggested_domain] = 1.5

    if has_theory_cues(intent_text):
        scores["core"] = float(scores.get("core", 0.0)) + 1.5
        evidence_signals.append("theory_cue")
    if is_mcq_prompt(intent_text):
        bio_score = float(scores.get("bio", 0.0))
        core_score = float(scores.get("core", 0.0))
        clear_bio_primary = bio_score >= 2.0 and bio_score > core_score and not has_theory_cues(intent_text)
        if not clear_bio_primary:
            scores["core"] = core_score + 0.75
            evidence_signals.append("mcq_backstop")

    ordered = sorted(
        (
            (domain_id, score)
            for domain_id, score in scores.items()
            if float(score) > 0.0
        ),
        key=lambda item: (-float(item[1]), item[0]),
    )
    if not ordered:
        fallback_domains = ["core"] if is_mcq_prompt(intent_text) else []
        return RouteDecision(
            selected_domains=fallback_domains,
            primary_domain=fallback_domains[0] if fallback_domains else None,
            secondary_domains=fallback_domains[1:],
            operation_intent=operation_intent,  # type: ignore[arg-type]
            artifact_modality=artifact_modality,  # type: ignore[arg-type]
            score_by_domain={},
            evidence_signals=evidence_signals,
            confidence=0.45 if fallback_domains else 0.0,
            reason="no_keyword_hits" if not fallback_domains else "mcq_core_backstop",
            route_source="fallback",
            used_model_classifier=False,
        )

    selected = [domain_id for domain_id, _score in ordered]
    top_score = float(ordered[0][1])
    second_score = float(ordered[1][1]) if len(ordered) > 1 else 0.0
    total_score = sum(float(score) for _domain_id, score in ordered)
    margin = max(0.0, top_score - second_score)
    confidence = min(0.97, 0.35 + min(top_score, 6.0) * 0.08 + margin * 0.07)

    if operation_intent == "diagnose" and artifact_modality in {"clinical_image", "clinical_volume"}:
        selected = ["medical", *[domain_id for domain_id in selected if domain_id != "medical"]]
        confidence = max(confidence, 0.82)
        evidence_signals.append("medical_guardrail")
    elif has_ecology_priority_cues and "ecology" in selected:
        selected = ["ecology", *[domain_id for domain_id in selected if domain_id != "ecology"]]
        confidence = max(confidence, 0.8 if operation_intent in {"detect", "count"} else confidence)
    elif artifact_modality == "microscopy_image" and "bio" in selected:
        selected = ["bio", *[domain_id for domain_id in selected if domain_id != "bio"]]
    elif artifact_modality in {"materials_image", "table"} and "materials" in selected:
        selected = ["materials", *[domain_id for domain_id in selected if domain_id != "materials"]]

    if len(selected) > 3:
        selected = selected[:3]
    score_by_domain = {domain_id: round(float(score), 3) for domain_id, score in ordered}
    reason = "keyword_match"
    if "medical_guardrail" in evidence_signals:
        reason = "intent_guardrail"
    elif operation_intent != "analyze":
        reason = f"{operation_intent}_intent"
    if total_score > 0 and margin <= 0.6 and len(selected) > 1:
        confidence = min(confidence, 0.62)
    return RouteDecision(
        selected_domains=selected,
        primary_domain=selected[0],
        secondary_domains=selected[1:],
        operation_intent=operation_intent,  # type: ignore[arg-type]
        artifact_modality=artifact_modality,  # type: ignore[arg-type]
        score_by_domain=score_by_domain,
        evidence_signals=evidence_signals,
        confidence=max(0.0, min(1.0, confidence)),
        reason=reason,
        route_source=("intent_guardrail" if "medical_guardrail" in evidence_signals else "heuristic"),
        used_model_classifier=False,
    )


def _normalize(text: str) -> str:
    tokenized = re.sub(r"[^a-z0-9\s\-_/]", " ", str(text or "").strip().lower())
    return re.sub(r"\s+", " ", tokenized).strip()


def _keyword_match(normalized_text: str, keyword: str) -> bool:
    token = str(keyword or "").strip().lower()
    if not token:
        return False
    if " " in token:
        return token in normalized_text
    return bool(re.search(rf"\b{re.escape(token)}\b", normalized_text))


def detect_domains_heuristic(user_text: str) -> list[DomainId]:
    """Return zero or more domains selected by keyword hits."""

    selected, _confidence, _reason = detect_domains_with_confidence(user_text)
    return selected


def detect_domains_with_confidence(user_text: str) -> tuple[list[DomainId], float, str]:
    """Return selected domains plus heuristic confidence and reason token."""
    route = compose_route_decision(
        TurnIntent(
            original_user_text=str(user_text or "").strip(),
            normalized_user_text=_normalize(user_text),
        )
    )
    return (
        coerce_domain_selection(route.selected_domains),
        float(route.confidence),
        str(route.reason or route.route_source or "keyword_match"),
    )


def has_theory_cues(user_text: str) -> bool:
    """Return True when the prompt appears mathematically/theoretically oriented."""

    normalized = _normalize(user_text)
    if not normalized:
        return False
    return any(cue in normalized for cue in _THEORY_CUES)


def _has_vision_detection_intent(normalized_text: str) -> bool:
    if not normalized_text:
        return False
    has_detection = any(_keyword_match(normalized_text, cue) for cue in _VISION_DETECTION_CUES)
    has_image_context = any(_keyword_match(normalized_text, cue) for cue in _IMAGE_CONTEXT_CUES)
    return bool(has_detection and has_image_context)


def _has_bisque_operation_intent(normalized_text: str) -> bool:
    if not normalized_text:
        return False
    return any(_keyword_match(normalized_text, cue) for cue in _BISQUE_OPERATION_CUES)


def _has_bisque_catalog_search_intent(normalized_text: str) -> bool:
    if not normalized_text or not _has_bisque_operation_intent(normalized_text):
        return False
    if any(_keyword_match(normalized_text, cue) for cue in _SEARCH_CUES):
        return True
    if any(_keyword_match(normalized_text, cue) for cue in _BISQUE_CATALOG_SEARCH_CUES):
        return True
    mentions_filetype = any(_keyword_match(normalized_text, cue) for cue in _BISQUE_FILETYPE_CUES)
    mentions_resource_noun = any(
        _keyword_match(normalized_text, cue)
        for cue in ("file", "files", "image", "images", "resource", "resources", "dataset", "datasets")
    )
    mentions_recency = any(_keyword_match(normalized_text, cue) for cue in ("recent", "latest", "uploaded"))
    starts_like_question = normalized_text.startswith(("what ", "which ", "show me ", "list "))
    return bool(starts_like_question and mentions_resource_noun and (mentions_filetype or mentions_recency))


def parse_domain_csv(value: str | None) -> list[DomainId]:
    """Parse orchestrator CSV output into known domain ids."""

    raw = str(value or "").strip().lower()
    if not raw:
        return []
    alias_map: dict[str, DomainId] = {
        "general": "core",
        "generalist": "core",
        "theory": "core",
        "physics": "core",
        "math": "core",
        "mathematics": "core",
        "wildlife": "ecology",
        "conservation": "ecology",
    }
    seen: set[DomainId] = set()
    selected: list[DomainId] = []
    for token in raw.split(","):
        cleaned = re.sub(r"[^a-z]", "", token)
        cleaned = alias_map.get(cleaned, cleaned)
        if cleaned in DOMAIN_PROFILES and cleaned not in seen:
            selected.append(cleaned)
            seen.add(cleaned)
    return selected


def coerce_domain_selection(candidates: Iterable[DomainId]) -> list[DomainId]:
    """Normalize domain sequence to unique known ids in stable order."""

    seen: set[DomainId] = set()
    out: list[DomainId] = []
    for domain_id in candidates:
        token = str(domain_id or "").strip().lower()
        if token in DOMAIN_PROFILES and token not in seen:
            seen.add(token)
            out.append(token)
    return out
