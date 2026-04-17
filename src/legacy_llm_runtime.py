"""OpenAI/vLLM client management and utilities."""

import json
import re
import time
from collections.abc import Iterator
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from openai import OpenAI, OpenAIError

from src.config import get_settings
from src.logger import logger
from src.tooling.domains import (
    ANALYZE_CSV_TOOL,
    BIOIO_LOAD_IMAGE_TOOL,
    BISQUE_ADD_GOBJECTS_TOOL,
    BISQUE_ADD_TO_DATASET_TOOL,
    BISQUE_ADVANCED_SEARCH_TOOL,
    BISQUE_CREATE_DATASET_TOOL,
    BISQUE_DELETE_TOOL,
    BISQUE_DOWNLOAD_DATASET_TOOL,
    BISQUE_DOWNLOAD_TOOL,
    BISQUE_FETCH_XML_TOOL,
    BISQUE_FIND_ASSETS_TOOL,
    BISQUE_LOAD_TOOL,
    BISQUE_PING_TOOL,
    BISQUE_RUN_MODULE_TOOL,
    BISQUE_SEARCH_TOOL,
    BISQUE_TAG_TOOL,
    BISQUE_UPLOAD_TOOL,
    CODEGEN_PYTHON_PLAN_TOOL,
    COMPARE_CONDITIONS_TOOL,
    DEPTH_PRO_ESTIMATE_TOOL,
    EXECUTE_PYTHON_JOB_TOOL,
    MEGASEG_SEGMENT_TOOL,
    QUANTIFY_OBJECTS_TOOL,
    QUANTIFY_SEGMENTATION_MASKS_TOOL,
    REPRO_REPORT_TOOL,
    SAM2_PROMPT_TOOL,
    SAM2_SEGMENT_TOOL,
    SAM2_VIDEO_TOOL,
    SAM3_SEGMENT_TOOL,
    SEGMENT_EVALUATE_BATCH_TOOL,
    SEGMENTATION_EVAL_TOOL,
    STATS_LIST_CURATED_TOOLS_TOOL,
    STATS_RUN_CURATED_TOOL,
    YOLO_DETECT_TOOL,
    YOLO_FINETUNE_DETECT_TOOL,
    YOLO_LIST_MODELS_TOOL,
)
from src.tooling.engine import ToolEngine
from src.tooling.progress import decode_progress_chunk, encode_progress_chunk
from src.tooling.workpad_orchestrator import upsert_markdown_section
from src.tools import execute_tool_call


def _resolved_provider(settings: object) -> str:
    return str(getattr(settings, "llm_provider", "openai"))


def _resolved_base_url(settings: object) -> str:
    value = getattr(settings, "resolved_llm_base_url", None) or getattr(
        settings, "openai_base_url", "http://localhost:8000/v1"
    )
    return str(value)


def _resolved_model(settings: object) -> str:
    value = (
        getattr(settings, "resolved_llm_model", None)
        or getattr(settings, "llm_model", None)
        or getattr(settings, "openai_model", "gpt-oss-120b")
    )
    return str(value)


def _resolved_api_key(settings: object) -> str | None:
    provider = _resolved_provider(settings)
    if hasattr(settings, "resolved_llm_api_key"):
        value = getattr(settings, "resolved_llm_api_key")
        if value:
            return str(value)
        if provider in {"ollama", "vllm"}:
            return "EMPTY"
        return None
    value = getattr(settings, "llm_api_key", None)
    if value:
        return str(value)
    value = getattr(settings, "openai_api_key", None)
    if value:
        return str(value)
    if provider in {"ollama", "vllm"}:
        return "EMPTY"
    return None


@lru_cache
def get_openai_client() -> OpenAI:
    """
    Get a cached OpenAI client configured for vLLM or OpenAI.

    Returns:
        Configured OpenAI client instance.
    """
    settings = get_settings()
    base_url = _resolved_base_url(settings)
    api_key = _resolved_api_key(settings)
    model_name = _resolved_model(settings)
    provider = _resolved_provider(settings)
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=settings.openai_timeout,
        max_retries=settings.openai_max_retries,
    )

    logger.info(
        "OpenAI-compatible client initialized: provider=%s base_url=%s model=%s",
        provider,
        base_url,
        model_name,
    )
    return client


def test_connection() -> tuple[bool, str]:
    """
    Test the connection to the vLLM/OpenAI API.

    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        settings = get_settings()
        client = get_openai_client()

        logger.info("Testing API connection...")

        response = client.chat.completions.create(
            model=_resolved_model(settings),
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5,
            stream=False,
        )

        if response.choices:
            logger.info("API connection test successful")
            return True, "✅ Connection successful!"
        else:
            logger.warning("API returned empty response")
            return False, "⚠️ API returned empty response"

    except OpenAIError as e:
        error_msg = f"API Error: {str(e)}"
        logger.error(f"Connection test failed: {error_msg}")
        return False, f"❌ {error_msg}"
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(f"Connection test failed: {error_msg}")
        return False, f"❌ {error_msg}"


def _latest_user_message(messages: list[dict[str, str]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user" and msg.get("content"):
            return str(msg.get("content"))
    return ""


def _fallback_title_from_messages(messages: list[dict[str, str]], max_words: int) -> str:
    source = _latest_user_message(messages).strip()
    if not source:
        return "New conversation"
    cleaned = re.sub(r"\s+", " ", source).strip().strip("\"'`")
    words = cleaned.split()
    if not words:
        return "New conversation"
    return " ".join(words[:max_words])


def _sanitize_title_text(raw: str, max_words: int) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    text = text.splitlines()[0].strip()
    text = re.sub(r"^(title)\s*[:\-]\s*", "", text, flags=re.IGNORECASE)
    text = text.strip().strip("\"'`“”‘’")
    text = re.sub(r"\s+", " ", text).strip()
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9_'’\-]*", text)
    if not tokens:
        return ""
    return " ".join(tokens[:max_words]).strip()


def generate_chat_title(
    messages: list[dict[str, str]],
    max_words: int = 4,
) -> tuple[str, str]:
    settings = get_settings()
    if bool(getattr(settings, "llm_mock_mode", False)):
        return _fallback_title_from_messages(messages, max_words=max_words), "fallback"

    client = get_openai_client()
    max_words_clamped = max(2, min(int(max_words or 4), 8))
    fallback = _fallback_title_from_messages(messages, max_words_clamped)

    user_messages = [
        str(item.get("content") or "").strip()
        for item in messages
        if str(item.get("role") or "") == "user" and str(item.get("content") or "").strip()
    ]
    if not user_messages:
        return fallback, "fallback"

    context_block = "\n".join(f"- {entry}" for entry in user_messages[-6:])
    title_prompt = [
        {
            "role": "system",
            "content": (
                "Generate a short conversation title.\n"
                f"Rules: max {max_words_clamped} words, no quotes, no trailing punctuation, title only."
            ),
        },
        {
            "role": "user",
            "content": f"Conversation snippets:\n{context_block}",
        },
    ]

    request_kwargs = {
        "model": _resolved_model(settings),
        "messages": title_prompt,
        "max_tokens": max(8, max_words_clamped * 5),
        "temperature": 0.1,
        "stream": False,
    }

    raw_title = ""
    try:
        completion = client.chat.completions.create(
            **request_kwargs,
            extra_body={"reasoning_effort": "low"},
        )
    except Exception:
        completion = client.chat.completions.create(**request_kwargs)

    try:
        if completion.choices and completion.choices[0].message:
            raw_title = str(completion.choices[0].message.content or "")
    except Exception:
        raw_title = ""

    normalized_title = _sanitize_title_text(raw_title, max_words_clamped)
    if normalized_title:
        return normalized_title, "llm"
    return fallback, "fallback"


def _tool_map(tools: list[dict]) -> dict[str, dict]:
    mapped: dict[str, dict] = {}
    for tool in tools:
        try:
            name = str(tool["function"]["name"])
        except Exception:
            continue
        mapped[name] = tool
    return mapped


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    haystack = str(text or "").lower()
    return any(needle in haystack for needle in needles)


def _needs_extended_next_steps(user_text: str) -> bool:
    text = str(user_text or "").lower()
    return _contains_any(
        text,
        (
            "final reflection",
            "top ",
            "weak point",
            "implementation priorit",
            "prioritize",
            "product-level reflection",
            "structured contract json",
            "based on this entire session",
            "from user experience perspective",
            "propose what should be automated",
        ),
    )


def _explicit_tool_mentions(user_text: str, tool_names: set[str]) -> set[str]:
    text = str(user_text or "").lower()
    return {name for name in tool_names if str(name).lower() in text}


def _is_no_tool_request(user_text: str) -> bool:
    text = str(user_text or "").lower()
    return _contains_any(
        text,
        (
            "do not call tools",
            "without calling tools",
            "without tools",
            "no tools",
        ),
    )


def _normalize_response_verbosity(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"concise", "balanced", "detailed"}:
        return normalized
    return "detailed"


def _resolve_response_verbosity(*, user_text: str, default_level: str) -> str:
    text = str(user_text or "").lower()
    if _contains_any(
        text,
        (
            "brief",
            "short answer",
            "keep it short",
            "concise",
            "tl;dr",
            "tldr",
        ),
    ):
        return "concise"
    if _contains_any(
        text,
        (
            "detailed",
            "comprehensive",
            "well thought out",
            "in depth",
            "in-depth",
            "thorough",
        ),
    ):
        return "detailed"
    return _normalize_response_verbosity(default_level)


def _response_verbosity_rule(level: str) -> str:
    if level == "concise":
        return (
            "Response depth: keep concise and direct while preserving required scientific "
            "contract fields."
        )
    if level == "balanced":
        return (
            "Response depth: use moderate detail with key methods, quantitative findings, "
            "and actionable next steps."
        )
    return (
        "Response depth: provide well-thought-out, detailed scientific explanations "
        "including approach, quantitative interpretation, limitations, and concrete next steps."
    )


def _normalize_user_response_text(value: str | None) -> str:
    text = str(value or "")
    if not text:
        return ""
    if "\\n" in text and "\n" not in text:
        text = text.replace("\\n", "\n")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _response_sentences(text: str) -> list[str]:
    normalized = _normalize_user_response_text(text)
    if not normalized:
        return []
    split_chunks = re.split(r"(?<=[.!?])\s+|\n+", normalized)
    sentences: list[str] = []
    for chunk in split_chunks:
        token = str(chunk or "").strip(" -•\t")
        if token:
            sentences.append(token)
    return sentences


def _is_meta_narration_sentence(sentence: str) -> bool:
    lower = str(sentence or "").strip().lower()
    if not lower:
        return False
    if not re.match(
        r"^(provided|presented|outlined|listed|summarized|reported|generated|executed|completed)\b",
        lower,
    ):
        return False
    marker_hits = 0
    for marker in (
        "comprehensive",
        "set of",
        "taxonomy",
        "covering",
        "recommended",
        "next-step",
        "next step",
        "pipeline",
    ):
        if marker in lower:
            marker_hits += 1
    return marker_hits >= 1


def _meta_narration_rate(text: str) -> float:
    sentences = _response_sentences(text)
    if not sentences:
        return 1.0
    meta_count = sum(1 for sentence in sentences if _is_meta_narration_sentence(sentence))
    return round(float(meta_count) / float(len(sentences)), 3)


def _numeric_detail_density(text: str) -> float:
    normalized = _normalize_user_response_text(text)
    if not normalized:
        return 0.0
    word_count = len(re.findall(r"\b[\w'-]+\b", normalized))
    if word_count <= 0:
        return 0.0
    numeric_hits = len(
        re.findall(r"\b\d+(?:\.\d+)?(?:e[+-]?\d+)?\b", normalized, flags=re.IGNORECASE)
    )
    return round((float(numeric_hits) * 100.0) / float(max(word_count, 1)), 3)


def _contains_action_verbs(text: str) -> bool:
    lower = str(text or "").lower()
    if not lower:
        return False
    return bool(
        re.search(
            r"\b(run|validate|compare|quantify|measure|collect|export|reproduce|inspect|tune|test|compute)\b",
            lower,
        )
    )


def _is_tool_progress_event(event: dict[str, Any]) -> bool:
    if not isinstance(event, dict):
        return False
    token = str(event.get("event") or "").strip().lower()
    if token not in {"started", "completed", "error", "heartbeat"}:
        return False
    return bool(str(event.get("tool") or "").strip())


def _expects_numeric_detail(
    *,
    user_text: str,
    contract_payload: dict[str, Any] | None,
    progress_events: list[dict[str, Any]] | None,
) -> bool:
    text = str(user_text or "").lower()
    if _contains_any(
        text,
        (
            "quantify",
            "metric",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "dice",
            "iou",
            "pca",
            "variance",
            "random forest",
            "statistics",
            "statistical",
        ),
    ):
        return True
    if isinstance(contract_payload, dict):
        measurements = contract_payload.get("measurements")
        if isinstance(measurements, list) and measurements:
            return True
    if isinstance(progress_events, list):
        for event in progress_events:
            if not isinstance(event, dict):
                continue
            summary = event.get("summary")
            if isinstance(summary, dict):
                metrics = summary.get("measurements")
                if isinstance(metrics, list) and metrics:
                    return True
    return False


def _answer_completeness_score(
    *,
    response_text: str,
    user_text: str,
    contract_payload: dict[str, Any] | None,
    progress_events: list[dict[str, Any]] | None,
) -> tuple[float, dict[str, bool]]:
    normalized = _normalize_user_response_text(response_text)
    lower = normalized.lower()
    words = re.findall(r"\b[\w'-]+\b", normalized)
    word_count = len(words)
    sentence_count = len(_response_sentences(normalized))
    tool_turn = any(_is_tool_progress_event(item) for item in (progress_events or []))
    expected_numeric = _expects_numeric_detail(
        user_text=user_text,
        contract_payload=contract_payload,
        progress_events=progress_events,
    )
    contract_measurements = 0
    contract_limitations = 0
    contract_next_steps = 0
    if isinstance(contract_payload, dict):
        measurements = contract_payload.get("measurements")
        if isinstance(measurements, list):
            contract_measurements = len(measurements)
        limitations = contract_payload.get("limitations")
        if isinstance(limitations, list):
            contract_limitations = len([item for item in limitations if str(item or "").strip()])
        next_steps = contract_payload.get("next_steps")
        if isinstance(next_steps, list):
            contract_next_steps = len(next_steps)

    checks: dict[str, bool] = {
        "direct_answer": bool(normalized)
        and word_count >= 32
        and _meta_narration_rate(normalized) < 0.5,
        "structured_detail": sentence_count >= 3 or ("\n-" in normalized) or ("\n1." in normalized),
        "limitations_covered": (
            ("limitation" in lower)
            or ("caveat" in lower)
            or ("uncertain" in lower)
            or contract_limitations > 0
        ),
        "next_steps_present": (
            ("next step" in lower)
            or ("recommend" in lower)
            or _contains_action_verbs(normalized)
            or contract_next_steps > 0
        ),
        "tool_status_covered": (
            (not tool_turn)
            or any(
                token in lower for token in ("completed", "partial", "failed", "succeeded", "error")
            )
        ),
        "numeric_detail_present": (
            (not expected_numeric)
            or _numeric_detail_density(normalized) >= 0.5
            or contract_measurements > 0
        ),
    }
    score = round(
        sum(1.0 for ok in checks.values() if ok) / float(max(len(checks), 1)),
        3,
    )
    return score, checks


def _compute_response_quality_metrics(
    *,
    response_text: str,
    user_text: str,
    contract_payload: dict[str, Any] | None,
    progress_events: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    normalized = _normalize_user_response_text(response_text)
    word_count = len(re.findall(r"\b[\w'-]+\b", normalized))
    meta_rate = _meta_narration_rate(normalized)
    completeness, completeness_checks = _answer_completeness_score(
        response_text=normalized,
        user_text=user_text,
        contract_payload=contract_payload,
        progress_events=progress_events,
    )
    numeric_density = _numeric_detail_density(normalized)
    return {
        "meta_narration_rate": meta_rate,
        "answer_completeness": completeness,
        "numeric_detail_density": numeric_density,
        "word_count": word_count,
        "completeness_checks": completeness_checks,
        "expects_numeric_detail": _expects_numeric_detail(
            user_text=user_text,
            contract_payload=contract_payload,
            progress_events=progress_events,
        ),
    }


def _estimate_phase_h_output_budget(
    *,
    user_text: str,
    scratchpad_text: str,
    fallback_response: str,
    verbosity_level: str,
    min_tokens: int,
    max_tokens: int,
) -> tuple[int, dict[str, Any]]:
    prompt_word_count = len(re.findall(r"\b[\w'-]+\b", str(user_text or "")))
    scratchpad_word_count = len(re.findall(r"\b[\w'-]+\b", str(scratchpad_text or "")))
    fallback_word_count = len(re.findall(r"\b[\w'-]+\b", str(fallback_response or "")))
    tool_section_count = len(
        re.findall(r"^##\s+Tool:\s+", str(scratchpad_text or ""), flags=re.MULTILINE)
    )
    numeric_tokens = len(
        re.findall(
            r"\b\d+(?:\.\d+)?(?:e[+-]?\d+)?\b", str(scratchpad_text or ""), flags=re.IGNORECASE
        )
    )

    complexity_score = (
        0.30 * min(float(prompt_word_count) / 220.0, 1.0)
        + 0.30 * min(float(scratchpad_word_count) / 2200.0, 1.0)
        + 0.20 * min(float(tool_section_count) / 7.0, 1.0)
        + 0.10 * min(float(numeric_tokens) / 80.0, 1.0)
        + 0.10 * min(float(fallback_word_count) / 350.0, 1.0)
    )
    if complexity_score < 0.34:
        tier = "low"
        base_budget = 1900
    elif complexity_score < 0.67:
        tier = "medium"
        base_budget = 3200
    else:
        tier = "high"
        base_budget = 5200

    verbosity_adjust = {"concise": -700, "balanced": -200, "detailed": 650}
    adjusted = base_budget + int(verbosity_adjust.get(str(verbosity_level), 0))
    clamped = max(int(min_tokens), min(int(max_tokens), adjusted))
    trace = {
        "tier": tier,
        "complexity_score": round(complexity_score, 3),
        "prompt_words": int(prompt_word_count),
        "scratchpad_words": int(scratchpad_word_count),
        "fallback_words": int(fallback_word_count),
        "tool_sections": int(tool_section_count),
        "numeric_tokens": int(numeric_tokens),
        "selected_max_tokens": int(clamped),
    }
    return int(clamped), trace


def _should_run_quality_repair(
    *,
    quality_metrics: dict[str, Any],
    max_meta_rate: float,
    min_completeness: float,
    min_numeric_density: float,
) -> bool:
    meta_rate = float(quality_metrics.get("meta_narration_rate") or 0.0)
    completeness = float(quality_metrics.get("answer_completeness") or 0.0)
    numeric_density = float(quality_metrics.get("numeric_detail_density") or 0.0)
    expects_numeric = bool(quality_metrics.get("expects_numeric_detail"))
    if meta_rate > float(max_meta_rate):
        return True
    if completeness < float(min_completeness):
        return True
    if expects_numeric and numeric_density < float(min_numeric_density):
        return True
    return False


def _quality_metrics_improved(
    *,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> bool:
    baseline_meta = float(baseline.get("meta_narration_rate") or 0.0)
    candidate_meta = float(candidate.get("meta_narration_rate") or 0.0)
    baseline_completeness = float(baseline.get("answer_completeness") or 0.0)
    candidate_completeness = float(candidate.get("answer_completeness") or 0.0)
    baseline_numeric = float(baseline.get("numeric_detail_density") or 0.0)
    candidate_numeric = float(candidate.get("numeric_detail_density") or 0.0)

    if candidate_meta + 0.05 < baseline_meta:
        return True
    if candidate_completeness > baseline_completeness + 0.08:
        return True
    if candidate_numeric > baseline_numeric + 0.45:
        return True
    return False


def _rewrite_low_quality_phase_h_response(
    *,
    client: Any,
    model: str,
    user_prompt: str,
    scratchpad_text: str,
    draft_response: str,
    contract_payload: dict[str, Any] | None,
    quality_metrics: dict[str, Any],
    max_tokens: int,
    verbosity_level: str,
) -> str:
    contract_block = ""
    if isinstance(contract_payload, dict) and contract_payload:
        contract_block = json.dumps(contract_payload, ensure_ascii=False, indent=2)
    depth_rule = _response_verbosity_rule(_normalize_response_verbosity(verbosity_level))
    request_kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Rewrite the scientific answer so it is direct, detailed, and user-facing.\n"
                    "Rules:\n"
                    "- Do not use meta narration prefixes like 'Provided...', 'Listed...', 'Outlined...'.\n"
                    "- Keep factual consistency with tool evidence and contract payload.\n"
                    "- Include quantitative details when available.\n"
                    "- Include limitations and next steps only when they materially help the user act on the answer.\n"
                    "- For simple conceptual or conversational turns, prefer plain paragraphs over faux-report framing.\n"
                    "- If the UI already shows figures, tables, or tool cards, keep the prose complementary instead of repeating them.\n"
                    "- Never mention scratchpad, internal IDs, or private local paths.\n"
                    "- Use markdown structure only when it improves readability.\n"
                    f"- {depth_rule}\n"
                    "- Return plain text only."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Original prompt:\n{user_prompt or '(empty)'}\n\n"
                    f"Current draft response:\n{draft_response or '(empty)'}\n\n"
                    f"Quality metrics:\n{json.dumps(quality_metrics, ensure_ascii=False)}\n\n"
                    f"Contract context:\n{contract_block or '(none)'}\n\n"
                    f"Scratchpad context:\n{scratchpad_text or '(none)'}"
                ),
            },
        ],
        "max_tokens": max(700, min(int(max_tokens), 2600)),
        "temperature": 0.1,
        "stream": False,
    }
    raw_text = ""
    try:
        completion = client.chat.completions.create(
            **request_kwargs,
            extra_body={"reasoning_effort": "high"},
        )
    except Exception:
        completion = client.chat.completions.create(**request_kwargs)
    try:
        if completion.choices and completion.choices[0].message:
            raw_text = str(completion.choices[0].message.content or "")
    except Exception:
        raw_text = ""
    return _normalize_user_response_text(raw_text)


def _extract_json_object_from_text(raw_text: str) -> dict[str, Any] | None:
    text = str(raw_text or "").strip()
    if not text:
        return None
    candidates: list[str] = [text]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        candidates.insert(0, str(fenced.group(1)).strip())
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        token = str(candidate or "").strip()
        if not token:
            continue
        try:
            parsed = json.loads(token)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _extract_latest_scratchpad_draft_text(scratchpad_text: str) -> str:
    text = str(scratchpad_text or "")
    if not text:
        return ""
    sections = re.findall(
        r"(?ms)^##\s+Draft\s+\(iteration\s+\d+\)\n(.*?)(?=^##\s+|\Z)",
        text,
    )
    if not sections:
        sections = re.findall(
            r"(?ms)^##\s+Draft\s+\(auto\)\n(.*?)(?=^##\s+|\Z)",
            text,
        )
    if not sections:
        sections = re.findall(
            r"(?ms)^##\s+Final Answer Draft \(high effort\)\n(.*?)(?=^##\s+|\Z)",
            text,
        )
    if not sections:
        return ""
    candidate = _normalize_user_response_text(str(sections[-1] or ""))
    if not candidate:
        return ""
    payload = _extract_json_object_from_text(candidate)
    if isinstance(payload, dict):
        result = _normalize_user_response_text(str(payload.get("result") or ""))
        if result:
            return result
        final_response = _normalize_user_response_text(str(payload.get("final_response") or ""))
        if final_response:
            return final_response
    fuzzy_result = _extract_result_from_contract_like_text(candidate)
    if fuzzy_result:
        return fuzzy_result
    return candidate


def _extract_result_from_contract_like_text(raw_text: str) -> str:
    payload = _extract_json_object_from_text(raw_text)
    if isinstance(payload, dict):
        result = _normalize_user_response_text(str(payload.get("result") or ""))
        if result:
            return result
        final_response = _normalize_user_response_text(str(payload.get("final_response") or ""))
        if final_response:
            return final_response
    match = re.search(
        r'"result"\s*:\s*"(?P<result>(?:\\.|[^"\\])*)"',
        str(raw_text or ""),
        flags=re.DOTALL,
    )
    if not match:
        return ""
    token = str(match.group("result") or "")
    token = token.replace('\\"', '"')
    token = token.replace("\\n", "\n").replace("\\t", "\t")
    token = token.replace("\\\\", "\\")
    return _normalize_user_response_text(token)


def _normalize_reframe_list(value: Any, *, max_items: int = 6) -> list[str]:
    values: list[Any]
    if isinstance(value, list):
        values = value
    elif isinstance(value, str):
        values = [value]
    else:
        values = []
    normalized: list[str] = []
    for item in values:
        text = re.sub(r"\s+", " ", str(item or "")).strip().strip("-•")
        if not text:
            continue
        normalized.append(text)
        if len(normalized) >= max_items:
            break
    return normalized


def _coerce_prompt_reframe_payload(
    payload: dict[str, Any] | None,
    *,
    fallback_prompt: str,
) -> dict[str, Any]:
    normalized_prompt = re.sub(r"\s+", " ", str(fallback_prompt or "")).strip() or "User request"
    source = payload if isinstance(payload, dict) else {}
    reframed_prompt = re.sub(
        r"\s+",
        " ",
        str(source.get("reframed_prompt") or source.get("reframe") or "").strip(),
    ).strip()
    if not reframed_prompt:
        reframed_prompt = normalized_prompt

    assumptions = _normalize_reframe_list(source.get("assumptions"))
    missing_information = _normalize_reframe_list(source.get("missing_information"))
    success_criteria = _normalize_reframe_list(source.get("success_criteria"))
    if not success_criteria:
        success_criteria = [
            "Answer the user request directly with scientifically actionable detail.",
            "State uncertainty or missing information explicitly when it affects conclusions.",
        ]
    return {
        "reframed_prompt": reframed_prompt,
        "assumptions": assumptions,
        "missing_information": missing_information,
        "success_criteria": success_criteria,
    }


def _format_prompt_reframe_markdown(payload: dict[str, Any]) -> str:
    reframed_prompt = str(payload.get("reframed_prompt") or "").strip() or "(empty)"
    assumptions = payload.get("assumptions") if isinstance(payload.get("assumptions"), list) else []
    missing_information = (
        payload.get("missing_information")
        if isinstance(payload.get("missing_information"), list)
        else []
    )
    success_criteria = (
        payload.get("success_criteria") if isinstance(payload.get("success_criteria"), list) else []
    )
    lines: list[str] = [
        "### Reframed Prompt",
        reframed_prompt,
        "",
        "### Assumptions",
    ]
    if assumptions:
        lines.extend(f"- {str(item)}" for item in assumptions)
    else:
        lines.append("- None.")
    lines.extend(["", "### Missing Information"])
    if missing_information:
        lines.extend(f"- {str(item)}" for item in missing_information)
    else:
        lines.append("- None identified.")
    lines.extend(["", "### Success Criteria"])
    lines.extend(f"- {str(item)}" for item in success_criteria)
    return "\n".join(lines).strip()


def _generate_prompt_reframe_payload(
    *,
    client: Any,
    model: str,
    messages: list[dict[str, str]],
) -> dict[str, Any]:
    latest_prompt = _latest_user_message(messages)
    context_rows: list[str] = []
    for item in messages[-10:]:
        role = str(item.get("role") or "").strip().lower()
        if role not in {"user", "assistant"}:
            continue
        content = re.sub(r"\s+", " ", str(item.get("content") or "")).strip()
        if not content:
            continue
        if len(content) > 280:
            content = content[:277].rstrip() + "..."
        context_rows.append(f"- {role}: {content}")
    context_block = "\n".join(context_rows[-8:]) if context_rows else "- (none)"

    phase_messages = [
        {
            "role": "system",
            "content": (
                "Reframe the scientific user request for execution planning.\n"
                "Return exactly one strict JSON object with keys:\n"
                '{\n  "reframed_prompt": string,\n  "assumptions": [string,...],\n'
                '  "missing_information": [string,...],\n  "success_criteria": [string,...]\n}\n'
                "Rules: be faithful to user intent, do not invent facts, keep assumptions minimal."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Current user prompt:\n{latest_prompt or '(empty)'}\n\n"
                f"Recent dialogue context:\n{context_block}"
            ),
        },
    ]
    request_kwargs: dict[str, Any] = {
        "model": model,
        "messages": phase_messages,
        "max_tokens": 500,
        "temperature": 0.1,
        "stream": False,
    }
    raw_text = ""
    try:
        completion = client.chat.completions.create(
            **request_kwargs,
            extra_body={"reasoning_effort": "low"},
        )
    except Exception:
        completion = client.chat.completions.create(**request_kwargs)
    try:
        if completion.choices and completion.choices[0].message:
            raw_text = str(completion.choices[0].message.content or "")
    except Exception:
        raw_text = ""
    payload = _extract_json_object_from_text(raw_text)
    return _coerce_prompt_reframe_payload(payload, fallback_prompt=latest_prompt)


def _normalize_plan_steps(
    value: Any,
    *,
    available_tool_names: set[str],
    max_items: int = 8,
) -> list[dict[str, str]]:
    items = value if isinstance(value, list) else []
    normalized: list[dict[str, str]] = []
    allowed_special = {"none", "no_tool", "none_required"}
    for item in items:
        tool = ""
        rationale = ""
        stopping_criteria = ""
        if isinstance(item, dict):
            tool = str(item.get("tool") or item.get("name") or "").strip()
            rationale = str(item.get("rationale") or item.get("why") or "").strip()
            stopping_criteria = str(
                item.get("stopping_criteria") or item.get("stop_when") or ""
            ).strip()
        elif isinstance(item, str):
            token = re.sub(r"\s+", " ", item).strip()
            if token:
                tool = token
        if not tool:
            continue
        normalized_tool = tool.lower()
        if normalized_tool not in available_tool_names and normalized_tool not in allowed_special:
            continue
        if not rationale:
            rationale = (
                "Directly addresses the user request while minimizing unnecessary tool calls."
            )
        if not stopping_criteria:
            stopping_criteria = "Stop when required evidence and outputs are obtained."
        normalized.append(
            {
                "tool": normalized_tool,
                "rationale": rationale,
                "stopping_criteria": stopping_criteria,
            }
        )
        if len(normalized) >= max_items:
            break
    return normalized


def _normalize_plan_list(value: Any, *, max_items: int = 8) -> list[str]:
    values = value if isinstance(value, list) else ([] if value is None else [value])
    normalized: list[str] = []
    for item in values:
        text = re.sub(r"\s+", " ", str(item or "")).strip().strip("-•")
        if not text:
            continue
        normalized.append(text)
        if len(normalized) >= max_items:
            break
    return normalized


def _coerce_workpad_plan_payload(
    payload: dict[str, Any] | None,
    *,
    user_text: str,
    available_tool_names: list[str],
    uploaded_file_count: int,
) -> dict[str, Any]:
    source = payload if isinstance(payload, dict) else {}
    available_set = {
        str(name or "").strip().lower() for name in available_tool_names if str(name or "").strip()
    }
    no_tool_requested = _is_no_tool_request(user_text)

    tool_plan = _normalize_plan_steps(
        source.get("tool_plan"),
        available_tool_names=available_set,
    )
    if no_tool_requested:
        tool_plan = [
            {
                "tool": "none",
                "rationale": "User explicitly asked to avoid tool calls.",
                "stopping_criteria": "Answer directly without tools unless user asks to enable tools.",
            }
        ]
    elif not tool_plan:
        if available_tool_names:
            first_tool = str(available_tool_names[0]).strip().lower()
            tool_plan = [
                {
                    "tool": first_tool,
                    "rationale": "Start with the highest-priority relevant tool and keep calls minimal.",
                    "stopping_criteria": "Stop when required quantitative evidence is collected.",
                }
            ]
        else:
            tool_plan = [
                {
                    "tool": "none",
                    "rationale": "No eligible tools are available for this prompt.",
                    "stopping_criteria": "Provide a direct scientific answer and note any limits.",
                }
            ]

    context_bolstering_plan = _normalize_plan_list(source.get("context_bolstering_plan"))
    if not context_bolstering_plan:
        context_bolstering_plan = [
            "Use uploaded files and system context first; avoid redundant asset searches.",
            "Do not invent missing metadata; mark unknown fields explicitly.",
        ]
        if uploaded_file_count <= 0:
            context_bolstering_plan[0] = (
                "Use conversation history and prior run context before requesting additional inputs."
            )

    answer_blueprint = _normalize_plan_list(source.get("answer_blueprint"))
    if not answer_blueprint:
        answer_blueprint = [
            "Direct answer to the user objective.",
            "Evidence and artifacts produced (if any).",
            "Quantitative measurements and interpretation.",
            "Confidence, QC warnings, and limitations.",
            "Concrete next steps.",
        ]

    fallback_behavior = _normalize_plan_list(source.get("fallback_behavior"))
    if not fallback_behavior:
        fallback_behavior = [
            "If a required tool fails, report partial completion and specific limitations.",
            "Do not expose internal IDs, scratchpad details, or private filesystem paths.",
        ]
        if no_tool_requested:
            fallback_behavior.insert(
                0,
                "Remain on a no-tool path unless the user explicitly changes instructions.",
            )

    return {
        "tool_plan": tool_plan,
        "context_bolstering_plan": context_bolstering_plan,
        "answer_blueprint": answer_blueprint,
        "fallback_behavior": fallback_behavior,
    }


def _format_workpad_plan_markdown(payload: dict[str, Any]) -> str:
    tool_plan = payload.get("tool_plan") if isinstance(payload.get("tool_plan"), list) else []
    context_bolstering_plan = (
        payload.get("context_bolstering_plan")
        if isinstance(payload.get("context_bolstering_plan"), list)
        else []
    )
    answer_blueprint = (
        payload.get("answer_blueprint") if isinstance(payload.get("answer_blueprint"), list) else []
    )
    fallback_behavior = (
        payload.get("fallback_behavior")
        if isinstance(payload.get("fallback_behavior"), list)
        else []
    )

    lines: list[str] = ["### Tool Plan"]
    if tool_plan:
        for idx, row in enumerate(tool_plan, start=1):
            if not isinstance(row, dict):
                continue
            tool_name = str(row.get("tool") or "none").strip()
            rationale = str(row.get("rationale") or "").strip()
            stop_when = str(row.get("stopping_criteria") or "").strip()
            lines.append(f"{idx}. `{tool_name}` — {rationale}")
            lines.append(f"   Stop when: {stop_when}")
    else:
        lines.append("1. `none` — No tool calls required.")
        lines.append("   Stop when: Direct answer is complete.")

    lines.extend(["", "### Context Bolstering Plan"])
    lines.extend(f"- {str(item)}" for item in context_bolstering_plan)
    lines.extend(["", "### Answer Blueprint"])
    lines.extend(f"- {str(item)}" for item in answer_blueprint)
    lines.extend(["", "### Fallback Behavior"])
    lines.extend(f"- {str(item)}" for item in fallback_behavior)
    return "\n".join(lines).strip()


def _planner_system_directive(payload: dict[str, Any]) -> str:
    tool_plan = payload.get("tool_plan") if isinstance(payload.get("tool_plan"), list) else []
    ordered_tools = [
        str(item.get("tool") or "").strip()
        for item in tool_plan
        if isinstance(item, dict) and str(item.get("tool") or "").strip()
    ]
    fallback_behavior = (
        payload.get("fallback_behavior")
        if isinstance(payload.get("fallback_behavior"), list)
        else []
    )
    answer_blueprint = (
        payload.get("answer_blueprint") if isinstance(payload.get("answer_blueprint"), list) else []
    )
    lines = [
        "Planner directive (medium effort):",
        "- Follow this tool plan unless new evidence requires deviation.",
        "- Keep tool calls minimal and skip redundant or duplicate calls.",
    ]
    if ordered_tools:
        lines.append("- Planned tool order: " + " -> ".join(ordered_tools))
    if fallback_behavior:
        lines.append("- Fallback policy: " + "; ".join(str(item) for item in fallback_behavior[:3]))
    if answer_blueprint:
        lines.append(
            "- Target response structure: " + "; ".join(str(item) for item in answer_blueprint[:5])
        )
    return "\n".join(lines)


def _generate_workpad_plan_payload(
    *,
    client: Any,
    model: str,
    user_text: str,
    reframe_payload: dict[str, Any] | None,
    available_tool_names: list[str],
    dynamic_rules: list[str],
    uploaded_file_count: int,
) -> dict[str, Any]:
    reframe_prompt = ""
    if isinstance(reframe_payload, dict):
        reframe_prompt = str(reframe_payload.get("reframed_prompt") or "").strip()
    rules_block = (
        "\n".join(f"- {rule}" for rule in dynamic_rules[:12]) if dynamic_rules else "- none"
    )
    available_tools_block = ", ".join(available_tool_names) if available_tool_names else "none"
    planner_messages = [
        {
            "role": "system",
            "content": (
                "Create a medium-effort execution plan for this scientific prompt.\n"
                "Return exactly one strict JSON object with keys:\n"
                "{\n"
                '  "tool_plan": [{"tool": string, "rationale": string, "stopping_criteria": string}],\n'
                '  "context_bolstering_plan": [string,...],\n'
                '  "answer_blueprint": [string,...],\n'
                '  "fallback_behavior": [string,...]\n'
                "}\n"
                "Rules: minimize tool usage, prefer no-tool path when appropriate, and include explicit failure handling."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Original prompt:\n{user_text or '(empty)'}\n\n"
                f"Reframed prompt:\n{reframe_prompt or '(not available)'}\n\n"
                f"Available tools:\n{available_tools_block}\n\n"
                f"Uploaded file count: {int(uploaded_file_count)}\n\n"
                f"Turn directives:\n{rules_block}"
            ),
        },
    ]
    request_kwargs: dict[str, Any] = {
        "model": model,
        "messages": planner_messages,
        "max_tokens": 900,
        "temperature": 0.1,
        "stream": False,
    }
    raw_text = ""
    try:
        completion = client.chat.completions.create(
            **request_kwargs,
            extra_body={"reasoning_effort": "medium"},
        )
    except Exception:
        completion = client.chat.completions.create(**request_kwargs)
    try:
        if completion.choices and completion.choices[0].message:
            raw_text = str(completion.choices[0].message.content or "")
    except Exception:
        raw_text = ""
    payload = _extract_json_object_from_text(raw_text)
    return _coerce_workpad_plan_payload(
        payload,
        user_text=user_text,
        available_tool_names=available_tool_names,
        uploaded_file_count=uploaded_file_count,
    )


def _coerce_phase_h_contract_payload(
    payload: dict[str, Any] | None,
    *,
    final_response: str,
) -> dict[str, Any]:
    source = payload if isinstance(payload, dict) else {}
    evidence_raw = source.get("evidence")
    evidence = evidence_raw if isinstance(evidence_raw, list) else []
    measurements_raw = source.get("measurements")
    measurements = measurements_raw if isinstance(measurements_raw, list) else []
    stat_raw = source.get("statistical_analysis")
    statistical_analysis = stat_raw if isinstance(stat_raw, list) else []
    qc_raw = source.get("qc_warnings")
    qc_warnings = _normalize_plan_list(qc_raw, max_items=10)
    limitations_raw = source.get("limitations")
    limitations = _normalize_plan_list(limitations_raw, max_items=10)
    next_steps_raw = source.get("next_steps")
    next_steps: list[dict[str, str]] = []
    if isinstance(next_steps_raw, list):
        for item in next_steps_raw:
            if isinstance(item, dict):
                action = re.sub(r"\s+", " ", str(item.get("action") or "")).strip()
            else:
                action = re.sub(r"\s+", " ", str(item or "")).strip()
            if action:
                next_steps.append({"action": action})
            if len(next_steps) >= 8:
                break
    if not next_steps:
        next_steps = [
            {"action": "Continue with the next highest-impact scientific validation step."}
        ]

    confidence_raw = source.get("confidence")
    confidence_level = "medium"
    confidence_why = ["Based on tool outputs and available run evidence."]
    if isinstance(confidence_raw, dict):
        level = str(confidence_raw.get("level") or "").strip().lower()
        if level in {"low", "medium", "high"}:
            confidence_level = level
        why = _normalize_plan_list(confidence_raw.get("why"), max_items=8)
        if why:
            confidence_why = why
    if not limitations:
        limitations = [
            "Conclusions may change with additional metadata, ground truth, or replication runs."
        ]
    contract = {
        "result": str(source.get("result") or "").strip() or final_response,
        "evidence": evidence,
        "measurements": measurements,
        "statistical_analysis": statistical_analysis,
        "confidence": {"level": confidence_level, "why": confidence_why},
        "qc_warnings": qc_warnings,
        "limitations": limitations,
        "next_steps": next_steps,
    }
    return contract


def _coerce_phase_h_payload(
    payload: dict[str, Any] | None,
    *,
    fallback_response: str,
) -> dict[str, Any]:
    source = payload if isinstance(payload, dict) else {}
    response = _normalize_user_response_text(str(source.get("final_response") or ""))
    if not response:
        response = _normalize_user_response_text(str(fallback_response or ""))
    if response.startswith("{") and '"result"' in response:
        extracted = _extract_result_from_contract_like_text(response)
        if extracted:
            response = extracted
    contract = _coerce_phase_h_contract_payload(
        source.get("contract") if isinstance(source.get("contract"), dict) else None,
        final_response=response,
    )
    return {"final_response": response, "contract": contract}


def _generate_phase_h_payload(
    *,
    client: Any,
    model: str,
    scratchpad_path: str,
    fallback_response: str,
    max_tokens: int,
    verbosity_level: str,
) -> dict[str, Any]:
    scratchpad_file = Path(str(scratchpad_path)).expanduser().resolve()
    scratchpad_text = ""
    try:
        scratchpad_text = scratchpad_file.read_text(encoding="utf-8")
    except Exception:
        scratchpad_text = ""

    messages = [
        {
            "role": "system",
            "content": (
                "You are the high-effort finalization phase for a scientific assistant.\n"
                "Return exactly one strict JSON object with keys:\n"
                "{\n"
                '  "final_response": string,\n'
                '  "contract": {\n'
                '    "result": string,\n'
                '    "evidence": array,\n'
                '    "measurements": array,\n'
                '    "statistical_analysis": array,\n'
                '    "confidence": {"level":"low|medium|high","why":[string,...]},\n'
                '    "qc_warnings": [string,...],\n'
                '    "limitations": [string,...],\n'
                '    "next_steps": [{"action": string}, ...]\n'
                "  }\n"
                "}\n"
                "Rules:\n"
                "- final_response must directly answer the user with substantive scientific detail.\n"
                "- Write final_response like a polished research brief: start with a 1-2 sentence bottom line, then use short paragraphs for interpretation, comparison, and implications.\n"
                "- Use final_response for synthesis and meaning. Do not simply restate contract headings, measurement bullet lists, or next-step lists verbatim.\n"
                "- Do not start final_response with meta narration such as 'Provided', 'Listed', 'Outlined', 'Summarized', or 'Reported'.\n"
                "- Do not mention scratchpad, internal IDs, or private local paths.\n"
                "- Keep contract aligned with final_response and observed evidence.\n"
                "- If equations are useful, write valid LaTeX only with explicit delimiters: inline \\\\(...\\\\) and display \\\\[...\\\\]. Avoid pseudo-LaTeX, unmatched delimiters, or bare bracket-wrapped math. Keep long chemical or IUPAC names in ordinary prose rather than math mode; if plain text must appear inside math, use \\\\text{...}. Escape backslashes in JSON strings.\n"
                f"- {_response_verbosity_rule(verbosity_level)}\n"
            ),
        },
        {
            "role": "user",
            "content": scratchpad_text or "(scratchpad unavailable)",
        },
    ]
    request_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max(800, int(max_tokens)),
        "temperature": 0.1,
        "stream": False,
    }
    raw_text = ""
    try:
        completion = client.chat.completions.create(
            **request_kwargs,
            extra_body={"reasoning_effort": "high"},
        )
    except Exception:
        completion = client.chat.completions.create(**request_kwargs)
    try:
        if completion.choices and completion.choices[0].message:
            raw_text = str(completion.choices[0].message.content or "")
    except Exception:
        raw_text = ""
    payload = _extract_json_object_from_text(raw_text)
    return _coerce_phase_h_payload(payload, fallback_response=fallback_response)


def _is_metadata_first_image_request(user_text: str) -> bool:
    text = str(user_text or "").lower()
    return _contains_any(
        text,
        (
            "what can you tell me about this image",
            "what can you tell me about these images",
            "tell me about this image",
            "tell me about these images",
            "describe this image",
            "describe these images",
            "summarize this image",
            "summarize these images",
            "quick look at this image",
            "quick look at these images",
            "first look at this image",
            "first look at these images",
            "what can you infer from this image",
            "what can you infer from these images",
            "metadata only for this image",
            "metadata only for these images",
            "before running analysis",
            "before running heavy analysis",
            "before running any analysis",
            "before running any tools",
            "just inspect this image",
            "just inspect these images",
        ),
    )


def _uploaded_path_looks_like_ground_truth(path: str) -> bool:
    lowered = str(path or "").strip().lower()
    if not lowered:
        return False
    if lowered.endswith(".nii.gz"):
        suffix = ".nii.gz"
    else:
        suffixes = Path(lowered).suffixes
        suffix = (
            "".join(suffixes[-2:]).lower() if len(suffixes) >= 2 else Path(lowered).suffix.lower()
        )
    if suffix not in {
        ".npy",
        ".npz",
        ".nii",
        ".nii.gz",
        ".png",
        ".tif",
        ".tiff",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".webp",
    }:
        return False
    normalized = re.sub(r"[^a-z0-9]+", " ", lowered).strip()
    words = set(normalized.split())
    if words.intersection(
        {"gt", "label", "labels", "annotation", "annotations", "annot", "manual", "target", "truth"}
    ):
        return True
    return "ground truth" in normalized or "groundtruth" in normalized


def _select_tool_subset(
    messages: list[dict[str, str]], uploaded_files: list | None, all_tools: list[dict]
) -> list[dict]:
    text = _latest_user_message(messages).lower()
    has_uploads = bool(uploaded_files)
    image_suffixes = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
        ".ome.tif",
        ".ome.tiff",
        ".czi",
        ".nd2",
        ".lif",
        ".lsm",
        ".svs",
        ".vsi",
        ".dv",
        ".r3d",
        ".nii",
        ".nii.gz",
        ".nrrd",
        ".mha",
        ".mhd",
    }
    video_suffixes = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    tabular_suffixes = {".csv", ".tsv", ".tab", ".txt", ".csv.gz", ".tsv.gz"}
    normalized_suffixes: set[str] = set()
    for path in uploaded_files or []:
        if not isinstance(path, str):
            continue
        lower = str(path).strip().lower()
        p = Path(lower)
        if p.suffix:
            normalized_suffixes.add(p.suffix)
        if len(p.suffixes) >= 2:
            normalized_suffixes.add("".join(p.suffixes[-2:]))
    has_pdf_upload = ".pdf" in normalized_suffixes
    has_image_upload = bool(normalized_suffixes.intersection(image_suffixes))
    has_video_upload = bool(normalized_suffixes.intersection(video_suffixes))
    has_table_upload = bool(normalized_suffixes.intersection(tabular_suffixes))
    has_ground_truth_upload = any(
        _uploaded_path_looks_like_ground_truth(str(path))
        for path in (uploaded_files or [])
        if isinstance(path, str)
    )

    bisque_core_tools = {
        "bisque_find_assets",
        "upload_to_bisque",
        "bisque_ping",
        "bisque_download_resource",
        "search_bisque_resources",
        "load_bisque_resource",
        "bisque_download_dataset",
        "bisque_create_dataset",
        "bisque_add_to_dataset",
    }
    bisque_extended_tools = {
        "delete_bisque_resource",
        "add_tags_to_resource",
        "bisque_fetch_xml",
        "bisque_add_gobjects",
        "bisque_advanced_search",
        "run_bisque_module",
    }
    detection_core_tools = {
        "yolo_list_finetuned_models",
        "yolo_detect",
        "yolo_finetune_detect",
        "quantify_objects",
    }
    segmentation_core_tools = {
        "bioio_load_image",
        "segment_image_megaseg",
        "segment_image_sam3",
        "quantify_segmentation_masks",
    }
    segmentation_interactive_tools = {
        "segment_image_sam2",
        "sam2_prompt_image",
        "segment_video_sam2",
    }
    depth_tools = {"estimate_depth_pro"}
    chemistry_tools = {
        "structure_report",
        "compare_structures",
        "propose_reactive_sites",
        "formula_balance_check",
    }
    stats_tools = {
        "compare_conditions",
        "stats_list_curated_tools",
        "stats_run_curated_tool",
    }
    tabular_tools = {"analyze_csv"}
    code_execution_tools = {"codegen_python_plan", "execute_python_job"}

    selected_names: set[str] = set()
    selected_names.add("repro_report")
    by_name = _tool_map(all_tools)
    explicit_mentions = _explicit_tool_mentions(text, set(by_name.keys()))
    mentions_bisque = _contains_any(
        text,
        ("bisque", "dataset", "resource", "module", "upload", "download"),
    )
    mentions_segmentation = _contains_any(
        text,
        ("segment", "segmentation", "sam2", "medsam2", "sam3", "mask", "track"),
    )
    mentions_megaseg = _contains_any(text, ("megaseg", "dynunet"))
    mentions_depth = _contains_any(
        text,
        ("depth", "depth map", "depth estimation", "monocular depth", "depthpro"),
    )
    mentions_chemistry = _contains_any(
        text,
        (
            "organic chemistry",
            "reaction sequence",
            "reaction pathway",
            "mechanism",
            "retrosynthesis",
            "wittig",
            "pdc",
            "tsoh",
            "smiles",
            "inchi",
            "molecule",
        ),
    )
    mentions_evaluation = _contains_any(
        text,
        (
            "evaluate",
            "evaluation",
            "ground truth",
            "ground_truth",
            "label",
            "dice",
            "iou",
            "benchmark",
        ),
    )
    mentions_detection = _contains_any(
        text,
        ("yolo", "detection", "detect", "bbox", "bounding box", "class table"),
    )
    mentions_yolo_detection = _contains_any(
        text,
        ("yolo", "object detection", "bounding box", "bbox", "class table"),
    )
    mentions_edge_module = _contains_any(
        text,
        (
            "edge detection",
            "canny",
            "canny edge",
            "edge map",
            "run edge module",
            "edgedetection module",
        ),
    )
    mentions_asset_discovery = _contains_any(
        text,
        (
            "search",
            "find",
            "list",
            "show me",
            "what files",
            "what images",
            "assets",
            "resources",
        ),
    )
    mentions_recent_bisque_catalog = _contains_any(
        text,
        (
            "recently uploaded",
            "recent uploads",
            "latest uploads",
            "most recent",
            "what jpg",
            "what jpeg",
            "what png",
            "what tif",
            "what tiff",
            "what files",
            "what images",
            "which files",
            "which images",
        ),
    )
    mentions_simple_bisque_catalog = bool(
        mentions_bisque and (mentions_asset_discovery or mentions_recent_bisque_catalog)
    )
    mentions_stats = _contains_any(
        text,
        (
            "compare",
            "condition",
            "treatment",
            "control",
            "effect size",
            "confidence interval",
            "p-value",
            "statistical",
            "hypothesis",
            "significant",
        ),
    )
    mentions_tabular = _contains_any(
        text,
        (
            "csv",
            "tsv",
            "dataframe",
            "pandas",
            "spreadsheet",
            "tabular",
            "groupby",
            "data cleaning",
            "malformed csv",
            "column ",
            "row ",
        ),
    )
    mentions_code_execution = _contains_any(
        text,
        (
            "write code",
            "run code",
            "python script",
            "python code",
            "sandbox",
            "execute python",
            "debug code",
            "fix code",
            "pca",
            "random forest",
            "scikit-learn",
            "sklearn",
            "opencv",
            "scipy",
            "numpy",
            "pandas",
        ),
    )
    mentions_image_measurements = _contains_any(
        text,
        (
            "measure",
            "measurement",
            "morphology",
            "shape analysis",
            "feature extraction",
            "mask statistics",
            "area fraction",
            "object size",
        ),
    )
    mentions_one_step = _contains_any(
        text,
        ("one step", "single step", "minimum necessary tools", "minimum tools only"),
    )
    mentions_sam2_flow = _contains_any(
        text,
        ("sam2", "medsam2", "prompt point", "interactive prompt", "track video"),
    )
    metadata_first_image_request = _is_metadata_first_image_request(text)

    if mentions_bisque:
        selected_names.update(bisque_core_tools)
    if mentions_detection:
        selected_names.update(detection_core_tools)
    if mentions_segmentation:
        selected_names.update(segmentation_core_tools)
    if mentions_megaseg:
        selected_names.add("segment_image_megaseg")
    if mentions_sam2_flow:
        selected_names.update(segmentation_interactive_tools)
    if mentions_depth:
        selected_names.update(depth_tools)
    if mentions_chemistry:
        selected_names.update(chemistry_tools)
    if any(
        k in text
        for k in (
            "quantify mask",
            "mask quantification",
            "segmentation quantification",
            "regionprops",
            "mask-based",
        )
    ):
        selected_names.update(
            {"quantify_segmentation_masks", "evaluate_segmentation_masks", "repro_report"}
        )
    if mentions_stats:
        selected_names.update(stats_tools)
    if mentions_tabular:
        selected_names.update(tabular_tools)
        selected_names.update({"stats_list_curated_tools", "stats_run_curated_tool"})
    if mentions_code_execution:
        selected_names.update(code_execution_tools)
    if mentions_image_measurements:
        selected_names.update(
            {
                "bioio_load_image",
                "segment_image_megaseg",
                "segment_image_sam3",
                "quantify_segmentation_masks",
            }
        )

    if mentions_segmentation and mentions_evaluation:
        selected_names.update({"evaluate_segmentation_masks", "quantify_segmentation_masks"})
        if not has_uploads or has_ground_truth_upload:
            selected_names.add("segment_evaluate_batch")
        if (
            "segment_evaluate_batch" in by_name
            and mentions_one_step
            and not (
                {"segment_image_sam3", "evaluate_segmentation_masks", "segment_evaluate_batch"}
                & explicit_mentions
            )
            and (not has_uploads or has_ground_truth_upload)
        ):
            selected_names.discard("segment_image_sam3")
            selected_names.discard("evaluate_segmentation_masks")
            selected_names = {
                "segment_evaluate_batch",
                "quantify_segmentation_masks",
                "repro_report",
                "bioio_load_image",
            }

    if mentions_segmentation and not mentions_detection:
        selected_names.add("quantify_segmentation_masks")
    if mentions_segmentation and "quantify_objects" not in explicit_mentions:
        selected_names.discard("quantify_objects")

    if _contains_any(text, ("delete resource", "remove resource", "delete from bisque")):
        selected_names.add("delete_bisque_resource")
    if _contains_any(text, ("add tag", "tag resource", "metadata tag", "annotate metadata")):
        selected_names.add("add_tags_to_resource")
    if _contains_any(text, ("xml", "mex", "raw metadata")):
        selected_names.add("bisque_fetch_xml")
    if _contains_any(text, ("advanced search", "lucene", "query xml")):
        selected_names.add("bisque_advanced_search")
    if _contains_any(text, ("run module", "bisque module")):
        selected_names.add("run_bisque_module")
    if mentions_edge_module:
        selected_names.add("run_bisque_module")
        if not has_uploads or mentions_asset_discovery:
            selected_names.update(bisque_core_tools)
    if _contains_any(text, ("gobject", "gobjects", "polygon annotation", "add annotations")):
        selected_names.add("bisque_add_gobjects")
    if selected_names.intersection(bisque_extended_tools):
        selected_names.update(bisque_core_tools)

    if mentions_edge_module and has_uploads and not mentions_asset_discovery:
        selected_names.discard("search_bisque_resources")
        selected_names.discard("bisque_find_assets")
        selected_names.discard("bisque_advanced_search")
        selected_names.discard("load_bisque_resource")
        selected_names.discard("bisque_download_resource")
        selected_names.discard("bisque_download_dataset")
        selected_names.discard("bisque_create_dataset")

    if mentions_edge_module and not mentions_yolo_detection:
        selected_names.discard("yolo_detect")
        selected_names.discard("yolo_finetune_detect")
        selected_names.discard("yolo_list_finetuned_models")
        selected_names.discard("quantify_objects")

    if mentions_simple_bisque_catalog:
        selected_names.add("search_bisque_resources")
        if not _contains_any(
            text, ("metadata", "dimensions", "header", "uri", "resource uri", "client view")
        ):
            selected_names.discard("load_bisque_resource")
        for tool_name in (
            "upload_to_bisque",
            "bisque_create_dataset",
            "bisque_add_to_dataset",
            "delete_bisque_resource",
            "add_tags_to_resource",
            "bisque_add_gobjects",
            "run_bisque_module",
            "bisque_download_resource",
            "bisque_download_dataset",
            "bisque_advanced_search",
            "bisque_find_assets",
        ):
            if tool_name not in explicit_mentions:
                selected_names.discard(tool_name)

    if has_uploads and not selected_names.intersection(
        detection_core_tools
        | segmentation_core_tools
        | segmentation_interactive_tools
        | depth_tools
    ):
        if has_table_upload and not has_image_upload and not has_video_upload:
            selected_names.update(
                {"analyze_csv", "stats_list_curated_tools", "stats_run_curated_tool"}
            )
        elif has_video_upload and not has_pdf_upload and not has_image_upload:
            selected_names.update(
                {
                    "bioio_load_image",
                    "segment_image_sam3",
                    "estimate_depth_pro",
                    "yolo_detect",
                    "quantify_segmentation_masks",
                }
            )
        elif has_image_upload:
            selected_names.update({"bioio_load_image"})

    if has_image_upload and metadata_first_image_request:
        selected_names.discard("segment_image_sam3")
        selected_names.discard("segment_evaluate_batch")
        selected_names.discard("evaluate_segmentation_masks")
        selected_names.discard("quantify_segmentation_masks")
        selected_names.discard("quantify_objects")
        selected_names.discard("yolo_detect")
        selected_names.discard("estimate_depth_pro")
        selected_names.add("bioio_load_image")

    if len(selected_names) <= 1:
        if has_table_upload:
            selected_names.update(
                {"analyze_csv", "stats_list_curated_tools", "stats_run_curated_tool"}
            )
        elif has_image_upload:
            selected_names.update({"bioio_load_image"})
        elif mentions_code_execution:
            selected_names.update(code_execution_tools)
        else:
            selected_names.update({"bisque_find_assets", "bioio_load_image", "analyze_csv"})

    selected_names.update(explicit_mentions)

    subset: list[dict] = []
    for name in selected_names:
        tool = by_name.get(name)
        if tool is not None:
            subset.append(tool)

    # Preserve original order for deterministic prompting.
    ordered = [tool for tool in all_tools if tool in subset]
    return ordered if ordered else all_tools


def stream_chat_completion_with_tools(
    messages: list[dict[str, str]],
    enable_tools: bool = True,
    uploaded_files: list = None,
    execution_plan: dict = None,
    scratchpad_path: str | None = None,
) -> Iterator[str]:
    """Stream a response via the tool-calling engine."""
    settings = get_settings()
    if bool(getattr(settings, "llm_mock_mode", False)):
        yield from _mock_stream_chat_completion(
            messages=messages,
            uploaded_files=uploaded_files,
        )
        return

    if not enable_tools:
        logger.warning("enable_tools=False is not supported; proceeding anyway.")

    client = get_openai_client()
    base_messages = list(messages)

    user_text = _latest_user_message(base_messages)
    code_execution_enabled = bool(getattr(settings, "code_execution_enabled", False))
    prompt_workpad_pipeline_enabled = bool(
        getattr(settings, "prompt_workpad_pipeline_enabled", False)
    )
    prompt_workpad_refinement_mode = (
        str(getattr(settings, "prompt_workpad_refinement_mode", "legacy") or "legacy")
        .strip()
        .lower()
    )
    if prompt_workpad_refinement_mode not in {"legacy", "phased"}:
        prompt_workpad_refinement_mode = "legacy"
    workpad_phase_l_enabled = (
        bool(scratchpad_path)
        and prompt_workpad_pipeline_enabled
        and prompt_workpad_refinement_mode == "phased"
    )
    response_verbosity = _resolve_response_verbosity(
        user_text=user_text,
        default_level=str(getattr(settings, "llm_response_verbosity", "detailed")),
    )
    verbosity_rule = _response_verbosity_rule(response_verbosity)
    reframe_payload_for_planning: dict[str, Any] | None = None
    has_ground_truth_upload = any(
        _uploaded_path_looks_like_ground_truth(str(path))
        for path in (uploaded_files or [])
        if isinstance(path, str)
    )

    if workpad_phase_l_enabled:
        phase_started = time.monotonic()
        yield encode_progress_chunk(
            {
                "event": "phase",
                "phase": "workpad_reframe",
                "status": "started",
                "mode": "phased",
                "message": "Generating low-effort prompt reframe.",
                "ts": datetime.utcnow().isoformat() + "Z",
            }
        )
        try:
            reframe_payload = _generate_prompt_reframe_payload(
                client=client,
                model=_resolved_model(settings),
                messages=base_messages,
            )
            reframe_payload_for_planning = dict(reframe_payload)
            wrote_reframe = upsert_markdown_section(
                path=Path(str(scratchpad_path)).expanduser().resolve(),
                title="Prompt (reframed - low effort)",
                body=_format_prompt_reframe_markdown(reframe_payload),
            )
            yield encode_progress_chunk(
                {
                    "event": "phase",
                    "phase": "workpad_reframe",
                    "status": "completed",
                    "mode": "phased",
                    "message": "Low-effort prompt reframe generated.",
                    "duration_seconds": round(time.monotonic() - phase_started, 3),
                    "wrote_workpad": bool(wrote_reframe),
                    "assumption_count": len(reframe_payload.get("assumptions") or []),
                    "missing_information_count": len(
                        reframe_payload.get("missing_information") or []
                    ),
                    "success_criteria_count": len(reframe_payload.get("success_criteria") or []),
                    "ts": datetime.utcnow().isoformat() + "Z",
                }
            )
        except Exception as exc:
            logger.warning("Prompt workpad reframe phase failed: %s", exc)
            yield encode_progress_chunk(
                {
                    "event": "phase",
                    "phase": "workpad_reframe",
                    "status": "error",
                    "mode": "phased",
                    "message": "Low-effort prompt reframe failed; continuing with original prompt.",
                    "error": str(exc),
                    "duration_seconds": round(time.monotonic() - phase_started, 3),
                    "ts": datetime.utcnow().isoformat() + "Z",
                }
            )

    tool_guide = (
        "You are a scientific assistant with deterministic BisQue tools.\n"
        "Rules:\n"
        "- Use the smallest number of tool calls needed for the request.\n"
        "- If a system message includes persistent tool-call memory or guardrails, treat it as hard policy for this turn.\n"
        "- Be conservative by default for uploaded images: start with metadata/header inspection before expensive inference.\n"
        "- If the user asks what an image contains or asks for a first look, call only bioio_load_image first and wait for user confirmation before running heavy inference. For scientific volumes such as NIfTI, report dimensions, axes, channels, and orientation/header cues in plain language.\n"
        "- When the prompt requests a pipeline (e.g., A then B), execute all requested stages in order; do not stop after stage one.\n"
        "- Never guess about BisQue assets. Use bisque_find_assets or search_bisque_resources.\n"
        "- For simple BisQue existence/listing questions, prefer one search_bisque_resources call before any metadata/download step.\n"
        "- If the user names a destination dataset in plain language (for example 'Upload this to Prairie_Dog_Active_Learning'), treat that string as a resolvable BisQue dataset target. Do not ask for a URI first when the dataset name is already present.\n"
        "- Use upload_to_bisque with dataset_name/dataset_uri when uploads should land in a dataset. Use bisque_add_to_dataset to aggregate existing BisQue resources into an existing or newly created dataset.\n"
        "- BisQue HDF5/DREAM3D assets are table resources. Search them as table/hdf5 resources rather than image resources.\n"
        "- If a BisQue tool fails because of authentication, permissions, or tool/runtime budget, report that exact blocker briefly. Do not replace it with a generic portal walkthrough.\n"
        "- Core researcher path: bioio_load_image -> segment_image_sam3 -> evaluate_segmentation_masks -> quantify_segmentation_masks -> repro_report.\n"
        "- For Megaseg or DynUNet microscopy inference requests, call segment_image_megaseg and preserve its binary mask artifacts for follow-up quantification.\n"
        "- For SAM3, default to preset='balanced'. Use preset='fast' for quick checks and preset='high_quality' only when user asks for best quality.\n"
        "- When the user names object(s) to segment in plain language, distill that wording into concept_prompt and use SAM3 concept mode. Use automatic prompting only when no specific object target is given.\n"
        "- For segmentation tasks that ask for measurements, summaries, morphology, overlap, counts, or region properties, run quantify_segmentation_masks on produced mask artifacts (preferred_upload_paths or prior segmentation mask refs), not on raw source images.\n"
        "- Prefer meaningful scientific outputs: coverage, object count, component-size distribution, and overlap metrics when ground truth is available.\n"
        "- Use segment_evaluate_batch only when ground_truth_paths are available from the user or uploaded labels. If labels are missing, run segment_image_sam3 and explain that evaluation requires ground-truth masks.\n"
        "- For object detection, call yolo_detect. Default pretrained baseline is yolo26x when no model is specified; do not silently opt into the newest local finetuned checkpoint unless the user explicitly asks for the latest finetuned detector or names a model.\n"
        "- If the user asks for the latest finetuned detector, call yolo_list_finetuned_models before yolo_detect.\n"
        "- For depth estimation, call estimate_depth_pro on image files.\n"
        "- For structure-heavy chemistry prompts, use structure_report, compare_structures, propose_reactive_sites, and formula_balance_check to ground ring strain, functional-group deltas, reactive handles, and formula bookkeeping before finalizing.\n"
        "- estimate_depth_pro, segment_image_sam3, and yolo_detect can process uploaded video/sequence files by sampling frames internally.\n"
        "- If the user asks to segment depth outputs, use estimate_depth_pro first, then call segment_image_sam3 with file_paths set to depth_map_paths from the depth result.\n"
        "- For CSV/tabular workflows, call analyze_csv to validate/repair parsing, document issues/fixes, and apply dataframe operations.\n"
        "- For uploaded images, use uploaded local paths directly; do not re-download the same files from BisQue.\n"
        "- For Python code execution workflows, first call codegen_python_plan, then call execute_python_job with returned job_id.\n"
        "- If execute_python_job fails, pass the failure payload to codegen_python_plan.previous_failure and iterate.\n"
        "- Keep code execution retries bounded; stop after budget is exhausted and report limitations clearly.\n"
        "- For code execution outputs, report concrete method + metrics + artifacts (for example dataset shape/class balance, PCA variance metrics, model performance metrics, output filenames) instead of a one-line status.\n"
        "- Use advanced/interactive tools (sam2_prompt_image, segment_image_sam2, segment_video_sam2, stats_*, bisque_advanced_search, run_bisque_module) only when explicitly requested or strictly necessary.\n"
        "- For segmentation uploads, use preferred_upload_path (or preferred_upload_paths) when available; do not upload overlay visualizations unless explicitly requested.\n"
        "- If a system message provides follow-up artifact context with local paths, reuse those paths for follow-up analyses/actions (for example bioio_load_image, segment_image_sam3, execute_python_job inputs, upload_to_bisque) before asking the user to regenerate outputs.\n"
        "- When sharing uploaded BisQue links, prefer client_view_url (or bisque_url) and include image_service_url when available; do not return only bare internal IDs.\n"
        "- In the final answer, avoid exposing internal storage paths (e.g., data/... absolute paths). Use short artifact labels and quantitative findings.\n"
        "- Report numeric metrics exactly as returned by tools.\n"
        "- In result, answer the user directly with substantive content. Do not use meta narration like 'Provided...', 'Listed...', 'Outlined...', or 'Summarized...'.\n"
        "- Treat result as a scientist-facing synthesis brief: begin with the bottom line, then explain the strongest interpretation, comparison, or implication in short paragraphs.\n"
        "- Prefer plain paragraphs for simple conceptual turns; reserve report-style structure for measured, comparative, or tool-backed outputs.\n"
        "- Do not duplicate the contract structure inside result. Let evidence, measurements, qc_warnings, limitations, and next_steps carry the list-like scaffolding.\n"
        "- Do not restate generic confidence, limitation, or next-step boilerplate inside result when the dedicated fields already cover it.\n"
        "- When figures, overlays, plots, or downloadable artifacts are produced, mention how to read the most important output and what pattern or comparison matters most.\n"
        "- For conceptual requests that do not require tools, include concrete scientific details in result (for example grouped metrics/methods with brief definitions), not a one-line abstract.\n"
        "- For tool-executed requests, summarize completion status (completed/partial/failed), key outputs produced, and any missing requested deliverables.\n"
        "- When writing equations, use valid LaTeX only with explicit delimiters: inline \\\\(...\\\\) and display \\\\[...\\\\]. Define symbols on first use, keep notation consistent, and avoid pseudo-LaTeX, unmatched delimiters, or bare bracket-wrapped math. Keep long chemical or IUPAC names in ordinary prose rather than math mode; if plain text must appear inside math, use \\\\text{...}.\n"
        "- Because the final answer is JSON, escape LaTeX backslashes inside strings (for example \\\\alpha, \\\\beta, \\\\frac{a}{b}).\n"
        "- Derive confidence, qc_warnings, and limitations from real tool signals (status, warnings, errors, metric/artifact completeness), not generic filler.\n"
        "- For reproducibility deliverables, call repro_report.\n"
        "- If the user explicitly names a tool, call that exact tool name (do not invent aliases).\n"
        "- Final answer MUST be strict JSON with keys: result, evidence, measurements, statistical_analysis, confidence, qc_warnings, limitations, next_steps.\n"
        "- next_steps must contain at least one concrete action item.\n"
        f"- {verbosity_rule}\n"
        "- Do NOT reveal chain-of-thought.\n"
    )
    if not code_execution_enabled:
        tool_guide = (
            tool_guide.replace(
                "- For Python code execution workflows, first call codegen_python_plan, then call execute_python_job with returned job_id.\n",
                "",
            )
            .replace(
                "- If execute_python_job fails, pass the failure payload to codegen_python_plan.previous_failure and iterate.\n",
                "",
            )
            .replace(
                "- Keep code execution retries bounded; stop after budget is exhausted and report limitations clearly.\n",
                "",
            )
            .replace(
                "- For code execution outputs, report concrete method + metrics + artifacts (for example dataset shape/class balance, PCA variance metrics, model performance metrics, output filenames) instead of a one-line status.\n",
                "",
            )
        )

    messages = [{"role": "system", "content": tool_guide}] + list(base_messages)

    extra_body = None

    all_tools = [
        BISQUE_FIND_ASSETS_TOOL,
        BISQUE_UPLOAD_TOOL,
        BISQUE_PING_TOOL,
        BISQUE_DOWNLOAD_TOOL,
        BISQUE_SEARCH_TOOL,
        BISQUE_LOAD_TOOL,
        BISQUE_DELETE_TOOL,
        BISQUE_TAG_TOOL,
        BISQUE_FETCH_XML_TOOL,
        BISQUE_DOWNLOAD_DATASET_TOOL,
        BISQUE_CREATE_DATASET_TOOL,
        BISQUE_ADD_TO_DATASET_TOOL,
        BISQUE_ADD_GOBJECTS_TOOL,
        BISQUE_ADVANCED_SEARCH_TOOL,
        BISQUE_RUN_MODULE_TOOL,
        BIOIO_LOAD_IMAGE_TOOL,
        DEPTH_PRO_ESTIMATE_TOOL,
        MEGASEG_SEGMENT_TOOL,
        SAM3_SEGMENT_TOOL,
        SEGMENT_EVALUATE_BATCH_TOOL,
        SEGMENTATION_EVAL_TOOL,
        SAM2_SEGMENT_TOOL,
        SAM2_PROMPT_TOOL,
        SAM2_VIDEO_TOOL,
        YOLO_LIST_MODELS_TOOL,
        YOLO_DETECT_TOOL,
        YOLO_FINETUNE_DETECT_TOOL,
        ANALYZE_CSV_TOOL,
        QUANTIFY_OBJECTS_TOOL,
        QUANTIFY_SEGMENTATION_MASKS_TOOL,
        COMPARE_CONDITIONS_TOOL,
        STATS_LIST_CURATED_TOOLS_TOOL,
        STATS_RUN_CURATED_TOOL,
        REPRO_REPORT_TOOL,
    ]
    if code_execution_enabled:
        all_tools.extend(
            [
                CODEGEN_PYTHON_PLAN_TOOL,
                EXECUTE_PYTHON_JOB_TOOL,
            ]
        )
    tools = _select_tool_subset(messages, uploaded_files=uploaded_files, all_tools=all_tools)
    selected_tool_set = {
        str(tool.get("function", {}).get("name", "")) for tool in tools if isinstance(tool, dict)
    }
    explicit_tool_names = sorted(_explicit_tool_mentions(user_text, selected_tool_set))
    dynamic_rules: list[str] = []
    if _is_no_tool_request(user_text):
        dynamic_rules.append(
            "The user explicitly asked to avoid tool calls in this turn. Do not call any tool unless the user later reverses this."
        )
    if _needs_extended_next_steps(user_text):
        dynamic_rules.append(
            "Return at least 3 concrete next_steps with clear actions and purpose."
        )
    if explicit_tool_names:
        dynamic_rules.append(
            "Explicitly requested tools for this turn: "
            + ", ".join(explicit_tool_names)
            + ". Call each at least once before finalizing."
        )
    if _contains_any(
        user_text,
        ("segment", "segmentation", "sam3", "sam2"),
    ) and _contains_any(
        user_text,
        ("evaluate", "evaluation", "ground truth", "label"),
    ):
        if has_ground_truth_upload or not bool(uploaded_files):
            dynamic_rules.append(
                "This prompt combines segmentation and evaluation and label masks are available. Use segment_evaluate_batch unless the user explicitly requests separate calls."
            )
        else:
            dynamic_rules.append(
                "This prompt asks for evaluation, but no uploaded ground-truth mask files were detected. Do not call segment_evaluate_batch unless ground_truth_paths are available; if labels are missing, run segmentation-only analysis or ask for the label masks."
            )
    if _contains_any(
        user_text, ("mask quantification", "quantify segmentation", "quantify_segmentation_masks")
    ):
        dynamic_rules.append(
            "This prompt is mask-centric; prefer quantify_segmentation_masks over quantify_objects unless boxes/detections are explicitly requested."
        )
    if _contains_any(user_text, ("depth", "depth map", "depthpro")):
        dynamic_rules.append(
            "Depth-related request: call estimate_depth_pro before any follow-up segmentation on derived depth maps."
        )
    if _contains_any(
        user_text,
        ("edge detection", "canny", "run edge module", "edgedetection module"),
    ) and bool(uploaded_files):
        dynamic_rules.append(
            "Edge-module request with uploaded image(s): call run_bisque_module directly on uploaded local file paths. "
            "Do not call search_bisque_resources/bisque_find_assets unless the user explicitly asks to search BisQue assets."
        )
    code_execution_prompt = _contains_any(
        user_text,
        (
            "write code",
            "run code",
            "python",
            "pca",
            "random forest",
            "sklearn",
            "opencv",
            "scipy",
            "numpy",
            "pandas",
        ),
    )
    if code_execution_prompt and code_execution_enabled:
        dynamic_rules.append(
            "Code execution workflow: use codegen_python_plan -> execute_python_job and keep retries within max 12 code-tool calls."
        )
        dynamic_rules.append(
            "Final JSON for code execution must include at least 3 quantitative measurements, at least 1 evidence item citing produced artifacts, and explicit limitations tied to observed tool signals."
        )
    elif code_execution_prompt and not code_execution_enabled:
        dynamic_rules.append(
            "Python code execution tools are currently disabled for stability. Do not call codegen_python_plan or execute_python_job."
        )
        dynamic_rules.append(
            "Explain the limitation clearly and continue with available analysis tools only."
        )
    if _is_metadata_first_image_request(user_text):
        dynamic_rules.append(
            "This is an initial image-understanding request: call only bioio_load_image to summarize metadata/headers "
            "(shape, channels, dtype, spacing, EXIF/filename hints). Call it exactly once unless it fails. "
            "Ask before running segmentation, detection, or depth."
        )

    planner_directive_message: str | None = None
    workpad_phase_m_enabled = workpad_phase_l_enabled
    if workpad_phase_m_enabled:
        phase_started = time.monotonic()
        yield encode_progress_chunk(
            {
                "event": "phase",
                "phase": "workpad_plan",
                "status": "started",
                "mode": "phased",
                "message": "Generating medium-effort execution plan.",
                "ts": datetime.utcnow().isoformat() + "Z",
            }
        )
        try:
            planner_payload = _generate_workpad_plan_payload(
                client=client,
                model=_resolved_model(settings),
                user_text=user_text,
                reframe_payload=reframe_payload_for_planning,
                available_tool_names=sorted(selected_tool_set),
                dynamic_rules=dynamic_rules,
                uploaded_file_count=len(uploaded_files or []),
            )
            wrote_plan = upsert_markdown_section(
                path=Path(str(scratchpad_path)).expanduser().resolve(),
                title="Plan (medium effort)",
                body=_format_workpad_plan_markdown(planner_payload),
            )
            planner_directive_message = _planner_system_directive(planner_payload)
            yield encode_progress_chunk(
                {
                    "event": "phase",
                    "phase": "workpad_plan",
                    "status": "completed",
                    "mode": "phased",
                    "message": "Medium-effort execution plan generated.",
                    "duration_seconds": round(time.monotonic() - phase_started, 3),
                    "wrote_workpad": bool(wrote_plan),
                    "planned_steps": len(planner_payload.get("tool_plan") or []),
                    "fallback_rules": len(planner_payload.get("fallback_behavior") or []),
                    "ts": datetime.utcnow().isoformat() + "Z",
                }
            )
        except Exception as exc:
            logger.warning("Prompt workpad planner phase failed: %s", exc)
            yield encode_progress_chunk(
                {
                    "event": "phase",
                    "phase": "workpad_plan",
                    "status": "error",
                    "mode": "phased",
                    "message": "Medium-effort plan failed; continuing with default rules.",
                    "error": str(exc),
                    "duration_seconds": round(time.monotonic() - phase_started, 3),
                    "ts": datetime.utcnow().isoformat() + "Z",
                }
            )

    selected_tool_names = ", ".join(str(tool.get("function", {}).get("name", "")) for tool in tools)
    pre_messages = [
        {
            "role": "system",
            "content": (
                "Selected tool subset for this request: "
                f"{selected_tool_names}. Only call tools from this subset unless absolutely required."
            ),
        },
        {
            "role": "system",
            "content": "Turn-specific directives:\n- " + "\n- ".join(dynamic_rules)
            if dynamic_rules
            else "Turn-specific directives: follow the default rules above.",
        },
    ]
    if planner_directive_message:
        pre_messages.append({"role": "system", "content": planner_directive_message})
    messages = pre_messages + messages

    code_execution_turn = code_execution_enabled and _contains_any(
        user_text,
        (
            "write code",
            "run code",
            "python",
            "pca",
            "random forest",
            "sklearn",
            "opencv",
            "scipy",
            "numpy",
            "pandas",
            "sandbox",
        ),
    )
    max_iterations = (
        int(getattr(settings, "code_execution_max_total_tool_calls", 12))
        if code_execution_turn
        else 6
    )
    use_legacy_workpad_refinement = (
        not prompt_workpad_pipeline_enabled
    ) or prompt_workpad_refinement_mode == "legacy"
    workpad_phase_h_enabled = (
        bool(scratchpad_path)
        and prompt_workpad_pipeline_enabled
        and prompt_workpad_refinement_mode == "phased"
    )
    phase_h_min_tokens = max(
        800,
        int(getattr(settings, "prompt_workpad_phase_h_min_tokens", 1400)),
    )
    phase_h_max_tokens = max(
        phase_h_min_tokens,
        int(getattr(settings, "prompt_workpad_phase_h_max_tokens", 6000)),
    )
    quality_repair_enabled = bool(getattr(settings, "prompt_workpad_quality_repair_enabled", True))
    quality_max_meta_rate = float(getattr(settings, "prompt_workpad_quality_max_meta_rate", 0.35))
    quality_min_completeness = float(
        getattr(settings, "prompt_workpad_quality_min_completeness", 0.58)
    )
    quality_min_numeric_density = float(
        getattr(settings, "prompt_workpad_quality_min_numeric_density", 0.45)
    )
    should_refine_from_scratchpad = bool(scratchpad_path) and use_legacy_workpad_refinement
    suppress_draft_output = bool(scratchpad_path) and use_legacy_workpad_refinement

    engine = ToolEngine(
        client=client,
        model=_resolved_model(settings),
        tools=tools,
        tool_executor=execute_tool_call,
        max_iterations=max_iterations,
        stream_delta_fix=True,
    )
    if workpad_phase_h_enabled:
        draft_chunks: list[str] = []
        phase_h_progress_events: list[dict[str, Any]] = []
        for chunk in engine.stream(
            messages,
            uploaded_files=uploaded_files,
            extra_body=extra_body,
            scratchpad_path=scratchpad_path,
            refine_response=False,
            suppress_draft_output=True,
            emit_suppressed_draft_when_no_refine=False,
            workpad_mode="phased",
        ):
            progress_payload = decode_progress_chunk(chunk)
            if progress_payload is not None:
                phase_h_progress_events.append(progress_payload)
                yield chunk
            else:
                draft_chunks.append(str(chunk or ""))
        draft_text = _normalize_user_response_text("".join(draft_chunks))
        scratchpad_snapshot = ""
        try:
            scratchpad_snapshot = (
                Path(str(scratchpad_path)).expanduser().resolve().read_text(encoding="utf-8")
            )
        except Exception:
            scratchpad_snapshot = ""
        if not draft_text and scratchpad_snapshot:
            draft_text = _extract_latest_scratchpad_draft_text(scratchpad_snapshot)
        phase_h_max_tokens, budget_trace = _estimate_phase_h_output_budget(
            user_text=user_text,
            scratchpad_text=scratchpad_snapshot,
            fallback_response=draft_text,
            verbosity_level=response_verbosity,
            min_tokens=phase_h_min_tokens,
            max_tokens=phase_h_max_tokens,
        )
        phase_started = time.monotonic()
        yield encode_progress_chunk(
            {
                "event": "phase",
                "phase": "workpad_finalize",
                "status": "started",
                "mode": "phased",
                "message": "Generating high-effort final response and contract sidecar.",
                "max_tokens": phase_h_max_tokens,
                "complexity_tier": budget_trace.get("tier"),
                "complexity_score": budget_trace.get("complexity_score"),
                "ts": datetime.utcnow().isoformat() + "Z",
            }
        )
        try:
            phase_h_payload = _generate_phase_h_payload(
                client=client,
                model=_resolved_model(settings),
                scratchpad_path=str(scratchpad_path),
                fallback_response=draft_text,
                max_tokens=phase_h_max_tokens,
                verbosity_level=response_verbosity,
            )
            final_response = _normalize_user_response_text(
                str(phase_h_payload.get("final_response") or "")
            )
            if not final_response:
                final_response = draft_text
            contract_payload = (
                phase_h_payload.get("contract")
                if isinstance(phase_h_payload.get("contract"), dict)
                else None
            )
            quality_metrics = _compute_response_quality_metrics(
                response_text=final_response,
                user_text=user_text,
                contract_payload=contract_payload,
                progress_events=phase_h_progress_events,
            )
            quality_payload: dict[str, Any] = {
                "event": "response_quality",
                "phase": "workpad_finalize",
                "mode": "phased",
                "metrics": quality_metrics,
                "repair_attempted": False,
                "repair_applied": False,
                "max_tokens": phase_h_max_tokens,
                "budget": budget_trace,
                "ts": datetime.utcnow().isoformat() + "Z",
            }
            if (
                quality_repair_enabled
                and final_response
                and _should_run_quality_repair(
                    quality_metrics=quality_metrics,
                    max_meta_rate=quality_max_meta_rate,
                    min_completeness=quality_min_completeness,
                    min_numeric_density=quality_min_numeric_density,
                )
            ):
                quality_payload["repair_attempted"] = True
                repaired_response = _rewrite_low_quality_phase_h_response(
                    client=client,
                    model=_resolved_model(settings),
                    user_prompt=user_text,
                    scratchpad_text=scratchpad_snapshot,
                    draft_response=final_response,
                    contract_payload=contract_payload,
                    quality_metrics=quality_metrics,
                    max_tokens=phase_h_max_tokens,
                    verbosity_level=response_verbosity,
                )
                if repaired_response:
                    repaired_metrics = _compute_response_quality_metrics(
                        response_text=repaired_response,
                        user_text=user_text,
                        contract_payload=contract_payload,
                        progress_events=phase_h_progress_events,
                    )
                    if _quality_metrics_improved(
                        baseline=quality_metrics,
                        candidate=repaired_metrics,
                    ):
                        final_response = repaired_response
                        quality_metrics = repaired_metrics
                        quality_payload["repair_applied"] = True
                quality_payload["post_repair_metrics"] = quality_metrics
            yield encode_progress_chunk(quality_payload)
            if isinstance(contract_payload, dict):
                contract_result = _normalize_user_response_text(
                    str(contract_payload.get("result") or "")
                )
                if (not contract_result) or _meta_narration_rate(contract_result) >= 0.5:
                    contract_payload["result"] = final_response
            wrote_answer = upsert_markdown_section(
                path=Path(str(scratchpad_path)).expanduser().resolve(),
                title="Final Answer Draft (high effort)",
                body=final_response or "(empty)",
            )
            wrote_contract = False
            if isinstance(contract_payload, dict):
                wrote_contract = upsert_markdown_section(
                    path=Path(str(scratchpad_path)).expanduser().resolve(),
                    title="Final Contract",
                    body="```json\n"
                    + json.dumps(contract_payload, indent=2, ensure_ascii=False)
                    + "\n```",
                )
                yield encode_progress_chunk(
                    {
                        "event": "workpad_contract",
                        "phase": "workpad_finalize",
                        "mode": "phased",
                        "contract": contract_payload,
                        "ts": datetime.utcnow().isoformat() + "Z",
                    }
                )
            yield encode_progress_chunk(
                {
                    "event": "phase",
                    "phase": "workpad_finalize",
                    "status": "completed",
                    "mode": "phased",
                    "message": "High-effort final response generated.",
                    "duration_seconds": round(time.monotonic() - phase_started, 3),
                    "wrote_workpad_answer": bool(wrote_answer),
                    "wrote_workpad_contract": bool(wrote_contract),
                    "quality": {
                        "meta_narration_rate": quality_metrics.get("meta_narration_rate"),
                        "answer_completeness": quality_metrics.get("answer_completeness"),
                        "numeric_detail_density": quality_metrics.get("numeric_detail_density"),
                    },
                    "repair_applied": bool(quality_payload.get("repair_applied")),
                    "ts": datetime.utcnow().isoformat() + "Z",
                }
            )
            if final_response:
                yield final_response
                return
            if draft_text:
                yield draft_text
                return
            yield "Unable to produce a final response."
            return
        except Exception as exc:
            logger.warning("Prompt workpad finalization phase failed: %s", exc)
            yield encode_progress_chunk(
                {
                    "event": "phase",
                    "phase": "workpad_finalize",
                    "status": "error",
                    "mode": "phased",
                    "message": "High-effort finalization failed; using draft response.",
                    "error": str(exc),
                    "duration_seconds": round(time.monotonic() - phase_started, 3),
                    "ts": datetime.utcnow().isoformat() + "Z",
                }
            )
            if draft_text:
                yield draft_text
                return
            yield "Unable to produce a final response."
            return

    yield from engine.stream(
        messages,
        uploaded_files=uploaded_files,
        extra_body=extra_body,
        scratchpad_path=scratchpad_path,
        refine_response=should_refine_from_scratchpad,
        suppress_draft_output=suppress_draft_output,
        workpad_mode=(
            prompt_workpad_refinement_mode if prompt_workpad_pipeline_enabled else "legacy"
        ),
    )


def stream_chat_completion(
    messages: list[dict[str, str]],
    uploaded_files: list = None,
    execution_plan: dict = None,
    scratchpad_path: str | None = None,
) -> Iterator[str]:
    """
    Stream a chat completion response (backward compatibility wrapper).

    Args:
        messages: List of message dictionaries with 'role' and 'content'.
        uploaded_files: Optional list of uploaded files for tool context.
        execution_plan: Optional execution plan for progress tracking.

    Yields:
        Streamed response chunks.
    """
    yield from stream_chat_completion_with_tools(
        messages,
        enable_tools=True,
        uploaded_files=uploaded_files,
        execution_plan=execution_plan,
        scratchpad_path=scratchpad_path,
    )


def _mock_stream_chat_completion(
    *,
    messages: list[dict[str, str]],
    uploaded_files: list | None = None,
) -> Iterator[str]:
    user_text = _latest_user_message(messages).lower()
    uploaded = [str(path) for path in (uploaded_files or []) if str(path).strip()]
    uploaded_count = len(uploaded)
    sample_artifact = str(
        Path(__file__).resolve().parents[1] / "tests" / "test_inputs" / "test.png"
    )

    def _emit_started(tool_name: str, message: str) -> str:
        return encode_progress_chunk(
            {
                "event": "started",
                "tool": tool_name,
                "message": message,
            }
        )

    def _emit_completed(
        tool_name: str,
        message: str,
        *,
        summary: dict | None = None,
        artifacts: list[dict] | None = None,
    ) -> str:
        payload: dict[str, object] = {
            "event": "completed",
            "tool": tool_name,
            "message": message,
        }
        if summary:
            payload["summary"] = summary
        if artifacts:
            payload["artifacts"] = artifacts
        return encode_progress_chunk(payload)

    def _contract_json(
        result: str,
        *,
        measurements: list[dict] | None = None,
        limitations: list[str] | None = None,
        next_steps: list[dict] | None = None,
    ) -> str:
        payload = {
            "result": result,
            "evidence": [],
            "measurements": measurements or [],
            "statistical_analysis": [],
            "confidence": {
                "level": "medium",
                "why": ["Deterministic mock mode output for integration testing."],
            },
            "qc_warnings": [],
            "limitations": limitations
            or ["This is a deterministic mock-mode response intended for E2E testing."],
            "next_steps": next_steps
            or [{"action": "Continue with the next requested analysis stage."}],
        }
        return json.dumps(payload, ensure_ascii=False)

    if "depth" in user_text and "segment" in user_text:
        yield _emit_started("estimate_depth_pro", "DepthPro started")
        yield _emit_completed(
            "estimate_depth_pro",
            "DepthPro completed",
            summary={
                "model": "depth-pro-mock",
                "processed": max(1, uploaded_count),
                "total_files": max(1, uploaded_count),
                "depth_mean_average": 0.44,
                "files": [
                    {
                        "file": sample_artifact,
                        "depth_mean": 0.44,
                        "depth_min": 0.02,
                        "depth_max": 0.98,
                    }
                ],
            },
            artifacts=[{"path": sample_artifact, "kind": "image", "title": "Depth preview"}],
        )
        yield _emit_started("segment_image_sam3", "SAM3 started")
        yield _emit_completed(
            "segment_image_sam3",
            "SAM3 completed",
            summary={
                "model": "sam3-mock",
                "processed": max(1, uploaded_count),
                "total_files": max(1, uploaded_count),
                "total_masks_generated": 4,
                "coverage_percent_mean": 26.2,
                "coverage_percent_min": 26.2,
                "coverage_percent_max": 26.2,
                "min_points": 64,
                "max_points": 64,
                "files": [
                    {
                        "file": sample_artifact,
                        "coverage_percent": 26.2,
                        "total_masks": 4,
                        "avg_points_per_window": 64,
                        "min_points": 64,
                        "max_points": 64,
                    }
                ],
            },
            artifacts=[{"path": sample_artifact, "kind": "image", "title": "SAM3 overlay"}],
        )
        yield _contract_json(
            "Depth estimation and SAM3 segmentation completed successfully.",
            measurements=[
                {"name": "depth_mean", "value": 0.44, "unit": "relative"},
                {"name": "mask_coverage_percent", "value": 26.2, "unit": "%"},
                {"name": "masks_generated", "value": 4, "unit": "count"},
            ],
            next_steps=[
                {"action": "Review overlay artifacts and verify segmentation boundaries."},
                {"action": "Run quantify_segmentation_masks for morphology statistics."},
            ],
        )
        return

    if "yolo" in user_text and ("segment" in user_text or "sam3" in user_text):
        yield _emit_started("yolo_detect", "YOLO detection started")
        yield _emit_completed(
            "yolo_detect",
            "YOLO detection completed",
            summary={
                "model_name": "yolo-mock",
                "total_boxes": 7,
                "avg_confidence": 0.81,
                "classes": [
                    {"class_name": "cell", "count": 5},
                    {"class_name": "artifact", "count": 2},
                ],
                "finetune_recommended": False,
            },
            artifacts=[{"path": sample_artifact, "kind": "image", "title": "YOLO detections"}],
        )
        yield _emit_started("segment_image_sam3", "SAM3 segmentation started")
        yield _emit_completed(
            "segment_image_sam3",
            "SAM3 segmentation completed",
            summary={
                "model": "sam3-mock",
                "processed": 1,
                "total_files": 1,
                "total_masks_generated": 3,
                "coverage_percent_mean": 19.8,
                "coverage_percent_min": 18.4,
                "coverage_percent_max": 21.0,
                "min_points": 64,
                "max_points": 64,
                "files": [
                    {
                        "file": sample_artifact,
                        "coverage_percent": 19.8,
                        "total_masks": 3,
                        "avg_points_per_window": 64,
                        "min_points": 64,
                        "max_points": 64,
                    }
                ],
            },
            artifacts=[{"path": sample_artifact, "kind": "image", "title": "SAM3 mask overlay"}],
        )
        yield _contract_json(
            "YOLO detection and SAM3 segmentation completed.",
            measurements=[
                {"name": "detected_boxes", "value": 7, "unit": "count"},
                {"name": "segmentation_masks", "value": 3, "unit": "count"},
            ],
        )
        return

    if "csv" in user_text or "dataframe" in user_text or "table" in user_text:
        yield _emit_started("analyze_csv", "CSV analysis started")
        yield _emit_completed(
            "analyze_csv",
            "CSV analysis completed",
            summary={
                "rows": 120,
                "columns": 8,
                "malformed_rows_fixed": 2,
                "operations_applied": ["trim_whitespace", "drop_empty_rows"],
            },
        )
        yield _contract_json(
            "CSV validation and cleanup completed.",
            measurements=[
                {"name": "rows", "value": 120, "unit": "count"},
                {"name": "columns", "value": 8, "unit": "count"},
                {"name": "malformed_rows_fixed", "value": 2, "unit": "count"},
            ],
        )
        return

    if "what analysis have i ran" in user_text or "previous chat" in user_text:
        yield _contract_json(
            "Recent analyses are available from persisted run history and can be listed on request.",
            next_steps=[
                {"action": "Ask: list my last 5 analyses."},
                {"action": "Ask: compare run metrics between two run IDs."},
            ],
        )
        return

    yield _contract_json(
        "Request processed successfully in deterministic mock mode.",
        measurements=[{"name": "uploaded_files", "value": uploaded_count, "unit": "count"}],
        next_steps=[{"action": "Provide a more specific scientific task for tool execution."}],
    )


def get_available_models() -> list[str]:
    """
    Get list of available models from the API.

    Returns:
        List of model names.
    """
    try:
        client = get_openai_client()
        models = client.models.list()
        model_list = [model.id for model in models.data]
        logger.info(f"Retrieved {len(model_list)} models from API")
        return model_list
    except Exception as e:
        logger.error(f"Failed to retrieve models: {str(e)}")
        settings = get_settings()
        return [_resolved_model(settings)]
