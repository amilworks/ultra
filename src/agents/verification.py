"""Verification helpers for SWGE-Lite."""

from __future__ import annotations

import json
import re
from typing import Any

from .contracts import AgentResult, RouteDecision, VerificationIssue, VerificationReport
from .reasoning import (
    extract_mcq_options,
    has_hesitation_language,
    is_mcq_prompt,
    normalize_text,
    parse_mcq_answer_letter,
)

_CLINICAL_DIRECTIVE_RE = re.compile(
    r"\b(prescribe|dosage|dose|medication|treatment plan|diagnose|diagnosis|clinical decision)\b",
    flags=re.IGNORECASE,
)
_DNA_BLOCK_RE = re.compile(r"[ACGT]{30,}", flags=re.IGNORECASE)
_PREMATURE_STOP_MENTION_RE = re.compile(
    r"\b(premature stop|stop codon|early termination|terminated translation|translation (?:stops|terminated)|truncat)\b",
    flags=re.IGNORECASE,
)
_LOSS_FUNCTION_RE = re.compile(
    r"\b(loss of function|loss-of-function|loss of [a-z\- ]*activity|inactiv|abolish(?:ed)? activity|reduced activity)\b",
    flags=re.IGNORECASE,
)
_GAIN_FUNCTION_RE = re.compile(
    r"\b(gain of function|gain-of-function|increased activity|constitutive(?:ly)? active)\b",
    flags=re.IGNORECASE,
)
_WILDTYPE_RE = re.compile(
    r"\b(wild[- ]type phenotype|normal phenotype|no phenotype)\b",
    flags=re.IGNORECASE,
)


def _extract_dna_sequences(user_text: str) -> list[str]:
    return [str(match.group(0)).upper() for match in _DNA_BLOCK_RE.finditer(str(user_text or ""))]


def _has_premature_inframe_stop(sequence: str) -> tuple[bool, str | None, int | None]:
    seq = str(sequence or "").upper()
    if not seq:
        return False, None, None
    start = seq.find("ATG")
    if start < 0:
        return False, None, None
    coding = seq[start:]
    codon_count = len(coding) // 3
    if codon_count < 2:
        return False, None, None
    stop_codons = {"TAA", "TAG", "TGA"}
    for idx in range(codon_count):
        codon = coding[idx * 3 : idx * 3 + 3]
        if codon in stop_codons and idx < codon_count - 1:
            return True, codon, idx
    return False, None, None


def _mcq_reasoning_option_contradiction(*, reasoning_text: str, chosen_option_text: str) -> bool:
    reasoning = (
        str(reasoning_text or "")
        .replace("‑", "-")
        .replace("–", "-")
        .replace("—", "-")
    )
    chosen = (
        str(chosen_option_text or "")
        .replace("‑", "-")
        .replace("–", "-")
        .replace("—", "-")
    )
    if not reasoning or not chosen:
        return False

    reasoning_loss = bool(_LOSS_FUNCTION_RE.search(reasoning))
    reasoning_gain = bool(_GAIN_FUNCTION_RE.search(reasoning))
    reasoning_wildtype = bool(_WILDTYPE_RE.search(reasoning))

    chosen_loss = bool(_LOSS_FUNCTION_RE.search(chosen))
    chosen_gain = bool(_GAIN_FUNCTION_RE.search(chosen))
    chosen_wildtype = bool(_WILDTYPE_RE.search(chosen))

    if reasoning_loss and (chosen_gain or chosen_wildtype):
        return True
    if reasoning_gain and (chosen_loss or chosen_wildtype):
        return True
    if reasoning_wildtype and (chosen_loss or chosen_gain):
        return True
    return False


def build_verification_prompt(
    *,
    user_text: str,
    route: RouteDecision,
    agent_results: dict[str, AgentResult],
) -> str:
    """Build verifier prompt with typed result payload."""

    serialized = []
    for domain_id, result in agent_results.items():
        serialized.append(
            {
                "domain_id": str(domain_id),
                "success": bool(result.success),
                "summary": str(result.summary or ""),
                "error": str(result.error or "") if result.error else None,
            }
        )
    payload = {
        "user_text": str(user_text or ""),
        "route": route.model_dump(mode="json"),
        "agent_results": serialized,
    }
    return (
        "Evaluate the domain outputs for correctness, contradiction, and policy compliance.\n"
        "Return JSON only with keys: passed, issues, retry_domains, notes.\n"
        "Issue severity must be low|medium|high.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def parse_verification_output(raw_text: str) -> VerificationReport | None:
    """Parse verifier model output into ``VerificationReport`` when possible."""

    text = str(raw_text or "").strip()
    if not text:
        return None
    candidates: list[str] = [text]
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE)
    if fenced and fenced.group(1):
        candidates.insert(0, str(fenced.group(1)).strip())
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        token = str(candidate or "").strip()
        if not token:
            continue
        try:
            parsed = json.loads(token)
        except Exception:
            continue
        if not isinstance(parsed, dict):
            continue
        try:
            return VerificationReport.model_validate(parsed)
        except Exception:
            continue
    return None


def heuristic_verification(
    *,
    user_text: str,
    route: RouteDecision,
    agent_results: dict[str, AgentResult],
) -> VerificationReport:
    """Apply deterministic verification checks to domain outputs."""

    issues: list[VerificationIssue] = []
    retry_domains: list[str] = []
    notes: list[str] = []
    combined_by_domain: dict[str, str] = {}

    for domain_id, result in agent_results.items():
        summary = str(result.summary or "").strip()
        raw_output = str(result.raw_output or "").strip()
        combined = f"{summary}\n{raw_output}".strip()
        combined_by_domain[domain_id] = combined
        metadata = dict(result.metadata or {}) if isinstance(result.metadata, dict) else {}
        if not result.success:
            issues.append(
                VerificationIssue(
                    code="solver_failed",
                    severity="high",
                    message=str(result.error or "Domain solver failed."),
                    domain_id=domain_id,
                    correctable=True,
                )
            )
            retry_domains.append(domain_id)
            continue
        if domain_id == "bio" and bool(metadata.get("bio_critic_disagreement")):
            issues.append(
                VerificationIssue(
                    code="bio_critic_disagreement",
                    severity="medium",
                    message="Independent biology critic disagreed with the primary biology answer.",
                    domain_id=domain_id,
                    correctable=True,
                )
            )
            retry_domains.append(domain_id)
        if not combined:
            issues.append(
                VerificationIssue(
                    code="empty_solver_output",
                    severity="high",
                    message="Domain solver returned empty output.",
                    domain_id=domain_id,
                    correctable=True,
                )
            )
            retry_domains.append(domain_id)
            continue
        if domain_id == "medical" and _CLINICAL_DIRECTIVE_RE.search(combined):
            issues.append(
                VerificationIssue(
                    code="medical_policy_directive",
                    severity="high",
                    message="Medical output appears to include clinical directives.",
                    domain_id=domain_id,
                    correctable=True,
                )
            )
            retry_domains.append(domain_id)

    normalized_user_text = normalize_text(user_text)
    combined_outputs = "\n".join(combined_by_domain.values())
    normalized_combined_outputs = normalize_text(combined_outputs)

    if is_mcq_prompt(user_text):
        options = extract_mcq_options(user_text)
        domain_votes: dict[str, str] = {}
        missing_format_domains: list[str] = []
        for domain_id, combined in combined_by_domain.items():
            if not combined:
                continue
            answer_letter = parse_mcq_answer_letter(combined, options=options)
            if answer_letter:
                domain_votes[domain_id] = answer_letter
            else:
                missing_format_domains.append(domain_id)
        if not domain_votes:
            issues.append(
                VerificationIssue(
                    code="mcq_no_explicit_answer",
                    severity="high",
                    message=(
                        "Multiple-choice prompt detected but no domain output returned a parseable "
                        "final option letter."
                    ),
                    domain_id=None,
                    correctable=True,
                )
            )
            retry_domains.extend(route.selected_domains)
        else:
            unique_votes = sorted(set(domain_votes.values()))
            if len(unique_votes) > 1:
                issues.append(
                    VerificationIssue(
                        code="mcq_domain_disagreement",
                        severity="high",
                        message=(
                            "Domain agents disagree on final multiple-choice answer: "
                            + ", ".join(f"{key}={value}" for key, value in sorted(domain_votes.items()))
                        ),
                        domain_id=None,
                        correctable=True,
                    )
                )
                retry_domains.extend(domain_votes.keys())
            for domain_id in missing_format_domains:
                issues.append(
                    VerificationIssue(
                        code="mcq_missing_answer_format",
                        severity="medium",
                        message="Domain output did not include a parseable final option letter.",
                        domain_id=domain_id,
                        correctable=True,
                    )
                )
                retry_domains.append(domain_id)
            for domain_id, combined in combined_by_domain.items():
                if domain_id not in domain_votes:
                    continue
                chosen_letter = str(domain_votes.get(domain_id) or "").upper()
                chosen_option_text = str(options.get(chosen_letter) or "")
                if _mcq_reasoning_option_contradiction(
                    reasoning_text=combined,
                    chosen_option_text=chosen_option_text,
                ):
                    issues.append(
                        VerificationIssue(
                            code="mcq_reasoning_option_mismatch",
                            severity="high",
                            message=(
                                "Reasoning content appears to contradict the selected option text; "
                                "re-evaluate all options before finalizing."
                            ),
                            domain_id=domain_id,
                            correctable=True,
                        )
                    )
                    retry_domains.append(domain_id)
                if has_hesitation_language(combined):
                    issues.append(
                        VerificationIssue(
                            code="mcq_low_commitment",
                            severity="medium",
                            message=(
                                "Answer appears hedged for a multiple-choice prompt; "
                                "provide stronger option-by-option justification."
                            ),
                            domain_id=domain_id,
                            correctable=True,
                        )
                    )
                    retry_domains.append(domain_id)
        notes.append("Applied generic MCQ consistency checks.")

    dna_sequences = _extract_dna_sequences(user_text)
    if (
        dna_sequences
        and any(token in normalized_user_text for token in ("overexpress", "construct", "coding sequence", "transgenic"))
    ):
        premature_stop_detected = False
        detected_stop = None
        for sequence in dna_sequences:
            has_stop, stop_codon, _codon_index = _has_premature_inframe_stop(sequence)
            if has_stop:
                premature_stop_detected = True
                detected_stop = stop_codon
                break
        if premature_stop_detected and not _PREMATURE_STOP_MENTION_RE.search(normalized_combined_outputs):
            target_domain = "core" if "core" in route.selected_domains else (
                str(route.selected_domains[0]) if route.selected_domains else "core"
            )
            issues.append(
                VerificationIssue(
                    code="sequence_stop_codon_missed",
                    severity="high",
                    message=(
                        "The sequence appears to contain an in-frame premature stop codon "
                        f"({detected_stop or 'TAA/TAG/TGA'}), so translation should terminate early."
                    ),
                    domain_id=target_domain,
                    correctable=True,
                )
            )
            retry_domains.append(target_domain)
            notes.append("Applied coding-sequence stop-codon check.")

    dedup_retry: list[str] = []
    seen_retry: set[str] = set()
    selected_domains = {str(token).strip() for token in route.selected_domains}
    for domain_id in retry_domains:
        token = str(domain_id or "").strip()
        if not token or token in seen_retry:
            continue
        if selected_domains and token not in selected_domains:
            continue
        seen_retry.add(token)
        dedup_retry.append(token)

    return VerificationReport(
        passed=len(issues) == 0,
        issues=issues,
        retry_domains=dedup_retry,
        notes=notes,
    )


def merge_reports(
    *,
    model_report: VerificationReport | None,
    heuristic_report: VerificationReport,
    selected_domains: list[str],
) -> VerificationReport:
    """Merge model-based and heuristic verifier reports into one output."""

    if model_report is None:
        return heuristic_report

    issues: list[VerificationIssue] = []
    seen_keys: set[tuple[str, str, str]] = set()
    for source in (model_report.issues, heuristic_report.issues):
        for issue in source:
            key = (str(issue.code), str(issue.domain_id or ""), str(issue.message))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            issues.append(issue)

    retry_domains: list[str] = []
    seen_retry: set[str] = set()
    selected = {str(token).strip() for token in selected_domains}
    for source in (model_report.retry_domains, heuristic_report.retry_domains):
        for domain_id in source:
            token = str(domain_id or "").strip()
            if not token or token in seen_retry:
                continue
            if selected and token not in selected:
                continue
            seen_retry.add(token)
            retry_domains.append(token)

    notes = list(model_report.notes) + [item for item in heuristic_report.notes if item not in model_report.notes]
    return VerificationReport(
        passed=(len(issues) == 0),
        issues=issues,
        retry_domains=retry_domains,
        notes=notes,
    )
