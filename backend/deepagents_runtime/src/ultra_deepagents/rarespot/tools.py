"""RareSpot goal classification.

The RareSpot prairie-dog detection tool/dispatch (``rarespot_ecology_inference``,
``create_rarespot_run``/``wait_for_rarespot_run``, the worker, and the run-anchored
control-plane auth) was retired in favour of the ``prairie-dog-detection`` Skill,
which runs the same ``run_rarespot_inference`` directly in the code sandbox. What
remains here is the lightweight goal classifier the agent still uses to recognise a
*report-only* RareSpot follow-up (e.g. "write a combined report across my RareSpot
runs") so it does not try to re-run inference.
"""

from __future__ import annotations

import re


def looks_report_only_rarespot_goal(goal: str) -> bool:
    text = " ".join(str(goal or "").lower().split())
    if "rarespot" not in text:
        return False
    if not re.search(r"\b(write|compile|combine|combined|synthes|summari[sz]e|report)\b", text):
        return False
    explicit_patterns = (
        r"\b(run|rerun|execute|perform|launch)\b.{0,80}\b(rarespot|inference|detect|detection|pass)\b",
        r"\b(stricter|permissive|new|another)\b.{0,80}\b(threshold|confidence|inference|pass)\b",
    )
    explicit_new_run = any(
        not _is_negated_rarespot_run_directive(text, match.start())
        for pattern in explicit_patterns
        for match in re.finditer(pattern, text)
    )
    return not explicit_new_run


def _is_negated_rarespot_run_directive(text: str, start: int) -> bool:
    prefix = text[max(0, start - 32):start]
    return bool(
        re.search(r"\b(?:do\s+not|don't|dont|never|without)\s+$", prefix)
        or re.search(r"\bno\s+(?:need\s+to\s+|new\s+)?$", prefix)
    )
