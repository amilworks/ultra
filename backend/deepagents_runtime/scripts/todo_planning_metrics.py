#!/usr/bin/env python3
"""Planning-effectiveness metrics for a run's write_todos usage.

Reads run events as JSONL (one object per line with ``event_kind`` and
``payload``; ``tool_call.started`` events carry ``payload.tool_name`` and
``payload.input``) and scores how the model actually used its todo list —
the instrument behind the todo-reminders A/B (todo_reminders.py).

Dump events for a run from the control-plane database like so:

    docker exec bisque-ultra-postgres psql -U postgres -d bisque_ultra -tA -c \\
      "SELECT row_to_json(t) FROM (SELECT sequence_number, event_kind, payload \\
       FROM control_run_events WHERE run_id='run_...' \\
       AND event_kind='tool_call.started' ORDER BY sequence_number) t" > events.jsonl

    python scripts/todo_planning_metrics.py events.jsonl

Metrics (higher is better unless noted):

- planned_first        the first substantive tool call was write_todos
- rewrites_total       number of write_todos calls
- max_stale_gap        LOWER is better: the longest stretch of non-todo tool
                       calls with no status update (from the first plan to the
                       end of the run — the end segment counts; a plan that is
                       never closed out is one long stale gap)
- incremental_transitions  status changes made in rewrites BETWEEN the first
                       plan and the final rewrite (mid-run maintenance)
- final_batch_close    LOWER is better: status transitions crammed into the
                       final rewrite (the "mark everything completed at the
                       end" smell)
- adaptations          items added or removed after the initial plan (plan
                       revised as reality moved)
- terminal_open_items  LOWER is better: items left pending/in_progress in the
                       final list (unreconciled plan)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PlanningMetrics:
    run_label: str
    tool_calls_total: int = 0
    rewrites_total: int = 0
    planned_first: bool = False
    max_stale_gap: int = 0
    incremental_transitions: int = 0
    final_batch_close: int = 0
    adaptations: int = 0
    terminal_open_items: int = 0
    final_statuses: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run": self.run_label,
            "tool_calls_total": self.tool_calls_total,
            "rewrites_total": self.rewrites_total,
            "planned_first": self.planned_first,
            "max_stale_gap": self.max_stale_gap,
            "incremental_transitions": self.incremental_transitions,
            "final_batch_close": self.final_batch_close,
            "adaptations": self.adaptations,
            "terminal_open_items": self.terminal_open_items,
            "final_statuses": self.final_statuses,
        }


def _tool_calls(events_jsonl: Path) -> list[tuple[str, dict[str, Any]]]:
    calls: list[tuple[str, dict[str, Any]]] = []
    for line in events_jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        event = json.loads(line)
        if event.get("event_kind") != "tool_call.started":
            continue
        payload = event.get("payload") or {}
        calls.append((str(payload.get("tool_name", "")), payload.get("input") or {}))
    return calls


def _status_transitions(previous: list[dict], current: list[dict]) -> int:
    prev_by_content = {str(t.get("content", "")): str(t.get("status", "")) for t in previous}
    changed = 0
    for todo in current:
        content = str(todo.get("content", ""))
        status = str(todo.get("status", ""))
        if content in prev_by_content and prev_by_content[content] != status:
            changed += 1
    return changed


def _membership_changes(previous: list[dict], current: list[dict]) -> int:
    prev_contents = {str(t.get("content", "")) for t in previous}
    cur_contents = {str(t.get("content", "")) for t in current}
    return len(prev_contents - cur_contents) + len(cur_contents - prev_contents)


def compute_metrics(events_jsonl: Path, run_label: str | None = None) -> PlanningMetrics:
    calls = _tool_calls(events_jsonl)
    metrics = PlanningMetrics(run_label=run_label or events_jsonl.stem)

    rewrites: list[list[dict]] = []
    gap = 0
    saw_first_rewrite = False
    for position, (tool_name, tool_input) in enumerate(calls):
        if tool_name == "write_todos":
            todos = list(tool_input.get("todos") or [])
            if saw_first_rewrite:
                metrics.max_stale_gap = max(metrics.max_stale_gap, gap)
            rewrites.append(todos)
            saw_first_rewrite = True
            gap = 0
            if position == 0:
                metrics.planned_first = True
        else:
            metrics.tool_calls_total += 1
            if saw_first_rewrite:
                gap += 1
    if saw_first_rewrite:
        # The tail after the last rewrite is a stale stretch too: a plan that is
        # never revisited scores its whole remaining run as one gap.
        metrics.max_stale_gap = max(metrics.max_stale_gap, gap)

    metrics.rewrites_total = len(rewrites)
    for index in range(1, len(rewrites)):
        transitions = _status_transitions(rewrites[index - 1], rewrites[index])
        if index == len(rewrites) - 1:
            metrics.final_batch_close = transitions
        else:
            metrics.incremental_transitions += transitions
        metrics.adaptations += _membership_changes(rewrites[index - 1], rewrites[index])

    if rewrites:
        final = rewrites[-1]
        for todo in final:
            status = str(todo.get("status", "")) or "pending"
            metrics.final_statuses[status] = metrics.final_statuses.get(status, 0) + 1
        metrics.terminal_open_items = sum(
            1 for todo in final if str(todo.get("status", "")) != "completed"
        )
    return metrics


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: todo_planning_metrics.py <events.jsonl> [<events.jsonl> ...]")
        return 2
    results = [compute_metrics(Path(arg)) for arg in argv]
    for metrics in results:
        print(json.dumps(metrics.as_dict(), indent=2))
    if len(results) == 2:
        a, b = results
        print(f"\n--- {a.run_label} vs {b.run_label} ---")
        for key in (
            "rewrites_total",
            "max_stale_gap",
            "incremental_transitions",
            "final_batch_close",
            "terminal_open_items",
        ):
            print(f"{key:26} {getattr(a, key):>6} -> {getattr(b, key):>6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
