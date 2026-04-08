from __future__ import annotations

import json
import time
from typing import Any

from src.logger import logger
from src.orchestration.models import RunStatus, ToolStep, WorkflowPlan
from src.orchestration.store import RunStore
from src.tools import execute_tool_call


class PlanExecutor:
    """Execute a WorkflowPlan step-by-step with durable trace events.
    """

    def __init__(self, store: RunStore):
        self.store = store

    def execute(self, run_id: str, plan: WorkflowPlan) -> None:
        """Execute each plan step with retries and append durable run events.
        
        Parameters
        ----------
        run_id : str
            Workflow run identifier.
        plan : WorkflowPlan
            Input argument.
        
        Returns
        -------
        None
            No return value.
        
        Notes
        -----
        Pass only validated plans to avoid runtime argument-resolution failures.
        Rely on emitted events for UI/monitoring status rather than ad-hoc logs.
        """
        self.store.update_status(run_id, RunStatus.RUNNING)
        self.store.append_event(run_id, "run_started", {"goal": plan.goal})

        context: dict[str, Any] = {"steps": {}}  # 1-based index -> {id, tool_name, result}

        try:
            for index, step in enumerate(plan.steps, 1):
                result = self._execute_step(run_id, index, step, context)
                context["steps"][str(index)] = {
                    "id": step.id,
                    "tool_name": step.tool_name,
                    "result": result,
                }

            self.store.update_status(run_id, RunStatus.SUCCEEDED)
            self.store.append_event(run_id, "run_succeeded", {})
        except Exception as e:
            logger.exception("Workflow run failed")
            self.store.update_status(run_id, RunStatus.FAILED, error=str(e))
            self.store.append_event(run_id, "run_failed", {"error": str(e)}, level="error")

    def _execute_step(
        self, run_id: str, index: int, step: ToolStep, context: dict[str, Any]
    ) -> dict[str, Any]:
        resolved_args = _resolve_refs(step.arguments, context)
        self.store.append_event(
            run_id,
            "step_started",
            {
                "step_index": index,
                "step_id": step.id,
                "tool_name": step.tool_name,
                "description": step.description,
                "arguments": resolved_args,
            },
        )

        start = time.time()
        last_error: str | None = None

        for attempt in range(step.retries + 1):
            failed_result_payload: Any = None
            try:
                raw = execute_tool_call(step.tool_name, resolved_args)
                result = json.loads(raw) if isinstance(raw, str) else raw
                if isinstance(result, dict) and result.get("success") is False:
                    failed_result_payload = _shrink_result(result)
                    raise RuntimeError(
                        str(result.get("error") or result.get("error_message") or "Step reported failure")
                    )

                duration = time.time() - start
                self.store.append_event(
                    run_id,
                    "step_succeeded",
                    {
                        "step_index": index,
                        "step_id": step.id,
                        "tool_name": step.tool_name,
                        "attempt": attempt + 1,
                        "duration_seconds": round(duration, 3),
                        "result": _shrink_result(result),
                    },
                )
                return result if isinstance(result, dict) else {"result": result}
            except Exception as e:
                last_error = str(e)
                self.store.append_event(
                    run_id,
                    "step_failed_attempt",
                    {
                        "step_index": index,
                        "step_id": step.id,
                        "tool_name": step.tool_name,
                        "attempt": attempt + 1,
                        "error": last_error,
                        "result": failed_result_payload,
                    },
                    level="error",
                )
                if attempt < step.retries:
                    time.sleep(0.5)

        raise RuntimeError(last_error or "Step failed")


def _resolve_refs(value: Any, context: dict[str, Any]) -> Any:
    """Resolve simple $ref pointers in step arguments.

    Supported form:
      {"$ref": "steps.1.result.some_key"}
    """
    if isinstance(value, dict):
        if set(value.keys()) == {"$ref"} and isinstance(value.get("$ref"), str):
            return _get_by_path(context, value["$ref"])
        return {k: _resolve_refs(v, context) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_refs(v, context) for v in value]
    return value


def _get_by_path(data: dict[str, Any], path: str) -> Any:
    cur: Any = data
    for part in path.split("."):
        if isinstance(cur, dict):
            if part not in cur:
                raise KeyError(f"Missing ref path segment: {part} in {path}")
            cur = cur[part]
        elif isinstance(cur, list):
            idx = int(part)
            cur = cur[idx]
        else:
            raise KeyError(f"Cannot resolve ref path {path}; segment {part}")
    return cur


def _shrink_result(result: Any) -> Any:
    """Prevent event log blowups; keep summaries, paths, and key metrics."""
    if isinstance(result, dict):
        # Keep common, useful keys if present
        keep_keys = {
            "success",
            "error",
            "processed",
            "total_files",
            "total_masks_generated",
            "total_frames_processed",
            "files_processed",
            "output_directory",
            "visualization_paths",
            "mex_url",
            "inputs",
            "message",
            "workflow_trace",
            # YOLO / detection / training
            "detections",
            "predictions",
            "counts_by_class",
            "prediction_images",
            "predictions_json",
            "dataset_dir",
            "dataset_yaml_path",
            "train_dir",
            "prepared_only",
            "model_name",
            "model_path",
            "registry_path",
            "metrics",
            # Scientific sandbox
            "session_id",
            "summary",
            "results",
            "ui_artifacts",
            # Python code execution
            "job_id",
            "execution_backend",
            "exit_code",
            "runtime_seconds",
            "timeout_seconds",
            "stdout_tail",
            "stderr_tail",
            "error_class",
            "error_message",
            "repair_hint",
            "attempt_id",
            "failure_signature",
            "durable_execution",
            "durable_run_id",
            "durable_status",
            "submitted_via",
            "execution_command",
            "docker_image",
            "artifacts",
            "output_files",
            "expected_outputs",
            "missing_expected_outputs",
            "analysis_outputs",
            "measurement_candidates",
            "key_measurements",
            "repair_cycles_used",
            "repair_stop_reason",
            "attempt_history",
        }
        slim = {k: v for k, v in result.items() if k in keep_keys}
        return slim if slim else result
    return result
