#!/usr/bin/env python3
"""App-side smoke test for service-backed code execution."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import httpx


def _tool_invocations(parsed: dict[str, object]) -> list[dict[str, object]]:
    metadata = parsed.get("metadata") if isinstance(parsed, dict) else {}
    invocations = metadata.get("tool_invocations") if isinstance(metadata, dict) else None
    if isinstance(invocations, list):
        return [item for item in invocations if isinstance(item, dict)]
    return []


def _app_chat_smoke(*, api_root: str) -> int:
    payload = {
        "messages": [
            {
                "role": "user",
                "content": (
                    "In Pro Mode, write and run Python to generate a synthetic classification dataset, "
                    "compare random forest to logistic regression, save result.json and outputs/feature_importance.png, "
                    "and explain whether overfitting is visible."
                ),
            }
        ],
        "conversation_id": "codeexec-production-smoke",
        "workflow_hint": {"id": "pro_mode", "source": "slash_menu"},
        "reasoning_mode": "deep",
        "debug": True,
        "budgets": {"max_tool_calls": 12, "max_runtime_seconds": 420},
    }
    with httpx.Client(timeout=480.0) as client:
        response = client.post(f"{api_root.rstrip('/')}/v1/chat", json=payload)
    response.raise_for_status()
    parsed = response.json()
    tool_names = [
        str(item.get("tool") or item.get("tool_name") or "").strip()
        for item in _tool_invocations(parsed)
    ]
    print(json.dumps(parsed, indent=2, default=str))
    if "codegen_python_plan" not in tool_names or "execute_python_job" not in tool_names:
        raise SystemExit(2)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--service-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--api-root")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="codeexec-app-smoke-") as tmp:
        tmp_dir = Path(tmp)
        os.environ["ARTIFACT_ROOT"] = str(tmp_dir / "artifacts")
        os.environ["CODE_EXECUTION_SERVICE_URL"] = args.service_url
        os.environ["CODE_EXECUTION_SERVICE_API_KEY"] = args.api_key
        os.environ["CODE_EXECUTION_ENABLED"] = "true"

        from src.tooling.code_execution import execute_python_job_once, persist_python_job_spec

        csv_path = tmp_dir / "smoke.csv"
        csv_path.write_text(
            "\n".join(
                [
                    "x1,x2,label",
                    "0.1,0.2,0",
                    "0.2,0.1,0",
                    "0.8,0.9,1",
                    "0.9,0.8,1",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        spec = {
            "entrypoint": "main.py",
            "command": "python main.py",
            "files": [
                {
                    "path": "main.py",
                    "content": "\n".join(
                        [
                            "from pathlib import Path",
                            "import json",
                            "import pandas as pd",
                            "from sklearn.ensemble import RandomForestClassifier",
                            "",
                            "df = pd.read_csv('/inputs/input_01_smoke.csv')",
                            "X = df[['x1', 'x2']]",
                            "y = df['label']",
                            "model = RandomForestClassifier(n_estimators=16, random_state=3)",
                            "model.fit(X, y)",
                            "accuracy = float(model.score(X, y))",
                            "Path('result.json').write_text(json.dumps({'accuracy': accuracy, 'rows': int(len(df))}, indent=2), encoding='utf-8')",
                            "print(json.dumps({'accuracy': accuracy, 'rows': int(len(df))}))",
                        ]
                    )
                    + "\n",
                }
            ],
            "expected_outputs": ["result.json"],
            "dependencies": ["pandas", "scikit-learn"],
        }
        persisted = persist_python_job_spec(
            job_id="smoke_app_codeexec_job",
            generated_spec=spec,
            task_summary="Train a tiny random forest on the supplied CSV and write result.json.",
            inputs=[
                {
                    "path": str(csv_path),
                    "kind": "file",
                    "description": "Toy CSV for the app-side code execution smoke test.",
                }
            ],
        )
        result = execute_python_job_once(job_id=str(persisted["job_id"]))
        print(json.dumps(result, indent=2, default=str))
        if not bool(result.get("success")):
            raise SystemExit(1)
        if args.api_root:
            return _app_chat_smoke(api_root=str(args.api_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
