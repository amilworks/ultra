#!/usr/bin/env python3
"""Operator smoke test for the dedicated code execution service."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from src.tooling.code_execution_jobs import build_service_submission_bundle
from src.tooling.code_execution_service_client import CodeExecutionServiceClient


def _build_smoke_job(tmp_dir: Path) -> tuple[Path, Path, list[Path], dict[str, object]]:
    csv_path = tmp_dir / "smoke.csv"
    csv_path.write_text(
        "\n".join(
            [
                "x1,x2,label",
                "0.1,0.2,0",
                "0.2,0.1,0",
                "0.8,0.9,1",
                "0.9,0.8,1",
                "0.7,0.85,1",
                "0.15,0.25,0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    job_dir = tmp_dir / "job"
    source_dir = job_dir / "source"
    source_dir.mkdir(parents=True)
    (source_dir / "main.py").write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "import json",
                "import pandas as pd",
                "from sklearn.ensemble import RandomForestClassifier",
                "import matplotlib.pyplot as plt",
                "",
                "df = pd.read_csv('/inputs/input_01_smoke.csv')",
                "X = df[['x1', 'x2']]",
                "y = df['label']",
                "model = RandomForestClassifier(n_estimators=32, random_state=7)",
                "model.fit(X, y)",
                "accuracy = float(model.score(X, y))",
                "Path('outputs').mkdir(parents=True, exist_ok=True)",
                "Path('result.json').write_text(json.dumps({'accuracy': accuracy, 'rows': int(len(df))}, indent=2), encoding='utf-8')",
                "plt.figure(figsize=(4, 3))",
                "plt.bar(['x1', 'x2'], model.feature_importances_)",
                "plt.tight_layout()",
                "plt.savefig('outputs/feature_importance.png', dpi=160)",
                "print(json.dumps({'accuracy': accuracy, 'rows': int(len(df))}))",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (job_dir / "job_spec.json").write_text(
        json.dumps(
            {
                "job_id": "smoke_codeexec_job",
                "entrypoint": "main.py",
                "command": "python main.py",
                "files": [{"path": "main.py"}],
                "expected_outputs": ["result.json", "outputs/feature_importance.png"],
                "inputs": [
                    {
                        "path": str(csv_path),
                        "kind": "file",
                        "name": "smoke.csv",
                        "sandbox_path": "/inputs/input_01_smoke.csv",
                        "description": "Toy CSV for the code execution service smoke test.",
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    bundle_path, _remote_inputs, local_inputs = build_service_submission_bundle(job_dir)
    request_payload = {
        "job_id": "smoke_codeexec_job",
        "timeout_seconds": 600,
        "cpu_limit": 2.0,
        "memory_mb": 4096,
        "expected_outputs": ["result.json", "outputs/feature_importance.png"],
        "inputs": [
            {
                "name": "smoke.csv",
                "kind": "file",
                "sandbox_path": "/inputs/input_01_smoke.csv",
                "description": "Toy CSV for the code execution service smoke test.",
            }
        ],
    }
    return job_dir, bundle_path, [Path(str(item["path"])) for item in local_inputs], request_payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    args = parser.parse_args()

    client = CodeExecutionServiceClient(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout_seconds=60,
    )
    with tempfile.TemporaryDirectory(prefix="codeexec-smoke-") as tmp:
        tmp_dir = Path(tmp)
        _job_dir, bundle_path, local_inputs, request_payload = _build_smoke_job(tmp_dir)
        try:
            submitted = client.submit_job(
                request_payload=request_payload,
                bundle_path=bundle_path,
                local_input_paths=local_inputs,
            )
            terminal = client.wait_for_job(job_id=str(submitted["job_id"]))
            if str(terminal.get("status") or "").strip().lower() != "succeeded":
                raise RuntimeError(f"Service job failed: {json.dumps(terminal, indent=2)}")
            destination = tmp_dir / "downloaded_result.json"
            client.download_artifact(
                job_id=str(submitted["job_id"]),
                artifact_name="result.json",
                destination=destination,
            )
            payload = json.loads(destination.read_text(encoding="utf-8"))
            print(json.dumps({"job_id": submitted["job_id"], "result": payload}, indent=2))
        finally:
            bundle_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
