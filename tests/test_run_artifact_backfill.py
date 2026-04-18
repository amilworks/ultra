from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from fastapi.testclient import TestClient

from src.api import main as api_main
from src.orchestration.models import RunStatus, WorkflowRun
from src.orchestration.store import RunStore


@contextmanager
def _test_client(
    monkeypatch,
    *,
    artifact_root: Path,
    run_store: RunStore,
    environment: str = "development",
    orchestrator_api_key: str = "bridge-secret",
) -> Iterator[TestClient]:
    settings = api_main.get_settings()
    monkeypatch.setattr(settings, "environment", environment, raising=False)
    monkeypatch.setattr(settings, "orchestrator_api_key", orchestrator_api_key, raising=False)
    monkeypatch.setattr(settings, "artifact_root", str(artifact_root), raising=False)
    monkeypatch.setattr(settings, "bisque_root", "https://bisque.example.org", raising=False)
    monkeypatch.setattr(settings, "bisque_auth_oidc_enabled", False, raising=False)
    monkeypatch.setattr(settings, "bisque_auth_oidc_via_bisque_login", False, raising=False)
    monkeypatch.setattr(api_main, "RunStore", lambda _path: run_store)

    app = api_main.create_app()
    with TestClient(app) as client:
        yield client


def test_list_artifacts_backfills_megaseg_progress_outputs_from_chat_done_payload(
    monkeypatch, tmp_path: Path
) -> None:
    artifact_root = tmp_path / "artifacts"
    run_store = RunStore(str(tmp_path / "runs.db"))
    external_output_dir = tmp_path / "science" / "megaseg_results" / "example"
    external_output_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = external_output_dir / "example__megaseg_overlay_mip.png"
    overlay_bytes = b"\x89PNG\r\n\x1a\nmegaseg"
    overlay_path.write_bytes(overlay_bytes)

    run = WorkflowRun.new(goal="Backfill stale megaseg artifacts")
    run.status = RunStatus.SUCCEEDED
    run_store.create_run(run)
    run_store.set_run_metadata(run.run_id, user_id=None, conversation_id="conv-backfill")

    run_dir = artifact_root / run.run_id
    tool_output_dir = run_dir / "tool_outputs"
    tool_output_dir.mkdir(parents=True, exist_ok=True)
    report_path = tool_output_dir / "megaseg_report.md"
    report_path.write_text("# report\n", encoding="utf-8")
    (run_dir / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "run_id": run.run_id,
                "artifacts": [
                    {
                        "path": "tool_outputs/megaseg_report.md",
                        "size_bytes": report_path.stat().st_size,
                        "mime_type": "text/markdown",
                        "modified_at": "2026-04-17T00:00:00Z",
                        "source_path": str(report_path),
                        "title": "Megaseg report",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    run_store.append_event(
        run.run_id,
        "chat_done_payload",
        {
            "response": {
                "progress_events": [
                    {
                        "event": "completed",
                        "tool": "segment_image_megaseg",
                        "summary": {
                            "success": True,
                            "result_group_id": "megaseg-group-1",
                            "output_directory": str(external_output_dir),
                        },
                        "artifacts": [
                            {
                                "path": str(overlay_path),
                                "title": "Megaseg overlay (MIP)",
                                "result_group_id": "megaseg-group-1",
                            }
                        ],
                    }
                ]
            }
        },
    )

    with _test_client(monkeypatch, artifact_root=artifact_root, run_store=run_store) as client:
        response = client.get(
            f"/v1/artifacts/{run.run_id}",
            headers={"X-API-Key": "bridge-secret"},
        )

        assert response.status_code == 200
        payload = response.json()
        overlay_artifact = next(
            artifact
            for artifact in payload["artifacts"]
            if artifact.get("source_path") == str(overlay_path)
        )
        assert str(overlay_artifact["path"]).startswith("tool_outputs/")
        assert overlay_artifact["source_path"] == str(overlay_path)

        copied_overlay_path = artifact_root / run.run_id / overlay_artifact["path"]
        assert copied_overlay_path.exists()
        assert copied_overlay_path.read_bytes() == overlay_bytes

        download = client.get(
            f"/v1/artifacts/{run.run_id}/download",
            params={"path": overlay_artifact["path"]},
            headers={"X-API-Key": "bridge-secret"},
        )

        assert download.status_code == 200
        assert download.content == overlay_bytes
