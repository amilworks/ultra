from __future__ import annotations

from pathlib import Path

from src.config import Settings
from src.tooling.code_execution_service_client import CodeExecutionServiceClient


def test_settings_expose_codeexec_service_fields() -> None:
    settings = Settings(
        _env_file=None,
        code_execution_service_url="http://codeexec.internal:8020",
        code_execution_service_api_key="secret-token",
        code_execution_service_timeout_seconds=45,
        code_execution_service_poll_interval_seconds=1.5,
        code_execution_service_wait_timeout_seconds=7200,
    )

    assert settings.code_execution_service_url == "http://codeexec.internal:8020"
    assert settings.code_execution_service_api_key == "secret-token"
    assert settings.code_execution_service_timeout_seconds == 45
    assert settings.code_execution_service_poll_interval_seconds == 1.5
    assert settings.code_execution_service_wait_timeout_seconds == 7200


def test_service_client_builds_authenticated_requests(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, str, dict]] = []
    bundle_path = tmp_path / "job.tar.gz"
    bundle_path.write_bytes(b"fake-bundle")

    class DummyResponse:
        def __init__(self, payload: dict):
            self._payload = payload
            self.content = b'{"accuracy": 0.95}'

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return self._payload

    class DummyClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def request(self, method: str, url: str, headers: dict, **kwargs):
            calls.append((method, url, {"headers": headers, **kwargs}))
            return DummyResponse({"job_id": "job-1", "status": "queued"})

    monkeypatch.setattr("src.tooling.code_execution_service_client.httpx.Client", DummyClient)
    client = CodeExecutionServiceClient(
        base_url="http://codeexec.internal:8020",
        api_key="secret-token",
        timeout_seconds=30,
    )

    response = client.submit_job(
        request_payload={"job_id": "job-1", "timeout_seconds": 60},
        bundle_path=bundle_path,
        local_input_paths=[],
    )

    assert response["job_id"] == "job-1"
    assert calls[0][0] == "POST"
    assert calls[0][2]["headers"]["Authorization"] == "Bearer secret-token"
