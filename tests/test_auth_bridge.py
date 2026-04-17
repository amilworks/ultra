from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from fastapi.testclient import TestClient
from src.api import main as api_main


@contextmanager
def _test_client(
    monkeypatch,
    *,
    environment: str = "development",
    allow_query_api_key_compat: bool | None = None,
    orchestrator_api_key: str = "bridge-secret",
) -> Iterator[TestClient]:
    settings = api_main.get_settings()
    monkeypatch.setattr(settings, "environment", environment, raising=False)
    monkeypatch.setattr(settings, "orchestrator_api_key", orchestrator_api_key, raising=False)
    monkeypatch.setattr(
        settings,
        "allow_query_api_key_compat",
        allow_query_api_key_compat,
        raising=False,
    )
    monkeypatch.setattr(settings, "bisque_root", "https://bisque.example.org", raising=False)
    monkeypatch.setattr(settings, "bisque_auth_oidc_enabled", False, raising=False)
    monkeypatch.setattr(settings, "bisque_auth_oidc_via_bisque_login", False, raising=False)

    app = api_main.create_app()
    with TestClient(app) as client:
        yield client


def test_auth_session_route_no_longer_requires_api_key(monkeypatch) -> None:
    with _test_client(monkeypatch) as client:
        response = client.get("/v1/auth/session")

    assert response.status_code == 200
    assert response.json()["authenticated"] is False
    assert response.json()["mode"] is None
    assert response.headers.get("X-Request-Id")


def test_query_param_api_key_compat_is_allowed_in_development(monkeypatch) -> None:
    with _test_client(
        monkeypatch,
        environment="development",
        allow_query_api_key_compat=None,
    ) as client:
        response = client.get("/v1/fun/weather/santa-barbara?api_key=bridge-secret")

    assert response.status_code == 200
    assert response.headers.get("Warning")
    assert response.headers.get("X-Request-Id")


def test_query_param_api_key_compat_is_rejected_in_production(monkeypatch) -> None:
    with _test_client(
        monkeypatch,
        environment="production",
        allow_query_api_key_compat=None,
    ) as client:
        response = client.get("/v1/fun/weather/santa-barbara?api_key=bridge-secret")

    assert response.status_code == 401
    assert response.json()["detail"] == "Unauthorized"


def test_browser_asset_routes_require_an_authenticated_session(monkeypatch) -> None:
    with _test_client(monkeypatch) as client:
        api_key_only = client.get(
            "/v1/resources/file-123/thumbnail",
            headers={"X-API-Key": "bridge-secret"},
        )
        guest_session = client.post(
            "/v1/auth/guest",
            json={
                "name": "Guest User",
                "email": "guest@example.org",
                "affiliation": "Bisque Ultra QA",
            },
        )
        guest_thumbnail = client.get("/v1/resources/file-123/thumbnail")

    assert api_key_only.status_code == 401
    assert guest_session.status_code == 200
    assert guest_session.json()["authenticated"] is True
    assert guest_session.json()["mode"] == "guest"
    assert guest_thumbnail.status_code == 404


def test_admin_routes_require_a_signed_in_admin_session(monkeypatch) -> None:
    with _test_client(monkeypatch) as client:
        api_key_only = client.get(
            "/v1/admin/overview",
            headers={"X-API-Key": "bridge-secret"},
        )
        guest_login = client.post(
            "/v1/auth/guest",
            json={
                "name": "Guest User",
                "email": "guest@example.org",
                "affiliation": "Bisque Ultra QA",
            },
        )
        guest_admin = client.get("/v1/admin/overview")

    assert api_key_only.status_code == 401
    assert guest_login.status_code == 200
    assert guest_admin.status_code == 403
    assert "signed-in bisque account" in str(guest_admin.json()["detail"]).lower()
