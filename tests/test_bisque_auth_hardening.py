from __future__ import annotations

from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient
from src.api import main as api_main
from starlette.requests import Request


def _api_headers() -> dict[str, str]:
    api_key = str(getattr(api_main.get_settings(), "orchestrator_api_key", "") or "").strip()
    return {"x-api-key": api_key} if api_key else {}


def test_public_request_origin_prefers_forwarded_headers() -> None:
    request = Request(
        {
            "type": "http",
            "scheme": "http",
            "method": "GET",
            "path": "/v1/auth/oidc/start",
            "raw_path": b"/v1/auth/oidc/start",
            "query_string": b"",
            "headers": [
                (b"host", b"internal-api:8000"),
                (b"x-forwarded-proto", b"https"),
                (b"x-forwarded-host", b"ultra.example.org"),
            ],
            "client": ("127.0.0.1", 12345),
            "server": ("internal-api", 8000),
        }
    )

    assert api_main._public_request_origin(request) == "https://ultra.example.org"


def test_build_bisque_links_uses_configured_root_for_user_facing_links() -> None:
    links = api_main._build_bisque_links_for_root(
        "http://internal-bisque:8080/data_service/00-SHAqHM6FMyPSub43wTiLWf",
        "https://bisque.example.org",
    )

    assert (
        links["resource_uri"] == "https://bisque.example.org/data_service/00-SHAqHM6FMyPSub43wTiLWf"
    )
    assert (
        links["client_view_url"]
        == "https://bisque.example.org/client_service/view?resource=https://bisque.example.org/data_service/00-SHAqHM6FMyPSub43wTiLWf"
    )
    assert (
        links["image_service_url"]
        == "https://bisque.example.org/image_service/00-SHAqHM6FMyPSub43wTiLWf"
    )


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    settings = api_main.get_settings()
    monkeypatch.setattr(settings, "bisque_root", "https://bisque.example.org", raising=False)
    monkeypatch.setattr(settings, "bisque_user", "service-user", raising=False)
    monkeypatch.setattr(settings, "bisque_password", "service-pass", raising=False)
    monkeypatch.setattr(settings, "bisque_auth_oidc_enabled", True, raising=False)
    monkeypatch.setattr(settings, "bisque_auth_oidc_via_bisque_login", False, raising=False)
    monkeypatch.setattr(settings, "bisque_auth_oidc_redirect_uri", None, raising=False)
    monkeypatch.setattr(
        settings,
        "bisque_auth_oidc_authorize_url",
        "https://idp.example.org/authorize",
        raising=False,
    )
    monkeypatch.setattr(
        settings,
        "bisque_auth_oidc_issuer_url",
        "https://idp.example.org/realms/ultra",
        raising=False,
    )
    monkeypatch.setattr(
        settings,
        "bisque_auth_oidc_client_id",
        "ultra-client",
        raising=False,
    )
    with TestClient(api_main.app) as test_client:
        yield test_client


def test_uploads_from_bisque_requires_authenticated_user_session(client: TestClient) -> None:
    response = client.post(
        "/v1/uploads/from-bisque",
        headers=_api_headers(),
        json={"resources": ["https://internal-bisque:8080/data_service/00-SHAqHM6FMyPSub43wTiLWf"]},
    )

    assert response.status_code == 401
    assert "authentication is required" in str(response.json().get("detail", "")).lower()


def test_oidc_start_uses_forwarded_public_origin_for_callback(client: TestClient) -> None:
    response = client.get(
        "/v1/auth/oidc/start",
        headers={
            **_api_headers(),
            "host": "internal-api:8000",
            "x-forwarded-host": "api.ultra.example.org",
            "x-forwarded-proto": "https",
        },
        follow_redirects=False,
    )

    assert response.status_code == 302
    parsed = urlparse(response.headers["location"])
    params = parse_qs(parsed.query)
    assert params["redirect_uri"] == ["https://api.ultra.example.org/v1/auth/oidc/callback"]
