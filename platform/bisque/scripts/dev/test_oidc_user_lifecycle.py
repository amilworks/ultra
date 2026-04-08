#!/usr/bin/env python3
"""Verify OIDC user lifecycle: create, login success, delete, login failure."""

from __future__ import annotations

import argparse
import json
import secrets
import string
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import requests

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.dev.test_auth_browser_flow import (  # noqa: E402
    _expect,
    _extract_keycloak_form,
    _follow_redirects,
    _location,
    _request,
)


def _rand_token(n: int = 6) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(n))


@dataclass
class KeycloakAdmin:
    base_url: str
    realm: str
    username: str
    password: str
    timeout: int
    _token: str | None = None

    def _headers(self) -> dict[str, str]:
        if not self._token:
            self._token = self.login()
        return {"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"}

    def login(self) -> str:
        token_url = f"{self.base_url.rstrip('/')}/realms/master/protocol/openid-connect/token"
        response = requests.post(
            token_url,
            data={
                "grant_type": "password",
                "client_id": "admin-cli",
                "username": self.username,
                "password": self.password,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        token = payload.get("access_token", "")
        _expect(bool(token), "Keycloak admin login did not return access_token")
        return token

    def create_user(self, username: str, password: str, email: str) -> str:
        users_url = f"{self.base_url.rstrip('/')}/admin/realms/{self.realm}/users"
        payload = {
            "username": username,
            "enabled": True,
            "emailVerified": True,
            "email": email,
            "firstName": "Soak",
            "lastName": "User",
            "requiredActions": [],
            "credentials": [{"type": "password", "value": password, "temporary": False}],
        }
        create_response = requests.post(
            users_url,
            headers=self._headers(),
            data=json.dumps(payload),
            timeout=self.timeout,
        )
        if create_response.status_code not in {201, 409}:
            raise RuntimeError(
                f"Keycloak create user failed status={create_response.status_code}: {create_response.text[:400]}"
            )
        user_id = self.find_user_id(username)
        _expect(bool(user_id), f"Keycloak did not return created user id for {username}")
        return user_id

    def find_user_id(self, username: str) -> str:
        users_url = f"{self.base_url.rstrip('/')}/admin/realms/{self.realm}/users"
        response = requests.get(
            users_url,
            headers=self._headers(),
            params={"username": username, "exact": "true"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            return ""
        for item in payload:
            if item.get("username") == username and item.get("id"):
                return str(item["id"])
        return ""

    def delete_user(self, user_id: str) -> None:
        delete_url = f"{self.base_url.rstrip('/')}/admin/realms/{self.realm}/users/{user_id}"
        response = requests.delete(delete_url, headers=self._headers(), timeout=self.timeout)
        if response.status_code not in {204, 404}:
            raise RuntimeError(
                f"Keycloak delete user failed status={response.status_code}: {response.text[:400]}"
            )


def _oidc_login(
    *,
    base_url: str,
    username: str,
    password: str,
    timeout: int,
) -> tuple[requests.Session, requests.Response]:
    session = requests.Session()
    session.headers.update({"User-Agent": "bisque-oidc-user-lifecycle/1.0"})

    login_url = f"{base_url.rstrip('/')}/auth_service/login"
    login_response = _request(session, "GET", login_url, timeout=timeout, allow_redirects=False)
    _expect(
        login_response.status_code in {301, 302, 303, 307, 308},
        f"Expected redirect from /auth_service/login, got {login_response.status_code}",
    )

    oidc_entry_url = _location(login_response, login_url)
    _expect("/auth_service/oidc_login" in oidc_entry_url, f"Unexpected login redirect: {oidc_entry_url}")
    oidc_entry = _request(session, "GET", oidc_entry_url, timeout=timeout, allow_redirects=False)
    _expect(
        oidc_entry.status_code in {301, 302, 303, 307, 308},
        f"Expected redirect from /auth_service/oidc_login, got {oidc_entry.status_code}",
    )

    keycloak_url = _location(oidc_entry, oidc_entry_url)
    provider_page = _request(session, "GET", keycloak_url, timeout=timeout, allow_redirects=False)
    _expect(provider_page.status_code == 200, f"Expected Keycloak login page, got {provider_page.status_code}")

    action_url, form_payload = _extract_keycloak_form(provider_page.text, provider_page.url)
    form_payload["username"] = username
    form_payload["password"] = password
    submit = _request(
        session,
        "POST",
        action_url,
        timeout=timeout,
        allow_redirects=False,
        data=form_payload,
    )

    final_response, _ = _follow_redirects(session, submit, timeout=timeout)
    return session, final_response


def _assert_login_success(base_url: str, username: str, password: str, timeout: int) -> dict[str, object]:
    session, final_response = _oidc_login(
        base_url=base_url,
        username=username,
        password=password,
        timeout=timeout,
    )
    _expect(final_response.status_code == 200, f"Expected login success status 200, got {final_response.status_code}")
    _expect(
        final_response.url.startswith(base_url.rstrip("/")),
        f"Expected successful login to return to BisQue URL, got {final_response.url}",
    )
    whoami = _request(
        session, "GET", f"{base_url.rstrip('/')}/auth_service/whoami", timeout=timeout, allow_redirects=False
    )
    _expect(whoami.status_code == 200, f"Expected whoami status 200, got {whoami.status_code}")
    _expect(
        f'name="name" value="{username}"' in whoami.text,
        f"Expected whoami name={username}, got body={whoami.text}",
    )
    return {"result": "PASS", "final_url": final_response.url}


def _assert_login_failure(base_url: str, username: str, password: str, timeout: int) -> dict[str, object]:
    session, final_response = _oidc_login(
        base_url=base_url,
        username=username,
        password=password,
        timeout=timeout,
    )
    login_form_present = "kc-form-login" in final_response.text
    error_text_present = (
        "Invalid username or password" in final_response.text
        or "invalid username or password" in final_response.text.lower()
        or "id=\"input-error\"" in final_response.text
    )
    stayed_on_provider = "/realms/" in final_response.url and "/login-actions/" in final_response.url
    _expect(
        login_form_present and error_text_present and stayed_on_provider,
        "Expected Keycloak login failure page with invalid credentials error",
    )

    whoami = _request(
        session, "GET", f"{base_url.rstrip('/')}/auth_service/whoami", timeout=timeout, allow_redirects=False
    )
    _expect(whoami.status_code == 200, f"Expected whoami status 200, got {whoami.status_code}")
    _expect(
        'name="name" value="anonymous"' in whoami.text,
        "Expected anonymous whoami after failed login",
    )
    return {"result": "PASS", "final_url": final_response.url}


def run_lifecycle(
    *,
    base_url: str,
    keycloak_base: str,
    realm: str,
    keycloak_admin_user: str,
    keycloak_admin_password: str,
    timeout: int,
) -> dict[str, object]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    username = f"soak_{timestamp}_{_rand_token(4)}"
    password = f"SoakPass-{_rand_token(10)}"
    email = f"{username}@local.invalid"

    report: dict[str, object] = {
        "base_url": base_url,
        "keycloak_base": keycloak_base,
        "realm": realm,
        "username": username,
    }

    keycloak = KeycloakAdmin(
        base_url=keycloak_base,
        realm=realm,
        username=keycloak_admin_user,
        password=keycloak_admin_password,
        timeout=timeout,
    )

    user_id = ""
    try:
        user_id = keycloak.create_user(username=username, password=password, email=email)
        report["created_user_id"] = user_id
        report["create_user"] = "PASS"

        login_ok = _assert_login_success(base_url=base_url, username=username, password=password, timeout=timeout)
        report["login_after_create"] = login_ok

        keycloak.delete_user(user_id)
        report["delete_user"] = "PASS"

        login_fail = _assert_login_failure(base_url=base_url, username=username, password=password, timeout=timeout)
        report["login_after_delete"] = login_fail

        report["result"] = "PASS"
        return report
    except Exception as exc:  # noqa: BLE001
        report["result"] = "FAIL"
        report["error"] = str(exc)
        return report
    finally:
        if user_id:
            try:
                keycloak.delete_user(user_id)
            except Exception:
                pass


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://localhost:8080", help="BisQue base URL")
    parser.add_argument("--keycloak-base", default="http://localhost:18080", help="Keycloak base URL")
    parser.add_argument("--realm", default="bisque", help="Keycloak realm used by BisQue")
    parser.add_argument("--keycloak-admin-user", default="admin", help="Keycloak admin username")
    parser.add_argument("--keycloak-admin-password", default="admin", help="Keycloak admin password")
    parser.add_argument("--timeout-seconds", type=int, default=30, help="HTTP timeout")
    parser.add_argument("--report-json", default="", help="Optional path to write JSON report")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    report = run_lifecycle(
        base_url=args.base,
        keycloak_base=args.keycloak_base,
        realm=args.realm,
        keycloak_admin_user=args.keycloak_admin_user,
        keycloak_admin_password=args.keycloak_admin_password,
        timeout=args.timeout_seconds,
    )
    if args.report_json:
        Path(args.report_json).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("result") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
