#!/usr/bin/env python3
"""Verify logout clears both BisQue session and OIDC SSO session."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import urlencode, urljoin

import requests

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.dev.test_auth_browser_flow import (
    _expect,
    _extract_keycloak_form,
    _follow_redirects,
    _location,
    _request,
)


def _login_via_oidc(
    session: requests.Session,
    *,
    base_url: str,
    username: str,
    password: str,
    timeout: int,
) -> dict[str, object]:
    login_url = f"{base_url}/auth_service/login"
    login_response = _request(session, "GET", login_url, timeout=timeout, allow_redirects=False)
    _expect(login_response.status_code in {301, 302, 303, 307, 308}, "Expected /auth_service/login redirect")
    oidc_entry_url = _location(login_response, login_url)
    _expect("/auth_service/oidc_login" in oidc_entry_url, f"Unexpected login redirect target: {oidc_entry_url}")

    oidc_entry = _request(session, "GET", oidc_entry_url, timeout=timeout, allow_redirects=False)
    _expect(oidc_entry.status_code in {301, 302, 303, 307, 308}, "Expected /auth_service/oidc_login redirect")
    provider_url = _location(oidc_entry, oidc_entry_url)

    provider_login = _request(session, "GET", provider_url, timeout=timeout, allow_redirects=False)
    _expect(provider_login.status_code == 200, f"Expected IdP login page 200, got {provider_login.status_code}")
    action_url, payload = _extract_keycloak_form(provider_login.text, provider_login.url)
    payload["username"] = username
    payload["password"] = password

    provider_submit = _request(
        session,
        "POST",
        action_url,
        timeout=timeout,
        allow_redirects=False,
        data=payload,
    )
    final_response, redirect_chain = _follow_redirects(session, provider_submit, timeout=timeout)
    _expect(final_response.status_code == 200, f"Expected post-login HTTP 200, got {final_response.status_code}")

    return {
        "provider_url": provider_url,
        "redirect_chain": redirect_chain,
        "final_url": final_response.url,
    }


def run_logout_flow(
    *,
    base_url: str,
    username: str,
    password: str,
    logout_path: str,
    timeout: int,
) -> dict[str, object]:
    base_url = base_url.rstrip("/")
    if not logout_path.startswith("/"):
        logout_path = "/" + logout_path

    session = requests.Session()
    session.headers.update({"User-Agent": "bisque-auth-logout-flow/1.0"})

    login_report = _login_via_oidc(
        session,
        base_url=base_url,
        username=username,
        password=password,
        timeout=timeout,
    )

    whoami_before = _request(session, "GET", f"{base_url}/auth_service/whoami", timeout=timeout, allow_redirects=False)
    _expect(
        f'name="name" value="{username}"' in whoami_before.text,
        f"Expected authenticated whoami before logout for user={username}",
    )

    logout_url = f"{base_url}{logout_path}?{urlencode({'came_from': '/'})}"
    logout_response = _request(session, "GET", logout_url, timeout=timeout, allow_redirects=False)
    _expect(
        logout_response.status_code in {301, 302, 303, 307, 308},
        f"Expected logout redirect, got {logout_response.status_code}",
    )
    final_logout, logout_chain = _follow_redirects(session, logout_response, timeout=timeout)
    _expect(
        final_logout.status_code in {200, 302},
        f"Unexpected final logout status: {final_logout.status_code}",
    )

    whoami_after = _request(session, "GET", f"{base_url}/auth_service/whoami", timeout=timeout, allow_redirects=False)
    _expect(
        'name="name" value="anonymous"' in whoami_after.text,
        f"Expected anonymous whoami after logout, got: {whoami_after.text}",
    )

    relogin_start = _request(session, "GET", f"{base_url}/auth_service/login", timeout=timeout, allow_redirects=False)
    _expect(relogin_start.status_code in {301, 302, 303, 307, 308}, "Expected relogin redirect to OIDC")
    relogin_oidc = _request(
        session,
        "GET",
        _location(relogin_start, f"{base_url}/auth_service/login"),
        timeout=timeout,
        allow_redirects=False,
    )
    provider_url = _location(relogin_oidc, relogin_oidc.url)
    provider_page = _request(session, "GET", provider_url, timeout=timeout, allow_redirects=False)
    _expect(
        provider_page.status_code == 200 and "kc-form-login" in provider_page.text,
        "Expected Keycloak login form after logout; got silent SSO redirect",
    )

    return {
        "result": "PASS",
        "base_url": base_url,
        "logout_path": logout_path,
        "logout_chain": logout_chain,
        "logout_final_url": final_logout.url,
        "login_final_url": login_report["final_url"],
        "relogin_provider_url": provider_url,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://localhost:8080", help="BisQue base URL")
    parser.add_argument("--user", default="admin", help="OIDC username")
    parser.add_argument("--password", default="admin", help="OIDC password")
    parser.add_argument(
        "--logout-path",
        default="/auth_service/logout_handler",
        help="Logout endpoint path to validate",
    )
    parser.add_argument("--timeout-seconds", type=int, default=30, help="HTTP timeout seconds")
    parser.add_argument("--report-json", default="", help="Optional report output path")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        report = run_logout_flow(
            base_url=args.base,
            username=args.user,
            password=args.password,
            logout_path=args.logout_path,
            timeout=args.timeout_seconds,
        )
        if args.report_json:
            with open(args.report_json, "w", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2, sort_keys=True)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001
        error = {"result": "FAIL", "error": str(exc)}
        if args.report_json:
            with open(args.report_json, "w", encoding="utf-8") as stream:
                json.dump(error, stream, indent=2, sort_keys=True)
        print(json.dumps(error, indent=2, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
