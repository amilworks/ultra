#!/usr/bin/env python3
"""Validate OIDC browser auth flow end-to-end with a real redirect chain."""

from __future__ import annotations

import argparse
import json
import sys
from http.cookies import SimpleCookie
from urllib.parse import urljoin

import requests
from lxml import html


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _request(
    session: requests.Session,
    method: str,
    url: str,
    *,
    timeout: int,
    allow_redirects: bool = False,
    data: dict[str, str] | None = None,
) -> requests.Response:
    response = session.request(
        method=method,
        url=url,
        timeout=timeout,
        allow_redirects=allow_redirects,
        data=data,
    )
    return response


def _location(response: requests.Response, base_url: str) -> str:
    location = response.headers.get("Location", "")
    _expect(bool(location), f"Missing redirect location for {response.url}")
    return urljoin(base_url, location)


def _follow_redirects(
    session: requests.Session,
    response: requests.Response,
    *,
    timeout: int,
    max_hops: int = 20,
) -> tuple[requests.Response, list[str]]:
    chain = [response.url]
    current = response
    hops = 0
    while current.is_redirect or current.is_permanent_redirect:
        _promote_localhost_auth_cookie(session, current)
        hops += 1
        _expect(hops <= max_hops, f"Too many redirects (>{max_hops})")
        next_url = _location(current, current.url)
        current = _request(session, "GET", next_url, timeout=timeout, allow_redirects=False)
        chain.append(current.url)
    _promote_localhost_auth_cookie(session, current)
    return current, chain


def _extract_keycloak_form(page_html: str, current_url: str) -> tuple[str, dict[str, str]]:
    tree = html.fromstring(page_html)
    forms = tree.xpath("//form[@id='kc-form-login']") or tree.xpath(
        "//form[contains(@action, 'login-actions/authenticate')]"
    )
    _expect(bool(forms), "Could not find Keycloak login form")
    form = forms[0]
    action = form.attrib.get("action") or current_url
    action_url = urljoin(current_url, action)

    payload: dict[str, str] = {}
    for node in form.xpath(".//input[@name]"):
        name = node.attrib.get("name", "").strip()
        if not name:
            continue
        input_type = node.attrib.get("type", "text").strip().lower()
        if input_type in {"submit", "button", "image"}:
            continue
        payload[name] = node.attrib.get("value", "")
    return action_url, payload


def _promote_localhost_auth_cookie(session: requests.Session, response: requests.Response) -> None:
    set_cookie = response.headers.get("Set-Cookie", "")
    if not set_cookie or "authtkt=" not in set_cookie:
        return
    parsed = SimpleCookie()
    parsed.load(set_cookie)
    morsel = parsed.get("authtkt")
    if morsel is None:
        return
    domain = (morsel["domain"] or "").strip().lower().lstrip(".")
    path = (morsel["path"] or "/").strip() or "/"
    if domain == "localhost":
        session.cookies.set("authtkt", morsel.value, path=path)
        session.cookies.set("authtkt", morsel.value, domain="localhost", path=path)


def run_browser_flow(
    base_url: str,
    mode: str,
    username: str,
    password: str,
    came_from: str,
    module_name: str,
    timeout: int,
) -> dict[str, object]:
    base_url = base_url.rstrip("/")
    login_url = f"{base_url}/auth_service/login"
    session_url = f"{base_url}/auth_service/session"
    whoami_url = f"{base_url}/auth_service/whoami"
    module_url = f"{base_url}/module_service/{module_name}/?wpublic=1"

    session = requests.Session()
    session.headers.update({"User-Agent": "bisque-auth-browser-flow/1.0"})

    report: dict[str, object] = {
        "base_url": base_url,
        "mode": mode,
        "module": module_name,
        "login_url": login_url,
    }

    login_response = _request(session, "GET", login_url, timeout=timeout, allow_redirects=False)
    report["login_status"] = login_response.status_code

    if mode == "legacy":
        _expect(
            login_response.status_code == 200,
            f"Expected HTTP 200 in legacy mode, got {login_response.status_code}",
        )
        report["result"] = "PASS"
        report["detail"] = "legacy mode login page rendered"
        return report

    _expect(
        login_response.status_code in {301, 302, 303, 307, 308},
        f"Expected redirect for /auth_service/login in {mode} mode, got {login_response.status_code}",
    )
    first_redirect = _location(login_response, login_url)
    report["first_redirect"] = first_redirect
    _expect(
        "/auth_service/oidc_login" in first_redirect,
        f"Expected /auth_service/oidc_login redirect, got {first_redirect}",
    )

    oidc_entry = _request(session, "GET", first_redirect, timeout=timeout, allow_redirects=False)
    _expect(
        oidc_entry.status_code in {301, 302, 303, 307, 308},
        f"Expected redirect from oidc_login, got {oidc_entry.status_code}",
    )
    keycloak_url = _location(oidc_entry, first_redirect)
    report["oidc_provider_url"] = keycloak_url

    provider_login_page = _request(session, "GET", keycloak_url, timeout=timeout, allow_redirects=False)
    _expect(
        provider_login_page.status_code == 200,
        f"Expected Keycloak login page HTTP 200, got {provider_login_page.status_code}",
    )
    form_action, payload = _extract_keycloak_form(provider_login_page.text, provider_login_page.url)
    payload["username"] = username
    payload["password"] = password

    provider_submit = _request(
        session,
        "POST",
        form_action,
        timeout=timeout,
        allow_redirects=False,
        data=payload,
    )
    final_response, redirect_chain = _follow_redirects(
        session,
        provider_submit,
        timeout=timeout,
    )
    report["redirect_chain"] = redirect_chain
    report["final_url"] = final_response.url

    _expect(
        final_response.status_code == 200,
        f"Expected final post-login HTTP 200, got {final_response.status_code}",
    )
    _expect(
        final_response.url.startswith(base_url),
        f"Expected final URL under {base_url}, got {final_response.url}",
    )
    if came_from:
        _expect(
            came_from in final_response.url,
            f"Expected final URL to include came_from={came_from}, got {final_response.url}",
        )

    whoami_response = _request(session, "GET", whoami_url, timeout=timeout, allow_redirects=False)
    _expect(
        whoami_response.status_code == 200,
        f"Expected /auth_service/whoami HTTP 200, got {whoami_response.status_code}",
    )
    _expect(
        f'name="name" value="{username}"' in whoami_response.text,
        f"Expected whoami response to include user={username}",
    )
    report["whoami_user"] = username

    session_response = _request(session, "GET", session_url, timeout=timeout, allow_redirects=False)
    _expect(
        session_response.status_code == 200,
        f"Expected /auth_service/session HTTP 200, got {session_response.status_code}",
    )
    _expect(
        'name="user"' in session_response.text,
        "Expected /auth_service/session response to include user tag",
    )

    module_response = _request(session, "GET", module_url, timeout=timeout, allow_redirects=False)
    _expect(
        module_response.status_code in {200, 301, 302, 303, 307, 308},
        f"Unexpected module page status: {module_response.status_code}",
    )
    if module_response.is_redirect or module_response.is_permanent_redirect:
        module_location = _location(module_response, module_url)
        _expect(
            "/auth_service/login" not in module_location,
            f"Module page redirected back to login: {module_location}",
        )
        report["module_redirect"] = module_location
    report["module_status"] = module_response.status_code
    report["result"] = "PASS"
    return report


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://localhost:8080", help="BisQue base URL")
    parser.add_argument(
        "--mode",
        default="dual",
        choices=["legacy", "dual", "oidc"],
        help="Expected auth mode",
    )
    parser.add_argument("--user", default="admin", help="OIDC user")
    parser.add_argument("--password", default="admin", help="OIDC user password")
    parser.add_argument(
        "--came-from",
        default="/client_service/",
        help="Expected post-login path fragment",
    )
    parser.add_argument(
        "--module",
        default="SimpleUniversalProcess",
        help="Module name to probe after login",
    )
    parser.add_argument("--timeout-seconds", type=int, default=30, help="HTTP timeout seconds")
    parser.add_argument(
        "--report-json",
        default="",
        help="Optional path to write machine-readable report JSON",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        report = run_browser_flow(
            base_url=args.base,
            mode=args.mode,
            username=args.user,
            password=args.password,
            came_from=args.came_from,
            module_name=args.module,
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
