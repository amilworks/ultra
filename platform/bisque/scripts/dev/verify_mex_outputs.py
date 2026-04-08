#!/usr/bin/env python3
"""Validate output resources from bqapi module runner summaries."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from urllib.parse import urljoin

import requests

DATA_SERVICE_RE = re.compile(r"/data_service/(00-[A-Za-z0-9]+)")


def _extract_resource_id(uri: str) -> str | None:
    if not uri:
        return None
    match = DATA_SERVICE_RE.search(uri)
    if match:
        return match.group(1)
    if uri.startswith("00-"):
        return uri.split("?", 1)[0]
    return None


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


def _check_url(
    session: requests.Session,
    url: str,
    timeout: int,
    *,
    basic_auth: tuple[str, str] | None,
    token: str,
) -> tuple[int, str]:
    response = session.get(
        url,
        timeout=timeout,
        auth=basic_auth,
        headers=_headers(token),
        allow_redirects=True,
    )
    return response.status_code, response.headers.get("Content-Type", "")


def verify_summary(
    summary_path: Path,
    base_url: str,
    timeout: int,
    user: str,
    password: str,
    token: str,
) -> dict[str, object]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    results = payload.get("results", [])
    if not isinstance(results, list):
        raise RuntimeError("Summary JSON is missing a valid `results` list")

    resource_ids: list[str] = []
    for row in results:
        if not isinstance(row, dict):
            continue
        rid = _extract_resource_id(str(row.get("output_image_uri", "")))
        if rid:
            resource_ids.append(rid)
    unique_ids = list(dict.fromkeys(resource_ids))
    if not unique_ids:
        raise RuntimeError("No output_image_uri values found in summary")

    session = requests.Session()
    base_url = base_url.rstrip("/")
    basic_auth = None if token else (user, password)

    checks: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    for rid in unique_ids:
        endpoints = {
            "data_short": f"{base_url}/data_service/{rid}?view=short",
            "image_meta": f"{base_url}/image_service/{rid}?meta",
            "image_thumbnail": f"{base_url}/image_service/{rid}?thumbnail=280,280",
        }
        for name, url in endpoints.items():
            code, content_type = _check_url(
                session,
                url,
                timeout,
                basic_auth=basic_auth,
                token=token,
            )
            row = {
                "resource_id": rid,
                "check": name,
                "url": url,
                "status_code": code,
                "content_type": content_type,
                "ok": code == 200,
            }
            checks.append(row)
            if code != 200:
                failures.append(row)

    return {
        "summary": str(summary_path),
        "base_url": base_url,
        "resources_checked": unique_ids,
        "checks_total": len(checks),
        "checks_failed": len(failures),
        "ok": not failures,
        "checks": checks,
        "failures": failures,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, help="Path to bqapi module runner summary JSON")
    parser.add_argument("--base", default="http://127.0.0.1:8080", help="BisQue base URL")
    parser.add_argument("--user", default="admin", help="Basic auth username")
    parser.add_argument("--password", default="admin", help="Basic auth password")
    parser.add_argument("--token", default="", help="Bearer token (optional, overrides basic)")
    parser.add_argument("--timeout-seconds", type=int, default=30, help="HTTP timeout")
    parser.add_argument("--report-json", default="", help="Optional output report path")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    summary_path = Path(args.summary)
    if not summary_path.exists():
        print(json.dumps({"result": "FAIL", "error": f"Missing summary: {summary_path}"}, indent=2))
        return 1

    try:
        report = verify_summary(
            summary_path=summary_path,
            base_url=args.base,
            timeout=args.timeout_seconds,
            user=args.user,
            password=args.password,
            token=args.token.strip(),
        )
        if args.report_json:
            Path(args.report_json).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report.get("ok") else 1
    except Exception as exc:  # noqa: BLE001
        error = {"result": "FAIL", "error": str(exc)}
        if args.report_json:
            Path(args.report_json).write_text(json.dumps(error, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(error, indent=2, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
