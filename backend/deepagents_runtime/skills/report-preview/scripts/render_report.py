#!/usr/bin/env python3
"""Render an HTML report headlessly and capture what a reader would see.

Produces, under /workspace/diagnostics/report_preview/ (scratch, never
/outputs — previews are diagnostics, not deliverables):
  <name>.png           full-page screenshot to inspect with inspect_images
  <name>.console.json  console messages, page errors, failed/blocked requests

The page renders from file:// with network access DISABLED except same-file
resources, mirroring the reading sandbox: the platform renders reports inside
a frame whose CSP blocks all outbound requests, so anything external that
"works" here with network on would still be broken for the reader.

Exit codes: 0 rendered clean · 1 usage/input error · 2 rendered WITH findings
(console errors or blocked/missing resources — read the .console.json) ·
3 no browser engine in this sandbox image (preview unavailable; say so and
move on rather than retrying).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PREVIEW_DIR = Path("/workspace/diagnostics/report_preview")
VIEWPORT = {"width": 1280, "height": 900}
RENDER_SETTLE_MS = 1200

_SRC_ATTR = re.compile(r"""(<img\b[^>]*?\bsrc=)(["'])([^"']+)\2""", re.IGNORECASE)


def remap_figure_sources(html: str, report_dir: Path) -> str:
    """Resolve figure references the way the platform does.

    Reports live IN outputs/ but conventionally reference figures as
    outputs/fig.png; the platform matcher forgives that, file:// does not.
    For each <img src> that is not data:/blob:/http(s), try the raw path and
    the path with a leading "outputs/" stripped, relative to the report;
    rewrite to the first that exists so the preview only flags figures that
    are GENUINELY missing.
    """

    def rewrite(match: re.Match[str]) -> str:
        prefix, quote, src = match.group(1), match.group(2), match.group(3)
        if re.match(r"^(data:|blob:|https?:|file:)", src, re.IGNORECASE):
            return match.group(0)
        candidates = [src]
        stripped = re.sub(r"^(\./)?outputs/", "", src)
        if stripped != src:
            candidates.append(stripped)
        for candidate in candidates:
            if (report_dir / candidate).is_file():
                return f"{prefix}{quote}{candidate}{quote}"
        return match.group(0)

    return _SRC_ATTR.sub(rewrite, html)


def fail_usage(message: str) -> int:
    print(f"render_report: {message}", file=sys.stderr)
    print(__doc__, file=sys.stderr)
    return 1


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        return fail_usage("usage: render_report.py <path/to/report.html>")
    report_path = Path(argv[1]).expanduser().resolve()
    if not report_path.is_file():
        return fail_usage(f"no such file: {report_path}")

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(
            "render_report: this sandbox image has no browser engine "
            "(playwright/chromium not installed). Report preview is "
            "unavailable here — state that in one line and continue; do not "
            "retry or attempt to install anything.",
            file=sys.stderr,
        )
        return 3

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    stem = report_path.stem or "report"
    shot_path = PREVIEW_DIR / f"{stem}.png"
    log_path = PREVIEW_DIR / f"{stem}.console.json"

    # Render a sibling copy with figure references resolved (see
    # remap_figure_sources); sibling placement keeps every relative
    # reference the report legitimately makes working.
    render_copy = report_path.parent / f".preview.{report_path.name}"
    render_copy.write_text(
        remap_figure_sources(
            report_path.read_text(encoding="utf-8", errors="replace"),
            report_path.parent,
        ),
        encoding="utf-8",
    )

    console_entries: list[dict[str, str]] = []
    page_errors: list[str] = []
    blocked_requests: list[str] = []
    failed_requests: list[str] = []

    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(args=["--disable-gpu"])
        except Exception as error:  # noqa: BLE001 - single actionable message
            print(
                "render_report: chromium failed to launch: "
                f"{error}. Preview unavailable in this sandbox.",
                file=sys.stderr,
            )
            return 3
        page = browser.new_page(viewport=VIEWPORT)

        # Parity with the reading sandbox: block every request that is not
        # the report file itself or a sibling file:// resource.
        def route_handler(route):  # type: ignore[no-untyped-def]
            url = route.request.url
            if url.startswith("file://"):
                route.continue_()
                return
            blocked_requests.append(url)
            route.abort()

        page.route("**/*", route_handler)
        page.on(
            "console",
            lambda message: console_entries.append(
                {"type": message.type, "text": message.text}
            ),
        )
        page.on("pageerror", lambda error: page_errors.append(str(error)))
        page.on(
            "requestfailed",
            lambda request: failed_requests.append(
                f"{request.url} — {request.failure}"
            ),
        )

        page.goto(render_copy.as_uri(), wait_until="load")
        page.wait_for_timeout(RENDER_SETTLE_MS)

        # Broken images are the most common report defect (a figure path that
        # never resolved); surface them explicitly instead of hoping the
        # screenshot makes them obvious.
        broken_images = page.evaluate(
            "() => [...document.images]"
            ".filter((img) => !(img.complete && img.naturalWidth > 0))"
            ".map((img) => img.getAttribute('src') || '(no src)')"
        )
        page.screenshot(path=str(shot_path), full_page=True)
        browser.close()
    render_copy.unlink(missing_ok=True)

    findings = {
        "console_errors": [
            entry for entry in console_entries if entry["type"] == "error"
        ],
        "page_errors": page_errors,
        "broken_images": broken_images,
        "blocked_external_requests": blocked_requests,
        "failed_requests": failed_requests,
    }
    log_path.write_text(
        json.dumps(
            {"report": str(report_path), "viewport": VIEWPORT, **findings},
            indent=2,
        )
    )

    has_findings = any(
        findings[key]
        for key in ("console_errors", "page_errors", "broken_images")
    )
    print(f"screenshot: {shot_path}")
    print(f"console log: {log_path}")
    if has_findings:
        print(
            "findings: "
            f"{len(findings['console_errors'])} console error(s), "
            f"{len(findings['page_errors'])} page error(s), "
            f"{len(findings['broken_images'])} broken image(s) — "
            "read the console log and fix the report before finishing."
        )
        return 2
    if blocked_requests:
        print(
            f"note: {len(blocked_requests)} external request(s) blocked "
            "(the reader's sandbox blocks them too); the report should not "
            "depend on them."
        )
    print("render clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
