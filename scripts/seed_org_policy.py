#!/usr/bin/env python3
"""Seed an organization's read-only policy memory.

Writes ``<policies_root>/<org_id>/lab_policy.md`` from a markdown file, so every
member of the org sees the same authoritative lab policy at the start of each
run. The agent can read it but never edit it (writes under ``/policies/`` are
denied by MEMORY_PERMISSIONS); only this script / application code populates it.

Usage:
    python scripts/seed_org_policy.py --org-id workos-org --file lab_policy.md
    ULTRA_DEEPAGENTS_POLICIES_ROOT=/srv/ultra/shared/deepagents/policies \\
        python scripts/seed_org_policy.py --org-id <org> --file <policy.md>
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path


def _org_slug(org_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", org_id).strip("_.")
    return slug[:128]


def _policies_root() -> Path:
    explicit = os.getenv("ULTRA_DEEPAGENTS_POLICIES_ROOT", "").strip()
    if explicit:
        return Path(explicit)
    memory_root = os.getenv("ULTRA_DEEPAGENTS_MEMORY_ROOT", "data/deepagents/memory").strip()
    return Path(memory_root) / "policies"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--org-id", required=True, help="Organization id (e.g. workos-org).")
    parser.add_argument("--file", required=True, help="Markdown file with the lab policy.")
    parser.add_argument(
        "--name",
        default="lab_policy.md",
        help="Policy filename under the org directory (default: lab_policy.md).",
    )
    args = parser.parse_args(argv)

    org_slug = _org_slug(args.org_id)
    if not org_slug:
        print("Invalid --org-id", file=sys.stderr)
        return 2
    source = Path(args.file).expanduser()
    if not source.is_file():
        print(f"Policy file not found: {source}", file=sys.stderr)
        return 2
    content = source.read_text(encoding="utf-8")

    target_dir = _policies_root() / org_slug
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / args.name
    target.write_text(content, encoding="utf-8")
    print(f"Seeded org policy: {target} ({len(content)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
