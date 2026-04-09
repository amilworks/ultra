#!/usr/bin/env python3
"""Render production nginx templates from server-side env files."""

from __future__ import annotations

import argparse
from pathlib import Path
from string import Template


def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ultra-env", required=True, help="Path to ultra-backend.env")
    parser.add_argument("--platform-env", required=True, help="Path to platform.env")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where rendered nginx configs should be written",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    template_dir = repo_root / "deploy" / "nginx"
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    env = {}
    env.update(load_env(Path(args.ultra_env).expanduser().resolve()))
    env.update(load_env(Path(args.platform_env).expanduser().resolve()))

    for template_path in sorted(template_dir.glob("*.template")):
        rendered = Template(template_path.read_text(encoding="utf-8")).substitute(env)
        target_name = template_path.name.removesuffix(".template")
        (output_dir / target_name).write_text(rendered, encoding="utf-8")

    print(f"Rendered nginx configs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
