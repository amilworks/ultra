#!/usr/bin/env python3
"""One-off cleanup for the pre-c7822b9c memory seeding bug.

That bug wrote the app-owned user-profile template into the agent-owned
``preferences.md``. This script finds per-user ``preferences.md`` files that are
still just the seed template (byte-identical to ``user_profile.md`` or carrying
the seed header) and removes them, so the agent recreates a clean preferences
file on its next learned write. Genuine agent-authored preferences are never
touched.

Usage:
    python scripts/purge_polluted_preferences.py --memory-root data/deepagents/memory
    python scripts/purge_polluted_preferences.py --memory-root /srv/ultra/shared/deepagents/memory --apply
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Signature of the app-seed template (produced only by _format_profile_markdown).
SEED_HEADER = "This researcher described themselves in their Ultra settings"


def _is_polluted(preferences: Path, user_profile: Path) -> bool:
    try:
        pref_text = preferences.read_text(encoding="utf-8")
    except OSError:
        return False
    if SEED_HEADER in pref_text:
        return True
    if user_profile.is_file():
        try:
            return pref_text == user_profile.read_text(encoding="utf-8")
        except OSError:
            return False
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--memory-root",
        required=True,
        help="Deep Agents memory root (ULTRA_DEEPAGENTS_MEMORY_ROOT).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete polluted files. Without this flag the script only reports (dry run).",
    )
    args = parser.parse_args(argv)

    users_dir = Path(args.memory_root).expanduser() / "users"
    if not users_dir.is_dir():
        print(f"No users directory under {args.memory_root!r}; nothing to do.")
        return 0

    polluted: list[Path] = []
    for user_dir in sorted(p for p in users_dir.iterdir() if p.is_dir()):
        preferences = user_dir / "preferences.md"
        if not preferences.is_file():
            continue
        if _is_polluted(preferences, user_dir / "user_profile.md"):
            polluted.append(preferences)

    if not polluted:
        print("No polluted preferences.md files found.")
        return 0

    verb = "Deleting" if args.apply else "Would delete (dry run)"
    for path in polluted:
        print(f"{verb}: {path}")
        if args.apply:
            try:
                path.unlink()
            except OSError as exc:
                print(f"  ERROR: {exc}", file=sys.stderr)

    print(f"\n{len(polluted)} polluted file(s) {'removed' if args.apply else 'found'}.")
    if not args.apply:
        print("Re-run with --apply to remove them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
