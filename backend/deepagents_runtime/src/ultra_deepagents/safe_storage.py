"""Root-anchored filesystem helpers for shared scientific-data storage.

The configured upload root is the trust anchor. Every server-owned descendant
is opened one component at a time with ``O_NOFOLLOW`` so a symlink cannot move
publication locks, tombstones, or output bytes outside that root.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)


def _safe_component(component: str) -> str:
    if not component or component in {".", ".."} or os.sep in component:
        raise ValueError("unsafe managed-storage path component")
    if os.altsep and os.altsep in component:
        raise ValueError("unsafe managed-storage path component")
    return component


@contextmanager
def open_directory_chain_no_follow(
    trusted_root: Path,
    components: Sequence[str],
    *,
    create: bool = False,
    mode: int = 0o700,
) -> Iterator[int]:
    """Open a descendant directory without following any managed component."""

    # The configured root itself is the explicit trust anchor. Resolve it once
    # so deployments that intentionally mount it through a symlink still work;
    # every component created or opened beneath it remains no-follow.
    root = Path(trusted_root).expanduser().resolve(strict=True)
    descriptor = os.open(root, _DIRECTORY_FLAGS)
    try:
        for raw_component in components:
            component = _safe_component(str(raw_component))
            if create:
                try:
                    os.mkdir(component, mode=mode, dir_fd=descriptor)
                except FileExistsError:
                    pass
                # The leaf journal is not durable if any newly-created
                # ancestor entry can disappear after a crash. Persist each
                # parent immediately before opening and trusting its child.
                # Repeat the barrier when the entry already exists: a previous
                # attempt may have completed mkdir but failed this fsync.
                os.fsync(descriptor)
            child = os.open(component, _DIRECTORY_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        yield descriptor
    finally:
        os.close(descriptor)
