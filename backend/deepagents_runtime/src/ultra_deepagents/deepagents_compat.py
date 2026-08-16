"""Pinned compatibility adapters for the Deep Agents 0.7 middleware stack."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from deepagents import FilesystemPermission
from deepagents.backends.protocol import BackendProtocol
from deepagents.middleware.filesystem import FilesystemMiddleware, FsToolName
from langchain.agents.middleware import TodoListMiddleware

# Explicit 0.7 policy decision: writable routes adopt upstream whole-file
# replacement semantics. Enforced permissions, not prose, protect read-only routes.
DEEPAGENTS_WRITE_FILE_DESCRIPTION = (
    "Create a new file or deliberately replace the entire contents of an existing "
    "file. Never use write_file on an existing file unless whole-file replacement "
    "is intended; use read_file and edit_file when preserving content."
)

DEEPAGENTS_FILESYSTEM_TOOLS: list[FsToolName] = [
    "ls",
    "read_file",
    "write_file",
    "edit_file",
    "glob",
    "grep",
    "execute",
]


def build_deepagents_07_middleware(
    *,
    backend: BackendProtocol,
    permissions: Sequence[FilesystemPermission],
) -> list[Any]:
    """Return fresh 0.6-compatible planning and restricted filesystem middleware.

    Deep Agents 0.7 removed todo planning from its default stack and added
    ``delete`` to the default filesystem suite. Each graph gets new middleware
    instances because ``FilesystemMiddleware`` owns executor state.
    """
    filesystem = FilesystemMiddleware(
        backend=backend,
        tools=DEEPAGENTS_FILESYSTEM_TOOLS,
        custom_tool_descriptions={
            "write_file": DEEPAGENTS_WRITE_FILE_DESCRIPTION,
        },
        # Pinned 0.7 bridge: replacing the SDK's default FilesystemMiddleware
        # otherwise discards the top-level permission denies.
        _permissions=list(permissions),
    )
    return [TodoListMiddleware(), filesystem]
