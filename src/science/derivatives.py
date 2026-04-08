"""Shared derivative cache for viewer/data-layer payloads.

This cache is keyed by the source file fingerprint and the semantic derivative
request, so identical requests can be reused across API endpoints.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Hashable

import numpy as np

_DEFAULT_MAX_ENTRIES = 128
_DEFAULT_MAX_VALUE_BYTES = 128 * 1024 * 1024


def _freeze_value(value: Any) -> Hashable:
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze_value(raw)) for key, raw in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze_value(item) for item in value))
    if isinstance(value, Path):
        return str(value.expanduser().resolve())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return ("ndarray", tuple(int(dim) for dim in value.shape), str(value.dtype))
    if isinstance(value, (str, bytes, int, float, bool, type(None))):
        return value
    return repr(value)


def _estimate_value_bytes(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bytes):
        return len(value)
    if isinstance(value, str):
        return len(value.encode("utf-8", errors="ignore"))
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, np.generic):
        return int(value.dtype.itemsize)
    if isinstance(value, dict):
        return sum(_estimate_value_bytes(key) + _estimate_value_bytes(raw) for key, raw in value.items())
    if isinstance(value, (list, tuple, set)):
        return sum(_estimate_value_bytes(item) for item in value)
    return 0


def file_derivative_key(
    *,
    derivative_kind: str,
    file_path: str | Path,
    **params: Any,
) -> tuple[Hashable, ...]:
    source = Path(file_path).expanduser().resolve()
    stat = source.stat()
    return (
        str(derivative_kind or "unknown"),
        str(source),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        _freeze_value(params),
    )


class FileDerivativeCache:
    def __init__(
        self,
        *,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
        max_value_bytes: int = _DEFAULT_MAX_VALUE_BYTES,
    ) -> None:
        self._max_entries = max(16, int(max_entries))
        self._max_value_bytes = max(1, int(max_value_bytes))
        self._lock = Lock()
        self._entries: OrderedDict[tuple[Hashable, ...], Any] = OrderedDict()

    def get_or_create(
        self,
        key: tuple[Hashable, ...],
        factory: Callable[[], Any],
        *,
        clone_result: bool = False,
    ) -> Any:
        with self._lock:
            if key in self._entries:
                value = self._entries.pop(key)
                self._entries[key] = value
                return deepcopy(value) if clone_result else value

        value = factory()
        estimated_size = _estimate_value_bytes(value)
        if estimated_size <= self._max_value_bytes:
            with self._lock:
                self._entries[key] = value
                self._entries.move_to_end(key)
                while len(self._entries) > self._max_entries:
                    self._entries.popitem(last=False)
        return deepcopy(value) if clone_result else value

    def invalidate_file(self, file_path: str | Path) -> None:
        source = str(Path(file_path).expanduser().resolve())
        with self._lock:
            stale = [key for key in self._entries if len(key) >= 2 and key[1] == source]
            for key in stale:
                self._entries.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


_FILE_DERIVATIVE_CACHE = FileDerivativeCache()


def get_cached_file_derivative(
    *,
    derivative_kind: str,
    file_path: str | Path,
    factory: Callable[[], Any],
    clone_result: bool = False,
    **params: Any,
) -> Any:
    key = file_derivative_key(
        derivative_kind=derivative_kind,
        file_path=file_path,
        **params,
    )
    return _FILE_DERIVATIVE_CACHE.get_or_create(
        key,
        factory,
        clone_result=clone_result,
    )


def invalidate_file_derivatives(file_path: str | Path) -> None:
    _FILE_DERIVATIVE_CACHE.invalidate_file(file_path)


def clear_file_derivatives() -> None:
    _FILE_DERIVATIVE_CACHE.clear()


__all__ = [
    "clear_file_derivatives",
    "file_derivative_key",
    "get_cached_file_derivative",
    "invalidate_file_derivatives",
]
