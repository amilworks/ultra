from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar, Token
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ExecuteProgressEvent:
    command: str
    stream: str
    text: str
    elapsed_seconds: float
    output_size_chars: int
    progress_index: int
    suppressed_line_count: int = 0

    def to_payload(self) -> dict[str, object]:
        return asdict(self)


ExecuteProgressSink = Callable[[ExecuteProgressEvent], None]

_execute_progress_sink: ContextVar[ExecuteProgressSink | None] = ContextVar(
    "ultra_execute_progress_sink",
    default=None,
)


def current_execute_progress_sink() -> ExecuteProgressSink | None:
    return _execute_progress_sink.get()


def set_execute_progress_sink(
    sink: ExecuteProgressSink | None,
) -> Token[ExecuteProgressSink | None]:
    return _execute_progress_sink.set(sink)


def reset_execute_progress_sink(token: Token[ExecuteProgressSink | None]) -> None:
    _execute_progress_sink.reset(token)
