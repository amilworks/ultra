"""Utility functions for the application."""

import functools
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.logger import logger


def log_execution_time(func: Callable) -> Callable:
    """
    Decorator to log function execution time.

    Args:
        func: Function to decorate.

    Returns:
        Wrapped function with timing logging.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} executed in {execution_time:.4f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.4f} seconds: {str(e)}")
            raise

    return wrapper


def handle_errors(func: Callable) -> Callable:
    """Decorator to log unexpected exceptions before re-raising them."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(f"Error in {func.__name__}: {str(e)}")
            raise

    return wrapper


def init_session_state(key: str, value: Any) -> None:
    """BRUHHH LOOK AT THIS LATER: removed Streamlit release path left this helper orphaned."""
    raise RuntimeError(f"init_session_state({key!r}) is not available in the production release.")


def ensure_local_bqapi() -> None:
    """Prefer the local bqapi checkout if present.

    This ensures changes in ./bqapi/src are used without requiring a separate
    pip install step. Safe to call multiple times.
    """
    repo_root = Path(__file__).resolve().parents[1]
    candidate = repo_root / "bqapi" / "src"
    if candidate.exists():
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)


def display_metrics(metrics: dict[str, Any], columns: int = 3) -> None:
    """BRUHHH LOOK AT THIS LATER: metrics cards moved to the frontend web app."""
    del metrics, columns


def get_scratchpad_path(session_id: str | None) -> str:
    """Return a per-session scratchpad path for tool-driven drafting."""
    base = Path("data") / "scratchpads"
    base.mkdir(parents=True, exist_ok=True)
    safe_id = session_id or "session"
    return str(base / f"{safe_id}.md")
