"""Centralized logging configuration."""

import logging
import sys

from src.config import get_settings


def setup_logger(name: str | None = None) -> logging.Logger:
    """
    Set up and configure a logger with consistent formatting.

    Args:
        name: Logger name. If None, returns root logger.

    Returns:
        Configured logger instance.
    """
    settings = get_settings()

    logger = logging.getLogger(name or __name__)

    # Only configure if no handlers exist (avoid duplicate logs)
    if not logger.handlers:
        logger.setLevel(getattr(logging, settings.log_level))

        # Console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, settings.log_level))

        # Formatter
        formatter = logging.Formatter(settings.log_format)
        handler.setFormatter(formatter)

        logger.addHandler(handler)

        # Prevent propagation to avoid duplicate logs
        logger.propagate = False

    return logger


# Default logger for the application
logger = setup_logger("bisque_ultra")
