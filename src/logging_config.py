"""Logging configuration helpers."""

from __future__ import annotations

import logging
import os

LOG_LEVEL_ENV = "APPROACH_ANALYSIS_LOG_LEVEL"
LOG_LEVELS = ("debug", "info", "warning", "error", "critical")


def configure_logging(level_name: str | None = None, *, force: bool = False) -> None:
    """Configure process-local logging."""
    resolved = (level_name or os.environ.get(LOG_LEVEL_ENV) or "warning").lower()
    if resolved not in LOG_LEVELS:
        raise ValueError(f"Unknown log level: {resolved}")

    os.environ[LOG_LEVEL_ENV] = resolved
    logging.basicConfig(
        level=getattr(logging, resolved.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=force,
    )


def configure_worker_logging() -> None:
    """Initializer for spawned worker processes."""
    configure_logging(os.environ.get(LOG_LEVEL_ENV), force=True)
