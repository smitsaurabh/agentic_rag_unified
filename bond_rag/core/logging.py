"""
Structured logging setup using Loguru.

Features
────────
• JSON-formatted logs when LOG_LEVEL=DEBUG or in production mode.
• Human-readable coloured output in development (default).
• Optional file sink with automatic rotation (10 MB) and retention (30 days).
• A single ``setup_logging()`` call wires everything up from Settings.
• ``get_logger(name)`` returns a context-bound logger for each module.

Usage::

    from bond_rag.core.logging import get_logger

    logger = get_logger(__name__)
    logger.info("Ingesting PDF", filename="bond.pdf", pages=400)
    logger.warning("ISIN not detected on page", page=42)
    logger.error("ChromaDB error", exc_info=True)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from loguru import logger


# ── Public API ─────────────────────────────────────────────────────────────────

def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    json_logs: bool = False,
) -> None:
    """
    Configure the global Loguru logger.

    Call this once at application startup (done automatically by
    ``BondRAGPipeline.__init__``).

    Parameters
    ----------
    level     : Log level string (DEBUG / INFO / WARNING / ERROR / CRITICAL)
    log_file  : Optional path to write logs to (in addition to stderr)
    json_logs : Emit JSON lines to stderr instead of coloured text
    """
    # Remove Loguru's default handler
    logger.remove()

    if json_logs:
        # Structured JSON for log aggregators (Loki, Datadog, etc.)
        fmt = (
            '{{"time":"{time:YYYY-MM-DDTHH:mm:ss.SSS}Z",'
            '"level":"{level.name}",'
            '"name":"{name}",'
            '"message":"{message}",'
            '"extra":{extra}}}'
        )
    else:
        # Human-readable coloured output
        fmt = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
            "{extra}"
        )

    logger.add(
        sys.stderr,
        level=level,
        format=fmt,
        colorize=not json_logs,
        backtrace=True,
        diagnose=(level == "DEBUG"),
    )

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_file),
            level=level,
            format=fmt,
            rotation="10 MB",
            retention="30 days",
            compression="gz",
            enqueue=True,      # thread-safe async writes
            backtrace=True,
        )

    logger.debug("Logging initialised", level=level, file=str(log_file))


def get_logger(name: str):  # type: ignore[return]
    """
    Return a context-bound logger for the given module name.

    Usage::

        logger = get_logger(__name__)
        logger.info("Starting ingestion", filename="bond.pdf")
    """
    return logger.bind(name=name)
