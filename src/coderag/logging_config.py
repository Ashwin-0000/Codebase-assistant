"""
Logging configuration for CodeRAG.

Sets up a consistent log format across the entire application:
- Coloured output when writing to a terminal (via Rich)
- Plain timestamped output when writing to a file

Call ``configure_logging()`` once at process start (CLI entry point does this).
All other modules simply call ``logging.getLogger(__name__)``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def configure_logging(level: str = "INFO", log_file: str = "") -> None:
    """Configure the root logger for CodeRAG.

    Args:
        level:    Log level string (DEBUG / INFO / WARNING / ERROR / CRITICAL).
        log_file: Optional path to a log file. If empty, logs go to stdout only.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Base formatter — used for file handler and non-TTY stdout
    plain_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handlers: list[logging.Handler] = []

    # --- stdout handler ---
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(numeric_level)

    # Use Rich for pretty output when stdout is a real terminal
    try:
        from rich.logging import RichHandler  # type: ignore[import-untyped]

        if sys.stdout.isatty():
            rich_handler = RichHandler(
                level=numeric_level,
                rich_tracebacks=True,
                markup=True,
                show_path=False,
            )
            handlers.append(rich_handler)
        else:
            stdout_handler.setFormatter(plain_formatter)
            handlers.append(stdout_handler)
    except ImportError:
        stdout_handler.setFormatter(plain_formatter)
        handlers.append(stdout_handler)

    # --- optional file handler ---
    if log_file:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(plain_formatter)
        handlers.append(file_handler)

    # Apply to root logger (all coderag.* loggers inherit from this)
    logging.basicConfig(level=numeric_level, handlers=handlers, force=True)

    # Quieten noisy third-party loggers
    for noisy in ("httpx", "httpcore", "urllib3", "chromadb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
