"""Centralized logging configuration using loguru.

This module provides a consistent logging interface across the application.
"""

import sys
from pathlib import Path
from typing import Any

from loguru import logger

from config.settings import get_settings


def setup_logger() -> Any:
    """Configure and setup application logger.

    Configures loguru logger with:
    - Console output with colorization
    - File rotation (10 MB per file, 10 files retention)
    - Structured logging format
    - Environment-specific log levels

    Returns:
        Any: Configured logger instance

    Examples:
        >>> log = setup_logger()
        >>> log.info("Application started")
    """
    settings = get_settings()

    # Remove default handler
    logger.remove()

    # Console handler with colors
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=settings.log_level,
        colorize=True,
    )

    # File handler with rotation
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logger.add(
        log_dir / "app_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=settings.log_level,
        rotation="10 MB",
        retention="10 files",
        compression="zip",
        enqueue=True,  # Thread-safe logging
    )

    # Error-specific file handler
    logger.add(
        log_dir / "error_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}\n{exception}",
        level="ERROR",
        rotation="10 MB",
        retention="30 files",
        compression="zip",
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )

    logger.info(f"Logger initialized with level: {settings.log_level}")
    logger.info(f"Environment: {settings.environment}")

    return logger


# Global logger instance
log = setup_logger()
