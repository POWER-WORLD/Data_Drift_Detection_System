"""
Utility functions for the Data Drift Detection System.
Provides logging setup with rotating file handler, timestamp helpers,
and common I/O operations.
"""

import logging
import logging.handlers
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from src.config.settings import LOG_FILE, LOG_FORMAT, LOG_DATE_FORMAT

# Rotating log configuration
_LOG_MAX_BYTES = 5 * 1024 * 1024   # 5 MB per file
_LOG_BACKUP_COUNT = 3               # Keep 3 old log files


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create and configure a logger with rotating file and console handlers.

    The file handler rotates at 5 MB and keeps 3 backup files, preventing
    unbounded log growth in production.

    Args:
        name: Logger name (typically module ``__name__``).
        level: Logging level (default INFO).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    # Rotating file handler
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.handlers.RotatingFileHandler(
        LOG_FILE,
        maxBytes=_LOG_MAX_BYTES,
        backupCount=_LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
    logger.addHandler(ch)

    return logger


def get_timestamp() -> str:
    """Return current timestamp as a formatted string for filenames."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_readable_timestamp() -> str:
    """Return current timestamp in human-readable format."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def save_json(data: Dict[str, Any], filepath: Path) -> None:
    """
    Save a dictionary as a formatted JSON file.

    Args:
        data: Dictionary to serialize.
        filepath: Destination path.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: Path) -> Dict[str, Any]:
    """
    Load a JSON file into a dictionary.

    Args:
        filepath: Source path.

    Returns:
        Parsed dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist and return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path
