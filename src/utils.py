"""
Utility functions for the Data Drift Detection System.

Provides robust logging configuration with rotating file handlers, custom JSON
serialization for numpy/datetime objects, safe atomic I/O operations, and timestamp helpers.
Follows PEP 8 and Global Naming Standardization.
"""

import json
import logging
import logging.handlers
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np

from src.config.settings import LOG_DATE_FORMAT, LOG_FILE, LOG_FORMAT

# Rotating log configuration
_LOG_MAX_BYTES: int = 5 * 1024 * 1024  # 5 MB per file
_LOG_BACKUP_COUNT: int = 3  # Keep 3 rotation backups


def json_default_serializer(obj: Any) -> Any:
    """
    Serialize custom objects including numpy primitives, arrays, dates, and paths.

    Args:
        obj: Object to serialize.

    Returns:
        JSON-compatible primitive or string representation.
    """
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32, np.float16, float)):
        if np.isnan(obj):
            return None
        if np.isneginf(obj):
            return "-Infinity"
        if np.isposinf(obj):
            return "Infinity"
        return float(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return str(obj)


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create and configure a thread-safe logger with rotating file and console handlers.

    The file handler rotates at 5 MB and keeps 3 backup files, preventing
    unbounded log growth in production environments.

    Args:
        name: Logger identifier (typically module ``__name__``).
        level: Logging verbosity level (default logging.INFO).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers on repeated calls
    has_rotating = any(
        isinstance(h, logging.handlers.RotatingFileHandler) for h in logger.handlers
    )
    has_stream = any(
        isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.handlers.RotatingFileHandler)
        for h in logger.handlers
    )

    if not has_rotating:
        try:
            LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                LOG_FILE,
                maxBytes=_LOG_MAX_BYTES,
                backupCount=_LOG_BACKUP_COUNT,
                encoding="utf-8",
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(
                logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
            )
            logger.addHandler(file_handler)
        except (OSError, PermissionError):
            # In read-only serverless environments (e.g. Vercel, AWS Lambda),
            # suppress file handler creation and rely exclusively on StreamHandler (stdout)
            pass

    if not has_stream:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(
            logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
        )
        logger.addHandler(console_handler)

    return logger


def get_timestamp() -> str:
    """Return the current timestamp formatted for file naming."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_readable_timestamp() -> str:
    """Return the current timestamp in a human-readable ISO-like format."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def save_json(data: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """
    Save a dictionary as a formatted JSON file atomically.

    Args:
        data: Dictionary payload to serialize.
        filepath: Destination file path.
    """
    target_path = Path(filepath)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(f"{target_path.suffix}.tmp")

    with open(temp_path, "w", encoding="utf-8") as file_handle:
        json.dump(data, file_handle, indent=2, default=json_default_serializer)

    temp_path.replace(target_path)


def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load and parse a JSON file into a Python dictionary.

    Args:
        filepath: Source file path.

    Returns:
        Parsed dictionary.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON data.
    """
    target_path = Path(filepath)
    with open(target_path, "r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure the target directory exists and return its resolved Path.

    Args:
        path: Directory path.

    Returns:
        Resolved Path object.
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path
