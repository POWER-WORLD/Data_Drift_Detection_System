"""
Unit tests for general utility functions (src/utils.py).

Tests logger configuration, handler idempotency, timestamp formatting,
directory creation, and JSON serialization/deserialization with error handling.
"""

from pathlib import Path
from typing import Any, Dict
import json
import logging
import re
import pytest

from src.utils import (
    ensure_dir,
    get_readable_timestamp,
    get_timestamp,
    load_json,
    save_json,
    setup_logger,
)


class TestLoggerSetup:
    """Tests for setup_logger function."""

    def test_logger_creation_and_level(self) -> None:
        """Verify logger is created with requested name and logging level."""
        logger = setup_logger("test_module_custom", level=logging.DEBUG)
        assert logger.name == "test_module_custom"
        assert logger.level == logging.DEBUG
        assert len(logger.handlers) >= 2  # RotatingFileHandler and StreamHandler

    def test_logger_idempotent_handlers(self) -> None:
        """Verify calling setup_logger multiple times does not accumulate duplicate handlers."""
        logger1 = setup_logger("test_idempotent")
        initial_handlers_count = len(logger1.handlers)

        logger2 = setup_logger("test_idempotent")
        assert len(logger2.handlers) == initial_handlers_count


class TestTimestampHelpers:
    """Tests for timestamp formatting functions."""

    def test_get_timestamp_format(self) -> None:
        """Verify get_timestamp matches YYYY-MM-DD_HH-MM-SS format suitable for filenames."""
        ts = get_timestamp()
        pattern = r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$"
        assert re.match(pattern, ts) is not None

    def test_get_readable_timestamp_format(self) -> None:
        """Verify get_readable_timestamp matches YYYY-MM-DD HH:MM:SS format."""
        ts = get_readable_timestamp()
        pattern = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$"
        assert re.match(pattern, ts) is not None


class TestJsonHelpers:
    """Tests for save_json and load_json functions."""

    def test_save_and_load_json_roundtrip(self, tmp_path: Path) -> None:
        """Verify save_json correctly serializes and load_json correctly deserializes data."""
        filepath = tmp_path / "test_data.json"
        data: Dict[str, Any] = {
            "name": "DriftDetection",
            "version": 1.0,
            "metrics": {"auc": 0.95, "f1": 0.88},
            "features": ["tenure", "MonthlyCharges"],
        }

        save_json(data, filepath)
        assert filepath.exists()

        loaded = load_json(filepath)
        assert loaded == data

    def test_save_json_with_non_serializable_types(self, tmp_path: Path) -> None:
        """Verify save_json handles non-primitive types (e.g. Path objects) using default=str."""
        filepath = tmp_path / "test_paths.json"
        data: Dict[str, Any] = {
            "path": Path("some/nested/file.txt"),
        }

        save_json(data, filepath)
        loaded = load_json(filepath)
        assert "file.txt" in loaded["path"]

    def test_load_json_nonexistent_file_raises(self, tmp_path: Path) -> None:
        """Verify load_json raises FileNotFoundError when target file does not exist."""
        fake_path = tmp_path / "does_not_exist.json"
        with pytest.raises(FileNotFoundError):
            load_json(fake_path)

    def test_load_json_corrupt_file_raises(self, tmp_path: Path) -> None:
        """Verify load_json raises json.JSONDecodeError when file contains invalid syntax."""
        corrupt_path = tmp_path / "corrupt.json"
        corrupt_path.write_text("{broken_json: true", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            load_json(corrupt_path)


class TestEnsureDir:
    """Tests for ensure_dir function."""

    def test_ensure_dir_creates_hierarchy(self, tmp_path: Path) -> None:
        """Verify ensure_dir recursively creates parents and returns directory path."""
        target_dir = tmp_path / "deep" / "nested" / "output"
        assert not target_dir.exists()

        returned_path = ensure_dir(target_dir)
        assert returned_path.exists()
        assert returned_path.is_dir()

    def test_ensure_dir_existing_path_no_error(self, tmp_path: Path) -> None:
        """Verify calling ensure_dir on an existing directory succeeds silently."""
        existing = tmp_path / "existing_dir"
        existing.mkdir()

        returned_path = ensure_dir(existing)
        assert returned_path.exists()
