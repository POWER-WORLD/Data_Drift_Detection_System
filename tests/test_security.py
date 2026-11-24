"""
Security & DevOps Assurance Tests for Data Drift Detection System.

Validates:
- Dockerfile hardening (multi-stage build, non-root user, healthcheck, security posture)
- Container exclusion rules (.dockerignore)
- Deployment security configurations (vercel.json)
- Input sanitization and path traversal defenses
- Data file validation boundaries
- Secret leakage prevention (.gitignore)
"""

import json
from pathlib import Path
import pytest

from src.data_validator import sanitize_filename, validate_file, DataValidationError
from src.config.settings import PROJECT_ROOT


class TestSecurityConfiguration:
    """Verifies that security configurations and manifests adhere to enterprise standards."""

    def test_dockerfile_hardening(self) -> None:
        """Verify Dockerfile uses multi-stage build, non-root execution, and hardened healthcheck."""
        dockerfile_path = PROJECT_ROOT / "Dockerfile"
        assert dockerfile_path.exists(), "Dockerfile must exist"

        content = dockerfile_path.read_text(encoding="utf-8")

        # Multi-stage build
        assert "AS builder" in content, "Dockerfile must use a builder stage"
        assert "AS runtime" in content, "Dockerfile must define a runtime stage"

        # Non-root user execution
        assert "10001" in content, "Dockerfile must declare non-root UID 10001"
        assert "USER 10001:10001" in content or "USER appuser" in content, "Container must switch to non-root user"

        # Healthcheck exists and doesn't rely on external uninstalled curl
        assert "HEALTHCHECK" in content, "Dockerfile must define a HEALTHCHECK"
        assert "urllib.request" in content, "Healthcheck should use python standard library for zero-binary footprint"

        # Security environment variables
        assert "STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true" in content, "Streamlit XSRF protection must be enabled"
        assert "STREAMLIT_SERVER_ENABLE_CORS=false" in content, "Streamlit CORS must be disabled"

    def test_dockerignore_coverage(self) -> None:
        """Verify .dockerignore excludes sensitive files, git history, and local virtualenvs."""
        dockerignore_path = PROJECT_ROOT / ".dockerignore"
        assert dockerignore_path.exists(), ".dockerignore must exist"

        content = dockerignore_path.read_text(encoding="utf-8")
        required_exclusions = [".git/", "venv/", ".env", "tests/", ".pytest_cache/"]

        for item in required_exclusions:
            assert item in content, f".dockerignore must exclude '{item}'"

    def test_gitignore_secret_protection(self) -> None:
        """Verify .gitignore blocks environment variables, credentials, and private keys."""
        gitignore_path = PROJECT_ROOT / ".gitignore"
        assert gitignore_path.exists(), ".gitignore must exist"

        content = gitignore_path.read_text(encoding="utf-8")
        required_ignores = [".env", "*.pem", "*.key", "credentials.json", ".streamlit/secrets.toml"]

        for item in required_ignores:
            assert item in content, f".gitignore must ignore sensitive file pattern '{item}'"

    def test_vercel_security_headers(self) -> None:
        """Verify vercel.json enforces OWASP security headers."""
        vercel_path = PROJECT_ROOT / "vercel.json"
        assert vercel_path.exists(), "vercel.json must exist for production deployment"

        data = json.loads(vercel_path.read_text(encoding="utf-8"))
        headers_list = data.get("headers", [{}])[0].get("headers", [])
        header_keys = {h["key"] for h in headers_list}

        expected_headers = {
            "Content-Security-Policy",
            "X-Content-Type-Options",
            "X-Frame-Options",
            "X-XSS-Protection",
            "Referrer-Policy",
            "Strict-Transport-Security",
            "Permissions-Policy",
        }
        missing = expected_headers - header_keys
        assert not missing, f"vercel.json missing required security headers: {missing}"

    def test_vercel_is_primary_deployment_target(self) -> None:
        """Verify vercel.json is the sole production deployment configuration and netlify.toml is removed."""
        netlify_path = PROJECT_ROOT / "netlify.toml"
        assert not netlify_path.exists(), "netlify.toml must be removed as hosting is dedicated to Vercel"

        vercel_path = PROJECT_ROOT / "vercel.json"
        assert vercel_path.exists(), "vercel.json must exist for Vercel deployment"


class TestInputSanitizationAndBoundaries:
    """Verifies input sanitization, file boundaries, and path traversal protection."""

    def test_sanitize_filename_prevents_path_traversal(self) -> None:
        """Verify that directory traversal sequences are eliminated."""
        malicious_inputs = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32\\cmd.exe",
            "/absolute/path/attack.csv",
            "test/subdir/batch.csv",
        ]
        for evil in malicious_inputs:
            safe = sanitize_filename(evil)
            assert "/" not in safe, f"Path traversal forward-slash detected in {safe}"
            assert "\\" not in safe, f"Path traversal backslash detected in {safe}"
            assert safe.endswith(".csv"), f"Filename must end in .csv: {safe}"

    def test_sanitize_filename_fallback_on_empty(self) -> None:
        """Verify fallback filename when an empty or illegal input is provided."""
        assert sanitize_filename("") == "uploaded_batch.csv"
        assert sanitize_filename(".csv") == "uploaded_batch.csv"

    def test_validate_file_rejects_non_csv(self, tmp_path: Path) -> None:
        """Verify that non-CSV files are rejected by validate_file."""
        non_csv = tmp_path / "malicious.sh"
        non_csv.write_text("#!/bin/sh\necho 'hacked'")

        with pytest.raises(DataValidationError, match="Invalid file type"):
            validate_file(non_csv)

    def test_validate_file_rejects_oversized_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify files exceeding MAX_FILE_SIZE_BYTES are rejected."""
        large_file = tmp_path / "oversized.csv"
        large_file.write_text("col_a,col_b\nval1,val2\n")

        # Monkeypatch MAX_FILE_SIZE_BYTES to 5 bytes to trigger size limit cleanly
        monkeypatch.setattr("src.data_validator.MAX_FILE_SIZE_BYTES", 5)

        with pytest.raises(DataValidationError, match="File too large"):
            validate_file(large_file)
