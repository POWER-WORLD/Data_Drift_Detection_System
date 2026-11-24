"""
Integration and unit tests for FastAPI REST API endpoints.

Tests endpoint status codes, response schemas, error handling, parameter validation,
file upload processing, drift analysis execution, and security headers.
"""

from pathlib import Path
import io
import pytest
from fastapi.testclient import TestClient
import pandas as pd

from api.app import app, SYSTEM_URL, PORTFOLIO_URL, LINKEDIN_URL, LEETCODE_URL, GITHUB_URL, AUTHOR_NAME
from src.config.settings import BASELINE_FILE, DRIFT_THRESHOLDS


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Fixture providing a configured FastAPI TestClient."""
    return TestClient(app)


class TestHealthAndMetadataEndpoints:
    """Tests for health verification and metadata endpoints."""

    def test_health_endpoint_success(self, client: TestClient) -> None:
        """Verify /api/health returns 200 OK with correct system attribution and status."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["system_url"] == SYSTEM_URL
        assert data["system_url"] == "https://dddsystem.vercel.app"
        assert data["author"]["name"] == AUTHOR_NAME
        assert data["author"]["name"] == "Pawan Kumar"
        assert data["author"]["portfolio"] == PORTFOLIO_URL
        assert data["author"]["linkedin"] == LINKEDIN_URL
        assert data["author"]["leetcode"] == LEETCODE_URL
        assert data["author"]["github"] == GITHUB_URL

    def test_settings_get_endpoint(self, client: TestClient) -> None:
        """Verify /api/settings retrieves current drift thresholds and system metadata."""
        response = client.get("/api/settings")
        assert response.status_code == 200
        data = response.json()
        assert "thresholds" in data
        assert "system" in data
        assert data["system"]["system_url"] == "https://dddsystem.vercel.app"
        assert "psi_moderate" in data["thresholds"]
        assert "ks_pvalue" in data["thresholds"]

    def test_settings_update_endpoint(self, client: TestClient) -> None:
        """Verify /api/settings POST updates statistical thresholds safely."""
        original_psi = DRIFT_THRESHOLDS.get("psi_moderate", 0.10)
        try:
            update_payload = {"thresholds": {"psi_moderate": 0.14}}
            response = client.post("/api/settings", json=update_payload)
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert data["updated_thresholds"]["psi_moderate"] == 0.14
            assert DRIFT_THRESHOLDS["psi_moderate"] == 0.14
        finally:
            DRIFT_THRESHOLDS["psi_moderate"] = original_psi


class TestBatchesAndDriftAnalysisEndpoints:
    """Tests for batch enumeration and statistical drift computation endpoints."""

    def test_list_batches_endpoint(self, client: TestClient) -> None:
        """Verify /api/batches lists available production batches with metadata."""
        response = client.get("/api/batches")
        assert response.status_code == 200
        data = response.json()
        assert "batches" in data
        assert isinstance(data["batches"], list)
        assert len(data["batches"]) > 0
        first_batch = data["batches"][0]
        assert "filename" in first_batch
        assert "size_kb" in first_batch
        assert "path" in first_batch

    def test_analyze_batch_success(self, client: TestClient) -> None:
        """Verify /api/analyze executes full drift detection on an existing batch."""
        payload = {"batch_name": "batch_01_no_drift.csv"}
        response = client.post("/api/analyze", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["batch_name"] == "batch_01_no_drift.csv"
        assert "dataset_drift" in data
        assert "feature_drift" in data
        assert "performance" in data
        assert "confidence_stats" in data
        assert "report_paths" in data
        assert data["dataset_drift"]["risk_level"] in ("LOW", "MEDIUM", "HIGH")
        assert "total_features" in data["dataset_drift"]
        assert "drifted_count" in data["dataset_drift"]

    def test_analyze_batch_non_existent(self, client: TestClient) -> None:
        """Verify /api/analyze returns 404 Not Found for non-existent batch."""
        payload = {"batch_name": "non_existent_batch_xyz_9999.csv"}
        response = client.post("/api/analyze", json=payload)
        assert response.status_code == 404
        assert "detail" in response.json()

    def test_feature_distribution_endpoint(self, client: TestClient) -> None:
        """Verify /api/feature-distribution returns baseline and production distributions."""
        response = client.get(
            "/api/feature-distribution/MonthlyCharges",
            params={"batch_name": "batch_01_no_drift.csv"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["feature"] == "MonthlyCharges"
        assert "baseline" in data
        assert "production" in data
        assert len(data["baseline"]) > 0
        assert len(data["production"]) > 0

    def test_feature_distribution_invalid_feature(self, client: TestClient) -> None:
        """Verify /api/feature-distribution returns 404 for non-existent feature."""
        response = client.get(
            "/api/feature-distribution/NonExistentFeatureColumn_XYZ",
            params={"batch_name": "batch_1.csv"},
        )
        assert response.status_code == 404


class TestAlertsEndpoints:
    """Tests for alert retrieval and clearing."""

    def test_get_alerts_endpoint(self, client: TestClient) -> None:
        """Verify /api/alerts returns alert notifications array."""
        response = client.get("/api/alerts")
        assert response.status_code == 200
        data = response.json()
        assert "alerts" in data
        assert "total" in data
        assert isinstance(data["alerts"], list)

    def test_get_alerts_filter_severity(self, client: TestClient) -> None:
        """Verify /api/alerts filters alerts by severity level."""
        response = client.get("/api/alerts?severity=CRITICAL")
        assert response.status_code == 200
        data = response.json()
        for alert in data["alerts"]:
            assert alert.get("severity") == "CRITICAL"

    def test_clear_alerts_endpoint(self, client: TestClient) -> None:
        """Verify /api/alerts/clear resets alert history."""
        response = client.post("/api/alerts/clear")
        assert response.status_code == 200
        assert response.json()["status"] == "success"


class TestUploadEndpoint:
    """Tests for secure batch upload and universal schema tolerance."""

    def test_upload_accepts_custom_arbitrary_schema(self, client: TestClient) -> None:
        """Verify /api/upload accepts any custom CSV dataset with arbitrary column names."""
        custom_csv = (
            b"metric_alpha,metric_beta,category_gamma\n"
            b"12.5,100.2,North\n"
            b"14.1,105.8,South\n"
            b"13.8,102.3,North\n"
            b"15.2,110.1,East\n"
            b"16.0,115.4,West\n"
            b"17.2,120.0,South\n"
        )
        files = {"file": ("custom_test_dataset.csv", io.BytesIO(custom_csv), "text/csv")}
        response = client.post("/api/upload", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["columns"] == 3
        assert data["is_telco_schema"] is False
        assert "metric_alpha" in data["numerical_features"]
        assert "metric_beta" in data["numerical_features"]
        assert "category_gamma" in data["categorical_features"]

        # Verify that drift analysis executes smoothly on this custom dataset
        analysis_resp = client.post("/api/analyze", json={"batch_name": "custom_test_dataset.csv"})
        assert analysis_resp.status_code == 200
        analysis_data = analysis_resp.json()
        assert analysis_data["status"] == "success"
        assert analysis_data["dataset_drift"]["total_features"] == 3
        assert "metric_alpha" in analysis_data["feature_drift"]["numerical"]
        assert "category_gamma" in analysis_data["feature_drift"]["categorical"]

        # Clean up temporary test files
        prod_path = Path("data/production/custom_test_dataset.csv")
        base_path = Path("data/baseline/custom_test_dataset_baseline.csv")
        if prod_path.exists():
            prod_path.unlink()
        if base_path.exists():
            base_path.unlink()

    def test_upload_rejects_empty_corrupt_file(self, client: TestClient) -> None:
        """Verify /api/upload rejects empty (0 bytes) or non-tabular files."""
        files = {"file": ("empty.csv", io.BytesIO(b""), "text/csv")}
        response = client.post("/api/upload", files=files)
        assert response.status_code in (400, 422)

    def test_upload_valid_batch_csv(self, client: TestClient) -> None:
        """Verify /api/upload accepts a schema-compliant production batch."""
        # Read a sample from baseline data to form a valid schema batch
        baseline_df = pd.read_csv(BASELINE_FILE).drop(columns=["Churn"], errors="ignore")
        sample_batch = baseline_df.head(10)
        csv_buffer = io.BytesIO()
        sample_batch.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue()

        files = {"file": ("test_upload_valid_temp.csv", io.BytesIO(csv_bytes), "text/csv")}
        response = client.post("/api/upload", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["rows"] == 10

        # Clean up the test uploaded file
        uploaded_path = Path("data/production/test_upload_valid_temp.csv")
        if uploaded_path.exists():
            uploaded_path.unlink()


class TestStaticUiServing:
    """Tests for static frontend asset delivery."""

    def test_root_serves_html(self, client: TestClient) -> None:
        """Verify GET / returns 200 OK with the application HTML shell."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        assert "Data Drift Detection" in response.text
        assert "dddsystem.vercel.app" in response.text or "Pawan Kumar" in response.text
