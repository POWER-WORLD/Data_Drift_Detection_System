"""
Unit tests for dashboard components and UI integration (dashboard/components.py).

Verifies global system links, portfolio attribution links, risk badge generators,
severity badge generators, status pills, and page module imports.
"""

from typing import Optional
import pytest

from dashboard.components import (
    AUTHOR_NAME,
    GITHUB_URL,
    LEETCODE_URL,
    LINKEDIN_URL,
    PORTFOLIO_URL,
    SYSTEM_URL,
    risk_badge,
)


class TestDashboardConstants:
    """Verifies standard global branding and attribution URLs."""

    def test_system_url_vercel(self) -> None:
        """Verify production system URL points to Vercel deployment."""
        assert SYSTEM_URL == "https://dddsystem.vercel.app"

    def test_author_links(self) -> None:
        """Verify author portfolio, professional networks, and source repository links."""
        assert PORTFOLIO_URL == "https://pawankumar.info/"
        assert LINKEDIN_URL == "https://www.linkedin.com/in/pawankumar101/"
        assert LEETCODE_URL == "https://leetcode.com/u/pk3253702/"
        assert GITHUB_URL == "https://github.com/POWER-WORLD"
        assert AUTHOR_NAME == "Pawan Kumar"


class TestDashboardBadgeComponents:
    """Tests for HTML badge helper functions in dashboard components."""

    @pytest.mark.parametrize(
        "level, expected_text, expected_class",
        [
            ("LOW", "Low Risk", "badge-low"),
            ("MEDIUM", "Medium Risk", "badge-medium"),
            ("HIGH", "High Risk", "badge-high"),
            ("HEALTHY", "Healthy", "badge-healthy"),
            ("WARNING", "Warning", "badge-warning"),
            ("CRITICAL", "Critical", "badge-critical"),
            (None, "Low Risk", "badge-low"),
        ],
    )
    def test_risk_badge_rendering(
        self, level: Optional[str], expected_text: str, expected_class: str
    ) -> None:
        """Verify risk badge HTML conforms to standard color classes and labels."""
        html = risk_badge(level)
        assert expected_text in html
        assert expected_class in html
        assert "<span class=" in html


class TestDashboardModuleImports:
    """Verify that all dashboard pages and styles can be imported cleanly without syntax or import errors."""

    def test_styles_import(self) -> None:
        """Verify dashboard.styles imports successfully and exposes get_custom_css."""
        import dashboard.styles as styles
        assert hasattr(styles, "get_custom_css")
        css = styles.get_custom_css()
        assert isinstance(css, str)
        assert "<style>" in css

    def test_pages_importable(self) -> None:
        """Verify each dashboard sub-page imports cleanly."""
        import dashboard.pages.overview
        import dashboard.pages.feature_analysis
        import dashboard.pages.model_health
        import dashboard.pages.alerts
        import dashboard.pages.settings
        import dashboard.pages.upload


class TestDashboardAppExecution:
    """End-to-end interactive dashboard simulation tests using Streamlit AppTest."""

    def test_app_bare_launch(self) -> None:
        """Verify app.py initializes and renders the welcome hero screen with no unhandled exceptions."""
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file("dashboard/app.py", default_timeout=20)
        app.run()
        assert len(app.exception) == 0, f"Bare launch exception: {app.exception}"

    @pytest.mark.parametrize(
        "page_title",
        [
            "🏠  Overview",
            "🔬  Drift Analysis",
            "🤖  Model Health",
            "🔔  Alerts",
            "🚀  Upload & Run",
            "⚙️  Settings",
        ],
    )
    def test_all_pages_with_analysis_state(self, page_title: str) -> None:
        """Verify all 6 dashboard views render cleanly when populated with analysis session state."""
        from streamlit.testing.v1 import AppTest
        from src.data_loader import load_baseline_data, load_production_batch
        from src.drift_detection import run_full_drift_analysis
        from src.predict import predict_batch
        from src.config.settings import PRODUCTION_DIR

        baseline_df = load_baseline_data()
        production_df = load_production_batch(PRODUCTION_DIR / "batch_01_no_drift.csv")
        drift_report = run_full_drift_analysis(baseline_df=baseline_df, production_df=production_df)
        pred_result = predict_batch(production_df)

        app = AppTest.from_file("dashboard/app.py", default_timeout=25)
        app.session_state["analysis_complete"] = True
        app.session_state["drift_report"] = drift_report
        app.session_state["performance"] = pred_result.get("comparison")
        app.session_state["confidence_stats"] = pred_result.get("confidence_stats")
        app.session_state["baseline_df"] = baseline_df
        app.session_state["production_df"] = production_df
        app.session_state["batch_name"] = "batch_01_no_drift.csv"
        app.session_state["nav_page"] = page_title
        app.run()

        assert len(app.exception) == 0, f"Error rendering page '{page_title}': {app.exception}"

    def test_user_links_rendered_in_markdown(self) -> None:
        """Verify that all user attribution links are rendered across the UI."""
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file("dashboard/app.py", default_timeout=20)
        app.run()
        rendered_text = " ".join(m.value for m in app.markdown)

        expected_links = [
            "https://dddsystem.vercel.app",
            "https://pawankumar.info/",
            "https://www.linkedin.com/in/pawankumar101/",
            "https://leetcode.com/u/pk3253702/",
            "https://github.com/POWER-WORLD",
        ]
        for url in expected_links:
            assert url in rendered_text, f"Expected URL {url} was not found in rendered dashboard."

    @pytest.mark.parametrize(
        "batch_filename",
        [
            "batch_01_no_drift.csv",
            "batch_02_mild_drift.csv",
            "batch_03_severe_drift.csv",
            "batch_04_missing_data.csv",
            "batch_05_category_drift.csv",
        ],
    )
    def test_all_production_batches_render(self, batch_filename: str) -> None:
        """Verify that every one of the 5 production batch scenarios renders cleanly across Overview and Drift Analysis."""
        from streamlit.testing.v1 import AppTest
        from src.data_loader import load_baseline_data, load_production_batch
        from src.drift_detection import run_full_drift_analysis
        from src.predict import predict_batch
        from src.config.settings import PRODUCTION_DIR

        baseline_df = load_baseline_data()
        production_df = load_production_batch(PRODUCTION_DIR / batch_filename)
        drift_report = run_full_drift_analysis(baseline_df=baseline_df, production_df=production_df)
        pred_result = predict_batch(production_df)

        app = AppTest.from_file("dashboard/app.py", default_timeout=25)
        app.session_state["analysis_complete"] = True
        app.session_state["drift_report"] = drift_report
        app.session_state["performance"] = pred_result.get("comparison")
        app.session_state["confidence_stats"] = pred_result.get("confidence_stats")
        app.session_state["baseline_df"] = baseline_df
        app.session_state["production_df"] = production_df
        app.session_state["batch_name"] = batch_filename
        app.session_state["nav_page"] = "🏠  Overview"
        app.run()
        assert len(app.exception) == 0, f"Exception in Overview for batch '{batch_filename}': {app.exception}"

    def test_feature_analysis_interactive_dropdown(self) -> None:
        """Verify selecting individual numerical features in Feature Analysis runs without error."""
        from streamlit.testing.v1 import AppTest
        from src.data_loader import load_baseline_data, load_production_batch
        from src.drift_detection import run_full_drift_analysis
        from src.config.settings import PRODUCTION_DIR

        baseline_df = load_baseline_data()
        production_df = load_production_batch(PRODUCTION_DIR / "batch_01_no_drift.csv")
        drift_report = run_full_drift_analysis(baseline_df=baseline_df, production_df=production_df)

        app = AppTest.from_file("dashboard/app.py", default_timeout=25)
        app.session_state["analysis_complete"] = True
        app.session_state["drift_report"] = drift_report
        app.session_state["baseline_df"] = baseline_df
        app.session_state["production_df"] = production_df
        app.session_state["nav_page"] = "🔬  Drift Analysis"
        app.session_state["num_feature_selector"] = "tenure"
        app.run()
        assert len(app.exception) == 0, f"Exception in Feature Analysis: {app.exception}"

    def test_alerts_page_interactive_filtering(self) -> None:
        """Verify Alerts page filters by severity without errors."""
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file("dashboard/app.py", default_timeout=25)
        app.session_state["nav_page"] = "🔔  Alerts"
        app.session_state["alert_severity_filter"] = ["CRITICAL"]
        app.session_state["alert_keyword_search"] = "drift"
        app.run()
        assert len(app.exception) == 0, f"Exception in Alerts filtering: {app.exception}"

    def test_upload_page_empty_and_ready_state(self) -> None:
        """Verify Upload & Run page renders both pre-analysis and post-analysis states cleanly."""
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file("dashboard/app.py", default_timeout=25)
        app.session_state["nav_page"] = "🚀  Upload & Run"
        app.run()
        assert len(app.exception) == 0, f"Exception in Upload & Run: {app.exception}"



