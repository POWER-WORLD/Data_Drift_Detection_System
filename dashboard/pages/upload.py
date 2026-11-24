"""
Upload & Run page — Batch upload, drift detection, threshold config, report export.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from dashboard.components import page_header, section_header, kpi_card, empty_state, risk_badge, CHART_COLORS
from src.config.settings import PRODUCTION_DIR, DRIFT_THRESHOLDS
from src.data_loader import load_baseline_data, load_production_batch, list_production_batches
from src.data_validator import sanitize_filename, validate_schema, validate_data_quality, MAX_FILE_SIZE_BYTES, DataValidationError
from src.drift_detection import run_full_drift_analysis
from src.predict import predict_batch
from src.alert import generate_alerts
from src.report import generate_drift_report


def render():
    """Render the upload & run page."""
    page_header("Upload & Run", "Upload production data, configure thresholds, and run drift detection")

    # ── Upload Section ────────────────────────────────────────────────────
    section_header("Upload New Batch", "📤")

    uploaded_file = st.file_uploader(
        "Drag and drop a CSV file here",
        type=["csv"],
        key="batch_uploader",
        help=f"Maximum file size: {MAX_FILE_SIZE_BYTES // (1024*1024)} MB",
    )

    if uploaded_file:
        file_size = uploaded_file.size
        if file_size > MAX_FILE_SIZE_BYTES:
            st.error(f"❌ File too large ({file_size / (1024*1024):.1f} MB). Max: {MAX_FILE_SIZE_BYTES // (1024*1024)} MB.")
        else:
            safe_name = sanitize_filename(uploaded_file.name)
            save_path = PRODUCTION_DIR / safe_name
            try:
                temp_df = pd.read_csv(uploaded_file)
                validate_schema(temp_df, require_target=False)
                uploaded_file.seek(0)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"✅ Saved as `{safe_name}` ({file_size/1024:.0f} KB)")
                quality = validate_data_quality(temp_df)
                if quality["issues"]:
                    with st.expander("⚠️ Data Quality Warnings"):
                        for issue in quality["issues"]:
                            st.warning(issue)
            except DataValidationError as e:
                st.error(f"❌ Validation failed: {e}")
            except Exception as e:
                st.error(f"❌ Error: {e}")

    # ── Batch Selection ───────────────────────────────────────────────────
    section_header("Select Batch", "📂")
    batches = list_production_batches()

    if not batches:
        empty_state("📂", "No Batches Found", "Upload a batch above or run model training first.")
        return

    selected = st.selectbox("Production batch:", [b.name for b in batches], key="batch_select")

    # ── Threshold Config ──────────────────────────────────────────────────
    section_header("Threshold Configuration", "⚙️")
    with st.expander("Customize Drift Thresholds", expanded=False):
        tc1, tc2, tc3 = st.columns(3)
        with tc1:
            ks_t = st.slider("KS p-value", 0.01, 0.10, 0.05, 0.01, key="ks_t")
        with tc2:
            psi_m = st.slider("PSI Moderate", 0.05, 0.30, 0.10, 0.01, key="psi_m")
        with tc3:
            psi_s = st.slider("PSI Severe", 0.15, 0.50, 0.25, 0.01, key="psi_s")

    # ── Run Button ────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("🔍 Run Drift Detection", type="primary", use_container_width=True, key="run_btn")

    if run_btn:
        DRIFT_THRESHOLDS["ks_pvalue"] = ks_t
        DRIFT_THRESHOLDS["psi_moderate"] = psi_m
        DRIFT_THRESHOLDS["psi_severe"] = psi_s

        with st.spinner("Running drift detection..."):
            try:
                selected_path = PRODUCTION_DIR / selected
                baseline_df = load_baseline_data()
                production_df = load_production_batch(selected_path)
                drift_report = run_full_drift_analysis(baseline_df, production_df)

                performance = None
                confidence_stats = None
                try:
                    pred_result = predict_batch(production_df)
                    performance = pred_result.get("comparison")
                    confidence_stats = pred_result.get("confidence_stats")
                except FileNotFoundError:
                    st.warning("⚠️ No trained model found.")
                except Exception as e:
                    st.warning(f"⚠️ Prediction error: {e}")

                alerts = generate_alerts(drift_report, batch_name=selected, performance_comparison=performance)
                report_paths = generate_drift_report(drift_report, batch_name=selected,
                                                      performance=performance, alerts=alerts)

                st.session_state.update({
                    "drift_report": drift_report,
                    "performance": performance,
                    "confidence_stats": confidence_stats,
                    "baseline_df": baseline_df,
                    "production_df": production_df,
                    "batch_name": selected,
                    "report_paths": report_paths,
                    "analysis_complete": True,
                })

                st.success("✅ Analysis complete!")
                st.balloons()

                # Quick summary
                ds = drift_report.get("dataset_drift", {})
                risk = ds.get("risk_level", "LOW")
                st.markdown("<br>", unsafe_allow_html=True)

                r1, r2, r3 = st.columns(3)
                risk_colors = {"LOW": "#10b981", "MEDIUM": "#f59e0b", "HIGH": "#ef4444"}
                with r1:
                    kpi_card("Risk Level", risk, "🛡️", risk_colors.get(risk, "#3b82f6"))
                with r2:
                    kpi_card("Drifted", f"{ds.get('drifted_count', 0)}/{ds.get('total_features', 0)}", "📊", CHART_COLORS["blue"])
                with r3:
                    h = performance.get("overall_health", "—") if performance else "—"
                    kpi_card("Model Health", h, "🤖", risk_colors.get(h, "#64748b"))

                # Export
                st.markdown("<br>", unsafe_allow_html=True)
                section_header("Export Reports", "📥")
                ec1, ec2 = st.columns(2)
                for i, (fmt, path) in enumerate(report_paths.items()):
                    col = ec1 if i == 0 else ec2
                    with col:
                        with open(path, "rb") as f:
                            st.download_button(f"📄 Download {fmt.upper()}", f.read(),
                                               path.name, key=f"dl_{fmt}")

            except DataValidationError as e:
                st.error(f"❌ Validation failed: {e}")
            except Exception as e:
                st.error(f"❌ Analysis failed: {e}")
                import traceback
                with st.expander("Error details"):
                    st.code(traceback.format_exc())
