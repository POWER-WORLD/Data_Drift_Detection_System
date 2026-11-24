"""
Upload & Run page — Production batch upload, schema validation, drift evaluation, and export.
Orchestrates end-to-end data validation, statistical drift detection, and report generation.
"""

from pathlib import Path
import streamlit as st
import pandas as pd
from dashboard.components import (
    page_header,
    section_header,
    kpi_card,
    empty_state,
    risk_badge,
    CHART_COLORS,
)
from src.config.settings import PRODUCTION_DIR, DRIFT_THRESHOLDS
from src.data_loader import (
    load_baseline_data,
    load_production_batch,
    list_production_batches,
)
from src.data_validator import (
    sanitize_filename,
    validate_schema,
    validate_data_quality,
    MAX_FILE_SIZE_BYTES,
    DataValidationError,
)
from src.drift_detection import run_full_drift_analysis
from src.predict import predict_batch
from src.alert import generate_alerts
from src.report import generate_drift_report


def render() -> None:
    """Render the upload, batch evaluation, and drift analysis page."""
    page_header(
        "Upload & Evaluate Production Batches",
        "Ingest production data batches, run 8 statistical drift algorithms, and generate audit reports",
    )

    # ── Upload New Batch Section ──────────────────────────────────────────
    section_header("Upload Production Batch (CSV)", "📤")

    max_mb = MAX_FILE_SIZE_BYTES // (1024 * 1024)
    uploaded_file = st.file_uploader(
        "Upload a production batch file",
        type=["csv"],
        key="production_batch_uploader",
        help=f"Accepts CSV files with identical or subset schema to baseline data. Maximum file size: {max_mb} MB.",
    )

    if uploaded_file is not None:
        file_size = uploaded_file.size
        if file_size > MAX_FILE_SIZE_BYTES:
            st.error(
                f"❌ File size exceeds allowed threshold ({file_size / (1024 * 1024):.1f} MB > {max_mb} MB)."
            )
        else:
            safe_filename = sanitize_filename(uploaded_file.name)
            destination_path = PRODUCTION_DIR / safe_filename
            try:
                temp_dataframe = pd.read_csv(uploaded_file)
                validate_schema(temp_dataframe, require_target=False)

                uploaded_file.seek(0)
                with open(destination_path, "wb") as f_out:
                    f_out.write(uploaded_file.getbuffer())

                st.success(f"✅ Ingested `{safe_filename}` ({file_size / 1024:.1f} KB, {len(temp_dataframe):,} rows)")

                quality_report = validate_data_quality(temp_dataframe)
                if quality_report.get("issues"):
                    with st.expander("⚠️ Ingestion Quality Warnings", expanded=True):
                        for issue_msg in quality_report["issues"]:
                            st.warning(issue_msg)
            except DataValidationError as val_err:
                st.error(f"❌ Schema validation failed: {val_err}")
            except Exception as exc:
                st.error(f"❌ Error during file ingestion: {exc}")

    # ── Batch Selection Section ───────────────────────────────────────────
    section_header("Select Production Batch to Evaluate", "📂")
    available_batches = list_production_batches()

    if not available_batches:
        empty_state(
            "📂",
            "Zero Production Batches Available",
            "Please upload a CSV file above or place batch files in the `data/production/` folder.",
        )
        return

    batch_names = [b.name for b in available_batches]
    selected_batch_name = st.selectbox(
        "Production Batch File:",
        batch_names,
        key="selected_batch_dropdown",
        help="Choose an existing or newly uploaded batch to run statistical drift evaluation against baseline.",
    )

    # ── Threshold Sensitivity Drawer ──────────────────────────────────────
    section_header("Runtime Threshold Sensitivity", "⚙️")
    with st.expander("Adjust Test Sensitivities for This Execution", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            ks_param = st.slider(
                "KS p-value (alpha)",
                0.01, 0.10,
                float(DRIFT_THRESHOLDS.get("ks_pvalue", 0.05)),
                0.01,
                key="run_ks_param",
            )
        with c2:
            psi_mod_param = st.slider(
                "PSI Moderate",
                0.05, 0.30,
                float(DRIFT_THRESHOLDS.get("psi_moderate", 0.10)),
                0.01,
                key="run_psi_mod_param",
            )
        with c3:
            psi_sev_param = st.slider(
                "PSI Severe",
                0.15, 0.50,
                float(DRIFT_THRESHOLDS.get("psi_severe", 0.25)),
                0.01,
                key="run_psi_sev_param",
            )

    # ── Action Button ─────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    execute_button = st.button(
        "🔍 Run Full Drift Detection & Telemetry Analysis",
        type="primary",
        width="stretch",
        key="run_full_drift_btn",
    )

    if execute_button:
        DRIFT_THRESHOLDS["ks_pvalue"] = ks_param
        DRIFT_THRESHOLDS["psi_moderate"] = psi_mod_param
        DRIFT_THRESHOLDS["psi_severe"] = psi_sev_param

        with st.spinner(f"Evaluating `{selected_batch_name}` across 8 statistical tests..."):
            try:
                selected_path = PRODUCTION_DIR / selected_batch_name
                baseline_data = load_baseline_data()
                production_data = load_production_batch(selected_path)

                drift_report = run_full_drift_analysis(baseline_data, production_data)

                performance_result = None
                confidence_telemetry = None
                try:
                    inference_output = predict_batch(production_data)
                    performance_result = inference_output.get("comparison")
                    confidence_telemetry = inference_output.get("confidence_stats")
                except FileNotFoundError:
                    st.warning("⚠️ No trained ML model found. Train model via `python -m src.train` to enable performance tracking.")
                except Exception as pred_err:
                    st.warning(f"⚠️ Inference evaluation notice: {pred_err}")

                alerts_list = generate_alerts(
                    drift_report,
                    batch_name=selected_batch_name,
                    performance_comparison=performance_result,
                )
                generated_reports = generate_drift_report(
                    drift_report,
                    batch_name=selected_batch_name,
                    performance=performance_result,
                    alerts=alerts_list,
                )

                st.session_state.update({
                    "drift_report": drift_report,
                    "performance": performance_result,
                    "confidence_stats": confidence_telemetry,
                    "baseline_df": baseline_data,
                    "production_df": production_data,
                    "batch_name": selected_batch_name,
                    "report_paths": generated_reports,
                    "analysis_complete": True,
                })

                st.success(f"✅ Drift detection completed successfully for `{selected_batch_name}`!")
                st.balloons()

                # Executive Summary Cards
                dataset_metrics = drift_report.get("dataset_drift", {})
                risk_level_found = dataset_metrics.get("risk_level", "LOW")
                drifted_n = dataset_metrics.get("drifted_count", 0)
                total_n = dataset_metrics.get("total_features", 0)

                st.markdown("<br>", unsafe_allow_html=True)
                k1, k2, k3 = st.columns(3)
                risk_colors = {"LOW": "#10b981", "MEDIUM": "#f59e0b", "HIGH": "#ef4444"}

                with k1:
                    kpi_card("Evaluated Risk Tier", risk_level_found, "🛡️", risk_colors.get(risk_level_found, "#3b82f6"))
                with k2:
                    kpi_card("Drifted Features", f"{drifted_n} / {total_n}", "📊", CHART_COLORS["blue"])
                with k3:
                    h_val = performance_result.get("overall_health", "No Target") if performance_result else "No Target"
                    kpi_card("Model Health", h_val, "🤖", risk_colors.get(h_val, "#94a3b8"))

                # Navigation Shortcut
                st.markdown("<br>", unsafe_allow_html=True)
                nav_col1, nav_col2 = st.columns([1, 1])
                with nav_col1:
                    if st.button("📊 View Full Overview Dashboard", type="secondary", width="stretch"):
                        st.session_state["nav_page"] = "🏠  Overview"
                        st.rerun()
                with nav_col2:
                    if st.button("🔬 Inspect Feature Distributions", type="secondary", width="stretch"):
                        st.session_state["nav_page"] = "🔬  Drift Analysis"
                        st.rerun()

                # Report Export Section
                st.markdown("<br>", unsafe_allow_html=True)
                section_header("Export Generated Audit Reports", "📥")
                dl_col1, dl_col2 = st.columns(2)
                for idx, (fmt, report_filepath) in enumerate(generated_reports.items()):
                    col_target = dl_col1 if idx == 0 else dl_col2
                    with col_target:
                        if Path(report_filepath).exists():
                            with open(report_filepath, "rb") as f_report:
                                st.download_button(
                                    label=f"📄 Download {fmt.upper()} Audit Report",
                                    data=f_report.read(),
                                    file_name=Path(report_filepath).name,
                                    mime="application/json" if fmt == "json" else "text/markdown",
                                    key=f"download_report_{fmt}",
                                    width="stretch",
                                )

            except DataValidationError as val_ex:
                st.error(f"❌ Ingestion schema validation failed: {val_ex}")
            except Exception as exec_ex:
                st.error(f"❌ Analysis failed: {exec_ex}")
                import traceback
                with st.expander("Diagnostic Traceback"):
                    st.code(traceback.format_exc())
