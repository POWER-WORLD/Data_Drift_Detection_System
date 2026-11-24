"""
Exhaustive verification harness for all backend functions in src/.
Tests every single function with standard inputs and extreme edge cases.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.settings import (
    BASELINE_FILE, PRODUCTION_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR,
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES, ALL_FEATURES, TARGET_COLUMN, ID_COLUMN
)
import src.utils as utils
import src.data_validator as validator
import src.data_loader as loader
import src.drift_detection as drift
import src.preprocessing as prep
import src.metrics as metrics
import src.train as train
import src.predict as predict
import src.alert as alert
import src.report as report

def run_tests():
    print("=" * 70)
    print("STARTING EXHAUSTIVE BACKEND FUNCTION VERIFICATION")
    print("=" * 70)
    
    # ── 1. src/utils.py ──────────────────────────────────────────────────
    print("\n[1] Testing src/utils.py...")
    # json_default_serializer
    assert utils.json_default_serializer(np.int64(42)) == 42
    assert utils.json_default_serializer(np.float64(3.14)) == 3.14
    assert utils.json_default_serializer(np.nan) is None
    assert utils.json_default_serializer(np.inf) == "Infinity"
    assert utils.json_default_serializer(-np.inf) == "-Infinity"
    assert utils.json_default_serializer(np.bool_(True)) is True
    assert utils.json_default_serializer(np.array([1, 2])) == [1, 2]
    assert utils.json_default_serializer(Path("foo/bar")) == "foo\\bar" or "foo/bar" in utils.json_default_serializer(Path("foo/bar"))
    
    # setup_logger
    logger = utils.setup_logger("test_exhaust_logger")
    logger.info("Test log line")
    
    # timestamps
    ts = utils.get_timestamp()
    rts = utils.get_readable_timestamp()
    assert len(ts) > 0 and len(rts) > 0
    
    # save_json & load_json (atomic)
    test_json_path = LOGS_DIR / "test_exhaust.json"
    test_data = {"key": "value", "np_num": np.int64(10), "nan_val": np.nan}
    utils.save_json(test_data, test_json_path)
    loaded = utils.load_json(test_json_path)
    import math
    assert loaded["key"] == "value"
    assert loaded["np_num"] == 10
    assert loaded["nan_val"] is None or math.isnan(loaded["nan_val"])
    if test_json_path.exists():
        test_json_path.unlink()
        
    # ensure_dir
    test_dir = LOGS_DIR / "sub_exhaust"
    assert utils.ensure_dir(test_dir).exists()
    test_dir.rmdir()
    print("✓ src/utils.py passed all tests.")

    # ── 2. src/data_validator.py ──────────────────────────────────────────
    print("\n[2] Testing src/data_validator.py...")
    # validate_file
    validator.validate_file(BASELINE_FILE)
    try:
        validator.validate_file(Path("non_existent_file.csv"))
        assert False, "Should raise DataValidationError"
    except validator.DataValidationError:
        pass
        
    # sanitize_filename
    assert validator.sanitize_filename("../../../etc/passwd.txt") == "passwd.txt.csv"
    assert validator.sanitize_filename("valid_batch.csv") == "valid_batch.csv"
    assert validator.sanitize_filename("   ") == "uploaded_batch.csv"
    assert validator.sanitize_filename(".csv") == "uploaded_batch.csv"

    # validate_schema
    base_df = pd.read_csv(BASELINE_FILE)
    warnings = validator.validate_schema(base_df, require_target=True)
    assert isinstance(warnings, list)
    
    # Missing columns
    bad_df = base_df.drop(columns=["tenure"])
    try:
        validator.validate_schema(bad_df)
        assert False, "Should raise for missing feature"
    except validator.DataValidationError:
        pass

    # validate_data_quality
    # 1. Normal
    q_norm = validator.validate_data_quality(base_df)
    assert q_norm["passed"] is True
    
    # 2. Empty df
    q_empty = validator.validate_data_quality(pd.DataFrame())
    assert q_empty["passed"] is False
    
    # 3. All NaN rows & high missingness & infinities
    edge_df = base_df.head(20).copy()
    edge_df.iloc[0:2] = np.nan
    edge_df.loc[3, "MonthlyCharges"] = np.inf
    edge_df.loc[4, "tenure"] = -np.inf
    edge_df["high_nan_col"] = np.nan
    q_edge = validator.validate_data_quality(edge_df)
    assert len(q_edge["issues"]) > 0
    
    # 4. Zero variance with and without inf
    zv_df = pd.DataFrame({
        "customerID": [f"id_{i}" for i in range(20)],
        "Churn": ["No"] * 20,
        "tenure": [10.0] * 19 + [np.inf],
        "MonthlyCharges": [50.0] * 20,
        "TotalCharges": [100.0] * 20,
        "SeniorCitizen": [0] * 20
    })
    for c in CATEGORICAL_FEATURES:
        zv_df[c] = "A"
    q_zv = validator.validate_data_quality(zv_df)
    assert any("Zero-variance" in str(iss) for iss in q_zv["issues"])
    
    # sanitize_dataframe
    sanitized = validator.sanitize_dataframe(edge_df)
    assert not np.isinf(sanitized["MonthlyCharges"]).any()
    assert not np.isinf(sanitized["tenure"]).any()
    print("✓ src/data_validator.py passed all tests.")

    # ── 3. src/data_loader.py ─────────────────────────────────────────────
    print("\n[3] Testing src/data_loader.py...")
    loaded_base = loader.load_baseline_data()
    assert len(loaded_base) > 0
    assert "TotalCharges" in loaded_base.columns
    assert loaded_base["TotalCharges"].dtype in (np.float64, float)
    
    batches = loader.list_production_batches()
    assert len(batches) >= 5
    for b in batches:
        loaded_b = loader.load_production_batch(b)
        assert len(loaded_b) > 0
    print("✓ src/data_loader.py passed all tests.")

    # ── 4. src/drift_detection.py ─────────────────────────────────────────
    print("\n[4] Testing src/drift_detection.py...")
    s1 = pd.Series(np.random.normal(50, 10, 500))
    s2 = pd.Series(np.random.normal(70, 15, 500))
    s_const = pd.Series([5.0] * 100)
    s_empty = pd.Series(dtype=float)
    s_nan = pd.Series([np.nan] * 50)
    s_inf = pd.Series([1.0, 2.0, np.inf, -np.inf, 3.0] * 20)
    s_small = pd.Series([1.0, 2.0])

    s_single = pd.Series([1.0])

    # ks_test
    assert drift.ks_test(s1, s2)["drift_detected"] is True
    assert drift.ks_test(s_empty, s1)["drift_detected"] is False
    assert drift.ks_test(s_nan, s1)["drift_detected"] is False
    assert drift.ks_test(s_small, s1)["warning"] == "Insufficient samples"
    assert drift.ks_test(s_inf, s1)["statistic"] is not None

    # compute_psi
    psi_res = drift.compute_psi(s1, s2)
    assert psi_res["value"] >= 0
    assert drift.compute_psi(s_const, s_const)["severity"] == "LOW"
    assert drift.compute_psi(s_const, s1)["severity"] in ("LOW", "MODERATE", "SEVERE")
    assert drift.compute_psi(s_empty, s1)["drift_detected"] is False

    # jensen_shannon_distance
    js_res = drift.jensen_shannon_distance(s1, s2)
    assert 0.0 <= js_res["value"] <= 1.0
    assert drift.jensen_shannon_distance(s_const, s_const)["value"] == 0.0
    assert drift.jensen_shannon_distance(s_empty, s1)["drift_detected"] is False

    # mean_shift_test
    ms_res = drift.mean_shift_test(s1, s2)
    assert ms_res["pct_change"] >= 0
    assert drift.mean_shift_test(pd.Series([0.0]*20), pd.Series([5.0]*20))["pct_change"] == 100.0
    assert drift.mean_shift_test(s_single, s1)["warning"] == "Insufficient samples"

    # variance_change_test
    vc_res = drift.variance_change_test(s1, s2)
    assert vc_res["pct_change"] >= 0
    assert drift.variance_change_test(pd.Series([5.0]*20), pd.Series([5.0]*20))["pct_change"] == 0.0
    assert drift.variance_change_test(s_single, s1)["warning"] == "Insufficient samples"

    # chi_square_test
    c1 = pd.Series(["A", "B", "C"] * 100)
    c2 = pd.Series(["A", "A", "B"] * 100)
    c_single = pd.Series(["A"] * 100)
    assert drift.chi_square_test(c1, c2)["test"] == "Chi-Square"
    assert drift.chi_square_test(c_single, c_single)["warning"] == "Only one unique category present"
    assert drift.chi_square_test(pd.Series(dtype=str), c1)["warning"] == "Insufficient samples"

    # category_frequency_shift
    cfs = drift.category_frequency_shift(c1, c2)
    assert "shifts" in cfs
    assert drift.category_frequency_shift(pd.Series(dtype=str), c1)["max_shift_pct"] == 0.0

    # new_category_detection
    ncd = drift.new_category_detection(c1, pd.Series(["A", "B", "NEW_CAT"] * 50))
    assert ncd["drift_detected"] is True
    assert "NEW_CAT" in ncd["new_categories"]

    # full analysis
    full_report = drift.run_full_drift_analysis(loaded_base.head(200), loaded_base.tail(200))
    assert "dataset_drift" in full_report
    assert "feature_drift" in full_report
    print("✓ src/drift_detection.py passed all tests.")

    # ── 5. src/preprocessing.py ───────────────────────────────────────────
    print("\n[5] Testing src/preprocessing.py...")
    p = prep.DataPreprocessor()
    X, y = p.fit_transform(loaded_base.head(200))
    assert len(X) == 200
    assert len(y) == 200
    assert p.is_fitted is True
    
    # transform with missing values and new categories
    prod_test = loaded_base.head(50).copy()
    prod_test.loc[0, "MonthlyCharges"] = np.nan
    prod_test.loc[1, "PaymentMethod"] = "NewUnknownMethod"
    prod_test = prod_test.drop(columns=["tenure"]) # missing column test
    X_p, y_p = p.transform(prod_test)
    assert len(X_p) == 50
    assert "tenure" in X_p.columns
    assert not X_p.isna().any().any()
    
    # save & load
    test_prep_path = MODELS_DIR / "test_prep.joblib"
    p.save(test_prep_path)
    p_loaded = prep.DataPreprocessor.load(test_prep_path)
    assert p_loaded.is_fitted is True
    test_prep_path.unlink()
    print("✓ src/preprocessing.py passed all tests.")

    # ── 6. src/metrics.py ─────────────────────────────────────────────────
    print("\n[6] Testing src/metrics.py...")
    y_t = np.array([0, 1, 0, 1, 0, 1])
    y_p = np.array([0, 1, 0, 0, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.4, 0.3, 0.8])
    
    m_res = metrics.evaluate_model(y_t, y_p, y_prob)
    assert m_res["accuracy"] > 0
    assert m_res["roc_auc"] is not None
    
    # Single class y_true
    m_single = metrics.evaluate_model(np.array([1, 1, 1]), np.array([1, 1, 1]), np.array([0.9, 0.8, 0.9]))
    assert m_single["roc_auc"] is None
    
    # Empty
    m_empty = metrics.evaluate_model(np.array([]), np.array([]))
    assert m_empty["accuracy"] == 0.0

    # compare_performance
    base_m = {"accuracy": 0.8, "precision": 0.8, "recall": 0.8, "f1_score": 0.8, "roc_auc": 0.8}
    curr_m = {"accuracy": 0.7, "precision": 0.7, "recall": 0.7, "f1_score": 0.7, "roc_auc": 0.7}
    comp = metrics.compare_performance(base_m, curr_m)
    assert comp["overall_health"] == "CRITICAL"
    assert comp["degraded_metrics_count"] == 5

    # confidence stats
    conf = metrics.get_prediction_confidence_stats(y_prob)
    assert 0.0 <= conf["mean_confidence"] <= 1.0
    conf_empty = metrics.get_prediction_confidence_stats(np.array([]))
    assert conf_empty["mean_confidence"] == 0.0
    print("✓ src/metrics.py passed all tests.")

    # ── 7. src/train.py & predict.py ──────────────────────────────────────
    print("\n[7] Testing src/train.py and src/predict.py...")
    model = train.load_trained_model()
    assert model is not None
    models_dict = train.get_models(scale_pos_weight=2.5)
    assert len(models_dict) == 4
    
    # predict_batch on baseline sample
    pred_res = predict.predict_batch(loaded_base.head(50))
    assert len(pred_res["predictions"]) == 50
    assert "comparison" in pred_res
    
    # unlabelled batch
    unlabelled = loaded_base.head(50).drop(columns=["Churn"])
    pred_unlabelled = predict.predict_batch(unlabelled)
    assert len(pred_unlabelled["predictions"]) == 50
    assert "comparison" not in pred_unlabelled
    
    # empty batch
    pred_empty = predict.predict_batch(pd.DataFrame())
    assert len(pred_empty["predictions"]) == 0
    print("✓ src/train.py and src/predict.py passed all tests.")

    # ── 8. src/alert.py & report.py ───────────────────────────────────────
    print("\n[8] Testing src/alert.py and src/report.py...")
    alert.clear_alerts()
    assert alert.get_all_alerts() == []
    
    sample_report = drift.run_full_drift_analysis(loaded_base.head(100), loaded_base.tail(100))
    alerts = alert.generate_alerts(sample_report, batch_name="exhaust_test")
    assert len(alerts) > 0
    assert len(alert.get_all_alerts()) == len(alerts)
    
    email = alert.simulate_email_alert(alerts[0])
    assert "DRIFT ALERT EMAIL SIMULATION" in email
    alert.clear_alerts()
    
    # generate_drift_report
    rep_paths = report.generate_drift_report(sample_report, batch_name="exhaust_test", alerts=alerts)
    assert rep_paths["json"].exists()
    assert rep_paths["csv"].exists()
    assert rep_paths["alerts_csv"].exists()
    
    # helpers in report.py
    rec_h = report._get_recommendation({"dataset_drift": {"risk_level": "HIGH"}})
    rec_m = report._get_recommendation({"dataset_drift": {"risk_level": "MEDIUM"}})
    rec_l = report._get_recommendation({"dataset_drift": {"risk_level": "LOW"}})
    assert "IMMEDIATE ACTION" in rec_h
    assert "SCHEDULE RETRAINING" in rec_m
    assert "NO ACTION NEEDED" in rec_l
    
    summary = report._summarize_features(sample_report)
    assert "total_features" in summary
    print("✓ src/alert.py and src/report.py passed all tests.")

    print("\n" + "=" * 70)
    print("ALL BACKEND FUNCTIONS AND EXTREME EDGE CASES VERIFIED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
