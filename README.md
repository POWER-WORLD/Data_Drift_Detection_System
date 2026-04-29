# 📊 Data Drift Detection System for Machine Learning Models

A **production-grade, enterprise-ready** monitoring system that detects when incoming real-world data distribution changes compared to historical training data, causing ML models to become less reliable. Built for production ML pipelines, professional portfolios, and real-world deployment.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?style=flat-square&logo=scikitlearn)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green?style=flat-square&logo=xgboost)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?style=flat-square&logo=streamlit)
![Tests](https://img.shields.io/badge/Tests-73%20passing-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## 🎯 Why Data Drift Matters

Machine learning models are trained on historical data, but real-world data changes over time. When the distribution of incoming data differs significantly from training data, model predictions become unreliable. This phenomenon is called **data drift**, and it's one of the most common reasons production ML models fail silently.

**This system solves that problem by:**
- Continuously monitoring incoming data for distribution changes
- Detecting drift at both feature and dataset level using rigorous statistical tests
- Tracking model performance degradation
- Generating actionable alerts and recommendations
- Providing an interactive dashboard for visualization
- Validating data quality before processing

---

## ✨ Features

### 🔬 Drift Detection Engine (8 Statistical Tests)
| Method | Type | Description |
|--------|------|-------------|
| Kolmogorov-Smirnov Test | Numerical | Statistical test for distribution difference |
| Population Stability Index (PSI) | Numerical | Industry-standard drift metric |
| Jensen-Shannon Distance | Numerical | Symmetric divergence measure |
| Mean Shift Detection | Numerical | Detects changes in central tendency |
| Variance Change Detection | Numerical | Detects changes in spread |
| Chi-Square Test | Categorical | Tests independence of distributions |
| Category Frequency Shift | Categorical | Measures category proportion changes |
| New Category Detection | Categorical | Identifies unseen categories |

All tests include **minimum sample-size guards** and return native Python types for safe JSON serialization.

### 🤖 Machine Learning Pipeline
- **4 Algorithms**: Logistic Regression, Random Forest, Gradient Boosting, XGBoost
- **Class Imbalance Handling**: `class_weight='balanced'` and `scale_pos_weight` for fair minority-class treatment
- **Stratified 5-Fold Cross-Validation**: Reliable model selection based on mean CV F1
- **Reproducibility**: Fixed `random_state=42` across all operations
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1, ROC AUC, Confusion Matrix

### 🛡️ Data Validation Layer
- Schema enforcement (required columns, types)
- Data quality checks (NaN, infinity, zero-variance, duplicates)
- File validation (size limits, extension checks)
- Input sanitization (filename cleaning, path traversal prevention)
- Graceful error messages for all failure modes

### 📊 Interactive Dashboard (Streamlit)
- **Overview**: Risk gauge, summary cards, drifted features breakdown
- **Feature Analysis**: PSI charts, distribution comparison, statistical test tables
- **Model Health**: Baseline vs production metrics, confusion matrix heatmap, prediction confidence
- **Alert Center**: Filterable alert timeline with severity levels
- **Upload & Run**: Upload batches, configure thresholds, export reports

### 🔔 Alert System
- **3 Severity Levels**: INFO, WARNING, CRITICAL
- **Automatic Generation**: Alerts triggered by drift and performance thresholds
- **Persistent Storage**: Alert history in JSON with corruption recovery

### 📈 Simulated Production Scenarios
| Scenario | Description |
|----------|-------------|
| No Drift | Random sample from baseline (control) |
| Mild Drift | Slight shifts in numerical features |
| Severe Drift | Large distribution changes across features |
| Missing Data | Sudden increase in null values |
| Category Drift | New unseen categorical values |

---

## 🏗️ Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌───────────────┐
│  Data Layer  │────▶│  Validation &    │────▶│   ML Layer    │
│  (CSV/Batch) │     │  Preprocessing   │     │ (Train/Pred)  │
└──────────────┘     └──────────────────┘     └───────┬───────┘
                                                       │
┌──────────────┐     ┌──────────────────┐     ┌───────▼───────┐
│  Dashboard   │◀────│ Monitoring Layer │◀────│ Drift Engine  │
│ (Streamlit)  │     │ (Alerts/Logs)    │     │ (Statistics)  │
└──────────────┘     └──────────────────┘     └───────────────┘
```

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Core** | Python 3.10+, pandas, NumPy |
| **ML** | scikit-learn, XGBoost |
| **Statistics** | SciPy |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Dashboard** | Streamlit |
| **Serialization** | Joblib |
| **Testing** | pytest (73 tests) |
| **Deployment** | Docker |

---

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/data-drift-detection-system.git
cd data-drift-detection-system
```

### 2. Create Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train Baseline Model & Generate Production Batches
```bash
python -m src.train
```

### 5. Launch Dashboard
```bash
streamlit run dashboard/app.py
```

### Docker (Alternative)
```bash
docker build -t drift-detector .
docker run -p 8501:8501 drift-detector
```

---

## 🚀 Usage

### Quick Start
1. Run `python -m src.train` to train the baseline model
2. Run `streamlit run dashboard/app.py` to launch the dashboard
3. Navigate to **Upload & Run** page
4. Select a production batch (e.g., `batch_03_severe_drift.csv`)
5. Click **Run Drift Detection**
6. Explore results across all dashboard pages

### Programmatic Usage
```python
from src.data_loader import load_baseline_data, load_production_batch
from src.drift_detection import run_full_drift_analysis
from src.predict import predict_batch

# Load data
baseline = load_baseline_data()
production = load_production_batch("data/production/batch_03_severe_drift.csv")

# Run drift analysis
drift_report = run_full_drift_analysis(baseline, production)
print(f"Risk Level: {drift_report['dataset_drift']['risk_level']}")
print(f"Drifted Features: {drift_report['dataset_drift']['drifted_count']}")

# Run predictions and compare performance
result = predict_batch(production)
print(f"Model Health: {result['comparison']['overall_health']}")
```

---

## 🧪 Testing

```bash
# Run all 73 tests
python -m pytest tests/ -v

# Run specific module
python -m pytest tests/test_drift_detection.py -v

# Run integration tests only
python -m pytest tests/test_integration.py -v
```

### Test Coverage
| Module | Tests | Coverage |
|--------|-------|----------|
| Data Loader | 8 | Schema validation, file loading |
| Data Validator | 17 | Schema, quality, sanitization, file checks |
| Drift Detection | 24 | All stats, edge cases, serialization |
| Preprocessing | 7 | Transform, NaN handling, save/load |
| Training | 5 | Artifacts, metrics, model selection |
| Integration | 12 | End-to-end pipeline, edge scenarios |
| **Total** | **73** | **All passing** |

---

## 📁 Project Structure

```
data-drift-detection-system/
├── data/
│   ├── baseline/
│   │   └── telco_churn.csv
│   └── production/
│       ├── batch_01_no_drift.csv
│       ├── batch_02_mild_drift.csv
│       ├── batch_03_severe_drift.csv
│       ├── batch_04_missing_data.csv
│       └── batch_05_category_drift.csv
├── models/
│   ├── best_model.joblib
│   ├── preprocessor.joblib
│   ├── baseline_metrics.json
│   └── training_summary.json
├── reports/
├── logs/
│   ├── drift_system.log (rotating, 5MB × 3)
│   └── alerts.json
├── src/
│   ├── config/
│   │   └── settings.py
│   ├── data_loader.py
│   ├── data_validator.py          # NEW — Schema + quality validation
│   ├── preprocessing.py
│   ├── train.py                   # Enhanced — CV + class weights
│   ├── predict.py
│   ├── drift_detection.py         # Enhanced — sample guards
│   ├── metrics.py
│   ├── alert.py
│   ├── report.py
│   └── utils.py                   # Enhanced — rotating logs
├── dashboard/
│   ├── app.py
│   ├── components.py
│   ├── styles.py
│   └── pages/
│       ├── overview.py
│       ├── feature_analysis.py
│       ├── model_health.py         # Enhanced — confusion matrix
│       ├── alerts.py
│       └── upload.py               # Enhanced — security hardening
├── tests/
│   ├── conftest.py                 # NEW — shared path setup
│   ├── test_data_loader.py
│   ├── test_data_validator.py      # NEW — 17 validation tests
│   ├── test_drift_detection.py     # Enhanced — 24 tests with edge cases
│   ├── test_integration.py         # NEW — 12 end-to-end tests
│   ├── test_preprocessing.py
│   └── test_train.py
├── .gitignore
├── .streamlit/config.toml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## ⚙️ Configuration

All thresholds and settings are configurable in `src/config/settings.py`:

```python
DRIFT_THRESHOLDS = {
    "ks_pvalue": 0.05,               # KS test significance level
    "psi_moderate": 0.10,            # PSI moderate drift threshold
    "psi_severe": 0.25,              # PSI severe drift threshold
    "chi2_pvalue": 0.05,             # Chi-Square significance level
    "mean_shift_pct": 0.10,          # 10% mean change threshold
    "variance_change_pct": 0.20,     # 20% variance change threshold
    "drifted_features_pct_moderate": 0.20,  # 20% features = moderate risk
    "drifted_features_pct_severe": 0.50,    # 50% features = high risk
}
```

Thresholds can also be adjusted per-run via the dashboard's **Threshold Configuration** panel.

---

## 🔐 Security Features

- **File size limits**: Uploads capped at 50 MB
- **Filename sanitization**: Path traversal characters stripped
- **Schema validation**: Uploaded CSVs verified before processing
- **Error isolation**: Corrupt files produce user-friendly error messages
- **No hardcoded credentials**: Configuration via settings module
- **Hidden Streamlit branding**: Production-ready UI appearance

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'src'` | Run from project root: `streamlit run dashboard/app.py` |
| `FileNotFoundError: best_model.joblib` | Run `python -m src.train` first |
| Dashboard shows "No analysis run yet" | Go to Upload & Run page and run detection |
| Tests fail with import errors | Ensure `conftest.py` exists in tests/ |
| Logs filling disk | Logs auto-rotate at 5MB (3 backups) |

---

## 🔮 Future Roadmap

- **REST API**: FastAPI endpoint for programmatic drift checks
- **Real-time Streaming**: Kafka/Spark integration for live monitoring
- **Cloud Deployment**: AWS/GCP/Azure deployment guides
- **Automated Retraining**: MLflow integration for model versioning
- **Role-based Access**: Dashboard login and permissions
- **Database Backend**: PostgreSQL for persistent drift history
- **CI/CD Pipeline**: GitHub Actions for automated testing
- **SHAP Explainability**: Feature importance for drifted predictions

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- **Dataset**: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle
- **Tools**: scikit-learn, XGBoost, Streamlit, Plotly, SciPy
