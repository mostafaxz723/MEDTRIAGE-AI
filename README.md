# 🏥 MEDTRIAGE AI - Clinical Emergency Triage & Decision Support System

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production%20Ready-009688.svg)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5+-F7931E.svg)](https://scikit-learn.org/)
[![Model Accuracy](https://img.shields.io/badge/Model%20Accuracy-94.27%25-brightgreen.svg)]()

Production-grade Machine Learning clinical decision support system for Emergency Departments (ED). The system automates patient triage acuity assignment according to the **Emergency Severity Index (ESI 4-Level Model)**, predicts probability distributions across urgency tiers, detects physiological red flags, and provides batch emergency intake processing alongside a high-throughput REST API.

---

## 📑 Table of Contents
- [Clinical Triage Hierarchy](#-clinical-triage-hierarchy)
- [Key Features](#-key-features)
- [Project Architecture](#-project-architecture)
- [Quick Start: Local Deployment](#-quick-start-local-deployment)
- [Cloud & Container Deployments](#-cloud--container-deployments)
  - [Docker & Docker Compose](#1-docker--docker-compose)
  - [Streamlit Community Cloud](#2-streamlit-community-cloud)
  - [Render / Heroku / Railway](#3-render--heroku--railway)
- [REST API Reference & EHR Integration](#-rest-api-reference--ehr-integration)
- [Model Provenance & Evaluation](#-model-provenance--evaluation)

---

## 🩺 Clinical Triage Hierarchy

The system classifies incoming emergency patients into 4 standardized acuity levels:

| Acuity Level | Short Name | Target Response | Clinical Protocol | Visual Indicator |
| :--- | :--- | :--- | :--- | :--- |
| **Level 0** | **Non-Urgent** | 120 - 240 mins | Safe for routine ambulatory queue or outpatient clinic evaluation. | 🟢 Green (`#10B981`) |
| **Level 1** | **Semi-Urgent** | Within 60 mins | Focused diagnostic workup (labs/imaging), symptom relief, and periodic vital checks. | 🟡 Amber (`#F59E0B`) |
| **Level 2** | **Urgent** | Within 15 - 30 mins | Acute physiological distress, marked vital sign deviations, or high pain. Immediate bed assignment. | 🟠 Orange (`#F97316`) |
| **Level 3** | **Critical / Resuscitation** | Immediate (0 mins) | **LIFE-THREATENING EMERGENCY**: Severe vital instability or impending collapse. Transfer directly to Resuscitation Bay. | 🔴 Red (`#EF4444`) |

---

## ✨ Key Features

1. **Interactive Single-Patient Triage Calculator**:
   - Real-time prediction with immediate visual feedback upon changing sliders and clinical inputs.
   - Pre-configured rapid clinical presets (Routine Adult, Moderate Pain, Acute Distress, Critical Resuscitation).
   - Multi-class probability distribution bar charts for clear transparency.
   - 1-click Downloadable Clinical Intake Slip (JSON).
2. **Automated Vital Sign Safety & Risk Alerts**:
   - Clinical threshold evaluation for:
     - Hypoxemia ($\text{SpO}_2 < 90\%$ critical, $90-93\%$ warning)
     - Tachycardia ($\text{HR} > 130$ critical, $> 100$ warning) & Bradycardia ($\text{HR} < 50$)
     - Hypertensive Crisis ($\text{SBP} \ge 180$ mmHg) & Severe Hypotension ($\text{SBP} < 90$ mmHg)
     - Hyperpyrexia ($\text{Temp} \ge 39.5^\circ\text{C}$) & Febrile illness ($\ge 38.3^\circ\text{C}$)
     - Severe acute pain ($\ge 8/10$)
3. **High-Volume Emergency Batch Triage**:
   - Upload CSV rosters of incoming patients.
   - Instant multi-acuity summary counters (Total, Critical, Urgent, Semi-Urgent, Non-Urgent).
   - Priority-filtered interactive queue and downloadable prioritized CSV.
4. **Standalone REST API**:
   - High-performance FastAPI backend (`api.py`) for EHR (Epic, Cerner), triage kiosks, and mobile applications.
   - Interactive OpenAPI / Swagger UI documentation at `/docs`.
5. **Turnkey Deployment Configurations**:
   - Standardized `requirements.txt`, `Dockerfile`, `docker-compose.yml`, `Procfile`, and Windows `.bat` launchers.

---

## 🏛️ Project Architecture

```
Medical/
├── Medical-Triage_model.pkl   # Serialized Multinomial Logistic Regression model
├── scaler.pkl                 # Serialized StandardScaler fitted on clinical features
├── synthetic_medical_triage.csv # Benchmark dataset (18,000 clinical records)
├── Medical_Triage_note.ipynb  # Original model training and EDA notebook
│
├── triage_predictor.py        # Core inference engine & clinical vital signs safety analyzer
├── app.py                     # Streamlit clinical dashboard & decision support UI
├── api.py                     # Production FastAPI REST service
│
├── requirements.txt           # Pinned production dependencies
├── runtime.txt                # Cloud build runtime specification (Python 3.11.9)
├── Dockerfile                 # Multi-stage container definition
├── docker-compose.yml         # Compose config for concurrent Web + API services
├── Procfile                   # Cloud platform process declaration (Render/Heroku)
├── render.yaml                # Render Infrastructure-as-Code specification
│
├── .streamlit/
│   └── config.toml            # Server port, headless mode, and healthcare theme settings
├── run_app.bat                # 1-Click Windows launcher for Web App
├── run_api.bat                # 1-Click Windows launcher for REST API
├── test_deployment.py         # Automated QA and deployment test suite
└── README.md                  # Comprehensive deployment & architecture manual
```

---

## 🚀 Quick Start: Local Deployment

### Option A: 1-Click Launchers (Windows)
Double-click `run_app.bat` to launch the web dashboard, or `run_api.bat` to start the REST API.

### Option B: Command Line (PowerShell / Bash)
1. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   # Windows PowerShell:
   .venv\Scripts\Activate.ps1
   # Linux / macOS:
   source .venv/bin/activate
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Streamlit Web Application**:
   ```bash
   streamlit run app.py
   ```
   Access the web app at: **`http://localhost:8501`**

4. **Run the REST API Service**:
   ```bash
   python api.py
   # or via uvicorn directly:
   uvicorn api:app --host 0.0.0.0 --port 8000 --reload
   ```
   Interactive API documentation: **`http://localhost:8000/docs`**

---

## ☁️ Cloud & Container Deployments

### 1. Docker & Docker Compose
To run both the Web App and REST API in isolated containers:
```bash
docker compose up --build
```
- Web Application: **`http://localhost:8501`**
- REST API Service: **`http://localhost:8000`**

### 2. Streamlit Community Cloud (Free 1-Click Deploy)
1. Push this repository to GitHub.
2. Visit [share.streamlit.io](https://share.streamlit.io/).
3. Select your repository, branch (`main`), and set Main file path to `app.py`.
4. Click **Deploy!**

### 3. Render / Heroku / Railway
The included `Procfile` and `render.yaml` automatically configure deployment:
```yaml
# Build Command:
pip install -r requirements.txt

# Start Command:
streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

---

## ⚡ REST API Reference & EHR Integration

### Endpoint: Health Check
`GET /health`
```json
{
  "status": "healthy",
  "model_loaded": true,
  "scaler_loaded": true,
  "features_expected": ["age", "heart_rate", "systolic_blood_pressure", "oxygen_saturation", "body_temperature", "pain_level", "chronic_disease_count", "previous_er_visits", "arrival_mode"]
}
```

### Endpoint: Predict Single Patient
`POST /predict`
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "PT-10492",
    "age": 62.0,
    "heart_rate": 135.0,
    "systolic_blood_pressure": 168.0,
    "oxygen_saturation": 89.5,
    "body_temperature": 39.1,
    "pain_level": 8,
    "chronic_disease_count": 3,
    "previous_er_visits": 2,
    "arrival_mode": "ambulance"
  }'
```

**Response**:
```json
{
  "patient_id": "PT-10492",
  "triage_level": 3,
  "triage_name": "Level 3 - Immediate (Resuscitation / Critical)",
  "short_name": "Critical / Resuscitation",
  "priority": "CRITICAL",
  "target_time": "IMMEDIATE (0 mins)",
  "clinical_action": "EMERGENCY: Imminent life threat or hemodynamic collapse. Transfer directly to Resuscitation / Trauma Suite.",
  "badge_color": "#EF4444",
  "confidence_score": 98.42,
  "class_probabilities": {
    "Level 0": 0.0,
    "Level 1": 0.0,
    "Level 2": 0.0158,
    "Level 3": 0.9842
  },
  "vital_flags": [
    {
      "severity": "critical",
      "parameter": "Oxygen Saturation",
      "value": "89.5%",
      "message": "Critical Hypoxemia (SpO2 < 90%). Administer supplemental O2 immediately."
    },
    {
      "severity": "critical",
      "parameter": "Heart Rate",
      "value": "135 bpm",
      "message": "Severe Tachycardia (HR > 130 bpm). Risk of hemodynamic instability."
    }
  ],
  "critical_flags_count": 2,
  "warning_flags_count": 0
}
```

---

## 📊 Model Provenance & Evaluation

- **Model Type**: Multinomial Logistic Regression (`max_iter=1000`)
- **Scaling**: `StandardScaler` (Mean-variance standardizer)
- **Acuity Classes**: 4 Acuity Tiers (`0: Non-Urgent`, `1: Semi-Urgent`, `2: Urgent`, `3: Critical`)
- **Pre-processing Protocol**: Heart rate clipped to `[40, 200] bpm`; categorical `arrival_mode` label-encoded (`ambulance: 0`, `walk_in: 1`, `wheelchair: 2`).
- **Benchmark Performance (18,000 cases)**:
  - **Overall Accuracy**: **94.27%**
  - **Level 0 (Non-Urgent)**: Precision 0.96 | Recall 0.98 | F1-Score 0.97
  - **Level 1 (Semi-Urgent)**: Precision 0.91 | Recall 0.88 | F1-Score 0.89
  - **Level 2 (Urgent)**: Precision 0.93 | Recall 0.92 | F1-Score 0.93
  - **Level 3 (Critical)**: Precision 0.96 | Recall 0.94 | F1-Score 0.95

---

## 🔒 Clinical Disclaimer
*MedTriage AI is intended as a clinical decision support tool to augment triage nursing and medical staff. It does not replace independent clinical judgment or certified emergency medical evaluation.*
