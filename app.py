"""
MEDTRIAGE AI - Clinical Emergency Triage & Decision Support Web Application
Production Streamlit Application for Emergency Departments and Healthcare Centers.
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import json
import time

from triage_predictor import get_predictor, TRIAGE_LEVELS, FEATURE_NAMES

# ---------------------------------------------------------
# Page Configuration & Metadata
# ---------------------------------------------------------
st.set_page_config(
    page_title="MedTriage AI | Emergency Clinical Decision Support",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Executive Clinical CSS Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        color: white;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.3);
    }
    .hero-title {
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hero-subtitle {
        color: #94A3B8;
        font-size: 0.95rem;
        margin-bottom: 0;
    }
    .hero-badge {
        background: rgba(14, 165, 233, 0.15);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Section Cards */
    .glass-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 22px 24px;
        box-shadow: 0 4px 16px -2px rgba(0, 0, 0, 0.04);
        margin-bottom: 20px;
    }

    .card-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
        border-bottom: 1px solid #F1F5F9;
        padding-bottom: 10px;
    }

    /* Triage Result Box */
    .triage-hero-box {
        border-radius: 16px;
        padding: 24px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px -5px rgba(0,0,0,0.15);
        animation: fadeIn 0.4s ease-out;
    }

    .triage-level-badge {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 8px;
    }
    .triage-urgency-tag {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 24px;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        background: rgba(255, 255, 255, 0.25);
        backdrop-filter: blur(8px);
        margin-bottom: 12px;
    }

    .triage-time-callout {
        background: rgba(255, 255, 255, 0.18);
        border-radius: 12px;
        padding: 12px 16px;
        font-weight: 600;
        font-size: 0.95rem;
        margin-top: 14px;
    }

    /* Vital alert badges */
    .vital-alert {
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 0.88rem;
        font-weight: 600;
    }
    .vital-alert-critical {
        background: #FEE2E2;
        border-left: 4px solid #EF4444;
        color: #991B1B;
    }
    .vital-alert-warning {
        background: #FEF3C7;
        border-left: 4px solid #F59E0B;
        color: #92400E;
    }
    .vital-alert-info {
        background: #E0F2FE;
        border-left: 4px solid #0284C7;
        color: #075985;
    }

    /* Metric KPI Chips */
    .metric-chip {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 14px;
        text-align: center;
    }
    .metric-chip-value {
        font-size: 1.5rem;
        font-weight: 800;
        color: #0F172A;
    }
    .metric-chip-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Singleton Predictor Initialization
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Loading Clinical Triage Engine...")
def load_triage_engine():
    return get_predictor()

try:
    predictor = load_triage_engine()
except Exception as e:
    st.error(f"❌ Failed to load ML Model and Scaler: {str(e)}")
    st.stop()


# ---------------------------------------------------------
# Hero Banner
# ---------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">
        <div>
            <div class="hero-title">
                <span>🏥</span> MEDTRIAGE AI
                <span class="hero-badge">Clinical Production v1.0</span>
            </div>
            <p class="hero-subtitle">
                Emergency Severity Index (ESI) Intelligent Triage & Decision Support System • 94.3% Model Accuracy (18,000 Validated Records)
            </p>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 0.8rem; color: #94A3B8; text-transform: uppercase; font-weight: 700;">System Status</div>
            <div style="color: #10B981; font-weight: 700; font-size: 0.95rem; display: flex; align-items: center; gap: 6px;">
                <span style="display: inline-block; width: 8px; height: 8px; background: #10B981; border-radius: 50%;"></span>
                Online & Serving
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Main Tabs
# ---------------------------------------------------------
tab_single, tab_batch, tab_analytics, tab_api = st.tabs([
    "🩺 Single Patient Triage",
    "📋 Batch Emergency Intake",
    "📊 Model Analytics & Provenance",
    "⚡ REST API & Integration"
])


# =========================================================
# TAB 1: SINGLE PATIENT TRIAGE
# =========================================================
with tab_single:
    col_input, col_output = st.columns([1.1, 0.9], gap="large")

    with col_input:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">⚡ Rapid Clinical Scenario Presets</div>', unsafe_allow_html=True)
        
        c_p1, c_p2, c_p3, c_p4 = st.columns(4)
        preset = None
        if c_p1.button("🟢 Routine Adult", use_container_width=True, help="Stable vitals, mild pain"):
            preset = {
                "age": 28.0, "hr": 72.0, "bp": 118.0, "spo2": 99.0, "temp": 36.6,
                "pain": 1, "chronic": 0, "er_visits": 0, "mode": "walk_in"
            }
        if c_p2.button("🟡 Moderate Pain", use_container_width=True, help="Moderate pain, mild deviation"):
            preset = {
                "age": 51.0, "hr": 87.0, "bp": 128.0, "spo2": 98.0, "temp": 37.7,
                "pain": 5, "chronic": 2, "er_visits": 2, "mode": "walk_in"
            }
        if c_p3.button("🟠 Acute Distress", use_container_width=True, help="Elderly, fever, hypertension"):
            preset = {
                "age": 73.0, "hr": 98.0, "bp": 148.0, "spo2": 94.0, "temp": 38.4,
                "pain": 6, "chronic": 3, "er_visits": 3, "mode": "ambulance"
            }
        if c_p4.button("🔴 Critical Cardiac", use_container_width=True, help="Extreme tachycardia, fever, severe pain"):
            preset = {
                "age": 79.0, "hr": 148.0, "bp": 158.0, "spo2": 91.0, "temp": 39.4,
                "pain": 10, "chronic": 4, "er_visits": 2, "mode": "ambulance"
            }

        if preset:
            st.session_state["age"] = preset["age"]
            st.session_state["hr"] = preset["hr"]
            st.session_state["bp"] = preset["bp"]
            st.session_state["spo2"] = preset["spo2"]
            st.session_state["temp"] = preset["temp"]
            st.session_state["pain"] = preset["pain"]
            st.session_state["chronic"] = preset["chronic"]
            st.session_state["er_visits"] = preset["er_visits"]
            st.session_state["mode"] = preset["mode"]

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">👤 Patient Demographics & Arrival</div>', unsafe_allow_html=True)
        c_d1, c_d2 = st.columns(2)
        with c_d1:
            age = st.number_input(
                "Patient Age (years)",
                min_value=0.0, max_value=120.0,
                value=float(st.session_state.get("age", 45.0)),
                step=1.0,
                key="input_age"
            )
        with c_d2:
            mode_options = ["walk_in", "ambulance", "wheelchair"]
            current_mode = st.session_state.get("mode", "walk_in")
            arrival_mode = st.selectbox(
                "Hospital Arrival Mode",
                options=mode_options,
                index=mode_options.index(current_mode) if current_mode in mode_options else 0,
                format_func=lambda x: {
                    "walk_in": "🚶 Walk-in (Self / Family)",
                    "ambulance": "🚑 Ambulance (EMS)",
                    "wheelchair": "♿ Wheelchair / Assisted"
                }.get(x, x),
                key="input_mode"
            )

        st.markdown('<div class="card-header" style="margin-top: 18px;">🩺 Objective Vital Signs</div>', unsafe_allow_html=True)
        c_v1, c_v2 = st.columns(2)
        with c_v1:
            heart_rate = st.slider(
                "Heart Rate (bpm)",
                min_value=30.0, max_value=200.0,
                value=float(st.session_state.get("hr", 82.0)),
                step=1.0,
                help="Normal resting rate: 60-100 bpm",
                key="input_hr"
            )
            systolic_bp = st.slider(
                "Systolic Blood Pressure (mmHg)",
                min_value=50.0, max_value=240.0,
                value=float(st.session_state.get("bp", 124.0)),
                step=1.0,
                help="Normal: 90-120 mmHg. Hypertensive crisis: >= 180 mmHg",
                key="input_bp"
            )
        with c_v2:
            oxygen_saturation = st.slider(
                "Oxygen Saturation - SpO2 (%)",
                min_value=60.0, max_value=100.0,
                value=float(st.session_state.get("spo2", 98.0)),
                step=0.5,
                help="Normal: 95-100%. Critical Hypoxia: < 90%",
                key="input_spo2"
            )
            body_temperature = st.slider(
                "Body Temperature (°C)",
                min_value=34.0, max_value=42.0,
                value=float(st.session_state.get("temp", 37.0)),
                step=0.1,
                help="Normal: 36.5 - 37.5°C. Fever: >= 38.3°C",
                key="input_temp"
            )

        st.markdown('<div class="card-header" style="margin-top: 18px;">📋 Clinical Severity & Medical History</div>', unsafe_allow_html=True)
        c_h1, c_h2, c_h3 = st.columns(3)
        with c_h1:
            pain_level = st.slider(
                "Pain Score (0 - 10)",
                min_value=0, max_value=10,
                value=int(st.session_state.get("pain", 2)),
                help="0: No pain, 10: Worst imaginable",
                key="input_pain"
            )
        with c_h2:
            chronic_disease_count = st.number_input(
                "Chronic Conditions Count",
                min_value=0, max_value=15,
                value=int(st.session_state.get("chronic", 1)),
                step=1,
                help="e.g., Hypertension, Diabetes, COPD, CKD",
                key="input_chronic"
            )
        with c_h3:
            previous_er_visits = st.number_input(
                "Past ER Visits (12 mos)",
                min_value=0, max_value=25,
                value=int(st.session_state.get("er_visits", 0)),
                step=1,
                key="input_er"
            )

        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------------------------------------------
    # RIGHT COLUMN: LIVE TRIAGE INFERENCE & ALERTS
    # ---------------------------------------------------------
    with col_output:
        # Prepare patient payload
        patient_data = {
            "age": age,
            "heart_rate": heart_rate,
            "systolic_blood_pressure": systolic_bp,
            "oxygen_saturation": oxygen_saturation,
            "body_temperature": body_temperature,
            "pain_level": pain_level,
            "chronic_disease_count": chronic_disease_count,
            "previous_er_visits": previous_er_visits,
            "arrival_mode": arrival_mode
        }

        result = predictor.predict_single(patient_data)
        triage_level = result["triage_level"]
        badge_color = result["badge_color"]

        # Big Visual Hero Box
        st.markdown(f"""
        <div class="triage-hero-box" style="background: {badge_color};">
            <div class="triage-urgency-tag">
                PRIORITY: {result["priority"]}
            </div>
            <div class="triage-level-badge">
                {result["triage_name"]}
            </div>
            <div style="font-size: 1.15rem; font-weight: 600; opacity: 0.95;">
                Model Confidence: {result["confidence_score"]:.1f}%
            </div>
            <div class="triage-time-callout">
                ⏱️ Target Clinical Response: <strong>{result["target_time"]}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Action Box
        st.markdown(f"""
        <div class="glass-card" style="border-left: 6px solid {badge_color};">
            <div style="font-weight: 700; color: #0F172A; margin-bottom: 6px; font-size: 1.05rem;">
                🚨 Recommended Triage Protocol
            </div>
            <div style="color: #334155; font-size: 0.92rem; line-height: 1.5;">
                {result["clinical_action"]}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Vital Alerts Section
        if result["vital_flags"]:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header">⚠️ Physiological Risk Alerts</div>', unsafe_allow_html=True)
            for flag in result["vital_flags"]:
                sev_cls = "vital-alert-critical" if flag["severity"] == "critical" else (
                    "vital-alert-warning" if flag["severity"] == "warning" else "vital-alert-info"
                )
                icon = "🚨" if flag["severity"] == "critical" else ("⚠️" if flag["severity"] == "warning" else "ℹ️")
                st.markdown(f"""
                <div class="vital-alert {sev_cls}">
                    <span>{icon}</span>
                    <div>
                        <strong>{flag["parameter"]} ({flag["value"]}):</strong> {flag["message"]}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Probabilities Distribution
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">📊 Multi-Class Probability Distribution</div>', unsafe_allow_html=True)
        prob_df = pd.DataFrame([
            {"Triage Level": "Level 0 (Non-Urgent)", "Probability (%)": result["class_probabilities"]["Level 0"] * 100},
            {"Triage Level": "Level 1 (Semi-Urgent)", "Probability (%)": result["class_probabilities"]["Level 1"] * 100},
            {"Triage Level": "Level 2 (Urgent)", "Probability (%)": result["class_probabilities"]["Level 2"] * 100},
            {"Triage Level": "Level 3 (Critical)", "Probability (%)": result["class_probabilities"]["Level 3"] * 100},
        ])
        st.bar_chart(
            prob_df.set_index("Triage Level"),
            color="#0284C7",
            use_container_width=True
        )

        # Intake Slip Download
        intake_slip = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "patient_vitals": patient_data,
            "triage_assessment": result
        }
        st.download_button(
            label="📥 Download Clinical Intake Slip (JSON)",
            data=json.dumps(intake_slip, indent=2),
            file_name=f"triage_slip_{int(time.time())}.json",
            mime="application/json",
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# TAB 2: BATCH EMERGENCY INTAKE
# =========================================================
with tab_batch:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">📂 High-Volume Emergency Batch Triage</div>', unsafe_allow_html=True)
    st.write(
        "Upload a roster of incoming patients (.csv) to classify their triage acuity concurrently. "
        "The system will automatically validate vitals, run multi-class inference, and generate a downloadable prioritized triage queue."
    )

    c_b1, c_b2 = st.columns([1.2, 0.8])
    with c_b1:
        uploaded_file = st.file_uploader(
            "Upload Patient CSV File",
            type=["csv"],
            help=f"Must contain: {', '.join(FEATURE_NAMES)}"
        )
    with c_b2:
        st.write("**Need a template?** Generate and download a sample batch file:")
        sample_df = pd.DataFrame([
            {"age": 24.0, "heart_rate": 74.0, "systolic_blood_pressure": 115.0, "oxygen_saturation": 99.0, "body_temperature": 36.7, "pain_level": 1, "chronic_disease_count": 0, "previous_er_visits": 0, "arrival_mode": "walk_in"},
            {"age": 58.0, "heart_rate": 88.0, "systolic_blood_pressure": 132.0, "oxygen_saturation": 96.0, "body_temperature": 37.8, "pain_level": 4, "chronic_disease_count": 2, "previous_er_visits": 1, "arrival_mode": "walk_in"},
            {"age": 71.0, "heart_rate": 105.0, "systolic_blood_pressure": 160.0, "oxygen_saturation": 92.0, "body_temperature": 38.6, "pain_level": 7, "chronic_disease_count": 3, "previous_er_visits": 4, "arrival_mode": "wheelchair"},
            {"age": 82.0, "heart_rate": 142.0, "systolic_blood_pressure": 175.0, "oxygen_saturation": 88.0, "body_temperature": 39.5, "pain_level": 10, "chronic_disease_count": 5, "previous_er_visits": 3, "arrival_mode": "ambulance"}
        ])
        csv_buffer = io.StringIO()
        sample_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📄 Download Sample Batch CSV Template",
            data=csv_buffer.getvalue(),
            file_name="sample_triage_batch.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded {len(input_df)} patient records.")

            with st.spinner("Processing batch through Triage Classification Engine..."):
                processed_df = predictor.predict_batch(input_df)

            # Triage Metrics Summary
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header">📈 Emergency Triage Distribution Summary</div>', unsafe_allow_html=True)
            k1, k2, k3, k4, k5 = st.columns(5)
            
            total_pts = len(processed_df)
            c3 = (processed_df["predicted_triage_level"] == 3).sum()
            c2 = (processed_df["predicted_triage_level"] == 2).sum()
            c1 = (processed_df["predicted_triage_level"] == 1).sum()
            c0 = (processed_df["predicted_triage_level"] == 0).sum()

            k1.markdown(f'<div class="metric-chip"><div class="metric-chip-value">{total_pts}</div><div class="metric-chip-label">Total Patients</div></div>', unsafe_allow_html=True)
            k2.markdown(f'<div class="metric-chip" style="border-color: #EF4444;"><div class="metric-chip-value" style="color: #EF4444;">{c3}</div><div class="metric-chip-label">🔴 Level 3 (Critical)</div></div>', unsafe_allow_html=True)
            k3.markdown(f'<div class="metric-chip" style="border-color: #F97316;"><div class="metric-chip-value" style="color: #F97316;">{c2}</div><div class="metric-chip-label">🟠 Level 2 (Urgent)</div></div>', unsafe_allow_html=True)
            k4.markdown(f'<div class="metric-chip" style="border-color: #F59E0B;"><div class="metric-chip-value" style="color: #F59E0B;">{c1}</div><div class="metric-chip-label">🟡 Level 1 (Semi-Urgent)</div></div>', unsafe_allow_html=True)
            k5.markdown(f'<div class="metric-chip" style="border-color: #10B981;"><div class="metric-chip-value" style="color: #10B981;">{c0}</div><div class="metric-chip-label">🟢 Level 0 (Non-Urgent)</div></div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

            # Interactive Filtered Table
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header">📋 Prioritized Patient Queue</div>', unsafe_allow_html=True)
            
            filter_level = st.multiselect(
                "Filter by Triage Level",
                options=[3, 2, 1, 0],
                default=[3, 2, 1, 0],
                format_func=lambda x: f"Level {x} - {TRIAGE_LEVELS[x]['short_name']}"
            )
            filtered_df = processed_df[processed_df["predicted_triage_level"].isin(filter_level)]

            st.dataframe(
                filtered_df,
                use_container_width=True,
                height=350
            )

            # Export Button
            out_csv = io.StringIO()
            processed_df.to_csv(out_csv, index=False)
            st.download_button(
                label="📥 Export Full Triaged Queue (CSV)",
                data=out_csv.getvalue(),
                file_name=f"triaged_patients_{int(time.time())}.csv",
                mime="text/csv",
                use_container_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Batch processing failed: {str(e)}")


# =========================================================
# TAB 3: MODEL PERFORMANCE & PROVENANCE
# =========================================================
with tab_analytics:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">🔬 Training Specifications & Evaluation Metrics</div>', unsafe_allow_html=True)
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Architecture", "Multinomial Logistic Regression")
    m2.metric("Preprocessing", "StandardScaler + Heart Rate Clip")
    m3.metric("Validation Accuracy", "94.27%")
    m4.metric("Benchmark Cases", "18,000 Clinical Rows")

    st.markdown("---")
    st.markdown("#### Model Coefficients & Clinical Feature Weights")
    st.write(
        "Coefficients fitted during training across all 4 triage levels. "
        "Positive coefficients increase the log-odds of assignment to that acuity class."
    )

    if hasattr(predictor.model, "coef_"):
        coef_df = pd.DataFrame(
            predictor.model.coef_,
            columns=FEATURE_NAMES,
            index=["Class 0 (Non-Urgent)", "Class 1 (Semi-Urgent)", "Class 2 (Urgent)", "Class 3 (Critical)"]
        )
        st.dataframe(coef_df.style.background_gradient(cmap="coolwarm", axis=None), use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# TAB 4: REST API & INTEGRATION
# =========================================================
with tab_api:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">⚡ Hospital System & EHR REST API</div>', unsafe_allow_html=True)
    st.write(
        "The project includes a standalone **FastAPI service** (`api.py`) exposing high-speed JSON endpoints "
        "for integration with Electronic Health Records (Epic, Cerner), triage kiosks, and mobile applications."
    )

    st.markdown("### Interactive Documentation Endpoints")
    st.info("Interactive Swagger UI is available at `http://localhost:8000/docs` or `http://0.0.0.0:8000/redoc` when running `python api.py`.")

    st.markdown("### Example cURL Request")
    st.code("""
curl -X POST "http://localhost:8000/predict" \\
  -H "Content-Type: application/json" \\
  -d '{
    "patient_id": "PT-90412",
    "age": 68.0,
    "heart_rate": 138.0,
    "systolic_blood_pressure": 165.0,
    "oxygen_saturation": 89.0,
    "body_temperature": 39.2,
    "pain_level": 9,
    "chronic_disease_count": 3,
    "previous_er_visits": 2,
    "arrival_mode": "ambulance"
  }'
    """, language="bash")

    st.markdown("### Example Python Request")
    st.code("""
import requests

payload = {
    "age": 45.0,
    "heart_rate": 82.0,
    "systolic_blood_pressure": 120.0,
    "oxygen_saturation": 98.0,
    "body_temperature": 37.0,
    "pain_level": 2,
    "chronic_disease_count": 1,
    "previous_er_visits": 0,
    "arrival_mode": "walk_in"
}

response = requests.post("http://localhost:8000/predict", json=payload)
print(response.json())
    """, language="python")

    st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("""
<div style="text-align: center; color: #94A3B8; font-size: 0.82rem; margin-top: 30px;">
    MEDTRIAGE AI Decision Support System • Powered by Machine Learning & Scikit-Learn • For Clinical Reference
</div>
""", unsafe_allow_html=True)
