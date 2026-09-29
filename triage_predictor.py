"""
Medical Triage Clinical Prediction & Decision Support Engine
Provides high-performance inference, feature validation, clinical vital sign risk flags,
and triage classification using the trained scikit-learn model and scaler.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional


# Official Clinical Triage Definitions based on Emergency Severity Index (ESI)
TRIAGE_LEVELS = {
    0: {
        "name": "Level 0 - Non-Urgent (Routine Care)",
        "short_name": "Non-Urgent",
        "badge_color": "#10B981",  # Green
        "light_color": "#E6F4EA",
        "text_color": "#065F46",
        "priority": "Low",
        "target_time": "120 - 240 mins",
        "clinical_action": "Safe for ambulatory waiting area or primary outpatient clinic assessment. Routine vital monitoring.",
        "icon": "🟢"
    },
    1: {
        "name": "Level 1 - Semi-Urgent (Low Acuity)",
        "short_name": "Semi-Urgent",
        "badge_color": "#F59E0B",  # Amber/Yellow
        "light_color": "#FEF3C7",
        "text_color": "#92400E",
        "priority": "Moderate",
        "target_time": "Within 60 mins",
        "clinical_action": "Requires focused emergency evaluation, routine diagnostic workup (labs/imaging), and symptom relief.",
        "icon": "🟡"
    },
    2: {
        "name": "Level 2 - Urgent (High Priority)",
        "short_name": "Urgent",
        "badge_color": "#F97316",  # Deep Orange
        "light_color": "#FFEDD5",
        "text_color": "#9A3412",
        "priority": "High",
        "target_time": "Within 15 - 30 mins",
        "clinical_action": "Acute distress, pronounced physiological abnormal vitals, or significant pain. Assign dedicated bed immediately.",
        "icon": "🟠"
    },
    3: {
        "name": "Level 3 - Immediate (Resuscitation / Critical)",
        "short_name": "Critical / Resuscitation",
        "badge_color": "#EF4444",  # Crimson Red
        "light_color": "#FEE2E2",
        "text_color": "#991B1B",
        "priority": "CRITICAL",
        "target_time": "IMMEDIATE (0 mins)",
        "clinical_action": "EMERGENCY: Imminent life threat or hemodynamic collapse. Transfer directly to Resuscitation / Trauma Suite.",
        "icon": "🔴"
    }
}

ARRIVAL_MODE_MAP = {
    "ambulance": 0,
    "walk_in": 1,
    "walk-in": 1,
    "walk in": 1,
    "walkin": 1,
    "wheelchair": 2
}

FEATURE_NAMES = [
    "age",
    "heart_rate",
    "systolic_blood_pressure",
    "oxygen_saturation",
    "body_temperature",
    "pain_level",
    "chronic_disease_count",
    "previous_er_visits",
    "arrival_mode"
]


class MedicalTriagePredictor:
    """Production Triage Predictor with embedded preprocessor and vital safety analysis."""

    def __init__(self, model_path: Optional[str] = None, scaler_path: Optional[str] = None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        if model_path is None:
            model_path = os.path.join(base_dir, "Medical-Triage_model.pkl")
        if scaler_path is None:
            scaler_path = os.path.join(base_dir, "scaler.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found at: {scaler_path}")

        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    @staticmethod
    def encode_arrival_mode(arrival_mode: Any) -> int:
        """Sanitizes and converts arrival mode strings or numbers to model integer encoding."""
        if isinstance(arrival_mode, (int, float)) and not pd.isna(arrival_mode):
            val = int(arrival_mode)
            if val in [0, 1, 2]:
                return val
        
        if isinstance(arrival_mode, str):
            clean_str = arrival_mode.strip().lower().replace(" ", "_").replace("-", "_")
            if clean_str in ARRIVAL_MODE_MAP:
                return ARRIVAL_MODE_MAP[clean_str]
        
        # Fallback to walk_in if unrecognized
        return 1

    @staticmethod
    def evaluate_vital_flags(
        heart_rate: float,
        systolic_bp: float,
        oxygen_saturation: float,
        body_temperature: float,
        pain_level: int
    ) -> List[Dict[str, str]]:
        """Evaluates clinical safety thresholds and flags alarming vital signs."""
        flags = []
        
        # Oxygen Saturation
        if oxygen_saturation < 90.0:
            flags.append({
                "severity": "critical",
                "parameter": "Oxygen Saturation",
                "value": f"{oxygen_saturation:.1f}%",
                "message": "Critical Hypoxemia (SpO2 < 90%). Administer supplemental O2 immediately."
            })
        elif oxygen_saturation < 94.0:
            flags.append({
                "severity": "warning",
                "parameter": "Oxygen Saturation",
                "value": f"{oxygen_saturation:.1f}%",
                "message": "Mild to Moderate Hypoxia (SpO2 90-93%). Continuous pulse oximetry indicated."
            })

        # Heart Rate
        if heart_rate > 130.0:
            flags.append({
                "severity": "critical",
                "parameter": "Heart Rate",
                "value": f"{heart_rate:.0f} bpm",
                "message": "Severe Tachycardia (HR > 130 bpm). Risk of hemodynamic instability."
            })
        elif heart_rate > 100.0:
            flags.append({
                "severity": "warning",
                "parameter": "Heart Rate",
                "value": f"{heart_rate:.0f} bpm",
                "message": "Tachycardia (HR > 100 bpm). Assess for pain, infection, or dehydration."
            })
        elif heart_rate < 50.0:
            flags.append({
                "severity": "warning",
                "parameter": "Heart Rate",
                "value": f"{heart_rate:.0f} bpm",
                "message": "Bradycardia (HR < 50 bpm). Monitor cardiac rhythm."
            })

        # Blood Pressure
        if systolic_bp >= 180.0:
            flags.append({
                "severity": "critical",
                "parameter": "Systolic BP",
                "value": f"{systolic_bp:.0f} mmHg",
                "message": "Hypertensive Crisis (SBP >= 180 mmHg). Assess for end-organ damage."
            })
        elif systolic_bp < 90.0:
            flags.append({
                "severity": "critical",
                "parameter": "Systolic BP",
                "value": f"{systolic_bp:.0f} mmHg",
                "message": "Severe Hypotension (SBP < 90 mmHg). Risk of circulatory shock."
            })
        elif systolic_bp >= 140.0:
            flags.append({
                "severity": "info",
                "parameter": "Systolic BP",
                "value": f"{systolic_bp:.0f} mmHg",
                "message": "Stage 2 Hypertension range (SBP >= 140 mmHg)."
            })

        # Body Temperature
        if body_temperature >= 39.5:
            flags.append({
                "severity": "critical",
                "parameter": "Body Temperature",
                "value": f"{body_temperature:.1f} °C",
                "message": "Hyperpyrexia (Temp >= 39.5 °C). High risk of severe sepsis or heat illness."
            })
        elif body_temperature >= 38.3:
            flags.append({
                "severity": "warning",
                "parameter": "Body Temperature",
                "value": f"{body_temperature:.1f} °C",
                "message": "Febrile state (Temp >= 38.3 °C). Sepsis workup consideration."
            })
        elif body_temperature < 35.5:
            flags.append({
                "severity": "warning",
                "parameter": "Body Temperature",
                "value": f"{body_temperature:.1f} °C",
                "message": "Hypothermia (Temp < 35.5 °C). Active warming recommended."
            })

        # Pain Level
        if pain_level >= 8:
            flags.append({
                "severity": "warning",
                "parameter": "Pain Score",
                "value": f"{pain_level}/10",
                "message": "Severe Acute Pain. Prompt analgesic intervention indicated."
            })

        return flags

    def prepare_features(self, patient_data: Dict[str, Any]) -> Tuple[np.ndarray, List[Dict[str, str]]]:
        """Extracts, validates, cleans, and structures features for inference."""
        age = float(patient_data.get("age", 45.0))
        raw_hr = float(patient_data.get("heart_rate", 80.0))
        # Notebook explicitly clipped heart_rate to [40, 200]
        heart_rate = float(np.clip(raw_hr, 40.0, 200.0))
        systolic_bp = float(patient_data.get("systolic_blood_pressure", 120.0))
        oxygen_sat = float(patient_data.get("oxygen_saturation", 98.0))
        body_temp = float(patient_data.get("body_temperature", 37.0))
        pain_level = int(patient_data.get("pain_level", 0))
        chronic_count = int(patient_data.get("chronic_disease_count", 0))
        er_visits = int(patient_data.get("previous_er_visits", 0))
        arrival_encoded = self.encode_arrival_mode(patient_data.get("arrival_mode", "walk_in"))

        features = np.array([[
            age,
            heart_rate,
            systolic_bp,
            oxygen_sat,
            body_temp,
            pain_level,
            chronic_count,
            er_visits,
            arrival_encoded
        ]], dtype=np.float64)

        vital_flags = self.evaluate_vital_flags(
            heart_rate=raw_hr,
            systolic_bp=systolic_bp,
            oxygen_saturation=oxygen_sat,
            body_temperature=body_temp,
            pain_level=pain_level
        )

        return features, vital_flags

    def predict_single(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Runs single-patient inference and returns full clinical triage report."""
        features, vital_flags = self.prepare_features(patient_data)
        
        features_df = pd.DataFrame(features, columns=FEATURE_NAMES)
        # Scale features
        scaled_features = self.scaler.transform(features_df)
        
        # Predict class & probabilities
        predicted_class = int(self.model.predict(scaled_features)[0])
        probabilities = self.model.predict_proba(scaled_features)[0]
        
        triage_info = TRIAGE_LEVELS.get(predicted_class, TRIAGE_LEVELS[0])
        confidence = float(probabilities[predicted_class])

        class_probabilities = {
            f"Level {cls_idx}": float(prob)
            for cls_idx, prob in enumerate(probabilities)
        }

        return {
            "triage_level": predicted_class,
            "triage_name": triage_info["name"],
            "short_name": triage_info["short_name"],
            "priority": triage_info["priority"],
            "badge_color": triage_info["badge_color"],
            "light_color": triage_info["light_color"],
            "text_color": triage_info["text_color"],
            "target_time": triage_info["target_time"],
            "clinical_action": triage_info["clinical_action"],
            "confidence_score": round(confidence * 100, 2),
            "class_probabilities": class_probabilities,
            "vital_flags": vital_flags,
            "critical_flags_count": sum(1 for f in vital_flags if f["severity"] == "critical"),
            "warning_flags_count": sum(1 for f in vital_flags if f["severity"] == "warning"),
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Batch inference on a pandas DataFrame with full classification and score enrichment."""
        df_work = df.copy()

        # Check required columns
        for col in FEATURE_NAMES:
            if col not in df_work.columns:
                raise ValueError(f"Required column '{col}' is missing from the dataset.")

        # Preprocessing
        clean_hr = df_work["heart_rate"].astype(float).clip(lower=40.0, upper=200.0)
        clean_arrival = df_work["arrival_mode"].apply(self.encode_arrival_mode)

        features_matrix = np.column_stack([
            df_work["age"].astype(float),
            clean_hr,
            df_work["systolic_blood_pressure"].astype(float),
            df_work["oxygen_saturation"].astype(float),
            df_work["body_temperature"].astype(float),
            df_work["pain_level"].astype(int),
            df_work["chronic_disease_count"].astype(int),
            df_work["previous_er_visits"].astype(int),
            clean_arrival
        ])

        matrix_df = pd.DataFrame(features_matrix, columns=FEATURE_NAMES)
        scaled_matrix = self.scaler.transform(matrix_df)
        predictions = self.model.predict(scaled_matrix)
        probabilities = self.model.predict_proba(scaled_matrix)

        df_work["predicted_triage_level"] = predictions
        df_work["triage_name"] = [TRIAGE_LEVELS[p]["short_name"] for p in predictions]
        df_work["priority"] = [TRIAGE_LEVELS[p]["priority"] for p in predictions]
        df_work["confidence_percent"] = [round(probabilities[i, p] * 100, 1) for i, p in enumerate(predictions)]
        df_work["prob_level_0"] = [round(probabilities[i, 0] * 100, 1) for i in range(len(df_work))]
        df_work["prob_level_1"] = [round(probabilities[i, 1] * 100, 1) for i in range(len(df_work))]
        df_work["prob_level_2"] = [round(probabilities[i, 2] * 100, 1) for i in range(len(df_work))]
        df_work["prob_level_3"] = [round(probabilities[i, 3] * 100, 1) for i in range(len(df_work))]

        return df_work


# Singleton instance helper
_predictor_instance = None

def get_predictor() -> MedicalTriagePredictor:
    """Returns singleton instance of MedicalTriagePredictor."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = MedicalTriagePredictor()
    return _predictor_instance
