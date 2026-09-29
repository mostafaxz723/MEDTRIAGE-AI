"""
Medical Triage REST API
Production FastAPI Service for EHR, Mobile, and Hospital System Integration.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os

from triage_predictor import get_predictor, TRIAGE_LEVELS, FEATURE_NAMES

app = FastAPI(
    title="Medical Triage Prediction & Clinical Decision Support API",
    description="High-performance machine learning inference service for emergency room triage classification (ESI Levels 0 - 3).",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for cross-origin web apps and EHR integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PatientInput(BaseModel):
    patient_id: Optional[str] = Field(default=None, description="Optional Hospital Patient ID or MRN")
    age: float = Field(..., ge=0.0, le=125.0, description="Patient age in years", examples=[45.0])
    heart_rate: float = Field(..., ge=20.0, le=300.0, description="Heart rate in beats per minute (bpm)", examples=[82.0])
    systolic_blood_pressure: float = Field(..., ge=40.0, le=300.0, description="Systolic blood pressure (mmHg)", examples=[122.0])
    oxygen_saturation: float = Field(..., ge=50.0, le=100.0, description="Blood oxygen saturation (SpO2 %)", examples=[98.0])
    body_temperature: float = Field(..., ge=30.0, le=45.0, description="Core body temperature in Celsius (deg C)", examples=[37.1])
    pain_level: int = Field(..., ge=0, le=10, description="Numeric pain rating scale (0 to 10)", examples=[2])
    chronic_disease_count: int = Field(..., ge=0, le=20, description="Number of pre-existing chronic conditions", examples=[1])
    previous_er_visits: int = Field(..., ge=0, le=50, description="Count of emergency visits in the past year", examples=[1])
    arrival_mode: str = Field(..., description="Method of hospital arrival: 'ambulance', 'walk_in', or 'wheelchair'", examples=["walk_in"])


class VitalFlag(BaseModel):
    severity: str
    parameter: str
    value: str
    message: str


class TriagePredictionResponse(BaseModel):
    patient_id: Optional[str]
    triage_level: int
    triage_name: str
    short_name: str
    priority: str
    target_time: str
    clinical_action: str
    badge_color: str
    confidence_score: float
    class_probabilities: Dict[str, float]
    vital_flags: List[VitalFlag]
    critical_flags_count: int
    warning_flags_count: int


class BatchPatientRequest(BaseModel):
    patients: List[PatientInput]


class BatchPredictionResponse(BaseModel):
    total_processed: int
    critical_count: int
    urgent_count: int
    semi_urgent_count: int
    non_urgent_count: int
    results: List[TriagePredictionResponse]


@app.get("/", tags=["System"])
def root():
    """Root endpoint welcoming clients and directing to docs."""
    return {
        "service": "Medical Triage ML Prediction API",
        "status": "operational",
        "documentation": "/docs",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_batch": "/predict-batch"
        }
    }


@app.get("/health", tags=["System"])
def health_check():
    """Service health and model readiness check."""
    try:
        predictor = get_predictor()
        return {
            "status": "healthy",
            "model_loaded": predictor.model is not None,
            "scaler_loaded": predictor.scaler is not None,
            "features_expected": FEATURE_NAMES
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}"
        )


@app.post("/predict", response_model=TriagePredictionResponse, tags=["Inference"])
def predict_patient(patient: PatientInput):
    """Predicts emergency triage level for a single incoming patient."""
    try:
        predictor = get_predictor()
        data_dict = patient.model_dump()
        result = predictor.predict_single(data_dict)
        result["patient_id"] = patient.patient_id
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )


@app.post("/predict-batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch(batch: BatchPatientRequest):
    """Predicts emergency triage level for multiple incoming patients."""
    try:
        predictor = get_predictor()
        results = []
        counts = {0: 0, 1: 0, 2: 0, 3: 0}

        for pt in batch.patients:
            data_dict = pt.model_dump()
            res = predictor.predict_single(data_dict)
            res["patient_id"] = pt.patient_id
            level = res["triage_level"]
            counts[level] = counts.get(level, 0) + 1
            results.append(res)

        return {
            "total_processed": len(results),
            "critical_count": counts.get(3, 0),
            "urgent_count": counts.get(2, 0),
            "semi_urgent_count": counts.get(1, 0),
            "non_urgent_count": counts.get(0, 0),
            "results": results
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)
