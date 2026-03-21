"""
Disease Risk Prediction API  —  Diabetes Edition
Euodia Ebalu: API Developer
FastAPI + Pydantic data validation
Model: RandomForestClassifier trained on the Pima Indians Diabetes Dataset
"""

import time
import logging
import traceback
import joblib
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

# ─────────────────────────────────────────────
# Logging Setup
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# FastAPI App Initialisation
# ─────────────────────────────────────────────
app = FastAPI(
    title="Diabetes Risk Prediction API",
    description=(
        "A RESTful API that accepts patient clinical features from the Pima Indians "
        "Diabetes Dataset and returns a diabetes risk probability score using a trained "
        "Random Forest classifier. Designed for integration by healthcare developers "
        "and clinicians."
    ),
    version="1.0.0",
    contact={
        "name": "API Developer — Person 3",
        "email": "developer@diabetesrisk.ai",
    },
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Load Trained Model
# ─────────────────────────────────────────────
MODEL_VERSION = "v1.0.0"
model = joblib.load("diabetes_model.pkl")
_app_start = time.time()

# Exact feature order the RandomForest was trained on
FEATURE_NAMES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

logger.info("Model loaded successfully — %s", MODEL_VERSION)


# ─────────────────────────────────────────────
# Pydantic Schemas — Input Validation
# ─────────────────────────────────────────────

class PatientFeatures(BaseModel):
    """
    Input schema for a single diabetes risk prediction request.
    Field ranges are derived from the Pima Indians Diabetes Dataset.
    All features are validated by Pydantic before the model is called.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "Pregnancies": 6,
                "Glucose": 148,
                "BloodPressure": 72,
                "SkinThickness": 35,
                "Insulin": 0,
                "BMI": 33.6,
                "DiabetesPedigreeFunction": 0.627,
                "Age": 50,
            }
        }
    )

    Pregnancies: int = Field(
        ...,
        ge=0,
        le=20,
        description="Number of times the patient has been pregnant (0–20).",
    )
    Glucose: float = Field(
        ...,
        ge=0.0,
        le=250.0,
        description="Plasma glucose concentration at 2 hours in an oral glucose tolerance test (mg/dL). Range: 0–250.",
    )
    BloodPressure: float = Field(
        ...,
        ge=0.0,
        le=150.0,
        description="Diastolic blood pressure in mmHg (0–150). 0 indicates a missing/unknown value.",
    )
    SkinThickness: float = Field(
        ...,
        ge=0.0,
        le=120.0,
        description="Triceps skinfold thickness in mm (0–120). 0 indicates a missing/unknown value.",
    )
    Insulin: float = Field(
        ...,
        ge=0.0,
        le=1000.0,
        description="2-hour serum insulin in µU/mL (0–1000). 0 indicates a missing/unknown value.",
    )
    BMI: float = Field(
        ...,
        ge=0.0,
        le=80.0,
        description="Body Mass Index in kg/m² (0–80). 0 indicates a missing/unknown value.",
    )
    DiabetesPedigreeFunction: float = Field(
        ...,
        ge=0.0,
        le=3.0,
        description="Diabetes pedigree function — scores likelihood of diabetes based on family history (0.0–3.0).",
    )
    Age: int = Field(
        ...,
        ge=21,
        le=90,
        description="Patient age in years (21–90). Minimum age in the dataset is 21.",
    )

    # ── Cross-field validator ───────────────────────────────────────────
    @model_validator(mode="after")
    def glucose_must_not_be_zero(self) -> "PatientFeatures":
        """Glucose of 0 is physiologically impossible — reject it explicitly."""
        if self.Glucose == 0:
            raise ValueError(
                "Glucose cannot be 0. A plasma glucose of zero is physiologically "
                "implausible. Please provide a valid glucose reading."
            )
        return self


class BatchPredictionRequest(BaseModel):
    """Input schema for batch prediction — up to 100 patients per call."""

    patients: list[PatientFeatures] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of patient records (1–100).",
    )


# ─────────────────────────────────────────────
# Pydantic Schemas — Response Models
# ─────────────────────────────────────────────

class PredictionResponse(BaseModel):
    """Single-patient prediction response."""

    risk_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Predicted probability of diabetes (0.0 = low risk, 1.0 = high risk).",
    )
    risk_category: str = Field(
        ...,
        description="Human-readable risk band: 'Low', 'Moderate', or 'High'.",
    )
    prediction: str = Field(
        ...,
        description="Binary outcome label: 'Diabetic' or 'Non-Diabetic'.",
    )
    model_version: str
    prediction_id: str = Field(..., description="UUID for audit logging.")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp.")
    processing_time_ms: float


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""

    results: list[PredictionResponse]
    total_patients: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_version: str
    uptime_seconds: float
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Model metadata response."""

    model_version: str
    algorithm: str
    training_dataset: str
    features: list[str]
    n_estimators: int
    metrics: dict
    last_trained: str


class ErrorResponse(BaseModel):
    """Standard error response body."""

    error: str
    detail: str
    status_code: int
    timestamp: str


# ─────────────────────────────────────────────
# Inference Helpers
# ─────────────────────────────────────────────

def _encode_features(p: PatientFeatures) -> pd.DataFrame:
    """
    Convert a validated PatientFeatures object into a single-row DataFrame
    with columns in the exact order the RandomForest was trained on.
    Using a DataFrame (not a raw numpy array) preserves feature names and
    avoids sklearn feature-name mismatch warnings.
    No scaling is applied — RandomForest does not require feature scaling.
    """
    return pd.DataFrame([{
        "Pregnancies":              p.Pregnancies,
        "Glucose":                  p.Glucose,
        "BloodPressure":            p.BloodPressure,
        "SkinThickness":            p.SkinThickness,
        "Insulin":                  p.Insulin,
        "BMI":                      p.BMI,
        "DiabetesPedigreeFunction": p.DiabetesPedigreeFunction,
        "Age":                      p.Age,
    }])


def _run_inference(features: pd.DataFrame) -> float:
    """
    Run the trained RandomForestClassifier and return the probability
    that the patient has diabetes (Outcome == 1).
    predict_proba returns [[P(0), P(1)]] — we take [0][1].
    No scaling step — RandomForest is invariant to feature scale.
    """
    proba = model.predict_proba(features)[0][1]
    return float(proba)


def _risk_category(score: float) -> str:
    """Threshold the probability score into a human-readable risk band."""
    if score < 0.33:
        return "Low"
    elif score < 0.66:
        return "Moderate"
    return "High"


def _prediction_label(score: float) -> str:
    """Binary outcome label using the standard 0.5 decision threshold."""
    return "Diabetic" if score >= 0.5 else "Non-Diabetic"


# ─────────────────────────────────────────────
# Global Exception Handler
# ─────────────────────────────────────────────

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s\n%s", exc, traceback.format_exc())
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="InternalServerError",
            detail="An unexpected error occurred. Please try again later.",
            status_code=500,
            timestamp=datetime.utcnow().isoformat() + "Z",
        ).model_dump(),
    )


# ─────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    tags=["Monitoring"],
    responses={
        200: {"description": "API is healthy and model is loaded."},
        503: {"description": "Service unavailable — model not loaded."},
    },
)
async def health_check():
    """
    Returns the current health status of the API and the version of the
    loaded Random Forest model. Used by Cloud Run health probes and
    external uptime monitors.from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("diabetes_model.pkl")

class PatientData(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

@app.get("/")
def home():
    return {"message": "Diabetes Risk Prediction API is running"}

@app.post("/predict")
def predict(data: PatientData):
    input_df = pd.DataFrame([data.model_dump()])
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if probability < 0.33:
        risk = "Low"
    elif probability < 0.66:
        risk = "Moderate"
    else:
        risk = "High"

    return {
        "prediction": int(prediction),
        "result": "Diabetic" if prediction == 1 else "Non-Diabetic",
        "risk_category": risk,
        "probability": round(float(probability), 4)
    }
    """
    return HealthResponse(
        status="healthy",
        model_version=MODEL_VERSION,
        uptime_seconds=round(time.time() - _app_start, 2),
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Single Patient Diabetes Risk Prediction",
    tags=["Prediction"],
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Prediction returned successfully."},
        422: {"description": "Validation error — input data failed schema checks."},
        500: {"description": "Internal server error during model inference."},
    },
)
async def predict(patient: PatientFeatures):
    """
    Accepts a single patient's 8 clinical features and returns a diabetes
    risk probability score produced by the Random Forest model.

    - **risk_score**: Probability of diabetes in [0.0, 1.0].
    - **risk_category**: Low (< 0.33) / Moderate (0.33–0.65) / High (> 0.65).
    - **prediction**: Binary label — 'Diabetic' or 'Non-Diabetic' (threshold 0.5).
    - **processing_time_ms**: Server-side latency for performance monitoring.
    """
    t0 = time.perf_counter()
    logger.info(
        "POST /predict — Age=%d, Glucose=%.1f, BMI=%.1f",
        patient.Age, patient.Glucose, patient.BMI,
    )

    try:
        features = _encode_features(patient)
        score    = _run_inference(features)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        return PredictionResponse(
            risk_score=round(score, 4),
            risk_category=_risk_category(score),
            prediction=_prediction_label(score),
            model_version=MODEL_VERSION,
            prediction_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow().isoformat() + "Z",
            processing_time_ms=elapsed_ms,
        )
    except Exception as exc:
        logger.error("Model inference failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model inference failed. Please contact support.",
        )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Batch Diabetes Risk Prediction",
    tags=["Prediction"],
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "All predictions returned successfully."},
        422: {"description": "Validation error in one or more patient records."},
        500: {"description": "Internal server error during batch inference."},
    },
)
async def predict_batch(batch: BatchPredictionRequest):
    """
    Accepts up to **100** patient records in a single request and returns
    a diabetes risk prediction for each. Useful for bulk screening workflows.
    """
    t0 = time.perf_counter()
    logger.info("POST /predict/batch — %d patients", len(batch.patients))

    results = []
    for patient in batch.patients:
        pt0      = time.perf_counter()
        features = _encode_features(patient)
        score    = _run_inference(features)
        elapsed_ms = round((time.perf_counter() - pt0) * 1000, 2)
        results.append(
            PredictionResponse(
                risk_score=round(score, 4),
                risk_category=_risk_category(score),
                prediction=_prediction_label(score),
                model_version=MODEL_VERSION,
                prediction_id=str(uuid.uuid4()),
                timestamp=datetime.utcnow().isoformat() + "Z",
                processing_time_ms=elapsed_ms,
            )
        )

    total_ms = round((time.perf_counter() - t0) * 1000, 2)
    return BatchPredictionResponse(
        results=results,
        total_patients=len(results),
        processing_time_ms=total_ms,
    )


@app.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="Model Metadata",
    tags=["Model"],
    responses={
        200: {"description": "Model metadata returned."},
    },
)
async def model_info():
    """
    Returns metadata about the currently loaded Random Forest model,
    including algorithm details, training dataset, feature list, and
    evaluation metrics. Fill in the metrics dict with Person 2's actual
    results once available.
    """
    return ModelInfoResponse(
        model_version=MODEL_VERSION,
        algorithm="Random Forest Classifier (sklearn.ensemble.RandomForestClassifier)",
        training_dataset="Pima Indians Diabetes Dataset — 768 records, 8 features",
        features=FEATURE_NAMES,
        n_estimators=model.n_estimators,
        metrics={
            "note": "Fill in with Person 2's actual evaluation metrics",
            "roc_auc":   0.00,
            "f1_score":  0.00,
            "precision": 0.00,
            "recall":    0.00,
        },
        last_trained="2025-01-01T00:00:00Z",
    )
