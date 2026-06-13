"""FastAPI service for T2D Risk Predictor.

Returns risk score + SHAP top features + model version for each prediction.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import shap
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

app = FastAPI(
    title="T2D Risk Predictor API",
    description=(
        "Fairness-aware Type 2 Diabetes risk prediction. "
        "Returns risk score, SHAP explanations, and model version."
    ),
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    age: float = Field(..., ge=18, le=120, description="Age in years")
    bmi: float = Field(..., ge=10, le=80, description="Body Mass Index")
    glucose: float = Field(..., ge=50, le=500, description="Fasting glucose (mg/dL)")
    hba1c: float = Field(..., ge=3.0, le=15.0, description="HbA1c (%)")
    systolic_bp: float = Field(..., ge=60, le=250, description="Systolic blood pressure")
    ethnicity: str = Field(..., description="Self-reported ethnicity (e.g. 'South Asian')")
    sex: str = Field(..., description="Biological sex ('M' or 'F')")


class SHAPFeature(BaseModel):
    feature: str
    shap_value: float
    direction: str  # 'increases_risk' | 'decreases_risk'


class PredictResponse(BaseModel):
    risk_score: float = Field(..., description="Predicted T2D probability (0.0 – 1.0)")
    risk_category: str = Field(..., description="Low / Moderate / High")
    top_features: list[SHAPFeature]
    model_version: str
    fairness_note: str


# ---------------------------------------------------------------------------
# Model loading (lazy, cached at module level)
# ---------------------------------------------------------------------------

_MODEL: Any = None
_EXPLAINER: Any = None
MODEL_VERSION = "0.1.0-nhanes"
MODEL_PATH = Path(__file__).parent / "model.joblib"


def _get_model():
    global _MODEL, _EXPLAINER
    if _MODEL is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. "
                "Run the training pipeline first (see notebooks/02_baseline.ipynb)."
            )
        _MODEL = joblib.load(MODEL_PATH)
        _EXPLAINER = shap.TreeExplainer(_MODEL)
    return _MODEL, _EXPLAINER


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    """Health check for Docker / load balancer."""
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """Return T2D risk score + SHAP explanations."""
    try:
        model, explainer = _get_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    features = np.array([[req.age, req.bmi, req.glucose, req.hba1c, req.systolic_bp]])
    feature_names = ["age", "bmi", "glucose", "hba1c", "systolic_bp"]

    risk_score = float(model.predict_proba(features)[0, 1])
    risk_category = (
        "High" if risk_score >= 0.7
        else "Moderate" if risk_score >= 0.4
        else "Low"
    )

    shap_values = explainer(features).values[0]
    top_features = sorted(
        [
            SHAPFeature(
                feature=name,
                shap_value=round(float(val), 4),
                direction="increases_risk" if val > 0 else "decreases_risk",
            )
            for name, val in zip(feature_names, shap_values)
        ],
        key=lambda f: abs(f.shap_value),
        reverse=True,
    )[:3]

    return PredictResponse(
        risk_score=round(risk_score, 4),
        risk_category=risk_category,
        top_features=top_features,
        model_version=MODEL_VERSION,
        fairness_note=(
            "This model was audited for demographic parity across ethnicity and sex "
            "subgroups using Fairlearn. See the README for the full fairness report."
        ),
    )
