"""FastAPI skeleton for T2D risk scores.

Nothing in this repository trains or saves a model. ``POST /predict`` returns
503 until a real artefact is placed at ``src/api/model.joblib``. No fairness
audit has been run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

# Column order from notebooks/02_baseline_model.ipynb (FEATURES).
# LBXGLU is excluded: notebooks/01_eda_nhanes.ipynb defines the label as
# fasting glucose >= 126 mg/dL OR a diabetes diagnosis (DIQ010 == 1).
# HbA1c is not in that feature list.
FEATURE_COLUMNS: tuple[str, ...] = (
    "RIDAGEYR",
    "BMXBMI",
    "BPXOSY1",
    "LBXTC",
    "INDFMPIR",
    "RIAGENDR",
    "RIDRETH3",
)

# Placeholder cut-points used only to label a probability. Not clinically set.
MODERATE_RISK_AT = 0.4
HIGH_RISK_AT = 0.7

FAIRNESS_NOTE = (
    "No fairness audit has been run yet. This is not a Fairlearn "
    "demographic-parity result, and there is no fairness report."
)

MODEL_MISSING_DETAIL = (
    "No model artefact at src/api/model.joblib. This repository does not "
    "train or ship a model. notebooks/02_baseline_model.ipynb is an "
    "unexecuted sketch and does not write that file. Refusing to score "
    "rather than returning a placeholder prediction."
)

MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"

# Reported only after an artefact loads. The notebooks do not stamp a version.
UNVERSIONED_MODEL = "unversioned"

app = FastAPI(
    title="T2D Risk Predictor API",
    description=(
        "Research skeleton for Type 2 diabetes risk scores. "
        "No model artefact is included and no fairness audit has been run. "
        "POST /predict returns 503 until src/api/model.joblib exists. "
        "Inputs follow the baseline notebook and exclude fasting glucose, "
        "which is part of the label."
    ),
    version="0.1.0",
)


class PredictRequest(BaseModel):
    """Same columns, same order, as the baseline notebook feature list."""

    model_config = ConfigDict(extra="forbid")

    RIDAGEYR: float = Field(..., ge=0, le=120, description="Age in years")
    BMXBMI: float = Field(..., ge=10, le=80, description="Body mass index")
    BPXOSY1: float = Field(
        ..., ge=60, le=250, description="Systolic blood pressure (mmHg)"
    )
    LBXTC: float = Field(
        ..., ge=50, le=800, description="Total cholesterol (mg/dL)"
    )
    INDFMPIR: float = Field(
        ..., ge=0, le=5, description="Family income to poverty ratio"
    )
    RIAGENDR: Literal[1, 2] = Field(
        ..., description="Sex code: 1 = male, 2 = female"
    )
    RIDRETH3: Literal[1, 2, 3, 4, 6, 7] = Field(
        ...,
        description=(
            "Race/ethnicity code: 1 Mexican American, 2 Other Hispanic, "
            "3 Non-Hispanic White, 4 Non-Hispanic Black, "
            "6 Non-Hispanic Asian, 7 Other/Multi"
        ),
    )


class SHAPFeature(BaseModel):
    feature: str
    shap_value: float
    direction: str  # "increases_risk" | "decreases_risk"


class PredictResponse(BaseModel):
    risk_score: float = Field(..., description="Predicted T2D probability (0–1)")
    risk_category: str = Field(..., description="Low / Moderate / High")
    top_features: list[SHAPFeature]
    model_version: str
    fairness_note: str


class HealthResponse(BaseModel):
    status: str
    model_artefact: str
    model_version: str | None
    fairness_audit: str
    fairness_note: str


def risk_category(score: float) -> str:
    """Map a probability to Low / Moderate / High.

    Moderate starts at 0.4 and High at 0.7. These cut-points are not
    clinically validated.
    """
    if score >= HIGH_RISK_AT:
        return "High"
    if score >= MODERATE_RISK_AT:
        return "Moderate"
    return "Low"


def feature_vector(req: PredictRequest) -> np.ndarray:
    """One row in ``FEATURE_COLUMNS`` order. No imputation or scaling."""
    values = [float(getattr(req, name)) for name in FEATURE_COLUMNS]
    return np.asarray([values], dtype=float)


_MODEL: Any = None
_EXPLAINER: Any = None


def _get_model() -> tuple[Any, Any]:
    global _MODEL, _EXPLAINER
    if _MODEL is None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(MODEL_MISSING_DETAIL)
        import joblib
        import shap

        _MODEL = joblib.load(MODEL_PATH)
        _EXPLAINER = shap.TreeExplainer(_MODEL)
    return _MODEL, _EXPLAINER


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Process is up. This does not mean a model is loaded."""
    present = MODEL_PATH.is_file()
    return HealthResponse(
        status="ok",
        model_artefact="present" if present else "missing",
        model_version=UNVERSIONED_MODEL if present else None,
        fairness_audit="not_run",
        fairness_note=FAIRNESS_NOTE,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """Score a request. Returns 503 when no model file exists."""
    try:
        model, explainer = _get_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    features = feature_vector(req)
    risk_score = float(model.predict_proba(features)[0, 1])
    shap_values = explainer(features).values[0]
    top_features = sorted(
        [
            SHAPFeature(
                feature=name,
                shap_value=round(float(val), 4),
                direction="increases_risk" if val > 0 else "decreases_risk",
            )
            for name, val in zip(FEATURE_COLUMNS, shap_values, strict=True)
        ],
        key=lambda item: abs(item.shap_value),
        reverse=True,
    )[:3]

    return PredictResponse(
        risk_score=round(risk_score, 4),
        risk_category=risk_category(risk_score),
        top_features=top_features,
        model_version=UNVERSIONED_MODEL,
        fairness_note=FAIRNESS_NOTE,
    )
