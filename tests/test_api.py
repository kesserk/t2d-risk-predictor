"""Tests against the API that actually ships in this repo."""

from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi.testclient import TestClient

from src.api.main import (
    FAIRNESS_NOTE,
    FEATURE_COLUMNS,
    HIGH_RISK_AT,
    MODEL_PATH,
    MODERATE_RISK_AT,
    PredictRequest,
    app,
    feature_vector,
    risk_category,
)

ROOT = Path(__file__).resolve().parents[1]

VALID_PAYLOAD = {
    "RIDAGEYR": 52,
    "BMXBMI": 27.4,
    "BPXOSY1": 128,
    "LBXTC": 190,
    "INDFMPIR": 2.1,
    "RIAGENDR": 1,
    "RIDRETH3": 6,
}


def _notebook_source(name: str) -> str:
    notebook = json.loads((ROOT / "notebooks" / name).read_text())
    chunks: list[str] = []
    for cell in notebook["cells"]:
        chunks.append("".join(cell["source"]))
    return "\n".join(chunks)


def test_risk_category_uses_api_thresholds() -> None:
    """Cut-points are 0.4 and 0.7, matching ``risk_category`` in the API."""
    assert MODERATE_RISK_AT == 0.4
    assert HIGH_RISK_AT == 0.7
    assert risk_category(0.0) == "Low"
    assert risk_category(0.35) == "Low"
    assert risk_category(0.399) == "Low"
    assert risk_category(0.4) == "Moderate"
    assert risk_category(0.65) == "Moderate"
    assert risk_category(0.699) == "Moderate"
    assert risk_category(0.7) == "High"
    assert risk_category(1.0) == "High"


def test_feature_schema_matches_training_notebook() -> None:
    source = _notebook_source("02_baseline_model.ipynb")
    match = re.search(r"FEATURES = \[(.*?)\]", source, re.DOTALL)
    assert match is not None
    notebook_features = re.findall(r"'([A-Z0-9]+)'", match.group(1))
    assert notebook_features == list(FEATURE_COLUMNS)

    label = _notebook_source("01_eda_nhanes.ipynb")
    assert "LBXGLU" in label
    assert ">= 126" in label
    assert "LBXGLU" not in FEATURE_COLUMNS
    assert "hba1c" not in {name.lower() for name in FEATURE_COLUMNS}


def test_feature_vector_follows_training_order() -> None:
    row = feature_vector(PredictRequest(**VALID_PAYLOAD))
    assert row.shape == (1, len(FEATURE_COLUMNS))
    assert list(row[0]) == [float(VALID_PAYLOAD[name]) for name in FEATURE_COLUMNS]


def test_leaked_and_unknown_fields_are_rejected() -> None:
    client = TestClient(app)
    leaked = {**VALID_PAYLOAD, "glucose": 130, "hba1c": 6.5}
    response = client.post("/predict", json=leaked)
    assert response.status_code == 422

    schema = app.openapi()["components"]["schemas"]["PredictRequest"]
    assert list(schema["properties"]) == list(FEATURE_COLUMNS)
    assert "glucose" not in schema["properties"]
    assert "hba1c" not in schema["properties"]
    assert "age" not in schema["properties"]


def test_predict_fails_clearly_when_model_is_missing() -> None:
    assert not MODEL_PATH.is_file()
    assert not (ROOT / "src" / "api" / "model.joblib").exists()

    client = TestClient(app)
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "model.joblib" in detail
    assert "does not train or ship a model" in detail
    assert "risk_score" not in response.json()


def test_health_does_not_claim_an_audit_or_a_model() -> None:
    client = TestClient(app)
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_artefact"] == "missing"
    assert body["model_version"] is None
    assert body["fairness_audit"] == "not_run"
    assert body["fairness_note"] == FAIRNESS_NOTE

    note = FAIRNESS_NOTE.lower()
    assert "no fairness audit has been run" in note
    assert "there is no fairness report" in note
    assert "this model was audited" not in note
    description = app.description.lower()
    assert "no fairness audit has been run" in description
