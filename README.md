# T2D Risk Predictor

> **Fairness-Aware Type 2 Diabetes Risk Prediction for South Asian Populations**
> PhD Research Project | University of Portsmouth | Kesser Karim

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-In%20Progress-orange)]()
[![CI](https://github.com/kesserk/t2d-risk-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/kesserk/t2d-risk-predictor/actions/workflows/ci.yml)

---

## The Problem

Type 2 Diabetes (T2D) affects South Asian populations at 2–4× the rate of white European populations, yet most clinical risk models are trained on predominantly white European datasets. When these models are deployed globally, they can systematically underestimate risk for South Asian patients — a calibration fairness failure with real clinical consequences.

This project builds a **fairness-audited** T2D risk model that explicitly measures and mitigates subgroup disparities, rather than optimising for aggregate accuracy alone.

**Status: work in progress.** This repo is being built in the open — the roadmap below reflects what is actually done and what isn't. Results will be published here (with full subgroup audits, not just headline metrics) once the evaluation pipeline is complete.

---

## Planned Evaluation

Models are compared on discrimination **and** calibration, overall and per subgroup:

- **Models:** logistic regression baseline, XGBoost, MLP, tabular transformer
- **Metrics:** AUROC, AUPRC, Brier score, expected calibration error (ECE)
- **Fairness audit:** Fairlearn `MetricFrame` across ethnicity, sex, and age bands — selection rates, equalised odds, calibration-by-group — with threshold optimisation for mitigation
- **Explainability:** SHAP global summaries and per-prediction explanations

No performance numbers are reported yet; they will appear here with the notebooks that produce them.

---

## Architecture

```
NHANES Data
    ↓
[src/data/] — ingestion, cleaning, feature engineering
    ↓
[src/models/] — baselines + gradient boosting + deep tabular models
    ↓
[src/fairness/] — Fairlearn MetricFrame audit + threshold optimisation
    ↓
[src/explain/] — SHAP global + per-prediction
    ↓
[src/api/] — FastAPI service: /predict returns risk score + explanation
    ↓
[Docker + GitHub Actions CI] — containerised, tested
    ↓
Deployment target: Hugging Face Spaces
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data | NHANES public dataset, pandas, NumPy |
| Modelling | scikit-learn, XGBoost, PyTorch, MLflow |
| Fairness | Fairlearn (MetricFrame, ThresholdOptimizer) |
| Interpretability | SHAP |
| API | FastAPI, Uvicorn, Pydantic |
| Infra | Docker, GitHub Actions, pytest |

---

## How to Run

```bash
# Clone
git clone https://github.com/kesserk/t2d-risk-predictor
cd t2d-risk-predictor

# Install (requires Python 3.11+)
pip install -e ".[dev]"

# Run API locally
uvicorn src.api.main:app --reload

# Run tests
pytest tests/ -v

# Docker
docker build -t t2d-api .
docker run -p 8000:8000 t2d-api
```

---

## Data Note

This project uses the **NHANES** (National Health and Nutrition Examination Survey) public dataset — not the NHS data used in my PhD research. NHS data cannot be shared publicly; this is a deliberate design choice that demonstrates the same fairness-audit methodology on open data.

> NHANES data: https://www.cdc.gov/nchs/nhanes/

---

## PhD Research Context

This project runs alongside my doctoral research at the **University of Portsmouth** (2024–present, part-time) on early detection of T2D risk in South Asian populations, with data partnerships spanning Connected Bradford and UK Biobank. The public NHANES implementation demonstrates the same pipeline architecture and fairness methodology on open data.

---

## Roadmap

- [x] Repo scaffold: FastAPI skeleton, CI, Docker, project structure
- [x] NHANES EDA notebook
- [x] Baseline model notebook
- [ ] Full data pipeline in `src/data/`
- [ ] Model comparison (logistic regression, XGBoost, MLP, tabular transformer)
- [ ] Fairlearn subgroup audit (AUROC, ECE, selection rate per subgroup)
- [ ] Threshold optimisation + SHAP explanations
- [ ] Published results tables in this README
- [ ] Hugging Face Spaces demo

**Known limitations:** NHANES has a limited South Asian sub-sample; results will need cautious interpretation and the model will not be validated for clinical use.

---

## Author

**Kesser Karim** — PhD Researcher, Healthcare AI | University of Portsmouth
[kesserkarim.com](https://kesserkarim.com) | [LinkedIn](https://www.linkedin.com/in/kesser-karim/) | [GitHub](https://github.com/kesserk)

---

*MIT License · Not a medical device — research and educational use only.*
