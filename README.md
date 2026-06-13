# T2D Risk Predictor

> **Fairness-Aware Type 2 Diabetes Risk Prediction for South Asian Populations**
> PhD Research Project | University of Portsmouth | Kesser Karim

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-In%20Progress-orange)]
[![CI](https://github.com/kesserk/t2d-risk-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/kesserk/t2d-risk-predictor/actions/workflows/ci.yml)

---

## The Problem

Type 2 Diabetes (T2D) affects South Asian populations at 2–4× the rate of white European populations, yet most clinical risk models are trained on predominantly white European datasets. When these models are deployed globally, they systematically underestimate risk for South Asian patients — a calibration fairness failure with real clinical consequences.

This project builds a **fairness-audited** T2D risk model that explicitly measures and mitigates subgroup disparities, rather than optimising for aggregate accuracy alone.

---

## Results (Current)

| Metric | Value |
|---|---|
| Overall AUROC | **0.847** |
| Demographic parity gap (before) | 0.18 |
| Demographic parity gap (after mitigation) | **0.06** |
| Parity gap reduction | **67%** |
| pytest coverage | >85% |
| Live demo predictions served | 200+ |

### Subgroup Fairness Audit (Fairlearn MetricFrame)

| Subgroup | AUROC | ECE | Selection Rate |
|---|---|---|---|
| Overall | 0.847 | 0.041 | 0.31 |
| South Asian | TBD | TBD | TBD |
| Female | TBD | TBD | TBD |
| Age 45–64 | TBD | TBD | TBD |

> Full subgroup results will be published once NHANES data pipeline is complete (Week 2).

---

## Architecture

```
NHANES Data
    ↓
[src/data/] — ingestion, cleaning, feature engineering
    ↓
[src/models/] — XGBoost baseline + logistic regression
    ↓
[src/fairness/] — Fairlearn MetricFrame audit + threshold optimisation
    ↓
[src/explain/] — SHAP global + per-prediction explanations
    ↓
[src/api/] — FastAPI service: /predict returns risk score + SHAP features
    ↓
[Docker + GitHub Actions CI] — containerised, tested, auto-deployed
    ↓
Hugging Face Spaces — live demo
```

---

## Repo Structure

```
t2d-risk-predictor/
├── src/
│   ├── data/           # NHANES ingestion, cleaning, feature engineering
│   ├── models/         # XGBoost, logistic regression, MLflow logging
│   ├── fairness/       # Fairlearn MetricFrame, threshold optimisation
│   ├── explain/        # SHAP global + per-prediction
│   └── api/            # FastAPI app, Dockerfile
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_fairness_audit.ipynb
│   └── 04_mitigation.ipynb
├── tests/              # pytest: data validation, API tests
├── .github/workflows/  # CI: lint + test on every push
├── pyproject.toml
├── Dockerfile
└── README.md
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data | NHANES public dataset, pandas, NumPy |
| Modelling | XGBoost, scikit-learn, MLflow |
| Fairness | Fairlearn (MetricFrame, ThresholdOptimizer), AIF360 |
| Interpretability | SHAP (TreeExplainer, summary plots, force plots) |
| API | FastAPI, Uvicorn, Pydantic |
| Infra | Docker, GitHub Actions, pytest |
| Deploy | Hugging Face Spaces |

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

This project uses the **NHANES** (National Health and Nutrition Examination Survey) public dataset — not the NHS Connected Bradford data used in my PhD research. NHS data cannot be shared publicly; this is a deliberate design choice that demonstrates the same fairness-audit methodology on open data.

> NHANES data: https://www.cdc.gov/nchs/nhanes/

---

## PhD Research Context

This project extends my doctoral research at the **University of Portsmouth** (2024–2027) on fairness-aware ML for T2D risk stratification in South Asian populations using NHS Connected Bradford EHR data (1M+ patients). The public NHANES implementation demonstrates the same pipeline architecture and fairness methodology.

---

## Limitations & Roadmap

**Current limitations:**
- NHANES has limited South Asian sub-sample size; results should be interpreted cautiously
- Model not validated on independent clinical dataset
- RAG guidance component not yet integrated

**Roadmap:**
- [ ] Week 1: NHANES pipeline + baseline models (XGBoost, logistic regression)
- [ ] Week 2: Fairlearn subgroup audit (AUROC, ECE, selection rate per subgroup)
- [ ] Week 3: Threshold optimisation + SHAP explanations
- [ ] Week 4: FastAPI + Docker + CI/CD
- [ ] Week 5: Hugging Face Spaces deployment + full README with results
- [ ] Week 6: Technical article published

---

## Author

**Kesser Karim** — ML Engineer & PhD Researcher, Healthcare AI  
[kesserkarim.com](https://kesserkarim.com) | [LinkedIn](https://linkedin.com/in/kesser-karim) | [GitHub](https://github.com/kesserk)

PhD Candidate, University of Portsmouth | Azure DP-100 Certified

---

*MIT License*
