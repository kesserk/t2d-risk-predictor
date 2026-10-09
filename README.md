# T2D Risk Predictor

> **Toward fairness-aware Type 2 diabetes risk prediction for South Asian populations**
> PhD Research Project | University of Portsmouth | Kesser Karim

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-In%20Progress-orange)]()
[![CI](https://github.com/kesserk/t2d-risk-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/kesserk/t2d-risk-predictor/actions/workflows/ci.yml)

---

## The Problem

Type 2 Diabetes (T2D) affects South Asian populations at 2–4× the rate of white European populations, yet most clinical risk models are trained on predominantly white European datasets. When these models are deployed globally, they can systematically underestimate risk for South Asian patients — a calibration fairness failure with real clinical consequences.

This project is working toward a T2D risk model that measures subgroup disparities, rather than optimising for aggregate accuracy alone. **No fairness audit has been run yet**, and there is no fairness report.

**Status: work in progress. No results yet.** The sections below separate what is in the repo from what is only planned. Performance numbers will be published here, with the notebooks that produce them, after an evaluation has actually been run.

---

## What exists today

| Path | State |
|---|---|
| `notebooks/01_eda_nhanes.ipynb` | Sketch only. Not executed. Defines the T2D label as fasting glucose (`LBXGLU`) ≥ 126 mg/dL **or** a prior diagnosis (`DIQ010 == 1`). |
| `notebooks/02_baseline_model.ipynb` | Sketch only. Not executed. Lists the training columns. Does **not** write `src/api/model.joblib`. |
| `src/api/main.py` | FastAPI skeleton. `GET /health` reports that the process is up. `POST /predict` returns **503** until a real model file exists. |
| `tests/` | Checks the 0.4 / 0.7 risk cut-points, the feature list, the missing-model response, and the fairness note. |
| `Dockerfile`, `.github/workflows/ci.yml` | Image build and CI. They do not deploy the service. |

There is **no drift monitoring**, and this is **not a production system**. Hugging Face Spaces is a possible later demo, not a current deployment.

### Feature schema

`POST /predict` accepts the same columns, in the same order, as `FEATURES` in `notebooks/02_baseline_model.ipynb`:

| Column | Meaning |
|---|---|
| `RIDAGEYR` | Age in years |
| `BMXBMI` | Body mass index |
| `BPXOSY1` | Systolic blood pressure (mmHg) |
| `LBXTC` | Total cholesterol (mg/dL) |
| `INDFMPIR` | Family income-to-poverty ratio |
| `RIAGENDR` | Sex code: 1 = male, 2 = female |
| `RIDRETH3` | NHANES race/ethnicity code: 1, 2, 3, 4, 6, or 7 |

Fasting glucose (`LBXGLU`) is **not** an input. The EDA notebook uses it in the label, so treating it as a feature would leak the outcome. HbA1c is not a training column either. Unknown fields are rejected.

Risk labels use cut-points of **0.4** (moderate) and **0.7** (high). They are placeholders, not clinical thresholds.

The baseline sketch fits an imputer and a scaler inside a scikit-learn pipeline and does not save either of them. This API does not impute or scale. A future `src/api/model.joblib` must be an estimator whose `predict_proba` accepts these seven raw columns. Nothing in the repo creates that file, and the API will not invent a score without it.

### Planned directories (not in the tree)

- `src/data/` — ingestion, cleaning, feature engineering
- `src/models/` — training entry points
- `src/fairness/` — Fairlearn audit
- `src/explain/` — saved SHAP reports

`GET /health` reports `fairness_audit: "not_run"`.

---

## Planned evaluation

When a real training pipeline exists, models should be compared on discrimination **and** calibration, overall and per subgroup:

- **Models:** logistic regression baseline, XGBoost, MLP, tabular transformer
- **Metrics:** AUROC, AUPRC, Brier score, expected calibration error (ECE)
- **Fairness audit:** Fairlearn `MetricFrame` across ethnicity, sex, and age bands — selection rates, equalised odds, calibration-by-group — with threshold optimisation for mitigation
- **Explainability:** SHAP global summaries and per-prediction explanations

None of that is implemented. `src/fairness/` and `src/explain/` do not exist. The API can call SHAP only after a real tree model is loaded; that path has not been run. No performance numbers are reported yet.

---

## Tech stack

Intended tools. Several are declared dependencies or imports in unexecuted notebooks; they are not all exercised by code that runs today.

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

`GET /health` returns 200 and states that no model is loaded and no fairness audit has been run. `POST /predict` returns 503 on a fresh clone. That is the expected result until a real `src/api/model.joblib` is produced outside this repository's current notebooks.

---

## Data Note

This project uses the **NHANES** (National Health and Nutrition Examination Survey) public dataset — not the NHS data used in my PhD research. NHS data cannot be shared publicly. The aim is to try the same kind of fairness checks on open data. Those checks are not implemented here yet.

> NHANES data: https://www.cdc.gov/nchs/nhanes/

The notebooks expect local files under `data/raw/` and `data/processed/`. Those directories are not in the repo, and the notebooks have not been executed.

---

## PhD Research Context

This project runs alongside my doctoral research at the **University of Portsmouth** (2024–present, part-time) on early detection of T2D risk in South Asian populations, with data partnerships spanning Connected Bradford and UK Biobank. The public NHANES notebooks are a sketch of that pipeline on open data. The pipeline itself is not built yet.

---

## Roadmap

- [x] Repo scaffold: FastAPI skeleton, CI, Docker
- [x] NHANES EDA notebook (sketch, not executed)
- [x] Baseline model notebook (sketch, not executed, does not save a model)
- [ ] Data pipeline in `src/data/` (directory does not exist yet)
- [ ] Model comparison (logistic regression, XGBoost, MLP, tabular transformer)
- [ ] Fairlearn subgroup audit (AUROC, ECE, selection rate per subgroup)
- [ ] Threshold optimisation + SHAP explanations that have actually been run
- [ ] A real `src/api/model.joblib` the API can load
- [ ] Published results tables in this README
- [ ] Hugging Face Spaces demo

**Known limitations:** NHANES has a limited South Asian sub-sample. There is no trained model, no fairness audit, and no drift monitoring. This is not a production service and not a medical device. Results, once they exist, will need cautious interpretation and will not be validated for clinical use.

---

## Author

**Kesser Karim** — PhD Researcher, Healthcare AI | University of Portsmouth
[kesserkarim.com](https://kesserkarim.com) | [LinkedIn](https://www.linkedin.com/in/kesser-karim/) | [GitHub](https://github.com/kesserk)

---

*MIT License · Not a medical device — research and educational use only.*
