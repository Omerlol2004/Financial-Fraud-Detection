# 8-Minute Presentation Outline

## 0:00–0:45 — Problem and Motivation
- Financial fraud detection is high impact and imbalanced.
- Accuracy alone is misleading; recall, PR-AUC, and F1 matter more.

## 0:45–1:45 — System Architecture
- Show flow from raw PaySim data to Airflow, MLflow, FastAPI, and monitoring.
- Emphasize roles: Airflow orchestrates, MLflow tracks/registers, FastAPI serves.

## 1:45–2:45 — Data Pipeline
- Validation checks required columns and binary target.
- Preprocessing normalizes transaction types and creates stratified splits.

## 2:45–4:00 — Experiments and Models
- Logistic Regression baseline.
- Random Forest.
- XGBoost.
- Optuna-tuned XGBoost.
- Show MLflow experiment page and tracked metrics/artifacts.

## 4:00–5:00 — Registry and Deployment
- Register best run as `FraudDetectionModel`.
- Promote from Staging to Production.
- FastAPI loads the Production model.

## 5:00–6:00 — API Demo
- `/health`.
- `/model-info`.
- `/predict` with a sample transaction.
- Show prediction logging CSV.

## 6:00–7:00 — Monitoring
- Custom monitoring summary.
- Evidently drift report HTML.
- Discuss how logs become current data for monitoring.

## 7:00–8:00 — Lessons and Future Work
- MLOps is lifecycle management, not only model training.
- Future: cost-sensitive thresholds, alerts, CI/CD, and stronger security.
