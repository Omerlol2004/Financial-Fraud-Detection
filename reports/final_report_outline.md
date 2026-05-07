# Final Report Outline

## 1. Executive Summary
- Problem: detect fraudulent financial transactions in an imbalanced dataset.
- Solution: local MLOps platform with automated training, registry, serving, and monitoring.

## 2. Dataset
- PaySim financial fraud dataset.
- Target: `isFraud`.
- Key fields: transaction step, type, amount, origin/destination balances.

## 3. Architecture
- Airflow orchestrates pipeline tasks.
- MLflow tracks experiments and manages model registry lifecycle.
- FastAPI serves the Production model.
- Evidently and custom summaries monitor live predictions.

## 4. Data Validation and Preprocessing
- Required-column checks.
- Binary target validation.
- Transaction type normalization.
- Numeric coercion and stratified train/test split.

## 5. Modeling
- Logistic Regression baseline.
- Random Forest.
- XGBoost.
- Optuna-tuned XGBoost final candidate.

## 6. Evaluation Strategy
- Accuracy is de-emphasized due to imbalance.
- Primary metrics: recall, PR-AUC, and F1-score.
- Operational metrics: false positives and false negatives.

## 7. MLflow Tracking and Registry
- Parameters, metrics, confusion matrix, classification report, feature list, signature, and model artifacts.
- Registered model name: `FraudDetectionModel`.
- Lifecycle stages: Staging and Production.

## 8. Deployment
- FastAPI endpoints: `/health`, `/model-info`, `/predict`, `/predict_batch`.
- Production model loaded from MLflow registry.
- Prediction logs persisted to CSV.

## 9. Monitoring
- Custom summary metrics.
- Evidently data drift and distribution report.
- Reports saved in `monitoring/reports/`.

## 10. Reproducibility
- Docker Compose services.
- Required commands and setup instructions.
- Known local development assumptions.

## 11. Future Work
- Add threshold optimization by cost function.
- Add CI/CD pipeline.
- Add authenticated API access.
- Add production-grade object storage and alerts.
