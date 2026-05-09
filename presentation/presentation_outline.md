# 8-Minute Presentation Outline

Use `PRESENTATION_SCRIPT.md`, `SLIDE_OUTLINE.md`, and `evidence/FINAL_EVIDENCE_REPORT.md` as the source of truth.

## 0:00-0:45 - Title and Problem

- End-to-End Financial Fraud Detection MLOps Platform.
- Fraud detection is highly imbalanced.
- Accuracy is misleading; focus on PR-AUC, recall, F1, false positives, and false negatives.

## 0:45-1:30 - Dataset

- PaySim transaction schema.
- Full source mirror: 6,362,620 rows.
- Final local run: 200,000-row stratified sample.
- Fraud rate: 0.129%.

## 1:30-2:20 - Architecture

- Docker Compose local platform.
- Airflow orchestrates.
- MLflow tracks experiments and manages registry.
- FastAPI serves Production model.
- Evidently monitors reference data vs. local prediction logs.

## 2:20-3:20 - Airflow Pipeline

- DAG: `fraud_detection_mlops_pipeline`.
- Final run: `qa_validation_tuning_20260509_142908`.
- Status: success.
- Covers validation, preprocessing, training, tuning, evaluation, registry, promotion, and monitoring.

## 3:20-4:20 - MLflow Experiment Tracking

- Runs: Logistic Regression, Random Forest, XGBoost, Tuned XGBoost.
- Logs parameters, metrics, confusion matrix, classification report, feature list, and model artifacts.
- Optuna optimizes validation PR-AUC; final metrics are reported on the held-out test set.

## 4:20-5:10 - Model Results

- Logistic Regression: high recall, too many false positives.
- Random Forest: more balanced.
- XGBoost: best PR-AUC at 0.8761 and selected.
- Tuned XGBoost: comparable thresholded metrics but slightly lower held-out test PR-AUC.

## 5:10-5:50 - Model Registry

- Registered model: `FraudDetectionModel`.
- Production version: 11.
- Status: READY.

## 5:50-6:40 - FastAPI Deployment

- Verified at `http://localhost:8000/docs`.
- Loads `models:/FraudDetectionModel/Production`.
- Logs every prediction to `data/predictions/prediction_logs.csv`.

## 6:40-7:20 - Monitoring

- Custom monitoring summary JSON.
- Real Evidently data drift report, not fallback HTML.
- Monitoring mechanics verified with local prediction logs.

## 7:20-8:00 - Conclusion and Future Work

- Complete local MLOps lifecycle verified.
- Future work: threshold optimization, larger training sample, label feedback loop, richer monitoring windows.
