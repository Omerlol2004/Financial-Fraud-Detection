# Slide Outline

## 1. Title

- End-to-End Financial Fraud Detection MLOps Platform
- Airflow, MLflow, FastAPI, Docker, PostgreSQL, Evidently AI
- Goal: demonstrate the full ML lifecycle, not only model training

## 2. Problem Statement

- Binary classification: fraud vs. non-fraud
- Fraud is rare and high-risk
- Accuracy is misleading under class imbalance
- Focus metrics: PR-AUC, recall, F1, false positives, false negatives

## 3. Dataset

- PaySim transaction schema
- Full source mirror: 6,362,620 rows
- Final local run: 200,000-row stratified sample
- Fraud rate: 0.129%
- Train/test split: 160,000 / 40,000 rows

## 4. MLOps Architecture

- Raw data to Airflow pipeline
- MLflow for tracking and registry
- FastAPI for serving
- Prediction logs for monitoring
- Evidently for drift reporting
- Docker Compose for local infrastructure

## 5. Airflow Pipeline

- DAG: `fraud_detection_mlops_pipeline`
- Final run: `qa_validation_tuning_20260509_142908`
- Status: success
- Tasks: validation, preprocessing, training, tuning, evaluation, registry, promotion, monitoring

## 6. MLflow Tracking

- Experiment: `fraud_detection_experiments`
- Runs: Logistic Regression, Random Forest, XGBoost, Tuned XGBoost
- Logged metrics, params, confusion matrix, classification report, feature list, model artifacts
- Optuna optimized validation PR-AUC; final metrics use the held-out test split
- Supports reproducible model comparison

## 7. Model Results

| Model | Recall | F1 | PR-AUC | FP | FN |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9615 | 0.0504 | 0.6253 | 1,883 | 2 |
| Random Forest | 0.7115 | 0.6981 | 0.7773 | 17 | 15 |
| XGBoost | 0.6346 | 0.7586 | 0.8761 | 2 | 19 |
| Tuned XGBoost | 0.6346 | 0.7586 | 0.8715 | 2 | 19 |

- Final selected model: XGBoost
- Selection: PR-AUC first, then recall, then F1

## 8. Model Registry and Deployment

- Registered model: `FraudDetectionModel`
- Production version: 11
- Status: READY
- FastAPI loads `models:/FraudDetectionModel/Production`
- Endpoints: `/health`, `/model-info`, `/predict`, `/predict_batch`

## 9. Monitoring

- Prediction logs appended to `data/predictions/prediction_logs.csv`
- Custom monitoring summary JSON
- Evidently drift report HTML
- Real Evidently output verified, not fallback HTML
- Monitoring mechanics verified with local prediction logs

## 10. Conclusion and Future Work

- Complete local MLOps lifecycle verified end-to-end
- Final DAG run succeeded
- Production model served through FastAPI
- Future work: threshold optimization, larger training sample, label feedback loop, richer monitoring windows
