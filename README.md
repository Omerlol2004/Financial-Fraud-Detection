# End-to-End Financial Fraud Detection MLOps Platform

An end-to-end local MLOps platform for financial fraud detection using the PaySim transaction schema. The project goes beyond model training by demonstrating data validation, preprocessing, experiment tracking, hyperparameter tuning, model registry promotion, API deployment, orchestration, prediction logging, and monitoring.

The final verified run used a real 200,000-row stratified PaySim sample for local execution. The production model is `FraudDetectionModel` version 11, promoted through MLflow Model Registry and served by FastAPI.

## Architecture

```text
PaySim dataset sample
  -> Airflow DAG: fraud_detection_mlops_pipeline
  -> data validation
  -> preprocessing + stratified train/test split
  -> Logistic Regression, Random Forest, XGBoost
  -> Optuna tuned XGBoost candidate using validation PR-AUC
  -> MLflow experiment tracking and artifacts
  -> MLflow Model Registry: FraudDetectionModel
  -> Production promotion
  -> FastAPI serving from models:/FraudDetectionModel/Production
  -> prediction_logs.csv
  -> custom monitoring + Evidently data drift report
```

## Tech Stack

| Area | Tools |
|---|---|
| Language | Python |
| Data processing | Pandas, NumPy |
| Machine learning | scikit-learn, XGBoost |
| Hyperparameter tuning | Optuna |
| Experiment tracking | MLflow |
| Model registry | MLflow Model Registry |
| Orchestration | Apache Airflow |
| API deployment | FastAPI, Uvicorn |
| Monitoring | Evidently AI, custom monitoring summary |
| Metadata storage | PostgreSQL |
| Object storage | MinIO included for local experimentation |
| Runtime | Docker Compose |
| Testing | pytest, compileall |

## Dataset

The final verification used a PaySim dataset mirror from Hugging Face: `purulalwani/Synthetic-Financial-Datasets-For-Fraud-Detection`.

The full source dataset contains 6,362,620 rows. To keep the project practical for local laptop execution, the final run used a 200,000-row class-stratified sample saved as `data/raw/paysim.csv`. The sample preserves the original fraud rate as closely as possible.

| Split | Rows | Non-Fraud | Fraud | Fraud % |
|---|---:|---:|---:|---:|
| Raw sample | 200,000 | 199,742 | 258 | 0.1290 |
| Train | 160,000 | 159,794 | 206 | 0.1288 |
| Test | 40,000 | 39,948 | 52 | 0.1300 |
| Reference | 160,000 | 159,794 | 206 | 0.1288 |

Target column: `isFraud`

Feature columns:

- `step`
- `type`
- `amount`
- `oldbalanceOrg`
- `newbalanceOrig`
- `oldbalanceDest`
- `newbalanceDest`

Dataset files are ignored by git. To reproduce the pipeline, place a PaySim-compatible CSV at:

```text
data/raw/paysim.csv
```

## Setup Instructions

Prerequisites:

- Docker Desktop
- Docker Compose
- A PaySim-compatible CSV at `data/raw/paysim.csv`

Start the stack:

```powershell
docker compose -p oozyb up -d --build
docker compose -p oozyb ps
```

Validate the Compose configuration:

```powershell
docker compose -p oozyb config
```

Run tests:

```powershell
docker compose -p oozyb exec -T airflow-webserver python -m compileall app src monitoring dags tests
docker compose -p oozyb exec -T airflow-webserver pytest -q
```

Final verified test result:

```text
5 passed
```

## Service URLs

| Service | URL |
|---|---|
| FastAPI Swagger | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |
| Airflow UI | http://localhost:8080 |
| MinIO console | http://localhost:9001 |

Airflow credentials:

```text
username: admin
password: admin
```

## Docker Compose Commands

```powershell
docker compose -p oozyb up -d --build
docker compose -p oozyb ps
docker compose -p oozyb logs -f airflow-webserver
docker compose -p oozyb logs -f fastapi
docker compose -p oozyb down
```

## Airflow Usage

The main DAG is:

```text
fraud_detection_mlops_pipeline
```

Trigger the pipeline:

```powershell
docker compose -p oozyb exec -T airflow-webserver airflow dags trigger fraud_detection_mlops_pipeline
```

Check DAG runs:

```powershell
docker compose -p oozyb exec -T airflow-webserver airflow dags list-runs -d fraud_detection_mlops_pipeline --no-backfill --output json
```

Final verified DAG run:

```text
run_id: qa_validation_tuning_20260509_142908
status: success
```

The successful run completed these tasks:

- `validate_data`
- `preprocess_data`
- `train_baseline_model`
- `train_random_forest`
- `train_xgboost_or_lightgbm`
- `tune_best_model`
- `evaluate_best_model`
- `register_best_model`
- `promote_model`
- `generate_monitoring_report`

## MLflow Usage

Open MLflow at:

```text
http://localhost:5000
```

Experiment:

```text
fraud_detection_experiments
```

Logged runs:

- `logistic_regression`
- `random_forest`
- `xgboost`
- `tuned_xgboost`

Each run logs:

- parameters
- precision, recall, F1, ROC-AUC, PR-AUC
- false positives and false negatives
- confusion matrix artifact
- classification report artifact
- feature list artifact
- model artifact

Optuna tuning uses a validation split created from the training set and optimizes validation PR-AUC. After tuning, the final tuned model is retrained on the full training set and evaluated once on the held-out test set. The final metrics below are held-out test metrics.

Final registered model:

```text
model name: FraudDetectionModel
version: 11
stage: Production
status: READY
```

## FastAPI Usage

FastAPI loads the model from:

```text
models:/FraudDetectionModel/Production
```

Endpoints:

- `GET /health`
- `GET /model-info`
- `POST /predict`
- `POST /predict_batch`

Example request:

```json
{
  "step": 439,
  "type": "TRANSFER",
  "amount": 3653274.35,
  "oldbalanceOrg": 3653274.35,
  "newbalanceOrig": 0.0,
  "oldbalanceDest": 0.0,
  "newbalanceDest": 0.0
}
```

Example response:

```json
{
  "prediction": 1,
  "label": "Fraud",
  "fraud_probability": 0.9992955923080444,
  "model_name": "FraudDetectionModel",
  "model_stage": "Production"
}
```

Prediction logs are appended to:

```text
data/predictions/prediction_logs.csv
```

## Monitoring Usage

Generate monitoring reports:

```powershell
docker compose -p oozyb exec -T airflow-webserver python -m monitoring.generate_report
```

Outputs:

- `monitoring/reports/evidently_drift_report.html`
- `monitoring/reports/custom_monitoring_summary.json`

The final verification generated real Evidently data drift output, not fallback HTML. Monitoring mechanics were verified with local prediction logs. Stable production drift conclusions are not claimed because only a small local prediction log was available.

## Final Metrics

Accuracy is not used as the main conclusion because fraud detection is highly imbalanced. The project prioritizes PR-AUC, recall, F1, false positives, and false negatives.

Selection criterion: PR-AUC first, then recall, then F1.

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.0259 | 0.9615 | 0.0504 | 0.9937 | 0.6253 | 1,883 | 2 |
| Random Forest | 0.6852 | 0.7115 | 0.6981 | 0.9957 | 0.7773 | 17 | 15 |
| XGBoost | 0.9429 | 0.6346 | 0.7586 | 0.9982 | 0.8761 | 2 | 19 |
| Tuned XGBoost | 0.9429 | 0.6346 | 0.7586 | 0.9989 | 0.8715 | 2 | 19 |

Final selected model: XGBoost.

Tuned XGBoost was optimized on validation PR-AUC and then evaluated on the held-out test set. It was comparable to base XGBoost on thresholded metrics, but base XGBoost retained the highest held-out test PR-AUC and was selected.

## Evidence

Primary evidence report:

```text
evidence/FINAL_EVIDENCE_REPORT.md
```

Key evidence files:

- `evidence/qa_final_docker_services.txt`
- `evidence/qa_final_airflow_task_states.txt`
- `evidence/qa_final_mlflow_runs_summary.json`
- `evidence/qa_final_model_registry.json`
- `evidence/qa_final_api_predict_response.json`
- `evidence/qa_final_monitoring_report_status.json`

## Limitations

- The final run used a 200,000-row stratified sample, not the full 6.36M-row dataset.
- The test set contains 52 fraud rows, so fraud metrics are useful for portfolio verification but still sample-sensitive.
- The model uses a fixed classification threshold of `0.5`.
- Monitoring was verified with local prediction logs; stable production drift conclusions are not claimed.
- Evidently currently reports feature/data drift. Prediction or target drift would require a larger current dataset with labels or a defined prediction reference window.
- MLflow stage APIs emit deprecation warnings in MLflow 2.18, but the requested Staging/Production workflow works.

## Resume Bullet

Built an end-to-end financial fraud detection MLOps platform using Airflow, MLflow, FastAPI, Docker, PostgreSQL, and Evidently AI. Automated data validation, preprocessing, model training, Optuna tuning, experiment tracking, model registry promotion, Production model serving, prediction logging, and monitoring on a real 200,000-row stratified PaySim sample.
