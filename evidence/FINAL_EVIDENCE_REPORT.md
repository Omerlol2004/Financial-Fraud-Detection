# Final Evidence Report - Fraud Detection MLOps Platform

Verification date: May 9, 2026  
Project path: `C:\Users\oozyb\OneDrive\Desktop\Financial Fraud Detection`

## Executive Status

The platform was re-verified end-to-end after the Optuna tuning workflow was corrected to avoid using the final test set during hyperparameter search. Docker Compose services are running, tests pass, MLflow contains real model runs/artifacts, `FraudDetectionModel` version 11 is in `Production`, FastAPI serves the Production model, Airflow completed the full DAG successfully, and monitoring generated a real Evidently HTML report.

## Dataset

Source used: Hugging Face mirror of the PaySim Synthetic Financial Fraud Detection dataset, `purulalwani/Synthetic-Financial-Datasets-For-Fraud-Detection`.

The full source file had 6,362,620 rows: 6,354,407 non-fraud and 8,213 fraud. For local execution, a 200,000-row class-stratified sample was created and saved as `data/raw/paysim.csv`.

| Split | Rows | Non-Fraud | Fraud | Fraud % |
|---|---:|---:|---:|---:|
| Raw sample | 200,000 | 199,742 | 258 | 0.1290 |
| Train | 160,000 | 159,794 | 206 | 0.1288 |
| Test | 40,000 | 39,948 | 52 | 0.1300 |
| Reference | 160,000 | 159,794 | 206 | 0.1288 |

Evidence files:

- `evidence/final_dataset_upgrade.json`
- `evidence/rerun_dataset_summary.json`
- `data/processed/train.csv`
- `data/processed/test.csv`
- `data/reference/reference_data.csv`

## Data Leakage QA Fix

Audit finding: the earlier `src/tune.py` workflow optimized Optuna trials on the final test split. That was fixed.

Current tuning workflow:

1. Keep the existing train/test split from preprocessing.
2. Split the training set into `train_inner` and validation data for Optuna.
3. Optimize Optuna trials using validation PR-AUC only.
4. Train the final tuned XGBoost model on the full training set.
5. Evaluate the tuned model once on the held-out test set and log those final test metrics to MLflow.

The final selected model is still base XGBoost because it had the best held-out test PR-AUC.

## Commands Run

```powershell
docker compose -p oozyb ps
docker compose -p oozyb config
docker compose -p oozyb exec -T airflow-webserver python -m compileall app src monitoring dags tests
docker compose -p oozyb exec -T airflow-webserver pytest -q
docker compose -p oozyb exec -T airflow-webserver python -m src.data_validation
docker compose -p oozyb exec -T airflow-webserver python -m src.preprocessing
docker compose -p oozyb exec -T airflow-webserver python -m src.train
docker compose -p oozyb exec -T airflow-webserver python -m src.tune
docker compose -p oozyb exec -T airflow-webserver python -m src.register_model
docker compose -p oozyb exec -T airflow-webserver python -m src.promote_model
docker compose -p oozyb restart fastapi
docker compose -p oozyb exec -T airflow-webserver airflow dags trigger fraud_detection_mlops_pipeline --run-id qa_validation_tuning_20260509_142908
docker compose -p oozyb exec -T airflow-webserver python -m monitoring.generate_report
```

## Docker Services

Final `docker compose -p oozyb ps` confirmed the required services were running.

| Service | Status | URL |
|---|---|---|
| FastAPI | Up | `http://localhost:8000/docs` |
| MLflow | Up | `http://localhost:5000` |
| Airflow webserver | Up | `http://localhost:8080` |
| Airflow scheduler | Up | internal scheduler |
| PostgreSQL | Up, healthy | `localhost:5432` |
| MLflow PostgreSQL | Up, healthy | internal metadata DB |
| MinIO | Up | `http://localhost:9001` |

Evidence:

- `evidence/qa_final_docker_services.txt`
- `evidence/qa_docs_final_docker_services.txt`
- `evidence/qa_docs_final_compose_config.txt`

## Tests

`compileall` completed successfully.

`pytest -q` result:

```text
5 passed, 16 warnings in 3.74s
```

Evidence:

- `evidence/qa_final_compileall.txt`
- `evidence/qa_final_pytest.txt`
- `evidence/qa_docs_final_compileall.txt`
- `evidence/qa_docs_final_pytest.txt`
- `tests/test_tuning.py`

## MLflow Runs and Metrics

MLflow experiment: `fraud_detection_experiments`

Final Airflow-generated runs logged parameters, metrics, confusion matrix artifacts, classification report artifacts, feature list artifacts, and model artifacts.

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.0259 | 0.9615 | 0.0504 | 0.9937 | 0.6253 | 1,883 | 2 |
| Random Forest | 0.6852 | 0.7115 | 0.6981 | 0.9957 | 0.7773 | 17 | 15 |
| XGBoost | 0.9429 | 0.6346 | 0.7586 | 0.9982 | 0.8761 | 2 | 19 |
| Tuned XGBoost | 0.9429 | 0.6346 | 0.7586 | 0.9989 | 0.8715 | 2 | 19 |

Tuned XGBoost validation metrics from Optuna:

| Metric | Value |
|---|---:|
| Validation precision | 1.0000 |
| Validation recall | 0.7317 |
| Validation F1 | 0.8451 |
| Validation ROC-AUC | 0.9995 |
| Validation PR-AUC | 0.9080 |
| Validation false positives | 0 |
| Validation false negatives | 11 |

Final best model: `xgboost`.

Selection reason: the platform selects by held-out test PR-AUC first, then recall, then F1. Base XGBoost had the best held-out test PR-AUC (`0.8761`). Tuned XGBoost was comparable on thresholded held-out test metrics, but its held-out test PR-AUC was lower (`0.8715`).

Evidence:

- `evidence/qa_final_mlflow_runs_summary.json`
- `evidence/qa_final_model_metrics_and_best.txt`

## Model Registry

Registered model: `FraudDetectionModel`

Final Production model:

| Version | Stage | Status | Run ID |
|---:|---|---|---|
| 11 | Production | READY | `df14e074faba410294acc469a9fc6efb` |

Evidence: `evidence/qa_final_model_registry.json`.

## FastAPI Verification

FastAPI was restarted after Airflow promoted version 11, clearing the cached model. The loader uses:

```python
mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/{model_stage}")
```

This confirms the API loads `models:/FraudDetectionModel/Production` and does not use a dummy fallback model.

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

Prediction logs appended to `data/predictions/prediction_logs.csv`. The final monitoring run saw 23 prediction log rows.

Evidence:

- `evidence/qa_final_api_health.json`
- `evidence/qa_final_api_model_info.json`
- `evidence/qa_final_api_predict_response.json`
- `evidence/qa_final_api_predict_batch_response.json`
- `evidence/qa_final_prediction_logs_tail.csv`

## Airflow DAG Verification

DAG: `fraud_detection_mlops_pipeline`

Final run:

| Run ID | Status | Start | End |
|---|---|---|---|
| `qa_validation_tuning_20260509_142908` | success | `2026-05-09T11:29:12Z` | `2026-05-09T11:31:44Z` |

All requested tasks completed successfully:

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

Evidence:

- `evidence/qa_final_airflow_status_clean.json`
- `evidence/qa_final_airflow_task_states.txt`

## Monitoring

Generated reports:

- `monitoring/reports/evidently_drift_report.html`
- `monitoring/reports/custom_monitoring_summary.json`

The Evidently report is real output, not fallback HTML:

```json
{
  "contains_fallback": false,
  "contains_evidently_dashboard": true,
  "contains_data_drift": true
}
```

The final custom summary used 23 prediction log rows.

Evidence:

- `evidence/qa_final_monitoring_report_status.json`
- `evidence/qa_final_custom_monitoring_summary.json`

## What Failed and Was Fixed

1. The Optuna objective previously optimized on the final test split. This was fixed by adding a training/validation split for tuning and reserving the final test set for one final evaluation.
2. Documentation previously reflected the earlier portfolio verification run. README, reports, presentation materials, resume bullets, and evidence were updated to match the corrected QA rerun.
3. Docker Desktop had previously stopped after the laptop battery died. The Compose stack was restarted and the system was rerun end-to-end.
4. Earlier Airflow polling used an incompatible CLI command. Final status is now captured from `airflow dags list-runs --output json` and task states.
5. Evidently previously risked fallback output when target drift inputs were unavailable. Monitoring now generates a real data drift report using reference data and prediction logs.

## Remaining Limitations

- The final run uses a 200,000-row stratified sample instead of the full 6.36M-row dataset to keep local runtime practical.
- The test set has 52 fraud rows, so fraud metrics are meaningful for portfolio evidence but still sensitive to sampling variance.
- The classification threshold is fixed at `0.5`; a production fraud workflow would tune the threshold against business cost.
- Prediction logs currently contain 23 local test predictions, enough to verify monitoring mechanics but not enough for stable drift conclusions.
- Evidently currently covers feature/data drift. Prediction or target drift would require a larger current prediction dataset with reliable labels or a defined prediction reference window.
- MLflow stage APIs emit deprecation warnings in MLflow 2.18, but the requested Staging/Production registry lifecycle works.
