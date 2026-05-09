# Fraud Detection MLOps Platform Evidence Report

Verification date: 2026-05-08  
Project path: `C:\Users\oozyb\OneDrive\Desktop\Financial Fraud Detection`

## Summary

The local MLOps platform runs end to end with Docker Compose, Airflow orchestration, MLflow experiment tracking and registry, FastAPI serving, prediction logging, and Evidently monitoring output.

The final Airflow evidence run was:

`manual_evidence_20260508231750`

Final status: `success`

## Service URLs

| Service | URL | Notes |
|---|---|---|
| FastAPI Swagger | http://localhost:8000/docs | Running and tested |
| MLflow UI | http://localhost:5000 | Running and contains experiment/model registry evidence |
| Airflow UI | http://localhost:8080 | Login: `admin / admin` |
| MinIO Console | http://localhost:9001 | Login: `minio / minio123` |

## Commands Run

```powershell
docker compose -p oozyb ps
docker compose -p oozyb config --quiet
python -m compileall app src monitoring dags tests
docker compose -p oozyb run --rm --no-deps fastapi python -c "import pandas, numpy, sklearn, xgboost, optuna, mlflow, fastapi, evidently; print('DEPENDENCY_IMPORTS_OK')"
docker compose -p oozyb run --rm --no-deps --volume "C:\Users\oozyb\OneDrive\Desktop\Financial Fraud Detection\tests:/app/tests" fastapi pytest -q /app/tests
docker compose -p oozyb exec -T airflow-webserver python -m src.data_validation
docker compose -p oozyb exec -T airflow-webserver python -m src.preprocessing
docker compose -p oozyb exec -T airflow-webserver python -m src.train
docker compose -p oozyb exec -T airflow-webserver python -m src.tune
docker compose -p oozyb exec -T airflow-webserver python -m src.register_model
docker compose -p oozyb exec -T airflow-webserver python -m src.promote_model
docker compose -p oozyb restart fastapi
curl.exe -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" --data-binary "@evidence\api_predict_request_final.json"
curl.exe -s -X POST http://localhost:8000/predict_batch -H "Content-Type: application/json" --data-binary "@evidence\api_predict_batch_request_final.json"
docker compose -p oozyb exec -T airflow-webserver airflow dags trigger fraud_detection_mlops_pipeline --run-id manual_evidence_20260508231750
docker compose -p oozyb exec -T airflow-webserver airflow tasks states-for-dag-run fraud_detection_mlops_pipeline manual_evidence_20260508231750
```

## Docker / Services

`docker compose -p oozyb ps` confirms the required services are running:

- FastAPI: `oozyb-fastapi-1`
- MLflow: `oozyb-mlflow-1`
- Airflow webserver: `oozyb-airflow-webserver-1`
- Airflow scheduler: `oozyb-airflow-scheduler-1`
- PostgreSQL: `oozyb-postgres-1`
- MLflow PostgreSQL: `oozyb-mlflow-postgres-1`
- MinIO: `oozyb-minio-1`

Compose config status: `COMPOSE_CONFIG_OK`.

Evidence:

- `evidence/docker_services_final.txt`
- `evidence/docker_services.png`
- `evidence/compose_config.txt`

## Dependencies and Tests

Dependency import check inside the FastAPI container returned:

`DEPENDENCY_IMPORTS_OK`

Compile check passed:

`python -m compileall app src monitoring dags tests`

Pytest passed:

`4 passed, 16 warnings`

Evidence:

- `evidence/dependency_imports.txt`
- `evidence/compileall.txt`
- `evidence/pytest.txt`

## Dataset Verification

Raw data exists at:

`data/raw/paysim.csv`

Target column:

`isFraud`

Preprocessing created:

- `data/processed/train.csv`
- `data/processed/test.csv`
- `data/reference/reference_data.csv`

Class distribution:

| Split | Rows | Non-fraud `0` | Fraud `1` |
|---|---:|---:|---:|
| Raw | 30 | 21 | 9 |
| Train | 24 | 17 | 7 |
| Test | 6 | 4 | 2 |
| Reference | 24 | 17 | 7 |

Evidence:

- `evidence/data_validation.txt`
- `evidence/preprocessing.txt`
- `evidence/dataset_summary.json`

## MLflow Training Verification

Experiment name:

`fraud_detection_experiments`

Required model runs were created and verified:

| Model | Run ID | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|---:|
| Logistic Regression | `8e4e6bf5e0a84e53ac59c33932f1e58d` | 0.667 | 1.000 | 0.800 | 0.875 | 0.833 |
| Random Forest | `b6a8e9a0ffcf4dfa8a72048c62df159a` | 0.500 | 0.500 | 0.500 | 0.875 | 0.833 |
| XGBoost | `c8a136ac24f643f784ec0a08fa3f3169` | 1.000 | 0.500 | 0.667 | 0.750 | 0.750 |

For each required model, MLflow contains:

- Parameters
- `precision`
- `recall`
- `f1`
- `roc_auc`
- `pr_auc`
- Confusion matrix artifact
- Classification report artifact
- Model artifact

The best untuned model was `logistic_regression`. Selection is implemented as `PR-AUC`, then `recall`, then `F1`. Logistic Regression and Random Forest tied on PR-AUC, so Logistic Regression won by higher recall and F1.

Evidence:

- `evidence/training_output.txt`
- `evidence/mlflow_training_runs.json`
- `evidence/training_model_metrics.json`
- `evidence/training_best_run.json`
- `evidence/mlflow_experiment_runs.png`

## Optuna Tuning Verification

Optuna tuning ran with 20 trials.

Manual tuning evidence run:

- Run ID: `6b5de31bc8284764bf2d6aa886b335b3`
- Best PR-AUC: 0.750
- Recall: 0.500
- F1: 0.667

Final end-to-end Airflow production run:

- Registered model version: `5`
- Run ID: `13ab71e92f5241c194db5d3120569580`
- Model: `tuned_xgboost`
- Best PR-AUC: 0.833
- Recall: 0.500
- F1: 0.667

Final production best parameters:

```json
{
  "n_estimators": "442",
  "max_depth": "10",
  "learning_rate": "0.05324262080824633",
  "subsample": "0.9509343735427023",
  "colsample_bytree": "0.804305317040194",
  "min_child_weight": "1"
}
```

The tuned model is comparable to the strongest baseline by PR-AUC and better than the untuned XGBoost PR-AUC in this smoke dataset. Its recall and F1 are lower than Logistic Regression on this tiny dataset, so the final metric conclusion should be rerun on the full PaySim dataset before presenting model quality.

Evidence:

- `evidence/optuna_tuning_output.txt`
- `evidence/optuna_summary.json`
- `evidence/mlflow_tuned_run.json`
- `evidence/production_model_run.json`

## MLflow Model Registry Verification

Registered model:

`FraudDetectionModel`

Model Registry contains multiple versions. Current Production version:

| Model | Version | Stage | Status | Run ID |
|---|---:|---|---|---|
| FraudDetectionModel | 5 | Production | READY | `13ab71e92f5241c194db5d3120569580` |

Evidence:

- `evidence/model_registry_after_airflow.json`
- `evidence/mlflow_registered_model.png`

## FastAPI Verification

FastAPI is running at:

http://localhost:8000/docs

Verified endpoints:

- `GET /health`
- `GET /model-info`
- `POST /predict`
- `POST /predict_batch`

`app/model_loader.py` loads the MLflow Production model with:

`models:/FraudDetectionModel/Production`

There is no dummy fallback path in the serving loader.

Example `/predict` request:

```json
{"step":15,"type":"TRANSFER","amount":250000.0,"oldbalanceOrg":250000.0,"newbalanceOrig":0.0,"oldbalanceDest":0.0,"newbalanceDest":0.0}
```

Example `/predict` response:

```json
{"prediction":1,"label":"Fraud","fraud_probability":0.92696613073349,"model_name":"FraudDetectionModel","model_stage":"Production"}
```

Prediction logs were appended to:

`data/predictions/prediction_logs.csv`

Evidence:

- `evidence/api_health.json`
- `evidence/api_model_info.json`
- `evidence/api_predict_request_final.json`
- `evidence/api_predict_response_final.json`
- `evidence/api_predict_batch_request_final.json`
- `evidence/api_predict_batch_response_final.json`
- `evidence/prediction_logs_tail.csv`
- `evidence/model_loader_source.txt`
- `evidence/fastapi_swagger.png`
- `evidence/predict_response.png`

## Airflow Verification

DAG:

`fraud_detection_mlops_pipeline`

Final manual run:

`manual_evidence_20260508231750`

Final status:

`success`

All DAG tasks completed successfully:

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

Import errors:

`No data found`

Evidence:

- `evidence/airflow_runs.txt`
- `evidence/airflow_task_states.txt`
- `evidence/airflow_import_errors.txt`
- `evidence/airflow_successful_dag_run.png`

## Monitoring Verification

Monitoring report output exists at:

`monitoring/reports/evidently_drift_report.html`

Custom monitoring summary exists at:

`monitoring/reports/custom_monitoring_summary.json`

The Evidently HTML report did not contain fallback markers:

`NO_FALLBACK_MARKERS_FOUND`

The report used:

- Reference data: `data/reference/reference_data.csv`
- Prediction logs: `data/predictions/prediction_logs.csv`

Evidence:

- `monitoring/reports/evidently_drift_report.html`
- `monitoring/reports/custom_monitoring_summary.json`
- `evidence/monitoring_report_files.txt`
- `evidence/monitoring_fallback_check.txt`
- `evidence/custom_monitoring_summary.json`
- `evidence/monitoring_report.png`

## What Failed and Was Fixed

1. The first full rebuild from the project folder hit a PyPI read timeout while downloading large dependencies. The already-built validated images were reused with `docker compose -p oozyb up -d --no-build`, and the services started cleanly.
2. Preprocessing originally wrote `data/reference/reference.csv`, while the required evidence path was `data/reference/reference_data.csv`. `src/config.py` was corrected and preprocessing was rerun.
3. A PowerShell/curl quoting attempt produced invalid JSON for `/predict`; the test was rerun using JSON request files and succeeded.
4. Earlier startup work had already resolved Docker Compose conflict markers, the NumPy/Evidently dependency conflict, shared Docker volume permissions, and the missing Airflow `promote_model` task.

## Remaining Limitations

1. `data/raw/paysim.csv` is currently a 30-row PaySim-shaped smoke dataset, not the full public PaySim dataset. The platform is verified end to end, but final portfolio metrics should be rerun on the full dataset.
2. The MLflow registry stage APIs emit deprecation warnings in MLflow 2.18. The current assignment requires Staging/Production lifecycle, so stages are still used intentionally.
3. MinIO is running, but MLflow artifacts are currently stored in the local MLflow artifact volume, not S3/MinIO.
4. `/model-info` confirms model name and stage but does not expose the exact MLflow version number. Registry evidence confirms Production version 5.

## Evidence Index

See:

`evidence/evidence_file_index.txt`
