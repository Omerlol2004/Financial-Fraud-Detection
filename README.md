# End-to-End Financial Fraud Detection MLOps Platform

This repository implements a local MLOps platform for binary fraud detection on the PaySim financial transaction dataset. The project emphasizes the complete ML lifecycle instead of a notebook-only model: validation, preprocessing, model comparison, Optuna tuning, MLflow tracking and registry, production serving with FastAPI, Airflow orchestration, prediction logging, and monitoring with Evidently AI.

## Architecture

```text
Raw PaySim CSV
→ Airflow DAG
→ Data Validation
→ Preprocessing / Train-Test Split
→ Logistic Regression + Random Forest + XGBoost
→ Optuna-tuned XGBoost
→ MLflow metrics/artifacts/model logging
→ Best model registration as FraudDetectionModel
→ Staging / Production lifecycle
→ FastAPI production model serving
→ Prediction logs
→ Custom + Evidently monitoring reports
```

## Tech Stack

- Python, Pandas, NumPy, scikit-learn, XGBoost
- Optuna for hyperparameter tuning
- MLflow for experiment tracking, artifacts, and model registry
- Apache Airflow for orchestration
- FastAPI for model serving
- Evidently AI for drift and distribution monitoring
- PostgreSQL for metadata stores
- Docker Compose for local services
- Optional MinIO service included for local object-storage experimentation

## Repository Layout

```text
dags/                         Airflow DAG definition
src/                          Pipeline modules called by Airflow
app/                          FastAPI application
monitoring/                   Evidently/custom monitoring entrypoint
data/raw/                     Place PaySim CSV here as paysim.csv
data/processed/               Generated train/test and metadata files
data/predictions/             Prediction logs
reports/                      Written report outline
presentation/                 8-minute presentation outline
docker/                       API and Airflow Dockerfiles
```

## Dataset Setup

Download the PaySim Financial Fraud Detection dataset and save the CSV as:

```bash
data/raw/paysim.csv
```

The expected target is `isFraud`. Required feature columns are `step`, `type`, `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, and `newbalanceDest`.

## Local Setup Without Docker

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.data_validation
python -m src.preprocessing
python -m src.train
python -m src.tune
python -m src.evaluate
python -m src.register_model
python -m src.promote_model
uvicorn app.main:app --reload
```

The API is available at <http://localhost:8000>. MLflow defaults to local file tracking when `MLFLOW_TRACKING_URI` is not set.

## Docker Compose Setup

```bash
cp .env.example .env
docker compose up --build
```

Services:

- MLflow: <http://localhost:5000>
- Airflow: <http://localhost:8080>
- FastAPI: <http://localhost:8000>
- MinIO console: <http://localhost:9001>

After the containers are running, open Airflow and trigger the `fraud_detection_mlops_pipeline` DAG.

## API Examples

### Health

```bash
curl http://localhost:8000/health
```

### Single Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "step": 1,
    "type": "TRANSFER",
    "amount": 1000.0,
    "oldbalanceOrg": 1000.0,
    "newbalanceOrig": 0.0,
    "oldbalanceDest": 0.0,
    "newbalanceDest": 1000.0
  }'
```

Example response:

```json
{
  "prediction": 1,
  "label": "Fraud",
  "fraud_probability": 0.94,
  "model_name": "FraudDetectionModel",
  "model_stage": "Production"
}
```

Predictions are appended to `data/predictions/prediction_logs.csv`.

## Metrics Tracked in MLflow

The pipeline prioritizes imbalanced-classification metrics over accuracy:

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- False positives
- False negatives
- Confusion matrix image
- Classification report
- Feature list
- Model signature and trained model artifact

## Monitoring

Generate monitoring artifacts with:

```bash
python monitoring/generate_report.py
```

Outputs are saved under `monitoring/reports/`:

- `custom_monitoring_summary.json`
- `evidently_drift_report.html`
