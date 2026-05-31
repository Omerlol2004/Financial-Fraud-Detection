# End-to-End Financial Fraud Detection MLOps Platform

An end-to-end local MLOps platform for financial fraud detection using the PaySim transaction schema. The project demonstrates real-world software engineering and data science practices, covering: data validation, preprocessing, model training, detailed experiment tracking, nested hyperparameter tuning, model registry management, containerized API deployment, orchestration, live prediction logging, and automated data drift monitoring.

The final verified run uses a real 200,000-row stratified PaySim sample for local execution. The champion model `FraudDetectionModel` was promoted to the Production stage and served through FastAPI using `models:/FraudDetectionModel/Production`.

---

## 🏗️ Architecture

```text
PaySim dataset sample
  -> Airflow DAG: fraud_detection_mlops_pipeline
  -> Data Validation (Schema & Type checks)
  -> Preprocessing + Stratified Train/Test Split
  -> Model Training (Logistic Regression, Random Forest, XGBoost)
  -> Optuna Hyperparameter Tuning (Nested MLflow Trial Runs)
  -> MLflow Experiment Tracking and Model Registry (Production Stage Promotion)
  -> FastAPI Serving directly from models:/FraudDetectionModel/Production
  -> Local Prediction Logs (prediction_logs.csv)
  -> Automated Monitoring (Custom summary & Evidently AI Data Drift Report)
```

---

## 🛠️ Technology Stack

| Component | Technology | Description |
|---|---|---|
| **Language** | Python 3.11 | Core programming language |
| **Data Processing** | Pandas, NumPy | Data wrangling and transformation |
| **Machine Learning** | scikit-learn, XGBoost | Preprocessing pipeline & model candidates |
| **Hyperparameter Tuning**| Optuna | Validation PR-AUC optimization |
| **Experiment Tracking** | MLflow | Model, parameter, metric, and artifact logging |
| **Model Registry** | MLflow Model Registry | Model version control and stage-based promotion |
| **Orchestration** | Apache Airflow | Full workflow scheduler and coordinator |
| **API Deployment** | FastAPI, Uvicorn | High-performance containerized prediction endpoints |
| **Monitoring** | Evidently AI, custom summary| Feature drift analysis and performance summary |
| **Metadata Storage** | PostgreSQL | Backend store for Airflow metadata and MLflow |
| **Object Storage** | MinIO | S3-compatible local bucket for artifact storage |
| **Runtime** | Docker Compose | Complete environment containerization |
| **Testing** | pytest, compileall | Quality assurance suite |

---

## 📊 Dataset Specifications

The final verification was run against a PaySim dataset mirror: `purulalwani/Synthetic-Financial-Datasets-For-Fraud-Detection`.
The full source dataset contains 6,362,620 rows. To keep the project practical for local execution, we use a **200,000-row class-stratified sample** saved as `data/raw/paysim.csv`.

| Split | Rows | Non-Fraud | Fraud | Fraud % |
|---|---:|---:|---:|---:|
| **Raw Sample** | 200,000 | 199,742 | 258 | 0.1290% |
| **Train Set** | 160,000 | 159,794 | 206 | 0.1288% |
| **Test Set** | 40,000 | 39,948 | 52 | 0.1300% |
| **Reference Set**| 160,000 | 159,794 | 206 | 0.1288% |

- **Target Column:** `isFraud`
- **Features Used:** `step`, `type` (categorical), `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`.

---

## 🚀 Execution & Setup Instructions

### 1. Prerequisites
- Docker Desktop installed and running
- Standard PowerShell or terminal
- Clone this repository locally

### 2. Prepare the Dataset
Place your PaySim-compatible CSV at:
```text
data/raw/paysim.csv
```
*(If you need a mock dataset for quick smoke-testing, run `python scripts/generate_mock_data_for_smoke_test.py` first)*.

### 3. Launch the Stack
Start all required backend and frontend services in the background:
```powershell
docker compose up -d --build
```
Verify all containers are up and running:
```powershell
docker compose ps
```

---

## 🌐 Local Service Directory

Once the stack is running, you can access the following services:

- **FastAPI Serving Endpoint:** [http://localhost:8000](http://localhost:8000)
- **FastAPI Swagger UI Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **MLflow Tracking Server:** [http://localhost:5000](http://localhost:5000)
- **Airflow Webserver Interface:** [http://localhost:8080](http://localhost:8080) *(Username: `admin` | Password: `admin`)*
- **MinIO Object Console:** [http://localhost:9001](http://localhost:9001) *(Username: `minio` | Password: `minio123`)*

---

## ⚙️ Triggering the Airflow DAG

You can trigger the pipeline in two ways:

1. **Airflow UI:** Navigate to `http://localhost:8080`, unpause `fraud_detection_mlops_pipeline`, and click the "Trigger DAG" play button.
2. **CLI Terminal Command:** Run the command directly within the webserver container:
   ```powershell
   docker compose exec -T airflow-webserver airflow dags trigger fraud_detection_mlops_pipeline
   ```

---

## 🧪 Testing and Quality Assurance

To compile the codebase and run the unit test suite inside the active container:
```powershell
# Compile the python code
docker compose exec -T airflow-webserver python -m compileall app src monitoring dags tests

# Run unit tests via pytest
docker compose exec -T airflow-webserver pytest -v
```

---

## 📡 API Usage & Curl Examples

### 1. Health Status check
```bash
curl http://localhost:8000/health
```
**Expected Response:**
```json
{"status": "ok"}
```

### 2. Model Stage Info
```bash
curl http://localhost:8000/model-info
```
**Expected Response:**
```json
{
  "model_name": "FraudDetectionModel",
  "model_stage": "Production",
  "model_uri": "models:/FraudDetectionModel/Production",
  "loaded": "true",
  "status": "active"
}
```

### 3. Predict Endpoint (Single Transaction)
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "step": 1,
       "type": "TRANSFER",
       "amount": 181000.0,
       "oldbalanceOrg": 181000.0,
       "newbalanceOrig": 0.0,
       "oldbalanceDest": 0.0,
       "newbalanceDest": 0.0
     }'
```
**Expected Response:**
```json
{
  "prediction": 1,
  "label": "Fraud",
  "fraud_probability": 0.9993,
  "model_name": "FraudDetectionModel",
  "model_stage": "Production"
}
```

---

## 📈 Monitoring & Drift Reports

Every request sent to `/predict` is appended to `data/predictions/prediction_logs.csv` with prediction scores and timestamps.
The Evidently AI monitoring task generates an interactive data drift report comparing baseline data with serving data.

- **Data Drift Report Path:** `monitoring/reports/data_drift_report.html` (mapped locally inside your repository).

---

## 📂 Verification & Evidence Directory

The `evidence/` directory contains verified visual screenshots demonstrating submission readiness:

1. **`airflow_dag_success.png`** - Showcases the Airflow DAG (`fraud_detection_mlops_pipeline`) completing successfully.
2. **`mlflow_experiment_runs.png`** - Shows candidate models (Logistic Regression, Random Forest, XGBoost, Tuned XGBoost) with parameters and metrics logged in MLflow.
3. **`mlflow_model_registry.png`** - Displays the registered model version promoted to the **Production** stage.
4. **`fastapi_swagger.png`** - Shows the interactive OpenAPI Swagger UI at `http://localhost:8000/docs`.
5. **`prediction_response.png`** - Shows a successful prediction request and response from the `/predict` API.
6. **`evidently_drift_report.png`** - Displays the interactive data drift report output generated by Evidently AI.
