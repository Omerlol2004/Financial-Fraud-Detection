# Final Report: End-to-End Financial Fraud Detection MLOps Platform

## 1. Introduction

Financial fraud detection is a high-impact machine learning problem because fraudulent transactions can cause direct monetary loss, customer harm, and operational risk. However, building only a classification model is not enough for a realistic fraud detection project. A useful production-oriented system must support repeatable training, experiment tracking, model versioning, deployment, prediction logging, monitoring, and orchestration.

This project implements an end-to-end local MLOps platform for financial fraud detection using the PaySim transaction schema. The platform integrates Apache Airflow, MLflow, FastAPI, Docker Compose, PostgreSQL, XGBoost, Optuna, and Evidently AI. The final verified pipeline ran successfully on a 200,000-row stratified PaySim sample and promoted `FraudDetectionModel` version 11 to Production.

## 2. Problem Statement

The objective is to predict whether a financial transaction is fraudulent. This is a binary classification task where the target variable is `isFraud`.

Fraud detection is challenging because the positive class is rare. In this project, the final dataset sample has a fraud rate of approximately 0.129%. As a result, accuracy is not a reliable primary metric. A model can achieve high accuracy by predicting nearly every transaction as non-fraud while failing to detect fraud. Therefore, this project evaluates models using precision, recall, F1-score, ROC-AUC, PR-AUC, false positives, and false negatives, with emphasis on PR-AUC, recall, and F1-score.

## 3. Dataset Description

The dataset follows the PaySim financial fraud detection schema. The full source dataset mirror contains 6,362,620 rows. For local execution, the platform used a 200,000-row class-stratified sample saved as `data/raw/paysim.csv`.

The final sample contains:

| Split | Rows | Non-Fraud | Fraud | Fraud % |
|---|---:|---:|---:|---:|
| Raw sample | 200,000 | 199,742 | 258 | 0.1290 |
| Train | 160,000 | 159,794 | 206 | 0.1288 |
| Test | 40,000 | 39,948 | 52 | 0.1300 |
| Reference | 160,000 | 159,794 | 206 | 0.1288 |

The target column is `isFraud`. The feature columns are `step`, `type`, `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, and `newbalanceDest`.

## 4. System Architecture

The platform uses a modular architecture:

```text
PaySim sample
  -> Airflow DAG
  -> data validation
  -> preprocessing and stratified train/test split
  -> model training and comparison
  -> Optuna tuning
  -> MLflow experiment tracking
  -> MLflow Model Registry
  -> Production model promotion
  -> FastAPI model serving
  -> prediction logging
  -> Evidently and custom monitoring reports
```

Airflow orchestrates the workflow. MLflow tracks experiments and manages the model registry. FastAPI serves the Production model. Evidently generates monitoring reports from reference data and local prediction logs. Docker Compose runs the system locally with PostgreSQL, MLflow, Airflow, FastAPI, and MinIO.

## 5. Tools and Technologies

| Component | Tool |
|---|---|
| Programming language | Python |
| Data processing | Pandas, NumPy |
| Machine learning | scikit-learn, XGBoost |
| Hyperparameter tuning | Optuna |
| Experiment tracking | MLflow |
| Model registry | MLflow Model Registry |
| Workflow orchestration | Apache Airflow |
| API deployment | FastAPI |
| Monitoring | Evidently AI and custom summaries |
| Metadata database | PostgreSQL |
| Local runtime | Docker Compose |

## 6. Data Preprocessing

The preprocessing stage reads `data/raw/paysim.csv`, validates the expected schema, normalizes the categorical transaction type, coerces numeric columns, drops invalid rows, and performs a stratified train/test split.

Stratification is important because the fraud class is rare. Without stratification, a small or unlucky test set could contain too few fraud examples, making recall and PR-AUC unstable. The training split is also saved as reference data for monitoring.

To avoid data leakage, the held-out test set is not used for hyperparameter selection. Optuna tuning creates a validation split from the training data, optimizes PR-AUC on that validation split, then retrains the tuned model on the full training set and evaluates it once on the held-out test set.

Generated files:

- `data/processed/train.csv`
- `data/processed/test.csv`
- `data/reference/reference_data.csv`

## 7. Model Training

The project trains and compares four model variants:

1. Logistic Regression baseline
2. Random Forest
3. XGBoost
4. Optuna-tuned XGBoost

The baseline model provides a simple reference point. Random Forest gives a stronger nonlinear tree-based comparison. XGBoost is the primary advanced model because gradient boosted trees are often strong for structured tabular data.

For each model, the system logs precision, recall, F1-score, ROC-AUC, PR-AUC, false positives, and false negatives. The final selection rule prioritizes PR-AUC first, then recall, then F1.

## 8. Experiment Tracking with MLflow

MLflow is the central experiment tracking and registry tool in this platform. For each training run, MLflow logs:

- model parameters
- evaluation metrics
- confusion matrix artifact
- classification report artifact
- feature list artifact
- model artifact
- model signature and input example

The final experiment name is `fraud_detection_experiments`. The final verified MLflow runs include `logistic_regression`, `random_forest`, `xgboost`, and `tuned_xgboost`.

## 9. Hyperparameter Tuning with Optuna

Optuna is used to tune XGBoost hyperparameters. The tuning objective maximizes validation PR-AUC, matching the project goal of prioritizing performance under class imbalance while keeping the final test split untouched.

During tuning, the original training set is split into `train_inner` and validation subsets. Each Optuna trial trains on `train_inner` and is scored only on the validation subset. After tuning, the best hyperparameters are used to train a final tuned model on the full training set, and that model is evaluated once on the held-out test set.

In the final verified run, tuned XGBoost was comparable to base XGBoost on thresholded test metrics, with the same precision, recall, F1-score, false positives, and false negatives. However, its held-out test PR-AUC was slightly lower than base XGBoost. Therefore, the selection logic correctly retained base XGBoost as the final best model.

## 10. Model Registry and Promotion

The best model is registered in MLflow Model Registry as:

```text
FraudDetectionModel
```

The final verified Production model is:

```text
model: FraudDetectionModel
version: 11
stage: Production
status: READY
```

This registry lifecycle demonstrates model versioning and controlled promotion. FastAPI loads the Production model from the registry instead of using a hardcoded local model file.

## 11. Deployment with FastAPI

FastAPI provides local model serving with the following endpoints:

- `GET /health`
- `GET /model-info`
- `POST /predict`
- `POST /predict_batch`

The API loads:

```text
models:/FraudDetectionModel/Production
```

A verified prediction request returned:

```json
{
  "prediction": 1,
  "label": "Fraud",
  "fraud_probability": 0.9992955923080444,
  "model_name": "FraudDetectionModel",
  "model_stage": "Production"
}
```

Every prediction is appended to `data/predictions/prediction_logs.csv`.

## 12. Pipeline Orchestration with Airflow

The main Airflow DAG is:

```text
fraud_detection_mlops_pipeline
```

The final verified DAG run was:

```text
run_id: qa_validation_tuning_20260509_142908
status: success
```

The DAG completed all major lifecycle tasks:

- data validation
- preprocessing
- baseline training
- Random Forest training
- XGBoost training
- Optuna tuning
- best model evaluation
- model registration
- Production promotion
- monitoring report generation

The DAG uses Python modules from `src/` and `monitoring/` instead of placing all logic directly inside the DAG file.

## 13. Monitoring with Evidently AI

Monitoring has two levels:

1. A custom JSON summary with prediction count, fraud prediction percentage, average fraud probability, transaction amount distribution, and transaction type distribution.
2. An Evidently data drift report comparing reference training data with current prediction logs.

The final monitoring outputs are:

- `monitoring/reports/evidently_drift_report.html`
- `monitoring/reports/custom_monitoring_summary.json`

The final Evidently output was verified as a real report, not fallback HTML. Stable production drift conclusions are not claimed because the local prediction log is small. The monitoring layer demonstrates that the system can collect prediction data and generate drift reports.

## 14. Results and Discussion

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.0259 | 0.9615 | 0.0504 | 0.9937 | 0.6253 | 1,883 | 2 |
| Random Forest | 0.6852 | 0.7115 | 0.6981 | 0.9957 | 0.7773 | 17 | 15 |
| XGBoost | 0.9429 | 0.6346 | 0.7586 | 0.9982 | 0.8761 | 2 | 19 |
| Tuned XGBoost | 0.9429 | 0.6346 | 0.7586 | 0.9989 | 0.8715 | 2 | 19 |

Logistic Regression achieved very high recall, detecting most fraud cases, but it produced 1,883 false positives. This would create a large operational review burden in a real fraud workflow.

Random Forest was more balanced. It reduced false positives dramatically while maintaining better F1-score than Logistic Regression. However, its PR-AUC was lower than XGBoost.

Base XGBoost achieved the best PR-AUC and was selected as the final model. It produced only 2 false positives and achieved the strongest precision/PR-AUC tradeoff among the evaluated models.

Tuned XGBoost was optimized on validation PR-AUC and then evaluated on the held-out test set. It matched base XGBoost on precision, recall, F1-score, false positives, and false negatives, but its held-out test PR-AUC was slightly lower. Because the project selection rule prioritizes PR-AUC first, then recall, then F1, base XGBoost remained the final selected model.

Accuracy is intentionally not emphasized. Since the dataset is highly imbalanced, a high accuracy score could hide poor fraud detection performance. PR-AUC, recall, F1-score, false positives, and false negatives provide a more meaningful view of model behavior.

### Threshold Optimization Discussion

The current API uses a classification threshold of `0.5`. This is a simple default threshold, but a real fraud detection system should tune the decision threshold based on business cost.

False negatives can allow fraudulent transactions to pass through, creating direct financial risk. False positives can block legitimate customers or send too many transactions to manual review. The optimal threshold depends on the relative cost of missed fraud, investigation workload, customer friction, and risk tolerance.

Future work should evaluate precision-recall tradeoffs across thresholds and select a threshold that matches the business objective rather than relying on the default `0.5`.

## 15. Limitations

- The final run used a 200,000-row stratified sample, not the full 6.36M-row dataset.
- The test split contained 52 fraud rows, so metrics are meaningful for local verification but still sample-sensitive.
- The model uses a fixed `0.5` threshold.
- Monitoring mechanics were verified with local prediction logs, but stable production drift conclusions are not claimed.
- Prediction or target drift would require a larger current dataset and, ideally, ground-truth labels.
- The platform is local and Docker-based; it does not include cloud deployment, Kubernetes, Prometheus, or Grafana.

## 16. Future Work

Future improvements include:

- threshold optimization based on fraud review cost
- full-dataset or larger-sample training on stronger hardware
- scheduled retraining based on drift or performance triggers
- richer monitoring with larger prediction windows
- label feedback loop for served predictions
- model explainability reports using SHAP or similar techniques
- CI checks for tests, formatting, and Docker Compose validation

## 17. Conclusion

This project implements a complete local MLOps platform for fraud detection. The final verified run used a real 200,000-row stratified PaySim sample, trained and compared multiple models, selected XGBoost by held-out test PR-AUC, registered `FraudDetectionModel` version 11 in Production, served predictions through FastAPI, orchestrated the full lifecycle with Airflow, and generated real monitoring outputs with Evidently.

The result is not just a machine learning model. It is a reproducible portfolio system that demonstrates the practical components required to move a fraud detection model through the MLOps lifecycle.
