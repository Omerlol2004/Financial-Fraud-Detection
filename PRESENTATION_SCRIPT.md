# 8-Minute Presentation Script

## 0:00-0:45 - Title and Problem

Hello, my project is an end-to-end Financial Fraud Detection MLOps Platform. The goal is to detect fraudulent financial transactions while also demonstrating the full machine learning lifecycle, not just model training.

Fraud detection is a highly imbalanced binary classification problem. Fraud transactions are rare, but missing them can be expensive. At the same time, too many false positives can create customer friction and overload review teams. Because of this, the project focuses on PR-AUC, recall, F1-score, false positives, and false negatives rather than accuracy.

## 0:45-1:30 - Dataset

The dataset follows the PaySim financial transaction schema. The full source dataset mirror contains 6,362,620 rows. For local execution on my laptop, I used a 200,000-row class-stratified sample.

The sample preserves the original fraud rate at about 0.129 percent. It contains 199,742 non-fraud transactions and 258 fraud transactions. After preprocessing, the train set has 160,000 rows and the test set has 40,000 rows, including 52 fraud cases.

The target column is `isFraud`, and the features include transaction step, type, amount, origin balances, and destination balances.

## 1:30-2:20 - Architecture

The system is designed as a local MLOps platform using Docker Compose. The main flow starts with the raw PaySim CSV, then Airflow orchestrates validation, preprocessing, model training, tuning, evaluation, registration, promotion, and monitoring.

MLflow is used for experiment tracking and the model registry. FastAPI serves the model from the MLflow Production stage. Prediction requests are logged to CSV, and Evidently generates a drift report using reference data and local prediction logs.

The main services are PostgreSQL, MLflow, Airflow webserver, Airflow scheduler, FastAPI, and MinIO.

## 2:20-3:20 - Airflow Pipeline

The main DAG is called `fraud_detection_mlops_pipeline`. The final verified run ID was `qa_validation_tuning_20260509_142908`, and it finished successfully.

The DAG tasks are data validation, preprocessing, Logistic Regression training, Random Forest training, XGBoost training, Optuna tuning, best model evaluation, model registration, model promotion, and monitoring report generation.

The important design point is that the DAG does not contain all the business logic. Instead, it calls modular Python scripts from the `src/` and `monitoring/` folders. This keeps orchestration separate from implementation.

## 3:20-4:20 - MLflow Experiment Tracking

MLflow is the main MLOps tool in this project. Each model run logs parameters, metrics, artifacts, and the trained model.

The tracked metrics include precision, recall, F1-score, ROC-AUC, PR-AUC, false positives, and false negatives. Each run also logs a confusion matrix, classification report, feature list, model signature, and model artifact.

The experiment contains runs for Logistic Regression, Random Forest, XGBoost, and tuned XGBoost. Optuna optimizes validation PR-AUC using a split from the training set, while the reported final metrics come from the held-out test set. This keeps the model comparison transparent and avoids tuning on the final test data.

## 4:20-5:10 - Model Results

The Logistic Regression model had very high recall at 0.9615, but it produced 1,883 false positives, which is too many for a practical review workflow.

Random Forest was more balanced, with precision 0.6852, recall 0.7115, F1 0.6981, and PR-AUC 0.7773.

XGBoost achieved the best PR-AUC at 0.8761 and was selected as the final model. Its precision was 0.9429, recall was 0.6346, and F1 was 0.7586.

Tuned XGBoost matched base XGBoost on precision, recall, F1, false positives, and false negatives after retraining on the full training set, but its held-out test PR-AUC was slightly lower at 0.8715. Since the selection rule is PR-AUC first, then recall, then F1, base XGBoost was selected.

## 5:10-5:50 - Model Registry

After the best model was selected, it was registered in MLflow as `FraudDetectionModel`.

The final Production model is version 11, and its status is READY. This shows the model lifecycle from experiment run to registered model version to Production promotion.

FastAPI does not load a local dummy model. It loads from the MLflow registry using `models:/FraudDetectionModel/Production`.

## 5:50-6:40 - FastAPI Deployment

FastAPI exposes four endpoints: `/health`, `/model-info`, `/predict`, and `/predict_batch`.

The final API was verified at `http://localhost:8000/docs`. One verified transaction returned a fraud prediction with probability about 0.9993, model name `FraudDetectionModel`, and model stage `Production`.

Every prediction is appended to `data/predictions/prediction_logs.csv`. This creates the link between serving and monitoring.

## 6:40-7:20 - Monitoring

Monitoring has two parts. First, a custom summary reports the number of predictions, fraud prediction percentage, average fraud probability, amount distribution, and transaction type distribution.

Second, Evidently generates a data drift HTML report. The final report was verified as real Evidently output, not fallback HTML.

I do not claim stable production drift conclusions because the local prediction log is small. The purpose here is to verify the monitoring mechanism and show how reference data and prediction logs can feed a monitoring report.

## 7:20-8:00 - Conclusion and Future Work

In conclusion, this project is a complete local MLOps platform for fraud detection. It includes Airflow orchestration, MLflow tracking and registry, FastAPI deployment, Docker Compose infrastructure, prediction logging, and Evidently monitoring.

The final selected model is XGBoost, chosen by PR-AUC, with PR-AUC 0.8761 and only 2 false positives on the test set.

Future work would include threshold optimization based on business cost, larger-sample or full-dataset training, a label feedback loop, and stronger production monitoring over larger prediction windows.
