# Resume Bullets

## Short Resume Bullet

Built an end-to-end financial fraud detection MLOps platform using Airflow, MLflow, FastAPI, Docker, PostgreSQL, and Evidently AI, with model training, registry promotion, Production serving, prediction logging, and monitoring on a 200,000-row stratified PaySim sample.

## Technical Resume Bullet

Implemented a local fraud detection MLOps system with Apache Airflow DAG orchestration, MLflow experiment tracking and Model Registry, Optuna validation-based tuning, XGBoost model selection by held-out test PR-AUC, FastAPI Production serving from `models:/FraudDetectionModel/Production`, Docker Compose infrastructure, PostgreSQL metadata stores, prediction logging, and Evidently data drift reporting. Verified final Airflow run `qa_validation_tuning_20260509_142908`, Production model version 11, and XGBoost PR-AUC of 0.8761 on a 200,000-row stratified PaySim sample.

## LinkedIn Project Description

I built an end-to-end Financial Fraud Detection MLOps Platform to demonstrate the full machine learning lifecycle beyond notebook-based modeling.

The system uses a 200,000-row stratified PaySim sample for local execution and includes data validation, preprocessing, model comparison, Optuna tuning, MLflow experiment tracking, MLflow Model Registry promotion, FastAPI serving, prediction logging, Apache Airflow orchestration, Docker Compose infrastructure, PostgreSQL metadata storage, and Evidently monitoring.

The final selected model was XGBoost, chosen by held-out test PR-AUC first, then recall and F1. It achieved PR-AUC 0.8761, precision 0.9429, recall 0.6346, and F1 0.7586. Optuna tuning optimized validation PR-AUC without using the final test set. The final Airflow DAG run completed successfully, and `FraudDetectionModel` version 11 was promoted to Production and served through FastAPI at `models:/FraudDetectionModel/Production`.

This project focuses on practical MLOps concerns: reproducible pipelines, experiment tracking, model versioning, deployment, prediction logs, and monitoring mechanics.
