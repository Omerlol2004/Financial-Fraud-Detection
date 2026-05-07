from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from monitoring.generate_report import generate_evidently_report
from src.data_validation import validate_data
from src.evaluate import evaluate_best_model
from src.preprocessing import preprocess_data
from src.register_model import register_best_model
from src.train import train_baseline_model, train_random_forest, train_xgboost
from src.tune import tune_best_model

with DAG(
    dag_id="fraud_detection_mlops_pipeline",
    description="End-to-end fraud detection MLflow pipeline orchestrated by Airflow.",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["fraud", "mlops", "mlflow"],
) as dag:
    validate_data_task = PythonOperator(task_id="validate_data", python_callable=validate_data)
    preprocess_data_task = PythonOperator(task_id="preprocess_data", python_callable=preprocess_data)
    train_baseline_task = PythonOperator(task_id="train_baseline_model", python_callable=train_baseline_model)
    train_random_forest_task = PythonOperator(task_id="train_random_forest", python_callable=train_random_forest)
    train_xgboost_task = PythonOperator(task_id="train_xgboost_or_lightgbm", python_callable=train_xgboost)
    tune_best_model_task = PythonOperator(task_id="tune_best_model", python_callable=tune_best_model, op_kwargs={"n_trials": 20})
    evaluate_best_model_task = PythonOperator(task_id="evaluate_best_model", python_callable=evaluate_best_model)
    register_best_model_task = PythonOperator(task_id="register_best_model", python_callable=register_best_model)
    generate_monitoring_report_task = PythonOperator(task_id="generate_monitoring_report", python_callable=generate_evidently_report)

    validate_data_task >> preprocess_data_task
    preprocess_data_task >> [train_baseline_task, train_random_forest_task, train_xgboost_task]
    [train_baseline_task, train_random_forest_task, train_xgboost_task] >> tune_best_model_task
    tune_best_model_task >> evaluate_best_model_task >> register_best_model_task >> generate_monitoring_report_task
