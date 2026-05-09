from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = Path(os.getenv("RAW_DATA_PATH", DATA_DIR / "raw" / "paysim.csv"))
PROCESSED_DIR = DATA_DIR / "processed"
REFERENCE_DIR = DATA_DIR / "reference"
PREDICTIONS_DIR = DATA_DIR / "predictions"
MONITORING_REPORTS_DIR = PROJECT_ROOT / "monitoring" / "reports"

TARGET_COLUMN = "isFraud"
REQUIRED_COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    TARGET_COLUMN,
]
FEATURE_COLUMNS = [column for column in REQUIRED_COLUMNS if column != TARGET_COLUMN]
CATEGORICAL_COLUMNS = ["type"]
NUMERIC_COLUMNS = [column for column in FEATURE_COLUMNS if column not in CATEGORICAL_COLUMNS]

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"file://{PROJECT_ROOT / 'mlruns'}")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "fraud_detection_experiments")
MODEL_NAME = os.getenv("MODEL_NAME", "FraudDetectionModel")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))

TRAIN_PATH = PROCESSED_DIR / "train.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"
REFERENCE_PATH = REFERENCE_DIR / "reference_data.csv"
METRICS_PATH = PROCESSED_DIR / "model_metrics.json"
BEST_RUN_PATH = PROCESSED_DIR / "best_run.json"
PREDICTION_LOG_PATH = PREDICTIONS_DIR / "prediction_logs.csv"


def ensure_directories() -> None:
    for directory in [PROCESSED_DIR, REFERENCE_DIR, PREDICTIONS_DIR, MONITORING_REPORTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
