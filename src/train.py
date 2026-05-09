from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.config import (
    BEST_RUN_PATH,
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    METRICS_PATH,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    NUMERIC_COLUMNS,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_PATH,
    TRAIN_PATH,
    ensure_directories,
)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_COLUMNS),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
        ]
    )


def make_pipeline(model: Any) -> Pipeline:
    return Pipeline([("preprocessor", build_preprocessor()), ("model", model)])


def get_model_candidates() -> dict[str, Any]:
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="aucpr",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def load_split() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return train[FEATURE_COLUMNS], train[TARGET_COLUMN], test[FEATURE_COLUMNS], test[TARGET_COLUMN]


def calculate_metrics(y_true: pd.Series, probabilities: np.ndarray, predictions: np.ndarray) -> dict[str, float | int]:
    return {
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "false_positives": int(((predictions == 1) & (y_true.to_numpy() == 0)).sum()),
        "false_negatives": int(((predictions == 0) & (y_true.to_numpy() == 1)).sum()),
    }


def log_artifacts(y_true: pd.Series, predictions: np.ndarray, feature_columns: list[str], prefix: str) -> None:
    artifact_dir = Path("artifacts") / prefix
    artifact_dir.mkdir(parents=True, exist_ok=True)

    matrix = confusion_matrix(y_true, predictions)
    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Fraud", "Fraud"], yticklabels=["Not Fraud", "Fraud"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    confusion_path = artifact_dir / "confusion_matrix.png"
    plt.savefig(confusion_path)
    plt.close()

    report_path = artifact_dir / "classification_report.json"
    report_path.write_text(json.dumps(classification_report(y_true, predictions, output_dict=True), indent=2), encoding="utf-8")
    feature_path = artifact_dir / "feature_list.json"
    feature_path.write_text(json.dumps(feature_columns, indent=2), encoding="utf-8")

    mlflow.log_artifact(str(confusion_path), artifact_path="evaluation")
    mlflow.log_artifact(str(report_path), artifact_path="evaluation")
    mlflow.log_artifact(str(feature_path), artifact_path="metadata")


def train_named_model(model_name: str, model: Any) -> dict[str, Any]:
    ensure_directories()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    x_train, y_train, x_test, y_test = load_split()
    pipeline = make_pipeline(model)

    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("target", TARGET_COLUMN)
        mlflow.log_param("features", ",".join(FEATURE_COLUMNS))
        pipeline.fit(x_train, y_train)
        probabilities = pipeline.predict_proba(x_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        metrics = calculate_metrics(y_test, probabilities, predictions)
        mlflow.log_metrics(metrics)
        log_artifacts(y_test, predictions, FEATURE_COLUMNS, model_name)
        signature = infer_signature(x_test.head(5), pipeline.predict_proba(x_test.head(5)))
        mlflow.sklearn.log_model(pipeline, artifact_path="model", signature=signature, input_example=x_test.head(2))
        result = {"run_id": run.info.run_id, "model_name": model_name, "metrics": metrics}
        record_model_result(result)
        return result


def record_model_result(result: dict[str, Any]) -> dict[str, Any]:
    ensure_directories()
    results = []
    if METRICS_PATH.exists():
        results = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    if not any(existing["run_id"] == result["run_id"] for existing in results):
        results.append(result)
    best = max(results, key=lambda item: (item["metrics"]["pr_auc"], item["metrics"]["recall"], item["metrics"]["f1"]))
    METRICS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    BEST_RUN_PATH.write_text(json.dumps(best, indent=2), encoding="utf-8")
    return best


def train_baseline_model() -> dict[str, Any]:
    return train_named_model("logistic_regression", get_model_candidates()["logistic_regression"])


def train_random_forest() -> dict[str, Any]:
    return train_named_model("random_forest", get_model_candidates()["random_forest"])


def train_xgboost() -> dict[str, Any]:
    return train_named_model("xgboost", get_model_candidates()["xgboost"])


def train_all_models() -> dict[str, Any]:
    results = [train_baseline_model(), train_random_forest(), train_xgboost()]
    best = max(results, key=lambda result: (result["metrics"]["pr_auc"], result["metrics"]["recall"], result["metrics"]["f1"]))
    METRICS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    BEST_RUN_PATH.write_text(json.dumps(best, indent=2), encoding="utf-8")
    return {"results": results, "best": best}


if __name__ == "__main__":
    print(json.dumps(train_all_models(), indent=2))
