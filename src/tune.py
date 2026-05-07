from __future__ import annotations

import json
from typing import Any

import mlflow
import mlflow.sklearn
import optuna
from mlflow.models import infer_signature
from xgboost import XGBClassifier

from src.config import BEST_RUN_PATH, FEATURE_COLUMNS, MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI, RANDOM_STATE, TARGET_COLUMN
from src.train import calculate_metrics, load_split, log_artifacts, make_pipeline


def objective(trial: optuna.Trial) -> float:
    x_train, y_train, x_test, y_test = load_split()
    model = XGBClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 500),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    pipeline = make_pipeline(model)
    pipeline.fit(x_train, y_train)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = calculate_metrics(y_test, probabilities, predictions)
    for key, value in metrics.items():
        trial.set_user_attr(key, value)
    return float(metrics["pr_auc"])


def tune_best_model(n_trials: int = 20) -> dict[str, Any]:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    study = optuna.create_study(direction="maximize", study_name="fraud_xgboost_tuning")
    study.optimize(objective, n_trials=n_trials)

    x_train, y_train, x_test, y_test = load_split()
    best_params = study.best_trial.params
    final_model = XGBClassifier(
        **best_params,
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    pipeline = make_pipeline(final_model)

    with mlflow.start_run(run_name="tuned_xgboost") as run:
        mlflow.log_params(best_params)
        mlflow.log_param("model_name", "tuned_xgboost")
        mlflow.log_param("target", TARGET_COLUMN)
        pipeline.fit(x_train, y_train)
        probabilities = pipeline.predict_proba(x_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        metrics = calculate_metrics(y_test, probabilities, predictions)
        mlflow.log_metrics(metrics)
        log_artifacts(y_test, predictions, FEATURE_COLUMNS, "tuned_xgboost")
        signature = infer_signature(x_test.head(5), pipeline.predict_proba(x_test.head(5)))
        mlflow.sklearn.log_model(pipeline, artifact_path="model", signature=signature, input_example=x_test.head(2))
        best = {"run_id": run.info.run_id, "model_name": "tuned_xgboost", "metrics": metrics, "params": best_params}
        BEST_RUN_PATH.write_text(json.dumps(best, indent=2), encoding="utf-8")
        return best


if __name__ == "__main__":
    print(json.dumps(tune_best_model(), indent=2))
