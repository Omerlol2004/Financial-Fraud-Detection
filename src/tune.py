from __future__ import annotations

import json
from typing import Any

import mlflow
import mlflow.sklearn
import optuna
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.config import FEATURE_COLUMNS, MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI, RANDOM_STATE, TARGET_COLUMN
from src.train import calculate_metrics, load_split, log_artifacts, make_pipeline, record_model_result

TUNING_VALIDATION_SIZE = 0.2


def create_tuning_validation_split(x_train, y_train):
    return train_test_split(
        x_train,
        y_train,
        test_size=TUNING_VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )


def build_trial_model(trial: optuna.Trial) -> XGBClassifier:
    return XGBClassifier(
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


def objective(trial: optuna.Trial, x_train_inner, y_train_inner, x_validation, y_validation) -> float:
    with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
        model = build_trial_model(trial)
        pipeline = make_pipeline(model)
        pipeline.fit(x_train_inner, y_train_inner)
        probabilities = pipeline.predict_proba(x_validation)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        metrics = calculate_metrics(y_validation, probabilities, predictions)
        
        mlflow.log_param("trial_number", trial.number)
        mlflow.log_params(trial.params)
        mlflow.log_metrics(metrics)
        
        for key, value in metrics.items():
            trial.set_user_attr(f"validation_{key}", value)
        return float(metrics["pr_auc"])


def make_final_model(best_params: dict[str, Any]) -> XGBClassifier:
    model = XGBClassifier(
        **best_params,
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return model


def tune_best_model(n_trials: int = 20) -> dict[str, Any]:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    x_train, y_train, x_test, y_test = load_split()
    x_train_inner, x_validation, y_train_inner, y_validation = create_tuning_validation_split(x_train, y_train)

    study = optuna.create_study(
        direction="maximize",
        study_name="fraud_xgboost_tuning",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    with mlflow.start_run(run_name="optuna_xgboost_tuning") as parent_run:
        study.optimize(
            lambda trial: objective(trial, x_train_inner, y_train_inner, x_validation, y_validation),
            n_trials=n_trials,
        )

    best_params = study.best_trial.params
    best_validation_metrics = {
        key.removeprefix("validation_"): value
        for key, value in study.best_trial.user_attrs.items()
        if key.startswith("validation_")
    }
    pipeline = make_pipeline(make_final_model(best_params))

    with mlflow.start_run(run_name="tuned_xgboost") as run:
        mlflow.log_params(best_params)
        mlflow.log_param("best_trial_number", study.best_trial.number)
        mlflow.log_param("model_name", "tuned_xgboost")
        mlflow.log_param("target", TARGET_COLUMN)
        mlflow.log_param("tuning_objective", "validation_pr_auc")
        mlflow.log_param("tuning_validation_size", TUNING_VALIDATION_SIZE)
        mlflow.log_param("final_evaluation_split", "held_out_test")
        mlflow.log_metrics({f"best_validation_{key}": value for key, value in best_validation_metrics.items()})
        pipeline.fit(x_train, y_train)
        probabilities = pipeline.predict_proba(x_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        metrics = calculate_metrics(y_test, probabilities, predictions)
        mlflow.log_metrics(metrics)
        log_artifacts(y_test, predictions, FEATURE_COLUMNS, "tuned_xgboost")
        signature = infer_signature(x_test.head(5), pipeline.predict_proba(x_test.head(5)))
        mlflow.sklearn.log_model(pipeline, artifact_path="model", signature=signature, input_example=x_test.head(2))
        tuned = {
            "run_id": run.info.run_id,
            "model_name": "tuned_xgboost",
            "metrics": metrics,
            "params": best_params,
            "validation_metrics": best_validation_metrics,
        }
        selected = record_model_result(tuned)
        return {"tuned": tuned, "selected_best": selected}


if __name__ == "__main__":
    print(json.dumps(tune_best_model(), indent=2))
