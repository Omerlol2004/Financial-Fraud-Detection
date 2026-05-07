from __future__ import annotations

import json

import mlflow
from mlflow.tracking import MlflowClient

from src.config import BEST_RUN_PATH, MLFLOW_TRACKING_URI, MODEL_NAME


def register_best_model() -> dict[str, str]:
    if not BEST_RUN_PATH.exists():
        raise FileNotFoundError("Best run metadata not found. Run evaluate_best_model first.")
    best = json.loads(BEST_RUN_PATH.read_text(encoding="utf-8"))
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"runs:/{best['run_id']}/model"
    result = mlflow.register_model(model_uri, MODEL_NAME)
    client = MlflowClient()
    client.transition_model_version_stage(MODEL_NAME, result.version, "Staging", archive_existing_versions=False)
    return {"model_name": MODEL_NAME, "version": str(result.version), "stage": "Staging", "model_uri": model_uri}


if __name__ == "__main__":
    print(json.dumps(register_best_model(), indent=2))
