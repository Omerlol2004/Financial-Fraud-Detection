from __future__ import annotations

import json

import mlflow
from mlflow.tracking import MlflowClient

from src.config import MLFLOW_TRACKING_URI, MODEL_NAME


def promote_latest_staging_to_production() -> dict[str, str]:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    staging_versions = client.get_latest_versions(MODEL_NAME, stages=["Staging"])
    if not staging_versions:
        raise ValueError(f"No Staging versions found for {MODEL_NAME}.")
    version = staging_versions[-1].version
    client.transition_model_version_stage(MODEL_NAME, version, "Production", archive_existing_versions=True)
    return {"model_name": MODEL_NAME, "version": str(version), "stage": "Production"}


if __name__ == "__main__":
    print(json.dumps(promote_latest_staging_to_production(), indent=2))
