from __future__ import annotations

import os
from functools import lru_cache

import mlflow.sklearn

from src.config import MLFLOW_TRACKING_URI, MODEL_NAME, MODEL_STAGE


@lru_cache(maxsize=1)
def load_production_model():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI))
    model_name = os.getenv("MODEL_NAME", MODEL_NAME)
    model_stage = os.getenv("MODEL_STAGE", MODEL_STAGE)
    return mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/{model_stage}")
