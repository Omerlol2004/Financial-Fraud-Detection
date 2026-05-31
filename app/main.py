from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from fastapi import FastAPI

from app.model_loader import load_production_model
from app.schemas import PredictionResponse, Transaction
from src.config import MODEL_NAME, MODEL_STAGE, PREDICTION_LOG_PATH, ensure_directories

app = FastAPI(title="Financial Fraud Detection API", version="1.0.0")


def _predict_frame(frame: pd.DataFrame) -> list[PredictionResponse]:
    model = load_production_model()
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(frame))[:, 1]
    else:
        probabilities = np.asarray(model.predict(frame))
    responses: list[PredictionResponse] = []
    log_rows = []
    for record, probability in zip(frame.to_dict(orient="records"), probabilities):
        fraud_probability = float(probability)
        prediction = int(fraud_probability >= 0.5)
        response = PredictionResponse(
            prediction=prediction,
            label="Fraud" if prediction else "Not Fraud",
            fraud_probability=fraud_probability,
            model_name=MODEL_NAME,
            model_stage=MODEL_STAGE,
        )
        responses.append(response)
        log_rows.append({**record, **response.model_dump(), "timestamp": datetime.now(timezone.utc).isoformat()})
    _append_prediction_logs(log_rows)
    return responses


def _append_prediction_logs(rows: list[dict]) -> None:
    ensure_directories()
    logs = pd.DataFrame(rows)
    header = not PREDICTION_LOG_PATH.exists()
    logs.to_csv(PREDICTION_LOG_PATH, mode="a", index=False, header=header)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model-info")
def model_info() -> dict[str, str]:
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    try:
        _ = load_production_model()
        loaded = "true"
        status = "active"
    except Exception as e:
        loaded = "false"
        status = f"inactive (error: {str(e)})"
    return {
        "model_name": MODEL_NAME,
        "model_stage": MODEL_STAGE,
        "model_uri": model_uri,
        "loaded": loaded,
        "status": status,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction) -> PredictionResponse:
    frame = pd.DataFrame([transaction.model_dump()])
    return _predict_frame(frame)[0]


@app.post("/predict_batch", response_model=list[PredictionResponse])
def predict_batch(transactions: list[Transaction]) -> list[PredictionResponse]:
    frame = pd.DataFrame([transaction.model_dump() for transaction in transactions])
    return _predict_frame(frame)
