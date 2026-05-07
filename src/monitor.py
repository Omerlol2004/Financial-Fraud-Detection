from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import MONITORING_REPORTS_DIR, PREDICTION_LOG_PATH, ensure_directories


def generate_custom_monitoring(log_path: Path = PREDICTION_LOG_PATH) -> dict[str, object]:
    ensure_directories()
    if not log_path.exists():
        summary = {"number_of_predictions": 0, "message": "No prediction log exists yet."}
    else:
        logs = pd.read_csv(log_path)
        summary = {
            "number_of_predictions": int(len(logs)),
            "fraud_prediction_percentage": float(logs["prediction"].mean() * 100) if len(logs) else 0.0,
            "average_fraud_probability": float(logs["fraud_probability"].mean()) if len(logs) else 0.0,
            "transaction_amount_distribution": logs["amount"].describe().to_dict() if "amount" in logs else {},
            "transaction_type_distribution": logs["type"].value_counts().to_dict() if "type" in logs else {},
        }
    output_path = MONITORING_REPORTS_DIR / "custom_monitoring_summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(generate_custom_monitoring(), indent=2))
