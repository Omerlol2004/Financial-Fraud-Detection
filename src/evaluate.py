from __future__ import annotations

import json

from src.config import BEST_RUN_PATH, METRICS_PATH


def evaluate_best_model() -> dict:
    if BEST_RUN_PATH.exists():
        return json.loads(BEST_RUN_PATH.read_text(encoding="utf-8"))
    if not METRICS_PATH.exists():
        raise FileNotFoundError("No model metrics found. Run training or tuning first.")
    results = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    best = max(results, key=lambda result: (result["metrics"]["pr_auc"], result["metrics"]["recall"], result["metrics"]["f1"]))
    BEST_RUN_PATH.write_text(json.dumps(best, indent=2), encoding="utf-8")
    return best


if __name__ == "__main__":
    print(json.dumps(evaluate_best_model(), indent=2))
