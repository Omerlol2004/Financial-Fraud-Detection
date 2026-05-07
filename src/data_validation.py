from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DIR, RAW_DATA_PATH, REQUIRED_COLUMNS, TARGET_COLUMN, ensure_directories


def validate_data(input_path: Path = RAW_DATA_PATH) -> dict[str, object]:
    ensure_directories()
    if not input_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {input_path}. Download PaySim and save it as data/raw/paysim.csv."
        )

    data = pd.read_csv(input_path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if data.empty:
        raise ValueError("Dataset is empty.")
    if data[TARGET_COLUMN].isna().any():
        raise ValueError(f"Target column {TARGET_COLUMN} contains null values.")
    invalid_targets = sorted(set(data[TARGET_COLUMN].dropna().unique()) - {0, 1})
    if invalid_targets:
        raise ValueError(f"Target column must be binary 0/1; found {invalid_targets}")

    summary = {
        "rows": int(len(data)),
        "columns": list(data.columns),
        "fraud_rate": float(data[TARGET_COLUMN].mean()),
        "missing_values": {column: int(count) for column, count in data.isna().sum().items()},
    }
    output_path = PROCESSED_DIR / "validation_summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(validate_data(), indent=2))
