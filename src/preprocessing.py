from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    BEST_RUN_PATH,
    ensure_directories,
    FEATURE_COLUMNS,
    METRICS_PATH,
    RANDOM_STATE,
    RAW_DATA_PATH,
    REFERENCE_PATH,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
    TEST_PATH,
    TEST_SIZE,
    TRAIN_PATH,
)


def load_raw_data(path=RAW_DATA_PATH) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data[REQUIRED_COLUMNS].copy()


def clean_transactions(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = data.copy()
    cleaned["type"] = cleaned["type"].astype(str).str.upper().str.strip()
    for column in FEATURE_COLUMNS:
        if column != "type":
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    cleaned[TARGET_COLUMN] = pd.to_numeric(cleaned[TARGET_COLUMN], errors="coerce").astype("Int64")
    cleaned = cleaned.dropna(subset=REQUIRED_COLUMNS)
    return cleaned


def preprocess_data() -> dict[str, str]:
    ensure_directories()
    data = clean_transactions(load_raw_data())
    train, test = train_test_split(
        data,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=data[TARGET_COLUMN],
    )
    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)
    train.to_csv(REFERENCE_PATH, index=False)
    for generated_file in [METRICS_PATH, BEST_RUN_PATH]:
        if generated_file.exists():
            generated_file.unlink()
    return {"train_path": str(TRAIN_PATH), "test_path": str(TEST_PATH), "reference_path": str(REFERENCE_PATH)}


if __name__ == "__main__":
    print(preprocess_data())
