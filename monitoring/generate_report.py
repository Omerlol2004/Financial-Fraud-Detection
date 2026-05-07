from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.report import Report

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import FEATURE_COLUMNS, MONITORING_REPORTS_DIR, PREDICTION_LOG_PATH, REFERENCE_PATH, ensure_directories
from src.monitor import generate_custom_monitoring


def generate_evidently_report() -> str:
    ensure_directories()
    generate_custom_monitoring()
    output_path = MONITORING_REPORTS_DIR / "evidently_drift_report.html"
    if not REFERENCE_PATH.exists() or not PREDICTION_LOG_PATH.exists():
        output_path.write_text("<html><body><h1>Monitoring Report</h1><p>Reference data or prediction logs are not available yet.</p></body></html>", encoding="utf-8")
        return str(output_path)

    reference = pd.read_csv(REFERENCE_PATH)
    current = pd.read_csv(PREDICTION_LOG_PATH)
    common_columns = [column for column in FEATURE_COLUMNS if column in reference.columns and column in current.columns]

    try:
        report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
        report.run(reference_data=reference[common_columns], current_data=current[common_columns])
        report.save_html(str(output_path))
    except Exception as exc:  # Evidently APIs differ across versions; keep pipeline resilient.
        summary = current[common_columns].describe(include="all").to_html()
        output_path.write_text(
            f"<html><body><h1>Fallback Monitoring Report</h1><p>Evidently unavailable: {exc}</p>{summary}</body></html>",
            encoding="utf-8",
        )
    return str(output_path)


if __name__ == "__main__":
    print(generate_evidently_report())
