# Final Report Outline

Use `FINAL_REPORT.md` and `evidence/FINAL_EVIDENCE_REPORT.md` as the source of truth.

## 1. Introduction

- End-to-end local MLOps platform for financial fraud detection.
- Goal is to demonstrate the full lifecycle, not only model training.

## 2. Problem Statement

- Binary classification: fraud vs. non-fraud.
- Highly imbalanced problem, so accuracy is not the main metric.
- Main metrics: PR-AUC, recall, F1, false positives, false negatives.

## 3. Dataset Description

- PaySim transaction schema.
- Full source mirror: 6,362,620 rows.
- Final local run: 200,000-row stratified sample.
- Fraud rate in sample: 0.129%.

## 4. System Architecture

- Airflow orchestrates the workflow.
- MLflow tracks experiments and manages registry lifecycle.
- FastAPI serves the Production model.
- Evidently and custom summaries monitor local prediction logs.

## 5. Results

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.0259 | 0.9615 | 0.0504 | 0.9937 | 0.6253 | 1,883 | 2 |
| Random Forest | 0.6852 | 0.7115 | 0.6981 | 0.9957 | 0.7773 | 17 | 15 |
| XGBoost | 0.9429 | 0.6346 | 0.7586 | 0.9982 | 0.8761 | 2 | 19 |
| Tuned XGBoost | 0.9429 | 0.6346 | 0.7586 | 0.9989 | 0.8715 | 2 | 19 |

- Final selected model: XGBoost.
- Selection criterion: held-out test PR-AUC first, then recall, then F1.
- Optuna tuning optimized validation PR-AUC and did not use the final test set for trial selection.
- Production model: `FraudDetectionModel` version 11, READY.

## 6. Verification Evidence

- Final Airflow run: `qa_validation_tuning_20260509_142908`.
- Airflow status: success.
- FastAPI verified at `http://localhost:8000/docs`.
- Evidently report exists and is real, not fallback HTML.
- Tests pass: `5 passed`.

## 7. Limitations and Future Work

- Used a 200,000-row stratified sample, not the full source dataset.
- Monitoring mechanics were verified with local prediction logs; stable production drift conclusions are not claimed.
- Future work: threshold optimization, larger training sample, label feedback loop, richer monitoring.
