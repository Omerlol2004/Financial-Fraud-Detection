import pandas as pd

from app.main import _predict_frame


class DummyModel:
    def predict_proba(self, frame):
        return [[0.1, 0.9] for _ in range(len(frame))]


def test_predict_frame_logs_probability(monkeypatch, tmp_path):
    import app.main as main

    log_path = tmp_path / "prediction_logs.csv"
    monkeypatch.setattr(main, "load_production_model", lambda: DummyModel())
    monkeypatch.setattr(main, "PREDICTION_LOG_PATH", log_path)

    frame = pd.DataFrame(
        [
            {
                "step": 1,
                "type": "TRANSFER",
                "amount": 100.0,
                "oldbalanceOrg": 200.0,
                "newbalanceOrig": 100.0,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 100.0,
            }
        ]
    )

    response = _predict_frame(frame)[0]

    assert response.prediction == 1
    assert response.fraud_probability == 0.9
    assert log_path.exists()
