import pandas as pd

from src.config import TARGET_COLUMN
from src.preprocessing import clean_transactions


def test_clean_transactions_standardizes_types_and_numeric_values():
    raw = pd.DataFrame(
        [
            {
                "step": "1",
                "type": " transfer ",
                "amount": "100.5",
                "oldbalanceOrg": "200",
                "newbalanceOrig": "99.5",
                "oldbalanceDest": "0",
                "newbalanceDest": "100.5",
                "isFraud": "1",
            }
        ]
    )

    cleaned = clean_transactions(raw)

    assert cleaned.iloc[0]["type"] == "TRANSFER"
    assert cleaned.iloc[0]["amount"] == 100.5
    assert int(cleaned.iloc[0][TARGET_COLUMN]) == 1
