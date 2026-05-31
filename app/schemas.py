from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

TransactionType = Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


class Transaction(BaseModel):
    step: int = Field(..., ge=0)
    type: TransactionType
    amount: float = Field(..., ge=0)
    oldbalanceOrg: float = Field(..., ge=0)
    newbalanceOrig: float = Field(..., ge=0)
    oldbalanceDest: float = Field(..., ge=0)
    newbalanceDest: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    fraud_probability: float
    model_name: str
    model_stage: str
