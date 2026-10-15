"""Pydantic request and response schemas for Random Forest training and inference."""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field, model_validator


TARGET_LEAKAGE_KEYS = {"is laundering", "islaundering", "is_laundering"}


class RFTrainRequest(BaseModel):
    sample_size: int = Field(50_000, ge=1_000, le=200_000, description="Chronological sample size to train on")
    n_estimators: int = Field(100, ge=10, le=300, description="Number of trees in forest")
    max_depth: int | None = Field(15, ge=2, le=30, description="Maximum tree depth")
    decision_threshold: float = Field(0.40, gt=0.0, lt=1.0, description="Decision threshold for class 1 (Laundering)")
    train_ratio: float = Field(0.70, ge=0.50, le=0.85, description="Proportion of chronological sample for training")
    val_ratio: float = Field(0.15, ge=0.05, le=0.25, description="Proportion of chronological sample for validation")
    run_in_background: bool = Field(False, description="If True, dispatches to background job runner")


class RFPredictItem(BaseModel):
    timestamp: str | None = Field("2022/09/01 12:00", alias="Timestamp")
    from_bank: int = Field(..., alias="From Bank")
    to_bank: int = Field(..., alias="To Bank")
    account: str = Field(..., alias="Account")
    account_1: str | None = Field(None, alias="Account.1")
    amount_received: float = Field(..., alias="Amount Received")
    amount_paid: float = Field(..., alias="Amount Paid")
    receiving_currency: str = Field(..., alias="Receiving Currency")
    payment_currency: str = Field(..., alias="Payment Currency")
    payment_format: str = Field(..., alias="Payment Format")

    class Config:
        populate_by_name = True
        extra = "allow"

    @model_validator(mode="before")
    @classmethod
    def check_no_target_leakage(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for k in data.keys():
                if str(k).strip().lower() in TARGET_LEAKAGE_KEYS:
                    raise ValueError(
                        f"Target leakage prevented: '{k}' must NOT be provided in inference features."
                    )
        return data


class RFPredictRequest(BaseModel):
    transactions: list[RFPredictItem] = Field(..., min_length=1, max_length=1_000)
    decision_threshold: float | None = Field(None, gt=0.0, lt=1.0, description="Override decision threshold")


class RFPredictionOutput(BaseModel):
    transaction_index: int
    probability_score: float
    predicted_class: int
    is_flagged: bool
    risk_level: str  # low, medium, high


class RFPredictResponse(BaseModel):
    total_transactions: int
    flagged_count: int
    threshold_used: float
    predictions: list[RFPredictionOutput]
    disclaimer: str = (
        "Model prediction scores represent tree ensemble voting proportions and ranking metrics, "
        "not calibrated real-world probabilities of money laundering or proof of guilt."
    )


class RFEvaluationResponse(BaseModel):
    model_available: bool
    model_path: str | None = None
    training_metadata: dict[str, Any] | None = None
    validation_metrics: dict[str, Any] | None = None
    test_metrics: dict[str, Any] | None = None
    top_features: list[dict[str, Any]] = Field(default_factory=list)
    disclaimer: str = (
        "Evaluation metrics are measured on an out-of-time chronological test slice. "
        "Accuracy is accompanied by Precision, Recall, F1, ROC-AUC, and PR-AUC due to class imbalance."
    )
