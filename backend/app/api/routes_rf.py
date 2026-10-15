"""Random Forest training, inference, and evaluation endpoints."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from app.algorithms.random_forest import AMLRandomForestPipeline
from app.core.config import (
    EVALUATION_METRICS_PATH,
    MODEL_ARTIFACT_PATH,
    get_dataset_path,
)
from app.data import loader
from app.data.features import build_rf_feature_matrix
from app.schemas.common import JobResponse
from app.schemas.random_forest import (
    RFEvaluationResponse,
    RFPredictionOutput,
    RFPredictRequest,
    RFPredictResponse,
    RFTrainRequest,
)
from app.services.job_service import job_service

router = APIRouter(prefix="/random-forest", tags=["Random Forest Classifier"])


def _execute_train_rf(req_dict: dict[str, Any]) -> dict[str, Any]:
    csv_path = get_dataset_path()
    sample_size = req_dict.get("sample_size", 50_000)
    df = next(loader.iter_chunks(csv_path, chunksize=sample_size, max_rows=sample_size))

    train_ratio = req_dict.get("train_ratio", 0.70)
    val_ratio = req_dict.get("val_ratio", 0.15)
    threshold = req_dict.get("decision_threshold", 0.40)

    train_df, val_df, test_df = AMLRandomForestPipeline.chronological_split(
        df, train_ratio=train_ratio, val_ratio=val_ratio
    )

    X_train, y_train = build_rf_feature_matrix(train_df)
    X_val, y_val = build_rf_feature_matrix(val_df)
    X_test, y_test = build_rf_feature_matrix(test_df)

    pipeline = AMLRandomForestPipeline(
        n_estimators=req_dict.get("n_estimators", 100),
        max_depth=req_dict.get("max_depth", 15),
        decision_threshold=threshold,
        random_state=42,
    )
    pipeline.fit(X_train, y_train)

    val_metrics = pipeline.evaluate(X_val, y_val, split_name="Validation", threshold=threshold)
    test_metrics = pipeline.evaluate(X_test, y_test, split_name="Test (Out-of-Time)", threshold=threshold)
    top_features = pipeline.get_feature_importances(top_n=15)

    MODEL_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pipeline.save(MODEL_ARTIFACT_PATH)

    eval_result = {
        "model_available": True,
        "model_path": str(MODEL_ARTIFACT_PATH),
        "sample_size": len(df),
        "training_metadata": pipeline.training_metadata_,
        "validation_metrics": val_metrics.to_dict(),
        "test_metrics": test_metrics.to_dict(),
        "top_features": top_features,
    }

    EVALUATION_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVALUATION_METRICS_PATH.write_text(json.dumps(eval_result, indent=2), encoding="utf-8")
    return eval_result


@router.post("/train", response_model=RFEvaluationResponse | JobResponse)
def train_random_forest(req: RFTrainRequest) -> Any:
    csv_path = get_dataset_path()
    if not csv_path.is_file():
        raise HTTPException(status_code=404, detail=f"Dataset CSV not found at {csv_path}.")

    req_dict = req.model_dump()
    if req.run_in_background:
        job_id = job_service.submit_job("random_forest_train", _execute_train_rf, req_dict)
        job_info = job_service.get_job(job_id)
        return JobResponse(**job_info)

    try:
        res = _execute_train_rf(req_dict)
        return RFEvaluationResponse(**res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Random Forest training failed: {e}")


@router.get("/evaluation", response_model=RFEvaluationResponse)
def get_evaluation() -> RFEvaluationResponse:
    if not MODEL_ARTIFACT_PATH.is_file():
        return RFEvaluationResponse(
            model_available=False,
            model_path=None,
            training_metadata=None,
            validation_metrics=None,
            test_metrics=None,
            top_features=[],
            disclaimer="No trained model artifact found on disk. Please trigger POST /api/random-forest/train first.",
        )

    if EVALUATION_METRICS_PATH.is_file():
        try:
            saved_eval = json.loads(EVALUATION_METRICS_PATH.read_text(encoding="utf-8"))
            if "random_forest" in saved_eval:
                rf_data = saved_eval["random_forest"]
                return RFEvaluationResponse(
                    model_available=True,
                    model_path=rf_data.get("saved_model_path", str(MODEL_ARTIFACT_PATH)),
                    training_metadata=rf_data.get("train_metrics"),
                    validation_metrics=rf_data.get("validation_metrics"),
                    test_metrics=rf_data.get("test_metrics"),
                    top_features=rf_data.get("top_features", []),
                )
            return RFEvaluationResponse(**saved_eval)
        except Exception:
            pass

    # Model exists but evaluation JSON is missing; load model metadata
    try:
        pipeline = AMLRandomForestPipeline.load(MODEL_ARTIFACT_PATH)
        return RFEvaluationResponse(
            model_available=True,
            model_path=str(MODEL_ARTIFACT_PATH),
            training_metadata=pipeline.training_metadata_,
            validation_metrics=None,
            test_metrics=None,
            top_features=pipeline.get_feature_importances(top_n=15),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed loading model evaluation: {e}")


@router.post("/predict", response_model=RFPredictResponse)
def predict_transactions(req: RFPredictRequest) -> RFPredictResponse:
    if not MODEL_ARTIFACT_PATH.is_file():
        raise HTTPException(
            status_code=404,
            detail="Trained Random Forest model artifact not found. Please train the model first via POST /api/random-forest/train.",
        )

    try:
        pipeline = AMLRandomForestPipeline.load(MODEL_ARTIFACT_PATH)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed loading trained model artifact: {e}")

    # Build DataFrame from incoming validated items
    raw_dicts = []
    for item in req.transactions:
        row = {
            "Timestamp": item.timestamp or "2022/09/01 12:00",
            "From Bank": item.from_bank,
            "To Bank": item.to_bank,
            "Account": item.account,
            "Account.1": item.account_1 or item.account,
            "Amount Received": item.amount_received,
            "Amount Paid": item.amount_paid,
            "Receiving Currency": item.receiving_currency,
            "Payment Currency": item.payment_currency,
            "Payment Format": item.payment_format,
            # Explicit dummy target for feature builder; isolated and discarded immediately
            "Is Laundering": 0,
        }
        raw_dicts.append(row)

    input_df = pd.DataFrame(raw_dicts)
    X, _ = build_rf_feature_matrix(input_df)

    threshold = req.decision_threshold if req.decision_threshold is not None else pipeline.decision_threshold
    probs = pipeline.predict_proba(X)
    preds = (probs >= threshold).astype(int)

    results: list[RFPredictionOutput] = []
    for idx, (p, c) in enumerate(zip(probs, preds)):
        risk = "high" if p >= 0.70 else ("medium" if p >= threshold else "low")
        results.append(
            RFPredictionOutput(
                transaction_index=idx,
                probability_score=round(float(p), 4),
                predicted_class=int(c),
                is_flagged=bool(c == 1),
                risk_level=risk,
            )
        )

    flagged_count = sum(1 for r in results if r.is_flagged)
    return RFPredictResponse(
        total_transactions=len(results),
        flagged_count=flagged_count,
        threshold_used=threshold,
        predictions=results,
    )
