"""Algorithms registry endpoint."""
from __future__ import annotations

from fastapi import APIRouter

from app.core.config import MODEL_ARTIFACT_PATH, get_dataset_path
from app.schemas.common import AlgorithmInfo, AlgorithmsResponse

router = APIRouter(tags=["Algorithms"])


@router.get("/algorithms", response_model=AlgorithmsResponse)
def get_algorithms() -> AlgorithmsResponse:
    dataset_ready = get_dataset_path().is_file()
    model_ready = MODEL_ARTIFACT_PATH.is_file()

    algos = [
        AlgorithmInfo(
            id="apriori",
            name="Standard Apriori",
            description=(
                "Level-wise frequent itemset mining with candidate generation, downward-closure "
                "pruning, and association rule generation (Support, Confidence, Lift)."
            ),
            category="Association Mining",
            readiness_status="ready" if dataset_ready else "requires_dataset",
            default_parameters={
                "min_support": 0.05,
                "min_confidence": 0.5,
                "max_len": 4,
                "sample_size": 25000,
            },
        ),
        AlgorithmInfo(
            id="adaptive_apriori",
            name="Adaptive Apriori (MSApriori)",
            description=(
                "Project-specific Multiple Minimum Supports Apriori. Solves the AML Rare Item Dilemma "
                "by assigning individual MIS thresholds to rare suspicious patterns while preserving "
                "valid downward-closure pruning through MIS item ordering."
            ),
            category="Association Mining",
            readiness_status="ready" if dataset_ready else "requires_dataset",
            default_parameters={
                "base_min_support": 0.10,
                "min_confidence": 0.4,
                "rare_target_mis": 0.0005,
                "max_len": 4,
                "sample_size": 25000,
            },
        ),
        AlgorithmInfo(
            id="utility_mining",
            name="Two-Phase High-Utility Itemset Mining (HUIM)",
            description=(
                "Discovers itemsets generating high cumulative monetary volume using Transaction-Weighted "
                "Utility (TWU) downward-closure upper-bound pruning, filtering overestimates in Phase 2."
            ),
            category="Utility Mining",
            readiness_status="ready" if dataset_ready else "requires_dataset",
            default_parameters={
                "min_utility_pct": 0.05,
                "max_len": 3,
                "sample_size": 25000,
            },
        ),
        AlgorithmInfo(
            id="random_forest",
            name="Random Forest Classifier",
            description=(
                "Supervised ensemble classification using chronological 70/15/15 train/val/test splitting, "
                "balanced class weighting for 0.10% positive imbalance, and strict target leakage prevention."
            ),
            category="Supervised ML",
            readiness_status="model_available" if model_ready else ("ready" if dataset_ready else "requires_dataset"),
            default_parameters={
                "n_estimators": 100,
                "max_depth": 15,
                "decision_threshold": 0.4,
                "sample_size": 50000,
            },
        ),
    ]
    return AlgorithmsResponse(algorithms=algos)
