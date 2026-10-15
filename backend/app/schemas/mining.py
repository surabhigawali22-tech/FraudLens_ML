"""Pydantic request and response schemas for pattern and utility mining endpoints."""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field, field_validator


class AprioriRequest(BaseModel):
    min_support: float = Field(0.05, gt=0.0, le=1.0, description="Minimum relative support threshold")
    min_confidence: float = Field(0.5, ge=0.0, le=1.0, description="Minimum association rule confidence")
    max_len: int | None = Field(4, ge=1, le=10, description="Maximum itemset length constraint")
    sample_size: int = Field(25_000, ge=100, le=100_000, description="Bounded sample size from dataset")
    include_retrospective_target: bool = Field(False, description="Include retrospective label in baskets")
    run_in_background: bool = Field(False, description="If True, dispatches to background job runner")


class AdaptiveAprioriRequest(BaseModel):
    base_min_support: float = Field(0.10, gt=0.0, le=1.0, description="Default MIS for unspecified items")
    min_confidence: float = Field(0.4, ge=0.0, le=1.0, description="Minimum rule confidence threshold")
    mis_dict: dict[str, float] | None = Field(
        default=None,
        description="Manual mapping of item name to individual MIS threshold",
    )
    use_auto_mis: bool = Field(False, description="Auto-compute MIS using beta * frequency")
    beta: float = Field(0.5, gt=0.0, le=1.0, description="Support difference scaling factor")
    lowest_support: float = Field(0.005, gt=0.0, le=1.0, description="Absolute floor for auto-computed MIS")
    max_len: int | None = Field(4, ge=1, le=10, description="Maximum itemset length constraint")
    sample_size: int = Field(25_000, ge=100, le=100_000, description="Bounded sample size from dataset")
    include_retrospective_target: bool = Field(True, description="Include retrospective label in baskets")
    run_in_background: bool = Field(False, description="If True, dispatches to background job runner")


class HUIMRequest(BaseModel):
    min_utility: float | None = Field(None, gt=0.0, description="Absolute minimum dollar volume utility")
    min_utility_pct: float | None = Field(
        0.05, gt=0.0, le=1.0, description="Minimum utility as percentage of total sample transaction volume"
    )
    max_len: int | None = Field(3, ge=1, le=10, description="Maximum itemset length constraint")
    sample_size: int = Field(25_000, ge=100, le=100_000, description="Bounded sample size from dataset")
    run_in_background: bool = Field(False, description="If True, dispatches to background job runner")


class MiningRuleItem(BaseModel):
    antecedent: list[str]
    consequent: list[str]
    support: float
    confidence: float
    lift: float


class MiningItemsetItem(BaseModel):
    items: list[str]
    support_count: int
    support: float | None = None
    utility: float | None = None
    twu: float | None = None
    min_support_threshold: float | None = None


class MiningResponse(BaseModel):
    algorithm: str
    sample_size: int
    runtime_seconds: float
    cached: bool = False
    result_id: str
    itemsets_count: int
    rules_count: int
    top_rules: list[MiningRuleItem] = Field(default_factory=list)
    top_itemsets: list[MiningItemsetItem] = Field(default_factory=list)
    export_url: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
