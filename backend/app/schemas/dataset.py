"""Pydantic models for Dataset summary and preview endpoints."""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class TargetDistributionItem(BaseModel):
    count: int
    percent: float | None = None


class DatasetSummaryResponse(BaseModel):
    path: str
    file_size_mb: float
    row_count: int
    columns: list[str]
    target_column: str | None = None
    target_distribution: dict[str, TargetDistributionItem] | None = None
    timestamp_column: str | None = None
    timestamp_min: str | None = None
    timestamp_max: str | None = None
    missing_values: dict[str, int] = Field(default_factory=dict)
    data_readiness_status: str  # e.g. "verified", "available", "not_found"


class DatasetPreviewResponse(BaseModel):
    sample_size: int
    total_columns: int
    columns: list[str]
    rows: list[dict[str, Any]]
