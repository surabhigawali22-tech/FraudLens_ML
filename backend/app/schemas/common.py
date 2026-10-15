"""Common Pydantic models for FraudLens API."""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., description="API operational status")
    version: str = Field(..., description="FraudLens API version")
    dataset_available: bool = Field(..., description="True if raw dataset CSV is detected")
    dataset_path: str | None = Field(None, description="Resolved path of dataset")
    model_available: bool = Field(..., description="True if trained Random Forest artifact exists")
    model_path: str | None = Field(None, description="Path of saved Random Forest model artifact")
    free_ram_mb: int | None = Field(None, description="Free host RAM in MB if psutil is available")


class AlgorithmInfo(BaseModel):
    id: str = Field(..., description="Unique algorithm key identifier")
    name: str = Field(..., description="Human-readable algorithm title")
    description: str = Field(..., description="Overview of theoretical formulation and role")
    category: str = Field(..., description="Algorithm type: 'Association Mining', 'Utility Mining', or 'Supervised ML'")
    readiness_status: str = Field(..., description="'ready' or 'requires_dataset' or 'model_available'")
    default_parameters: dict[str, Any] = Field(default_factory=dict)


class AlgorithmsResponse(BaseModel):
    algorithms: list[AlgorithmInfo]


class JobResponse(BaseModel):
    job_id: str
    job_type: str
    status: str  # queued, running, completed, failed
    created_at: str
    completed_at: str | None = None
    runtime_seconds: float | None = None
    error: str | None = None
    result_id: str | None = None
    result: dict[str, Any] | None = None
