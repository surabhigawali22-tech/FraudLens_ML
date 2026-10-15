"""Health and system readiness endpoint."""
from __future__ import annotations

import os
from fastapi import APIRouter

from app.core.config import API_VERSION, MODEL_ARTIFACT_PATH, get_dataset_path
from app.data.inspection import _free_ram_bytes
from app.schemas.common import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    dataset_p = get_dataset_path()
    dataset_ok = dataset_p.is_file()

    model_p = MODEL_ARTIFACT_PATH
    model_ok = model_p.is_file()

    free_ram = _free_ram_bytes()
    free_ram_mb = round(free_ram / 1024 / 1024) if free_ram is not None else None

    return HealthResponse(
        status="ok",
        version=API_VERSION,
        dataset_available=dataset_ok,
        dataset_path=str(dataset_p) if dataset_ok else None,
        model_available=model_ok,
        model_path=str(model_p) if model_ok else None,
        free_ram_mb=free_ram_mb,
    )
