"""Dataset inspection and preview endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.schemas.dataset import DatasetPreviewResponse, DatasetSummaryResponse
from app.services.dataset_service import dataset_service

router = APIRouter(prefix="/dataset", tags=["Dataset"])


@router.get("/summary", response_model=DatasetSummaryResponse)
def get_dataset_summary(force_refresh: bool = False) -> DatasetSummaryResponse:
    try:
        return dataset_service.get_summary(force_refresh=force_refresh)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dataset inspection failed: {e}")


@router.get("/preview", response_model=DatasetPreviewResponse)
def get_dataset_preview(
    limit: int = Query(10, ge=1, le=100, description="Number of sample records to preview")
) -> DatasetPreviewResponse:
    try:
        return dataset_service.get_preview(limit=limit)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dataset preview failed: {e}")
