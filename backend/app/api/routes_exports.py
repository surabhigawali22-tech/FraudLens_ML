"""Endpoints for downloading CSV exports of mining and prediction results."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response

from app.services.export_service import export_service

router = APIRouter(prefix="/exports", tags=["Exports"])


@router.get("/{result_id}")
def download_export_csv(result_id: str) -> Response:
    export_result = export_service.get_export_csv(result_id)
    if not export_result:
        raise HTTPException(
            status_code=404,
            detail=f"Export result '{result_id}' not found or expired from session.",
        )

    filename, csv_content = export_result
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
