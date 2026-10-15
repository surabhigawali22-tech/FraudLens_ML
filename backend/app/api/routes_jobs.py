"""Endpoints for background job status retrieval."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas.common import JobResponse
from app.services.job_service import job_service

router = APIRouter(prefix="/jobs", tags=["Jobs"])


@router.get("/{job_id}", response_model=JobResponse)
def get_job_status(job_id: str) -> JobResponse:
    job = job_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return JobResponse(**job)


@router.get("", response_model=list[JobResponse])
def list_jobs(limit: int = 20) -> list[JobResponse]:
    jobs = job_service.list_jobs(limit=limit)
    return [JobResponse(**j) for j in jobs]
