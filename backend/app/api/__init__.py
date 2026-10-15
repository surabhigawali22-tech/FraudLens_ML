"""API router aggregation for FraudLens."""
from __future__ import annotations

from fastapi import APIRouter

from app.api.routes_algorithms import router as router_algorithms
from app.api.routes_dataset import router as router_dataset
from app.api.routes_exports import router as router_exports
from app.api.routes_health import router as router_health
from app.api.routes_jobs import router as router_jobs
from app.api.routes_mining import router as router_mining
from app.api.routes_rf import router as router_rf

api_router = APIRouter()

api_router.include_router(router_health)
api_router.include_router(router_dataset)
api_router.include_router(router_algorithms)
api_router.include_router(router_mining)
api_router.include_router(router_rf)
api_router.include_router(router_jobs)
api_router.include_router(router_exports)
