"""Main FastAPI application for FraudLens."""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import api_router
from app.core.config import API_PREFIX, API_TITLE, API_VERSION, CORS_ORIGINS


def create_app() -> FastAPI:
    app = FastAPI(
        title=API_TITLE,
        version=API_VERSION,
        description=(
            "FraudLens Backend API for Anti-Money Laundering (AML) transaction-pattern mining "
            "and predictive classification using IBM AML HI-Small transaction data."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS configuration restricted to local dev servers
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global exception handler for ValueError (e.g. invalid arguments or target leakage)
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"detail": str(exc), "error_type": "ValidationError"},
        )

    # Mount API routes under /api
    app.include_router(api_router, prefix=API_PREFIX)

    @app.get("/", tags=["Root"])
    def root() -> dict[str, str]:
        return {
            "name": API_TITLE,
            "version": API_VERSION,
            "status": "online",
            "documentation": "/docs",
            "api_prefix": API_PREFIX,
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
