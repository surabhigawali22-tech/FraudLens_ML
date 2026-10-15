"""Dataset inspection and preview service."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.config import OUTPUTS_DIR, get_dataset_path
from app.data import inspection, loader
from app.schemas.dataset import (
    DatasetPreviewResponse,
    DatasetSummaryResponse,
    TargetDistributionItem,
)


class DatasetService:
    @staticmethod
    def get_summary(force_refresh: bool = False) -> DatasetSummaryResponse:
        csv_path = get_dataset_path()
        if not csv_path.is_file():
            raise FileNotFoundError(
                f"Dataset CSV not found at {csv_path}. Please place HI-Small_Trans.csv "
                "in data/raw/ or set FRAUDLENS_DATA_PATH."
            )

        cached_json_path = OUTPUTS_DIR / "dataset_inspection.json"
        data: dict[str, Any] | None = None

        if not force_refresh and cached_json_path.is_file():
            try:
                cached_data = json.loads(cached_json_path.read_text(encoding="utf-8"))
                # Check if cached data corresponds to the same path
                if cached_data.get("path") == str(csv_path):
                    data = cached_data
            except Exception:
                data = None

        if data is None:
            # Run inspection
            data = inspection.inspect_csv(csv_path)
            OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
            cached_json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        target_dist = None
        if data.get("target_distribution"):
            target_dist = {
                k: TargetDistributionItem(
                    count=v["count"],
                    percent=v["percent"],
                )
                for k, v in data["target_distribution"].items()
            }

        return DatasetSummaryResponse(
            path=str(csv_path),
            file_size_mb=data.get("file_size_mb", 0.0),
            row_count=data.get("row_count", 0),
            columns=data.get("pandas_columns", []),
            target_column=data.get("target_column"),
            target_distribution=target_dist,
            timestamp_column=data.get("timestamp_column"),
            timestamp_min=data.get("timestamp_min"),
            timestamp_max=data.get("timestamp_max"),
            missing_values=data.get("missing_values", {}),
            data_readiness_status="verified" if data.get("target_column") else "available",
        )

    @staticmethod
    def get_preview(limit: int = 10) -> DatasetPreviewResponse:
        if not (1 <= limit <= 100):
            raise ValueError(f"Preview limit must be between 1 and 100, got {limit}")

        csv_path = get_dataset_path()
        if not csv_path.is_file():
            raise FileNotFoundError(f"Dataset CSV not found at {csv_path}.")

        df_sample = loader.read_sample(csv_path, n=limit)
        # Ensure values are clean standard JSON types
        records = json.loads(df_sample.to_json(orient="records"))
        return DatasetPreviewResponse(
            sample_size=len(records),
            total_columns=len(df_sample.columns),
            columns=list(df_sample.columns),
            rows=records,
        )


dataset_service = DatasetService()
