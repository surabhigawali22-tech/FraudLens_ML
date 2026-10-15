"""Central configuration. Nothing here reads the dataset; it only says where it is.

Override the dataset location without editing code:
    PowerShell:  $env:FRAUDLENS_DATA_PATH = "D:\\data\\HI-Small_Trans.csv"
"""
import os
from pathlib import Path

# config.py -> core -> app -> backend -> FraudLens (project root)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_RAW_CSV = PROJECT_ROOT / "data" / "raw" / "HI-Small_Trans.csv"
FALLBACK_RAW_CSV = PROJECT_ROOT.parent / "HI-Small_Trans.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "data" / "artifacts"
OUTPUTS_DIR = PROJECT_ROOT / "data" / "outputs"

# Rows per chunk when streaming the CSV. 200_000 rows is a modest amount of RAM.
DEFAULT_CHUNK_ROWS = 200_000

# API Settings
API_TITLE = "FraudLens AML Pattern Mining & Prediction API"
API_VERSION = "1.0.0"
API_PREFIX = "/api"

# CORS configuration restricted to local dev origins
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

# Resource & Sample Limits for interactive requests
DEFAULT_SAMPLE_SIZE = 25_000
MAX_INTERACTIVE_SAMPLE_SIZE = 100_000

# Persistent Artifact & Evaluation Paths
MODEL_ARTIFACT_PATH = ARTIFACTS_DIR / "random_forest_model.joblib"
EVALUATION_METRICS_PATH = OUTPUTS_DIR / "experiment_results_rf.json"


def get_dataset_path() -> Path:
    """Dataset path: env var FRAUDLENS_DATA_PATH if set, else data/raw/, else workspace root."""
    override = os.environ.get("FRAUDLENS_DATA_PATH")
    if override:
        return Path(override).expanduser()
    if DEFAULT_RAW_CSV.is_file():
        return DEFAULT_RAW_CSV
    if FALLBACK_RAW_CSV.is_file():
        return FALLBACK_RAW_CSV
    return DEFAULT_RAW_CSV
