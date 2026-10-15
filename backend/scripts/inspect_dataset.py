"""Run from the backend folder:
    python scripts/inspect_dataset.py
    python scripts/inspect_dataset.py --path "D:\\data\\HI-Small_Trans.csv"
    python scripts/inspect_dataset.py --max-rows 500000      (quick trial)
Writes data/outputs/dataset_inspection.json and prints a text report.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `app` importable

from app.core.config import DEFAULT_CHUNK_ROWS, OUTPUTS_DIR  # noqa: E402
from app.data.inspection import format_report, inspect_csv  # noqa: E402
from app.data.loader import DatasetNotFoundError  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect the AML CSV without loading it fully.")
    ap.add_argument("--path", default=None, help="CSV path (default: env FRAUDLENS_DATA_PATH or data/raw/HI-Small_Trans.csv)")
    ap.add_argument("--chunksize", type=int, default=DEFAULT_CHUNK_ROWS)
    ap.add_argument("--sample-rows", type=int, default=5)
    ap.add_argument("--max-rows", type=int, default=None, help="Stop after N rows (partial run)")
    a = ap.parse_args()
    try:
        result = inspect_csv(a.path, a.chunksize, a.sample_rows, a.max_rows)
    except DatasetNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(format_report(result))
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUTS_DIR / "dataset_inspection.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
