"""Memory-efficient dataset inspection (chunked). Reports only what it measures."""
import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from app.core.config import DEFAULT_CHUNK_ROWS
from app.data import loader

TARGET_CANDIDATES = ("is laundering", "is_laundering", "islaundering")


def find_target_column(columns: list[str]) -> Optional[str]:
    """Case-insensitive match against known spellings. None if not found."""
    for c in columns:
        if c.strip().lower() in TARGET_CANDIDATES:
            return c
    return None


def _free_ram_bytes() -> Optional[int]:
    try:
        import psutil  # optional dependency
        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def inspect_csv(
    path: Optional[str | Path] = None,
    chunksize: int = DEFAULT_CHUNK_ROWS,
    sample_rows: int = 5,
    max_rows: Optional[int] = None,
) -> dict:
    p = loader.resolve_path(path)
    raw_header = loader.read_raw_header(p)

    n_rows = 0
    n_chunks = 0
    dtypes: dict[str, set] = {}
    missing: dict[str, int] = {}
    columns: list[str] = []
    target_col: Optional[str] = None
    target_counts: dict[str, int] = {}
    ts_col: Optional[str] = None
    ts_min = ts_max = None
    ts_unparsed = 0
    sample = None

    for chunk in loader.iter_chunks(p, chunksize=chunksize, max_rows=max_rows):
        if n_chunks == 0:
            columns = list(chunk.columns)
            target_col = find_target_column(columns)
            ts_col = next((c for c in columns if c.strip().lower() == "timestamp"), None)
            sample = chunk.head(sample_rows)
            missing = {c: 0 for c in columns}
        n_chunks += 1
        n_rows += len(chunk)
        for c in columns:
            dtypes.setdefault(c, set()).add(str(chunk[c].dtype))
            missing[c] += int(chunk[c].isna().sum())
        if target_col is not None:
            for k, v in chunk[target_col].value_counts(dropna=False).items():
                key = "NaN" if pd.isna(k) else str(k)
                target_counts[key] = target_counts.get(key, 0) + int(v)
        if ts_col is not None:
            parsed = pd.to_datetime(chunk[ts_col], errors="coerce")
            ts_unparsed += int(parsed.isna().sum())
            if parsed.notna().any():
                lo, hi = parsed.min(), parsed.max()
                ts_min = lo if ts_min is None or lo < ts_min else ts_min
                ts_max = hi if ts_max is None or hi > ts_max else ts_max

    result = {
        "path": str(p),
        "file_size_mb": round(os.stat(p).st_size / 1024 / 1024, 2),
        "free_ram_mb": None,
        "chunksize": chunksize,
        "partial_run_max_rows": max_rows,
        "raw_header": raw_header,
        "pandas_columns": columns,
        "duplicate_header_names": sorted({c for c in raw_header if raw_header.count(c) > 1}),
        "row_count": n_rows,
        "dtypes": {c: sorted(v) for c, v in dtypes.items()},
        "missing_values": missing,
        "target_column": target_col,
        "target_distribution": None,
        "timestamp_column": ts_col,
        "timestamp_min": None if ts_min is None else str(ts_min),
        "timestamp_max": None if ts_max is None else str(ts_max),
        "timestamp_unparsed": ts_unparsed if ts_col else None,
        "sample": [] if sample is None else json.loads(sample.to_json(orient="records")),
    }
    ram = _free_ram_bytes()
    if ram is not None:
        result["free_ram_mb"] = round(ram / 1024 / 1024)

    if target_col is not None:
        total = sum(target_counts.values())
        result["target_distribution"] = {
            k: {"count": v, "percent": round(100 * v / total, 4) if total else None}
            for k, v in sorted(target_counts.items())
        }
    return result


def format_report(r: dict) -> str:
    L = []
    add = L.append
    add("=" * 70)
    add("FRAUDLENS DATASET INSPECTION")
    add("=" * 70)
    add(f"File            : {r['path']}")
    add(f"File size       : {r['file_size_mb']} MB")
    add(f"Free RAM        : {r['free_ram_mb']} MB" if r["free_ram_mb"] is not None
        else "Free RAM        : unknown (install psutil to report this)")
    add(f"Chunk size      : {r['chunksize']} rows")
    if r["partial_run_max_rows"]:
        add(f"!! PARTIAL RUN  : only first {r['partial_run_max_rows']} rows were read")
    add(f"Rows read       : {r['row_count']}")
    add("")
    add("RAW HEADER (as written in the file):")
    for i, c in enumerate(r["raw_header"]):
        add(f"  [{i}] {c}")
    if r["duplicate_header_names"]:
        add(f"  NOTE: duplicate header name(s): {r['duplicate_header_names']} "
            "(pandas renames the second one, e.g. 'Account.1')")
    add("")
    add("COLUMNS AS PANDAS SEES THEM -> dtype(s), missing values:")
    for c in r["pandas_columns"]:
        add(f"  {c:<22} {','.join(r['dtypes'][c]):<10} missing={r['missing_values'][c]}")
    add("")
    if r["target_column"] is None:
        add("TARGET COLUMN: NOT FOUND (looked for 'Is Laundering' and spelling variants)")
    else:
        add(f"TARGET COLUMN: found -> '{r['target_column']}'")
        for k, v in r["target_distribution"].items():
            add(f"  value {k}: {v['count']} rows ({v['percent']}%)")
    add("")
    if r["timestamp_column"]:
        add(f"TIMESTAMP: column '{r['timestamp_column']}', min={r['timestamp_min']}, "
            f"max={r['timestamp_max']}, unparseable={r['timestamp_unparsed']}")
    else:
        add("TIMESTAMP: column not found")
    add("")
    add("SAMPLE RECORDS:")
    for row in r["sample"]:
        add("  " + json.dumps(row))
    add("=" * 70)
    return "\n".join(L)
