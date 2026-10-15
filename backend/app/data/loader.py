"""Dataset loading interface. Streams the CSV; never loads it whole.

Deliberately makes NO assumptions about column names. Column-specific logic
(feature engineering, baskets) will be added after the real schema is confirmed.
"""
import csv
from pathlib import Path
from typing import Iterator, Optional, Sequence

import pandas as pd

from app.core.config import DEFAULT_CHUNK_ROWS, get_dataset_path


class DatasetNotFoundError(FileNotFoundError):
    pass


def resolve_path(path: Optional[str | Path] = None) -> Path:
    """Return an existing dataset path, or raise a clear error."""
    p = Path(path).expanduser() if path else get_dataset_path()
    if not p.is_file():
        raise DatasetNotFoundError(
            f"Dataset file not found: {p}\n"
            "Place HI-Small_Trans.csv in data/raw/, pass --path, "
            "or set the FRAUDLENS_DATA_PATH environment variable."
        )
    return p


def read_raw_header(path: Optional[str | Path] = None) -> list[str]:
    """Header exactly as written in the file (pandas would rename duplicates)."""
    p = resolve_path(path)
    with open(p, "r", encoding="utf-8-sig", newline="") as f:
        return next(csv.reader(f))


def iter_chunks(
    path: Optional[str | Path] = None,
    chunksize: int = DEFAULT_CHUNK_ROWS,
    usecols: Optional[Sequence[str]] = None,
    max_rows: Optional[int] = None,
) -> Iterator[pd.DataFrame]:
    """Yield DataFrame chunks. max_rows stops early (for quick trial runs)."""
    p = resolve_path(path)
    seen = 0
    with pd.read_csv(p, chunksize=chunksize, usecols=usecols) as reader:
        for chunk in reader:
            if max_rows is not None and seen + len(chunk) > max_rows:
                chunk = chunk.iloc[: max_rows - seen]
            seen += len(chunk)
            yield chunk
            if max_rows is not None and seen >= max_rows:
                break


def read_sample(path: Optional[str | Path] = None, n: int = 5) -> pd.DataFrame:
    """First n rows only."""
    p = resolve_path(path)
    return pd.read_csv(p, nrows=n)
