"""In-memory cache for FraudLens algorithm results and dataset previews.

Computes deterministic cache keys incorporating:
- Algorithm name
- Normalized parameter dictionary
- Sample size / row limits
- Dataset path and dataset file modification timestamp
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Mapping


class ResultCache:
    """Thread-safe in-memory cache for algorithm results."""

    def __init__(self, max_entries: int = 50) -> None:
        self._cache: dict[str, dict[str, Any]] = {}
        self._max_entries = max_entries
        self._lock = threading.Lock()

    @staticmethod
    def generate_key(
        namespace: str,
        params: Mapping[str, Any],
        dataset_path: str | Path | None = None,
    ) -> str:
        """Create a deterministic SHA-256 hash key for a request."""
        mtime = None
        if dataset_path:
            p = Path(dataset_path)
            if p.is_file():
                try:
                    mtime = os.stat(p).st_mtime_ns
                except OSError:
                    mtime = None

        key_payload = {
            "namespace": namespace,
            "params": params,
            "dataset_path": str(dataset_path) if dataset_path else None,
            "dataset_mtime": mtime,
        }
        serialized = json.dumps(key_payload, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        """Retrieve cached result if present."""
        with self._lock:
            return self._cache.get(key)

    def set(self, key: str, value: dict[str, Any]) -> None:
        """Store result in cache, evicting oldest entry if limit reached."""
        with self._lock:
            if len(self._cache) >= self._max_entries and key not in self._cache:
                # Evict first inserted key (FIFO)
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            self._cache[key] = value

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._cache)


# Global singleton instance for the application
global_cache = ResultCache()
