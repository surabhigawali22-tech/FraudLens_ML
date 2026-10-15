"""Thread-safe background job execution and status tracking service."""
from __future__ import annotations

import datetime
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable


class JobService:
    def __init__(self, max_workers: int = 2) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def submit_job(
        self,
        job_type: str,
        target_fn: Callable[..., dict[str, Any]],
        *args: Any,
        **kwargs: Any,
    ) -> str:
        job_id = uuid.uuid4().hex
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()

        job_info: dict[str, Any] = {
            "job_id": job_id,
            "job_type": job_type,
            "status": "queued",
            "created_at": now,
            "completed_at": None,
            "runtime_seconds": None,
            "error": None,
            "result_id": None,
            "result": None,
        }

        with self._lock:
            self._jobs[job_id] = job_info

        def _worker_wrapper() -> None:
            with self._lock:
                self._jobs[job_id]["status"] = "running"

            start_t = time.time()
            try:
                res = target_fn(*args, **kwargs)
                elapsed = time.time() - start_t
                completed_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

                with self._lock:
                    self._jobs[job_id]["status"] = "completed"
                    self._jobs[job_id]["completed_at"] = completed_iso
                    self._jobs[job_id]["runtime_seconds"] = round(elapsed, 4)
                    self._jobs[job_id]["result"] = res
                    self._jobs[job_id]["result_id"] = res.get("result_id") if isinstance(res, dict) else None
            except Exception as e:
                elapsed = time.time() - start_t
                completed_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
                with self._lock:
                    self._jobs[job_id]["status"] = "failed"
                    self._jobs[job_id]["completed_at"] = completed_iso
                    self._jobs[job_id]["runtime_seconds"] = round(elapsed, 4)
                    self._jobs[job_id]["error"] = str(e)

        self._executor.submit(_worker_wrapper)
        return job_id

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def list_jobs(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            items = list(self._jobs.values())
        return sorted(items, key=lambda x: str(x.get("created_at")), reverse=True)[:limit]


# Global singleton instance
job_service = JobService()
