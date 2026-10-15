"""Tests for background job management and CSV export endpoints."""
import time
import unittest
from fastapi.testclient import TestClient

from app.main import app


class TestApiJobsAndExports(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_background_job_lifecycle(self):
        # 1. Submit background job
        payload = {
            "min_support": 0.20,
            "min_confidence": 0.5,
            "sample_size": 1_000,
            "run_in_background": True,
        }
        resp = self.client.post("/api/apriori/run", json=payload)
        self.assertEqual(resp.status_code, 200)
        job_data = resp.json()
        self.assertIn("job_id", job_data)
        job_id = job_data["job_id"]
        self.assertIn(job_data["status"], ("queued", "running", "completed"))

        # 2. Poll job status
        for _ in range(20):
            time.sleep(0.1)
            poll_resp = self.client.get(f"/api/jobs/{job_id}")
            self.assertEqual(poll_resp.status_code, 200)
            status = poll_resp.json()["status"]
            if status in ("completed", "failed"):
                break

        self.assertEqual(status, "completed")
        self.assertIsNotNone(poll_resp.json()["result"])

    def test_list_jobs_endpoint(self):
        resp = self.client.get("/api/jobs")
        self.assertEqual(resp.status_code, 200)
        self.assertIsInstance(resp.json(), list)

    def test_missing_job_returns_404(self):
        resp = self.client.get("/api/jobs/non_existent_job_id")
        self.assertEqual(resp.status_code, 404)

    def test_csv_export_endpoint(self):
        # Run synchronous Apriori to obtain result_id
        payload = {
            "min_support": 0.20,
            "min_confidence": 0.5,
            "sample_size": 1_000,
            "run_in_background": False,
        }
        run_resp = self.client.post("/api/apriori/run", json=payload)
        self.assertEqual(run_resp.status_code, 200)
        result_id = run_resp.json()["result_id"]

        # Request CSV export
        export_resp = self.client.get(f"/api/exports/{result_id}")
        self.assertEqual(export_resp.status_code, 200)
        self.assertIn("text/csv", export_resp.headers["content-type"])
        self.assertIn("attachment", export_resp.headers["content-disposition"])
        self.assertTrue(export_resp.headers["content-disposition"].endswith('.csv"'))

        csv_text = export_resp.text
        self.assertIn("Antecedent", csv_text)
        self.assertIn("Consequent", csv_text)
        self.assertIn("Support", csv_text)

    def test_missing_export_returns_404(self):
        resp = self.client.get("/api/exports/invalid_export_id_12345")
        self.assertEqual(resp.status_code, 404)


if __name__ == "__main__":
    unittest.main()
