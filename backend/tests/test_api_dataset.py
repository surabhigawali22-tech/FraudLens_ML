"""Tests for dataset summary and preview endpoints."""
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient

from app.main import app


class TestApiDataset(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_dataset_summary_endpoint(self):
        resp = self.client.get("/api/dataset/summary")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("row_count", data)
        self.assertIn("columns", data)
        self.assertEqual(data["target_column"], "Is Laundering")
        self.assertEqual(data["data_readiness_status"], "verified")
        self.assertIn("target_distribution", data)

    def test_dataset_preview_bounded_sample(self):
        resp = self.client.get("/api/dataset/preview?limit=5")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["sample_size"], 5)
        self.assertEqual(len(data["rows"]), 5)
        self.assertIn("columns", data)
        # Check that records have expected fields
        first_row = data["rows"][0]
        self.assertIn("Timestamp", first_row)
        self.assertIn("Payment Format", first_row)

    def test_dataset_preview_validation_limits(self):
        # limit < 1
        resp_low = self.client.get("/api/dataset/preview?limit=0")
        self.assertEqual(resp_low.status_code, 422)

        # limit > 100
        resp_high = self.client.get("/api/dataset/preview?limit=101")
        self.assertEqual(resp_high.status_code, 422)

    @patch("app.services.dataset_service.get_dataset_path")
    def test_missing_dataset_returns_404(self, mock_path):
        mock_path.return_value = Path("non/existent/path/HI-Small_Trans.csv")
        resp = self.client.get("/api/dataset/summary?force_refresh=true")
        self.assertEqual(resp.status_code, 404)
        self.assertIn("not found", resp.json()["detail"].lower())


if __name__ == "__main__":
    unittest.main()
