"""Tests for pattern and utility mining endpoints."""
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient

from app.core.cache import global_cache
from app.main import app


class TestApiMining(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        global_cache.clear()

    def test_apriori_run_success(self):
        payload = {
            "min_support": 0.15,
            "min_confidence": 0.6,
            "max_len": 3,
            "sample_size": 1_000,
            "include_retrospective_target": False,
        }
        resp = self.client.post("/api/apriori/run", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["algorithm"], "Standard Apriori")
        self.assertEqual(data["sample_size"], 1_000)
        self.assertIn("result_id", data)
        self.assertIn("top_rules", data)
        self.assertIn("top_itemsets", data)
        self.assertIn("export_url", data)

    def test_apriori_validation_errors(self):
        # min_support <= 0
        resp1 = self.client.post("/api/apriori/run", json={"min_support": 0.0})
        self.assertEqual(resp1.status_code, 422)

        # min_confidence > 1.0
        resp2 = self.client.post("/api/apriori/run", json={"min_confidence": 1.5})
        self.assertEqual(resp2.status_code, 422)

    def test_adaptive_apriori_run_success(self):
        payload = {
            "base_min_support": 0.20,
            "min_confidence": 0.5,
            "max_len": 3,
            "sample_size": 1_000,
            "include_retrospective_target": True,
        }
        resp = self.client.post("/api/adaptive-apriori/run", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("Adaptive Apriori", data["algorithm"])
        self.assertIn("result_id", data)

    def test_utility_mining_run_success(self):
        payload = {
            "min_utility_pct": 0.10,
            "max_len": 3,
            "sample_size": 1_000,
        }
        resp = self.client.post("/api/utility-mining/run", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["algorithm"], "Two-Phase High-Utility Itemset Mining (HUIM)")
        self.assertIn("effective_min_utility", data["parameters"])
        self.assertIn("total_sample_volume", data["parameters"])

    def test_caching_returns_cached_flag(self):
        payload = {
            "min_support": 0.25,
            "min_confidence": 0.7,
            "sample_size": 500,
        }
        resp1 = self.client.post("/api/apriori/run", json=payload)
        self.assertEqual(resp1.status_code, 200)
        self.assertFalse(resp1.json()["cached"])

        resp2 = self.client.post("/api/apriori/run", json=payload)
        self.assertEqual(resp2.status_code, 200)
        self.assertTrue(resp2.json()["cached"])

    @patch("app.api.routes_mining.get_dataset_path")
    def test_missing_dataset_returns_404(self, mock_path):
        mock_path.return_value = Path("non/existent/HI-Small_Trans.csv")
        resp = self.client.post("/api/apriori/run", json={"min_support": 0.1})
        self.assertEqual(resp.status_code, 404)


if __name__ == "__main__":
    unittest.main()
