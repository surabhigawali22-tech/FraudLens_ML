"""Tests for health, root, and algorithm registry endpoints."""
import unittest
from fastapi.testclient import TestClient

from app.main import app


class TestApiHealth(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_root_endpoint(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "online")
        self.assertEqual(data["api_prefix"], "/api")
        self.assertIn("documentation", data)

    def test_health_endpoint(self):
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("dataset_available", data)
        self.assertIn("model_available", data)
        self.assertIn("version", data)

    def test_algorithms_registry_contains_four_required_algorithms(self):
        resp = self.client.get("/api/algorithms")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        algos = data["algorithms"]
        self.assertEqual(len(algos), 4)

        algo_ids = {a["id"] for a in algos}
        expected_ids = {"apriori", "adaptive_apriori", "utility_mining", "random_forest"}
        self.assertEqual(algo_ids, expected_ids)

        for a in algos:
            self.assertIn("name", a)
            self.assertIn("description", a)
            self.assertIn("category", a)
            self.assertIn("readiness_status", a)
            self.assertIsInstance(a["default_parameters"], dict)


if __name__ == "__main__":
    unittest.main()
