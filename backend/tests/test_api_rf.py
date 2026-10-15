"""Tests for Random Forest API endpoints."""
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient

from app.main import app


class TestApiRandomForest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.sample_tx = {
            "Timestamp": "2022/09/01 14:30",
            "From Bank": 10,
            "To Bank": 20,
            "Account": "ACC_12345",
            "Account.1": "ACC_67890",
            "Amount Received": 4500.0,
            "Amount Paid": 4500.0,
            "Receiving Currency": "US Dollar",
            "Payment Currency": "US Dollar",
            "Payment Format": "Wire",
        }

    def test_evaluation_endpoint(self):
        resp = self.client.get("/api/random-forest/evaluation")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("model_available", data)
        self.assertIn("disclaimer", data)

    def test_predict_success(self):
        payload = {
            "transactions": [self.sample_tx],
            "decision_threshold": 0.45,
        }
        resp = self.client.post("/api/random-forest/predict", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["total_transactions"], 1)
        self.assertEqual(data["threshold_used"], 0.45)
        self.assertIn("disclaimer", data)
        self.assertIn("predictions", data)

        pred = data["predictions"][0]
        self.assertIn("probability_score", pred)
        self.assertIn("predicted_class", pred)
        self.assertIn("is_flagged", pred)
        self.assertIn("risk_level", pred)

    def test_target_leakage_prevented_in_prediction(self):
        """Verify that attempting to pass target 'Is Laundering' in inference features is rejected."""
        leaky_tx = dict(self.sample_tx)
        leaky_tx["Is Laundering"] = 1

        payload = {"transactions": [leaky_tx]}
        resp = self.client.post("/api/random-forest/predict", json=payload)
        # Must be rejected with 422 Validation Error
        self.assertEqual(resp.status_code, 422)
        err_msg = resp.text.lower()
        self.assertTrue(
            "target leakage" in err_msg or "is laundering" in err_msg,
            f"Expected target leakage error, got: {resp.text}",
        )

    @patch("app.api.routes_rf.MODEL_ARTIFACT_PATH")
    def test_missing_model_returns_404(self, mock_path):
        mock_path.is_file.return_value = False
        payload = {"transactions": [self.sample_tx]}
        resp = self.client.post("/api/random-forest/predict", json=payload)
        self.assertEqual(resp.status_code, 404)
        self.assertIn("not found", resp.json()["detail"].lower())

    def test_train_rf_bounded_sample(self):
        import tempfile
        payload = {
            "sample_size": 2_000,
            "n_estimators": 10,
            "max_depth": 5,
            "decision_threshold": 0.4,
            "run_in_background": False,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_model = Path(tmpdir) / "test_rf.joblib"
            tmp_eval = Path(tmpdir) / "test_eval.json"
            with patch("app.api.routes_rf.MODEL_ARTIFACT_PATH", tmp_model), \
                 patch("app.api.routes_rf.EVALUATION_METRICS_PATH", tmp_eval):
                resp = self.client.post("/api/random-forest/train", json=payload)
                self.assertEqual(resp.status_code, 200)
                data = resp.json()
                self.assertTrue(data["model_available"])
                self.assertIn("training_metadata", data)
                self.assertIn("top_features", data)


if __name__ == "__main__":
    unittest.main()
