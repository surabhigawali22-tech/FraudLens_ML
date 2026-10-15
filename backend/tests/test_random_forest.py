"""Unit tests for AML Random Forest Classifier Pipeline.
Verifies target leakage prevention, chronological splitting, training,
prediction, threshold adjustments, evaluation metrics, and persistence.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from app.algorithms.random_forest import AMLRandomForestPipeline
from app.data.features import build_rf_feature_matrix


class TestRandomForest(unittest.TestCase):
    def setUp(self):
        # Create a synthetic dataset mimicking HI-Small schema with 100 chronological rows
        n = 100
        dates = pd.date_range("2022-09-01 00:00", periods=n, freq="15min")
        self.df = pd.DataFrame({
            "Timestamp": dates.strftime("%Y/%m/%d %H:%M"),
            "From Bank": np.random.choice([10, 20, 30], size=n),
            "Account": [f"ACC_{i % 15}" for i in range(n)],
            "To Bank": np.random.choice([10, 20, 40], size=n),
            "Account.1": [f"ACC_{(i + 1) % 15}" for i in range(n)],
            "Amount Received": np.random.uniform(50, 50000, size=n),
            "Receiving Currency": np.random.choice(["US Dollar", "Euro"], size=n),
            "Amount Paid": np.random.uniform(50, 50000, size=n),
            "Payment Currency": np.random.choice(["US Dollar", "Euro"], size=n),
            "Payment Format": np.random.choice(["ACH", "Wire", "Cash", "Cheque"], size=n),
            # 10% positives
            "Is Laundering": np.array([1 if i % 10 == 0 else 0 for i in range(n)]),
        })

    def test_target_leakage_prevention(self):
        """Verify that target column is strictly isolated and X does not contain Is Laundering."""
        X, y = build_rf_feature_matrix(self.df)
        self.assertNotIn("Is Laundering", X.columns)
        self.assertEqual(len(X), len(self.df))
        self.assertEqual(len(y), len(self.df))

        # Explicitly trying to fit pipeline with target column in X must raise ValueError
        pipeline = AMLRandomForestPipeline(n_estimators=10)
        X_with_target = X.copy()
        X_with_target["Is Laundering"] = y
        with self.assertRaises(ValueError):
            pipeline.fit(X_with_target, y)

    def test_chronological_splitting(self):
        """Verify chronological split divides dataset into proper time windows without shuffling."""
        train_df, val_df, test_df = AMLRandomForestPipeline.chronological_split(
            self.df, train_ratio=0.70, val_ratio=0.15
        )
        self.assertEqual(len(train_df), 70)
        self.assertEqual(len(val_df), 15)
        self.assertEqual(len(test_df), 15)
        self.assertEqual(len(train_df) + len(val_df) + len(test_df), len(self.df))

        # Check temporal order: train timestamps <= val timestamps <= test timestamps
        max_train_ts = train_df["Timestamp"].max()
        min_val_ts = val_df["Timestamp"].min()
        max_val_ts = val_df["Timestamp"].max()
        min_test_ts = test_df["Timestamp"].min()

        self.assertLessEqual(max_train_ts, min_val_ts)
        self.assertLessEqual(max_val_ts, min_test_ts)

    def test_fit_and_predict(self):
        """Verify model training and binary prediction."""
        X, y = build_rf_feature_matrix(self.df)
        pipeline = AMLRandomForestPipeline(n_estimators=15, max_depth=5, random_state=42)
        pipeline.fit(X.iloc[:80], y.iloc[:80])

        probs = pipeline.predict_proba(X.iloc[80:])
        preds = pipeline.predict(X.iloc[80:])

        self.assertEqual(len(probs), 20)
        self.assertEqual(len(preds), 20)
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))
        self.assertTrue(set(preds).issubset({0, 1}))

    def test_configurable_decision_threshold(self):
        """Verify that lowering decision threshold increases or maintains detected positives."""
        X, y = build_rf_feature_matrix(self.df)
        pipeline = AMLRandomForestPipeline(n_estimators=15, max_depth=5, random_state=42)
        pipeline.fit(X.iloc[:80], y.iloc[:80])

        preds_high = pipeline.predict(X.iloc[80:], threshold=0.8)
        preds_low = pipeline.predict(X.iloc[80:], threshold=0.2)

        # Lower threshold should classify equal or more samples as positive
        self.assertGreaterEqual(preds_low.sum(), preds_high.sum())

    def test_evaluation_metrics_integrity(self):
        """Verify evaluation metrics: confusion matrix adds up to sample size."""
        X, y = build_rf_feature_matrix(self.df)
        pipeline = AMLRandomForestPipeline(n_estimators=15, max_depth=5, random_state=42)
        pipeline.fit(X.iloc[:80], y.iloc[:80])

        metrics = pipeline.evaluate(X.iloc[80:], y.iloc[80:], split_name="Test")
        cm = metrics.confusion_matrix

        total_cm = cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"]
        self.assertEqual(total_cm, len(y.iloc[80:]))
        self.assertEqual(metrics.sample_size, 20)

    def test_save_and_load_persistence(self):
        """Verify model serialization and loaded prediction fidelity."""
        X, y = build_rf_feature_matrix(self.df)
        pipeline = AMLRandomForestPipeline(n_estimators=15, max_depth=5, random_state=42)
        pipeline.fit(X.iloc[:80], y.iloc[:80])

        original_probs = pipeline.predict_proba(X.iloc[80:])

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "rf_model.joblib"
            pipeline.save(model_path)
            self.assertTrue(model_path.is_file())

            loaded = AMLRandomForestPipeline.load(model_path)
            loaded_probs = loaded.predict_proba(X.iloc[80:])

            np.testing.assert_allclose(original_probs, loaded_probs, rtol=1e-5)

    def test_positive_class_identification_and_probability_extraction(self):
        """Verify that class 1 is correctly identified in model.classes_ and probabilities target class 1."""
        X, y = build_rf_feature_matrix(self.df)
        pipeline = AMLRandomForestPipeline(n_estimators=10, max_depth=3, random_state=42)
        pipeline.fit(X.iloc[:80], y.iloc[:80])

        self.assertIn(1, pipeline.classifier_.classes_)
        self.assertIn(0, pipeline.classifier_.classes_)

        probs = pipeline.predict_proba(X.iloc[80:])
        full_proba = pipeline.classifier_.predict_proba(pipeline.preprocessor_.transform(X.iloc[80:]))
        idx_1 = list(pipeline.classifier_.classes_).index(1)

        np.testing.assert_allclose(probs, full_proba[:, idx_1], rtol=1e-5)

    def test_confusion_matrix_label_order(self):
        """Verify confusion matrix follows exact [0, 1] label order."""
        X, y = build_rf_feature_matrix(self.df)
        pipeline = AMLRandomForestPipeline(n_estimators=10, max_depth=3, decision_threshold=0.4, random_state=42)
        pipeline.fit(X.iloc[:70], y.iloc[:70])

        metrics = pipeline.evaluate(X.iloc[70:], y.iloc[70:], threshold=0.4)
        cm = metrics.confusion_matrix

        # Hand calculate expected counts
        probs = pipeline.predict_proba(X.iloc[70:])
        preds = (probs >= 0.4).astype(int)
        y_true = y.iloc[70:].values.astype(int)

        expected_tn = int(((y_true == 0) & (preds == 0)).sum())
        expected_fp = int(((y_true == 0) & (preds == 1)).sum())
        expected_fn = int(((y_true == 1) & (preds == 0)).sum())
        expected_tp = int(((y_true == 1) & (preds == 1)).sum())

        self.assertEqual(cm["tn"], expected_tn)
        self.assertEqual(cm["fp"], expected_fp)
        self.assertEqual(cm["fn"], expected_fn)
        self.assertEqual(cm["tp"], expected_tp)

    def test_evaluation_both_classes_metrics(self):
        """Verify precision, recall, F1, and continuous ROC-AUC are computed when both classes exist."""
        X, y = build_rf_feature_matrix(self.df)
        pipeline = AMLRandomForestPipeline(n_estimators=10, max_depth=3, random_state=42)
        pipeline.fit(X.iloc[:70], y.iloc[:70])

        metrics = pipeline.evaluate(X.iloc[70:], y.iloc[70:], threshold=0.4)
        self.assertGreater(metrics.positive_count, 0)
        self.assertGreater(metrics.negative_count, 0)
        self.assertIsNotNone(metrics.roc_auc)
        self.assertIsNotNone(metrics.pr_auc)
        self.assertIsNotNone(metrics.precision)
        self.assertIsNotNone(metrics.recall)
        self.assertIsNotNone(metrics.f1)
        self.assertGreaterEqual(metrics.roc_auc, 0.0)
        self.assertLessEqual(metrics.roc_auc, 1.0)

    def test_evaluation_single_class_graceful_handling(self):
        """Verify that when test set contains only one class, metrics explain undefined status without crashing."""
        X, y = build_rf_feature_matrix(self.df)
        pipeline = AMLRandomForestPipeline(n_estimators=10, max_depth=3, random_state=42)
        pipeline.fit(X.iloc[:70], y.iloc[:70])

        # Create purely negative evaluation slice
        X_neg = X.iloc[70:].copy()
        y_neg = pd.Series([0] * len(X_neg), index=X_neg.index)

        metrics = pipeline.evaluate(X_neg, y_neg, threshold=0.4)
        self.assertEqual(metrics.positive_count, 0)
        self.assertEqual(metrics.negative_count, len(X_neg))
        self.assertIsNone(metrics.roc_auc)
        self.assertIsNone(metrics.precision)
        self.assertIsNone(metrics.recall)
        self.assertIsNotNone(metrics.notes)
        self.assertIn("Single-class", metrics.notes)
        self.assertEqual(metrics.confusion_matrix["tn"] + metrics.confusion_matrix["fp"], len(X_neg))
        self.assertEqual(metrics.confusion_matrix["tp"], 0)
        self.assertEqual(metrics.confusion_matrix["fn"], 0)

    def test_chronological_split_smart_holdout_adjustment(self):
        """Verify that chronological_split shifts val_end backward if test slice would otherwise have 0 positives."""
        # Create dataset where positive is at index 82 (in validation slice 70..85)
        n = 100
        df = self.df.copy()
        df["Is Laundering"] = 0
        df.loc[10, "Is Laundering"] = 1   # in train (0..70)
        df.loc[75, "Is Laundering"] = 1   # in val (70..85)

        train_df, val_df, test_df = AMLRandomForestPipeline.chronological_split(
            df, train_ratio=0.70, val_ratio=0.15, target_col="Is Laundering"
        )

        # val_end should have shifted backward to 75 so test_df includes index 75
        self.assertEqual((train_df["Is Laundering"] == 1).sum(), 1)
        self.assertEqual((test_df["Is Laundering"] == 1).sum(), 1)
        self.assertTrue(train_df["Timestamp"].max() <= val_df["Timestamp"].min())
        self.assertTrue(val_df["Timestamp"].max() <= test_df["Timestamp"].min())


if __name__ == "__main__":
    unittest.main()
