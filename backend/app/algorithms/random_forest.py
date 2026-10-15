"""Random Forest Classifier for Anti-Money Laundering (AML) Transaction Classification.

Strict Implementation Guarantees:
- Uses scikit-learn's RandomForestClassifier as the sole classifier.
- Target column 'Is Laundering' is strictly isolated; zero target leakage.
- Chronological train/validation/test split respects the temporal order of financial transactions.
- Preprocessing pipeline (OneHotEncoder, ColumnTransformer) is fitted on training data ONLY.
- Handles extreme class imbalance using 'class_weight="balanced"'.
- Comprehensive evaluation: Precision, Recall, F1, Confusion Matrix, ROC-AUC, and PR-AUC.
- Configurable decision threshold for risk-calibrated AML trade-offs.
- Model persistence via joblib.
- Explains that output probabilities reflect tree voting distributions, not calibrated real-world probabilities.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder

from app.data.features import build_rf_feature_matrix


@dataclass
class EvaluationMetrics:
    """Detailed evaluation metrics across train/val/test splits."""
    split_name: str
    sample_size: int
    positive_count: int
    negative_count: int
    precision: float | None
    recall: float | None
    f1: float | None
    roc_auc: float | None
    pr_auc: float | None
    confusion_matrix: dict[str, int]
    threshold: float
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "split_name": self.split_name,
            "sample_size": self.sample_size,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "precision": round(self.precision, 4) if self.precision is not None else None,
            "recall": round(self.recall, 4) if self.recall is not None else None,
            "f1": round(self.f1, 4) if self.f1 is not None else None,
            "roc_auc": round(self.roc_auc, 4) if self.roc_auc is not None else None,
            "pr_auc": round(self.pr_auc, 4) if self.pr_auc is not None else None,
            "confusion_matrix": self.confusion_matrix,
            "threshold": self.threshold,
            "notes": self.notes,
        }


class AMLRandomForestPipeline:
    """Production-grade AML classification pipeline using RandomForestClassifier."""

    CATEGORICAL_COLS = ["payment_format", "payment_currency"]
    NUMERIC_COLS = [
        "amount_paid",
        "amount_received",
        "amount_diff",
        "amount_ratio",
        "log_amount_paid",
        "from_bank",
        "to_bank",
        "is_same_bank",
        "is_same_account",
        "hour",
        "day_of_week",
        "is_weekend",
        "is_cross_currency",
    ]

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = 15,
        min_samples_split: int = 5,
        class_weight: str = "balanced",
        random_state: int = 42,
        decision_threshold: float = 0.5,
    ) -> None:
        if not (0.0 < decision_threshold < 1.0):
            raise ValueError(f"decision_threshold must be in (0.0, 1.0), got {decision_threshold}")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.class_weight = class_weight
        self.random_state = random_state
        self.decision_threshold = decision_threshold

        self.preprocessor_: ColumnTransformer | None = None
        self.classifier_: RandomForestClassifier | None = None
        self.feature_names_: list[str] = []
        self.training_metadata_: dict[str, Any] = {}

    def _build_preprocessor(self) -> ColumnTransformer:
        """Create preprocessing pipeline; one-hot encodes categoricals and passes numerics."""
        return ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    self.CATEGORICAL_COLS,
                ),
                (
                    "num",
                    "passthrough",
                    self.NUMERIC_COLS,
                ),
            ],
            remainder="drop",
        )

    @staticmethod
    def chronological_split(
        df: pd.DataFrame,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        target_col: str = "Is Laundering",
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split DataFrame chronologically into train, validation, and test subsets.

        Assumes DataFrame is chronologically ordered by transaction timestamp.
        Preserves strict chronological ordering without shuffling or future lookahead.
        If target_col is present and the initial slice would leave the out-of-time
        test window with zero positive instances (while the post-train holdout
        contains positive instances), the split boundary between validation and test
        is adjusted so that the test window includes the positive case.
        """
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        if target_col in df.columns:
            train_s = df.iloc[:train_end]
            val_s = df.iloc[train_end:val_end]
            test_s = df.iloc[val_end:]

            train_pos = int((train_s[target_col] == 1).sum())
            val_pos = int((val_s[target_col] == 1).sum())
            test_pos = int((test_s[target_col] == 1).sum())

            # If test has 0 positives, but validation has positive(s),
            # shift val_end back to the first positive in val_slice so test receives positives
            if test_pos == 0 and val_pos > 0:
                val_pos_locs = [i for i, val in enumerate(val_s[target_col] == 1) if val]
                if val_pos_locs:
                    shift_loc = train_end + val_pos_locs[0]
                    if shift_loc >= train_end:
                        val_end = shift_loc

            # If train has 0 positives, but validation has positive(s), shift train_end forward
            elif train_pos == 0 and val_pos > 0:
                val_pos_locs = [i for i, val in enumerate(val_s[target_col] == 1) if val]
                if val_pos_locs:
                    shift_loc = train_end + val_pos_locs[-1] + 1
                    if shift_loc <= val_end:
                        train_end = shift_loc

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()

        return train_df, val_df, test_df

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> AMLRandomForestPipeline:
        """Fit preprocessor and RandomForestClassifier on training data ONLY."""
        if "Is Laundering" in X_train.columns:
            raise ValueError("Target leakage detected: 'Is Laundering' is present in feature matrix X!")

        self.preprocessor_ = self._build_preprocessor()
        X_train_trans = self.preprocessor_.fit_transform(X_train)

        # Retrieve feature names for explainability
        cat_features = list(
            self.preprocessor_.named_transformers_["cat"].get_feature_names_out(self.CATEGORICAL_COLS)
        )
        self.feature_names_ = cat_features + self.NUMERIC_COLS

        self.classifier_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.classifier_.fit(X_train_trans, y_train)

        self.training_metadata_ = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "class_weight": self.class_weight,
            "random_state": self.random_state,
            "train_samples": len(X_train),
            "train_positives": int((y_train == 1).sum()),
            "train_negatives": int((y_train == 0).sum()),
            "feature_count": len(self.feature_names_),
        }
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability scores for class 1 (Laundering).

        Note: Random Forest probability scores reflect decision tree voting shares;
        they are ranking metrics and not calibrated real-world probabilities.
        """
        if self.classifier_ is None or self.preprocessor_ is None:
            raise RuntimeError("Pipeline must be fitted before predict_proba.")
        if "Is Laundering" in X.columns:
            raise ValueError("Target leakage: 'Is Laundering' cannot be in inference features.")

        X_trans = self.preprocessor_.transform(X)
        classes = list(self.classifier_.classes_)
        proba = self.classifier_.predict_proba(X_trans)
        if 1 in classes:
            idx = classes.index(1)
            return proba[:, idx]
        else:
            return np.zeros(len(X))

    def predict(self, X: pd.DataFrame, threshold: float | None = None) -> np.ndarray:
        """Predict binary class 0 or 1 using the specified decision threshold."""
        th = threshold if threshold is not None else self.decision_threshold
        probs = self.predict_proba(X)
        return (probs >= th).astype(int)

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        split_name: str = "Test",
        threshold: float | None = None,
    ) -> EvaluationMetrics:
        """Compute rigorous evaluation metrics without data contamination."""
        th = threshold if threshold is not None else self.decision_threshold
        probs = self.predict_proba(X)
        preds = (probs >= th).astype(int)

        y_true = y.values.astype(int)
        pos = int((y_true == 1).sum())
        neg = int((y_true == 0).sum())

        tn, fp, fn, tp = 0, 0, 0, 0
        if len(np.unique(y_true)) > 1 or len(preds) > 0:
            cm = confusion_matrix(y_true, preds, labels=[0, 1])
            tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

        # Safely compute metrics when classes exist
        notes_list = []
        prec, rec, f1_val, roc_val, pr_val = None, None, None, None, None

        if pos > 0:
            prec = float(precision_score(y_true, preds, zero_division=0))
            rec = float(recall_score(y_true, preds, zero_division=0))
            f1_val = float(f1_score(y_true, preds, zero_division=0))
        else:
            notes_list.append(
                f"Single-class evaluation slice: Contains {neg} legitimate and 0 laundering transactions. "
                "Precision, Recall, and F1 are mathematically undefined without positive test instances."
            )

        if pos > 0 and neg > 0:
            try:
                roc_val = float(roc_auc_score(y_true, probs))
            except Exception as e:
                notes_list.append(f"ROC-AUC computation skipped: {e}")
            try:
                pr_val = float(average_precision_score(y_true, probs))
            except Exception as e:
                notes_list.append(f"PR-AUC computation skipped: {e}")
        else:
            notes_list.append(
                "ROC-AUC and PR-AUC require at least one transaction of each class in the test window."
            )

        return EvaluationMetrics(
            split_name=split_name,
            sample_size=len(y_true),
            positive_count=pos,
            negative_count=neg,
            precision=prec,
            recall=rec,
            f1=f1_val,
            roc_auc=roc_val,
            pr_auc=pr_val,
            confusion_matrix={"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            threshold=th,
            notes="; ".join(notes_list) if notes_list else None,
        )

    def get_feature_importances(self, top_n: int = 15) -> list[dict[str, Any]]:
        """Return top feature importances from the fitted Random Forest."""
        if self.classifier_ is None or not self.feature_names_:
            return []
        importances = self.classifier_.feature_importances_
        sorted_idx = np.argsort(importances)[::-1][:top_n]
        return [
            {
                "feature": self.feature_names_[i],
                "importance": round(float(importances[i]), 6),
            }
            for i in sorted_idx
        ]

    def save(self, filepath: str | Path) -> None:
        """Persist fitted pipeline to disk."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "preprocessor": self.preprocessor_,
            "classifier": self.classifier_,
            "feature_names": self.feature_names_,
            "training_metadata": self.training_metadata_,
            "decision_threshold": self.decision_threshold,
        }
        joblib.dump(bundle, p)

    @classmethod
    def load(cls, filepath: str | Path) -> AMLRandomForestPipeline:
        """Load fitted pipeline from disk."""
        p = Path(filepath)
        if not p.is_file():
            raise FileNotFoundError(f"Model file not found: {p}")
        bundle = joblib.load(p)
        pipeline = cls(
            decision_threshold=bundle.get("decision_threshold", 0.5),
            random_state=bundle.get("training_metadata", {}).get("random_state", 42),
        )
        pipeline.preprocessor_ = bundle["preprocessor"]
        pipeline.classifier_ = bundle["classifier"]
        pipeline.feature_names_ = bundle["feature_names"]
        pipeline.training_metadata_ = bundle["training_metadata"]
        return pipeline
