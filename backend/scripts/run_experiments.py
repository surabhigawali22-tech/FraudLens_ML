"""Run AML Experiments (Phases 2-5) on the IBM AML HI-Small Dataset.

CLI tool allowing reproducible, bounded development experiments and model evaluation.
Example PowerShell runs from FraudLens/backend:
    python scripts/run_experiments.py --algorithm apriori --max-rows 20000
    python scripts/run_experiments.py --algorithm adaptive --max-rows 20000
    python scripts/run_experiments.py --algorithm huim --max-rows 20000
    python scripts/run_experiments.py --algorithm rf --max-rows 50000
    python scripts/run_experiments.py --algorithm all --max-rows 20000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from app.algorithms.adaptive_apriori import AdaptiveAprioriMiner
from app.algorithms.apriori import AprioriMiner
from app.algorithms.random_forest import AMLRandomForestPipeline
from app.algorithms.utility_mining import TwoPhaseHUIMiner
from app.core.config import ARTIFACTS_DIR, OUTPUTS_DIR, get_dataset_path
from app.data import loader
from app.data.features import build_association_baskets, build_rf_feature_matrix


def run_standard_apriori(df: pd.DataFrame, min_support: float = 0.05, min_confidence: float = 0.5) -> dict:
    print(f"\n[PHASE 2] Running Standard Apriori (N={len(df)}, min_support={min_support}, min_confidence={min_confidence})...")
    start = time.time()
    baskets, _ = build_association_baskets(df, include_retrospective_target=True)
    miner = AprioriMiner(min_support=min_support, min_confidence=min_confidence, max_len=4)
    miner.fit(baskets)
    elapsed = time.time() - start

    itemsets = miner.get_frequent_itemsets()
    rules = miner.get_rules()
    print(f"  Discovered {len(itemsets)} frequent itemsets and {len(rules)} rules in {elapsed:.2f}s.")
    if rules:
        print("  Top 3 Association Rules by Lift:")
        for r in rules[:3]:
            print(f"    {' + '.join(r['antecedent'])} -> {' + '.join(r['consequent'])} "
                  f"(supp={r['support']:.4f}, conf={r['confidence']:.4f}, lift={r['lift']:.2f})")

    return {
        "algorithm": "Standard Apriori",
        "sample_size": len(df),
        "runtime_seconds": round(elapsed, 4),
        "min_support": min_support,
        "min_confidence": min_confidence,
        "frequent_itemsets_count": len(itemsets),
        "rules_count": len(rules),
        "top_rules": rules[:10],
    }


def run_adaptive_apriori(df: pd.DataFrame, base_min_support: float = 0.10) -> dict:
    print(f"\n[PHASE 3] Running Adaptive Apriori (MSApriori, N={len(df)}, base_min_support={base_min_support})...")
    start = time.time()
    baskets, _ = build_association_baskets(df, include_retrospective_target=True)

    # Assign lower MIS to rare suspicious targets / payment formats
    domain_mis = {
        "Laundering:True": 0.0005,
        "Format:Bitcoin": 0.005,
        "Currency:CrossCurrency": 0.02,
        "AmountBand:VeryHigh(>50k)": 0.02,
    }
    miner = AdaptiveAprioriMiner(
        base_min_support=base_min_support,
        min_confidence=0.4,
        mis_dict=domain_mis,
        max_len=4,
    )
    miner.fit(baskets)
    elapsed = time.time() - start

    itemsets = miner.get_frequent_itemsets()
    rules = miner.get_rules()
    print(f"  Discovered {len(itemsets)} adaptive frequent itemsets and {len(rules)} rules in {elapsed:.2f}s.")
    if rules:
        print("  Top 3 Adaptive Rules by Lift:")
        for r in rules[:3]:
            print(f"    {' + '.join(r['antecedent'])} -> {' + '.join(r['consequent'])} "
                  f"(supp={r['support']:.4f}, conf={r['confidence']:.4f}, lift={r['lift']:.2f})")

    return {
        "algorithm": "Adaptive Apriori (MSApriori)",
        "sample_size": len(df),
        "runtime_seconds": round(elapsed, 4),
        "base_min_support": base_min_support,
        "domain_mis": domain_mis,
        "frequent_itemsets_count": len(itemsets),
        "rules_count": len(rules),
        "top_rules": rules[:10],
    }


def run_huim(df: pd.DataFrame, min_utility: float | None = None) -> dict:
    print(f"\n[PHASE 4] Running Genuine Two-Phase HUIM with TWU Pruning (N={len(df)})...")
    start = time.time()
    baskets, amounts = build_association_baskets(df, include_retrospective_target=False)
    total_db_utility = sum(amounts)

    # If min_utility not specified, default to 5% of total transaction monetary volume
    th = min_utility if min_utility is not None else 0.05 * total_db_utility

    miner = TwoPhaseHUIMiner(min_utility=th, max_len=3)
    miner.fit(baskets, quantities=amounts)
    elapsed = time.time() - start

    huis = miner.get_high_utility_itemsets()
    print(f"  Total Transaction Volume Utility: ${total_db_utility:,.2f}")
    print(f"  min_utility threshold: ${th:,.2f}")
    print(f"  Discovered {len(huis)} High-Utility Itemsets in {elapsed:.2f}s.")
    if huis:
        print("  Top 3 High-Utility Itemsets by Utility:")
        for h in huis[:3]:
            print(f"    {' + '.join(h['items'])}: Utility=${h['utility']:,.2f} (TWU=${h['twu']:,.2f}, count={h['support_count']})")

    return {
        "algorithm": "Two-Phase High-Utility Itemset Mining (HUIM)",
        "sample_size": len(df),
        "total_volume_utility": round(total_db_utility, 2),
        "min_utility_threshold": round(th, 2),
        "runtime_seconds": round(elapsed, 4),
        "high_utility_itemsets_count": len(huis),
        "top_itemsets": huis[:10],
    }


def run_random_forest(df: pd.DataFrame, n_estimators: int = 100, threshold: float = 0.4) -> dict:
    print(f"\n[PHASE 5] Training AML Random Forest Classifier (N={len(df)}, n_estimators={n_estimators}, threshold={threshold})...")
    start = time.time()

    # Chronological train/val/test split (70% / 15% / 15%)
    train_df, val_df, test_df = AMLRandomForestPipeline.chronological_split(df, train_ratio=0.70, val_ratio=0.15)
    print(f"  Chronological Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    X_train, y_train = build_rf_feature_matrix(train_df)
    X_val, y_val = build_rf_feature_matrix(val_df)
    X_test, y_test = build_rf_feature_matrix(test_df)

    pos_train = int((y_train == 1).sum())
    pos_test = int((y_test == 1).sum())
    print(f"  Train Class Balance: Legitimate={len(y_train) - pos_train}, Laundering={pos_train} ({100*pos_train/len(y_train):.3f}%)")
    print(f"  Test Class Balance : Legitimate={len(y_test) - pos_test}, Laundering={pos_test} ({100*pos_test/len(y_test):.3f}%)")

    pipeline = AMLRandomForestPipeline(
        n_estimators=n_estimators,
        max_depth=15,
        class_weight="balanced",
        decision_threshold=threshold,
        random_state=42,
    )
    pipeline.fit(X_train, y_train)

    val_metrics = pipeline.evaluate(X_val, y_val, split_name="Validation", threshold=threshold)
    test_metrics = pipeline.evaluate(X_test, y_test, split_name="Test (Out-of-Time)", threshold=threshold)
    elapsed = time.time() - start

    # Save model artifact
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACTS_DIR / "random_forest_model.joblib"
    pipeline.save(model_path)

    print(f"  Model saved to: {model_path}")
    print(f"  Test Evaluation (Threshold={threshold}):")
    print(f"    Precision : {test_metrics.precision}")
    print(f"    Recall    : {test_metrics.recall}")
    print(f"    F1-Score  : {test_metrics.f1}")
    print(f"    ROC-AUC   : {test_metrics.roc_auc}")
    print(f"    PR-AUC    : {test_metrics.pr_auc}")
    print(f"    Confusion Matrix: {test_metrics.confusion_matrix}")

    top_features = pipeline.get_feature_importances(top_n=10)
    print("  Top 5 Predictive Features:")
    for f in top_features[:5]:
        print(f"    - {f['feature']}: {f['importance']:.4f}")

    return {
        "algorithm": "Random Forest Classifier",
        "sample_size": len(df),
        "runtime_seconds": round(elapsed, 4),
        "split_strategy": "Chronological (70% train, 15% val, 15% test)",
        "train_metrics": pipeline.training_metadata_,
        "validation_metrics": val_metrics.to_dict(),
        "test_metrics": test_metrics.to_dict(),
        "top_features": top_features,
        "saved_model_path": str(model_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run FraudLens ML and Mining Experiments on HI-Small.")
    parser.add_argument("--algorithm", choices=["apriori", "adaptive", "huim", "rf", "all"], default="all")
    parser.add_argument("--max-rows", type=int, default=25000, help="Number of rows to load (bounded development run)")
    parser.add_argument("--path", default=None, help="Path to HI-Small_Trans.csv")
    args = parser.parse_args()

    csv_path = loader.resolve_path(args.path)
    print(f"Loading dataset from: {csv_path} (reading {args.max_rows:,} rows)...")
    start = time.time()
    # Read bounded sample in chronological order
    df = next(loader.iter_chunks(csv_path, chunksize=args.max_rows, max_rows=args.max_rows))
    print(f"Loaded {len(df):,} rows in {time.time() - start:.2f}s.")

    results: dict[str, Any] = {"dataset": str(csv_path), "rows": len(df)}

    if args.algorithm in ("apriori", "all"):
        results["apriori"] = run_standard_apriori(df)
    if args.algorithm in ("adaptive", "all"):
        results["adaptive_apriori"] = run_adaptive_apriori(df)
    if args.algorithm in ("huim", "all"):
        results["huim"] = run_huim(df)
    if args.algorithm in ("rf", "all"):
        results["random_forest"] = run_random_forest(df)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUTS_DIR / f"experiment_results_{args.algorithm}.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[DONE] Full experiment results saved to: {out_file}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
