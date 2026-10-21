#!/usr/bin/env python3
"""Optional demo: the same GBDT on the Wisconsin diagnostic dataset.

**This is not the dataset in the repository.**  The repository ships the UCI
Ljubljana *recurrence* dataset (286 rows, 9 categorical attributes), where the
leakage-free ceiling is roughly 72-78%.  The scikit-learn Wisconsin dataset
(569 rows, 30 continuous features) is a much easier problem and is used here
only to demonstrate that the from-scratch GBDT implementation is correct and
capable of >90% test accuracy when the data supports it.

Run:
    python wisconsin_demo.py
    python wisconsin_demo.py --seed 7 --no-cv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from pandas_dt import (
    GBDTClassifier,
    binary_metrics,
    save_json,
    save_model,
    stratified_kfold,
    stratified_split,
)
from pandas_dt.metrics import bootstrap_accuracy_ci, roc_auc_score

ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GBDT demo on the sklearn Wisconsin breast-cancer dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default="results")
    parser.add_argument("--no-cv", action="store_true")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--max-features", type=float, default=0.8)
    parser.add_argument("--early-stopping-rounds", type=int, default=30)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        from sklearn.datasets import load_breast_cancer
    except ImportError:
        print(
            "scikit-learn is required for this optional demo: "
            "python -m pip install scikit-learn",
            file=sys.stderr,
        )
        return 1

    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    dataset = load_breast_cancer()
    X = np.asarray(dataset.data, dtype=np.float32)
    y = np.asarray(dataset.target, dtype=np.int8)
    feature_names = list(dataset.feature_names)

    print("=" * 78)
    print(" Wisconsin Diagnostic Breast Cancer (sklearn) - from-scratch GBDT")
    print("=" * 78)
    print("NOTE: this is a different, easier dataset than breast-cancer.data.")
    print(
        f"rows={len(y)}  features={X.shape[1]}  "
        f"malignant={(y == 0).sum()}  benign={(y == 1).sum()}"
    )
    print("binary label: 0=malignant, 1=benign; precision/recall below are for benign")

    train_idx, val_idx, test_idx = stratified_split(
        y, test_size=0.2, val_size=0.2, random_state=args.seed
    )
    print(
        f"split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} "
        f"(stratified, seed={args.seed})"
    )

    params = dict(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        reg_lambda=args.reg_lambda,
        subsample=args.subsample,
        max_features=args.max_features,
        random_state=args.seed,
    )

    started = time.perf_counter()
    selection = GBDTClassifier(
        **params, early_stopping_rounds=args.early_stopping_rounds
    )
    selection.fit(X[train_idx], y[train_idx], eval_set=(X[val_idx], y[val_idx]))
    n_trees = max(1, selection.n_used_trees)

    final = GBDTClassifier(**{**params, "n_estimators": n_trees}, early_stopping_rounds=None)
    fit_idx = np.concatenate([train_idx, val_idx])
    final.fit(X[fit_idx], y[fit_idx])
    elapsed = time.perf_counter() - started

    proba_test = final.predict_proba(X[test_idx])[:, 1]
    metrics = binary_metrics(y[test_idx], proba_test)
    metrics["roc_auc"] = roc_auc_score(y[test_idx], proba_test)
    metrics["train_accuracy"] = float(
        np.mean((final.predict_proba(X[fit_idx])[:, 1] >= 0.5).astype(int) == y[fit_idx])
    )
    lo, hi = bootstrap_accuracy_ci(
        y[test_idx], (proba_test >= 0.5).astype(int), n_bootstrap=2000,
        random_state=args.seed,
    )

    print(f"GBDT: {n_trees} trees selected, refit on {len(fit_idx)} rows "
          f"({elapsed:.2f}s)")
    print(f"  train accuracy : {metrics['train_accuracy']:.4f}")
    print(f"  test accuracy  : {metrics['accuracy']:.4f}  "
          f"95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"  precision={metrics['precision']:.4f}  recall={metrics['recall']:.4f}  "
          f"f1={metrics['f1']:.4f}")
    print(f"  ROC AUC={metrics['roc_auc']:.4f}  logloss={metrics['log_loss']:.4f}")
    print(f"  confusion matrix (rows=true, cols=pred): {metrics['confusion_matrix']}")

    cv_report = None
    if not args.no_cv:
        print("\n5-fold CV (fixed tree count, no validation leakage):")
        fold_accuracies: list[float] = []
        started = time.perf_counter()
        for train_cv, test_cv in stratified_kfold(y, n_splits=5, random_state=args.seed):
            fold_model = GBDTClassifier(
                **{**params, "n_estimators": n_trees}, early_stopping_rounds=None
            )
            fold_model.fit(X[train_cv], y[train_cv])
            fold_pred = (fold_model.predict_proba(X[test_cv])[:, 1] >= 0.5).astype(int)
            fold_accuracies.append(float(np.mean(fold_pred == y[test_cv])))
        cv_report = {
            "folds": 5,
            "accuracy": float(np.mean(fold_accuracies)),
            "accuracy_std": float(np.std(fold_accuracies)),
            "fold_accuracies": fold_accuracies,
            "seconds": round(time.perf_counter() - started, 3),
        }
        print(
            f"  accuracy={cv_report['accuracy']:.4f} "
            f"± {cv_report['accuracy_std']:.4f}  "
            f"({cv_report['seconds']:.2f}s)"
        )

    target = 0.90
    print("\n" + "-" * 78)
    print(f"90% target: {'MET' if metrics['accuracy'] >= target else 'NOT MET'} "
          f"on the held-out Wisconsin test set")
    print("Reminder: this demo is not a result on breast-cancer.data.")
    print("-" * 78)

    report = {
        "dataset": "sklearn.datasets.load_breast_cancer (Wisconsin Diagnostic)",
        "note": "different dataset from breast-cancer.data",
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "split": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
            "seed": args.seed,
        },
        "params": params,
        "trees_used": int(n_trees),
        "test_metrics": metrics,
        "test_accuracy_95ci": [lo, hi],
        "cv": cv_report,
        "target_met": bool(metrics["accuracy"] >= target),
    }
    save_json(report, outdir / "wisconsin_metrics.json")
    save_model(
        final,
        outdir / "wisconsin_gbdt_model.pkl",
        metadata={"feature_names": feature_names, "threshold": 0.5},
    )
    print(f"\nArtifacts written to: {outdir}/wisconsin_*.json|pkl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
