#!/usr/bin/env python3
"""Reproducible experiment: fixed ID3/C4.5 + from-scratch GBDT.

Examples
--------
    python ml_exp.py
    python ml_exp.py --benchmark
    python ml_exp.py --cv-folds 5 --cv-repeats 3 --seed 7
    python ml_exp.py --no-cv --outdir results_quick

The script deliberately reports *both* a stratified holdout and a nested
cross-validation estimate.  On the 286-row UCI Ljubljana dataset the honest
accuracy is roughly 72-77%; see README.md for the ceiling analysis.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pandas_dt import (
    GBDTClassifier,
    ID3Classifier,
    accuracy_score,
    binary_metrics,
    encode_categorical,
    encode_numeric,
    encode_onehot,
    load_raw,
    save_json,
    save_model,
    stratified_kfold,
    stratified_split,
    tune_threshold,
)
from pandas_dt.data import NEGATIVE_LABEL, POSITIVE_LABEL, FeatureInfo
from pandas_dt.metrics import (
    bootstrap_accuracy_ci,
    log_loss_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="UCI breast-cancer: fixed ID3/C4.5 tree and from-scratch GBDT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", default="breast-cancer.data", help="path to the CSV file")
    parser.add_argument("--outdir", default="results", help="directory for models and reports")
    parser.add_argument("--seed", type=int, default=42, help="random seed for the holdout split")
    parser.add_argument("--test-size", type=float, default=0.2, help="fraction used for the test set")
    parser.add_argument("--val-size", type=float, default=0.2, help="fraction used for validation")
    parser.add_argument("--target", type=float, default=0.90, help="accuracy target to check")

    parser.add_argument("--no-cv", action="store_true", help="skip nested cross-validation")
    parser.add_argument("--cv-folds", type=int, default=5, help="outer folds for nested CV")
    parser.add_argument("--cv-repeats", type=int, default=3, help="repetitions of nested CV")
    parser.add_argument("--benchmark", action="store_true", help="run scikit-learn reference models")

    parser.add_argument("--id3-criterion", choices=["gain", "gain_ratio"], default="gain_ratio")
    parser.add_argument("--id3-max-depth", type=int, default=None)
    parser.add_argument("--id3-min-samples-leaf", type=int, default=2)

    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--min-samples-leaf", type=int, default=8)
    parser.add_argument("--min-child-weight", type=float, default=1.0)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--max-features", type=float, default=0.8)
    parser.add_argument("--early-stopping-rounds", type=int, default=30)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if not 0.0 < args.test_size < 1.0:
        raise SystemExit("--test-size must be in (0, 1)")
    if not 0.0 <= args.val_size < 1.0:
        raise SystemExit("--val-size must be in [0, 1)")
    if args.cv_folds < 2:
        raise SystemExit("--cv-folds must be >= 2")
    if args.cv_repeats < 1:
        raise SystemExit("--cv-repeats must be >= 1")
    return args


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _proba_metrics(y: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    metrics = binary_metrics(y, proba, threshold=threshold)
    metrics["roc_auc"] = roc_auc_score(y, proba)
    return metrics


def _fit_id3(
    X: np.ndarray,
    y: np.ndarray,
    info: FeatureInfo,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    args: argparse.Namespace,
) -> tuple[ID3Classifier, dict[str, Any]]:
    model = ID3Classifier(
        criterion=args.id3_criterion,
        max_depth=args.id3_max_depth,
        min_samples_leaf=args.id3_min_samples_leaf,
        prune=bool(len(val_idx)),
    )
    started = time.perf_counter()
    eval_set = (X[val_idx], y[val_idx]) if len(val_idx) else None
    model.fit(X[train_idx], y[train_idx], feature_info=info, eval_set=eval_set)
    elapsed = time.perf_counter() - started

    info_dict: dict[str, Any] = {
        "criterion": args.id3_criterion,
        "max_depth": args.id3_max_depth,
        "min_samples_leaf": args.id3_min_samples_leaf,
        "n_nodes": model.n_nodes,
        "n_leaves": model.n_leaves,
        "depth": model.depth,
        "pruned_nodes": model.pruned_nodes_,
        "fit_seconds": round(elapsed, 6),
    }
    return model, info_dict


def _fit_gbdt(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    args: argparse.Namespace,
    *,
    random_state: int | None = None,
) -> tuple[GBDTClassifier, dict[str, Any]]:
    params = {
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "min_child_weight": args.min_child_weight,
        "reg_lambda": args.reg_lambda,
        "subsample": args.subsample,
        "max_features": args.max_features,
        "random_state": args.seed if random_state is None else random_state,
        "early_stopping_rounds": args.early_stopping_rounds if len(val_idx) else None,
    }
    model = GBDTClassifier(**params)
    started = time.perf_counter()
    eval_set = (X[val_idx], y[val_idx]) if len(val_idx) else None
    model.fit(X[train_idx], y[train_idx], eval_set=eval_set)
    elapsed = time.perf_counter() - started
    meta = {
        **params,
        "trees_trained": len(model.trees_),
        "trees_used": model.n_used_trees,
        "best_iteration": model.best_iteration_,
        "train_loss_first": model.train_loss_[0] if model.train_loss_ else None,
        "train_loss_last": model.train_loss_[-1] if model.train_loss_ else None,
        "fit_seconds": round(elapsed, 6),
    }
    return model, meta


def _refit_gbdt(
    X: np.ndarray,
    y: np.ndarray,
    fit_idx: np.ndarray,
    n_estimators: int,
    args: argparse.Namespace,
    random_state: int | None = None,
) -> GBDTClassifier:
    """Refit on more data with the iteration count chosen on validation."""

    model = GBDTClassifier(
        n_estimators=max(1, int(n_estimators)),
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        min_child_weight=args.min_child_weight,
        reg_lambda=args.reg_lambda,
        subsample=args.subsample,
        max_features=args.max_features,
        random_state=args.seed if random_state is None else random_state,
        early_stopping_rounds=None,
    )
    model.fit(X[fit_idx], y[fit_idx])
    return model


def _nested_cv_gbdt(
    X: np.ndarray,
    y: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Leakage-free nested CV.

    The outer fold is never used to choose the number of trees or the
    decision threshold.  An inner stratified split of the outer training
    data is used for early stopping and threshold selection; the refitted
    model is then evaluated on the untouched outer fold.
    """

    accuracies: list[float] = []
    accuracies_tuned: list[float] = []
    log_losses: list[float] = []
    iterations: list[int] = []
    thresholds: list[float] = []
    started = time.perf_counter()

    for repeat in range(args.cv_repeats):
        folds = stratified_kfold(
            y, n_splits=args.cv_folds, random_state=args.seed + 1000 + repeat
        )
        for outer_train, outer_valid in folds:
            inner_train_rel, inner_val_rel, _ = stratified_split(
                y[outer_train],
                test_size=0.0,
                val_size=0.2,
                random_state=args.seed + 7919 * (repeat + 1),
            )
            inner_train = outer_train[inner_train_rel]
            inner_val = outer_train[inner_val_rel]

            selection_model, _ = _fit_gbdt(X, y, inner_train, inner_val, args)
            threshold, _ = tune_threshold(
                y[inner_val], selection_model.predict_proba(X[inner_val])[:, 1]
            )
            n_trees = max(1, selection_model.n_used_trees)

            final_model = _refit_gbdt(
                X, y, outer_train, n_trees, args, random_state=args.seed + repeat
            )
            proba = final_model.predict_proba(X[outer_valid])[:, 1]
            accuracies.append(accuracy_score(y[outer_valid], (proba >= 0.5).astype(int)))
            accuracies_tuned.append(
                accuracy_score(y[outer_valid], (proba >= threshold).astype(int))
            )
            log_losses.append(log_loss_score(y[outer_valid], proba))
            iterations.append(n_trees)
            thresholds.append(float(threshold))

    elapsed = time.perf_counter() - started
    return {
        "protocol": (
            "nested stratified CV; inner 80/20 split for early stopping and "
            "threshold selection; refit on outer train"
        ),
        "folds": args.cv_folds,
        "repeats": args.cv_repeats,
        "n_evaluations": len(accuracies),
        "accuracy": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "accuracy_tuned_threshold": float(np.mean(accuracies_tuned)),
        "accuracy_tuned_threshold_std": float(np.std(accuracies_tuned)),
        "log_loss": float(np.mean(log_losses)),
        "mean_trees": float(np.mean(iterations)),
        "mean_threshold": float(np.mean(thresholds)),
        "seconds": round(elapsed, 3),
        "fold_accuracies": [float(a) for a in accuracies],
    }


def _majority_baseline(y_train: np.ndarray, y_eval: np.ndarray) -> float:
    majority = int(np.round(float(np.mean(y_train))))
    return float(np.mean(y_eval == majority))


def _run_sklearn_benchmark(
    X: np.ndarray,
    y: np.ndarray,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Optional ceiling check with well-known scikit-learn models.

    scikit-learn is not a core dependency; if it is missing the benchmark is
    skipped gracefully.  A purely one-hot encoded matrix is used here: it is
    the strongest representation for the reference models, so the ceiling is
    not understated.
    """

    try:
        import warnings

        from sklearn.dummy import DummyClassifier
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return [{"model": "skipped", "reason": "scikit-learn is not installed"}]

    cv = RepeatedStratifiedKFold(
        n_splits=args.cv_folds, n_repeats=2, random_state=args.seed
    )
    models = {
        "DummyClassifier (majority)": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=args.seed)
        ),
        "RandomForest (500 trees)": RandomForestClassifier(
            n_estimators=500, random_state=args.seed, n_jobs=-1
        ),
        "GradientBoosting (sklearn)": GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=3,
            min_samples_leaf=8, subsample=0.8, max_features=0.8,
            random_state=args.seed,
        ),
    }
    results: list[dict[str, Any]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, model in models.items():
            scores = cross_validate(
                model, X, y, cv=cv, scoring=("accuracy", "roc_auc", "neg_log_loss"),
                n_jobs=1,
            )
            results.append(
                {
                    "model": name,
                    "accuracy_mean": float(np.mean(scores["test_accuracy"])),
                    "accuracy_std": float(np.std(scores["test_accuracy"])),
                    "roc_auc_mean": float(np.mean(scores["test_roc_auc"])),
                    "log_loss_mean": float(-np.mean(scores["test_neg_log_loss"])),
                }
            )
    return results


def _feature_importance_table(
    names: tuple[str, ...], importances: np.ndarray
) -> pd.DataFrame:
    frame = pd.DataFrame({"feature": list(names), "importance": importances})
    return frame.sort_values("importance", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(" UCI Breast Cancer (Ljubljana) - fixed ID3/C4.5 + from-scratch GBDT")
    print("=" * 78)

    raw = load_raw(data_path)
    X_cat, y, cat_info = encode_categorical(raw)
    X_num, _, num_info = encode_numeric(raw)
    X_bench, _, _ = encode_onehot(raw)

    n = len(raw)
    n_pos = int(y.sum())
    n_neg = n - n_pos
    missing_counts = {
        col: int((raw[col] == "?").sum())
        for col in raw.columns
        if (raw[col] == "?").any()
    }
    print(f"data            : {data_path.name}  ({n} rows)")
    print(f"classes         : no-recurrence={n_neg}  recurrence={n_pos}")
    print(f"missing values  : {missing_counts or 'none'}")
    print(
        f"features        : {X_cat.shape[1]} categorical (ID3) / "
        f"{X_num.shape[1]} numeric-expanded (GBDT)"
    )

    train_idx, val_idx, test_idx = stratified_split(
        y, test_size=args.test_size, val_size=args.val_size, random_state=args.seed
    )
    print(
        f"holdout split   : train={len(train_idx)} val={len(val_idx)} "
        f"test={len(test_idx)} (stratified, seed={args.seed})"
    )
    print(
        f"majority baseline (test): {_majority_baseline(y[train_idx], y[test_idx]):.4f}"
    )

    report: dict[str, Any] = {
        "dataset": {
            "path": str(data_path),
            "n_samples": n,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "missing_values": missing_counts,
            "pos_label": POSITIVE_LABEL,
            "neg_label": NEGATIVE_LABEL,
        },
        "split": {
            "seed": args.seed,
            "test_size": args.test_size,
            "val_size": args.val_size,
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
            "majority_baseline_test": _majority_baseline(y[train_idx], y[test_idx]),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "platform": platform.platform(),
        },
        "gbdt_params": {
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "min_child_weight": args.min_child_weight,
            "reg_lambda": args.reg_lambda,
            "subsample": args.subsample,
            "max_features": args.max_features,
            "early_stopping_rounds": args.early_stopping_rounds,
        },
    }

    # ------------------------------------------------------------------
    # 1. ID3 / C4.5
    # ------------------------------------------------------------------
    print("\n[1/3] ID3 / C4.5 (multi-way categorical splits, reduced-error pruning)")
    id3, id3_info = _fit_id3(X_cat, y, cat_info, train_idx, val_idx, args)
    id3_proba = {
        "train": id3.predict_proba(X_cat[train_idx])[:, 1],
        "val": id3.predict_proba(X_cat[val_idx])[:, 1] if len(val_idx) else np.empty(0),
        "test": id3.predict_proba(X_cat[test_idx])[:, 1],
    }
    id3_metrics = {
        "train": _proba_metrics(y[train_idx], id3_proba["train"]),
        "test": _proba_metrics(y[test_idx], id3_proba["test"]),
    }
    if len(val_idx):
        id3_metrics["val"] = _proba_metrics(y[val_idx], id3_proba["val"])
        id3_threshold, _ = tune_threshold(y[val_idx], id3_proba["val"])
        id3_metrics["test_tuned"] = _proba_metrics(
            y[test_idx], id3_proba["test"], threshold=id3_threshold
        )
    else:
        id3_threshold = 0.5

    print(
        f"    nodes={id3_info['n_nodes']} leaves={id3_info['n_leaves']} "
        f"depth={id3_info['depth']} pruned={id3_info['pruned_nodes']} "
        f"({id3_info['fit_seconds']:.4f}s)"
    )
    print(
        f"    train {id3_metrics['train']['accuracy']:.4f} | "
        f"val {id3_metrics.get('val', {}).get('accuracy', float('nan')):.4f} | "
        f"test {id3_metrics['test']['accuracy']:.4f} "
        f"(test@{id3_threshold:.2f}: {id3_metrics.get('test_tuned', id3_metrics['test'])['accuracy']:.4f})"
    )
    rules_path = id3.export_rules(outdir / "id3_rules.txt")
    (outdir / "id3_tree.txt").write_text(str(id3) + "\n", encoding="utf-8")
    save_model(id3, outdir / "id3_model.pkl", metadata={**id3_info, "split_seed": args.seed})

    report["id3"] = {
        "info": id3_info,
        "threshold": float(id3_threshold),
        "metrics": id3_metrics,
        "feature_importance": dict(
            zip(cat_info.names, [float(v) for v in id3.feature_importances_])
        ),
        "rules_file": rules_path.name,
    }

    # ------------------------------------------------------------------
    # 2. GBDT
    # ------------------------------------------------------------------
    print("\n[2/3] GBDT (Newton boosting, logistic loss, early stopping)")
    gbdt, gbdt_info = _fit_gbdt(X_num, y, train_idx, val_idx, args)
    gbdt_metrics = {
        "train": _proba_metrics(y[train_idx], gbdt.predict_proba(X_num[train_idx])[:, 1]),
        "test": _proba_metrics(y[test_idx], gbdt.predict_proba(X_num[test_idx])[:, 1]),
    }
    if len(val_idx):
        val_proba = gbdt.predict_proba(X_num[val_idx])[:, 1]
        gbdt_metrics["val"] = _proba_metrics(y[val_idx], val_proba)
        gbdt_threshold, _ = tune_threshold(y[val_idx], val_proba)
        gbdt_metrics["test_tuned"] = _proba_metrics(
            y[test_idx],
            gbdt.predict_proba(X_num[test_idx])[:, 1],
            threshold=gbdt_threshold,
        )
    else:
        gbdt_threshold = 0.5

    print(
        f"    trees used={gbdt_info['trees_used']}/{gbdt_info['trees_trained']} "
        f"(best iteration {gbdt_info['best_iteration']}) "
        f"({gbdt_info['fit_seconds']:.4f}s)"
    )
    print(
        f"    train {gbdt_metrics['train']['accuracy']:.4f} | "
        f"val {gbdt_metrics.get('val', {}).get('accuracy', float('nan')):.4f} | "
        f"test {gbdt_metrics['test']['accuracy']:.4f} "
        f"(test@{gbdt_threshold:.2f}: {gbdt_metrics.get('test_tuned', gbdt_metrics['test'])['accuracy']:.4f})"
    )

    # Refit on train+validation using the iteration count chosen above.
    fit_idx = np.concatenate([train_idx, val_idx])
    refit_n = max(1, gbdt.n_used_trees)
    gbdt_refit = _refit_gbdt(X_num, y, fit_idx, refit_n, args)
    refit_metrics = {
        "fit_train": _proba_metrics(
            y[fit_idx], gbdt_refit.predict_proba(X_num[fit_idx])[:, 1]
        ),
        "test": _proba_metrics(
            y[test_idx], gbdt_refit.predict_proba(X_num[test_idx])[:, 1]
        ),
    }
    if len(val_idx) and gbdt_threshold != 0.5:
        refit_metrics["test_tuned"] = _proba_metrics(
            y[test_idx],
            gbdt_refit.predict_proba(X_num[test_idx])[:, 1],
            threshold=gbdt_threshold,
        )
    print(
        f"    refit on train+val ({len(fit_idx)} rows, {refit_n} trees): "
        f"fit acc {refit_metrics['fit_train']['accuracy']:.4f} | "
        f"test {refit_metrics['test']['accuracy']:.4f}"
    )

    importance = _feature_importance_table(num_info.names, gbdt.feature_importances_)
    importance.to_csv(outdir / "gbdt_feature_importance.csv", index=False)
    save_model(
        gbdt_refit,
        outdir / "gbdt_model.pkl",
        metadata={
            "split_seed": args.seed,
            "n_trees": refit_n,
            "feature_names": list(num_info.names),
            "threshold": float(gbdt_threshold),
            "trained_on": "train+val",
        },
    )

    report["gbdt"] = {
        "info": gbdt_info,
        "threshold": float(gbdt_threshold),
        "metrics": gbdt_metrics,
        "refit": {
            "n_trees": int(refit_n),
            "n_rows": int(len(fit_idx)),
            "metrics": refit_metrics,
        },
        "top_features": importance.head(10).to_dict(orient="records"),
    }

    # Capacity check: deliberately show that >90% train accuracy is easy and
    # meaningless here.  It is not used as the reported test model.
    capacity = GBDTClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        min_child_weight=args.min_child_weight,
        reg_lambda=args.reg_lambda,
        subsample=args.subsample,
        max_features=args.max_features,
        random_state=args.seed,
        early_stopping_rounds=None,
    )
    capacity.fit(X_num[fit_idx], y[fit_idx])
    capacity_metrics = {
        "fit_train": _proba_metrics(
            y[fit_idx], capacity.predict_proba(X_num[fit_idx])[:, 1]
        ),
        "test": _proba_metrics(
            y[test_idx], capacity.predict_proba(X_num[test_idx])[:, 1]
        ),
    }
    report["gbdt"]["capacity_check"] = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "note": "in-sample capacity demonstration; not the recommended model",
        "metrics": capacity_metrics,
    }
    print(
        f"    capacity check (300 trees, no early stop): "
        f"train {capacity_metrics['fit_train']['accuracy']:.4f} | "
        f"test {capacity_metrics['test']['accuracy']:.4f}  "
        f"<- train>90% is memorization, not generalization"
    )

    # ------------------------------------------------------------------
    # 3. Nested CV
    # ------------------------------------------------------------------
    cv_report: dict[str, Any] | None = None
    if not args.no_cv:
        print(
            f"\n[3/3] Nested {args.cv_folds}-fold CV x{args.cv_repeats} "
            "(leakage-free: early stopping/threshold use only inner folds)"
        )
        cv_report = _nested_cv_gbdt(X_num, y, args)
        print(
            f"    accuracy@0.5      : {cv_report['accuracy']:.4f} "
            f"± {cv_report['accuracy_std']:.4f}"
        )
        print(
            f"    accuracy@tuned    : {cv_report['accuracy_tuned_threshold']:.4f} "
            f"± {cv_report['accuracy_tuned_threshold_std']:.4f}"
        )
        print(
            f"    mean trees={cv_report['mean_trees']:.1f} "
            f"mean threshold={cv_report['mean_threshold']:.3f} "
            f"({cv_report['seconds']:.1f}s)"
        )
        report["nested_cv"] = cv_report

    # ------------------------------------------------------------------
    # Optional sklearn ceiling check
    # ------------------------------------------------------------------
    benchmark_report: list[dict[str, Any]] | None = None
    if args.benchmark:
        print("\n[benchmark] Reference models (scikit-learn, repeated stratified CV)")
        benchmark_report = _run_sklearn_benchmark(X_bench, y, args)
        for row in benchmark_report:
            if "accuracy_mean" in row:
                print(
                    f"    {row['model']:<32s} acc={row['accuracy_mean']:.4f} "
                    f"± {row['accuracy_std']:.4f}  auc={row['roc_auc_mean']:.4f}  "
                    f"logloss={row['log_loss_mean']:.4f}"
                )
            else:
                print(f"    {row.get('model')}: {row.get('reason')}")
        report["benchmark"] = benchmark_report

    # ------------------------------------------------------------------
    # Target analysis
    # ------------------------------------------------------------------
    test_accuracy = float(refit_metrics["test"]["accuracy"])
    test_accuracy_tuned = float(
        refit_metrics.get("test_tuned", refit_metrics["test"])["accuracy"]
    )
    cv_accuracy = (
        float(cv_report["accuracy"]) if cv_report is not None else float("nan")
    )
    train_accuracy = float(refit_metrics["fit_train"]["accuracy"])
    early_stop_test_accuracy = float(gbdt_metrics["test"]["accuracy"])
    capacity_train_accuracy = float(capacity_metrics["fit_train"]["accuracy"])
    capacity_test_accuracy = float(capacity_metrics["test"]["accuracy"])
    target_met_holdout = test_accuracy >= args.target
    target_met_cv = (not np.isnan(cv_accuracy)) and cv_accuracy >= args.target
    target_met_train = capacity_train_accuracy >= args.target

    lo, hi = bootstrap_accuracy_ci(
        y[test_idx],
        (gbdt_refit.predict_proba(X_num[test_idx])[:, 1] >= 0.5).astype(int),
        n_bootstrap=2000,
        random_state=args.seed,
    )
    benchmark_best = None
    if benchmark_report:
        scored = [r for r in benchmark_report if "accuracy_mean" in r]
        if scored:
            benchmark_best = max(scored, key=lambda r: r["accuracy_mean"])

    report["target"] = {
        "value": float(args.target),
        "train_accuracy": train_accuracy,
        "holdout_accuracy_early_stop": early_stop_test_accuracy,
        "holdout_accuracy": test_accuracy,
        "holdout_accuracy_tuned": test_accuracy_tuned,
        "holdout_accuracy_95ci": [lo, hi],
        "nested_cv_accuracy": cv_accuracy,
        "capacity_check_train_accuracy": capacity_train_accuracy,
        "capacity_check_test_accuracy": capacity_test_accuracy,
        "met_on_holdout": bool(target_met_holdout),
        "met_on_nested_cv": bool(target_met_cv),
        "met_on_train": bool(target_met_train),
        "best_sklearn_cv_accuracy": (
            float(benchmark_best["accuracy_mean"]) if benchmark_best else None
        ),
    }

    print("\n" + "-" * 78)
    print(f"SUMMARY  (target: {args.target:.0%} accuracy)")
    print("-" * 78)
    print(
        "  Majority-class baseline (holdout) : "
        f"{_majority_baseline(y[train_idx], y[test_idx]):.4f}"
    )
    print(f"  ID3/C4.5 holdout accuracy         : {id3_metrics['test']['accuracy']:.4f}")
    print(f"  GBDT holdout (early-stopped)      : {early_stop_test_accuracy:.4f}")
    print(f"  GBDT holdout (refit train+val)    : {test_accuracy:.4f}")
    print(f"  GBDT holdout (tuned threshold)    : {test_accuracy_tuned:.4f}")
    if cv_report is not None:
        print(f"  GBDT nested-CV accuracy           : {cv_accuracy:.4f}")
    print(f"  GBDT training accuracy            : {train_accuracy:.4f}")
    print(
        f"  High-capacity train/test          : "
        f"{capacity_train_accuracy:.4f} / {capacity_test_accuracy:.4f}"
    )
    print(f"  Test 95% bootstrap CI             : [{lo:.3f}, {hi:.3f}]")
    if benchmark_best is not None:
        print(
            f"  Best sklearn CV ({benchmark_best['model']}): "
            f"{benchmark_best['accuracy_mean']:.4f}"
        )
    if target_met_holdout or target_met_cv:
        print(f"  RESULT: {args.target:.0%} target met on held-out data.")
    else:
        print(f"  RESULT: {args.target:.0%} target NOT met on held-out data.")
        if target_met_train:
            print(
                f"          A 300-tree variant reaches {capacity_train_accuracy:.1%} "
                "on its training rows,\n          but its test accuracy is "
                f"{capacity_test_accuracy:.1%}: that is memorization, not generalization."
            )
        print(
            "          This 286-row dataset has strong class overlap; the "
            "leakage-free ceiling\n          is roughly 0.72-0.78. See README.md "
            "for the evidence and the Wisconsin\n          breast-cancer demo "
            "(same GBDT, 95%+ test accuracy)."
        )

    save_json(report, outdir / "metrics.json")
    print(f"\nArtifacts written to: {outdir}")
    print("  metrics.json, id3_rules.txt, id3_tree.txt, id3_model.pkl,")
    print("  gbdt_model.pkl, gbdt_feature_importance.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
