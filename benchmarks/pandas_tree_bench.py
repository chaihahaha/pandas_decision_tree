#!/usr/bin/env python3
"""Benchmark pandas-first decision trees.

Compares three levels:

1. the split-gain kernel (the original per-value filtering style vs a single
   ``groupby`` vs a compact integer ``bincount``);
2. full tree fitting (``PandasID3Classifier`` / ``GroupbyID3Classifier`` /
   the integer ``ID3Classifier`` core);
3. scaling to 1k / 5k / 20k synthetic rows.

Run:
    python benchmarks/pandas_tree_bench.py
    python benchmarks/pandas_tree_bench.py --reps 100
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pandas_dt import (  # noqa: E402
    GroupbyID3Classifier,
    ID3Classifier,
    PandasID3Classifier,
    load_raw,
)
from pandas_dt.data import encode_categorical, stratified_split  # noqa: E402

TARGET = "Class"
FEATURES = [
    "age",
    "menopause",
    "tumor-size",
    "inv-nodes",
    "node-caps",
    "deg-malig",
    "breast",
    "breast-quad",
    "irradiat",
]


def entropy_counts(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def gain_naive_filter(frame: pd.DataFrame, attribute: str) -> float:
    """Original-project style: filter the DataFrame once per feature value."""

    target = frame[TARGET]
    total = len(frame)
    parent = entropy_counts(target.value_counts().to_numpy(dtype=float))
    weighted = 0.0
    for value in frame[attribute].unique():
        subset = frame[frame[attribute] == value]
        if len(subset) == 0:
            continue
        weighted += (
            len(subset) / total
        ) * entropy_counts(subset[TARGET].value_counts().to_numpy(dtype=float))
    return parent - weighted


def gain_groupby(frame: pd.DataFrame, attribute: str) -> float:
    """One groupby -> contingency table; no per-value frame copies."""

    table = (
        frame.groupby([attribute, TARGET], observed=True, sort=False, dropna=False)
        .size()
        .unstack(fill_value=0)
    )
    matrix = table.to_numpy(dtype=float)
    row_counts = matrix.sum(axis=1)
    total = matrix.sum()
    parent = entropy_counts(matrix.sum(axis=0))
    weighted = 0.0
    for i in range(len(row_counts)):
        if row_counts[i]:
            weighted += (row_counts[i] / total) * entropy_counts(matrix[i])
    return parent - weighted


def gain_bincount(codes: np.ndarray, y: np.ndarray, cardinality: int) -> float:
    joint = np.bincount(
        codes.astype(np.int64) * 2 + y, minlength=2 * cardinality
    ).reshape(cardinality, 2)
    total = joint.sum()
    parent = entropy_counts(joint.sum(axis=0))
    weighted = 0.0
    for i in range(cardinality):
        rows = joint[i].sum()
        if rows:
            weighted += (rows / total) * entropy_counts(joint[i])
    return parent - weighted


def time_per_call(fn, *args, reps: int) -> float:
    fn(*args)
    gc.collect()
    started = time.perf_counter()
    for _ in range(reps):
        fn(*args)
    return (time.perf_counter() - started) / reps * 1000.0


def peak_kib(fn, *args) -> float:
    gc.collect()
    tracemalloc.start()
    fn(*args)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024.0


def make_synthetic(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {
        column: pd.Categorical(rng.integers(0, 8, size=n).astype(str))
        for column in FEATURES
    }
    score = (
        data["deg-malig"].codes * 0.35
        + data["inv-nodes"].codes * 0.25
        + rng.normal(size=n)
    )
    data[TARGET] = pd.Categorical(
        np.where(score > np.median(score), "recurrence-events", "no-recurrence-events")
    )
    return pd.DataFrame(data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--data", default=str(ROOT / "breast-cancer.data"))
    args = parser.parse_args(argv)

    raw = load_raw(args.data)
    X_all, y_all, info = encode_categorical(raw)
    train_idx, val_idx, test_idx = stratified_split(
        y_all, test_size=0.2, val_size=0.2, random_state=42
    )
    train = raw.iloc[train_idx].reset_index(drop=True)
    val = raw.iloc[val_idx].reset_index(drop=True)
    test = raw.iloc[test_idx].reset_index(drop=True)
    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]
    X_test, y_test = X_all[test_idx], y_all[test_idx]

    # ------------------------------------------------------------------
    # 1. Gain kernel
    # ------------------------------------------------------------------
    print("=" * 78)
    print("1. split-gain kernel (all 9 attributes, one call = one node)")
    print("=" * 78)
    codes = {column: X_train[:, j] for j, column in enumerate(FEATURES)}
    cardinalities = {column: info.cardinalities[j] for j, column in enumerate(FEATURES)}

    naive_vals = [gain_naive_filter(train, c) for c in FEATURES]
    groupby_vals = [gain_groupby(train, c) for c in FEATURES]
    bincount_vals = [
        gain_bincount(codes[c], y_train, cardinalities[c]) for c in FEATURES
    ]
    np.testing.assert_allclose(naive_vals, groupby_vals, atol=1e-12)
    np.testing.assert_allclose(naive_vals, bincount_vals, atol=1e-12)

    methods = {
        "original filter (df[df[col]==v])": lambda: [
            gain_naive_filter(train, c) for c in FEATURES
        ],
        "pandas groupby": lambda: [gain_groupby(train, c) for c in FEATURES],
        "integer bincount": lambda: [
            gain_bincount(codes[c], y_train, cardinalities[c]) for c in FEATURES
        ],
    }
    print(f"{'implementation':<34s} {'ms/call':>9s} {'peak KiB':>10s}")
    print("-" * 58)
    times = {}
    for name, fn in methods.items():
        times[name] = time_per_call(fn, reps=args.reps)
        print(f"{name:<34s} {times[name]:>9.3f} {peak_kib(fn):>10.1f}")
    base = times["original filter (df[df[col]==v])"]
    print(
        f"\ngroupby speedup: {base / times['pandas groupby']:.2f}x   "
        f"bincount speedup: {base / times['integer bincount']:.2f}x"
    )

    # ------------------------------------------------------------------
    # 2. Full tree fit (strings in, pandas Series out)
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("2. full tree fit on breast-cancer.data (build + prune on validation)")
    print("=" * 78)
    implementations = {
        "PandasID3Classifier (pandas API)": lambda: PandasID3Classifier(
            target=TARGET, min_samples_leaf=2
        ).fit(train, eval_set=val),
        "GroupbyID3Classifier (pure pandas)": lambda: GroupbyID3Classifier(
            target=TARGET, min_samples_leaf=2
        ).fit(train, eval_set=val),
        "ID3Classifier (integer codes)": lambda: ID3Classifier(
            criterion="gain_ratio", min_samples_leaf=2
        ).fit(
            X_train,
            y_train,
            feature_info=info,
            eval_set=(X_val, y_val),
        ),
    }
    print(f"{'implementation':<38s} {'fit ms':>8s} {'nodes':>6s} {'test acc':>9s}")
    print("-" * 66)
    for name, builder in implementations.items():
        started = time.perf_counter()
        model = builder()
        fit_ms = (time.perf_counter() - started) * 1000.0
        if isinstance(model, ID3Classifier):
            pred = model.predict(X_test)
            accuracy = float(np.mean(pred == y_test))
        else:
            pred = model.predict(test)
            accuracy = float(np.mean(pred.to_numpy() == test[TARGET].to_numpy()))
        print(
            f"{name:<38s} {fit_ms:>8.2f} {model.n_nodes:>6d} {accuracy:>9.4f}"
        )

    # ------------------------------------------------------------------
    # 3. Scaling
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("3. scaling: build time (max_depth=5, min_samples_leaf=5)")
    print("=" * 78)
    print(f"{'rows':>7s} {'PandasID3 ms':>14s} {'GroupbyID3 ms':>15s} {'speedup':>9s}")
    print("-" * 52)
    for n in (1_000, 5_000, 20_000):
        frame = make_synthetic(n)
        started = time.perf_counter()
        PandasID3Classifier(
            target=TARGET, max_depth=5, min_samples_leaf=5
        ).fit(frame)
        t_api = (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        GroupbyID3Classifier(
            target=TARGET, max_depth=5, min_samples_leaf=5
        ).fit(frame)
        t_groupby = (time.perf_counter() - started) * 1000.0
        print(
            f"{n:>7d} {t_api:>14.1f} {t_groupby:>15.1f} "
            f"{t_groupby / t_api:>8.1f}x"
        )

    print("\nNotes:")
    print("- The gain kernels are numerically equivalent; only the access pattern differs.")
    print("- PandasID3Classifier keeps the DataFrame/string API but encodes once and")
    print("  runs the split search on compact integer codes.")
    print("- GroupbyID3Classifier is fully pandas-native: one groupby per feature per")
    print("  node, no manual encoding; it is a good reference, not the fastest option.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
