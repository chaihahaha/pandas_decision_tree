#!/usr/bin/env python3
"""Microbenchmark: pandas vs NumPy for tree-learning hot loops.

The point is *not* that pandas is bad.  pandas is excellent for loading,
joining, grouping and exploratory work.  But a DataFrame is a higher-level
container: it stores NumPy arrays plus an index, a block manager and dtype
metadata.  Vectorized pandas operations eventually call NumPy, while adding
index alignment, temporary Series/Index objects and copies.  For a GBDT the
hot loop is "sort one feature, take prefix sums, evaluate split gains", which
maps directly to contiguous NumPy arrays.

Run:
    python benchmarks/pandas_vs_numpy.py --reps 300
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

from pandas_dt import load_raw  # noqa: E402
from pandas_dt.data import encode_categorical, encode_numeric  # noqa: E402


def best_split_numpy(X: np.ndarray, g: np.ndarray, h: np.ndarray, lam: float = 1.0):
    return _best_split_columns((X[:, j] for j in range(X.shape[1])), g, h, lam)


def best_split_pandas_native(
    df: pd.DataFrame, g: pd.Series, h: pd.Series, lam: float = 1.0
):
    """Pandas-native version: Series.argsort / Series.iloc / Series.cumsum."""

    n_features = df.shape[1]
    G, H = float(g.sum()), float(h.sum())
    base = G * G / (H + lam)
    best = (-1, 0.0, 0.0)
    for j in range(n_features):
        s = df.iloc[:, j]
        order = s.argsort(kind="mergesort")
        xs = s.iloc[order]
        gs = g.iloc[order].cumsum()
        hs = h.iloc[order].cumsum()
        x_values = xs.to_numpy()
        valid = np.flatnonzero(x_values[:-1] < x_values[1:])
        if valid.size == 0:
            continue
        GL, HL = gs.iloc[valid], hs.iloc[valid]
        GR, HR = G - GL, H - HL
        gains = GL * GL / (HL + lam) + GR * GR / (HR + lam) - base
        k = int(gains.to_numpy().argmax())
        if float(gains.iloc[k]) > best[1]:
            best = (
                j,
                float(gains.iloc[k]),
                float((x_values[valid[k]] + x_values[valid[k] + 1]) / 2.0),
            )
    return best


def best_split_pandas_container(
    df: pd.DataFrame, g: np.ndarray, h: np.ndarray, lam: float = 1.0
):
    """Use a DataFrame only as a container; run the NumPy kernel on its columns."""

    return _best_split_columns(
        (df.iloc[:, j].to_numpy() for j in range(df.shape[1])), g, h, lam
    )


def _best_split_columns(columns, g: np.ndarray, h: np.ndarray, lam: float):
    G, H = float(g.sum()), float(h.sum())
    base = G * G / (H + lam)
    best = (-1, 0.0, 0.0)
    for j, x in enumerate(columns):
        order = np.argsort(x, kind="mergesort")
        xs = x[order]
        gs = np.cumsum(g[order])
        hs = np.cumsum(h[order])
        valid = np.flatnonzero(xs[:-1] < xs[1:])
        if valid.size == 0:
            continue
        GL, HL = gs[valid], hs[valid]
        GR, HR = G - GL, H - HL
        gains = GL * GL / (HL + lam) + GR * GR / (HR + lam) - base
        k = int(np.argmax(gains))
        if float(gains[k]) > best[1]:
            best = (
                j,
                float(gains[k]),
                float((xs[valid[k]] + xs[valid[k] + 1]) / 2.0),
            )
    return best


def gain_pandas_filtering(frame: pd.DataFrame, target: str, attribute: str) -> float:
    """Original-project style: boolean-filter the DataFrame once per value."""

    total = len(frame)
    parent = 0.0
    for cls in frame[target].unique():
        p = (frame[target] == cls).sum() / total
        if p:
            parent -= p * np.log2(p)
    gain = parent
    for value in frame[attribute].unique():
        subset = frame[frame[attribute] == value]
        entropy = 0.0
        for cls in subset[target].unique():
            p = (subset[target] == cls).sum() / len(subset)
            if p:
                entropy -= p * np.log2(p)
        gain -= len(subset) / total * entropy
    return float(gain)


def gain_numpy_bincount(codes: np.ndarray, y: np.ndarray, cardinality: int) -> float:
    """One contingency table instead of repeated DataFrame filtering."""

    joint = np.bincount(
        codes.astype(np.int64) * 2 + y, minlength=2 * cardinality
    ).reshape(cardinality, 2)

    def entropy(counts: np.ndarray) -> float:
        total = counts.sum()
        if total == 0:
            return 0.0
        p = counts[counts > 0] / total
        return float(-(p * np.log2(p)).sum())

    total = joint.sum()
    weighted = sum(
        (joint[v].sum() / total) * entropy(joint[v])
        for v in range(cardinality)
        if joint[v].sum() > 0
    )
    return float(entropy(joint.sum(axis=0)) - weighted)


def time_per_call(fn, *args, reps: int) -> float:
    fn(*args)  # warmup / JIT / caches
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(ROOT / "breast-cancer.data"))
    parser.add_argument("--reps", type=int, default=300)
    args = parser.parse_args(argv)

    raw = load_raw(args.data)
    X32, y, info = encode_numeric(raw)
    # float64 for both backends so dtype promotion is not the explanation.
    X = X32.astype(np.float64)
    df = pd.DataFrame(X, columns=list(info.names))
    g = np.random.default_rng(0).normal(size=len(y))
    h = np.random.default_rng(1).uniform(0.05, 0.25, size=len(y))
    g_series = pd.Series(g)
    h_series = pd.Series(h)

    n_features = X.shape[1]
    results = {
        "NumPy array": (
            best_split_numpy(X, g, h),
            time_per_call(best_split_numpy, X, g, h, reps=args.reps),
            peak_kib(best_split_numpy, X, g, h),
        ),
        "pandas native (Series ops)": (
            best_split_pandas_native(df, g_series, h_series),
            time_per_call(best_split_pandas_native, df, g_series, h_series, reps=args.reps),
            peak_kib(best_split_pandas_native, df, g_series, h_series),
        ),
        "pandas container + NumPy kernel": (
            best_split_pandas_container(df, g, h),
            time_per_call(best_split_pandas_container, df, g, h, reps=args.reps),
            peak_kib(best_split_pandas_container, df, g, h),
        ),
    }

    reference = results["NumPy array"][0]
    print(f"data: {len(y)} rows x {n_features} features, {args.reps} reps")
    print("best-split search (all features), one call = one tree node\n")
    print(f"{'implementation':<34s} {'ms/call':>9s} {'vs NumPy':>9s} {'peak KiB':>9s}")
    print("-" * 66)
    for name, (split, ms, mem) in results.items():
        assert split[:2] == reference[:2], f"split mismatch for {name}"
        print(f"{name:<34s} {ms:>9.3f} {ms / results['NumPy array'][1]:>8.2f}x {mem:>9.1f}")

    # ID3-style information gain on one attribute.
    Xc, yc, cat_info = encode_categorical(raw)
    frame = raw.copy()
    frame["Class"] = (frame["Class"] == "recurrence-events").astype(int)
    attr_index = cat_info.names.index("tumor-size")
    codes = Xc[:, attr_index]
    cardinality = cat_info.cardinalities[attr_index]
    assert abs(
        gain_pandas_filtering(frame, "Class", "tumor-size")
        - gain_numpy_bincount(codes, yc, cardinality)
    ) < 1e-12
    gain_pd = time_per_call(
        gain_pandas_filtering, frame, "Class", "tumor-size", reps=args.reps
    )
    gain_np = time_per_call(
        gain_numpy_bincount, codes, yc, cardinality, reps=args.reps
    )
    print("\nID3 information gain on one attribute (tumor-size)\n")
    print(f"{'pandas boolean filtering':<34s} {gain_pd:>9.3f} ms/call")
    print(f"{'NumPy bincount':<34s} {gain_np:>9.3f} ms/call"
          f"   {gain_pd / gain_np:>5.1f}x faster")
    print(f"\npandas native / NumPy time ratio: "
          f"{results['pandas native (Series ops)'][1] / results['NumPy array'][1]:.1f}x")

    # Static memory footprint: the win comes from encoding, not from pandas.
    raw_bytes = int(raw.memory_usage(deep=True).sum())
    codes_bytes = int(Xc.nbytes)
    numeric_bytes = int(X32.nbytes)
    print("\nstatic memory footprint\n")
    print(f"raw DataFrame (object/string columns): {raw_bytes:>8,d} B")
    print(f"encoded category codes (uint8)      : {codes_bytes:>8,d} B")
    print(f"encoded numeric matrix (float32)    : {numeric_bytes:>8,d} B")
    print(f"numeric matrix (float64)            : {X.nbytes:>8,d} B")
    print(f"pandas DataFrame metadata+data      : "
          f"{int(df.memory_usage(deep=True).sum()):>8,d} B")
    print("\nNote: on 286 rows the absolute time is tiny; the relative overhead of")
    print("pandas' Series/Index layer is what dominates the hot loop.  Encoding")
    print("strings to ints and using compact dtypes is the real memory win.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
