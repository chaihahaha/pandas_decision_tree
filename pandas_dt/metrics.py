"""Evaluation helpers: binary metrics, bootstrap CI, threshold tuning."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def accuracy_score(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if len(y_true_arr) == 0:
        raise ValueError("cannot score an empty array")
    return float(np.mean(y_true_arr == y_pred_arr))


def log_loss_score(y_true: Iterable[int], proba: Iterable[float]) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.clip(np.asarray(proba, dtype=np.float64), 1e-15, 1.0 - 1e-15)
    if y.shape != p.shape:
        raise ValueError("y_true and proba must have the same shape")
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def brier_score(y_true: Iterable[int], proba: Iterable[float]) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(proba, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def confusion_matrix(y_true: Iterable[int], y_pred: Iterable[int]) -> np.ndarray:
    """Return ``[[TN, FP], [FN, TP]]`` for labels 0/1."""

    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_pred, dtype=int)
    if y.shape != p.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    matrix = np.zeros((2, 2), dtype=int)
    for true_value in (0, 1):
        for pred_value in (0, 1):
            matrix[true_value, pred_value] = int(
                np.sum((y == true_value) & (p == pred_value))
            )
    return matrix


def binary_metrics(
    y_true: Iterable[int],
    proba: Iterable[float],
    threshold: float = 0.5,
) -> dict[str, float | int | list[list[int]]]:
    """Compute a compact, JSON-friendly metric dictionary."""

    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=np.float64)
    if y.shape != p.shape:
        raise ValueError("y_true and proba must have the same shape")
    pred = (p >= threshold).astype(int)
    matrix = confusion_matrix(y, pred)
    tn, fp, fn, tp = matrix.ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "n": int(len(y)),
        "accuracy": float(np.mean(pred == y)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "log_loss": log_loss_score(y, p),
        "brier": brier_score(y, p),
        "confusion_matrix": matrix.tolist(),
        "threshold": float(threshold),
    }


def roc_auc_score(y_true: Iterable[int], proba: Iterable[float]) -> float:
    """Rank-based AUC; ties get average ranks (Mann-Whitney U)."""

    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=np.float64)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=np.float64)
    sorted_p = p[order]
    i = 0
    while i < len(p):
        j = i
        while j + 1 < len(p) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        average_rank = (i + j + 2) / 2.0  # ranks are 1-based
        ranks[order[i : j + 1]] = average_rank
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def tune_threshold(
    y_true: Iterable[int],
    proba: Iterable[float],
    grid: np.ndarray | None = None,
) -> tuple[float, float]:
    """Return ``(best_threshold, best_accuracy)`` on the provided data.

    Used only on validation data; never on the test set.
    """

    if grid is None:
        grid = np.linspace(0.05, 0.95, 91)
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=np.float64)
    best_threshold = 0.5
    best_accuracy = -1.0
    for threshold in grid:
        accuracy = float(np.mean((p >= threshold).astype(int) == y))
        if accuracy > best_accuracy + 1e-12:
            best_accuracy = accuracy
            best_threshold = float(threshold)
    return best_threshold, best_accuracy


def bootstrap_accuracy_ci(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    random_state: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for accuracy.

    The interval is useful context for a 57-sample test set: a single
    accuracy number has a very wide uncertainty band.
    """

    y = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    if len(y) != len(pred):
        raise ValueError("y_true and y_pred must have equal length")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    rng = np.random.default_rng(random_state)
    n = len(y)
    correct = (y == pred).astype(np.float64)
    estimates = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        sample = rng.integers(0, n, size=n)
        estimates[i] = correct[sample].mean()
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(estimates, alpha)), float(np.quantile(estimates, 1.0 - alpha))


def format_metrics(metrics: dict, name: str = "") -> str:
    """One-line human readable summary used by the CLI."""

    prefix = f"{name:<18}" if name else ""
    return (
        f"{prefix}acc={metrics['accuracy']:.4f}  "
        f"prec={metrics['precision']:.3f}  rec={metrics['recall']:.3f}  "
        f"f1={metrics['f1']:.3f}  logloss={metrics['log_loss']:.4f}"
    )
