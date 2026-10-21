"""Gradient boosted decision trees (binary logistic loss), from scratch.

The ensemble uses second-order (Newton) boosting, the same objective that
XGBoost optimizes:

    gain = G_L^2/(H_L + λ) + G_R^2/(H_R + λ) - G^2/(H + λ)

where for logistic loss ``g = p - y`` and ``h = p(1-p)``.  Leaf values are
``-G/(H + λ)`` and each tree is shrunk by ``learning_rate``.

Only NumPy is used for the learner, so the whole model fits in one readable
file and runs comfortably on the 286-row dataset.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

_LEAF = -1


class _RegressionNode:
    __slots__ = ("feature", "threshold", "left", "right", "value", "n_samples")

    def __init__(self, value: float, n_samples: int) -> None:
        self.feature = _LEAF
        self.threshold = 0.0
        self.left: _RegressionNode | None = None
        self.right: _RegressionNode | None = None
        self.value = float(value)
        self.n_samples = int(n_samples)

    @property
    def is_leaf(self) -> bool:
        return self.feature == _LEAF


class RegressionTree:
    """CART regression tree fitted to gradients/hessians.

    This is not a standalone regressor; it is the base learner used by
    :class:`GBDTClassifier`.  Exposed publicly to keep the implementation
    testable.
    """

    def __init__(
        self,
        max_depth: int = 3,
        min_samples_leaf: int = 5,
        min_child_weight: float = 1.0,
        reg_lambda: float = 1.0,
        max_features: float | int | None = None,
        random_state: int | None = None,
    ) -> None:
        if max_depth < 0:
            raise ValueError("max_depth must be >= 0")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1")
        if min_child_weight < 0:
            raise ValueError("min_child_weight must be >= 0")
        if reg_lambda < 0:
            raise ValueError("reg_lambda must be >= 0")

        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.max_features = max_features
        self.random_state = random_state

        self.root_: _RegressionNode | None = None
        self.feature_importances_: np.ndarray | None = None
        self.n_features_: int = 0
        self._rng = np.random.default_rng(random_state)

    def fit(
        self,
        X: np.ndarray,
        gradients: np.ndarray,
        hessians: np.ndarray,
    ) -> "RegressionTree":
        X = np.asarray(X, dtype=np.float32)
        gradients = np.asarray(gradients, dtype=np.float64)
        hessians = np.asarray(hessians, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional")
        if len(X) != len(gradients) or len(X) != len(hessians):
            raise ValueError("X, gradients and hessians have incompatible lengths")
        if len(X) == 0:
            raise ValueError("cannot fit on an empty dataset")
        if np.any(hessians < 0):
            raise ValueError("hessians must be non-negative")

        self._rng = np.random.default_rng(self.random_state)
        self.n_features_ = X.shape[1]
        importances = np.zeros(self.n_features_, dtype=np.float64)
        self.root_ = self._build(
            np.arange(len(X)), X, gradients, hessians, depth=0, importances=importances
        )
        total = importances.sum()
        if total > 0:
            importances /= total
        self.feature_importances_ = importances
        return self

    def _build(
        self,
        idx: np.ndarray,
        X: np.ndarray,
        gradients: np.ndarray,
        hessians: np.ndarray,
        depth: int,
        importances: np.ndarray,
    ) -> _RegressionNode:
        g = gradients[idx]
        h = hessians[idx]
        G = float(g.sum())
        H = float(h.sum())
        node = _RegressionNode(-G / (H + self.reg_lambda), len(idx))

        if depth >= self.max_depth:
            return node
        if len(idx) < 2 * self.min_samples_leaf:
            return node
        if H < self.min_child_weight:
            return node

        n = len(idx)
        parent_score = (G * G) / (H + self.reg_lambda)
        best_gain = 0.0
        best_feature = -1
        best_threshold = 0.0

        features = self._select_features()
        for feature in features:
            x = X[idx, feature]
            if x.min() == x.max():
                continue

            order = np.argsort(x, kind="mergesort")
            xs = x[order]
            if xs[0] == xs[-1]:
                continue
            gs = np.cumsum(g[order])
            hs = np.cumsum(h[order])

            # Split after position p: left = xs[:p+1], right = xs[p+1:].
            valid = np.flatnonzero(xs[:-1] < xs[1:])
            if valid.size == 0:
                continue
            left_count = valid + 1
            valid = valid[
                (left_count >= self.min_samples_leaf)
                & ((n - left_count) >= self.min_samples_leaf)
            ]
            if valid.size == 0:
                continue

            GL = gs[valid]
            HL = hs[valid]
            GR = G - GL
            HR = H - HL
            keep = (HL >= self.min_child_weight) & (HR >= self.min_child_weight)
            if not keep.any():
                continue
            GL = GL[keep]
            HL = HL[keep]
            GR = GR[keep]
            HR = HR[keep]
            valid = valid[keep]
            gains = (GL * GL) / (HL + self.reg_lambda) + (GR * GR) / (
                HR + self.reg_lambda
            )
            gains -= parent_score
            local = int(np.argmax(gains))
            gain = float(gains[local])
            if gain > best_gain + 1e-12:
                best_gain = gain
                best_feature = int(feature)
                position = int(valid[local])
                best_threshold = float((xs[position] + xs[position + 1]) / 2.0)

        if best_feature < 0:
            return node

        importances[best_feature] += best_gain
        node.feature = best_feature
        node.threshold = best_threshold
        x_best = X[idx, best_feature]
        left_mask = x_best <= best_threshold
        left_idx = idx[left_mask]
        right_idx = idx[~left_mask]
        # ``best_threshold`` lies strictly between two observed values, so
        # both sides are non-empty by construction; keep the guard for safety.
        if left_idx.size == 0 or right_idx.size == 0:
            return node
        node.left = self._build(
            left_idx, X, gradients, hessians, depth + 1, importances
        )
        node.right = self._build(
            right_idx, X, gradients, hessians, depth + 1, importances
        )
        return node

    def _select_features(self) -> np.ndarray:
        n_features = self.n_features_
        mf = self.max_features
        if mf is None or mf == 1.0 or mf == n_features:
            return np.arange(n_features)
        if isinstance(mf, (int, np.integer)):
            if mf <= 0:
                raise ValueError("max_features must be positive")
            size = min(int(mf), n_features)
        else:
            if not 0.0 < float(mf) <= 1.0:
                raise ValueError("float max_features must be in (0, 1]")
            size = max(1, int(np.ceil(float(mf) * n_features)))
        return self._rng.choice(n_features, size=size, replace=False)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.root_ is None:
            raise RuntimeError("call fit before predict")
        X = np.asarray(X, dtype=np.float32)
        out = np.empty(len(X), dtype=np.float64)
        stack: list[tuple[_RegressionNode, np.ndarray]] = [
            (self.root_, np.arange(len(X)))
        ]
        while stack:
            node, idx = stack.pop()
            if idx.size == 0:
                continue
            if node.is_leaf:
                out[idx] = node.value
                continue
            assert node.left is not None and node.right is not None
            mask = X[idx, node.feature] <= node.threshold
            stack.append((node.left, idx[mask]))
            stack.append((node.right, idx[~mask]))
        return out

    @property
    def n_leaves(self) -> int:
        return _count_leaves(self.root_) if self.root_ is not None else 0


class GBDTClassifier:
    """Binary gradient boosted decision trees with logistic loss.

    Parameters mirror the usual GBDT knobs:
    ``n_estimators``, ``learning_rate``, tree depth, leaf regularisation,
    row/column subsampling and early stopping on a validation set.

    The model is deliberately small enough to run in a second on the
    breast-cancer dataset, but the implementation is generic.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 3,
        min_samples_leaf: int = 8,
        min_child_weight: float = 1.0,
        reg_lambda: float = 1.0,
        subsample: float = 0.8,
        max_features: float | int | None = 0.8,
        random_state: int = 0,
        early_stopping_rounds: int | None = 30,
        eval_metric: str = "logloss",
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be >= 1")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if not 0.0 < subsample <= 1.0:
            raise ValueError("subsample must be in (0, 1]")
        if eval_metric not in {"logloss", "error"}:
            raise ValueError("eval_metric must be 'logloss' or 'error'")
        if early_stopping_rounds is not None and early_stopping_rounds < 1:
            raise ValueError("early_stopping_rounds must be >= 1 or None")

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.subsample = subsample
        self.max_features = max_features
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric

        self.trees_: list[RegressionTree] = []
        self.base_score_: float = 0.0
        self.best_iteration_: int = 0
        self.train_loss_: list[float] = []
        self.val_loss_: list[float] = []
        self.feature_importances_: np.ndarray | None = None
        self._n_features: int = 0

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> "GBDTClassifier":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional")
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("X and y have incompatible shapes")
        if not np.isin(y, (0.0, 1.0)).all():
            raise ValueError("GBDTClassifier supports binary labels 0/1")
        if len(y) == 0:
            raise ValueError("cannot fit on an empty dataset")

        n_features = X.shape[1]
        self._n_features = n_features
        self.trees_ = []
        self.train_loss_ = []
        self.val_loss_ = []

        positive_rate = float(np.clip(y.mean(), 1e-6, 1.0 - 1e-6))
        self.base_score_ = float(np.log(positive_rate / (1.0 - positive_rate)))
        raw = np.full(len(y), self.base_score_, dtype=np.float64)

        if eval_set is not None:
            X_val = np.asarray(eval_set[0], dtype=np.float32)
            y_val = np.asarray(eval_set[1], dtype=np.float64)
            if X_val.ndim != 2 or X_val.shape[1] != n_features:
                raise ValueError("eval X has the wrong shape")
            if y_val.ndim != 1 or len(y_val) != len(X_val):
                raise ValueError("eval X and y have incompatible shapes")
            raw_val = np.full(len(y_val), self.base_score_, dtype=np.float64)
        else:
            X_val = y_val = raw_val = None

        best_score = np.inf
        best_iteration = 0
        rounds_without_improvement = 0
        rng = np.random.default_rng(self.random_state)
        n_samples = len(y)

        for iteration in range(1, self.n_estimators + 1):
            p = _sigmoid(raw)
            gradient = p - y
            hessian = np.maximum(p * (1.0 - p), 1e-6)

            if self.subsample < 1.0:
                size = max(2 * self.min_samples_leaf, int(round(n_samples * self.subsample)))
                size = min(size, n_samples)
                rows = rng.choice(n_samples, size=size, replace=False)
            else:
                rows = np.arange(n_samples)

            tree = RegressionTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_child_weight=self.min_child_weight,
                reg_lambda=self.reg_lambda,
                max_features=self.max_features,
                random_state=self.random_state + iteration,
            )
            tree.fit(X[rows], gradient[rows], hessian[rows])
            raw += self.learning_rate * tree.predict(X)
            self.trees_.append(tree)

            self.train_loss_.append(_log_loss(y, _sigmoid(raw)))
            if X_val is not None and y_val is not None and raw_val is not None:
                raw_val = raw_val + self.learning_rate * tree.predict(X_val)
                score = (
                    _log_loss(y_val, _sigmoid(raw_val))
                    if self.eval_metric == "logloss"
                    else 1.0 - _accuracy(y_val, raw_val)
                )
                self.val_loss_.append(score)
                if score < best_score - 1e-12:
                    best_score = score
                    best_iteration = iteration
                    rounds_without_improvement = 0
                else:
                    rounds_without_improvement += 1

                if (
                    self.early_stopping_rounds is not None
                    and rounds_without_improvement >= self.early_stopping_rounds
                ):
                    break
            else:
                best_iteration = iteration

        self.best_iteration_ = best_iteration if best_iteration > 0 else len(self.trees_)

        importances = np.zeros(n_features, dtype=np.float64)
        for tree in self.trees_[: self.best_iteration_]:
            if tree.feature_importances_ is not None:
                importances += tree.feature_importances_
        total = importances.sum()
        self.feature_importances_ = importances / total if total > 0 else importances
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = np.asarray(X, dtype=np.float32)
        raw = np.full(len(X), self.base_score_, dtype=np.float64)
        for tree in self.trees_[: self.best_iteration_]:
            raw += self.learning_rate * tree.predict(X)
        return raw

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = _sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - p, p])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        return (self.decision_function(X) >= np.log(threshold / (1.0 - threshold))).astype(
            np.int8
        )

    def staged_predict_proba(self, X: np.ndarray):
        """Yield ``(iteration, probabilities)`` for inspection/curves."""

        self._check_fitted()
        X = np.asarray(X, dtype=np.float32)
        raw = np.full(len(X), self.base_score_, dtype=np.float64)
        for iteration, tree in enumerate(self.trees_, start=1):
            raw = raw + self.learning_rate * tree.predict(X)
            p = _sigmoid(raw)
            yield iteration, np.column_stack([1.0 - p, p])

    def _check_fitted(self) -> None:
        if not self.trees_ or self.feature_importances_ is None:
            raise RuntimeError("call fit before predict")

    @property
    def n_used_trees(self) -> int:
        return min(self.best_iteration_, len(self.trees_)) if self.trees_ else 0


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""

    out = np.empty_like(x, dtype=np.float64)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def _log_loss(y_true: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-15, 1.0 - 1e-15)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


def _accuracy(y_true: np.ndarray, raw: np.ndarray) -> float:
    return float(np.mean((raw >= 0.0).astype(np.float64) == y_true))


def _count_leaves(node: _RegressionNode) -> int:
    if node.is_leaf:
        return 1
    assert node.left is not None and node.right is not None
    return _count_leaves(node.left) + _count_leaves(node.right)


def accuracy_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Tiny dependency-free accuracy helper (also convenient for callers)."""

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if len(y_true_arr) == 0:
        raise ValueError("cannot score an empty array")
    return float(np.mean(y_true_arr == y_pred_arr))


def log_loss_score(y_true: Sequence[int], proba: Sequence[float]) -> float:
    """Binary log loss for probability-of-class-1 predictions."""

    y = np.asarray(y_true, dtype=np.float64)
    p = np.clip(np.asarray(proba, dtype=np.float64), 1e-15, 1.0 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
