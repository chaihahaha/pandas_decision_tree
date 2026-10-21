"""ID3 / C4.5 decision tree with multi-way categorical splits.

This is a corrected, vectorized rewrite of the original educational
implementation.  Key behavioural fixes:

* information gain is computed from contingency tables instead of
  re-filtering a DataFrame once per feature value (same result, much faster);
* the recursion checks the *child* set for emptiness (the original checked
  the parent and then called ``mode()`` on an empty frame);
* missing values are an explicit branch instead of being dropped from the
  tree and randomly guessed at prediction time;
* unseen categories fall back to the stored majority class of the node,
  making prediction deterministic;
* reduced-error pruning is done bottom-up using vectorized predictions on a
  held-out validation set instead of re-scoring the whole tree per node.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .data import FeatureInfo

_LEAF = -1


class ID3Node:
    """A node in an ID3/C4.5 tree.

    ``prediction`` is always the majority class of the training samples that
    reach the node, including internal nodes.  It is used when a future
    sample has a category that was never observed at training time.
    """

    __slots__ = (
        "feature",
        "children",
        "prediction",
        "proba",
        "n_samples",
        "class_counts",
        "gain",
    )

    def __init__(
        self,
        prediction: int,
        proba: np.ndarray,
        n_samples: int,
        class_counts: np.ndarray,
        feature: int = _LEAF,
        gain: float = 0.0,
    ) -> None:
        self.feature = feature
        self.children: dict[int, "ID3Node"] = {}
        self.prediction = int(prediction)
        self.proba = proba
        self.n_samples = int(n_samples)
        self.class_counts = class_counts
        self.gain = float(gain)

    @property
    def is_leaf(self) -> bool:
        return self.feature == _LEAF

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        if self.is_leaf:
            return f"ID3Node(leaf, pred={self.prediction}, n={self.n_samples})"
        return f"ID3Node(feature={self.feature}, children={list(self.children)}, n={self.n_samples})"


class ID3Classifier:
    """ID3 (information gain) or C4.5 (gain ratio) classifier.

    Parameters
    ----------
    criterion:
        ``"gain_ratio"`` (default, C4.5-style, reduces the bias towards
        high-cardinality attributes) or ``"gain"`` (classic ID3).
    max_depth:
        Maximum number of internal nodes on a root-to-leaf path.
        ``None`` means unlimited (bounded by the number of attributes).
    min_samples_split / min_samples_leaf:
        Pre-pruning controls.  ``min_samples_leaf`` is enforced for every
        branch of a candidate split.
    min_gain:
        Minimum information gain required to split.
    prune:
        If true, call :meth:`prune` automatically after ``fit`` when an
        evaluation set is supplied to ``fit``.
    """

    def __init__(
        self,
        criterion: str = "gain_ratio",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_gain: float = 1e-9,
        prune: bool = True,
        random_state: int | None = None,
    ) -> None:
        if criterion not in {"gain", "gain_ratio"}:
            raise ValueError("criterion must be 'gain' or 'gain_ratio'")
        if max_depth is not None and max_depth < 0:
            raise ValueError("max_depth must be >= 0 or None")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be >= 2")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1")

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_gain = min_gain
        self.prune = prune
        # Kept for API symmetry / reproducibility; ID3 itself is deterministic.
        self.random_state = random_state

        self.root_: ID3Node | None = None
        self.feature_info_: FeatureInfo | None = None
        self.n_features_: int = 0
        self.n_classes_: int = 0
        self.pruned_nodes_: int = 0
        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._feature_names: tuple[str, ...] = ()
        self._value_labels: tuple[tuple[str, ...], ...] = ()

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray | Sequence[Sequence[int]],
        y: np.ndarray | Sequence[int],
        feature_info: FeatureInfo | None = None,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> "ID3Classifier":
        """Build the tree.

        ``X`` must contain integer category codes in
        ``[0, cardinality)``.  ``feature_info`` supplies human-readable names
        and cardinalities; when omitted, cardinalities are inferred from the
        training data and names are ``f0, f1, ...``.
        """

        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        if X_arr.ndim != 2:
            raise ValueError("X must be 2-dimensional")
        if y_arr.ndim != 1 or len(y_arr) != len(X_arr):
            raise ValueError("X and y have incompatible shapes")
        if len(y_arr) == 0:
            raise ValueError("cannot fit on an empty dataset")
        if not np.issubdtype(X_arr.dtype, np.integer):
            if np.any(X_arr != np.floor(X_arr)):
                raise ValueError("ID3 requires integer category codes")
            X_arr = X_arr.astype(np.int64)
        if np.any(X_arr < 0):
            raise ValueError("category codes must be non-negative")
        if not np.isin(y_arr, (0, 1)).all():
            raise ValueError("ID3Classifier supports binary labels 0/1")

        self._feature_names, self._value_labels, cardinalities = _resolve_feature_info(
            X_arr, feature_info
        )
        if np.any(X_arr >= np.asarray(cardinalities, dtype=X_arr.dtype)):
            raise ValueError("category code exceeds the declared cardinality")

        self.n_features_ = X_arr.shape[1]
        self.n_classes_ = 2
        self.feature_info_ = feature_info
        self.pruned_nodes_ = 0
        self._X = X_arr
        self._y = y_arr

        all_idx = np.arange(len(y_arr))
        available = np.ones(self.n_features_, dtype=bool)
        self.root_ = self._grow(all_idx, available, depth=0)

        if (
            self.prune
            and eval_set is not None
            and self.root_ is not None
            and not self.root_.is_leaf
        ):
            X_val, y_val = eval_set
            self.prune_tree(X_val, y_val)

        return self

    def _grow(
        self,
        idx: np.ndarray,
        available: np.ndarray,
        depth: int,
    ) -> ID3Node:
        assert self._X is not None and self._y is not None
        y_node = self._y[idx]
        n = len(idx)
        counts = np.bincount(y_node, minlength=2).astype(np.int64)
        proba = counts / max(n, 1)
        prediction = int(np.argmax(counts))
        node = ID3Node(prediction, proba, n, counts)

        if n < self.min_samples_split or counts.min() == 0:
            return node
        if self.max_depth is not None and depth >= self.max_depth:
            return node
        if not available.any():
            return node

        parent_entropy = _entropy(counts)
        best_feature = -1
        best_score = 0.0
        best_gain = 0.0
        best_present: np.ndarray | None = None

        for feature in np.flatnonzero(available):
            feature = int(feature)
            cardinality = len(self._value_labels[feature])
            xj = self._X[idx, feature]
            # Contingency table: rows are category codes, columns are classes.
            joint = np.bincount(
                xj.astype(np.int64) * 2 + y_node,
                minlength=2 * cardinality,
            ).reshape(cardinality, 2)
            n_v = joint.sum(axis=1)
            present = n_v > 0
            n_present = int(present.sum())
            if n_present < 2:
                continue
            if int(n_v[present].min()) < self.min_samples_leaf:
                continue

            weighted_entropy = 0.0
            for v in np.flatnonzero(present):
                weighted_entropy += (n_v[v] / n) * _entropy(joint[v])
            gain = parent_entropy - weighted_entropy
            if gain <= self.min_gain:
                continue

            if self.criterion == "gain_ratio":
                split_info = _split_information(n_v[present], n)
                if split_info <= 1e-12:
                    continue
                score = gain / split_info
            else:
                score = gain

            if score > best_score + 1e-12:
                best_feature = feature
                best_score = score
                best_gain = gain
                best_present = present

        if best_feature < 0 or best_present is None:
            return node

        node.feature = best_feature
        node.gain = best_gain
        x_best = self._X[idx, best_feature]
        child_available = available.copy()
        child_available[best_feature] = False
        for value in np.flatnonzero(best_present):
            mask = x_best == value
            child_idx = idx[mask]
            if child_idx.size < self.min_samples_leaf:
                # Defensive: candidates are pre-filtered, but a value can be
                # duplicated in the index array only in pathological callers.
                continue
            node.children[int(value)] = self._grow(
                child_idx, child_available, depth + 1
            )
        return node

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------
    def prune_tree(
        self,
        X_val: np.ndarray | Sequence[Sequence[int]],
        y_val: np.ndarray | Sequence[int],
    ) -> "ID3Classifier":
        """Bottom-up reduced-error pruning on a validation set.

        Replacing a subtree by its majority leaf is accepted when it does not
        reduce validation accuracy.  Unlike the original implementation, the
        pruning decision is local to the validation samples that actually
        reach the node, and only one traversal is required.
        """

        if self.root_ is None:
            raise RuntimeError("call fit before prune_tree")
        X_val_arr = np.asarray(X_val)
        y_val_arr = np.asarray(y_val)
        if X_val_arr.ndim != 2 or X_val_arr.shape[1] != self.n_features_:
            raise ValueError("validation X has the wrong shape")
        if y_val_arr.ndim != 1 or len(y_val_arr) != len(X_val_arr):
            raise ValueError("validation X and y have incompatible shapes")
        if self.root_.is_leaf or len(y_val_arr) == 0:
            return self

        self.pruned_nodes_ = 0
        self._prune_node(self.root_, X_val_arr, y_val_arr, np.arange(len(y_val_arr)))
        return self

    def _prune_node(
        self,
        node: ID3Node,
        X_val: np.ndarray,
        y_val: np.ndarray,
        idx: np.ndarray,
    ) -> np.ndarray:
        if idx.size == 0:
            return np.empty(0, dtype=np.int8)
        if node.is_leaf:
            return np.full(idx.size, node.prediction, dtype=np.int8)

        values = X_val[idx, node.feature]
        subtree_predictions = np.full(idx.size, node.prediction, dtype=np.int8)
        for value, child in node.children.items():
            mask = values == value
            if mask.any():
                subtree_predictions[mask] = self._prune_node(
                    child, X_val, y_val, idx[mask]
                )

        subtree_accuracy = float(np.mean(subtree_predictions == y_val[idx]))
        leaf_accuracy = float(np.mean(node.prediction == y_val[idx]))
        if leaf_accuracy + 1e-12 >= subtree_accuracy:
            node.feature = _LEAF
            node.children.clear()
            self.pruned_nodes_ += 1
            return np.full(idx.size, node.prediction, dtype=np.int8)
        return subtree_predictions

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict_proba(self, X: np.ndarray | Sequence[Sequence[int]]) -> np.ndarray:
        if self.root_ is None:
            raise RuntimeError("call fit before predict")
        X_arr = np.asarray(X)
        if X_arr.ndim != 2 or X_arr.shape[1] != self.n_features_:
            raise ValueError("X has the wrong shape")
        out = np.empty((len(X_arr), 2), dtype=np.float64)
        self._predict_node(self.root_, X_arr, np.arange(len(X_arr)), out)
        return out

    def _predict_node(
        self,
        node: ID3Node,
        X: np.ndarray,
        idx: np.ndarray,
        out: np.ndarray,
    ) -> None:
        if idx.size == 0:
            return
        if node.is_leaf:
            out[idx] = node.proba
            return
        values = X[idx, node.feature]
        assigned = np.zeros(idx.size, dtype=bool)
        for value, child in node.children.items():
            mask = values == value
            if mask.any():
                assigned |= mask
                self._predict_node(child, X, idx[mask], out)
        # Unseen/missing category: deterministic majority fallback.
        if not assigned.all():
            out[idx[~assigned]] = node.proba

    def predict(
        self,
        X: np.ndarray | Sequence[Sequence[int]],
        threshold: float = 0.5,
    ) -> np.ndarray:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        return (self.predict_proba(X)[:, 1] >= threshold).astype(np.int8)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------
    @property
    def n_nodes(self) -> int:
        return _count_nodes(self.root_) if self.root_ is not None else 0

    @property
    def n_leaves(self) -> int:
        return _count_leaves(self.root_) if self.root_ is not None else 0

    @property
    def depth(self) -> int:
        return _tree_depth(self.root_) if self.root_ is not None else 0

    @property
    def feature_importances_(self) -> np.ndarray:
        """Sum of information gain attributed to each feature."""

        importances = np.zeros(self.n_features_, dtype=np.float64)
        if self.root_ is not None:
            _accumulate_importance(self.root_, importances)
        total = importances.sum()
        if total > 0:
            importances /= total
        return importances

    def decision_rules(self) -> list[tuple[list[tuple[str, str]], str]]:
        """Return ``(conditions, predicted_label)`` pairs for every leaf."""

        rules: list[tuple[list[tuple[str, str]], str]] = []
        if self.root_ is None:
            return rules
        self._collect_rules(self.root_, [], rules)
        return rules

    def _collect_rules(
        self,
        node: ID3Node,
        conditions: list[tuple[str, str]],
        out: list[tuple[list[tuple[str, str]], str]],
    ) -> None:
        if node.is_leaf:
            out.append((list(conditions), _label(node.prediction)))
            return
        feature_name = self._feature_names[node.feature]
        for value, child in node.children.items():
            labels = self._value_labels[node.feature]
            value_label = labels[value] if value < len(labels) else str(value)
            conditions.append((feature_name, value_label))
            self._collect_rules(child, conditions, out)
            conditions.pop()

    def export_rules(self, path: str | Path) -> Path:
        """Write human-readable rules using the original file's format."""

        path = Path(path)
        lines: list[str] = []
        for conditions, label in self.decision_rules():
            if conditions:
                body = " ∧ ".join(f"[{name} = {value}]" for name, value in conditions)
            else:
                body = "[any]"
            lines.append(f"{body} → [Class = {label}]")
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return path

    def __str__(self) -> str:
        if self.root_ is None:
            return "<unfitted ID3Classifier>"
        lines: list[str] = []
        self._render(self.root_, "", "", lines)
        return "\n".join(lines)

    def _render(
        self,
        node: ID3Node,
        prefix: str,
        condition: str,
        lines: list[str],
    ) -> None:
        label = _label(node.prediction)
        detail = f"n={node.n_samples}, p={node.proba[1]:.3f}"
        if node.is_leaf:
            lines.append(f"{prefix}{condition}→ {label} ({detail})")
            return
        feature_name = self._feature_names[node.feature]
        lines.append(
            f"{prefix}{condition}[{feature_name}?] gain={node.gain:.4f} ({detail})"
        )
        for value, child in node.children.items():
            labels = self._value_labels[node.feature]
            value_label = labels[value] if value < len(labels) else str(value)
            self._render(child, prefix + "  ", f"{feature_name}={value_label} ", lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_feature_info(
    X: np.ndarray,
    feature_info: FeatureInfo | None,
) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...], tuple[int, ...]]:
    if feature_info is not None:
        if len(feature_info.names) != X.shape[1]:
            raise ValueError("feature_info does not match the number of columns")
        if not (
            len(feature_info.names)
            == len(feature_info.cardinalities)
            == len(feature_info.value_labels)
            == len(feature_info.ordinal)
        ):
            raise ValueError("feature_info fields have inconsistent lengths")
        for cardinality, labels in zip(
            feature_info.cardinalities, feature_info.value_labels
        ):
            if int(cardinality) != len(labels):
                raise ValueError(
                    "feature_info cardinality does not match the number of value labels"
                )
        return (
            tuple(feature_info.names),
            tuple(tuple(v) for v in feature_info.value_labels),
            tuple(int(c) for c in feature_info.cardinalities),
        )

    cardinalities = tuple(int(v) for v in (X.max(axis=0) + 1)) if X.size else ()
    if any(c < 1 for c in cardinalities):
        raise ValueError("could not infer feature cardinalities")
    names = tuple(f"f{i}" for i in range(X.shape[1]))
    value_labels = tuple(tuple(str(i) for i in range(c)) for c in cardinalities)
    return names, value_labels, cardinalities


def _entropy(counts: np.ndarray) -> float:
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts:
        if count > 0:
            p = count / total
            entropy -= p * np.log2(p)
    return float(entropy)


def _split_information(counts: np.ndarray, total: int) -> float:
    info = 0.0
    for count in counts:
        if count > 0:
            p = count / total
            info -= p * np.log2(p)
    return float(info)


def _label(prediction: int) -> str:
    return "recurrence-events" if prediction == 1 else "no-recurrence-events"


def _count_nodes(node: ID3Node) -> int:
    return 1 + sum(_count_nodes(child) for child in node.children.values())


def _count_leaves(node: ID3Node) -> int:
    if node.is_leaf:
        return 1
    return sum(_count_leaves(child) for child in node.children.values())


def _tree_depth(node: ID3Node) -> int:
    if node.is_leaf:
        return 0
    return 1 + max(_tree_depth(child) for child in node.children.values())


def _accumulate_importance(node: ID3Node, importances: np.ndarray) -> None:
    if node.is_leaf:
        return
    importances[node.feature] += node.gain
    for child in node.children.values():
        _accumulate_importance(child, importances)
