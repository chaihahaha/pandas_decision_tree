"""pandas-first decision trees.

Two complementary implementations are provided:

``PandasID3Classifier``
    Recommended for real use.  It accepts a DataFrame with arbitrary string /
    categorical / numeric columns, handles missing values and preserves the
    index and column names.  pandas performs the one-time encoding; the
    per-node split search uses a compact integer kernel (the same code as
    :class:`pandas_dt.tree.ID3Classifier`).  This keeps the pandas API while
    avoiding repeated DataFrame filtering.

``GroupbyID3Classifier``
    A fully pandas-native reference implementation.  Every split gain is
    computed with ``DataFrame.groupby`` / ``Series.value_counts``; strings are
    handled directly and no manual encoding is required.  It is slower than
    the wrapper but still much faster than the original per-value
    ``df[df[col] == value]`` style, especially as the data grows.

Both classes support binary classification, explicit missing values, unseen
categories (deterministic majority fallback), gain / gain-ratio criteria and
optional reduced-error pruning on a validation DataFrame.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .data import FeatureInfo
from .tree import ID3Classifier

MISSING_SENTINEL = "__pdt_missing__"
_LEAF = None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _unique_sorted(values: Sequence[Any]) -> list[Any]:
    """Unique values, sorted when the type supports it."""

    unique = list(pd.unique(values))
    try:
        return sorted(unique)
    except TypeError:
        # Mixed types (e.g. int + str) may not be orderable.
        return unique


def _entropy_counts(counts: np.ndarray) -> float:
    total = float(np.sum(counts))
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def _entropy_rows(counts: np.ndarray) -> np.ndarray:
    """Row-wise entropy for a 2-D count matrix."""

    counts = np.asarray(counts, dtype=np.float64)
    totals = counts.sum(axis=1)
    result = np.zeros(len(counts), dtype=np.float64)
    valid = totals > 0
    if valid.any():
        p = counts[valid] / totals[valid, None]
        logp = np.zeros_like(p)
        np.log2(p, out=logp, where=p > 0)
        result[valid] = -np.sum(p * logp, axis=1)
    return result


def _choose_sentinel(series: pd.Series) -> str:
    """Pick a missing-value sentinel that cannot collide with real values."""

    existing = {str(v) for v in pd.unique(series.dropna())}
    sentinel = MISSING_SENTINEL
    while sentinel in existing:
        sentinel += "_"
    return sentinel


def _check_frame(df: pd.DataFrame, target: str, features: Sequence[str] | None):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("expected a pandas DataFrame")
    if target not in df.columns:
        raise ValueError(f"target column {target!r} is not in the DataFrame")
    if len(df) == 0:
        raise ValueError("cannot fit on an empty DataFrame")
    if features is None:
        features = [c for c in df.columns if c != target]
    features = list(features)
    if not features:
        raise ValueError("at least one feature column is required")
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"feature column(s) not found: {missing}")
    if target in features:
        raise ValueError("target column must not be used as a feature")
    if df[target].isna().any():
        raise ValueError("target column contains missing values")
    return features


# ---------------------------------------------------------------------------
# pandas API + compact integer kernel (recommended)
# ---------------------------------------------------------------------------


class PandasID3Classifier:
    """ID3/C4.5 decision tree with a pandas DataFrame API.

    Parameters
    ----------
    target:
        Name of the target column.
    criterion:
        ``"gain_ratio"`` (default) or ``"gain"``.
    max_depth / min_samples_split / min_samples_leaf / min_gain:
        Pre-pruning controls.
    prune:
        If an ``eval_set`` is supplied to :meth:`fit`, perform bottom-up
        reduced-error pruning on it.

    Examples
    --------
    >>> tree = PandasID3Classifier(target="Class").fit(train_df, eval_set=val_df)
    >>> pred = tree.predict(test_df)       # pd.Series, original index
    >>> proba = tree.predict_proba(test_df)  # pd.DataFrame, class columns
    """

    def __init__(
        self,
        target: str = "Class",
        criterion: str = "gain_ratio",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 2,
        min_gain: float = 1e-9,
        prune: bool = True,
    ) -> None:
        if criterion not in {"gain", "gain_ratio"}:
            raise ValueError("criterion must be 'gain' or 'gain_ratio'")
        if max_depth is not None and max_depth < 0:
            raise ValueError("max_depth must be >= 0 or None")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be >= 2")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1")
        if min_gain < 0:
            raise ValueError("min_gain must be >= 0")

        self.target = target
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_gain = min_gain
        self.prune = prune

        self.model_: ID3Classifier | None = None
        self.feature_names_: list[str] = []
        self.categories_: dict[str, tuple[Any, ...]] = {}
        self.missing_sentinels_: dict[str, str | None] = {}
        self.classes_: list[Any] = []
        self.target_: str = target
        self.info_: FeatureInfo | None = None
        self.pruned_nodes_: int = 0
        self._fitted = False

    # -- encoding ------------------------------------------------------
    def _encode_fit(
        self, df: pd.DataFrame, features: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        self.feature_names_ = features
        self.categories_ = {}
        self.missing_sentinels_ = {}
        columns: list[np.ndarray] = []

        for column in features:
            series = df[column]
            if isinstance(series.dtype, pd.CategoricalDtype):
                categories = list(series.cat.categories)
                values = series.astype(object)
            else:
                categories = _unique_sorted(series.dropna())
                values = series

            if series.isna().any():
                sentinel = _choose_sentinel(series)
                if sentinel not in categories:
                    categories.append(sentinel)
                values = values.where(series.notna(), sentinel)
                self.missing_sentinels_[column] = sentinel
            else:
                self.missing_sentinels_[column] = None

            if not categories:  # entire column missing
                sentinel = MISSING_SENTINEL
                categories = [sentinel]
                values = pd.Series(sentinel, index=series.index)
                self.missing_sentinels_[column] = sentinel

            self.categories_[column] = tuple(categories)
            codes = pd.Categorical(values, categories=categories).codes
            columns.append(codes.astype(np.int64))

        X = (
            np.column_stack(columns)
            if columns
            else np.empty((len(df), 0), dtype=np.int64)
        )
        y = self._encode_target(df[self.target])
        return X, y

    def _encode_target(self, series: pd.Series) -> np.ndarray:
        codes = pd.Categorical(series, categories=self.classes_).codes
        if np.any(codes < 0):
            unknown = pd.unique(series[codes < 0]).tolist()
            raise ValueError(f"unknown target label(s): {unknown}")
        return codes.astype(np.int8)

    def _transform(self, df: pd.DataFrame) -> np.ndarray:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("expected a pandas DataFrame")
        missing = [c for c in self.feature_names_ if c not in df.columns]
        if missing:
            raise ValueError(f"missing feature column(s): {missing}")

        columns: list[np.ndarray] = []
        for column in self.feature_names_:
            series = df[column]
            sentinel = self.missing_sentinels_[column]
            if series.isna().any() and sentinel is not None:
                values = series.astype(object).where(series.notna(), sentinel)
            else:
                values = series
            categories = self.categories_[column]
            codes = pd.Categorical(values, categories=categories).codes
            columns.append(codes.astype(np.int64))
        return (
            np.column_stack(columns)
            if columns
            else np.empty((len(df), 0), dtype=np.int64)
        )

    def _build_info(self) -> FeatureInfo:
        return FeatureInfo(
            names=tuple(self.feature_names_),
            cardinalities=tuple(
                len(self.categories_[c]) for c in self.feature_names_
            ),
            value_labels=tuple(
                tuple(str(v) for v in self.categories_[c]) for c in self.feature_names_
            ),
            ordinal=tuple(False for _ in self.feature_names_),
        )

    # -- fit / predict -------------------------------------------------
    def fit(
        self,
        df: pd.DataFrame,
        eval_set: pd.DataFrame | None = None,
        features: Sequence[str] | None = None,
    ) -> "PandasID3Classifier":
        features = _check_frame(df, self.target, features)
        self.target_ = self.target

        classes = _unique_sorted(df[self.target].dropna())
        if len(classes) != 2:
            raise ValueError(
                "PandasID3Classifier is binary; "
                f"found {len(classes)} target classes"
            )
        self.classes_ = classes

        X, y = self._encode_fit(df, features)
        self.info_ = self._build_info()

        eval_encoded = None
        if eval_set is not None:
            if not isinstance(eval_set, pd.DataFrame):
                raise TypeError("eval_set must be a pandas DataFrame")
            if self.target not in eval_set.columns:
                raise ValueError(f"eval_set is missing target {self.target!r}")
            X_val = self._transform(eval_set)
            y_val = self._encode_target(eval_set[self.target])
            eval_encoded = (X_val, y_val)

        self.model_ = ID3Classifier(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_gain=self.min_gain,
            prune=self.prune and eval_encoded is not None,
        )
        self.model_.fit(X, y, feature_info=self.info_, eval_set=eval_encoded)
        self.pruned_nodes_ = self.model_.pruned_nodes_
        self._fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self._fitted or self.model_ is None:
            raise RuntimeError("call fit before predict")

    def predict(self, df: pd.DataFrame) -> pd.Series:
        self._check_fitted()
        codes = self.model_.predict(self._transform(df))
        labels = np.asarray(self.classes_, dtype=object)[codes]
        return pd.Series(labels, index=df.index, name=self.target_)

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        proba = self.model_.predict_proba(self._transform(df))
        return pd.DataFrame(proba, index=df.index, columns=list(self.classes_))

    def score(self, df: pd.DataFrame) -> float:
        self._check_fitted()
        y_true = self._encode_target(df[self.target_])
        y_pred = self.model_.predict(self._transform(df))
        return float(np.mean(y_true == y_pred))

    # -- inspection ----------------------------------------------------
    @property
    def feature_importances_(self) -> pd.Series:
        self._check_fitted()
        assert self.model_ is not None
        return pd.Series(
            self.model_.feature_importances_,
            index=self.feature_names_,
            name="importance",
        )

    @property
    def n_nodes(self) -> int:
        self._check_fitted()
        assert self.model_ is not None
        return self.model_.n_nodes

    @property
    def n_leaves(self) -> int:
        self._check_fitted()
        assert self.model_ is not None
        return self.model_.n_leaves

    @property
    def depth(self) -> int:
        self._check_fitted()
        assert self.model_ is not None
        return self.model_.depth

    def _value_label(self, feature: str, code: int) -> str:
        categories = self.categories_[feature]
        if 0 <= code < len(categories):
            return str(categories[code])
        return str(code)

    def decision_rules(self) -> list[tuple[list[tuple[str, str]], Any]]:
        self._check_fitted()
        assert self.model_ is not None and self.model_.root_ is not None
        rules: list[tuple[list[tuple[str, str]], Any]] = []

        def visit(node, conditions):
            if node.is_leaf:
                rules.append((list(conditions), self.classes_[node.prediction]))
                return
            feature = self.feature_names_[node.feature]
            for code, child in node.children.items():
                conditions.append((feature, self._value_label(feature, code)))
                visit(child, conditions)
                conditions.pop()

        visit(self.model_.root_, [])
        return rules

    def export_rules(self, path: str | Path) -> Path:
        path = Path(path)
        lines = []
        for conditions, label in self.decision_rules():
            body = (
                " ∧ ".join(f"[{name} = {value}]" for name, value in conditions)
                if conditions
                else "[any]"
            )
            lines.append(f"{body} → [{self.target_} = {label}]")
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return path

    def __str__(self) -> str:
        self._check_fitted()
        assert self.model_ is not None and self.model_.root_ is not None
        lines: list[str] = []

        def visit(node, prefix: str, condition: str) -> None:
            if node.is_leaf:
                lines.append(
                    f"{prefix}{condition}→ {self.classes_[node.prediction]} "
                    f"(n={node.n_samples})"
                )
                return
            feature = self.feature_names_[node.feature]
            lines.append(
                f"{prefix}{condition}[{feature}?] gain={node.gain:.4f} "
                f"(n={node.n_samples})"
            )
            for code, child in node.children.items():
                visit(
                    child,
                    prefix + "  ",
                    f"{feature}={self._value_label(feature, code)} ",
                )

        visit(self.model_.root_, "", "")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fully pandas-native reference implementation
# ---------------------------------------------------------------------------


class _PandasNode:
    __slots__ = ("feature", "children", "prediction", "proba", "n_samples", "gain")

    def __init__(
        self,
        prediction: Any,
        proba: np.ndarray,
        n_samples: int,
    ) -> None:
        self.feature: str | None = _LEAF
        self.children: dict[Any, "_PandasNode"] = {}
        self.prediction = prediction
        self.proba = proba
        self.n_samples = int(n_samples)
        self.gain = 0.0

    @property
    def is_leaf(self) -> bool:
        return self.feature is None


class GroupbyID3Classifier:
    """Fully pandas-native ID3/C4.5 tree using ``DataFrame.groupby``.

    Every node's gain is computed by grouping the current frame once per
    candidate feature (``groupby([feature, target]).size().unstack()``), rather
    than filtering the frame once per feature value.  Missing values become a
    dedicated categorical level; unseen levels fall back deterministically to
    the node's majority class.

    This is a reference implementation for pandas users.  Only the small
    contingency-table counts are converted to NumPy for the entropy math;
    all grouping, string handling and missing-value handling stays in pandas.
    For maximum speed use :class:`PandasID3Classifier`, which encodes once and
    then uses a compact integer kernel.
    """

    def __init__(
        self,
        target: str = "Class",
        criterion: str = "gain_ratio",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 2,
        min_gain: float = 1e-9,
        prune: bool = True,
    ) -> None:
        if criterion not in {"gain", "gain_ratio"}:
            raise ValueError("criterion must be 'gain' or 'gain_ratio'")
        if max_depth is not None and max_depth < 0:
            raise ValueError("max_depth must be >= 0 or None")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be >= 2")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1")
        if min_gain < 0:
            raise ValueError("min_gain must be >= 0")

        self.target = target
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_gain = min_gain
        self.prune = prune

        self.root_: _PandasNode | None = None
        self.feature_names_: list[str] = []
        self.classes_: list[Any] = []
        self.target_: str = target
        self.missing_sentinels_: dict[str, str | None] = {}
        self.pruned_nodes_: int = 0
        self._fitted = False

    # -- preprocessing -------------------------------------------------
    def _choose_sentinels(self, df: pd.DataFrame) -> dict[str, str | None]:
        sentinels: dict[str, str | None] = {}
        for column in self.feature_names_:
            series = df[column]
            sentinels[column] = _choose_sentinel(series) if series.isna().any() else None
        return sentinels

    def _prepare_features(
        self, df: pd.DataFrame, sentinels: dict[str, str | None]
    ) -> pd.DataFrame:
        prepared = df[self.feature_names_].copy()
        for column in self.feature_names_:
            sentinel = sentinels.get(column)
            if sentinel is None:
                continue
            series = prepared[column]
            if isinstance(series.dtype, pd.CategoricalDtype):
                if sentinel not in series.cat.categories:
                    series = series.cat.add_categories([sentinel])
                prepared[column] = series.fillna(sentinel)
            else:
                prepared[column] = series.where(series.notna(), sentinel)
        return prepared

    def _attach_target(self, frame: pd.DataFrame, source: pd.DataFrame) -> None:
        """Attach target values without relying on index alignment."""

        series = source[self.target]
        if isinstance(series.dtype, pd.CategoricalDtype):
            frame[self.target] = pd.Categorical(
                series.to_numpy(), categories=series.cat.categories
            )
        else:
            frame[self.target] = series.to_numpy()

    @staticmethod
    def _class_probabilities(
        counts: pd.Series, classes: list[Any], n: int
    ) -> np.ndarray:
        values = np.zeros(len(classes), dtype=np.float64)
        for i, label in enumerate(classes):
            values[i] = counts.get(label, 0) / n
        return values

    # -- fit -----------------------------------------------------------
    def fit(
        self,
        df: pd.DataFrame,
        eval_set: pd.DataFrame | None = None,
        features: Sequence[str] | None = None,
    ) -> "GroupbyID3Classifier":
        features = _check_frame(df, self.target, features)
        self.feature_names_ = features
        self.target_ = self.target
        classes = _unique_sorted(df[self.target].dropna())
        if len(classes) != 2:
            raise ValueError(
                "GroupbyID3Classifier is binary; "
                f"found {len(classes)} target classes"
            )
        self.classes_ = classes
        self.missing_sentinels_ = self._choose_sentinels(df)
        self.pruned_nodes_ = 0

        frame = self._prepare_features(df, self.missing_sentinels_)
        self._attach_target(frame, df)
        self.root_ = self._grow(frame, self.feature_names_, depth=0)

        if eval_set is not None:
            if self.target not in eval_set.columns:
                raise ValueError(f"eval_set is missing target {self.target!r}")
            eval_frame = self._prepare_features(eval_set, self.missing_sentinels_)
            self._attach_target(eval_frame, eval_set)
            if self.prune:
                self._prune_node(self.root_, eval_frame)

        self._fitted = True
        return self

    def _grow(
        self, frame: pd.DataFrame, features: list[str], depth: int
    ) -> _PandasNode:
        target = self.target_
        counts = frame[target].value_counts(dropna=False)
        n = len(frame)
        prediction = counts.index[0]
        proba = self._class_probabilities(counts, self.classes_, n)
        node = _PandasNode(prediction, proba, n)

        if n < self.min_samples_split or len(counts) == 1 or not features:
            return node
        if self.max_depth is not None and depth >= self.max_depth:
            return node

        parent_counts = counts.reindex(self.classes_, fill_value=0).to_numpy(float)
        parent_entropy = _entropy_counts(parent_counts)
        best_score = 0.0
        best_feature: str | None = None
        best_gain = 0.0

        for feature in features:
            # One groupby gives the full contingency table for this feature.
            table = (
                frame.groupby(
                    [feature, target], observed=True, sort=False, dropna=False
                )
                .size()
                .unstack(fill_value=0)
            )
            table = table.reindex(columns=self.classes_, fill_value=0)
            matrix = table.to_numpy(dtype=np.float64)
            row_counts = matrix.sum(axis=1)
            total = matrix.sum()
            valid = row_counts >= self.min_samples_leaf
            if int(valid.sum()) < 2:
                continue

            entropies = _entropy_rows(matrix)
            weighted = float(
                ((row_counts[valid] / total) * entropies[valid]).sum()
            )
            gain = parent_entropy - weighted
            if gain <= self.min_gain:
                continue

            if self.criterion == "gain_ratio":
                p = row_counts[valid] / total
                split_info = float(-(p * np.log2(p)).sum())
                score = gain / split_info if split_info > 1e-12 else 0.0
            else:
                score = gain

            if score > best_score + 1e-12:
                best_score = score
                best_feature = feature
                best_gain = gain

        if best_feature is None:
            return node

        node.feature = best_feature
        node.gain = best_gain
        child_features = [f for f in features if f != best_feature]
        for value, subset in frame.groupby(
            best_feature, observed=True, sort=False, dropna=False
        ):
            if len(subset) < self.min_samples_leaf:
                continue
            node.children[value] = self._grow(subset, child_features, depth + 1)

        if not node.children:
            node.feature = None
            node.gain = 0.0
        return node

    # -- prediction ----------------------------------------------------
    def _prepare_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("expected a pandas DataFrame")
        missing = [c for c in self.feature_names_ if c not in df.columns]
        if missing:
            raise ValueError(f"missing feature column(s): {missing}")
        return self._prepare_features(df, self.missing_sentinels_)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if not self._fitted or self.root_ is None:
            raise RuntimeError("call fit before predict")
        frame = self._prepare_predict(df)
        out = pd.Series(index=df.index, dtype=object)
        self._predict_node(self.root_, frame, out)
        return out.rename(self.target_)

    def _predict_node(
        self, node: _PandasNode, frame: pd.DataFrame, out: pd.Series
    ) -> None:
        if frame.empty:
            return
        if node.is_leaf:
            out.loc[frame.index] = node.prediction
            return
        values = frame[node.feature]
        assigned = pd.Series(False, index=frame.index)
        for value, child in node.children.items():
            mask = values.eq(value)
            if mask.any():
                assigned |= mask
                self._predict_node(child, frame[mask], out)
        if not assigned.all():
            out.loc[frame.index[~assigned]] = node.prediction

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted or self.root_ is None:
            raise RuntimeError("call fit before predict_proba")
        frame = self._prepare_predict(df)
        out = pd.DataFrame(
            0.0, index=df.index, columns=list(self.classes_), dtype=float
        )
        self._predict_proba_node(self.root_, frame, out)
        return out

    def _predict_proba_node(
        self, node: _PandasNode, frame: pd.DataFrame, out: pd.DataFrame
    ) -> None:
        if frame.empty:
            return
        if node.is_leaf:
            out.loc[frame.index, :] = node.proba
            return
        values = frame[node.feature]
        assigned = pd.Series(False, index=frame.index)
        for value, child in node.children.items():
            mask = values.eq(value)
            if mask.any():
                assigned |= mask
                self._predict_proba_node(child, frame[mask], out)
        if not assigned.all():
            out.loc[frame.index[~assigned], :] = node.proba

    def score(self, df: pd.DataFrame) -> float:
        if self.target not in df.columns:
            raise ValueError(f"target column {self.target!r} is not in the DataFrame")
        predictions = self.predict(df)
        return float(np.mean(predictions.to_numpy() == df[self.target].to_numpy()))

    # -- pruning -------------------------------------------------------
    def _prune_node(
        self, node: _PandasNode, frame: pd.DataFrame
    ) -> pd.Series:
        """Bottom-up reduced-error pruning; returns predictions for ``frame``."""

        if frame.empty:
            return pd.Series(index=frame.index, dtype=object)
        if node.is_leaf:
            return pd.Series(node.prediction, index=frame.index, dtype=object)

        predictions = pd.Series(node.prediction, index=frame.index, dtype=object)
        values = frame[node.feature]
        for value, child in node.children.items():
            mask = values.eq(value)
            if mask.any():
                predictions.loc[mask] = self._prune_node(
                    child, frame[mask]
                ).to_numpy()

        leaf_predictions = pd.Series(
            node.prediction, index=frame.index, dtype=object
        )
        leaf_accuracy = float(
            np.mean(leaf_predictions.to_numpy() == frame[self.target].to_numpy())
        )
        subtree_accuracy = float(
            np.mean(predictions.to_numpy() == frame[self.target].to_numpy())
        )
        if leaf_accuracy + 1e-12 >= subtree_accuracy:
            node.feature = None
            node.children.clear()
            self.pruned_nodes_ += 1
            return leaf_predictions
        return predictions

    # -- inspection ----------------------------------------------------
    @property
    def feature_importances_(self) -> pd.Series:
        if not self._fitted or self.root_ is None:
            raise RuntimeError("call fit first")
        importances = np.zeros(len(self.feature_names_), dtype=float)
        index = {name: i for i, name in enumerate(self.feature_names_)}

        def visit(node: _PandasNode) -> None:
            if node.is_leaf:
                return
            importances[index[node.feature]] += node.gain
            for child in node.children.values():
                visit(child)

        visit(self.root_)
        total = importances.sum()
        if total > 0:
            importances /= total
        return pd.Series(importances, index=self.feature_names_, name="importance")

    @property
    def n_nodes(self) -> int:
        return _count_nodes(self.root_)

    @property
    def n_leaves(self) -> int:
        return _count_leaves(self.root_)

    @property
    def depth(self) -> int:
        return _node_depth(self.root_)

    def decision_rules(self) -> list[tuple[list[tuple[str, str]], Any]]:
        if not self._fitted or self.root_ is None:
            raise RuntimeError("call fit first")
        rules: list[tuple[list[tuple[str, str]], Any]] = []

        def visit(node: _PandasNode, conditions: list[tuple[str, str]]) -> None:
            if node.is_leaf:
                rules.append((list(conditions), node.prediction))
                return
            for value, child in node.children.items():
                conditions.append((node.feature, str(value)))
                visit(child, conditions)
                conditions.pop()

        visit(self.root_, [])
        return rules

    def export_rules(self, path: str | Path) -> Path:
        path = Path(path)
        lines = []
        for conditions, label in self.decision_rules():
            body = (
                " ∧ ".join(f"[{name} = {value}]" for name, value in conditions)
                if conditions
                else "[any]"
            )
            lines.append(f"{body} → [{self.target_} = {label}]")
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return path

    def __str__(self) -> str:
        if not self._fitted or self.root_ is None:
            return "<unfitted GroupbyID3Classifier>"
        lines: list[str] = []

        def visit(node: _PandasNode, prefix: str, condition: str) -> None:
            label = node.prediction
            if node.is_leaf:
                lines.append(
                    f"{prefix}{condition}→ {label} (n={node.n_samples})"
                )
                return
            lines.append(
                f"{prefix}{condition}[{node.feature}?] gain={node.gain:.4f} "
                f"(n={node.n_samples})"
            )
            for value, child in node.children.items():
                visit(child, prefix + "  ", f"{node.feature}={value} ")

        visit(self.root_, "", "")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Small tree helpers
# ---------------------------------------------------------------------------


def _count_nodes(node: _PandasNode | None) -> int:
    if node is None:
        return 0
    return 1 + sum(_count_nodes(child) for child in node.children.values())


def _count_leaves(node: _PandasNode | None) -> int:
    if node is None:
        return 0
    if node.is_leaf:
        return 1
    return sum(_count_leaves(child) for child in node.children.values())


def _node_depth(node: _PandasNode | None) -> int:
    if node is None or node.is_leaf:
        return 0
    return 1 + max(_node_depth(child) for child in node.children.values())
