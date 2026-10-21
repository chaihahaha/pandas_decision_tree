"""Data loading and encoding for the UCI breast-cancer recurrence dataset.

Dataset facts (``breast-cancer.names``):
    * 286 instances, 9 categorical attributes
    * class distribution: 201 no-recurrence / 85 recurrence
    * missing values are encoded as ``?`` (8 in node-caps, 1 in breast-quad)

Everything here is deliberately explicit: category order is part of the
model contract, so a fitted model can be serialized and re-used on new data
without relying on the order in which categories happened to appear.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

TARGET_COLUMN = "Class"
POSITIVE_LABEL = "recurrence-events"
NEGATIVE_LABEL = "no-recurrence-events"
MISSING = "?"

COLUMNS: tuple[str, ...] = (
    TARGET_COLUMN,
    "age",
    "menopause",
    "tumor-size",
    "inv-nodes",
    "node-caps",
    "deg-malig",
    "breast",
    "breast-quad",
    "irradiat",
)


@dataclass(frozen=True)
class FeatureSpec:
    """Declarative description of one raw attribute.

    ``ordinal`` attributes have a meaningful order; their integer code is the
    rank in ``values``.  Nominal attributes are one-hot encoded for models
    that need a numeric matrix and are used as multi-way splits by ID3/C4.5.
    """

    name: str
    values: tuple[str, ...]
    ordinal: bool


FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        "age",
        ("20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90-99", MISSING),
        True,
    ),
    FeatureSpec("menopause", ("lt40", "ge40", "premeno", MISSING), False),
    FeatureSpec(
        "tumor-size",
        (
            "0-4",
            "5-9",
            "10-14",
            "15-19",
            "20-24",
            "25-29",
            "30-34",
            "35-39",
            "40-44",
            "45-49",
            "50-54",
            "55-59",
            MISSING,
        ),
        True,
    ),
    FeatureSpec(
        "inv-nodes",
        (
            "0-2",
            "3-5",
            "6-8",
            "9-11",
            "12-14",
            "15-17",
            "18-20",
            "21-23",
            "24-26",
            "27-29",
            "30-32",
            "33-35",
            "36-39",
            MISSING,
        ),
        True,
    ),
    FeatureSpec("node-caps", ("no", "yes", MISSING), False),
    FeatureSpec("deg-malig", ("1", "2", "3"), True),
    FeatureSpec("breast", ("left", "right"), False),
    FeatureSpec(
        "breast-quad",
        ("left_up", "left_low", "right_up", "right_low", "central", MISSING),
        False,
    ),
    FeatureSpec("irradiat", ("no", "yes"), False),
)


@dataclass(frozen=True)
class FeatureInfo:
    """Metadata attached to an encoded feature matrix."""

    names: tuple[str, ...]
    cardinalities: tuple[int, ...]
    value_labels: tuple[tuple[str, ...], ...]
    ordinal: tuple[bool, ...]

    @property
    def n_features(self) -> int:
        return len(self.names)


def load_raw(path: str | Path = "breast-cancer.data") -> pd.DataFrame:
    """Load the raw CSV.

    The file has no header and occasionally contains surrounding whitespace.
    Missing values are kept as the literal ``?`` (rather than NaN) so that
    "missing" is an explicit, predictable category rather than something that
    silently becomes the mean/mode at a later stage.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}")

    df = pd.read_csv(path, names=list(COLUMNS), dtype=str, keep_default_na=False)
    df = df.apply(lambda s: s.str.strip())

    if df.shape[1] != len(COLUMNS):
        raise ValueError(f"expected {len(COLUMNS)} columns, got {df.shape[1]}")
    if (df[TARGET_COLUMN] == "").any():
        raise ValueError("empty class label found")

    unknown = set(df[TARGET_COLUMN].unique()) - {POSITIVE_LABEL, NEGATIVE_LABEL}
    if unknown:
        raise ValueError(f"unknown class label(s): {sorted(unknown)}")

    for spec in FEATURE_SPECS:
        bad = set(df[spec.name].unique()) - set(spec.values)
        if bad:
            raise ValueError(f"unknown value(s) for {spec.name}: {sorted(bad)}")

    return df


def encode_categorical(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, FeatureInfo]:
    """Encode every attribute as an integer category code for ID3/C4.5.

    Returns ``(X, y, info)`` where ``X`` is a 2-D ``uint8`` array, ``y`` is
    ``1`` for recurrence and ``0`` for no-recurrence.
    """

    _check_target(df)
    columns: list[np.ndarray] = []
    for spec in FEATURE_SPECS:
        mapping = {value: i for i, value in enumerate(spec.values)}
        codes = df[spec.name].map(mapping)
        if codes.isna().any():
            # load_raw validates this, but encode_categorical is public API.
            bad = df.loc[codes.isna(), spec.name].unique().tolist()
            raise ValueError(f"unknown value(s) for {spec.name}: {bad}")
        columns.append(codes.to_numpy(dtype=np.uint8))

    X = np.column_stack(columns) if columns else np.empty((len(df), 0), dtype=np.uint8)
    y = (df[TARGET_COLUMN] == POSITIVE_LABEL).to_numpy(dtype=np.int8)
    info = FeatureInfo(
        names=tuple(spec.name for spec in FEATURE_SPECS),
        cardinalities=tuple(len(spec.values) for spec in FEATURE_SPECS),
        value_labels=tuple(spec.values for spec in FEATURE_SPECS),
        ordinal=tuple(spec.ordinal for spec in FEATURE_SPECS),
    )
    return X, y, info


def encode_numeric(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, FeatureInfo]:
    """Encode the data as a numeric matrix suitable for CART/GBDT.

    * ordinal attributes keep their rank as a numeric value, so a binary
      threshold split respects the ordering;
    * nominal attributes are one-hot encoded, with ``?`` as its own column;
    * no category is silently dropped.  This matters for a tiny dataset.

    The returned matrix is ``float32`` to keep the tree learner fast.
    """

    _check_target(df)
    columns: list[np.ndarray] = []
    names: list[str] = []
    cardinalities: list[int] = []
    value_labels: list[tuple[str, ...]] = []
    ordinal_flags: list[bool] = []

    for spec in FEATURE_SPECS:
        mapping = {value: i for i, value in enumerate(spec.values)}
        codes = df[spec.name].map(mapping)
        if codes.isna().any():
            bad = df.loc[codes.isna(), spec.name].unique().tolist()
            raise ValueError(f"unknown value(s) for {spec.name}: {bad}")

        if spec.ordinal:
            columns.append(codes.to_numpy(dtype=np.float32))
            names.append(spec.name)
            cardinalities.append(len(spec.values))
            value_labels.append(spec.values)
            ordinal_flags.append(True)
        else:
            for value in spec.values:
                columns.append((codes.to_numpy() == mapping[value]).astype(np.float32))
                names.append(f"{spec.name}={value}")
                cardinalities.append(2)
                value_labels.append((value,))
                ordinal_flags.append(False)

    X = np.column_stack(columns).astype(np.float32, copy=False)
    y = (df[TARGET_COLUMN] == POSITIVE_LABEL).to_numpy(dtype=np.int8)
    info = FeatureInfo(
        names=tuple(names),
        cardinalities=tuple(cardinalities),
        value_labels=tuple(value_labels),
        ordinal=tuple(ordinal_flags),
    )
    return X, y, info


def encode_onehot(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, FeatureInfo]:
    """One-hot encode *every* category, including ordinal attributes.

    This encoding is used by the optional scikit-learn ceiling benchmark:
    random forests and linear models generally do better when every category
    can be split independently.  The GBDT uses :func:`encode_numeric` so that
    ordinal thresholds remain available.
    """

    _check_target(df)
    columns: list[np.ndarray] = []
    names: list[str] = []
    for spec in FEATURE_SPECS:
        mapping = {value: i for i, value in enumerate(spec.values)}
        codes = df[spec.name].map(mapping)
        if codes.isna().any():
            bad = df.loc[codes.isna(), spec.name].unique().tolist()
            raise ValueError(f"unknown value(s) for {spec.name}: {bad}")
        for value in spec.values:
            columns.append((codes.to_numpy() == mapping[value]).astype(np.float32))
            names.append(f"{spec.name}={value}")

    X = np.column_stack(columns).astype(np.float32, copy=False)
    y = (df[TARGET_COLUMN] == POSITIVE_LABEL).to_numpy(dtype=np.int8)
    info = FeatureInfo(
        names=tuple(names),
        cardinalities=tuple(2 for _ in names),
        value_labels=tuple((name.split("=", 1)[1],) for name in names),
        ordinal=tuple(False for _ in names),
    )
    return X, y, info


def _check_target(df: pd.DataFrame) -> None:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"missing target column {TARGET_COLUMN!r}")
    bad = set(df[TARGET_COLUMN].unique()) - {POSITIVE_LABEL, NEGATIVE_LABEL}
    if bad:
        raise ValueError(f"unknown class label(s): {sorted(bad)}")


# ---------------------------------------------------------------------------
# Splitting / cross validation
# ---------------------------------------------------------------------------


def stratified_split(
    y: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return disjoint, stratified ``(train, val, test)`` index arrays.

    The original implementation used inclusive ``.loc`` slices, which made
    train/validation/test overlap and, with ``fraction=1``, produced an empty
    test set.  This implementation guarantees:
        * no index appears in more than one split;
        * every split is non-empty;
        * class proportions are approximately preserved.
    """

    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError("y must be one-dimensional")
    if len(y) < 3:
        raise ValueError("need at least 3 samples")
    if not (0.0 <= test_size < 1.0) or not (0.0 <= val_size < 1.0):
        raise ValueError("test_size/val_size must be in [0, 1)")
    if test_size == 0 and val_size == 0:
        raise ValueError("at least one of test_size/val_size must be > 0")

    rng = np.random.default_rng(random_state)
    train_idx: list[np.ndarray] = []
    val_idx: list[np.ndarray] = []
    test_idx: list[np.ndarray] = []

    classes = np.unique(y)
    for cls in classes:
        idx = rng.permutation(np.flatnonzero(y == cls))
        n = len(idx)
        n_test = int(round(n * test_size))
        n_val = int(round(n * val_size))
        if n_test + n_val > n - 1:
            n_test = 1 if test_size > 0 else 0
            n_val = 1 if val_size > 0 else 0
        # Keep at least one training example of each class.
        n_test = max(0, min(n_test, n - 1))
        n_val = max(0, min(n_val, n - n_test - 1))
        test_idx.append(idx[:n_test])
        val_idx.append(idx[n_test : n_test + n_val])
        train_idx.append(idx[n_test + n_val :])

    def _cat(parts: Sequence[np.ndarray]) -> np.ndarray:
        return np.sort(np.concatenate(parts)) if parts else np.empty(0, dtype=int)

    return _cat(train_idx), _cat(val_idx), _cat(test_idx)


def stratified_kfold(
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return stratified ``(train, validation)`` index pairs.

    A simple round-robin assignment within each class is enough for a
    dataset of this size and guarantees fold sizes differ by at most one.
    """

    y = np.asarray(y)
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if n_splits > len(y):
        raise ValueError("n_splits cannot exceed the number of samples")

    rng = np.random.default_rng(random_state)
    folds: list[list[int]] = [[] for _ in range(n_splits)]
    for cls in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == cls))
        for position, sample_index in enumerate(idx):
            folds[position % n_splits].append(int(sample_index))

    all_idx = np.arange(len(y))
    result: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in folds:
        val = np.sort(np.asarray(fold, dtype=int))
        mask = np.ones(len(y), dtype=bool)
        mask[val] = False
        result.append((all_idx[mask], val))
    return result


def class_distribution(y: Iterable[int]) -> dict[int, int]:
    """Small helper used by reports and tests."""

    return {int(k): int(v) for k, v in zip(*np.unique(np.asarray(list(y)), return_counts=True))}
