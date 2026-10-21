"""A small, dependency-light decision-tree / GBDT toolkit built on pandas + NumPy.

The package is intentionally self contained: the core learners use only
``numpy`` and ``pandas`` so that the algorithms are transparent and easy to
audit.  scikit-learn is only used by the optional benchmark script.
"""

from .data import (
    COLUMNS,
    FEATURE_SPECS,
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    FeatureInfo,
    encode_categorical,
    encode_numeric,
    encode_onehot,
    load_raw,
    stratified_kfold,
    stratified_split,
)
from .tree import ID3Classifier, ID3Node
from .pandas_tree import GroupbyID3Classifier, PandasID3Classifier
from .gbdt import GBDTClassifier, RegressionTree
from .metrics import (
    accuracy_score,
    binary_metrics,
    bootstrap_accuracy_ci,
    log_loss_score,
    roc_auc_score,
    tune_threshold,
)
from .utils import load_model, save_json, save_model

__all__ = [
    "COLUMNS",
    "FEATURE_SPECS",
    "NEGATIVE_LABEL",
    "POSITIVE_LABEL",
    "FeatureInfo",
    "encode_categorical",
    "encode_numeric",
    "encode_onehot",
    "load_raw",
    "stratified_kfold",
    "stratified_split",
    "ID3Classifier",
    "PandasID3Classifier",
    "GroupbyID3Classifier",
    "ID3Node",
    "GBDTClassifier",
    "RegressionTree",
    "accuracy_score",
    "binary_metrics",
    "bootstrap_accuracy_ci",
    "log_loss_score",
    "roc_auc_score",
    "tune_threshold",
    "load_model",
    "save_json",
    "save_model",
]
