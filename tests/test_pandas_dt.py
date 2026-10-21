"""Unit tests for the fixed decision tree and from-scratch GBDT.

Run with:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from pandas_dt import (
    GBDTClassifier,
    ID3Classifier,
    accuracy_score,
    binary_metrics,
    bootstrap_accuracy_ci,
    encode_categorical,
    encode_numeric,
    encode_onehot,
    load_model,
    load_raw,
    roc_auc_score,
    save_model,
    stratified_kfold,
    stratified_split,
    tune_threshold,
)
from pandas_dt.data import FeatureInfo
from pandas_dt.gbdt import RegressionTree, _sigmoid, log_loss_score
from pandas_dt.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "breast-cancer.data"


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = load_raw(DATA)

    def test_raw_shape_and_classes(self) -> None:
        self.assertEqual(self.raw.shape, (286, 10))
        counts = self.raw["Class"].value_counts().to_dict()
        self.assertEqual(counts["no-recurrence-events"], 201)
        self.assertEqual(counts["recurrence-events"], 85)

    def test_missing_values_are_explicit(self) -> None:
        self.assertEqual(int((self.raw["node-caps"] == "?").sum()), 8)
        self.assertEqual(int((self.raw["breast-quad"] == "?").sum()), 1)

    def test_categorical_encoding(self) -> None:
        X, y, info = encode_categorical(self.raw)
        self.assertEqual(X.shape, (286, 9))
        self.assertEqual(y.shape, (286,))
        self.assertEqual(set(np.unique(y)), {0, 1})
        self.assertEqual(len(info.names), 9)
        for j, cardinality in enumerate(info.cardinalities):
            self.assertTrue(np.all(X[:, j] < cardinality))

    def test_numeric_encoding(self) -> None:
        X, y, _ = encode_numeric(self.raw)
        self.assertEqual(X.shape, (286, 21))
        self.assertEqual(X.dtype, np.float32)
        self.assertFalse(np.isnan(X).any())
        self.assertEqual(y.sum(), 85)

    def test_onehot_encoding(self) -> None:
        X, y, info = encode_onehot(self.raw)
        self.assertEqual(X.shape[0], 286)
        self.assertEqual(X.shape[1], len(info.names))
        self.assertEqual(set(np.unique(X)), {0.0, 1.0})
        self.assertEqual(y.sum(), 85)

    def test_bad_category_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.data"
            lines = [
                "no-recurrence-events,30-39,premeno,30-34,0-2,no,3,left,left_low,no"
            ]
            lines.append(
                "recurrence-events,30-39,premeno,30-34,0-2,no,3,left,bogus,no"
            )
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_raw(path)

    def test_missing_file_is_rejected(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_raw(ROOT / "does-not-exist.data")

    def test_split_is_disjoint_and_stratified(self) -> None:
        _, y, _ = encode_categorical(self.raw)
        train, val, test = stratified_split(y, 0.2, 0.2, random_state=42)
        self.assertEqual(len(train) + len(val) + len(test), len(y))
        self.assertEqual(len(set(train) & set(val)), 0)
        self.assertEqual(len(set(train) & set(test)), 0)
        self.assertEqual(len(set(val) & set(test)), 0)
        np.testing.assert_array_equal(
            np.sort(np.concatenate([train, val, test])), np.arange(len(y))
        )
        self.assertGreater(len(val), 0)
        self.assertGreater(len(test), 0)
        # Class proportions should be close to the full data.
        full_rate = y.mean()
        for part in (train, val, test):
            self.assertLess(abs(y[part].mean() - full_rate), 0.06)

    def test_split_shuffles_class_sorted_file(self) -> None:
        # The raw file is sorted by class: rows 0..200 are negative and
        # 201..285 are positive.  A correct split must not simply take the
        # first/last rows, otherwise accuracy would be trivially inflated.
        _, y, _ = encode_categorical(self.raw)
        train, val, test = stratified_split(y, 0.2, 0.2, random_state=42)
        self.assertGreater(len(set(train) & set(range(201))), 0)
        self.assertGreater(len(set(train) & set(range(201, 286))), 0)
        self.assertGreater(len(set(test) & set(range(201))), 0)
        self.assertGreater(len(set(test) & set(range(201, 286))), 0)
        self.assertLess(abs(y[test].mean() - y.mean()), 0.06)

    def test_split_without_test_set(self) -> None:
        _, y, _ = encode_categorical(self.raw)
        train, val, test = stratified_split(y, test_size=0.0, val_size=0.2, random_state=1)
        self.assertEqual(len(test), 0)
        self.assertGreater(len(train), 0)
        self.assertGreater(len(val), 0)
        self.assertEqual(len(set(train) & set(val)), 0)

    def test_kfold_partitions_every_sample_once(self) -> None:
        _, y, _ = encode_categorical(self.raw)
        folds = stratified_kfold(y, n_splits=5, random_state=3)
        seen = np.zeros(len(y), dtype=int)
        for _, valid in folds:
            seen[valid] += 1
        np.testing.assert_array_equal(seen, np.ones(len(y), dtype=int))
        for train, valid in folds:
            self.assertEqual(len(set(train) & set(valid)), 0)
            self.assertEqual(len(train) + len(valid), len(y))


class ID3Tests(unittest.TestCase):
    def test_perfect_split(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.integers(0, 4, size=(200, 2))
        y = (X[:, 0] >= 2).astype(np.int8)
        info = FeatureInfo(
            names=("a", "b"),
            cardinalities=(4, 4),
            value_labels=(("0", "1", "2", "3"), ("0", "1", "2", "3")),
            ordinal=(False, False),
        )
        model = ID3Classifier(min_samples_leaf=1, prune=False).fit(X, y, feature_info=info)
        self.assertGreaterEqual(accuracy_score(y, model.predict(X)), 0.99)
        self.assertEqual(model.n_features_, 2)
        self.assertGreater(model.n_nodes, 1)

    def test_missing_and_unseen_values_are_deterministic(self) -> None:
        X = np.array([[0], [1], [0], [1], [2], [2]], dtype=int)
        y = np.array([0, 1, 0, 1, 0, 1], dtype=np.int8)
        info = FeatureInfo(
            names=("a",),
            cardinalities=(4,),
            value_labels=(("0", "1", "2", "3"),),
            ordinal=(False,),
        )
        model = ID3Classifier(min_samples_leaf=1, prune=False).fit(X, y, feature_info=info)
        probe = np.array([[2], [3], [3], [2]], dtype=int)
        first = model.predict_proba(probe)
        second = model.predict_proba(probe)
        np.testing.assert_allclose(first, second)
        # Code 3 never appeared in training; it must fall back to a node's
        # stored majority, not a random branch.
        self.assertTrue(np.isfinite(first).all())
        self.assertTrue(np.allclose(first.sum(axis=1), 1.0))

    def test_single_class_training(self) -> None:
        X = np.array([[0, 1], [1, 1], [0, 2]], dtype=int)
        y = np.zeros(3, dtype=np.int8)
        model = ID3Classifier(prune=False).fit(X, y)
        self.assertTrue(model.root_.is_leaf)
        self.assertEqual(model.root_.prediction, 0)
        np.testing.assert_array_equal(model.predict(X), np.zeros(3, dtype=np.int8))

    def test_pruning_reduces_or_keeps_nodes(self) -> None:
        rng = np.random.default_rng(1)
        X = rng.integers(0, 5, size=(300, 3))
        y = ((X[:, 0] + X[:, 1] > 4) ^ (X[:, 2] > 2)).astype(np.int8)
        model = ID3Classifier(min_samples_leaf=1, prune=False).fit(X, y)
        before = model.n_nodes
        X_val = rng.integers(0, 5, size=(200, 3))
        y_val = ((X_val[:, 0] + X_val[:, 1] > 4) ^ (X_val[:, 2] > 2)).astype(np.int8)
        model.prune_tree(X_val, y_val)
        self.assertLessEqual(model.n_nodes, before)
        self.assertGreaterEqual(model.pruned_nodes_, 0)

    def test_empty_validation_does_not_prune(self) -> None:
        X = np.array([[0], [1], [0], [1]], dtype=int)
        y = np.array([0, 0, 1, 1], dtype=np.int8)
        model = ID3Classifier(prune=False).fit(X, y)
        before = model.n_nodes
        model.prune_tree(np.empty((0, 1), dtype=int), np.empty(0, dtype=np.int8))
        self.assertEqual(model.n_nodes, before)

    def test_rules_export(self) -> None:
        X = np.array([[0], [1], [0], [1]], dtype=int)
        y = np.array([0, 1, 0, 1], dtype=np.int8)
        model = ID3Classifier(prune=False).fit(X, y)
        with tempfile.TemporaryDirectory() as tmp:
            path = model.export_rules(Path(tmp) / "rules.txt")
            text = path.read_text(encoding="utf-8")
        self.assertIn("Class =", text)
        self.assertIn("→", text)
        self.assertGreaterEqual(len(model.decision_rules()), 1)
        self.assertAlmostEqual(float(model.feature_importances_.sum()), 1.0)

    def test_bad_inputs(self) -> None:
        X = np.array([[0], [1]], dtype=int)
        y = np.array([0, 1], dtype=np.int8)
        with self.assertRaises(ValueError):
            ID3Classifier().fit(X.astype(float) + 0.5, y)
        with self.assertRaises(ValueError):
            ID3Classifier().fit(X, np.array([0, 2], dtype=np.int8))
        with self.assertRaises(ValueError):
            ID3Classifier().fit(np.empty((0, 1)), np.empty(0, dtype=np.int8))
        model = ID3Classifier().fit(X, np.array([0, 0], dtype=np.int8))
        with self.assertRaises(ValueError):
            model.predict(np.zeros((2, 3), dtype=int))


class GBDTTests(unittest.TestCase):
    def _xor_data(self, n: int = 600, seed: int = 5):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, 4)).astype(np.float32)
        y = (((X[:, 0] + X[:, 1]) > 0) ^ (X[:, 2] > 0)).astype(np.int8)
        return X, y

    def test_sigmoid_is_stable(self) -> None:
        values = np.array([-1000.0, -50.0, 0.0, 50.0, 1000.0])
        p = _sigmoid(values)
        self.assertTrue(np.isfinite(p).all())
        np.testing.assert_allclose(p[[0, 4]], [0.0, 1.0], atol=1e-12)
        self.assertAlmostEqual(float(p[2]), 0.5)

    def test_regression_tree_fits_constant_gradients(self) -> None:
        X = np.array([[0.0], [0.0], [1.0], [1.0]], dtype=np.float32)
        g = np.array([-1.0, -1.0, 1.0, 1.0])
        h = np.ones(4)
        tree = RegressionTree(max_depth=1, min_samples_leaf=1, random_state=0)
        tree.fit(X, g, h)
        pred = tree.predict(X)
        self.assertEqual(pred[0], pred[1])
        self.assertEqual(pred[2], pred[3])
        # g = p - y, so a positive leaf value increases the raw score for
        # samples whose gradient is negative (positive class).
        self.assertGreater(pred[0], pred[2])
        self.assertGreaterEqual(tree.n_leaves, 2)

    def test_learns_xor(self) -> None:
        X, y = self._xor_data()
        model = GBDTClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=3,
            min_samples_leaf=5,
            subsample=0.8,
            max_features=0.8,
            random_state=0,
            early_stopping_rounds=None,
        )
        model.fit(X, y)
        accuracy = accuracy_score(y, model.predict(X))
        self.assertGreater(accuracy, 0.85)
        proba = model.predict_proba(X)
        self.assertEqual(proba.shape, (len(y), 2))
        self.assertTrue(np.all((proba >= 0) & (proba <= 1)))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)

    def test_early_stopping_and_threshold(self) -> None:
        X, y = self._xor_data(n=400, seed=9)
        train, val = X[:300], X[300:]
        y_train, y_val = y[:300], y[300:]
        model = GBDTClassifier(
            n_estimators=300, learning_rate=0.1, max_depth=3, min_samples_leaf=5,
            random_state=0, early_stopping_rounds=10,
        ).fit(train, y_train, eval_set=(val, y_val))
        self.assertLessEqual(model.n_used_trees, 300)
        self.assertGreaterEqual(model.n_used_trees, 1)
        low = model.predict(val, threshold=0.2)
        high = model.predict(val, threshold=0.8)
        self.assertGreaterEqual(int(low.sum()), int(high.sum()))

    def test_single_class_training(self) -> None:
        rng = np.random.default_rng(2)
        X = rng.normal(size=(50, 3)).astype(np.float32)
        y = np.zeros(50, dtype=np.int8)
        model = GBDTClassifier(n_estimators=20, random_state=0).fit(X, y)
        proba = model.predict_proba(X)
        self.assertTrue(np.isfinite(proba).all())
        self.assertTrue(np.all(proba[:, 1] < 0.01))
        np.testing.assert_array_equal(model.predict(X), np.zeros(50, dtype=np.int8))

    def test_feature_importance_sums_to_one(self) -> None:
        X, y = self._xor_data(n=200)
        model = GBDTClassifier(n_estimators=50, random_state=0).fit(X, y)
        self.assertAlmostEqual(float(model.feature_importances_.sum()), 1.0, places=6)

    def test_log_loss(self) -> None:
        self.assertLess(log_loss_score([0, 0, 1, 1], [0.01, 0.01, 0.99, 0.99]), 0.1)
        self.assertGreater(log_loss_score([0, 0, 1, 1], [0.49, 0.49, 0.51, 0.51]), 0.6)


class MetricTests(unittest.TestCase):
    def test_accuracy_and_confusion(self) -> None:
        y = [0, 0, 1, 1]
        p = [0.1, 0.2, 0.8, 0.9]
        self.assertEqual(accuracy_score(y, (np.asarray(p) >= 0.5).astype(int)), 1.0)
        matrix = confusion_matrix(y, (np.asarray(p) >= 0.5).astype(int))
        np.testing.assert_array_equal(matrix, np.array([[2, 0], [0, 2]]))
        metrics = binary_metrics(y, p)
        self.assertAlmostEqual(metrics["accuracy"], 1.0)
        self.assertAlmostEqual(metrics["precision"], 1.0)
        self.assertAlmostEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["confusion_matrix"], [[2, 0], [0, 2]])

    def test_auc(self) -> None:
        self.assertAlmostEqual(roc_auc_score([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertAlmostEqual(roc_auc_score([0, 1], [0.5, 0.5]), 0.5)

    def test_threshold_tuning(self) -> None:
        threshold, accuracy = tune_threshold([0, 0, 0, 1], [0.1, 0.2, 0.6, 0.7])
        self.assertGreater(threshold, 0.6)
        self.assertEqual(accuracy, 1.0)

    def test_bootstrap_ci(self) -> None:
        lo, hi = bootstrap_accuracy_ci([0, 1, 0, 1] * 10, [0, 1, 0, 1] * 10, n_bootstrap=200)
        self.assertLessEqual(lo, hi)
        self.assertAlmostEqual(lo, 1.0)
        self.assertAlmostEqual(hi, 1.0)


class SerializationTests(unittest.TestCase):
    def test_gbdt_roundtrip(self) -> None:
        rng = np.random.default_rng(3)
        X = rng.normal(size=(100, 3)).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.int8)
        model = GBDTClassifier(n_estimators=30, max_depth=2, random_state=0).fit(X, y)
        expected = model.predict_proba(X)
        with tempfile.TemporaryDirectory() as tmp:
            path = save_model(model, Path(tmp) / "model.pkl", metadata={"seed": 0})
            loaded, metadata = load_model(path)
        np.testing.assert_allclose(loaded.predict_proba(X), expected)
        self.assertEqual(metadata["seed"], 0)

    def test_id3_roundtrip(self) -> None:
        X = np.array([[0], [1], [0], [1]], dtype=int)
        y = np.array([0, 1, 0, 1], dtype=np.int8)
        model = ID3Classifier(prune=False).fit(X, y)
        expected = model.predict(X)
        with tempfile.TemporaryDirectory() as tmp:
            path = save_model(model, Path(tmp) / "tree.pkl")
            loaded, _ = load_model(path)
        np.testing.assert_array_equal(loaded.predict(X), expected)


class RealDataSmokeTests(unittest.TestCase):
    def test_both_models_run_on_real_data(self) -> None:
        raw = load_raw(DATA)
        X_cat, y, cat_info = encode_categorical(raw)
        X_num, _, _ = encode_numeric(raw)
        train, val, test = stratified_split(y, 0.2, 0.2, random_state=42)

        tree = ID3Classifier(prune=True).fit(
            X_cat[train], y[train], feature_info=cat_info, eval_set=(X_cat[val], y[val])
        )
        self.assertEqual(tree.predict(X_cat[test]).shape, (len(test),))

        gbdt = GBDTClassifier(n_estimators=40, random_state=0).fit(
            X_num[train], y[train], eval_set=(X_num[val], y[val])
        )
        proba = gbdt.predict_proba(X_num[test])[:, 1]
        self.assertEqual(proba.shape, (len(test),))
        self.assertTrue(np.all((proba >= 0) & (proba <= 1)))


if __name__ == "__main__":
    unittest.main()
