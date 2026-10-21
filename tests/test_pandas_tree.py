"""Tests for the pandas-first tree implementations."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pandas_dt import (
    GroupbyID3Classifier,
    PandasID3Classifier,
    load_raw,
)
from pandas_dt.data import encode_categorical, stratified_split


def make_frame(n: int = 240, seed: int = 0, missing: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    colors = rng.choice(["red", "blue", "green"], size=n)
    sizes = rng.choice(["S", "M", "L"], size=n)
    shapes = rng.choice(["round", "square"], size=n)
    label = np.where(
        (colors == "red") | (sizes == "L"),
        "positive",
        "negative",
    )
    # A little label noise makes the pruning test meaningful.
    flip = rng.random(n) < 0.08
    label = np.where(flip, np.where(label == "positive", "negative", "positive"), label)
    frame = pd.DataFrame(
        {
            "color": colors,
            "size": sizes,
            "shape": shapes,
            "label": label,
        },
        index=np.arange(1000, 1000 + n),
    )
    if missing:
        frame.loc[frame.index[:5], "color"] = None
    return frame


class PandasID3Tests(unittest.TestCase):
    def test_string_dataframe_and_index(self) -> None:
        train = make_frame(seed=1)
        test = make_frame(seed=2)
        tree = PandasID3Classifier(target="label", min_samples_leaf=2).fit(train)
        pred = tree.predict(test)
        self.assertIsInstance(pred, pd.Series)
        self.assertTrue(pred.index.equals(test.index))
        self.assertEqual(pred.name, "label")
        self.assertGreaterEqual(tree.score(test), 0.8)
        self.assertGreater(tree.n_nodes, 1)
        self.assertGreaterEqual(tree.n_leaves, 1)
        self.assertGreaterEqual(tree.depth, 1)

    def test_probabilities_and_importances(self) -> None:
        train = make_frame(seed=3)
        tree = PandasID3Classifier(target="label").fit(train)
        proba = tree.predict_proba(train)
        self.assertIsInstance(proba, pd.DataFrame)
        self.assertEqual(list(proba.columns), ["negative", "positive"])
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
        self.assertTrue(proba.index.equals(train.index))
        importances = tree.feature_importances_
        self.assertIsInstance(importances, pd.Series)
        self.assertEqual(list(importances.index), ["color", "size", "shape"])
        self.assertAlmostEqual(float(importances.sum()), 1.0)

    def test_missing_values(self) -> None:
        train = make_frame(seed=4, missing=True)
        test = make_frame(seed=5, missing=True)
        tree = PandasID3Classifier(target="label").fit(train)
        pred = tree.predict(test)
        self.assertEqual(len(pred), len(test))
        self.assertTrue(set(pred.unique()).issubset({"negative", "positive"}))
        # Prediction must be deterministic for missing/unseen values.
        np.testing.assert_array_equal(pred.to_numpy(), tree.predict(test).to_numpy())

    def test_unseen_category_fallback(self) -> None:
        train = make_frame(seed=6)
        tree = PandasID3Classifier(target="label").fit(train)
        probe = pd.DataFrame(
            {
                "color": ["purple"],
                "size": ["XL"],
                "shape": ["triangle"],
                "label": ["positive"],
            },
            index=[42],
        )
        first = tree.predict(probe).iloc[0]
        second = tree.predict(probe).iloc[0]
        self.assertEqual(first, second)
        self.assertIn(first, {"negative", "positive"})

    def test_categorical_dtype(self) -> None:
        train = make_frame(seed=7)
        train["color"] = train["color"].astype("category")
        train["size"] = train["size"].astype("category")
        tree = PandasID3Classifier(target="label").fit(train)
        self.assertGreaterEqual(tree.score(train), 0.75)

    def test_feature_subset(self) -> None:
        train = make_frame(seed=8)
        tree = PandasID3Classifier(target="label").fit(
            train, features=["color", "size"]
        )
        self.assertEqual(tree.feature_names_, ["color", "size"])
        self.assertGreaterEqual(tree.score(train), 0.75)

    def test_pruning_and_rules(self) -> None:
        train = make_frame(seed=9)
        val = make_frame(seed=10)
        unpruned = PandasID3Classifier(target="label", prune=False).fit(train)
        pruned = PandasID3Classifier(target="label", prune=True).fit(
            train, eval_set=val
        )
        self.assertLessEqual(pruned.n_nodes, unpruned.n_nodes)
        rules = pruned.decision_rules()
        self.assertGreater(len(rules), 0)
        with tempfile.TemporaryDirectory() as tmp:
            path = pruned.export_rules(Path(tmp) / "rules.txt")
            text = path.read_text(encoding="utf-8")
        self.assertIn("label =", text)
        self.assertIn("→", text)

    def test_invalid_inputs(self) -> None:
        frame = make_frame()
        with self.assertRaises(ValueError):
            PandasID3Classifier(target="missing").fit(frame)
        with self.assertRaises(ValueError):
            PandasID3Classifier(target="label").fit(frame, features=["color", "nope"])
        with self.assertRaises(ValueError):
            PandasID3Classifier(target="label").fit(frame.iloc[:0])

        three = frame.copy()
        three.loc[three.index[0], "label"] = "third"
        with self.assertRaises(ValueError):
            PandasID3Classifier(target="label").fit(three)

    def test_unfitted_predict_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            PandasID3Classifier(target="label").predict(make_frame())


class GroupbyID3Tests(unittest.TestCase):
    def test_string_dataframe_and_index(self) -> None:
        train = make_frame(seed=11)
        test = make_frame(seed=12)
        tree = GroupbyID3Classifier(target="label", min_samples_leaf=2).fit(train)
        pred = tree.predict(test)
        self.assertIsInstance(pred, pd.Series)
        self.assertTrue(pred.index.equals(test.index))
        self.assertEqual(pred.name, "label")
        self.assertGreaterEqual(tree.score(test), 0.75)
        self.assertGreater(tree.n_nodes, 1)

    def test_probabilities(self) -> None:
        train = make_frame(seed=13)
        tree = GroupbyID3Classifier(target="label").fit(train)
        proba = tree.predict_proba(train)
        self.assertIsInstance(proba, pd.DataFrame)
        self.assertEqual(list(proba.columns), ["negative", "positive"])
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
        self.assertTrue(proba.index.equals(train.index))

    def test_missing_unseen_and_rules(self) -> None:
        train = make_frame(seed=14, missing=True)
        test = make_frame(seed=15, missing=True)
        tree = GroupbyID3Classifier(target="label").fit(train)
        pred = tree.predict(test)
        np.testing.assert_array_equal(pred.to_numpy(), tree.predict(test).to_numpy())
        probe = pd.DataFrame(
            {
                "color": ["purple"],
                "size": ["XL"],
                "shape": ["triangle"],
                "label": ["negative"],
            }
        )
        np.testing.assert_array_equal(
            tree.predict(probe).to_numpy(), tree.predict(probe).to_numpy()
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = tree.export_rules(Path(tmp) / "rules.txt")
            text = path.read_text(encoding="utf-8")
        self.assertIn("label =", text)
        self.assertIn("→", text)

    def test_pruning_reduces_or_keeps_nodes(self) -> None:
        train = make_frame(seed=16)
        val = make_frame(seed=17)
        unpruned = GroupbyID3Classifier(target="label", prune=False).fit(train)
        pruned = GroupbyID3Classifier(target="label", prune=True).fit(
            train, eval_set=val
        )
        self.assertLessEqual(pruned.n_nodes, unpruned.n_nodes)

    def test_invalid_inputs(self) -> None:
        frame = make_frame()
        with self.assertRaises(ValueError):
            GroupbyID3Classifier(target="label").fit(frame.iloc[:0])
        with self.assertRaises(ValueError):
            GroupbyID3Classifier(target="label").fit(frame, features=[])
        with self.assertRaises(TypeError):
            GroupbyID3Classifier(target="label").fit("not a dataframe")  # type: ignore[arg-type]

    def test_unfitted_predict_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            GroupbyID3Classifier(target="label").predict(make_frame())


class RealDataPandasTreeTests(unittest.TestCase):
    def test_both_pandas_trees_on_breast_cancer(self) -> None:
        raw = load_raw()
        _, y, _ = encode_categorical(raw)
        train_idx, val_idx, test_idx = stratified_split(
            y, test_size=0.2, val_size=0.2, random_state=42
        )
        train = raw.iloc[train_idx]
        val = raw.iloc[val_idx]
        test = raw.iloc[test_idx]
        for cls in (PandasID3Classifier, GroupbyID3Classifier):
            tree = cls(target="Class").fit(train, eval_set=val)
            pred = tree.predict(test)
            self.assertEqual(len(pred), len(test))
            self.assertTrue(pred.index.equals(test.index))
            self.assertTrue(set(pred.unique()).issubset(
                {"no-recurrence-events", "recurrence-events"}
            ))
            self.assertGreaterEqual(tree.score(train), 0.6)


if __name__ == "__main__":
    unittest.main()
