"""Tests for StructuralClassifier and GasTrendClassifier: fit/predict/save/load.

The save/load round trip specifically regression-tests a real bug: load()
used to return a raw lgb.Booster with no predict_proba/classes_, so a model
that trained and evaluated perfectly would still fail at inference after
being reloaded from disk - exactly the kind of gap that only shows up in
production, never in a training notebook.
"""

import numpy as np
import pytest

from nexcrawl_ml.change_detection.features import FEATURE_NAMES
from nexcrawl_ml.models.gas_classifier import (
    GAS_FEATURE_NAMES,
    GasTrendClassifier,
)
from nexcrawl_ml.models.structural_classifier import (
    CLASS_SEVERITY,
    STRUCTURAL_CLASSES,
    StructuralClassifier,
    stratified_split,
)


def _structural_data(n_per_class=40, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for i, cls in enumerate(STRUCTURAL_CLASSES):
        centre = np.full(len(FEATURE_NAMES), i * 2.0)
        X.append(rng.normal(centre, 0.3, (n_per_class, len(FEATURE_NAMES))))
        y += [cls] * n_per_class
    return np.abs(np.vstack(X)), np.array(y)


def _gas_data(n_per_class=40, seed=1):
    from nexcrawl_ml.models.gas_classifier import GAS_CLASSES

    rng = np.random.default_rng(seed)
    X, y = [], []
    for i, cls in enumerate(GAS_CLASSES):
        centre = np.full(len(GAS_FEATURE_NAMES), i * 1.5)
        X.append(rng.normal(centre, 0.2, (n_per_class, len(GAS_FEATURE_NAMES))))
        y += [cls] * n_per_class
    return np.vstack(X), np.array(y)


class TestStructuralClassifier:
    def test_fit_predict_roundtrip(self):
        X, y = _structural_data()
        clf = StructuralClassifier(n_estimators=50)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert len(preds) == len(y)
        assert set(clf.classes_) == set(STRUCTURAL_CLASSES)

    def test_predict_proba_sums_to_one(self):
        X, y = _structural_data()
        clf = StructuralClassifier(n_estimators=50).fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (len(X), len(STRUCTURAL_CLASSES))
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_predict_with_severity_uses_class_severity(self):
        X, y = _structural_data()
        clf = StructuralClassifier(n_estimators=50).fit(X, y)
        results = clf.predict_with_severity(X[:5])
        for r in results:
            assert r["severity"] == CLASS_SEVERITY[r["label"]]
            assert 0.0 <= r["confidence"] <= 1.0
            assert "distribution" in r

    def test_predict_before_fit_raises(self):
        clf = StructuralClassifier()
        with pytest.raises(RuntimeError):
            clf.predict(np.zeros((1, len(FEATURE_NAMES))))

    def test_wrong_feature_count_raises(self):
        X, y = _structural_data()
        clf = StructuralClassifier(n_estimators=20)
        with pytest.raises(ValueError):
            clf.fit(X[:, :2], y)

    def test_feature_importance_keys_match_feature_names(self):
        X, y = _structural_data()
        clf = StructuralClassifier(n_estimators=50).fit(X, y)
        importance = clf.feature_importance()
        assert set(importance.keys()) == set(FEATURE_NAMES)

    def test_save_load_roundtrip_booster_supports_full_api(self, tmp_path):
        """Regression test: load() must return something with a working
        predict_proba/classes_/feature_importance, not a bare Booster."""
        X, y = _structural_data()
        clf = StructuralClassifier(n_estimators=50).fit(X, y)
        model_path = tmp_path / "structural_model.txt"
        clf.save(model_path)

        assert model_path.exists()
        assert model_path.with_suffix(".meta.json").exists()

        loaded = StructuralClassifier.load(model_path)
        assert loaded.classes_ == clf.classes_

        proba_orig = clf.predict_proba(X[:5])
        proba_loaded = loaded.predict_proba(X[:5])
        assert np.allclose(proba_orig, proba_loaded, atol=1e-6)

        preds_loaded = loaded.predict(X[:5])
        assert len(preds_loaded) == 5

        importance = loaded.feature_importance()
        assert set(importance.keys()) == set(FEATURE_NAMES)

    def test_stratified_split_produces_three_disjoint_sets(self):
        X, y = _structural_data(n_per_class=60)
        X_train, y_train, X_calib, y_calib, X_test, y_test = stratified_split(X, y)
        total = len(X_train) + len(X_calib) + len(X_test)
        assert total == len(X)
        # every class must appear in every split (stratification)
        for split_y in (y_train, y_calib, y_test):
            assert set(split_y) == set(STRUCTURAL_CLASSES)


class TestGasTrendClassifier:
    def test_fit_predict_roundtrip(self):
        X, y = _gas_data()
        clf = GasTrendClassifier(n_estimators=50).fit(X, y)
        preds = clf.predict(X)
        assert len(preds) == len(y)

    def test_predict_with_severity(self):
        X, y = _gas_data()
        clf = GasTrendClassifier(n_estimators=50).fit(X, y)
        results = clf.predict_with_severity(X[:5])
        for r in results:
            assert "severity" in r and "confidence" in r

    def test_feature_importance_keys_match_gas_feature_names(self):
        X, y = _gas_data()
        clf = GasTrendClassifier(n_estimators=50).fit(X, y)
        importance = clf.feature_importance()
        assert set(importance.keys()) == set(GAS_FEATURE_NAMES)

    def test_save_load_roundtrip(self, tmp_path):
        X, y = _gas_data()
        clf = GasTrendClassifier(n_estimators=50).fit(X, y)
        path = tmp_path / "gas_model.txt"
        clf.save(path)

        loaded = GasTrendClassifier.load(path)
        assert np.allclose(
            clf.predict_proba(X[:5]), loaded.predict_proba(X[:5]), atol=1e-6
        )
        # This exact call used to be a latent NameError risk (FEATURE_NAMES
        # referenced but never imported in gas_classifier.py).
        importance = loaded.feature_importance()
        assert set(importance.keys()) == set(GAS_FEATURE_NAMES)
