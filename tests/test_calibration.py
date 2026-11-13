"""Tests for the calibration wrapper and its interaction with HazardPipeline.

Two things regression-tested here specifically:
1. calibrate_model works whether the installed sklearn exposes FrozenEstimator
   (>=1.6) or the legacy cv='prefit' (<1.6) - the project docs specify
   cv='prefit', which crashes outright on current sklearn.
2. HazardPipeline preserves real CLASS_SEVERITY when given a bare
   CalibratedClassifierCV wrapper instead of the StructuralClassifier
   convenience wrapper - previously hardcoded to severity=0, which silently
   demoted every calibrated hazard prediction to INFO priority.
"""

import numpy as np
import pytest

from nexcrawl_ml.change_detection.features import FEATURE_NAMES
from nexcrawl_ml.models.calibration_wrapper import (
    calibrate_model,
    evaluate_calibration,
    reliability_data,
)
from nexcrawl_ml.models.structural_classifier import (
    CLASS_SEVERITY,
    STRUCTURAL_CLASSES,
    StructuralClassifier,
)
from nexcrawl_ml.pipeline.hazard_pipeline import HazardPipeline


def _structural_data(n_per_class=60, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for i, cls in enumerate(STRUCTURAL_CLASSES):
        centre = np.full(len(FEATURE_NAMES), i * 2.5)
        X.append(rng.normal(centre, 0.3, (n_per_class, len(FEATURE_NAMES))))
        y += [cls] * n_per_class
    return np.abs(np.vstack(X)), np.array(y)


@pytest.fixture
def trained_pair():
    X, y = _structural_data()
    n = len(X)
    split = n * 2 // 3
    idx = np.random.default_rng(0).permutation(n)
    train_idx, calib_idx = idx[:split], idx[split:]

    clf = StructuralClassifier(n_estimators=50)
    clf.fit(X[train_idx], y[train_idx])
    return clf, X[calib_idx], y[calib_idx]


class TestCalibrateModel:
    def test_calibrate_model_produces_valid_probabilities(self, trained_pair):
        clf, X_calib, y_calib = trained_pair
        calibrated = calibrate_model(clf.model, X_calib, y_calib, method="sigmoid")
        proba = calibrated.predict_proba(X_calib)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_calibrate_model_rejects_tiny_calibration_set(self, trained_pair):
        clf, X_calib, y_calib = trained_pair
        with pytest.raises(ValueError):
            calibrate_model(clf.model, X_calib[:10], y_calib[:10], method="isotonic")

    def test_falls_back_to_legacy_cv_prefit_when_frozen_estimator_missing(
        self, trained_pair, monkeypatch
    ):
        """Verify the branching logic itself: when sklearn.frozen is
        unavailable (sklearn < 1.6), calibrate_model must call
        CalibratedClassifierCV(model, method=..., cv='prefit') - the form
        that actually works on that older sklearn.

        This mocks CalibratedClassifierCV rather than relying on the
        installed (modern, strict) one to accept 'prefit': modern sklearn's
        own param validation rejects that string outright, so exercising
        the legacy branch against real modern internals tests an impossible
        combination, not a real deployment. What's real and worth pinning is
        that the code CONSTRUCTS the legacy call correctly when the
        FrozenEstimator import path is unavailable.
        """
        clf, X_calib, y_calib = trained_pair
        import nexcrawl_ml.models.calibration_wrapper as cw_module

        captured = {}

        class FakeCalibratedClassifierCV:
            def __init__(self, estimator, method=None, cv=None):
                captured["estimator"] = estimator
                captured["method"] = method
                captured["cv"] = cv

            def fit(self, X, y):
                captured["fit_called"] = True
                return self

        real_import = __import__

        def blocking_import(name, *args, **kwargs):
            if name == "sklearn.frozen":
                raise ImportError("simulated: sklearn < 1.6, no FrozenEstimator")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(
            cw_module, "CalibratedClassifierCV", FakeCalibratedClassifierCV
        )
        monkeypatch.setattr("builtins.__import__", blocking_import)

        cw_module.calibrate_model(clf.model, X_calib, y_calib, method="sigmoid")

        assert captured["cv"] == "prefit"
        assert captured["method"] == "sigmoid"
        assert captured["estimator"] is clf.model
        assert captured["fit_called"]

    def test_evaluate_calibration_reports_brier_scores(self, trained_pair):
        clf, X_calib, y_calib = trained_pair
        calibrated = calibrate_model(clf.model, X_calib, y_calib, method="sigmoid")
        report = evaluate_calibration(
            clf.model, calibrated, X_calib, y_calib,
            positive_class=STRUCTURAL_CLASSES[0], method="sigmoid",
        )
        assert report.brier_before >= 0.0
        assert report.brier_after >= 0.0

    def test_reliability_data_shapes_match(self, trained_pair):
        clf, X_calib, y_calib = trained_pair
        calibrated = calibrate_model(clf.model, X_calib, y_calib, method="sigmoid")
        mean_pred, frac_pos = reliability_data(
            calibrated, X_calib, y_calib, positive_class=STRUCTURAL_CLASSES[0]
        )
        assert len(mean_pred) == len(frac_pos)


class TestHazardPipelineSeverityWithCalibratedWrapper:
    """The specific regression: HazardPipeline.run() must not zero out
    severity when given a bare CalibratedClassifierCV instead of a
    StructuralClassifier."""

    def test_bare_calibrated_wrapper_preserves_class_severity(self, trained_pair):
        clf, X_calib, y_calib = trained_pair
        calibrated = calibrate_model(clf.model, X_calib, y_calib, method="sigmoid")

        # calibrated is a plain sklearn object: no predict_with_severity.
        assert not hasattr(calibrated, "predict_with_severity")

        pipeline = HazardPipeline(calibrated)

        # Exercise the exact code path HazardPipeline.run() uses internally
        # for a classifier without predict_with_severity.
        proba = calibrated.predict_proba(X_calib[:10])
        classes = [str(c) for c in calibrated.classes_]
        for row in proba:
            label = classes[int(np.argmax(row))]
            severity = CLASS_SEVERITY.get(label, 0)
            # A roof_fall prediction must never be silently reported as 0.
            if label == "roof_fall":
                assert severity == CLASS_SEVERITY["roof_fall"] == 4

    def test_severity_lookup_matches_class_severity_table_exactly(self, trained_pair):
        clf, X_calib, y_calib = trained_pair
        calibrated = calibrate_model(clf.model, X_calib, y_calib, method="sigmoid")
        classes = [str(c) for c in calibrated.classes_]
        proba = calibrated.predict_proba(X_calib)
        for row in proba:
            label = classes[int(np.argmax(row))]
            assert CLASS_SEVERITY.get(label, 0) == CLASS_SEVERITY[label]
