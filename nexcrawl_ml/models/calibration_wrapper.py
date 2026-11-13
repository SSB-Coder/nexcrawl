"""
Post-hoc probability calibration for the LightGBM classifiers.

Why this exists: raw LightGBM predict_proba output is typically
overconfident near 0 and 1. In a safety system that matters - a "94% chance
of roof fall" that is really 60% will get people hurt or will train the
operators to ignore alerts.

IMPORTANT - cv='prefit' is used deliberately.

    cv='prefit'  -> calibrates the ALREADY-TRAINED model on a held-out set.
                    One LightGBM model ships. Predictable, small, fast.
    cv=5 (default) -> internally CLONES AND REFITS LightGBM on 5 folds and
                    averages them. You end up shipping 5 models, and on a
                    small hand-staged dataset each fold is data-starved.

With the ~300-sample staged dataset this project uses, 'prefit' is the
correct choice. Do not silently change it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from sklearn.calibration import CalibratedClassifierCV, calibration_curve
    from sklearn.metrics import brier_score_loss

    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    _HAS_SKLEARN = False


@dataclass
class CalibrationReport:
    brier_before: float
    brier_after: float
    improvement: float
    method: str
    n_calibration_samples: int

    def summary(self) -> str:
        direction = "improved" if self.improvement > 0 else "WORSENED"
        return (
            f"Brier {self.brier_before:.4f} -> {self.brier_after:.4f} "
            f"({direction} by {abs(self.improvement):.4f}) "
            f"via {self.method} on {self.n_calibration_samples} samples"
        )


def calibrate_model(
    fitted_model,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    method: str = "isotonic",
):
    """Wrap an already-fitted classifier with isotonic calibration.

    Note on method choice: isotonic is non-parametric and more flexible than
    sigmoid/Platt, but it needs more data and can overfit below ~100
    calibration samples. If your held-out set is small, sigmoid is safer.
    """
    if not _HAS_SKLEARN:
        raise ImportError("scikit-learn is required: pip install scikit-learn")

    if len(X_calib) < 50:
        raise ValueError(
            f"only {len(X_calib)} calibration samples; isotonic calibration "
            "will overfit. Use method='sigmoid' or collect more data."
        )

    # sklearn 1.6 removed cv='prefit' in favour of wrapping the estimator in
    # FrozenEstimator. Both express the same intent - "do not refit the base
    # model" - so support whichever this environment provides.
    try:
        from sklearn.frozen import FrozenEstimator

        calibrated = CalibratedClassifierCV(
            FrozenEstimator(fitted_model), method=method
        )
    except ImportError:  # sklearn < 1.6
        calibrated = CalibratedClassifierCV(
            fitted_model, method=method, cv="prefit"
        )

    calibrated.fit(X_calib, y_calib)
    return calibrated


def evaluate_calibration(
    raw_model,
    calibrated_model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    positive_class: int = 1,
    method: str = "isotonic",
) -> CalibrationReport:
    """Measure whether calibration actually helped, via Brier score.

    Always run this. Calibration is not guaranteed to improve things on a
    small dataset, and shipping a calibration layer that made things worse
    is a real failure mode.
    """
    if not _HAS_SKLEARN:
        raise ImportError("scikit-learn is required")

    y_binary = (np.asarray(y_test) == positive_class).astype(int)

    def _positive_proba(model):
        proba = model.predict_proba(X_test)
        classes = list(getattr(model, "classes_", range(proba.shape[1])))
        col = classes.index(positive_class) if positive_class in classes else 1
        return proba[:, col]

    brier_before = float(brier_score_loss(y_binary, _positive_proba(raw_model)))
    brier_after = float(brier_score_loss(y_binary, _positive_proba(calibrated_model)))

    return CalibrationReport(
        brier_before=brier_before,
        brier_after=brier_after,
        improvement=brier_before - brier_after,
        method=method,
        n_calibration_samples=len(X_test),
    )


def reliability_data(
    model, X: np.ndarray, y: np.ndarray, positive_class: int = 1, n_bins: int = 10
):
    """Reliability-diagram data. Plot this for the judges - a calibration
    curve hugging the diagonal is a very legible 'we did this properly'."""
    if not _HAS_SKLEARN:
        raise ImportError("scikit-learn is required")

    y_binary = (np.asarray(y) == positive_class).astype(int)
    proba = model.predict_proba(X)
    classes = list(getattr(model, "classes_", range(proba.shape[1])))
    col = classes.index(positive_class) if positive_class in classes else 1

    n_bins = min(n_bins, max(2, len(y) // 10))
    frac_pos, mean_pred = calibration_curve(y_binary, proba[:, col], n_bins=n_bins)
    return mean_pred, frac_pos
