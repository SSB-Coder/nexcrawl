"""
LightGBM Model 2 - Gas Trend & Sensor Fault Classifier.

This is a SEPARATE model from the structural classifier: different features,
different label space, different training data, saved to a different file.
Same library, same three lines of fitting code - but not the same model, and
not a second training run of the same task.

Input : rolling-window statistics over the gas sensor stream
Output: {SAFE, SLOW_LEAK, RAPID_ACCUMULATION, SENSOR_FAULT}

Why trend features and not raw readings? A raw threshold alarm ("CH4 > 1.25%")
is a solved problem that needs no ML at all. The value added here is
detecting the *trajectory* - a slow leak that is still below threshold but
climbing steadily, or a sensor that has quietly died and is reporting a
suspiciously flat line.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import lightgbm as lgb

    _HAS_LGB = True
except ImportError:  # pragma: no cover
    _HAS_LGB = False

try:
    from sklearn.metrics import classification_report, confusion_matrix

    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    _HAS_SKLEARN = False

GAS_CLASSES: list[str] = [
    "SAFE",
    "SLOW_LEAK",
    "RAPID_ACCUMULATION",
    "SENSOR_FAULT",
]

GAS_FEATURE_NAMES: list[str] = [
    "mean",
    "std",
    "rate_of_rise",
    "max_minus_min",
    "jitter",
    "flatline_score",
    "last_value",
    "trend_r2",
]

GAS_SEVERITY: dict[str, int] = {
    "RAPID_ACCUMULATION": 4,
    "SLOW_LEAK": 3,
    "SENSOR_FAULT": 2,
    "SAFE": 0,
}


@dataclass
class GasTrainingReport:
    accuracy: float
    per_class_report: str
    confusion: np.ndarray
    feature_importance: dict[str, float]
    n_test: int


class GasTrendClassifier:
    """LightGBM classifier over rolling-window gas sensor statistics."""

    def __init__(
        self,
        n_estimators: int = 250,
        learning_rate: float = 0.05,
        num_leaves: int = 20,
        max_depth: int = 6,
        min_child_samples: int = 15,
        class_weight: str | dict | None = "balanced",
        random_state: int = 42,
    ):
        if not _HAS_LGB:
            raise ImportError("lightgbm is required: pip install lightgbm")

        self.params = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            min_child_samples=min_child_samples,
            class_weight=class_weight,
            random_state=random_state,
            verbose=-1,
        )
        self.model = lgb.LGBMClassifier(**self.params)
        self.classes_: list[str] = []
        self.is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> "GasTrendClassifier":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        if X.shape[1] != len(GAS_FEATURE_NAMES):
            raise ValueError(
                f"expected {len(GAS_FEATURE_NAMES)} features, got {X.shape[1]}"
            )

        fit_kwargs = {}
        if eval_set is not None:
            fit_kwargs["eval_set"] = [eval_set]
            fit_kwargs["callbacks"] = [lgb.early_stopping(30, verbose=False)]

        # See StructuralClassifier.fit for why feature_name is omitted.
        self.model.fit(X, y, **fit_kwargs)
        self.classes_ = [str(c) for c in self.model.classes_]
        self.n_train_samples = len(X)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        if self._is_booster():
            proba = self.predict_proba(X)
            return np.array([self.classes_[i] for i in proba.argmax(axis=1)])
        return self.model.predict(np.asarray(X, dtype=np.float64))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        if self._is_booster():
            proba = self.model.predict(X)
            # Binary boosters return a 1-D vector of P(positive class).
            if proba.ndim == 1:
                proba = np.column_stack([1.0 - proba, proba])
            return proba
        return self.model.predict_proba(X)

    def predict_with_severity(self, X: np.ndarray) -> list[dict]:
        self._check_fitted()
        proba = self.predict_proba(X)
        out = []
        for row in proba:
            idx = int(np.argmax(row))
            label = self.classes_[idx]
            out.append(
                {
                    "label": label,
                    "confidence": float(row[idx]),
                    "severity": GAS_SEVERITY.get(label, 0),
                    "distribution": {
                        cls: float(p) for cls, p in zip(self.classes_, row)
                    },
                }
            )
        return out


    def _is_booster(self) -> bool:
        """True when this instance was restored from a saved Booster file.

        A raw lgb.Booster exposes predict() returning class probabilities for
        multiclass, but has no predict_proba()/classes_. The save/load round
        trip therefore has to be handled explicitly rather than assuming the
        sklearn API is always present.
        """
        return not hasattr(self.model, "predict_proba")

    def feature_importance(self) -> dict[str, float]:
        self._check_fitted()
        if self._is_booster():
            return {
                n: float(s)
                for n, s in zip(GAS_FEATURE_NAMES, self.model.feature_importance())
            }
        return {
            n: float(s)
            for n, s in zip(GAS_FEATURE_NAMES, self.model.feature_importances_)
        }

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> GasTrainingReport:
        self._check_fitted()
        if not _HAS_SKLEARN:
            raise ImportError("scikit-learn is required")
        preds = self.predict(X_test)
        return GasTrainingReport(
            accuracy=float(np.mean(preds == np.asarray(y_test))),
            per_class_report=classification_report(y_test, preds, zero_division=0),
            confusion=confusion_matrix(y_test, preds),
            feature_importance=self.feature_importance(),
            n_test=len(y_test),
        )

    def save(self, path: str | Path) -> None:
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.booster_.save_model(str(path))
        path.with_suffix(".meta.json").write_text(
            json.dumps(
                {
                    "classes": self.classes_,
                    "features": GAS_FEATURE_NAMES,
                    "params": self.params,
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, path: str | Path) -> "GasTrendClassifier":
        path = Path(path)
        obj = cls.__new__(cls)
        obj.model = lgb.Booster(model_file=str(path))
        meta_path = path.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        obj.classes_ = meta.get("classes", GAS_CLASSES)
        obj.params = meta.get("params", {})
        obj.is_fitted = True
        return obj

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("model is not fitted - call fit() first")
