"""
LightGBM Model 1 - Structural Change Classifier.

Input : 5 geometric features from a DBSCAN hazard blob
Output: {roof_fall, rib_spall, floor_heave, equipment_moved, artefact}

Why LightGBM and not a deep 3D model (PointNet / 3D CNN)? With ~300 staged
training samples, a deep point-cloud network will memorise the training set.
Gradient-boosted trees on 5 hand-designed, physically-meaningful features
are the right capacity for this data budget - and they give you feature
importances you can actually explain to a judge.
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
    import joblib
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split

    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    _HAS_SKLEARN = False

from ..change_detection.features import FEATURE_NAMES

STRUCTURAL_CLASSES: list[str] = [
    "roof_fall",
    "rib_spall",
    "floor_heave",
    "equipment_moved",
    "artefact",
]

# Severity ordering drives alert priority downstream.
CLASS_SEVERITY: dict[str, int] = {
    "roof_fall": 4,
    "rib_spall": 3,
    "floor_heave": 2,
    "equipment_moved": 1,
    "artefact": 0,
}


@dataclass
class TrainingReport:
    accuracy: float
    per_class_report: str
    confusion: np.ndarray
    feature_importance: dict[str, float]
    n_train: int
    n_test: int

    def summary(self) -> str:
        top = sorted(
            self.feature_importance.items(), key=lambda kv: kv[1], reverse=True
        )
        lines = [
            f"Accuracy: {self.accuracy:.3f}  (train={self.n_train}, test={self.n_test})",
            "Feature importance:",
        ]
        lines += [f"  {name:24s} {score:.1f}" for name, score in top]
        return "\n".join(lines)


class StructuralClassifier:
    """Thin, explicit wrapper around LGBMClassifier for hazard typing."""

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        num_leaves: int = 15,
        max_depth: int = 5,
        min_child_samples: int = 10,
        class_weight: str | dict | None = "balanced",
        random_state: int = 42,
    ):
        """Hyperparameters are deliberately conservative.

        num_leaves=15 and max_depth=5 are small by LightGBM standards. With
        only 5 features and a few hundred rows, the default num_leaves=31
        overfits badly. class_weight='balanced' matters because real staged
        datasets are lopsided - you will have far more 'artefact' rows than
        'roof_fall' rows, and an unweighted model learns to just never
        predict the dangerous class.
        """
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
    ) -> "StructuralClassifier":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        if X.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                f"expected {len(FEATURE_NAMES)} features {FEATURE_NAMES}, "
                f"got {X.shape[1]}"
            )

        fit_kwargs = {}
        if eval_set is not None:
            fit_kwargs["eval_set"] = [eval_set]
            fit_kwargs["callbacks"] = [lgb.early_stopping(30, verbose=False)]

        # feature_name is deliberately NOT passed: it makes LightGBM expect
        # named input downstream, which trips a warning when sklearn's
        # calibration wrapper hands it a plain array. Importances are mapped
        # positionally in feature_importance() instead.
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
        """Prediction + confidence + severity, ready for the alert layer."""
        self._check_fitted()
        proba = self.predict_proba(X)
        results = []
        for row in proba:
            idx = int(np.argmax(row))
            label = self.classes_[idx]
            results.append(
                {
                    "label": label,
                    "confidence": float(row[idx]),
                    "severity": CLASS_SEVERITY.get(label, 0),
                    "distribution": {
                        cls: float(p) for cls, p in zip(self.classes_, row)
                    },
                }
            )
        return results


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
                for n, s in zip(FEATURE_NAMES, self.model.feature_importance())
            }
        return {
            name: float(score)
            for name, score in zip(FEATURE_NAMES, self.model.feature_importances_)
        }

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> TrainingReport:
        self._check_fitted()
        if not _HAS_SKLEARN:
            raise ImportError("scikit-learn is required")

        preds = self.predict(X_test)
        accuracy = float(np.mean(preds == np.asarray(y_test)))
        return TrainingReport(
            accuracy=accuracy,
            per_class_report=classification_report(y_test, preds, zero_division=0),
            confusion=confusion_matrix(y_test, preds),
            feature_importance=self.feature_importance(),
            n_train=int(getattr(self, "n_train_samples", 0)),
            n_test=len(y_test),
        )

    def save(self, path: str | Path) -> None:
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.booster_.save_model(str(path))
        meta = {
            "classes": self.classes_,
            "features": FEATURE_NAMES,
            "params": self.params,
        }
        path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "StructuralClassifier":
        path = Path(path)
        obj = cls.__new__(cls)
        obj.model = lgb.Booster(model_file=str(path))
        meta_path = path.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        obj.classes_ = meta.get("classes", STRUCTURAL_CLASSES)
        obj.params = meta.get("params", {})
        obj.is_fitted = True
        return obj

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("model is not fitted - call fit() first")


def stratified_split(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.2, calib_size: float = 0.2,
    random_state: int = 42,
):
    """Three-way split: train / calibration / test.

    The calibration split is NOT optional. Using cv='prefit' calibration
    requires data the base model has never seen; calibrating on the training
    set produces meaningless (over-optimistic) probabilities.
    """
    if not _HAS_SKLEARN:
        raise ImportError("scikit-learn is required")

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    relative_calib = calib_size / (1.0 - test_size)
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_calib,
        stratify=y_temp,
        random_state=random_state,
    )
    return X_train, y_train, X_calib, y_calib, X_test, y_test
