#!/usr/bin/env python3
"""
Train + calibrate the LightGBM structural change classifier (Model 1).

Three-way split is mandatory here: train / calibration / test.
cv='prefit' calibration requires data the base model has never seen, so
calibrating on the training split would produce meaningless probabilities.

Usage:
    python train_structural.py --data ../data/raw/structural_synthetic.npz \
                               --out ../data/models
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nexcrawl_ml.models.calibration_wrapper import (
    calibrate_model,
    evaluate_calibration,
)
from nexcrawl_ml.models.structural_classifier import (
    StructuralClassifier,
    stratified_split,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help=".npz with arrays X and y")
    parser.add_argument("--out", default="data/models")
    parser.add_argument("--calib-method", default="isotonic",
                        choices=["isotonic", "sigmoid"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = np.load(args.data, allow_pickle=True)
    X, y = data["X"], data["y"]
    print(f"loaded {X.shape[0]} samples, {X.shape[1]} features")
    unique, counts = np.unique(y, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  {cls:18s} {cnt:4d}  ({cnt/len(y)*100:.1f}%)")

    X_train, y_train, X_calib, y_calib, X_test, y_test = stratified_split(
        X, y, random_state=args.seed
    )
    print(f"\nsplit -> train={len(X_train)} calib={len(X_calib)} test={len(X_test)}")

    print("\n[1/3] training LightGBM...")
    clf = StructuralClassifier(random_state=args.seed)
    clf.fit(X_train, y_train)

    report = clf.evaluate(X_test, y_test)
    print("\n" + report.summary())
    print("\n" + report.per_class_report)

    print("[2/3] calibrating (cv='prefit')...")
    if len(X_calib) < 50 and args.calib_method == "isotonic":
        print("  calibration set is small - falling back to sigmoid")
        method = "sigmoid"
    else:
        method = args.calib_method

    calibrated = calibrate_model(clf.model, X_calib, y_calib, method=method)

    positive_class = "roof_fall" if "roof_fall" in set(y) else str(unique[0])
    calib_report = evaluate_calibration(
        clf.model, calibrated, X_test, y_test,
        positive_class=positive_class, method=method,
    )
    print("  " + calib_report.summary())
    if calib_report.improvement <= 0:
        print(
            "  WARNING: calibration did not improve the Brier score. On a small\n"
            "  dataset this can happen. Consider shipping the uncalibrated model,\n"
            "  or collect more calibration data."
        )

    print("\n[3/3] saving...")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    clf.save(out / "structural_model.txt")

    try:
        import joblib

        joblib.dump(calibrated, out / "structural_model_calibrated.joblib")
    except ImportError:
        print("  joblib unavailable - calibrated wrapper not saved")

    (out / "structural_training_report.json").write_text(
        json.dumps(
            {
                "accuracy": report.accuracy,
                "feature_importance": report.feature_importance,
                "classes": clf.classes_,
                "n_train": len(X_train),
                "n_calib": len(X_calib),
                "n_test": len(X_test),
                "calibration": {
                    "method": method,
                    "cv_mode": "prefit",
                    "brier_before": calib_report.brier_before,
                    "brier_after": calib_report.brier_after,
                    "improvement": calib_report.improvement,
                },
                "confusion_matrix": report.confusion.tolist(),
            },
            indent=2,
        )
    )
    print(f"  wrote model + report to {out}")


if __name__ == "__main__":
    main()
