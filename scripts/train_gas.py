#!/usr/bin/env python3
"""
Train + calibrate the LightGBM gas trend classifier (Model 2).

This is a genuinely separate model from the structural classifier: different
features, different labels, different file. Same library, different job.

Usage:
    python train_gas.py --data ../data/raw/gas_synthetic.npz --out ../data/models
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nexcrawl_ml.models.calibration_wrapper import (
    calibrate_model,
    evaluate_calibration,
)
from nexcrawl_ml.models.gas_classifier import GasTrendClassifier
from nexcrawl_ml.models.structural_classifier import stratified_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="data/models")
    parser.add_argument("--calib-method", default="isotonic",
                        choices=["isotonic", "sigmoid"])
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()

    data = np.load(args.data, allow_pickle=True)
    X, y = data["X"], data["y"]
    print(f"loaded {X.shape[0]} samples, {X.shape[1]} features")
    unique, counts = np.unique(y, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  {cls:22s} {cnt:4d}  ({cnt/len(y)*100:.1f}%)")

    X_train, y_train, X_calib, y_calib, X_test, y_test = stratified_split(
        X, y, random_state=args.seed
    )
    print(f"\nsplit -> train={len(X_train)} calib={len(X_calib)} test={len(X_test)}")

    print("\n[1/3] training LightGBM...")
    clf = GasTrendClassifier(random_state=args.seed)
    clf.fit(X_train, y_train)

    report = clf.evaluate(X_test, y_test)
    print(f"\nAccuracy: {report.accuracy:.3f}")
    if report.accuracy >= 0.99:
        print(
            "  WARNING: near-perfect accuracy. On a real staged dataset this almost\n"
            "  always means leakage or classes that are trivially separable - not a\n"
            "  good model. Expect a large drop on real sensor data; do not quote\n"
            "  this number to judges."
        )
    print(report.per_class_report)
    print("Feature importance:")
    for name, score in sorted(
        report.feature_importance.items(), key=lambda kv: kv[1], reverse=True
    ):
        print(f"  {name:18s} {score:.1f}")

    print("\n[2/3] calibrating (cv='prefit')...")
    method = args.calib_method if len(X_calib) >= 50 else "sigmoid"
    calibrated = calibrate_model(clf.model, X_calib, y_calib, method=method)

    positive = "RAPID_ACCUMULATION" if "RAPID_ACCUMULATION" in set(y) else str(unique[0])
    calib_report = evaluate_calibration(
        clf.model, calibrated, X_test, y_test, positive_class=positive, method=method
    )
    print("  " + calib_report.summary())
    if calib_report.improvement <= 0:
        print(
            "  WARNING: calibration did not improve the Brier score. Isotonic\n"
            "  calibration needs a reasonably large held-out set; on small data it\n"
            "  can overfit the calibration curve. Try --calib-method sigmoid, or\n"
            "  ship the uncalibrated model and say so."
        )

    print("\n[3/3] saving...")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    clf.save(out / "gas_model.txt")

    try:
        import joblib

        joblib.dump(calibrated, out / "gas_model_calibrated.joblib")
    except ImportError:
        print("  joblib unavailable - calibrated wrapper not saved")

    (out / "gas_training_report.json").write_text(
        json.dumps(
            {
                "accuracy": report.accuracy,
                "feature_importance": report.feature_importance,
                "classes": clf.classes_,
                "calibration": {
                    "method": method,
                    "cv_mode": "prefit",
                    "brier_before": calib_report.brier_before,
                    "brier_after": calib_report.brier_after,
                },
                "confusion_matrix": report.confusion.tolist(),
            },
            indent=2,
        )
    )
    print(f"  wrote model + report to {out}")


if __name__ == "__main__":
    main()
