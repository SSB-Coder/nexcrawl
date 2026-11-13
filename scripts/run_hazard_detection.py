#!/usr/bin/env python3
"""
Compare a current scan against the stored baseline and report hazards.

Usage:
    python run_hazard_detection.py --baseline ../data/processed/dome.npy \
                                   --current  ../data/processed/after.npy \
                                   --model    ../data/models/structural_model.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nexcrawl_ml.models.structural_classifier import StructuralClassifier
from nexcrawl_ml.pipeline.alert_fusion import fuse_alerts, structural_alerts
from nexcrawl_ml.pipeline.hazard_pipeline import HazardPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--threshold", type=float, default=0.08)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    baseline = np.load(args.baseline)
    current = np.load(args.current)
    print(f"baseline: {baseline.shape[0]:,} points")
    print(f"current : {current.shape[0]:,} points")

    clf = StructuralClassifier.load(args.model)
    pipeline = HazardPipeline(clf, diff_threshold_m=args.threshold)

    report = pipeline.run(current, baseline)

    print(f"\nregistration: fitness={report.registration_fitness:.3f} "
          f"rmse={report.registration_rmse*1000:.1f}mm")
    print(f"changed points: +{report.n_added_points:,} / -{report.n_removed_points:,}")
    print(f"clusters: {report.n_blobs} (eps={report.eps_used:.3f}m)")

    for w in report.warnings:
        print(f"\nWARNING: {w}")

    if not report.hazards:
        print("\nno hazards above threshold")
    else:
        print(f"\n{len(report.hazards)} hazard(s):")
        for h in report.hazards:
            print(f"  [{h.severity}] {h.label:16s} conf={h.confidence:.2f} "
                  f"at ({h.centroid[0]:.2f}, {h.centroid[1]:.2f}, {h.centroid[2]:.2f}) "
                  f"vol={h.features['volume_m3']:.3f}m3")

    alerts = fuse_alerts(structural_alerts(report, min_confidence=args.min_confidence))
    if alerts:
        print(f"\n{len(alerts)} alert(s) for the operator:")
        for a in alerts:
            print(f"  {a.priority.name:8s} {a.message}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(
                {
                    "report": report.as_dict(),
                    "alerts": [a.as_dict() for a in alerts],
                },
                indent=2,
            )
        )
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
