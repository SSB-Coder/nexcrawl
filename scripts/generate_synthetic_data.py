#!/usr/bin/env python3
"""
Synthetic dataset generator - lets you build and validate the full training
pipeline BEFORE the hardware and the staged mock-mine data exist.

IMPORTANT: synthetic data is scaffolding, not a substitute. Models trained
here must be retrained on the real staged dataset (Week 9) before the demo.
The distributions below are plausible, not measured, and a model that scores
well here can still fail on real rock.

Usage:
    python generate_synthetic_data.py --structural 400 --gas 600 --out ../data/raw
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

STRUCTURAL_CLASSES = [
    "roof_fall",
    "rib_spall",
    "floor_heave",
    "equipment_moved",
    "artefact",
]
GAS_CLASSES = ["SAFE", "SLOW_LEAK", "RAPID_ACCUMULATION", "SENSOR_FAULT"]


def generate_structural(n: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate (volume, height, aspect_ratio, roughness, planarity) samples.

    Class priors are deliberately imbalanced to mirror reality: most detected
    changes are artefacts, roof falls are rare. This is exactly why the
    classifier uses class_weight='balanced'.
    """
    rng = np.random.default_rng(seed)
    priors = np.array([0.15, 0.20, 0.15, 0.20, 0.30])
    labels = rng.choice(STRUCTURAL_CLASSES, size=n, p=priors)

    rows = []
    for label in labels:
        if label == "roof_fall":
            volume = rng.lognormal(np.log(0.8), 0.6)
            height = rng.normal(2.2, 0.4)
            aspect = rng.normal(2.2, 0.6)
            rough = rng.beta(4, 6)
            planar = rng.beta(2, 5)
        elif label == "rib_spall":
            volume = rng.lognormal(np.log(0.25), 0.7)
            height = rng.normal(1.2, 0.5)
            aspect = rng.normal(3.5, 1.0)
            rough = rng.beta(4, 5)
            planar = rng.beta(4, 3)
        elif label == "floor_heave":
            volume = rng.lognormal(np.log(0.5), 0.6)
            height = rng.normal(0.25, 0.15)
            aspect = rng.normal(4.5, 1.2)
            rough = rng.beta(3, 7)
            planar = rng.beta(5, 3)
        elif label == "equipment_moved":
            volume = rng.lognormal(np.log(0.6), 0.5)
            height = rng.normal(0.8, 0.4)
            aspect = rng.normal(1.6, 0.4)   # boxy, regular
            rough = rng.beta(2, 12)         # smooth manufactured surfaces
            planar = rng.beta(7, 2)         # flat faces
        else:  # artefact
            volume = rng.lognormal(np.log(0.05), 1.0)
            height = rng.uniform(0.0, 2.5)
            aspect = rng.normal(6.0, 2.5)   # thin ghost sheets
            rough = rng.beta(2, 8)
            planar = rng.beta(8, 2)

        rows.append(
            [
                max(volume, 1e-4),
                max(height, 0.0),
                max(aspect, 1.0),
                float(np.clip(rough, 0, 1)),
                float(np.clip(planar, 0, 1)),
            ]
        )

    return np.array(rows), labels


def generate_gas(
    n: int, seed: int = 43, overlap: float = 0.35
) -> tuple[np.ndarray, np.ndarray]:
    """Generate 8-feature rolling-window samples for the gas classifier.

    `overlap` deliberately blurs the class boundaries. Without it the
    generated classes are almost linearly separable and LightGBM scores a
    perfect 1.000 - which looks great and means nothing, because real sensor
    data has ambiguous windows (a slow leak early on looks exactly like
    drifting SAFE readings). A synthetic benchmark you cannot fail is not
    telling you the pipeline works.
    """
    rng = np.random.default_rng(seed)
    priors = np.array([0.45, 0.20, 0.15, 0.20])
    labels = rng.choice(GAS_CLASSES, size=n, p=priors)

    rows = []
    for label in labels:
        if label == "SAFE":
            mean = rng.uniform(0.05, 0.4)
            slope = rng.normal(0.0, 0.0008)
            jitter = rng.uniform(0.004, 0.02)
            r2 = rng.beta(1.5, 5)
        elif label == "SLOW_LEAK":
            mean = rng.uniform(0.3, 1.1)
            slope = rng.uniform(0.002, 0.012)
            jitter = rng.uniform(0.004, 0.018)
            r2 = rng.beta(6, 2)   # steady, linear climb
        elif label == "RAPID_ACCUMULATION":
            mean = rng.uniform(0.8, 3.2)
            slope = rng.uniform(0.05, 0.35)
            jitter = rng.uniform(0.01, 0.05)
            r2 = rng.beta(7, 2)
        else:  # SENSOR_FAULT
            mean = rng.uniform(0.0, 2.0)
            slope = rng.normal(0.0, 0.0002)
            jitter = rng.uniform(1e-6, 5e-5)   # the tell: no noise at all
            r2 = rng.beta(1, 8)

        std = jitter * rng.uniform(2, 5)
        span = jitter * rng.uniform(4, 12) + abs(slope) * 10
        scale = max(abs(mean), 1e-6)
        flatline = float(np.clip(np.exp(-jitter / (scale * 1e-3)), 0, 1))
        last = mean + slope * 5 + rng.normal(0, jitter)

        rows.append([mean, std, slope, span, jitter, flatline, last, r2])

    X = np.array(rows)

    # Blur the boundaries: jitter every feature by a fraction of its own
    # spread, and additionally soften the two features that would otherwise
    # be perfect giveaways (flatline_score and trend_r2).
    if overlap > 0:
        scale = X.std(axis=0, keepdims=True)
        X = X + rng.normal(0, overlap, X.shape) * scale
        X[:, 5] = np.clip(X[:, 5], 0.0, 1.0)   # flatline_score
        X[:, 7] = np.clip(X[:, 7], 0.0, 1.0)   # trend_r2
        X[:, 4] = np.abs(X[:, 4])              # jitter is non-negative

    return X, labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structural", type=int, default=400)
    parser.add_argument("--gas", type=int, default=600)
    parser.add_argument("--out", type=str, default="data/raw")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    Xs, ys = generate_structural(args.structural, args.seed)
    np.savez(out / "structural_synthetic.npz", X=Xs, y=ys)
    print(f"structural: {Xs.shape} -> {out / 'structural_synthetic.npz'}")

    Xg, yg = generate_gas(args.gas, args.seed + 1)
    np.savez(out / "gas_synthetic.npz", X=Xg, y=yg)
    print(f"gas:        {Xg.shape} -> {out / 'gas_synthetic.npz'}")

    (out / "SYNTHETIC_README.txt").write_text(
        "These files are SYNTHETIC scaffolding data.\n"
        "They exist so the training/calibration/serving pipeline can be built\n"
        "and tested before the staged mock-mine dataset is collected.\n\n"
        "Do NOT ship models trained on this data. Retrain on real staged\n"
        "measurements before the demo, and re-run calibration afterwards.\n"
    )


if __name__ == "__main__":
    main()
