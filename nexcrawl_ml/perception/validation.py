"""
Validation tests for the reconstruction rig (Section 14 of the plan).

Run these before trusting the system in front of judges. Each returns a
dataclass with a hard number you can quote.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PlanarityResult:
    rms_error_m: float
    max_error_m: float
    n_points: int
    plane_normal: tuple[float, float, float]
    passed: bool


@dataclass
class DimensionResult:
    measured_m: tuple[float, float, float]
    expected_m: tuple[float, float, float]
    percent_error: tuple[float, float, float]
    passed: bool


@dataclass
class RepeatabilityResult:
    mean_distance_m: float
    rms_distance_m: float
    systematic_bias_m: tuple[float, float, float]
    bias_ratio: float
    verdict: str


def flat_wall_test(points: np.ndarray, tolerance_m: float = 0.02) -> PlanarityResult:
    """Test 1: scan a known flat wall; points should collapse onto a plane.

    This is the direct check on lever-arm calibration. A large RMS here means
    calibration, not sensor noise.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 10:
        raise ValueError("need at least 10 points")

    centroid = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - centroid, full_matrices=False)
    normal = vh[-1]
    distances = (pts - centroid) @ normal

    rms = float(np.sqrt(np.mean(distances ** 2)))
    return PlanarityResult(
        rms_error_m=rms,
        max_error_m=float(np.abs(distances).max()),
        n_points=len(pts),
        plane_normal=tuple(float(v) for v in normal),
        passed=rms <= tolerance_m,
    )


def known_dimension_test(
    points: np.ndarray,
    expected_dims_m: tuple[float, float, float],
    tolerance_percent: float = 5.0,
) -> DimensionResult:
    """Test 2: scan an object of known size, compare the bounding box."""
    pts = np.asarray(points, dtype=np.float64)
    measured = tuple(float(v) for v in (pts.max(axis=0) - pts.min(axis=0)))
    errors = tuple(
        float(abs(m - e) / e * 100.0) if e > 0 else float("inf")
        for m, e in zip(measured, expected_dims_m)
    )
    return DimensionResult(
        measured_m=measured,
        expected_m=expected_dims_m,
        percent_error=errors,
        passed=all(e <= tolerance_percent for e in errors),
    )


def repeatability_test(
    cloud_a: np.ndarray,
    cloud_b: np.ndarray,
    bias_threshold: float = 0.3,
) -> RepeatabilityResult:
    """Test 3: scan the same static scene twice.

    The key diagnostic is not the magnitude of the difference but its
    *character*. Random scatter = sensor noise (fine). A consistent
    directional offset = a calibration error you have not fixed yet.
    """
    a = np.asarray(cloud_a, dtype=np.float64)
    b = np.asarray(cloud_b, dtype=np.float64)

    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(b)
        dists, idx = tree.query(a, k=1)
        deltas = b[idx] - a
    except ImportError:  # pragma: no cover
        n = min(len(a), len(b))
        deltas = b[:n] - a[:n]
        dists = np.linalg.norm(deltas, axis=1)

    mean_d = float(np.mean(dists))
    rms_d = float(np.sqrt(np.mean(dists ** 2)))
    bias = deltas.mean(axis=0)
    bias_mag = float(np.linalg.norm(bias))
    ratio = bias_mag / mean_d if mean_d > 0 else 0.0

    if ratio > bias_threshold:
        verdict = (
            "SYSTEMATIC BIAS detected - re-run lever-arm / zero-offset "
            "calibration before trusting the mesh"
        )
    else:
        verdict = "Random scatter consistent with sensor noise - calibration OK"

    return RepeatabilityResult(
        mean_distance_m=mean_d,
        rms_distance_m=rms_d,
        systematic_bias_m=tuple(float(v) for v in bias),
        bias_ratio=ratio,
        verdict=verdict,
    )


def timing_budget(
    n_tilt_steps: int,
    settle_time_s: float,
    revolution_time_s: float,
    reconstruction_s: float,
) -> dict:
    """Test 4: end-to-end latency, checked against the demo script window."""
    sweep = n_tilt_steps * (settle_time_s + revolution_time_s)
    total = sweep + reconstruction_s
    return {
        "sweep_seconds": sweep,
        "reconstruction_seconds": reconstruction_s,
        "total_seconds": total,
        "fits_7min_demo": total < 420,
        "per_step_seconds": settle_time_s + revolution_time_s,
    }
