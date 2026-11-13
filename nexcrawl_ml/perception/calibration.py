"""
Rig calibration for the actuated 2D LiDAR.

Three calibrations, in dependency order:
  A. servo zero-offset   -> phi_zero_offset_rad
  B. lever-arm / axis offset (flat-wall least-squares fit)
  C. mast height          -> measured directly, stored here for completeness

(B) is the one most DIY builds skip. The tilt axis of the rotating mechanism
should ideally pass through the optical rotation centre of the 2D LiDAR; in
practice assembly misalignment means it does not, and the residual offset
must be estimated numerically.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from .transforms import spherical_to_local, apply_lever_arm


@dataclass
class RigCalibration:
    """Everything the transform pipeline needs to know about the physical rig."""

    phi_zero_offset_rad: float = 0.0
    lever_arm_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mast_height_m: float = 0.25
    residual_rms_m: float | None = None
    n_calibration_points: int | None = None

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        payload["lever_arm_m"] = list(self.lever_arm_m)
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "RigCalibration":
        data = json.loads(Path(path).read_text())
        data["lever_arm_m"] = tuple(data["lever_arm_m"])
        return cls(**data)


def calibrate_servo_zero(raw_encoder_rad_at_level: float) -> float:
    """Calibration A.

    Level the LiDAR against a bubble level / flat reference, read the raw
    encoder value, and store it. Every later reading gets this subtracted.
    """
    return float(raw_encoder_rad_at_level)


def _plane_residuals(
    params: np.ndarray,
    scans: list[tuple[np.ndarray, np.ndarray, float]],
) -> np.ndarray:
    """Residual function for the flat-wall lever-arm fit.

    Free parameters:
        params[0:3] -> lever arm (dx, dy, dz)
        params[3]   -> phi zero-offset correction (radians)
        params[4:7] -> wall plane normal (unnormalised)
        params[7]   -> wall plane offset d

    Cost: point-to-plane distance for every point of every scan. If the
    calibration is right, scans taken at *different* tilt angles all collapse
    onto the same physical plane.
    """
    lever = params[0:3]
    dphi = params[3]
    normal = params[4:7]
    d = params[7]

    norm = np.linalg.norm(normal)
    if norm < 1e-9:
        return np.full(sum(len(s[0]) for s in scans), 1e6)
    normal = normal / norm

    residuals = []
    for ranges, beam_angles, phi_raw in scans:
        pts = spherical_to_local(ranges, beam_angles, phi_raw - dphi)
        pts = apply_lever_arm(pts, lever)
        residuals.append(pts @ normal - d)
    return np.concatenate(residuals)


def calibrate_lever_arm(
    scans: list[tuple[np.ndarray, np.ndarray, float]],
    initial_lever: tuple[float, float, float] = (0.0, 0.0, 0.0),
    initial_dphi: float = 0.0,
    max_lever_m: float = 0.15,
) -> RigCalibration:
    """Calibration B: flat-wall least-squares estimation of the axis offset.

    Procedure:
      1. Park the rover facing a known flat wall.
      2. Capture one full LiDAR revolution at several distinct tilt angles
         (e.g. -30, -15, 0, +15, +30 degrees).
      3. Pass them in as (ranges, beam_angles, raw_phi) tuples.

    The optimiser finds the lever arm + zero-offset that best makes every
    scan agree on a single plane. Uses Levenberg-Marquardt style non-linear
    least squares, which is the standard tool in the tilting-LiDAR
    calibration literature.

    Returns a RigCalibration with residual_rms_m populated - quote that
    number to judges as your measured accuracy.
    """
    if len(scans) < 3:
        raise ValueError(
            "need at least 3 scans at different tilt angles for a stable fit"
        )

    # Seed the plane from the middle scan, assuming roughly forward-facing wall.
    mid_ranges, mid_angles, mid_phi = scans[len(scans) // 2]
    seed_pts = spherical_to_local(mid_ranges, mid_angles, mid_phi - initial_dphi)
    centroid = seed_pts.mean(axis=0)
    # Smallest-eigenvector of the covariance = best-fit plane normal (PCA).
    _, _, vh = np.linalg.svd(seed_pts - centroid, full_matrices=False)
    seed_normal = vh[-1]
    seed_d = float(seed_normal @ centroid)

    x0 = np.concatenate(
        [np.asarray(initial_lever, dtype=float), [initial_dphi], seed_normal, [seed_d]]
    )
    lower = np.array(
        [-max_lever_m] * 3 + [-np.deg2rad(10)] + [-np.inf] * 3 + [-np.inf]
    )
    upper = np.array(
        [max_lever_m] * 3 + [np.deg2rad(10)] + [np.inf] * 3 + [np.inf]
    )

    result = least_squares(
        _plane_residuals,
        x0,
        bounds=(lower, upper),
        args=(scans,),
        method="trf",
        loss="soft_l1",  # robust against a few wall-edge outliers
    )

    rms = float(np.sqrt(np.mean(result.fun ** 2)))
    n_pts = int(sum(len(s[0]) for s in scans))

    return RigCalibration(
        phi_zero_offset_rad=float(result.x[3]),
        lever_arm_m=tuple(float(v) for v in result.x[0:3]),
        residual_rms_m=rms,
        n_calibration_points=n_pts,
    )
