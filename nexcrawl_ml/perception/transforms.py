"""
Coordinate transforms for the actuated (nodding) 2D LiDAR.

Frame chain (see Section 0 of the implementation plan):
    sensor -> tilt -> mast -> chassis -> world

These are *pure* geometric functions: no training, no state, fully testable.
"""

from __future__ import annotations

import numpy as np


def spherical_to_local(
    ranges: np.ndarray,
    beam_angles: np.ndarray,
    tilt_angle_rad: float,
) -> np.ndarray:
    """Lift one 2D LiDAR revolution into local 3D Cartesian coordinates.

    CORRECTED MODEL. A 2D LiDAR disc spins its beam inside its own flat
    plane: p_sensor = (r cos(theta), r sin(theta), 0). The servo tilts that
    *entire rigid disc* by pitch angle phi about the Y axis (the axis running
    left-right through the disc, perpendicular to the direction of travel).

    That is a rigid-body rotation R_y(phi) applied identically to every
    point regardless of theta - NOT the spherical/latitude formula
    (x=r*cos(phi)*cos(theta), y=r*cos(phi)*sin(theta), z=r*sin(phi)) that
    would apply if phi were a per-beam elevation angle, as on a fixed-channel
    multi-beam LiDAR (e.g. Velodyne). This project has no such per-beam
    elevation: the whole disc, including the beams pointing left and right,
    moves together as one rigid body.

    The practical difference matters a lot: under the (wrong) spherical
    model, side beams (theta = +-90 deg) get lifted by z=r*sin(phi) AND
    their apparent distance shrinks by cos(phi) - so a levelled corridor
    would appear to narrow every time the servo tilts, and a rear beam
    would erroneously lift UP instead of dipping DOWN. Under the correct
    rigid-body model, points at theta = +-90 deg sit exactly on the tilt
    axis and are completely unaffected by phi - side-wall distance and
    tunnel width stay constant at every tilt angle, which is what a rigid
    tilting disc actually does.

    Frame convention:
        +X: forward along rover heading
        +Y: left, along the servo's pitch rotation axis
        +Z: up, toward the mine roof

    Derivation: p_tilt = R_y(phi) @ (r*cos(theta), r*sin(theta), 0)^T using
        R_y(phi) = [[cos(phi), 0, sin(phi)],
                    [0,         1, 0        ],
                    [-sin(phi), 0, cos(phi)]]
    with the sign convention that positive phi tilts the front (theta=0) UP:
        x = r * cos(phi) * cos(theta)
        y = r * sin(theta)                  <- unaffected by phi (rotation axis)
        z = r * sin(phi) * cos(theta)

    Args:
        ranges: (N,) range measurements in metres.
        beam_angles: (N,) azimuth angles in radians (LiDAR's own frame).
        tilt_angle_rad: phi, the *calibrated* tilt angle in radians.

    Returns:
        (N, 3) array of local XYZ points.
    """
    r = np.asarray(ranges, dtype=np.float64)
    theta = np.asarray(beam_angles, dtype=np.float64)
    phi = float(tilt_angle_rad)

    if r.shape != theta.shape:
        raise ValueError(
            f"ranges {r.shape} and beam_angles {theta.shape} must match"
        )

    cos_theta = np.cos(theta)
    x = r * np.cos(phi) * cos_theta
    y = r * np.sin(theta)
    z = r * np.sin(phi) * cos_theta
    return np.column_stack([x, y, z])


def spherical_to_local_varying_phi(
    ranges: np.ndarray,
    beam_angles: np.ndarray,
    tilt_angles_rad: np.ndarray,
) -> np.ndarray:
    """Per-point version for continuous-nod mode.

    Same corrected rigid-body pitch model as spherical_to_local, applied
    per-point since each point carries its own interpolated phi in
    continuous-nod mode.
    """
    r = np.asarray(ranges, dtype=np.float64)
    theta = np.asarray(beam_angles, dtype=np.float64)
    phi = np.asarray(tilt_angles_rad, dtype=np.float64)

    if not (r.shape == theta.shape == phi.shape):
        raise ValueError("ranges, beam_angles and tilt_angles must all match")

    cos_theta = np.cos(theta)
    x = r * np.cos(phi) * cos_theta
    y = r * np.sin(theta)
    z = r * np.sin(phi) * cos_theta
    return np.column_stack([x, y, z])


def apply_lever_arm(points_local: np.ndarray, lever_arm: np.ndarray) -> np.ndarray:
    """Apply the static translation between LiDAR optical centre and tilt axis.

    The servo's rotation axis will not pass exactly through the LiDAR's
    internal scan centre on any realistic DIY bracket. Without this
    correction the error grows with range and shows up as a consistent
    'smear' in the reconstructed mesh.
    """
    lever_arm = np.asarray(lever_arm, dtype=np.float64).reshape(3)
    return points_local + lever_arm


def rotation_z(yaw_rad: float) -> np.ndarray:
    """Standard right-handed rotation matrix about the Z (yaw) axis."""
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def local_to_world(
    points_local: np.ndarray,
    rover_x: float = 0.0,
    rover_y: float = 0.0,
    rover_yaw: float = 0.0,
    mast_height_m: float = 0.25,
) -> np.ndarray:
    """Project local (mast-frame) points into the global mine frame.

    For a stationary dome scan the pose is constant across the whole sweep,
    so this can be called once per revolution with fixed arguments.
    """
    pts = np.asarray(points_local, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"expected (N,3) points, got {pts.shape}")

    rotated = pts @ rotation_z(rover_yaw).T
    rotated[:, 0] += rover_x
    rotated[:, 1] += rover_y
    rotated[:, 2] += mast_height_m
    return rotated


def range_gate(
    ranges: np.ndarray,
    beam_angles: np.ndarray,
    min_range: float = 0.15,
    max_range: float = 10.0,
    quality: np.ndarray | None = None,
    min_quality: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop self-occlusion hits, out-of-range returns and low-quality points.

    Returns the filtered (ranges, beam_angles, mask).
    """
    r = np.asarray(ranges, dtype=np.float64)
    theta = np.asarray(beam_angles, dtype=np.float64)

    mask = (r > min_range) & (r < max_range) & np.isfinite(r)
    if quality is not None and min_quality > 0:
        mask &= np.asarray(quality) >= min_quality

    return r[mask], theta[mask], mask
