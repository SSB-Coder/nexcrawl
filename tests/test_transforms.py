"""Tests for the geometric transform chain.

These use closed-form expected values rather than golden files, so a
regression here points at a real math error, not a changed fixture.
"""

import numpy as np
import pytest

from nexcrawl_ml.perception.transforms import (
    apply_lever_arm,
    local_to_world,
    range_gate,
    rotation_z,
    spherical_to_local,
    spherical_to_local_varying_phi,
)


def test_zero_tilt_stays_in_plane():
    """At phi=0 every point must lie exactly in the z=0 plane."""
    ranges = np.array([1.0, 2.0, 3.0])
    angles = np.array([0.0, np.pi / 2, np.pi])
    pts = spherical_to_local(ranges, angles, 0.0)
    assert np.allclose(pts[:, 2], 0.0)


def test_straight_up_tilt():
    """At phi=90deg the disc's plane has rotated into the vertical XZ plane.

    theta=0 (was pointing +X/forward) now points +Z (up).
    theta=180 (was pointing -X/backward) now points -Z (down).
    theta=+-90 (side beams) are unaffected - they sit on the rotation axis.
    This replaces the old (incorrect) assumption that every beam points
    straight up regardless of azimuth, which only holds for a per-beam
    elevation angle on a fixed-channel LiDAR, not a rigidly tilted 2D disc.
    """
    r = 2.0
    pts = spherical_to_local(
        np.array([r, r, r]), np.array([0.0, np.pi / 2, np.pi]), np.pi / 2
    )
    assert np.allclose(pts[0], [0.0, 0.0, r], atol=1e-9)   # front -> straight up
    assert np.allclose(pts[1], [0.0, r, 0.0], atol=1e-9)   # side -> unaffected
    assert np.allclose(pts[2], [0.0, 0.0, -r], atol=1e-9)  # rear -> straight down


def test_side_beams_unaffected_by_tilt():
    """theta=+-90deg sits exactly on the pitch rotation axis: z=0 and range
    is fully preserved in y, at every tilt angle. This is the direct
    behavioural check on the rigid-body model - tunnel width must not
    shrink just because the sensor tilts."""
    r = 3.0
    for phi in [-1.2, -0.5, 0.0, 0.7, 1.4]:
        pts = spherical_to_local(
            np.array([r, r]), np.array([np.pi / 2, -np.pi / 2]), phi
        )
        assert np.allclose(pts[:, 2], 0.0, atol=1e-9), f"side beam lifted at phi={phi}"
        assert np.allclose(np.abs(pts[:, 1]), r, atol=1e-9), f"side range changed at phi={phi}"


def test_front_beam_height_matches_r_sin_phi():
    """theta=0 (directly forward) is the one azimuth where the full tilt
    shows up in z, matching the rigid-body derivation z = r*sin(phi)*cos(theta)
    with cos(0)=1."""
    r = 4.0
    for phi in [-0.9, -0.3, 0.3, 0.9]:
        pts = spherical_to_local(np.array([r]), np.array([0.0]), phi)
        assert np.isclose(pts[0, 2], r * np.sin(phi))
        assert np.isclose(pts[0, 0], r * np.cos(phi))


def test_range_preserved_under_tilt():
    """Tilting rotates points; it must not change their distance."""
    ranges = np.array([1.0, 2.5, 4.0])
    angles = np.array([0.3, 1.1, 2.7])
    for phi in [-0.8, -0.2, 0.0, 0.4, 1.2]:
        pts = spherical_to_local(ranges, angles, phi)
        assert np.allclose(np.linalg.norm(pts, axis=1), ranges)


def test_known_45_degree_case():
    r = 1.0
    pts = spherical_to_local(np.array([r]), np.array([0.0]), np.pi / 4)
    expected = r / np.sqrt(2)
    assert np.allclose(pts[0], [expected, 0.0, expected])


def test_varying_phi_matches_scalar():
    ranges = np.array([1.0, 2.0, 3.0])
    angles = np.array([0.1, 0.2, 0.3])
    phi = 0.35
    a = spherical_to_local(ranges, angles, phi)
    b = spherical_to_local_varying_phi(ranges, angles, np.full(3, phi))
    assert np.allclose(a, b)


def test_lever_arm_is_pure_translation():
    pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    lever = np.array([0.01, -0.02, 0.03])
    out = apply_lever_arm(pts, lever)
    assert np.allclose(out - pts, lever)


def test_rotation_z_orthonormal():
    for yaw in [0.0, 0.5, np.pi, -2.0]:
        R = rotation_z(yaw)
        assert np.allclose(R @ R.T, np.eye(3))
        assert np.isclose(np.linalg.det(R), 1.0)


def test_local_to_world_identity():
    pts = np.array([[1.0, 0.0, 0.0]])
    out = local_to_world(pts, 0.0, 0.0, 0.0, 0.0)
    assert np.allclose(out, pts)


def test_local_to_world_yaw_90():
    """A point on +X, rotated 90deg yaw, must land on +Y."""
    pts = np.array([[1.0, 0.0, 0.0]])
    out = local_to_world(pts, 0.0, 0.0, np.pi / 2, 0.0)
    assert np.allclose(out, [[0.0, 1.0, 0.0]], atol=1e-9)


def test_mast_height_offsets_z():
    pts = np.zeros((3, 3))
    out = local_to_world(pts, mast_height_m=0.25)
    assert np.allclose(out[:, 2], 0.25)


def test_range_gate_filters_correctly():
    ranges = np.array([0.05, 0.5, 5.0, 50.0, np.nan])
    angles = np.arange(5, dtype=float)
    r, theta, mask = range_gate(ranges, angles, 0.15, 10.0)
    assert len(r) == 2
    assert np.allclose(r, [0.5, 5.0])
    assert np.allclose(theta, [1.0, 2.0])


def test_mismatched_shapes_raise():
    with pytest.raises(ValueError):
        spherical_to_local(np.array([1.0, 2.0]), np.array([0.0]), 0.0)
