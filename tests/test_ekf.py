"""Tests for the pose EKF."""

import numpy as np

from nexcrawl_ml.fusion.ekf import Pose2D, PoseEKF, ticks_to_distance, wrap_angle


def test_wrap_angle_range():
    assert np.isclose(wrap_angle(3 * np.pi), np.pi)
    assert np.isclose(wrap_angle(-3 * np.pi), np.pi)
    assert np.isclose(wrap_angle(0.5), 0.5)


def test_straight_line_motion():
    ekf = PoseEKF()
    for _ in range(10):
        ekf.predict(0.1, 0.1)
    pose = ekf.pose
    assert np.isclose(pose.x, 1.0, rtol=0.01)
    assert np.isclose(pose.y, 0.0, atol=1e-6)
    assert np.isclose(pose.theta, 0.0, atol=1e-6)


def test_pure_rotation_does_not_translate():
    ekf = PoseEKF(wheel_base_m=0.3)
    ekf.predict(-0.1, 0.1)
    assert np.isclose(ekf.pose.x, 0.0, atol=1e-9)
    assert np.isclose(ekf.pose.y, 0.0, atol=1e-9)
    assert ekf.pose.theta > 0


def test_freeze_holds_pose_during_dome_scan():
    """The whole point of freezing: a parked rover must not accumulate drift."""
    ekf = PoseEKF()
    ekf.predict(0.5, 0.5)
    before = ekf.pose.as_tuple()

    ekf.freeze()
    for _ in range(50):
        ekf.predict(0.1, 0.1)
        ekf.update_imu_yaw(1.5)
    assert ekf.pose.as_tuple() == before

    ekf.unfreeze()
    ekf.predict(0.1, 0.1)
    assert ekf.pose.as_tuple() != before


def test_imu_update_pulls_heading():
    ekf = PoseEKF()
    ekf.predict(0.1, 0.12)   # small odometry-derived heading change
    before = ekf.pose.theta
    for _ in range(10):
        ekf.update_imu_yaw(0.5)
    assert abs(ekf.pose.theta - 0.5) < abs(before - 0.5)


def test_uncertainty_grows_with_motion():
    ekf = PoseEKF()
    initial = ekf.uncertainty[0]
    for _ in range(20):
        ekf.predict(0.2, 0.2)
    assert ekf.uncertainty[0] > initial


def test_ticks_to_distance():
    d = ticks_to_distance(ticks=360, ticks_per_revolution=360, wheel_radius_m=0.05)
    assert np.isclose(d, 2 * np.pi * 0.05)


def test_reset():
    ekf = PoseEKF()
    ekf.predict(1.0, 1.0)
    ekf.reset(Pose2D(5.0, 5.0, 1.0))
    assert ekf.pose.as_tuple() == (5.0, 5.0, 1.0)
