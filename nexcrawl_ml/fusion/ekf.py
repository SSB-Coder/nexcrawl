"""
Extended Kalman Filter for rover pose: wheel odometry + IMU yaw.

State: [x, y, theta]

Scoping note that matters for this project: the EKF is only needed while
DRIVING. During a stationary dome scan the rover is parked, pose is
constant, and running the filter just injects drift into a measurement that
should be exact. The pipeline therefore holds pose fixed during dome scans -
a deliberate deviation from always-on EKF usage, and one less error source
in the flagship demo.

Also note what this does NOT do: there is no LiDAR-based correction
anywhere. Pose is pure dead reckoning, so it drifts without bound over a
long traverse. Any loop-closure or scan-matching correction would be a
separate addition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def wrap_angle(angle: float) -> float:
    """Wrap to (-pi, pi]. Skipping this is the classic EKF heading bug:
    the filter tries to rotate 359 degrees instead of -1.

    The explicit -pi -> +pi fixup keeps the returned range half-open as
    documented; the modulo alone lands exact multiples of pi on -pi.
    """
    wrapped = float((angle + np.pi) % (2 * np.pi) - np.pi)
    return np.pi if wrapped == -np.pi else wrapped


@dataclass
class Pose2D:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0

    def as_tuple(self) -> tuple[float, float, float]:
        return self.x, self.y, self.theta


class PoseEKF:
    """3-state EKF fusing differential-drive odometry with IMU heading."""

    def __init__(
        self,
        wheel_base_m: float = 0.30,
        process_noise_xy: float = 0.02,
        process_noise_theta: float = 0.01,
        imu_yaw_noise: float = 0.02,
        initial_pose: Pose2D | None = None,
    ):
        self.wheel_base_m = wheel_base_m
        pose = initial_pose or Pose2D()
        self.state = np.array([pose.x, pose.y, pose.theta], dtype=np.float64)
        self.P = np.diag([0.01, 0.01, 0.01])
        self.Q = np.diag(
            [process_noise_xy ** 2, process_noise_xy ** 2, process_noise_theta ** 2]
        )
        self.R_yaw = np.array([[imu_yaw_noise ** 2]])
        self.frozen = False

    # -------------------------------------------------------------- #
    # Dome-scan support
    # -------------------------------------------------------------- #

    def freeze(self) -> None:
        """Hold pose constant - call before a stationary dome scan."""
        self.frozen = True

    def unfreeze(self) -> None:
        self.frozen = False

    # -------------------------------------------------------------- #
    # Filter
    # -------------------------------------------------------------- #

    def predict(self, d_left_m: float, d_right_m: float) -> Pose2D:
        """Motion update from wheel encoder distances since the last call."""
        if self.frozen:
            return self.pose

        d_center = (d_left_m + d_right_m) / 2.0
        d_theta = (d_right_m - d_left_m) / self.wheel_base_m
        theta = self.state[2]
        mid_theta = theta + d_theta / 2.0

        self.state[0] += d_center * np.cos(mid_theta)
        self.state[1] += d_center * np.sin(mid_theta)
        self.state[2] = wrap_angle(theta + d_theta)

        # Jacobian of the motion model w.r.t. state.
        F = np.array(
            [
                [1.0, 0.0, -d_center * np.sin(mid_theta)],
                [0.0, 1.0, d_center * np.cos(mid_theta)],
                [0.0, 0.0, 1.0],
            ]
        )
        # Scale process noise with distance travelled - standing still
        # should not accumulate uncertainty.
        scale = max(abs(d_center), 1e-4)
        self.P = F @ self.P @ F.T + self.Q * scale

        return self.pose

    def update_imu_yaw(self, measured_yaw_rad: float) -> Pose2D:
        """Measurement update from the BNO085's fused heading."""
        if self.frozen:
            return self.pose

        H = np.array([[0.0, 0.0, 1.0]])
        innovation = wrap_angle(measured_yaw_rad - self.state[2])
        S = H @ self.P @ H.T + self.R_yaw
        K = self.P @ H.T @ np.linalg.inv(S)

        self.state = self.state + (K @ np.array([innovation])).ravel()
        self.state[2] = wrap_angle(self.state[2])
        self.P = (np.eye(3) - K @ H) @ self.P

        return self.pose

    @property
    def pose(self) -> Pose2D:
        return Pose2D(
            float(self.state[0]), float(self.state[1]), float(self.state[2])
        )

    @property
    def uncertainty(self) -> tuple[float, float, float]:
        """1-sigma uncertainty. Worth surfacing on the UI: when this grows
        past ~0.5 m the operator should know the map is drifting."""
        return tuple(float(np.sqrt(v)) for v in np.diag(self.P))

    def reset(self, pose: Pose2D | None = None) -> None:
        pose = pose or Pose2D()
        self.state = np.array([pose.x, pose.y, pose.theta], dtype=np.float64)
        self.P = np.diag([0.01, 0.01, 0.01])


def ticks_to_distance(
    ticks: int, ticks_per_revolution: int, wheel_radius_m: float
) -> float:
    """Convert raw encoder ticks to linear distance in metres."""
    return float(2 * np.pi * wheel_radius_m * ticks / ticks_per_revolution)
