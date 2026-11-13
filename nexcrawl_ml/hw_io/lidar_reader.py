"""
RPLidar C1 reader, plus a simulator so the whole ML pipeline can be
developed and tested before the hardware exists.

The simulator is not a toy: it generates a synthetic tunnel with correct
geometry, so every downstream stage (transform, meshing, differencing,
classification) can be validated end-to-end on a laptop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

try:
    from rplidar import RPLidar

    _HAS_RPLIDAR = True
except ImportError:  # pragma: no cover
    _HAS_RPLIDAR = False


@dataclass
class Revolution:
    """One complete 360-degree LiDAR sweep."""

    ranges: np.ndarray
    beam_angles: np.ndarray
    quality: np.ndarray
    timestamp: float
    point_timestamps: np.ndarray | None = None  # per-point, for continuous mode

    def __len__(self) -> int:
        return len(self.ranges)


class RPLidarC1Reader:
    """Thin wrapper that yields complete revolutions.

    The vendor SDK emits individual measurements; grouping them into
    revolutions is what the step-and-stare strategy needs, since the whole
    revolution must be tagged with one tilt angle.
    """

    def __init__(self, port: str = "/dev/ttyUSB0", baudrate: int = 460800,
                 timeout: float = 3.0):
        if not _HAS_RPLIDAR:
            raise ImportError("rplidar-roboticia is required: pip install rplidar-roboticia")
        self.lidar = RPLidar(port, baudrate=baudrate, timeout=timeout)

    def info(self) -> dict:
        return {"info": self.lidar.get_info(), "health": self.lidar.get_health()}

    def revolutions(self, max_buf_meas: int = 3000):
        """Generator yielding Revolution objects."""
        try:
            for scan in self.lidar.iter_scans(max_buf_meas=max_buf_meas):
                arr = np.array(scan, dtype=np.float64)  # (N, 3): quality, angle, dist
                yield Revolution(
                    ranges=arr[:, 2] / 1000.0,              # mm -> m
                    beam_angles=np.deg2rad(arr[:, 1]),
                    quality=arr[:, 0],
                    timestamp=time.time(),
                )
        finally:
            self.stop()

    def stop(self) -> None:
        try:
            self.lidar.stop()
            self.lidar.stop_motor()
            self.lidar.disconnect()
        except Exception:
            pass


class SimulatedLidar:
    """Synthetic tunnel generator for offline development.

    Models a rectangular-ish mine drift with optional rockfall obstacles.
    Adds realistic range noise and dropout so downstream filtering is
    exercised properly rather than being handed perfect data.
    """

    def __init__(
        self,
        tunnel_width_m: float = 4.0,
        tunnel_height_m: float = 3.0,
        tunnel_length_m: float = 20.0,
        points_per_rev: int = 500,
        range_noise_m: float = 0.015,
        dropout_rate: float = 0.03,
        seed: int = 42,
    ):
        self.width = tunnel_width_m
        self.height = tunnel_height_m
        self.length = tunnel_length_m
        self.points_per_rev = points_per_rev
        self.range_noise_m = range_noise_m
        self.dropout_rate = dropout_rate
        self.rng = np.random.default_rng(seed)
        self.obstacles: list[dict] = []

    def add_rockfall(
        self, center: tuple[float, float, float], radius: float = 0.5
    ) -> None:
        """Place a spherical debris pile - used to generate 'after' scans."""
        self.obstacles.append({"center": np.array(center), "radius": radius})

    def clear_obstacles(self) -> None:
        self.obstacles.clear()

    def _ray_to_box(self, direction: np.ndarray, origin: np.ndarray) -> float:
        """Slab-method ray/box intersection against the tunnel interior."""
        bounds_min = np.array([-self.length / 2, -self.width / 2, 0.0])
        bounds_max = np.array([self.length / 2, self.width / 2, self.height])

        t_max = np.inf
        for axis in range(3):
            d = direction[axis]
            if abs(d) < 1e-9:
                continue
            t1 = (bounds_min[axis] - origin[axis]) / d
            t2 = (bounds_max[axis] - origin[axis]) / d
            t_far = max(t1, t2)
            t_max = min(t_max, t_far)
        return float(t_max) if np.isfinite(t_max) else 12.0

    def _ray_to_obstacles(self, direction: np.ndarray, origin: np.ndarray) -> float:
        """Analytic ray/sphere intersection, nearest hit wins."""
        best = np.inf
        for obs in self.obstacles:
            oc = origin - obs["center"]
            b = 2.0 * float(oc @ direction)
            c = float(oc @ oc) - obs["radius"] ** 2
            disc = b * b - 4 * c
            if disc < 0:
                continue
            t = (-b - np.sqrt(disc)) / 2.0
            if 0 < t < best:
                best = t
        return float(best)

    def revolution_at_tilt(
        self, tilt_rad: float, sensor_height_m: float = 0.25
    ) -> Revolution:
        """Generate one full 360-degree revolution at a given tilt angle.

        Ray direction uses the same rigid-body pitch rotation as
        perception.transforms.spherical_to_local - see that function's
        docstring for why this is NOT the spherical/latitude formula. Kept
        in sync deliberately: a simulator using different geometry than the
        real transform pipeline would validate nothing.
        """
        theta = np.linspace(0, 2 * np.pi, self.points_per_rev, endpoint=False)
        origin = np.array([0.0, 0.0, sensor_height_m])

        ranges = np.zeros(len(theta))
        for i, th in enumerate(theta):
            d = np.array(
                [
                    np.cos(tilt_rad) * np.cos(th),
                    np.sin(th),
                    np.sin(tilt_rad) * np.cos(th),
                ]
            )
            n = np.linalg.norm(d)
            if n < 1e-9:
                ranges[i] = 0.0
                continue
            d = d / n
            ranges[i] = min(
                self._ray_to_box(d, origin), self._ray_to_obstacles(d, origin)
            )

        ranges += self.rng.normal(0, self.range_noise_m, len(ranges))
        ranges = np.clip(ranges, 0.0, 12.0)

        keep = self.rng.random(len(ranges)) > self.dropout_rate
        quality = np.where(keep, 47.0, 0.0)
        ranges = np.where(keep, ranges, 0.0)

        return Revolution(
            ranges=ranges,
            beam_angles=theta,
            quality=quality,
            timestamp=time.time(),
        )

    def dome_sweep(
        self,
        phi_start_deg: float = -45.0,
        phi_end_deg: float = 45.0,
        step_deg: float = 2.5,
        sensor_height_m: float = 0.25,
    ) -> list[tuple[Revolution, float]]:
        """Full simulated step-and-stare dome scan."""
        angles = np.deg2rad(
            np.arange(phi_start_deg, phi_end_deg + 1e-9, step_deg)
        )
        return [
            (self.revolution_at_tilt(phi, sensor_height_m), float(phi))
            for phi in angles
        ]
