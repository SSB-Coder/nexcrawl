"""
End-to-end dome scan orchestrator.

Ties together: servo stepping -> revolution capture -> transform ->
accumulation -> mesh reconstruction, with optional progressive callbacks so
the frontend can render the dome building up live rather than showing a
spinner and then a finished mesh.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from ..hw_io.lidar_reader import Revolution, SimulatedLidar
from ..hw_io.serial_reader import SimulatedServoLink
from ..perception.calibration import RigCalibration
from ..perception.reconstructor import (
    ReconstructionStats,
    TiltedDomeLidarReconstructor,
)


@dataclass
class DomeScanConfig:
    phi_start_deg: float = -45.0
    phi_end_deg: float = 45.0
    step_deg: float = 2.5
    settle_time_s: float = 0.12
    revolution_timeout_s: float = 1.0
    method: str = "bpa"
    progressive_updates: bool = True
    update_every_n_steps: int = 5

    @property
    def n_steps(self) -> int:
        return int(
            np.floor((self.phi_end_deg - self.phi_start_deg) / self.step_deg)
        ) + 1


@dataclass
class DomeScanResult:
    mesh: object | None
    stats: ReconstructionStats
    point_cloud: np.ndarray
    steps_completed: int
    steps_attempted: int
    duration_s: float
    output_path: Path | None = None

    @property
    def success_rate(self) -> float:
        return self.steps_completed / max(self.steps_attempted, 1)


class DomeScanner:
    """Runs a complete stationary dome scan.

    Pose is deliberately held fixed for the whole sweep: the rover is
    parked, so querying the EKF per slice would only add drift to a
    measurement that is exactly known.
    """

    def __init__(
        self,
        lidar,
        servo,
        calibration: RigCalibration | None = None,
        config: DomeScanConfig | None = None,
        reconstructor_kwargs: dict | None = None,
    ):
        self.lidar = lidar
        self.servo = servo
        self.config = config or DomeScanConfig()
        self.calib = calibration or RigCalibration()
        self.reconstructor = TiltedDomeLidarReconstructor(
            calibration=self.calib, **(reconstructor_kwargs or {})
        )

    def _tilt_targets(self) -> np.ndarray:
        c = self.config
        return np.deg2rad(
            np.arange(c.phi_start_deg, c.phi_end_deg + 1e-9, c.step_deg)
        )

    def run(
        self,
        rover_pose: tuple[float, float, float] = (0.0, 0.0, 0.0),
        output_path: str | Path | None = None,
        on_slice: Callable[[int, int, int], None] | None = None,
        on_progressive_mesh: Callable[[object, float], None] | None = None,
    ) -> DomeScanResult:
        """Execute the sweep.

        on_slice(step_index, n_steps, points_added) fires per tilt step.
        on_progressive_mesh(mesh, progress) fires periodically if enabled.
        """
        t0 = time.perf_counter()
        self.reconstructor.clear()

        rover_x, rover_y, rover_yaw = rover_pose
        targets = self._tilt_targets()
        completed = 0

        for i, phi_target in enumerate(targets):
            self.servo.set_tilt(phi_target)
            settled = self.servo.wait_until_settled(
                phi_target, settle_time_s=self.config.settle_time_s
            )
            if not settled:
                continue

            rev = self._capture_revolution(phi_target)
            if rev is None:
                continue

            status = self.servo.latest
            phi_actual = status.angle_rad if status else float(phi_target)

            n_added = self.reconstructor.process_slice(
                rev.ranges,
                rev.beam_angles,
                phi_actual,
                rover_x=rover_x,
                rover_y=rover_y,
                rover_yaw=rover_yaw,
                quality=rev.quality,
            )
            completed += 1

            if on_slice:
                on_slice(i, len(targets), n_added)

            if (
                self.config.progressive_updates
                and on_progressive_mesh
                and completed % self.config.update_every_n_steps == 0
            ):
                try:
                    partial = self.reconstructor.reconstruct(
                        method=self.config.method
                    )
                    on_progressive_mesh(partial, (i + 1) / len(targets))
                except Exception:
                    pass  # partial clouds can legitimately be too sparse to mesh

        mesh = None
        try:
            mesh = self.reconstructor.reconstruct(
                method=self.config.method, output_path=output_path
            )
        except Exception as exc:
            print(f"[dome_scan] final reconstruction failed: {exc}")

        return DomeScanResult(
            mesh=mesh,
            stats=self.reconstructor.stats,
            point_cloud=self.reconstructor.merged_points(),
            steps_completed=completed,
            steps_attempted=len(targets),
            duration_s=time.perf_counter() - t0,
            output_path=Path(output_path) if output_path else None,
        )

    def _capture_revolution(self, phi: float) -> Revolution | None:
        """Grab one full revolution at the settled tilt angle."""
        if isinstance(self.lidar, SimulatedLidar):
            return self.lidar.revolution_at_tilt(
                phi, sensor_height_m=self.calib.mast_height_m
            )
        try:
            return next(self.lidar.revolutions())
        except (StopIteration, Exception):
            return None


def run_simulated_dome_scan(
    config: DomeScanConfig | None = None,
    calibration: RigCalibration | None = None,
    with_rockfall: bool = False,
    output_path: str | Path | None = None,
) -> DomeScanResult:
    """Convenience entry point for offline development and CI."""
    lidar = SimulatedLidar()
    if with_rockfall:
        lidar.add_rockfall(center=(3.0, 0.5, 0.4), radius=0.6)

    scanner = DomeScanner(
        lidar=lidar,
        servo=SimulatedServoLink(),
        calibration=calibration,
        config=config,
    )
    return scanner.run(output_path=output_path)
