"""Perception: 2D LiDAR slices -> calibrated 3D mesh."""

from .calibration import RigCalibration, calibrate_lever_arm, calibrate_servo_zero
from .reconstructor import (
    ReconstructionStats,
    TiltedDomeLidarReconstructor,
    mesh_to_payload,
)
from .time_sync import AngleBuffer, ClockOffsetEstimator, StepAndStareController
from .transforms import (
    apply_lever_arm,
    local_to_world,
    range_gate,
    spherical_to_local,
    spherical_to_local_varying_phi,
)

__all__ = [
    "RigCalibration",
    "calibrate_lever_arm",
    "calibrate_servo_zero",
    "TiltedDomeLidarReconstructor",
    "ReconstructionStats",
    "mesh_to_payload",
    "AngleBuffer",
    "ClockOffsetEstimator",
    "StepAndStareController",
    "spherical_to_local",
    "spherical_to_local_varying_phi",
    "apply_lever_arm",
    "local_to_world",
    "range_gate",
]
