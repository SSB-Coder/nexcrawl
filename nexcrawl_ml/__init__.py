"""
NexCrawl-M ML stack (SIH 2026, PS 26039).

Perception & Machine Learning subsystem for the underground mine safety
and rescue rover.

Layout:
    perception/       2D LiDAR slices -> calibrated 3D mesh
    change_detection/ ICP + differencing + DBSCAN + geometric features
    models/           LightGBM x2, KnockNet CNN, calibration wrappers
    audio/            mel features + GCC-PHAT bearing
    gas/              rolling-window features + explosibility physics
    fusion/           wheel + IMU EKF pose
    hw_io/            LiDAR and ESP32 links, plus simulators
    pipeline/         end-to-end orchestrators
"""

__version__ = "1.0.0"
__all__ = [
    "perception",
    "change_detection",
    "models",
    "audio",
    "gas",
    "fusion",
    "hw_io",
    "pipeline",
]
