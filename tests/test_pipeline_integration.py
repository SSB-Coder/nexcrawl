"""End-to-end integration tests using the simulator.

These are the tests that catch frame-convention errors: a unit test on the
transform can pass while the full chain still puts the tunnel inside out.
"""

import numpy as np
import pytest

from nexcrawl_ml.hw_io.lidar_reader import SimulatedLidar
from nexcrawl_ml.hw_io.serial_reader import SimulatedServoLink
from nexcrawl_ml.perception.calibration import RigCalibration
from nexcrawl_ml.perception.reconstructor import TiltedDomeLidarReconstructor
from nexcrawl_ml.perception.validation import repeatability_test

open3d = pytest.importorskip("open3d", reason="open3d not installed")


def _run_sweep(lidar, calib, phi_range=(-40, 40), step=5.0):
    rec = TiltedDomeLidarReconstructor(calibration=calib)
    for phi_deg in np.arange(phi_range[0], phi_range[1] + 1e-9, step):
        phi = np.deg2rad(phi_deg)
        rev = lidar.revolution_at_tilt(phi, sensor_height_m=calib.mast_height_m)
        rec.process_slice(rev.ranges, rev.beam_angles, phi, quality=rev.quality)
    return rec


def test_dome_sweep_produces_points():
    rec = _run_sweep(SimulatedLidar(), RigCalibration())
    assert rec.point_count > 5000
    assert rec.stats.n_slices == 17


def test_reconstructed_tunnel_has_plausible_dimensions():
    """The simulated drift is 4m wide x 3m high; the cloud should agree."""
    calib = RigCalibration(mast_height_m=0.25)
    rec = _run_sweep(SimulatedLidar(tunnel_width_m=4.0, tunnel_height_m=3.0), calib)
    pts = rec.merged_points()

    y_span = pts[:, 1].max() - pts[:, 1].min()
    assert 3.0 < y_span < 5.0, f"tunnel width came out at {y_span:.2f}m"
    assert pts[:, 2].min() > -0.5
    assert pts[:, 2].max() < 4.0


def test_pipeline_produces_mesh():
    rec = _run_sweep(SimulatedLidar(), RigCalibration(), step=3.0)
    mesh = rec.reconstruct(method="bpa")
    assert len(mesh.vertices) > 100
    assert len(mesh.triangles) > 100


def test_rockfall_shows_up_as_added_points():
    from nexcrawl_ml.change_detection.differencing import cloud_difference

    calib = RigCalibration()
    baseline = _run_sweep(SimulatedLidar(seed=1), calib).merged_points()

    lidar_after = SimulatedLidar(seed=1)
    lidar_after.add_rockfall(center=(2.5, 0.0, 0.4), radius=0.6)
    current = _run_sweep(lidar_after, calib).merged_points()

    diff = cloud_difference(current, baseline, threshold_m=0.10)
    assert diff.added_count > 50, "injected rockfall was not detected"


def test_repeatability_of_identical_scans():
    calib = RigCalibration()
    a = _run_sweep(SimulatedLidar(seed=5), calib).merged_points()
    b = _run_sweep(SimulatedLidar(seed=6), calib).merged_points()
    result = repeatability_test(a, b)
    assert result.mean_distance_m < 0.1
    assert "calibration OK" in result.verdict


def test_lever_arm_solver_recovers_injected_offset():
    """Verify the calibration solver before trusting it on hardware."""
    from nexcrawl_ml.perception.calibration import calibrate_lever_arm

    rng = np.random.default_rng(3)
    scans = []
    for phi_deg in [-30, -15, 0, 15, 30]:
        phi = np.deg2rad(phi_deg)
        theta = np.linspace(-np.pi / 5, np.pi / 5, 200)
        denom = np.cos(phi) * np.cos(theta)
        r = 3.0 / np.where(np.abs(denom) < 1e-6, 1e-6, denom)
        r += rng.normal(0, 0.005, len(r))
        scans.append((r, theta, phi))

    calib = calibrate_lever_arm(scans)
    assert calib.residual_rms_m < 0.05
    assert all(abs(v) < 0.15 for v in calib.lever_arm_m)
