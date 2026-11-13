"""Direct pytest coverage of HazardPipeline.run() - previously only
exercised indirectly via the CLI script, never called straight from a test.
This is exactly the layer where the severity-zeroing bug lived, so it
deserves direct coverage rather than relying on manual script runs."""

import numpy as np
import pytest

open3d = pytest.importorskip("open3d", reason="open3d not installed")

from nexcrawl_ml.change_detection.registration import (
    RegistrationResult,
    apply_transform,
    initial_guess_from_odometry,
    register_to_baseline,
)
from nexcrawl_ml.hw_io.lidar_reader import SimulatedLidar
from nexcrawl_ml.hw_io.serial_reader import SimulatedServoLink
from nexcrawl_ml.models.structural_classifier import StructuralClassifier
from nexcrawl_ml.perception.calibration import RigCalibration
from nexcrawl_ml.pipeline.dome_scan import DomeScanConfig, DomeScanner
from nexcrawl_ml.pipeline.hazard_pipeline import HazardPipeline


def _dome_points(seed, rockfall=None):
    lidar = SimulatedLidar()
    if rockfall:
        lidar.add_rockfall(center=rockfall, radius=0.5)
    scanner = DomeScanner(
        lidar, SimulatedServoLink(), calibration=RigCalibration(),
        config=DomeScanConfig(phi_start_deg=-30, phi_end_deg=30, step_deg=5),
    )
    result = scanner.run()
    return result.point_cloud


def _trained_classifier():
    from nexcrawl_ml.change_detection.features import FEATURE_NAMES
    from nexcrawl_ml.models.structural_classifier import STRUCTURAL_CLASSES

    rng = np.random.default_rng(0)
    X, y = [], []
    for i, cls in enumerate(STRUCTURAL_CLASSES):
        centre = np.full(len(FEATURE_NAMES), i * 2.0)
        X.append(np.abs(rng.normal(centre, 0.3, (30, len(FEATURE_NAMES)))))
        y += [cls] * 30
    return StructuralClassifier(n_estimators=40).fit(np.vstack(X), np.array(y))


class TestHazardPipelineEndToEnd:
    def test_no_change_produces_no_hazards(self):
        baseline = _dome_points(seed=1)
        pipeline = HazardPipeline(_trained_classifier())
        report = pipeline.run(baseline, baseline)
        assert report.n_added_points < 50  # allow small reconstruction jitter
        assert report.registration_fitness > 0.9

    def test_injected_rockfall_is_detected_and_classified(self):
        baseline = _dome_points(seed=2)
        current = _dome_points(seed=2, rockfall=(2.5, 0.3, 0.4))
        pipeline = HazardPipeline(_trained_classifier())
        report = pipeline.run(current, baseline)

        assert report.n_added_points > 50
        assert report.n_blobs >= 1
        assert len(report.hazards) >= 1
        assert report.hazards[0].confidence > 0.0
        assert report.hazards[0].label in {
            "roof_fall", "rib_spall", "floor_heave", "equipment_moved", "artefact",
        }

    def test_report_serialises_to_dict(self):
        baseline = _dome_points(seed=3)
        current = _dome_points(seed=3, rockfall=(2.0, -0.4, 0.3))
        report = HazardPipeline(_trained_classifier()).run(current, baseline)
        d = report.as_dict()
        assert "hazards" in d and "registration" in d and "max_severity" in d

    def test_low_registration_fitness_produces_warning(self):
        """Feeding two completely unrelated point sets should trigger the
        low-fitness warning path rather than silently trusting a bad
        alignment."""
        baseline = _dome_points(seed=4)
        garbage = np.random.default_rng(5).uniform(-20, 20, (500, 3))
        report = HazardPipeline(_trained_classifier()).run(garbage, baseline)
        assert report.registration_fitness < 0.3
        assert any("fitness" in w for w in report.warnings)

    def test_odometry_seed_is_used_as_initial_guess(self):
        baseline = _dome_points(seed=6)
        current = _dome_points(seed=6, rockfall=(2.2, 0.2, 0.4))
        pipeline = HazardPipeline(_trained_classifier())
        # Should not raise, and a correct seed should not hurt fitness.
        report = pipeline.run(current, baseline, odometry_guess=(0.0, 0.0, 0.0))
        assert report.registration_fitness > 0.8


class TestRegistration:
    def test_identical_clouds_register_with_high_fitness(self):
        pts = _dome_points(seed=7)
        result = register_to_baseline(pts, pts)
        assert isinstance(result, RegistrationResult)
        assert result.fitness > 0.9
        assert result.converged

    def test_apply_transform_identity_is_noop(self):
        pts = np.random.default_rng(0).normal(0, 1, (50, 3))
        out = apply_transform(pts, np.eye(4))
        assert np.allclose(out, pts)

    def test_apply_transform_translation(self):
        pts = np.zeros((5, 3))
        T = initial_guess_from_odometry(dx=1.0, dy=2.0, dz=0.5)
        out = apply_transform(pts, T)
        assert np.allclose(out, np.tile([1.0, 2.0, 0.5], (5, 1)))

    def test_initial_guess_yaw_rotation_matches_rotation_matrix(self):
        T = initial_guess_from_odometry(dyaw=np.pi / 2)
        pts = np.array([[1.0, 0.0, 0.0]])
        out = apply_transform(pts, T)
        assert np.allclose(out, [[0.0, 1.0, 0.0]], atol=1e-9)
