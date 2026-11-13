"""Tests for alert fusion: deterministic physics must outrank probabilistic
ML, and severe hazards must map to CRITICAL priority."""

import numpy as np

from nexcrawl_ml.gas.physics import classify_atmosphere
from nexcrawl_ml.pipeline.alert_fusion import (
    Alert,
    AlertPriority,
    acoustic_alerts,
    fuse_alerts,
    gas_alerts,
    structural_alerts,
)
from nexcrawl_ml.pipeline.hazard_pipeline import Hazard, HazardReport


def _hazard(label, confidence, severity, n_points=100):
    return Hazard(
        label=label,
        confidence=confidence,
        severity=severity,
        centroid=(1.0, 2.0, 0.5),
        n_points=n_points,
        features={"volume_m3": 0.5},
    )


def _report(*hazards):
    return HazardReport(
        hazards=list(hazards),
        registration_fitness=0.95,
        registration_rmse=0.005,
        n_added_points=200,
        n_removed_points=0,
        n_blobs=len(hazards),
        eps_used=0.1,
        duration_s=0.1,
    )


class TestStructuralAlerts:
    def test_roof_fall_maps_to_critical(self):
        report = _report(_hazard("roof_fall", 0.9, severity=4))
        alerts = structural_alerts(report)
        assert len(alerts) == 1
        assert alerts[0].priority == AlertPriority.CRITICAL

    def test_artefact_is_filtered_out_by_default(self):
        report = _report(_hazard("artefact", 0.9, severity=0))
        assert structural_alerts(report) == []

    def test_low_confidence_hazard_is_filtered(self):
        report = _report(_hazard("roof_fall", 0.2, severity=4))
        assert structural_alerts(report, min_confidence=0.5) == []

    def test_equipment_moved_is_low_priority(self):
        report = _report(_hazard("equipment_moved", 0.8, severity=1))
        alerts = structural_alerts(report)
        assert alerts[0].priority == AlertPriority.LOW


class TestGasAlertsDeterministicPriority:
    def test_explosive_atmosphere_is_critical(self):
        atmosphere = classify_atmosphere(ch4_percent=9.5, co_ppm=50, o2_percent=20.9)
        alerts = gas_alerts(atmosphere)
        assert any(
            a.priority == AlertPriority.CRITICAL and a.deterministic for a in alerts
        )

    def test_open_fire_grahams_ratio_is_critical(self):
        # Verified against grahams_ratio directly: co=2000ppm, o2=13.0%,
        # ch4=0.1% -> ratio ~2.01, which is in the open_fire_likely band.
        atmosphere = classify_atmosphere(
            ch4_percent=0.1, co_ppm=2000, o2_percent=13.0
        )
        assert atmosphere["grahams_interpretation"] == "open_fire_likely"
        alerts = gas_alerts(atmosphere)
        assert any(
            a.label == "open_fire_likely" and a.priority == AlertPriority.CRITICAL
            for a in alerts
        )

    def test_sensor_tamper_produces_high_priority_deterministic_alert(self):
        atmosphere = classify_atmosphere(
            ch4_percent=2.0, co_ppm=10, o2_percent=20.9,
            prior_ch4_percent=0.3, prior_o2_percent=20.89,
        )
        alerts = gas_alerts(atmosphere)
        tamper_alerts = [a for a in alerts if a.label == "SENSOR_TAMPER_SUSPECTED"]
        assert len(tamper_alerts) == 1
        assert tamper_alerts[0].priority == AlertPriority.HIGH
        assert tamper_alerts[0].deterministic

    def test_safe_atmosphere_produces_no_physics_alerts(self):
        atmosphere = classify_atmosphere(ch4_percent=0.1, co_ppm=5, o2_percent=20.9)
        alerts = gas_alerts(atmosphere)
        assert alerts == []


class TestPhysicsOutranksML:
    """The core safety invariant: a deterministic explosibility fact must
    sort ahead of a probabilistic ML alert, priority tie or not."""

    def test_deterministic_wins_priority_tie_against_ml(self):
        ml_alert = Alert(
            source="structural", priority=AlertPriority.CRITICAL,
            label="roof_fall", message="ML says roof fall", confidence=0.95,
            deterministic=False,
        )
        physics_alert = Alert(
            source="physics", priority=AlertPriority.CRITICAL,
            label="EXPLOSIVE", message="Explosive atmosphere",
            deterministic=True,
        )
        fused = fuse_alerts([ml_alert], [physics_alert])
        assert fused[0].deterministic
        assert fused[0].source == "physics"

    def test_higher_priority_always_wins_regardless_of_determinism(self):
        low_priority_physics = Alert(
            source="physics", priority=AlertPriority.LOW,
            label="minor", message="minor", deterministic=True,
        )
        high_priority_ml = Alert(
            source="structural", priority=AlertPriority.CRITICAL,
            label="roof_fall", message="roof fall", confidence=0.99,
            deterministic=False,
        )
        fused = fuse_alerts([high_priority_ml], [low_priority_physics])
        assert fused[0].source == "structural"

    def test_end_to_end_explosive_plus_roof_fall_puts_physics_first(self):
        atmosphere = classify_atmosphere(ch4_percent=9.5, co_ppm=50, o2_percent=20.9)
        gas = gas_alerts(atmosphere)
        report = _report(_hazard("roof_fall", 0.99, severity=4))
        structural = structural_alerts(report)
        fused = fuse_alerts(gas, structural)
        assert fused[0].source == "physics"
        assert fused[0].deterministic


class TestAcousticAlerts:
    def test_knock_above_threshold_is_critical(self):
        pred = {"label": "knock", "confidence": 0.9}
        alerts = acoustic_alerts(pred)
        assert len(alerts) == 1
        assert alerts[0].priority == AlertPriority.CRITICAL

    def test_voice_is_not_treated_as_knock_alert(self):
        """With the 4-class KnockNet, a voice prediction must not trigger
        the knock alert path - confusing the two defeats the point of
        splitting them into separate classes."""
        pred = {"label": "voice", "confidence": 0.9}
        assert acoustic_alerts(pred) == []

    def test_low_confidence_knock_is_suppressed(self):
        pred = {"label": "knock", "confidence": 0.3}
        assert acoustic_alerts(pred, min_confidence=0.6) == []

    def test_bearing_and_mirror_both_reported(self):
        class FakeDOA:
            bearing_deg = 30.0
            ambiguous_mirror_deg = 150.0

        pred = {"label": "knock", "confidence": 0.9}
        alerts = acoustic_alerts(pred, doa_result=FakeDOA())
        assert "+30" in alerts[0].message
        assert "150" in alerts[0].message
