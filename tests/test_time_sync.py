"""Tests for encoder/LiDAR time synchronisation."""

import numpy as np
import pytest

from nexcrawl_ml.perception.time_sync import (
    AngleBuffer,
    ClockOffsetEstimator,
    StepAndStareController,
)


def test_interpolate_midpoint():
    buf = AngleBuffer()
    buf.push(0.0, 0.0)
    buf.push(1.0, 1.0)
    assert np.isclose(buf.interpolate(0.5), 0.5)


def test_interpolate_outside_window_returns_none():
    """Extrapolating past a servo direction reversal produces badly wrong
    geometry, so out-of-window queries must fail loudly rather than guess."""
    buf = AngleBuffer()
    buf.push(1.0, 0.0)
    buf.push(2.0, 1.0)
    assert buf.interpolate(0.5) is None
    assert buf.interpolate(3.0) is None


def test_interpolate_insufficient_samples():
    buf = AngleBuffer()
    assert buf.interpolate(0.5) is None
    buf.push(0.0, 0.0)
    assert buf.interpolate(0.0) is None


def test_interpolate_many_masks_out_of_range():
    buf = AngleBuffer()
    for i in range(5):
        buf.push(float(i), float(i) * 0.1)
    phi, valid = buf.interpolate_many(np.array([-1.0, 0.5, 2.5, 10.0]))
    assert list(valid) == [False, True, True, False]
    assert np.isclose(phi[1], 0.05)


def test_clock_offset_median_robust_to_outlier():
    est = ClockOffsetEstimator()
    for _ in range(10):
        est.add_sync_pulse(t_pi=100.0, t_esp=99.0)
    est.add_sync_pulse(t_pi=100.0, t_esp=50.0)  # one wild packet
    assert np.isclose(est.offset, 1.0)


def test_step_and_stare_requires_settle_time():
    ctrl = StepAndStareController(
        phi_start_rad=0.0, phi_end_rad=0.1, step_rad=0.05, settle_time_s=0.1
    )
    assert not ctrl.update(ctrl.current_target, now=0.0)   # just arrived
    assert not ctrl.update(ctrl.current_target, now=0.05)  # not settled yet
    assert ctrl.update(ctrl.current_target, now=0.15)      # settled


def test_step_and_stare_resets_on_drift():
    ctrl = StepAndStareController(
        phi_start_rad=0.0, phi_end_rad=0.1, step_rad=0.05, settle_time_s=0.1
    )
    ctrl.update(0.0, now=0.0)
    ctrl.update(0.5, now=0.05)          # drifted away
    assert not ctrl.update(0.0, now=0.12)  # settle timer must have restarted


def test_step_and_stare_advances_and_finishes():
    ctrl = StepAndStareController(
        phi_start_rad=0.0, phi_end_rad=0.1, step_rad=0.05, settle_time_s=0.0
    )
    n = ctrl.n_steps
    for _ in range(n):
        ctrl.advance()
    assert ctrl.finished
