"""Tests for gas windowing features and explosibility physics."""

import numpy as np

from nexcrawl_ml.gas.physics import (
    CH4_LEL_PERCENT,
    check_sensor_tamper,
    classify_atmosphere,
    coward_explosibility,
    grahams_ratio,
    interpret_grahams_ratio,
    time_to_lel,
)
from nexcrawl_ml.gas.windowing import (
    RollingGasBuffer,
    extract_window_features,
    windows_from_series,
)


def test_rate_of_rise_on_linear_ramp():
    t = np.arange(20, dtype=float)
    v = 0.5 + 0.05 * t
    f = extract_window_features(v, t)
    assert np.isclose(f.rate_of_rise, 0.05, rtol=0.01)
    assert f.trend_r2 > 0.99


def test_flat_signal_has_zero_slope():
    f = extract_window_features(np.full(20, 0.5), np.arange(20, dtype=float))
    assert np.isclose(f.rate_of_rise, 0.0, atol=1e-9)


def test_flatline_score_detects_dead_sensor():
    """A perfectly constant reading means a dead sensor, not perfect air."""
    dead = extract_window_features(np.full(20, 0.5), np.arange(20, dtype=float))
    rng = np.random.default_rng(0)
    alive = extract_window_features(
        0.5 + rng.normal(0, 0.01, 20), np.arange(20, dtype=float)
    )
    assert dead.flatline_score > 0.9
    assert alive.flatline_score < 0.1


def test_rolling_buffer_needs_minimum_samples():
    buf = RollingGasBuffer(window_seconds=10.0, sample_rate_hz=1.0)
    assert not buf.ready
    assert buf.features() is None
    for i in range(5):
        buf.push(0.5, float(i))
    assert buf.ready
    assert buf.features() is not None


def test_windows_from_series_shapes():
    t = np.arange(0, 60, 1.0)
    v = 0.2 + 0.01 * t
    X, spans = windows_from_series(v, t, window_seconds=10.0, stride_seconds=5.0)
    assert X.shape[1] == 8
    assert len(spans) == X.shape[0]
    assert X.shape[0] > 5


def test_coward_explosive_range():
    state = coward_explosibility(ch4_percent=9.5, o2_percent=20.9)
    assert state.in_explosive_range
    assert state.risk_label == "EXPLOSIVE"


def test_coward_inert_when_oxygen_starved():
    """Fuel-rich but oxygen-poor is NOT safe - it explodes when air arrives."""
    state = coward_explosibility(ch4_percent=9.5, o2_percent=8.0)
    assert not state.in_explosive_range
    assert state.risk_label == "INERT_BUT_UNSTABLE"


def test_coward_safe_below_thresholds():
    assert coward_explosibility(0.1, 20.9).risk_label == "SAFE"


def test_coward_flags_approaching_lel():
    assert coward_explosibility(1.5, 20.9).risk_label == "APPROACHING_LEL"


def test_grahams_ratio_zero_without_o2_deficiency():
    assert grahams_ratio(co_ppm=50, o2_percent=20.9) == 0.0


def test_grahams_ratio_increases_with_co():
    low = grahams_ratio(co_ppm=20, o2_percent=19.9)
    high = grahams_ratio(co_ppm=200, o2_percent=19.9)
    assert high > low


def test_grahams_interpretation_bands():
    assert interpret_grahams_ratio(0.1) == "normal"
    assert interpret_grahams_ratio(0.6) == "heating_suspected"
    assert interpret_grahams_ratio(1.5) == "active_heating"
    assert interpret_grahams_ratio(3.0) == "open_fire_likely"


def test_time_to_lel_none_when_not_rising():
    assert time_to_lel(1.0, 0.0) is None
    assert time_to_lel(1.0, -0.01) is None


def test_time_to_lel_arithmetic():
    ttl = time_to_lel(current_ch4_percent=3.0, rate_of_rise_per_s=0.5)
    assert np.isclose(ttl, (CH4_LEL_PERCENT - 3.0) / 0.5)


def test_classify_atmosphere_safe_case():
    result = classify_atmosphere(0.1, 5, 20.9)
    assert result["overall_safe"]


def test_classify_atmosphere_flags_low_oxygen():
    result = classify_atmosphere(0.1, 5, 18.0)
    assert result["o2_status"] == "OXYGEN_DEFICIENT"
    assert not result["overall_safe"]


def test_grahams_ratio_matches_nitrogen_balance_by_hand():
    """Pin the exact DGMS/master-guide nitrogen-balance arithmetic so a
    future edit can't silently drift back toward the old simplified form."""
    co_ppm, o2, ch4, co2_ppm = 200.0, 19.9, 0.3, 500.0
    co_pct = co_ppm / 10000.0
    co2_pct = co2_ppm / 10000.0
    n2_pct = 100.0 - (ch4 + co2_pct + co_pct + o2)
    expected = round((co_pct / (0.265 * n2_pct - o2)) * 100.0, 2)
    assert grahams_ratio(co_ppm, o2, ch4_percent=ch4, co2_ppm=co2_ppm) == expected


def test_grahams_ratio_n2_floor_prevents_blowup():
    """Physically implausible gas sums (should trip tamper detection
    elsewhere) must not send this ratio to +-infinity."""
    result = grahams_ratio(co_ppm=500, o2_percent=15.0, ch4_percent=20.0, co2_ppm=2000.0)
    assert np.isfinite(result)


def test_grahams_ratio_changes_with_ch4_and_co2():
    """The whole point of the nitrogen-balance form vs. the old simplified
    one is that CH4/CO2 change the result at fixed CO and O2."""
    baseline = grahams_ratio(co_ppm=150, o2_percent=19.5, ch4_percent=0.0)
    with_ch4 = grahams_ratio(co_ppm=150, o2_percent=19.5, ch4_percent=3.0)
    assert not np.isclose(baseline, with_ch4)


def test_tamper_flags_ch4_jump_with_flat_o2():
    assert check_sensor_tamper(
        ch4_curr=2.0, ch4_prev=0.3, o2_curr=20.9, o2_prev=20.89
    ) == "WARNING: CH4_SENSOR_FAULT_OR_TAMPER"


def test_tamper_normal_when_o2_drops_with_ch4():
    """A real gas event: CH4 rising WITH O2 dropping is physically
    consistent and must NOT be flagged as tamper."""
    assert check_sensor_tamper(
        ch4_curr=2.0, ch4_prev=0.3, o2_curr=19.5, o2_prev=20.9
    ) == "NORMAL"


def test_tamper_normal_for_small_ch4_change():
    assert check_sensor_tamper(
        ch4_curr=0.5, ch4_prev=0.3, o2_curr=20.9, o2_prev=20.89
    ) == "NORMAL"


def test_classify_atmosphere_reports_unknown_tamper_without_prior():
    result = classify_atmosphere(0.5, 20, 20.9)
    assert result["sensor_tamper_status"] == "UNKNOWN"


def test_classify_atmosphere_detects_tamper_with_prior():
    result = classify_atmosphere(
        ch4_percent=2.0, co_ppm=20, o2_percent=20.9,
        prior_ch4_percent=0.3, prior_o2_percent=20.89,
    )
    assert result["sensor_tamper_status"] == "WARNING: CH4_SENSOR_FAULT_OR_TAMPER"
    assert not result["overall_safe"]
