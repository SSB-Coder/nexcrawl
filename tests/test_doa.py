"""Tests for GCC-PHAT bearing estimation on synthetic time-delayed audio."""

import numpy as np

from nexcrawl_ml.audio.doa import SPEED_OF_SOUND_MS, estimate_bearing, gcc_phat


def _delayed_signal(duration_s=0.05, sr=48000, seed=0, delay_samples=0):
    """Broadband noise burst, not a pure tone.

    GCC-PHAT flattens every frequency bin's magnitude to 1 before
    correlating, keeping only phase. For a near-pure sinusoid, almost the
    entire spectrum outside the one active frequency is just FFT leakage/
    numerical noise - PHAT amplifies that noise to full weight right along
    with the real signal, and the delay estimate becomes unreliable. This is
    a genuine, known GCC-PHAT limitation on narrowband/tonal sources, not a
    quirk of this implementation - broadband transients (which is what a
    real knock's impact signature actually looks like, per the project's own
    'impact transients vs horizontal bands' description of knock vs. motor
    noise) are exactly the case PHAT is designed for and tested against here.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * sr)
    base = rng.normal(0, 1, n) * np.hanning(n)
    delayed = np.zeros(n)
    if delay_samples >= 0:
        delayed[delay_samples:] = base[: n - delay_samples]
    else:
        delayed[: n + delay_samples] = base[-delay_samples:]
    return base, delayed


def test_gcc_phat_recovers_known_positive_delay():
    sr = 48000
    delay_samples = 10
    ref, sig = _delayed_signal(sr=sr, delay_samples=delay_samples)
    tdoa, confidence = gcc_phat(sig, ref, sr)
    expected = delay_samples / sr
    assert np.isclose(tdoa, expected, atol=1e-4)
    assert confidence > 0


def test_gcc_phat_recovers_known_negative_delay():
    sr = 48000
    delay_samples = -8
    ref, sig = _delayed_signal(sr=sr, delay_samples=delay_samples)
    tdoa, _ = gcc_phat(sig, ref, sr)
    expected = delay_samples / sr
    assert np.isclose(tdoa, expected, atol=1e-4)


def test_gcc_phat_zero_delay_for_identical_signals():
    sr = 48000
    ref, _ = _delayed_signal(sr=sr, delay_samples=0)
    tdoa, confidence = gcc_phat(ref, ref, sr)
    assert np.isclose(tdoa, 0.0, atol=1e-4)
    assert confidence > 0.5


def test_estimate_bearing_broadside_source_is_near_zero():
    """A source directly ahead (equidistant from both mics) has ~zero TDOA
    and should report a bearing near 0 degrees."""
    sr = 48000
    ref, sig = _delayed_signal(sr=sr, delay_samples=0)
    result = estimate_bearing(sig, ref, sr, mic_spacing_m=0.40)
    assert abs(result.bearing_deg) < 5.0


def test_estimate_bearing_endfire_source_is_near_90():
    """A source directly to one side produces the maximum physically
    possible TDOA for the given mic spacing: d/c seconds."""
    sr = 48000
    mic_spacing_m = 0.40
    max_delay_s = mic_spacing_m / SPEED_OF_SOUND_MS
    delay_samples = int(round(max_delay_s * sr))
    ref, sig = _delayed_signal(sr=sr, delay_samples=delay_samples)
    result = estimate_bearing(sig, ref, sr, mic_spacing_m=mic_spacing_m)
    assert abs(result.bearing_deg) > 60.0


def test_estimate_bearing_reports_ambiguous_mirror():
    """Two mics cannot distinguish front from back - the mirror bearing
    must always be reported explicitly rather than hidden."""
    sr = 48000
    ref, sig = _delayed_signal(sr=sr, delay_samples=5)
    result = estimate_bearing(sig, ref, sr, mic_spacing_m=0.40)
    assert result.ambiguous_mirror_deg != result.bearing_deg
    assert np.isfinite(result.ambiguous_mirror_deg)


def test_estimate_bearing_scales_with_mic_spacing():
    """Same physical delay, different assumed spacing, must give a
    different bearing - this is exactly why mic_spacing_m must match the
    physical rig (config default fixed to 0.40m to match the technical
    proposal's front-bumper baseline)."""
    sr = 48000
    ref, sig = _delayed_signal(sr=sr, delay_samples=6)
    narrow = estimate_bearing(sig, ref, sr, mic_spacing_m=0.15)
    wide = estimate_bearing(sig, ref, sr, mic_spacing_m=0.40)
    assert not np.isclose(narrow.bearing_deg, wide.bearing_deg)
