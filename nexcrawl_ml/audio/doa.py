"""
GCC-PHAT direction-of-arrival estimation for knock localisation.

Once KnockNet says "that was a knock", the next operational question is
"which way?". With two microphones a known distance apart, the time
difference of arrival gives the bearing.

PHAT weighting (rather than plain cross-correlation) is used because it
whitens the signal magnitude and keeps only phase information, which is
substantially more robust in the reverberant, echo-heavy conditions of an
underground tunnel.

Note the inherent limitation: a two-mic linear array cannot distinguish
front from back. That ambiguity is real and worth stating plainly rather
than hiding - resolving it needs a third microphone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SPEED_OF_SOUND_MS = 343.0


@dataclass
class DOAResult:
    tdoa_s: float
    bearing_deg: float
    confidence: float
    ambiguous_mirror_deg: float

    def summary(self) -> str:
        return (
            f"bearing {self.bearing_deg:+.1f} deg "
            f"(or mirror {self.ambiguous_mirror_deg:+.1f}), "
            f"tdoa={self.tdoa_s*1e6:.1f} us, conf={self.confidence:.2f}"
        )


def gcc_phat(
    sig: np.ndarray,
    ref: np.ndarray,
    sample_rate: int,
    max_tau: float | None = None,
    interp: int = 16,
) -> tuple[float, float]:
    """Generalised cross-correlation with phase transform.

    Returns (tdoa_seconds, confidence). The interp factor upsamples the
    correlation peak so the resolution is not limited to one sample period -
    at 22.05 kHz one sample is ~45 us, which alone is too coarse for a
    short mic baseline.

    Known limitation: PHAT weighting flattens every frequency bin's
    magnitude to 1 before correlating, keeping only phase. On a broadband
    transient (a real knock's impact signature) this is exactly what makes
    PHAT robust to reverberation. On a near-pure tone, most of the spectrum
    outside the one active frequency carries no real signal at all - PHAT
    still amplifies that near-silent noise to full weight, and the delay
    estimate can become unreliable. If a real knock signal turns out to
    carry strong single-frequency resonance (e.g. striking a metal pipe),
    bearing estimation confidence should be expected to degrade accordingly.
    """
    sig = np.asarray(sig, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    n = len(sig) + len(ref)

    SIG = np.fft.rfft(sig, n=n)
    REF = np.fft.rfft(ref, n=n)
    cross = SIG * np.conj(REF)

    denom = np.abs(cross)
    denom[denom < 1e-12] = 1e-12
    cross_phat = cross / denom  # PHAT weighting: keep phase, discard magnitude

    cc = np.fft.irfft(cross_phat, n=interp * n)

    max_shift = interp * n // 2
    if max_tau is not None:
        max_shift = min(int(interp * sample_rate * max_tau), max_shift)

    cc = np.concatenate([cc[-max_shift:], cc[: max_shift + 1]])
    peak_index = int(np.argmax(np.abs(cc)))
    shift = peak_index - max_shift

    peak_value = float(np.abs(cc[peak_index]))
    mean_value = float(np.mean(np.abs(cc)))
    confidence = min(peak_value / (mean_value + 1e-12) / 10.0, 1.0)

    return shift / float(interp * sample_rate), confidence


def estimate_bearing(
    mic_left: np.ndarray,
    mic_right: np.ndarray,
    sample_rate: int,
    mic_spacing_m: float = 0.15,
) -> DOAResult:
    """Two-microphone bearing estimate.

    mic_spacing_m must be measured on the real rig, not assumed. Getting it
    wrong scales every bearing estimate by a constant factor.
    """
    max_tau = mic_spacing_m / SPEED_OF_SOUND_MS
    tdoa, confidence = gcc_phat(
        mic_left, mic_right, sample_rate, max_tau=max_tau
    )

    ratio = np.clip(tdoa * SPEED_OF_SOUND_MS / mic_spacing_m, -1.0, 1.0)
    bearing = float(np.degrees(np.arcsin(ratio)))

    return DOAResult(
        tdoa_s=float(tdoa),
        bearing_deg=bearing,
        confidence=float(confidence),
        ambiguous_mirror_deg=180.0 - bearing if bearing >= 0 else -180.0 - bearing,
    )
