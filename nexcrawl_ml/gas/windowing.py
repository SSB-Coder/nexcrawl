"""
Rolling-window feature extraction for the gas trend classifier.

The model never sees a raw reading. It sees the shape of the last N seconds:
is the level climbing, how fast, how noisily, and does the sensor still look
alive at all.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from ..models.gas_classifier import GAS_FEATURE_NAMES


@dataclass
class GasWindowFeatures:
    mean: float
    std: float
    rate_of_rise: float       # units per second, from a linear fit
    max_minus_min: float
    jitter: float             # mean absolute consecutive difference
    flatline_score: float     # 1.0 = suspiciously constant -> sensor fault
    last_value: float
    trend_r2: float           # how linear the trend is

    def to_array(self) -> np.ndarray:
        return np.array(
            [
                self.mean,
                self.std,
                self.rate_of_rise,
                self.max_minus_min,
                self.jitter,
                self.flatline_score,
                self.last_value,
                self.trend_r2,
            ],
            dtype=np.float64,
        )

    def as_dict(self) -> dict:
        return dict(zip(GAS_FEATURE_NAMES, self.to_array().tolist()))


def extract_window_features(
    values: np.ndarray, timestamps: np.ndarray | None = None
) -> GasWindowFeatures:
    """Compute the 8 trend features over one window of readings.

    flatline_score is the sensor-fault detector: a real gas sensor always
    has a little noise. A perfectly constant reading usually means the
    sensor has died, not that the air is perfectly stable - and a dead
    sensor reporting 'safe' is more dangerous than one reporting an alarm.
    """
    v = np.asarray(values, dtype=np.float64)
    if len(v) < 2:
        return GasWindowFeatures(
            float(v[0]) if len(v) else 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            float(v[0]) if len(v) else 0.0, 0.0,
        )

    t = (
        np.asarray(timestamps, dtype=np.float64)
        if timestamps is not None
        else np.arange(len(v), dtype=np.float64)
    )

    mean = float(v.mean())
    std = float(v.std())

    # Linear fit -> rate of rise and R^2.
    t_centred = t - t.mean()
    denom = float((t_centred ** 2).sum())
    if denom > 1e-12:
        slope = float((t_centred * (v - mean)).sum() / denom)
        predicted = mean + slope * t_centred
        ss_res = float(((v - predicted) ** 2).sum())
        ss_tot = float(((v - mean) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    else:
        slope, r2 = 0.0, 0.0

    diffs = np.diff(v)
    jitter = float(np.mean(np.abs(diffs)))

    # Near-zero jitter relative to the signal level = flatline.
    scale = max(abs(mean), 1e-6)
    flatline = float(np.exp(-jitter / (scale * 1e-3)))

    return GasWindowFeatures(
        mean=mean,
        std=std,
        rate_of_rise=slope,
        max_minus_min=float(v.max() - v.min()),
        jitter=jitter,
        flatline_score=min(flatline, 1.0),
        last_value=float(v[-1]),
        trend_r2=float(np.clip(r2, 0.0, 1.0)),
    )


class RollingGasBuffer:
    """Streaming buffer that emits a feature vector on demand."""

    def __init__(self, window_seconds: float = 10.0, sample_rate_hz: float = 1.0):
        self.window_seconds = window_seconds
        maxlen = max(int(window_seconds * sample_rate_hz), 4)
        self._values: deque[float] = deque(maxlen=maxlen)
        self._times: deque[float] = deque(maxlen=maxlen)

    def push(self, value: float, timestamp: float) -> None:
        self._values.append(float(value))
        self._times.append(float(timestamp))

    @property
    def ready(self) -> bool:
        return len(self._values) >= 4

    def features(self) -> GasWindowFeatures | None:
        if not self.ready:
            return None
        return extract_window_features(
            np.array(self._values), np.array(self._times)
        )

    def clear(self) -> None:
        self._values.clear()
        self._times.clear()


def windows_from_series(
    values: np.ndarray,
    timestamps: np.ndarray,
    window_seconds: float = 10.0,
    stride_seconds: float = 2.0,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Slice a logged series into overlapping windows for training.

    Returns (feature_matrix, window_time_spans).
    """
    values = np.asarray(values, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)

    features, spans = [], []
    t_start = timestamps[0]
    while t_start + window_seconds <= timestamps[-1]:
        t_end = t_start + window_seconds
        mask = (timestamps >= t_start) & (timestamps < t_end)
        if mask.sum() >= 4:
            features.append(
                extract_window_features(values[mask], timestamps[mask]).to_array()
            )
            spans.append((float(t_start), float(t_end)))
        t_start += stride_seconds

    matrix = np.vstack(features) if features else np.empty((0, len(GAS_FEATURE_NAMES)))
    return matrix, spans
