"""
Time synchronisation between the LiDAR USB stream and the ESP32 encoder stream.

Two strategies (see Section 3 of the plan):

  Strategy 1 - STEP AND STARE  (recommended default / demo path)
      Servo stops, one full revolution is captured, every point in that
      revolution shares one confirmed phi. No interpolation, no clock
      alignment, nothing to get wrong.

  Strategy 2 - CONTINUOUS NOD  (stretch goal, driving mode)
      Servo sweeps while the LiDAR spins. Each point's phi must be
      interpolated from the bracketing encoder samples, which also requires
      knowing the ESP32 <-> Pi clock offset.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class AngleSample:
    """One (timestamp, angle) pair from the STS3215 magnetic encoder."""

    t: float          # seconds, ESP32 clock (already offset-corrected)
    phi_rad: float    # raw encoder angle, zero-offset NOT yet applied


class AngleBuffer:
    """Rolling buffer of encoder samples, queryable by timestamp.

    Kept deliberately small: at 100 Hz a 5 second window is 500 samples,
    which is plenty to bracket any LiDAR point still in flight.
    """

    def __init__(self, maxlen: int = 2000):
        self._samples: deque[AngleSample] = deque(maxlen=maxlen)

    def push(self, t: float, phi_rad: float) -> None:
        self._samples.append(AngleSample(t, phi_rad))

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def times(self) -> list[float]:
        return [s.t for s in self._samples]

    def latest(self) -> AngleSample | None:
        return self._samples[-1] if self._samples else None

    def interpolate(self, t: float) -> float | None:
        """Linear interpolation of phi at an arbitrary timestamp.

            phi(t) = phi0 + (phi1 - phi0) * (t - t0) / (t1 - t0)

        Returns None when t falls outside the buffered window - the caller
        should drop those points rather than extrapolate, because
        extrapolating a servo angle through a direction reversal produces
        badly wrong geometry.
        """
        if len(self._samples) < 2:
            return None

        samples = list(self._samples)
        times = [s.t for s in samples]

        if t < times[0] or t > times[-1]:
            return None

        idx = bisect_left(times, t)
        if idx == 0:
            return samples[0].phi_rad
        if idx >= len(samples):
            return samples[-1].phi_rad

        s0, s1 = samples[idx - 1], samples[idx]
        dt = s1.t - s0.t
        if dt <= 0:
            return s0.phi_rad
        alpha = (t - s0.t) / dt
        return s0.phi_rad + alpha * (s1.phi_rad - s0.phi_rad)

    def interpolate_many(self, timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised interpolation for a whole revolution of points.

        Returns (phi_values, valid_mask). Points outside the buffered window
        come back masked out.
        """
        timestamps = np.asarray(timestamps, dtype=np.float64)
        if len(self._samples) < 2:
            return np.zeros_like(timestamps), np.zeros_like(timestamps, dtype=bool)

        samples = list(self._samples)
        t_arr = np.array([s.t for s in samples])
        phi_arr = np.array([s.phi_rad for s in samples])

        valid = (timestamps >= t_arr[0]) & (timestamps <= t_arr[-1])
        phi_out = np.zeros_like(timestamps)
        if valid.any():
            phi_out[valid] = np.interp(timestamps[valid], t_arr, phi_arr)
        return phi_out, valid


class ClockOffsetEstimator:
    """Estimate the ESP32 -> Pi clock offset.

    Two supported methods:
      - 'sync_pulse': ESP32 toggles a GPIO the Pi also timestamps. Most
        accurate; needed if you pursue continuous-nod mode seriously.
      - 'round_trip': ping/echo over serial, offset ~= (t_recv - t_send)/2.
        Cruder but needs no extra wiring.

    A running median is used rather than a mean because serial latency has a
    long right tail - one delayed packet should not drag the estimate.
    """

    def __init__(self, window: int = 51):
        self._offsets: deque[float] = deque(maxlen=window)

    def add_round_trip(self, t_send: float, t_esp: float, t_recv: float) -> float:
        one_way = (t_recv - t_send) / 2.0
        self._offsets.append((t_send + one_way) - t_esp)
        return self.offset

    def add_sync_pulse(self, t_pi: float, t_esp: float) -> float:
        self._offsets.append(t_pi - t_esp)
        return self.offset

    @property
    def offset(self) -> float:
        if not self._offsets:
            return 0.0
        return float(np.median(self._offsets))

    def to_pi_clock(self, t_esp: float) -> float:
        return t_esp + self.offset


class StepAndStareController:
    """Sequencer for Strategy 1.

    State machine per tilt step:
        COMMAND -> SETTLING -> CAPTURING -> DONE

    The important guarantee: a revolution is only accepted if the servo was
    confirmed stationary for its entire duration. That is what makes the
    'one phi for the whole revolution' assumption safe.
    """

    def __init__(
        self,
        phi_start_rad: float = np.deg2rad(-45.0),
        phi_end_rad: float = np.deg2rad(45.0),
        step_rad: float = np.deg2rad(2.5),
        settle_time_s: float = 0.12,
        angle_tolerance_rad: float = np.deg2rad(0.3),
    ):
        self.targets = np.arange(phi_start_rad, phi_end_rad + 1e-9, step_rad)
        self.settle_time_s = settle_time_s
        self.angle_tolerance_rad = angle_tolerance_rad
        self.reset()

    def reset(self) -> None:
        self._index = 0
        self._arrived_at: float | None = None

    @property
    def n_steps(self) -> int:
        return len(self.targets)

    @property
    def finished(self) -> bool:
        return self._index >= len(self.targets)

    @property
    def current_target(self) -> float:
        return float(self.targets[min(self._index, len(self.targets) - 1)])

    @property
    def progress(self) -> float:
        return self._index / max(len(self.targets), 1)

    def update(self, measured_phi: float, now: float) -> bool:
        """Feed in the latest encoder reading. Returns True when it is safe
        to capture a revolution at the current target angle."""
        if self.finished:
            return False

        within = abs(measured_phi - self.current_target) <= self.angle_tolerance_rad
        if not within:
            self._arrived_at = None
            return False

        if self._arrived_at is None:
            self._arrived_at = now
            return False

        return (now - self._arrived_at) >= self.settle_time_s

    def advance(self) -> None:
        """Call after a revolution has been successfully captured."""
        self._index += 1
        self._arrived_at = None
