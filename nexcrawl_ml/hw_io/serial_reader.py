"""
ESP32-S3 serial link: servo angle telemetry in, tilt commands out.

Wire protocol (line-oriented ASCII - deliberately simple so it can be
debugged with a serial monitor):

    ESP32 -> Pi :  A <timestamp_us> <angle_ticks> <moving_flag>
    Pi -> ESP32 :  T <target_ticks>       set tilt target
                   S                      request status
                   P                      sync ping

The STS3215 reports 12-bit absolute position (4096 ticks over 360 degrees),
which is where the 0.088 deg/tick resolution comes from.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

try:
    import serial

    _HAS_SERIAL = True
except ImportError:  # pragma: no cover
    _HAS_SERIAL = False

from ..perception.time_sync import AngleBuffer, ClockOffsetEstimator

STS3215_TICKS_PER_REV = 4096
STS3215_DEG_PER_TICK = 360.0 / STS3215_TICKS_PER_REV  # ~0.088 degrees


def ticks_to_rad(ticks: int) -> float:
    return float(np.deg2rad(ticks * STS3215_DEG_PER_TICK))


def rad_to_ticks(rad: float) -> int:
    return int(round(np.degrees(rad) / STS3215_DEG_PER_TICK))


@dataclass
class ServoStatus:
    angle_rad: float
    moving: bool
    timestamp: float


class ESP32ServoLink:
    """Threaded reader for the ESP32 angle stream."""

    def __init__(
        self,
        port: str = "/dev/ttyACM0",
        baudrate: int = 921600,
        buffer_size: int = 2000,
    ):
        if not _HAS_SERIAL:
            raise ImportError("pyserial is required: pip install pyserial")
        self.serial = serial.Serial(port, baudrate, timeout=0.1)
        self.angle_buffer = AngleBuffer(maxlen=buffer_size)
        self.clock = ClockOffsetEstimator()
        self._latest: ServoStatus | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        try:
            self.serial.close()
        except Exception:
            pass

    def _read_loop(self) -> None:
        while self._running:
            try:
                line = self.serial.readline().decode("ascii", errors="ignore").strip()
                if not line.startswith("A "):
                    continue
                parts = line.split()
                if len(parts) < 4:
                    continue

                t_esp = int(parts[1]) / 1e6
                angle = ticks_to_rad(int(parts[2]))
                moving = parts[3] == "1"
                t_pi = self.clock.to_pi_clock(t_esp)

                with self._lock:
                    self.angle_buffer.push(t_pi, angle)
                    self._latest = ServoStatus(angle, moving, t_pi)
            except Exception:
                time.sleep(0.01)

    @property
    def latest(self) -> ServoStatus | None:
        with self._lock:
            return self._latest

    def set_tilt(self, angle_rad: float) -> None:
        self.serial.write(f"T {rad_to_ticks(angle_rad)}\n".encode("ascii"))

    def sync_clock(self, n_pings: int = 20) -> float:
        """Round-trip clock offset estimation. Run once at startup."""
        for _ in range(n_pings):
            t_send = time.time()
            self.serial.write(b"P\n")
            line = self.serial.readline().decode("ascii", errors="ignore").strip()
            t_recv = time.time()
            if line.startswith("P "):
                try:
                    self.clock.add_round_trip(
                        t_send, int(line.split()[1]) / 1e6, t_recv
                    )
                except (ValueError, IndexError):
                    continue
        return self.clock.offset

    def wait_until_settled(
        self, target_rad: float, tolerance_rad: float = np.deg2rad(0.3),
        settle_time_s: float = 0.12, timeout_s: float = 3.0
    ) -> bool:
        """Block until the servo has physically arrived and held position."""
        deadline = time.time() + timeout_s
        arrived_at = None
        while time.time() < deadline:
            status = self.latest
            if status and abs(status.angle_rad - target_rad) <= tolerance_rad:
                if arrived_at is None:
                    arrived_at = time.time()
                elif time.time() - arrived_at >= settle_time_s:
                    return True
            else:
                arrived_at = None
            time.sleep(0.005)
        return False


class SimulatedServoLink:
    """Drop-in stand-in for ESP32ServoLink during offline development."""

    def __init__(self, noise_rad: float = np.deg2rad(0.05), seed: int = 0):
        self.angle_buffer = AngleBuffer()
        self.rng = np.random.default_rng(seed)
        self.noise_rad = noise_rad
        self._target = 0.0
        self._current = 0.0

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def set_tilt(self, angle_rad: float) -> None:
        self._target = angle_rad
        self._current = angle_rad + self.rng.normal(0, self.noise_rad)
        self.angle_buffer.push(time.time(), self._current)

    @property
    def latest(self) -> ServoStatus:
        return ServoStatus(self._current, False, time.time())

    def wait_until_settled(self, target_rad: float, **kwargs) -> bool:
        return True

    def sync_clock(self, n_pings: int = 20) -> float:
        return 0.0
