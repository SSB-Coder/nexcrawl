"""
Alert fusion: merge structural, gas and acoustic branches into one
prioritised operator view.

Design rule enforced here: a deterministic physics result always outranks a
model prediction. If the Coward triangle says the atmosphere is explosive,
that is a fact and goes top of the list regardless of what any classifier's
confidence score says.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum


class AlertPriority(IntEnum):
    INFO = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Alert:
    source: str          # 'structural' | 'gas' | 'acoustic' | 'physics'
    priority: AlertPriority
    label: str
    message: str
    confidence: float | None = None
    location: tuple[float, float, float] | None = None
    bearing_deg: float | None = None
    deterministic: bool = False
    timestamp: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "priority": int(self.priority),
            "priority_name": self.priority.name,
            "label": self.label,
            "message": self.message,
            "confidence": self.confidence,
            "location": list(self.location) if self.location else None,
            "bearing_deg": self.bearing_deg,
            "deterministic": self.deterministic,
            "timestamp": self.timestamp,
        }


SEVERITY_TO_PRIORITY = {
    4: AlertPriority.CRITICAL,
    3: AlertPriority.HIGH,
    2: AlertPriority.MEDIUM,
    1: AlertPriority.LOW,
    0: AlertPriority.INFO,
}


def structural_alerts(
    report, min_confidence: float = 0.5, ignore_artefacts: bool = True
) -> list[Alert]:
    alerts = []
    for h in report.hazards:
        if ignore_artefacts and h.label == "artefact":
            continue
        if h.confidence < min_confidence:
            continue
        alerts.append(
            Alert(
                source="structural",
                priority=SEVERITY_TO_PRIORITY.get(h.severity, AlertPriority.LOW),
                label=h.label,
                message=(
                    f"{h.label.replace('_', ' ')} detected "
                    f"({h.n_points} points, "
                    f"{h.features.get('volume_m3', 0):.2f} m3 bounding volume)"
                ),
                confidence=h.confidence,
                location=h.centroid,
            )
        )
    return alerts


def gas_alerts(
    atmosphere: dict, trend_prediction: dict | None = None,
    time_to_lel_s: float | None = None,
) -> list[Alert]:
    """Deterministic physics first, then the ML trend prediction."""
    alerts = []

    explos = atmosphere.get("explosibility", {})
    risk = explos.get("risk_label", "SAFE")
    if risk != "SAFE":
        priority = (
            AlertPriority.CRITICAL
            if risk in ("EXPLOSIVE", "INERT_BUT_UNSTABLE")
            else AlertPriority.HIGH
            if risk == "APPROACHING_LEL"
            else AlertPriority.MEDIUM
        )
        alerts.append(
            Alert(
                source="physics",
                priority=priority,
                label=risk,
                message=explos.get("action", ""),
                deterministic=True,
            )
        )

    gr_label = atmosphere.get("grahams_interpretation", "normal")
    if gr_label != "normal":
        alerts.append(
            Alert(
                source="physics",
                priority=AlertPriority.CRITICAL
                if gr_label == "open_fire_likely"
                else AlertPriority.HIGH,
                label=gr_label,
                message=(
                    f"Graham's Ratio {atmosphere.get('grahams_ratio', 0):.2f} "
                    f"indicates {gr_label.replace('_', ' ')}"
                ),
                deterministic=True,
            )
        )

    if atmosphere.get("o2_status") != "NORMAL":
        alerts.append(
            Alert(
                source="physics",
                priority=AlertPriority.CRITICAL,
                label=atmosphere["o2_status"],
                message="Oxygen outside safe range - breathing apparatus required",
                deterministic=True,
            )
        )

    tamper = atmosphere.get("sensor_tamper_status")
    if tamper == "WARNING: CH4_SENSOR_FAULT_OR_TAMPER":
        # Deliberately distinct from a genuine gas alert: this says "the
        # reading is not trustworthy", not "the air is dangerous". A CH4
        # spike with a static O2 reading is not physically consistent with
        # a real atmospheric event - treat the sensor as suspect, not the
        # air, until it is checked.
        alerts.append(
            Alert(
                source="physics",
                priority=AlertPriority.HIGH,
                label="SENSOR_TAMPER_SUSPECTED",
                message=(
                    "CH4 rose sharply while O2 stayed flat - physically "
                    "inconsistent with a real gas event. Check for a masked "
                    "or clogged intake before trusting this reading."
                ),
                deterministic=True,
            )
        )

    if trend_prediction and trend_prediction.get("label") not in (None, "SAFE"):
        label = trend_prediction["label"]
        msg = f"Gas trend classified as {label.replace('_', ' ').lower()}"
        if time_to_lel_s is not None and time_to_lel_s < 600:
            msg += f"; extrapolated ~{time_to_lel_s/60:.1f} min to LEL at current rate"
        alerts.append(
            Alert(
                source="gas",
                priority=SEVERITY_TO_PRIORITY.get(
                    trend_prediction.get("severity", 1), AlertPriority.MEDIUM
                ),
                label=label,
                message=msg,
                confidence=trend_prediction.get("confidence"),
            )
        )

    return alerts


def acoustic_alerts(
    knock_prediction: dict, doa_result=None, min_confidence: float = 0.6
) -> list[Alert]:
    if knock_prediction.get("label") != "knock":
        return []
    if knock_prediction.get("confidence", 0.0) < min_confidence:
        return []

    bearing = getattr(doa_result, "bearing_deg", None)
    msg = "Possible human knock detected"
    if bearing is not None:
        mirror = getattr(doa_result, "ambiguous_mirror_deg", None)
        msg += f" at bearing {bearing:+.0f} deg"
        if mirror is not None:
            msg += f" (or {mirror:+.0f} deg - two-mic front/back ambiguity)"

    return [
        Alert(
            source="acoustic",
            priority=AlertPriority.CRITICAL,
            label="knock_detected",
            message=msg,
            confidence=knock_prediction.get("confidence"),
            bearing_deg=bearing,
        )
    ]


def fuse_alerts(*alert_lists: list[Alert]) -> list[Alert]:
    """Merge and sort. Deterministic physics alerts win ties against models."""
    merged: list[Alert] = []
    for lst in alert_lists:
        merged.extend(lst or [])
    merged.sort(
        key=lambda a: (int(a.priority), a.deterministic, a.confidence or 0.0),
        reverse=True,
    )
    return merged
