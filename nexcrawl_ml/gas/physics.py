"""
Deterministic gas physics. No ML, no training, no tuning - these are fixed
relationships from mine ventilation engineering.

Kept strictly separate from the ML classifier on purpose: a hard physical
explosibility check should never be subject to a model's confidence score.
The ML layer predicts *trend*; this layer states *fact*.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Methane explosive range in air, percent by volume.
CH4_LEL_PERCENT = 5.0     # lower explosive limit
CH4_UEL_PERCENT = 15.0    # upper explosive limit
CH4_MOST_EXPLOSIVE = 9.5

# Statutory-style action thresholds commonly used underground (% by volume).
CH4_ACTION_THRESHOLDS = {
    "normal": 0.25,
    "caution": 1.0,
    "withdraw": 1.25,
    "danger": 2.0,
}

CO_PPM_THRESHOLDS = {"normal": 25, "caution": 50, "danger": 100, "critical": 400}
O2_PERCENT_THRESHOLDS = {"critical_low": 19.5, "normal": 20.9, "critical_high": 23.5}


@dataclass
class ExplosibilityState:
    ch4_percent: float
    in_explosive_range: bool
    distance_to_lel: float
    risk_label: str
    action: str


def grahams_ratio(
    co_ppm: float,
    o2_percent: float,
    ch4_percent: float = 0.0,
    co2_ppm: float = 400.0,
) -> float:
    """Graham's Ratio via the standard mining-engineering nitrogen balance.

        N2% = 100 - (CH4% + CO2% + CO% + O2%)          (by difference)
        R = CO% / (0.265 * N2% - O2%) * 100

    This replaces a simplified `100 * CO / (20.9 - O2)` approximation. The
    simplified form assumes ambient air is the only source of the oxygen
    deficit; the nitrogen-balance form instead computes how much oxygen
    *should* accompany the measured nitrogen if the air were unvitiated
    (0.265 = N2:O2 ratio in normal air, ~79/20.9*20.9... i.e. the fraction of
    N2 that corresponds to one part O2), and compares that against what is
    actually present. This matters most in afterdamp - air already depleted
    of oxygen from a prior fire or explosion - where CH4 and CO2 measurably
    change the nitrogen-by-difference term and the simplified formula
    quietly drifts from the DGMS-standard value.

    N2% is floored at 70% to avoid a division blow-up on physically
    implausible sensor combinations (e.g. CH4+CO2+CO+O2 summing above 30%,
    which should trip check_sensor_tamper rather than distort this ratio).

    Interpretation (industry convention, see interpret_grahams_ratio):
        < 0.4   normal
        0.4-1.0 heating suspected
        1.0-2.0 active heating
        > 2.0   open fire likely
    """
    co_pct = co_ppm / 10000.0
    co2_pct = co2_ppm / 10000.0

    n2_pct = max(100.0 - (ch4_percent + co2_pct + co_pct + o2_percent), 70.0)
    o2_consumed = (0.265 * n2_pct) - o2_percent

    if o2_consumed <= 0.05:
        return 0.0
    return float(round((co_pct / o2_consumed) * 100.0, 2))


def interpret_grahams_ratio(gr: float) -> str:
    if gr < 0.4:
        return "normal"
    if gr < 1.0:
        return "heating_suspected"
    if gr < 2.0:
        return "active_heating"
    return "open_fire_likely"


def coward_explosibility(
    ch4_percent: float, o2_percent: float
) -> ExplosibilityState:
    """Coward explosibility triangle check for a methane/air mixture.

    A mixture is only explosive when methane is between LEL and UEL AND
    there is enough oxygen to sustain combustion (~12% is the usual nose
    limit). Below that oxygen level a high-methane mixture is inert - but
    becomes explosive the moment fresh air reaches it, which is exactly why
    a 'safe' reading in a sealed area is not actually reassuring.
    """
    enough_oxygen = o2_percent >= 12.0
    in_range = CH4_LEL_PERCENT <= ch4_percent <= CH4_UEL_PERCENT
    explosive = in_range and enough_oxygen

    if explosive:
        risk, action = "EXPLOSIVE", "EVACUATE IMMEDIATELY - do not operate equipment"
    elif in_range and not enough_oxygen:
        risk = "INERT_BUT_UNSTABLE"
        action = "Fuel-rich and oxygen-starved: explosive if fresh air enters"
    elif ch4_percent >= CH4_ACTION_THRESHOLDS["withdraw"]:
        risk, action = "APPROACHING_LEL", "Withdraw personnel, increase ventilation"
    elif ch4_percent >= CH4_ACTION_THRESHOLDS["caution"]:
        risk, action = "ELEVATED", "Increase ventilation, monitor closely"
    else:
        risk, action = "SAFE", "Continue normal operation"

    return ExplosibilityState(
        ch4_percent=float(ch4_percent),
        in_explosive_range=explosive,
        distance_to_lel=float(CH4_LEL_PERCENT - ch4_percent),
        risk_label=risk,
        action=action,
    )


def time_to_lel(
    current_ch4_percent: float, rate_of_rise_per_s: float
) -> float | None:
    """Extrapolate seconds until the lower explosive limit is reached.

    Returns None when not rising. This is a linear extrapolation and should
    be presented as an estimate, not a countdown - real gas accumulation is
    rarely linear.
    """
    if rate_of_rise_per_s <= 1e-9:
        return None
    remaining = CH4_LEL_PERCENT - current_ch4_percent
    return float(remaining / rate_of_rise_per_s) if remaining > 0 else 0.0


def check_sensor_tamper(
    ch4_curr: float,
    ch4_prev: float,
    o2_curr: float,
    o2_prev: float,
    ch4_jump_threshold: float = 1.2,
    o2_stability_threshold: float = 0.05,
) -> str:
    """Flag a physically implausible reading: methane rising sharply while
    oxygen stays essentially flat.

    In a real combustion or displacement event, CH4 rising means something is
    physically displacing air, so O2 should measurably drop in step. A large
    CH4 jump with a nearly-static O2 reading is not a real atmospheric event -
    it is the signature of a masked/clogged intake (dust, a rag, condensation)
    or a sensor fault, called out explicitly in the project's safety
    requirements as a case to catch rather than alert on as a genuine gas
    event.
    """
    ch4_delta = ch4_curr - ch4_prev
    o2_delta = abs(o2_curr - o2_prev)
    if ch4_delta > ch4_jump_threshold and o2_delta < o2_stability_threshold:
        return "WARNING: CH4_SENSOR_FAULT_OR_TAMPER"
    return "NORMAL"


def classify_atmosphere(
    ch4_percent: float,
    co_ppm: float,
    o2_percent: float,
    co2_ppm: float = 400.0,
    prior_ch4_percent: float | None = None,
    prior_o2_percent: float | None = None,
) -> dict:
    """Combined deterministic assessment across all three gases.

    prior_ch4_percent / prior_o2_percent are optional: pass the previous
    reading to enable the cross-sensor tamper check. Without them, tamper
    status is reported as UNKNOWN rather than silently skipped, so a caller
    can tell the difference between "checked, looks fine" and "not checked".
    """
    explosibility = coward_explosibility(ch4_percent, o2_percent)
    gr = grahams_ratio(co_ppm, o2_percent, ch4_percent=ch4_percent, co2_ppm=co2_ppm)

    if prior_ch4_percent is not None and prior_o2_percent is not None:
        tamper_status = check_sensor_tamper(
            ch4_percent, prior_ch4_percent, o2_percent, prior_o2_percent
        )
    else:
        tamper_status = "UNKNOWN"

    if o2_percent < O2_PERCENT_THRESHOLDS["critical_low"]:
        o2_status = "OXYGEN_DEFICIENT"
    elif o2_percent > O2_PERCENT_THRESHOLDS["critical_high"]:
        o2_status = "OXYGEN_ENRICHED"
    else:
        o2_status = "NORMAL"

    if co_ppm >= CO_PPM_THRESHOLDS["critical"]:
        co_status = "CRITICAL"
    elif co_ppm >= CO_PPM_THRESHOLDS["danger"]:
        co_status = "DANGER"
    elif co_ppm >= CO_PPM_THRESHOLDS["caution"]:
        co_status = "CAUTION"
    else:
        co_status = "NORMAL"

    return {
        "explosibility": explosibility.__dict__,
        "grahams_ratio": gr,
        "grahams_interpretation": interpret_grahams_ratio(gr),
        "sensor_tamper_status": tamper_status,
        "o2_status": o2_status,
        "co_status": co_status,
        "overall_safe": (
            explosibility.risk_label == "SAFE"
            and o2_status == "NORMAL"
            and co_status == "NORMAL"
            and tamper_status != "WARNING: CH4_SENSOR_FAULT_OR_TAMPER"
        ),
    }
