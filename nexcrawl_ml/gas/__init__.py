"""Gas: rolling-window ML features + deterministic explosibility physics."""

from .physics import (
    CH4_LEL_PERCENT,
    CH4_UEL_PERCENT,
    ExplosibilityState,
    check_sensor_tamper,
    classify_atmosphere,
    coward_explosibility,
    grahams_ratio,
    interpret_grahams_ratio,
    time_to_lel,
)
from .windowing import (
    GasWindowFeatures,
    RollingGasBuffer,
    extract_window_features,
    windows_from_series,
)

__all__ = [
    "GasWindowFeatures",
    "extract_window_features",
    "RollingGasBuffer",
    "windows_from_series",
    "grahams_ratio",
    "interpret_grahams_ratio",
    "coward_explosibility",
    "ExplosibilityState",
    "time_to_lel",
    "classify_atmosphere",
    "check_sensor_tamper",
    "CH4_LEL_PERCENT",
    "CH4_UEL_PERCENT",
]
