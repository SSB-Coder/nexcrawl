"""Audio: mel features for KnockNet + GCC-PHAT bearing estimation."""

from .doa import DOAResult, estimate_bearing, gcc_phat
from .features import (
    AudioConfig,
    augment_waveform,
    compute_log_mel,
    load_audio,
    spec_augment,
)

__all__ = [
    "AudioConfig",
    "compute_log_mel",
    "load_audio",
    "augment_waveform",
    "spec_augment",
    "gcc_phat",
    "estimate_bearing",
    "DOAResult",
]
