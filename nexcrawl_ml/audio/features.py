"""
Audio feature extraction for KnockNet.

Log-mel spectrograms rather than raw waveform: with ~400 training clips
there is nowhere near enough data to learn a filterbank from scratch, and
mel scaling encodes the perceptual frequency resolution that separates a
sharp metallic knock from broadband motor noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import librosa

    _HAS_LIBROSA = True
except ImportError:  # pragma: no cover
    _HAS_LIBROSA = False


@dataclass
class AudioConfig:
    sample_rate: int = 22050
    n_mels: int = 64
    n_fft: int = 1024
    hop_length: int = 256
    clip_duration_s: float = 1.0
    fmin: float = 50.0
    fmax: float = 8000.0

    @property
    def n_frames(self) -> int:
        return int(np.ceil(self.sample_rate * self.clip_duration_s / self.hop_length))


def load_audio(path: str, config: AudioConfig | None = None) -> np.ndarray:
    if not _HAS_LIBROSA:
        raise ImportError("librosa is required: pip install librosa")
    config = config or AudioConfig()
    y, _ = librosa.load(path, sr=config.sample_rate, mono=True)
    return y


def compute_log_mel(
    waveform: np.ndarray, config: AudioConfig | None = None, normalize: bool = True
) -> np.ndarray:
    """Waveform -> (n_mels, n_frames) log-mel spectrogram.

    Per-clip normalisation is on by default so the model keys on spectral
    *shape* rather than absolute loudness - a distant knock and a close
    knock should land in the same class.
    """
    if not _HAS_LIBROSA:
        raise ImportError("librosa is required: pip install librosa")

    config = config or AudioConfig()
    target_len = int(config.sample_rate * config.clip_duration_s)

    y = np.asarray(waveform, dtype=np.float32)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        fmin=config.fmin,
        fmax=config.fmax,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)

    if normalize:
        std = log_mel.std()
        log_mel = (log_mel - log_mel.mean()) / (std if std > 1e-6 else 1.0)

    return log_mel.astype(np.float32)


def augment_waveform(
    waveform: np.ndarray,
    config: AudioConfig | None = None,
    noise_level: float = 0.005,
    time_shift_max: float = 0.2,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Light augmentation to stretch a small training set.

    Deliberately conservative: pitch-shifting a knock too far turns it into
    something that is no longer a knock, which poisons the labels.
    """
    config = config or AudioConfig()
    rng = rng or np.random.default_rng()
    y = np.asarray(waveform, dtype=np.float32).copy()

    shift = int(rng.uniform(-time_shift_max, time_shift_max) * len(y))
    y = np.roll(y, shift)

    if noise_level > 0:
        y = y + rng.normal(0, noise_level, len(y)).astype(np.float32)

    gain = float(rng.uniform(0.8, 1.2))
    return y * gain


def spec_augment(
    mel: np.ndarray,
    n_freq_masks: int = 1,
    n_time_masks: int = 1,
    freq_mask_width: int = 8,
    time_mask_width: int = 10,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """SpecAugment-style masking applied on the spectrogram directly."""
    rng = rng or np.random.default_rng()
    out = np.asarray(mel, dtype=np.float32).copy()
    n_mels, n_frames = out.shape

    for _ in range(n_freq_masks):
        w = int(rng.integers(0, freq_mask_width + 1))
        if w > 0 and n_mels > w:
            f0 = int(rng.integers(0, n_mels - w))
            out[f0 : f0 + w, :] = 0.0

    for _ in range(n_time_masks):
        w = int(rng.integers(0, time_mask_width + 1))
        if w > 0 and n_frames > w:
            t0 = int(rng.integers(0, n_frames - w))
            out[:, t0 : t0 + w] = 0.0

    return out
