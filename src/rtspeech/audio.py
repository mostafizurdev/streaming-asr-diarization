"""Audio helpers: wire format conversion and resampling."""

from __future__ import annotations

from math import gcd

import numpy as np

SUPPORTED_SAMPLE_RATES = (8000, 16000, 22050, 24000, 32000, 44100, 48000)


def pcm16_to_float32(data: bytes) -> np.ndarray:
    """Little-endian signed 16-bit PCM bytes -> float32 samples in [-1, 1)."""
    if len(data) % 2:
        raise ValueError("PCM16 payload must have an even number of bytes")
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def float32_to_pcm16(samples: np.ndarray) -> bytes:
    """float32 samples -> little-endian signed 16-bit PCM bytes (clipped)."""
    clipped = np.clip(samples, -1.0, 32767 / 32768)
    return (clipped * 32768.0).astype("<i2").tobytes()


def resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Polyphase resampling for whole signals (not used on the streaming path)."""
    if src_rate == dst_rate:
        return samples.astype(np.float32, copy=False)
    from scipy.signal import resample_poly

    g = gcd(src_rate, dst_rate)
    return resample_poly(samples, dst_rate // g, src_rate // g).astype(np.float32)


class StreamResampler:
    """Stateful resampler for chunked input (libsoxr), so chunk edges leave no artifacts."""

    def __init__(self, src_rate: int, dst_rate: int = 16000):
        import soxr

        self.passthrough = src_rate == dst_rate
        self._rs = None if self.passthrough else soxr.ResampleStream(src_rate, dst_rate, 1, dtype="float32")

    def __call__(self, samples: np.ndarray, last: bool = False) -> np.ndarray:
        if self.passthrough:
            return samples.astype(np.float32, copy=False)
        return self._rs.resample_chunk(samples.astype(np.float32, copy=False), last=last)


def to_mono(samples: np.ndarray) -> np.ndarray:
    return samples if samples.ndim == 1 else samples.mean(axis=1).astype(np.float32)
