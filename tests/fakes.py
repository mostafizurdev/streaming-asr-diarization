"""Model-free stand-ins for the ASR backend and speaker embedder, used by the unit tests.

FakeAsr "recognises" one token per 0.3 s of non-silent audio and signals an endpoint after
0.8 s of trailing silence, which is enough to exercise segmentation, carry-over and
batching without downloading models. FakeEmbedder turns a pure tone into a vector that
depends on its frequency, so two tones behave like two different speakers.
"""

from __future__ import annotations

import numpy as np

from rtspeech.asr import AsrResult

SR = 16000
FRAME = 160  # 10 ms


def tone(freq: float, seconds: float, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), np.float32)


class FakeStream:
    def __init__(self) -> None:
        self.chunks: list[np.ndarray] = []
        self.finished = False

    @property
    def audio(self) -> np.ndarray:
        return np.concatenate(self.chunks) if self.chunks else np.zeros(0, np.float32)


class FakeAsr:
    def __init__(self, endpoint_silence: float = 0.8) -> None:
        self.endpoint_silence = endpoint_silence
        self.batch_sizes: list[int] = []

    def create_stream(self) -> FakeStream:
        return FakeStream()

    def accept(self, stream: FakeStream, sample_rate: int, samples: np.ndarray) -> None:
        assert sample_rate == SR
        stream.chunks.append(np.asarray(samples, np.float32))

    def input_finished(self, stream: FakeStream) -> None:
        stream.finished = True

    def decode_batch(self, streams) -> None:
        self.batch_sizes.append(len(streams))

    def _voiced(self, stream: FakeStream) -> np.ndarray:
        x = stream.audio
        n = x.size // FRAME
        return np.abs(x[: n * FRAME].reshape(n, FRAME)).max(axis=1) > 0.01

    def result(self, stream: FakeStream) -> AsrResult:
        voiced = self._voiced(stream)
        tokens, times, run = [], [], 0
        for i, v in enumerate(voiced):
            run = run + 1 if v else 0
            if v and run % 30 == 1:  # a token every 0.3 s of speech
                tokens.append(f" W{len(tokens) + 1}")
                times.append(i * FRAME / SR + 0.2)  # emitted with ~0.2 s lag, like the real model
        return AsrResult(text=" ".join(t.strip() for t in tokens), tokens=tokens, token_times=times)

    def is_endpoint(self, stream: FakeStream) -> bool:
        voiced = self._voiced(stream)
        if not voiced.any():
            return voiced.size * FRAME / SR >= 2.4
        trailing = voiced.size - 1 - int(np.where(voiced)[0][-1])
        return trailing * FRAME / SR >= self.endpoint_silence


class FakeEmbedder:
    """Unit vector of the energy in 8 frequency bands: different tones -> different 'voices'."""

    EDGES = np.arange(0, 1300, 150)  # 8 bands of 150 Hz

    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        spec = np.abs(np.fft.rfft(samples)) ** 2
        freqs = np.fft.rfftfreq(samples.size, 1 / sample_rate)
        bands = np.array(
            [spec[(freqs >= lo) & (freqs < hi)].sum() for lo, hi in zip(self.EDGES[:-1], self.EDGES[1:], strict=True)],
            dtype=np.float32,
        )
        bands = np.sqrt(bands)
        return bands / (np.linalg.norm(bands) + 1e-9)
