"""Streaming ASR backend: a thin wrapper around sherpa-onnx's streaming Zipformer transducer.

The wrapper keeps the rest of the code independent of sherpa-onnx, so the session logic,
scheduler and server can be unit-tested with a fake backend and no model files.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .config import Settings


@dataclass
class AsrResult:
    text: str
    # BPE tokens (a leading space starts a new word) and their times in seconds
    # from the start of this stream.
    tokens: list[str] = field(default_factory=list)
    token_times: list[float] = field(default_factory=list)


class StreamingAsr(Protocol):
    """What the session and scheduler need from an ASR backend."""

    def create_stream(self) -> Any: ...
    def accept(self, stream: Any, sample_rate: int, samples: np.ndarray) -> None: ...
    def decode_batch(self, streams: Sequence[Any]) -> None: ...
    def result(self, stream: Any) -> AsrResult: ...
    def is_endpoint(self, stream: Any) -> bool: ...
    def input_finished(self, stream: Any) -> None: ...


def _pick(model_dir: Path, prefix: str, int8: bool) -> str:
    candidates = sorted(model_dir.glob(f"{prefix}-*.onnx"))
    if int8:
        chosen = [c for c in candidates if c.name.endswith(".int8.onnx")]
    else:
        chosen = [c for c in candidates if not c.name.endswith(".int8.onnx")]
    if not chosen:
        raise FileNotFoundError(f"No {prefix} model ({'int8' if int8 else 'fp32'}) in {model_dir}")
    return str(chosen[0])


class SherpaStreamingAsr:
    """Streaming Zipformer transducer (greedy search) with rule-based endpointing."""

    def __init__(self, settings: Settings):
        import sherpa_onnx  # imported lazily so tests run without the native package

        d = settings.asr_model_dir
        self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(d / "tokens.txt"),
            encoder=_pick(d, "encoder", settings.asr_int8),
            decoder=_pick(d, "decoder", settings.asr_int8),
            joiner=_pick(d, "joiner", settings.asr_int8),
            num_threads=settings.num_threads,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
            enable_endpoint_detection=True,
            # rule1: no speech yet, long silence; rule2: speech then `endpoint_silence`
            # of trailing silence; rule3: force a cut on very long segments.
            rule1_min_trailing_silence=2.4,
            rule2_min_trailing_silence=settings.endpoint_silence,
            rule3_min_utterance_length=settings.max_segment_seconds,
        )

    def create_stream(self):
        return self.recognizer.create_stream()

    def accept(self, stream, sample_rate: int, samples: np.ndarray) -> None:
        # sherpa-onnx resamples internally when sample_rate != 16 kHz.
        stream.accept_waveform(sample_rate, samples)

    def decode_batch(self, streams) -> None:
        rec = self.recognizer
        pending = list(streams)
        while pending:
            ready = [s for s in pending if rec.is_ready(s)]
            if not ready:
                return
            rec.decode_streams(ready)
            pending = ready

    def result(self, stream) -> AsrResult:
        r = self.recognizer.get_result_all(stream)
        times = [r.start_time + t for t in r.timestamps]
        return AsrResult(text=r.text.strip(), tokens=list(r.tokens), token_times=times)

    def is_endpoint(self, stream) -> bool:
        return self.recognizer.is_endpoint(stream)

    def input_finished(self, stream) -> None:
        stream.input_finished()
