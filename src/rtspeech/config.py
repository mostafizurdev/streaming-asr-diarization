"""Runtime settings, read from environment variables (12-factor style)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    # Models
    asr_model_dir: Path = Path("models/sherpa-onnx-streaming-zipformer-en-2023-06-26")
    asr_int8: bool = True
    speaker_model: Path = Path("models/nemo_en_titanet_small.onnx")
    num_threads: int = 2

    # Endpointing: a segment is closed after this much trailing silence (seconds).
    endpoint_silence: float = 0.8
    # Hard cap on a single segment's length (seconds).
    max_segment_seconds: float = 20.0

    # Online speaker clustering
    speaker_threshold: float = 0.65  # cosine similarity needed to join an existing speaker (tuned on dev set)
    max_speakers: int = 8
    min_embed_seconds: float = 1.0  # shorter segments inherit the previous speaker
    detect_speaker_changes: bool = True  # split segments at in-segment speaker changes
    change_threshold: float = 0.55  # window-to-window similarity below this marks a change (tuned on dev set)
    change_hop: float = 0.25  # seconds between change-detection windows (0.5 halves embedding work)

    # Serving
    max_batch: int = 16  # streams decoded together in one ONNX call
    max_wait_ms: float = 4.0  # how long the scheduler waits to fill a batch
    max_message_seconds: float = 2.0  # largest audio chunk accepted per WebSocket message

    @classmethod
    def from_env(cls) -> Settings:
        d = cls()
        return cls(
            asr_model_dir=Path(_env("ASR_MODEL_DIR", str(d.asr_model_dir))),
            asr_int8=_env("ASR_INT8", "1") not in ("0", "false", "False"),
            speaker_model=Path(_env("SPEAKER_MODEL", str(d.speaker_model))),
            num_threads=int(_env("NUM_THREADS", str(d.num_threads))),
            endpoint_silence=float(_env("ENDPOINT_SILENCE", str(d.endpoint_silence))),
            max_segment_seconds=float(_env("MAX_SEGMENT_SECONDS", str(d.max_segment_seconds))),
            speaker_threshold=float(_env("SPEAKER_THRESHOLD", str(d.speaker_threshold))),
            max_speakers=int(_env("MAX_SPEAKERS", str(d.max_speakers))),
            min_embed_seconds=float(_env("MIN_EMBED_SECONDS", str(d.min_embed_seconds))),
            detect_speaker_changes=_env("DETECT_SPEAKER_CHANGES", "1") not in ("0", "false", "False"),
            change_threshold=float(_env("CHANGE_THRESHOLD", str(d.change_threshold))),
            change_hop=float(_env("CHANGE_HOP", str(d.change_hop))),
            max_batch=int(_env("MAX_BATCH", str(d.max_batch))),
            max_wait_ms=float(_env("MAX_WAIT_MS", str(d.max_wait_ms))),
            max_message_seconds=float(_env("MAX_MESSAGE_SECONDS", str(d.max_message_seconds))),
        )
