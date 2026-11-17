"""Evaluation helpers: text normalisation, WER counts, diarization error rate, in-process runs."""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from rtspeech.asr import StreamingAsr
from rtspeech.config import Settings
from rtspeech.scheduler import DecodeScheduler
from rtspeech.session import StreamingSession
from rtspeech.speaker import Embedder, OnlineSpeakerClustering


def normalize(text: str) -> str:
    text = text.upper().replace("-", " ")
    text = re.sub(r"[^A-Z' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class WordCounts:
    ref_words: int = 0
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0

    def add(self, ref: str, hyp: str) -> None:
        import jiwer

        out = jiwer.process_words(normalize(ref) or "<empty>", normalize(hyp) or "<empty>")
        self.ref_words += len(normalize(ref).split())
        self.substitutions += out.substitutions
        self.deletions += out.deletions
        self.insertions += out.insertions

    @property
    def wer(self) -> float:
        return (self.substitutions + self.deletions + self.insertions) / max(self.ref_words, 1)


def _frames(segments: list[tuple[str, float, float]], labels: list[str], n: int, step: float) -> np.ndarray:
    """One-hot speaker activity matrix [frames, speakers]."""
    m = np.zeros((n, len(labels)), dtype=bool)
    for spk, s, e in segments:
        m[int(round(s / step)) : int(round(e / step)), labels.index(spk)] = True
    return m


def diarization_error(
    ref: list[tuple[str, float, float]],
    hyp: list[tuple[str, float, float]],
    collar: float = 0.0,
    step: float = 0.01,
) -> dict:
    """Frame-level DER with optimal one-to-one speaker mapping (no overlap in reference).

    `collar` removes +/- collar seconds around every reference boundary from scoring,
    as in the standard NIST md-eval convention.
    """
    end = max([e for _, _, e in ref + hyp] + [0.0])
    n = int(np.ceil(end / step)) + 1
    rl, hl = sorted({s for s, _, _ in ref}), sorted({s for s, _, _ in hyp})
    R = _frames(ref, rl, n, step)
    H = _frames(hyp, hl, n, step) if hl else np.zeros((n, 0), bool)

    scored = np.ones(n, dtype=bool)
    if collar > 0:
        for _, s, e in ref:
            for b in (s, e):
                scored[max(0, int((b - collar) / step)) : int((b + collar) / step)] = False
    R, H = R[scored], H[scored]

    ref_speech = R.any(axis=1)
    hyp_speech = H.any(axis=1) if H.shape[1] else np.zeros(len(R), bool)
    total = int(ref_speech.sum())
    missed = int((ref_speech & ~hyp_speech).sum())
    false_alarm = int((~ref_speech & hyp_speech).sum())

    confusion = 0
    if H.shape[1]:
        overlap = R.T.astype(np.int64) @ H.astype(np.int64)  # [ref_spk, hyp_spk]
        r_idx, h_idx = linear_sum_assignment(-overlap)
        mapping = dict(zip(h_idx, r_idx, strict=True))
        both = ref_speech & hyp_speech
        ref_spk = R.argmax(axis=1)
        hyp_spk = H.argmax(axis=1)
        mapped = np.array([mapping.get(h, -1) for h in hyp_spk])
        confusion = int((both & (mapped != ref_spk)).sum())

    return {
        "der": (missed + false_alarm + confusion) / max(total, 1),
        "missed": missed / max(total, 1),
        "false_alarm": false_alarm / max(total, 1),
        "confusion": confusion / max(total, 1),
        "ref_speakers": len(rl),
        "hyp_speakers": len(hl),
    }


async def run_in_process(
    audio: np.ndarray,
    asr: StreamingAsr,
    embedder: Embedder | None,
    settings: Settings,
    chunk_ms: int = 100,
) -> tuple[list[dict], float]:
    """Stream `audio` (16 kHz) through a session as fast as possible; return finals and compute time."""
    scheduler = DecodeScheduler(asr, settings.max_batch, max_wait_ms=0.0)
    await scheduler.start()
    session = StreamingSession(
        asr,
        scheduler,
        OnlineSpeakerClustering(settings.speaker_threshold, settings.max_speakers),
        embedder=embedder,
        min_embed_seconds=settings.min_embed_seconds,
        detect_changes=settings.detect_speaker_changes,
        change_threshold=settings.change_threshold,
        change_hop=settings.change_hop,
    )
    step = 16000 * chunk_ms // 1000
    finals: list[dict] = []
    t0 = time.perf_counter()
    for i in range(0, audio.size, step):
        finals += [e for e in await session.feed(audio[i : i + step]) if e["type"] == "final"]
    finals += await session.finish()
    elapsed = time.perf_counter() - t0
    await scheduler.stop()
    return finals, elapsed


def run_sync(*args, **kwargs):
    return asyncio.run(run_in_process(*args, **kwargs))
