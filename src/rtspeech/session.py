"""One streaming session: audio in, partial and final (speaker-labelled) transcripts out."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any

import numpy as np

from .asr import AsrResult, StreamingAsr
from .audio import StreamResampler
from .scheduler import DecodeScheduler
from .segmenter import SubSegment, Word, speaker_pass, words_from_tokens
from .speaker import Embedder, OnlineSpeakerClustering

# Runs a blocking function off the event loop (the server passes a thread-pool runner).
RunBlocking = Callable[..., Awaitable[Any]]
# Receives final events as soon as they are ready (the server sends them to the client).
OnFinal = Callable[[list[dict]], Awaitable[None]]

log = logging.getLogger("rtspeech.session")

# Silence appended at end of input so the transducer flushes its right context.
_FLUSH_PAD_S = 0.6
# Audio replayed into the next stream after a silence-only endpoint.
_CARRY_SILENT_S = 0.5
# Time kept after the last token of an utterance (end of the last word).
_TAIL_S = 0.35


def display_text(raw: str) -> str:
    """Model output is upper-case BPE text; make it readable."""
    text = raw.strip().lower()
    text = re.sub(r"\bi\b", "I", text)
    text = re.sub(r"\bi'(m|ll|ve|d)\b", lambda m: "I'" + m.group(1), text)
    return text[:1].upper() + text[1:]


async def _run_inline(fn, *args):
    return fn(*args)


class StreamingSession:
    """Owns one client's ASR stream, endpointing, and the ordered speaker-labelling queue.

    Events:
      partial  {"utterance", "text", "audio_end"}       while someone is speaking
      final    {"segment", "utterance", "text", "start", "end", "speaker", ...}
               one or more per utterance, after speaker-change splitting
    """

    def __init__(
        self,
        asr: StreamingAsr,
        scheduler: DecodeScheduler,
        clustering: OnlineSpeakerClustering,
        embedder: Embedder | None = None,
        run_blocking: RunBlocking = _run_inline,
        input_sample_rate: int = 16000,
        min_embed_seconds: float = 1.0,
        detect_changes: bool = True,
        change_threshold: float = 0.55,
        change_hop: float = 0.25,
        on_final: OnFinal | None = None,
    ):
        self.asr = asr
        self.scheduler = scheduler
        self.clustering = clustering
        self.embedder = embedder
        self.run_blocking = run_blocking
        # Everything inside the session runs at 16 kHz; other client rates are converted
        # once, with a streaming resampler, on the way in.
        self.sample_rate = 16000
        self._resample = StreamResampler(input_sample_rate, self.sample_rate)
        self.min_embed_seconds = min_embed_seconds
        self.detect_changes = detect_changes
        self.change_threshold = change_threshold
        self.change_hop = change_hop
        self.on_final = on_final

        self.stream = asr.create_stream()
        self.samples_in = 0
        self.utterance_index = 0  # endpoint-delimited stretch of speech
        self.segment_index = 0  # speaker-homogeneous final segment
        self.segments: list[dict] = []
        self._seg_chunks: list[np.ndarray] = []
        self._seg_offset = 0  # session sample index where the buffered audio starts
        self._stream_origin = 0.0  # session time (s) at which the current stream starts
        self._last_partial = ""
        self._finished = False
        self._lock = asyncio.Lock()
        self._outbox: list[dict] = []  # finals not yet returned (used when on_final is None)
        self._label_task: asyncio.Task | None = None

    @property
    def audio_seconds(self) -> float:
        return self.samples_in / self.sample_rate

    async def feed(self, samples: np.ndarray) -> list[dict]:
        """Push one chunk of audio; returns new partials and any finals that are ready."""
        async with self._lock:
            if self._finished:
                raise RuntimeError("session already finished")
            samples = self._resample(samples)
            if samples.size == 0:
                return self._drain()
            self.asr.accept(self.stream, self.sample_rate, samples)
            self._seg_chunks.append(samples)
            self.samples_in += samples.size
            await self.scheduler.decode(self.stream)

            events = self._drain()
            res = self._result()
            if res.text and res.text != self._last_partial:
                self._last_partial = res.text
                events.append(
                    {
                        "type": "partial",
                        "utterance": self.utterance_index,
                        "text": display_text(res.text),
                        "audio_end": round(self.audio_seconds, 3),
                    }
                )
            if self.asr.is_endpoint(self.stream):
                await self._close_utterance(res)
            return events

    async def finish(self) -> list[dict]:
        """Flush the model, label the last utterance and return the remaining finals."""
        async with self._lock:
            if self._finished:
                return []
            self._finished = True
            tail = self._resample(np.zeros(0, np.float32), last=True)
            if tail.size:
                self.asr.accept(self.stream, self.sample_rate, tail)
                self._seg_chunks.append(tail)
                self.samples_in += tail.size
            pad = np.zeros(int(_FLUSH_PAD_S * self.sample_rate), dtype=np.float32)
            self.asr.accept(self.stream, self.sample_rate, pad)
            self.asr.input_finished(self.stream)
            await self.scheduler.decode(self.stream)
            await self._close_utterance(self._result(), carry_over=False)
            if self._label_task is not None:
                await self._label_task
            return self._drain()

    def _drain(self) -> list[dict]:
        out, self._outbox = self._outbox, []
        return out

    def _result(self) -> AsrResult:
        res = self.asr.result(self.stream)
        res.token_times = [self._stream_origin + t for t in res.token_times]
        return res

    async def _close_utterance(self, res: AsrResult, carry_over: bool = True) -> None:
        """Queue the finished utterance for speaker labelling and open a fresh stream.

        Labelling (change detection + embeddings) runs as a background task, chained so
        finals stay in order, so audio keeps flowing and partials keep coming while the
        previous utterance is being labelled.

        The endpoint fires `endpoint_silence` after the last *emitted* token, but the
        transducer emits tokens with some delay, so the next utterance may already have
        started inside that window. Instead of `reset()`-ing the stream (which in our
        tests dropped the words of an utterance that was already under way), we open a
        new stream and replay the audio after the last token into it ("carry-over").
        """
        sr = self.sample_rate
        audio = np.concatenate(self._seg_chunks) if self._seg_chunks else np.zeros(0, np.float32)
        seg_start = self._seg_offset / sr
        seg_end = self.samples_in / sr

        if res.text:
            words = words_from_tokens(res.tokens, res.token_times)
            self._label_task = asyncio.create_task(
                self._label(self._label_task, audio, seg_start, seg_end, words, res.text, self.utterance_index)
            )
            self.utterance_index += 1
            carry_from = min(seg_end, res.token_times[-1] + _TAIL_S) if res.token_times else seg_end
        else:
            # Silence-only endpoint: keep a short tail in case speech starts right at the cut.
            carry_from = max(seg_start, seg_end - _CARRY_SILENT_S)

        self._last_partial = ""
        if not carry_over:
            return

        cut = min(audio.size, max(0, int(round(carry_from * sr)) - self._seg_offset))
        carry = audio[cut:]
        self.stream = self.asr.create_stream()
        self._seg_chunks = [carry] if carry.size else []
        self._seg_offset = self.samples_in - carry.size
        self._stream_origin = self._seg_offset / sr
        if carry.size:
            self.asr.accept(self.stream, sr, carry)
            await self.scheduler.decode(self.stream)

    async def _label(
        self,
        previous: asyncio.Task | None,
        audio: np.ndarray,
        seg_start: float,
        seg_end: float,
        words: list[Word],
        raw_text: str,
        utterance: int,
    ) -> None:
        """Speaker pass for one utterance; emits its final segment(s) after earlier ones."""
        if previous is not None:
            await previous
        try:
            subs: list[SubSegment] = await self.run_blocking(
                speaker_pass,
                self.embedder,
                audio,
                self.sample_rate,
                seg_start,
                words,
                seg_start,
                seg_end,
                self.min_embed_seconds,
                self.detect_changes,
                self.change_threshold,
                self.change_hop,
            )
        except Exception:  # never lose the transcript because labelling failed
            log.exception("speaker pass failed")
            subs = [SubSegment(words, seg_start, seg_end)]

        events = []
        for sub in subs:
            who = self.clustering.assign(sub.embedding, weight=max(sub.end - sub.start, 1e-3))
            event = {
                "type": "final",
                "segment": self.segment_index,
                "utterance": utterance,
                "text": display_text(sub.text if sub.words else raw_text),
                "start": round(sub.start, 2),
                "end": round(sub.end, 2),
                "speaker": who.label,
                "speaker_score": None if who.score is None else round(who.score, 3),
                "new_speaker": who.is_new,
            }
            self.segments.append(event)
            events.append(event)
            self.segment_index += 1

        if self.on_final is not None:
            try:
                await self.on_final(events)
            except Exception:  # the client may already be gone
                log.debug("could not deliver final events", exc_info=True)
        else:
            self._outbox.extend(events)
