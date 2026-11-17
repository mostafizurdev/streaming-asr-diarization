"""Speaker-change detection inside an ASR segment.

Endpointing only cuts a segment after `endpoint_silence` seconds of silence, so in a quick
exchange ("Sounds good." / "Okay, I'll do it.") two speakers can land in one segment.
This module looks for speaker changes inside a closed segment:

1. Embed sliding windows (1.5 s long, 0.25 s hop) with the speaker model.
2. At each candidate boundary t (every 0.25 s), compare the window just before t with the window just
   after it. A local minimum of cosine similarity below `threshold` is a change point.
3. Snap each change point to the longest pause between words within +/- `max_snap` s,
   so text is never split in the middle of a word.

Everything here is synchronous and CPU-bound; the server runs it on a worker thread.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .speaker import Embedder


@dataclass
class Word:
    text: str
    start: float  # time of the first sub-word token (s, session time)
    last: float  # time of the last sub-word token


@dataclass
class SubSegment:
    words: list[Word]
    start: float
    end: float
    embedding: np.ndarray | None = None
    change_scores: list[float] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


# Streaming transducers emit a token a little after the sound it stands for. Measured on
# our data, token times trail word onsets by about 0.2 s, so we shift them back.
TOKEN_LAG_S = 0.2


def words_from_tokens(tokens: list[str], times: list[float], lag: float = TOKEN_LAG_S) -> list[Word]:
    """BPE tokens with a leading space start a new word (sherpa-onnx convention)."""
    words: list[Word] = []
    for tok, t in zip(tokens, times, strict=False):
        t = max(0.0, t - lag)
        if tok.startswith(" ") or not words:
            words.append(Word(tok.strip(), t, t))
        else:
            words[-1].text += tok
            words[-1].last = t
    return [w for w in words if w.text]


def voiced_fraction(x: np.ndarray, sr: int, ref_energy: float) -> float:
    """Share of 20 ms frames whose energy is within 26 dB of the clip's loud frames."""
    n = int(0.02 * sr)
    frames = x[: x.size // n * n].reshape(-1, n)
    if frames.size == 0:
        return 0.0
    energy = (frames**2).mean(axis=1)
    return float((energy > max(ref_energy * 0.0025, 1e-7)).mean())


def change_points(
    speech: np.ndarray,
    sr: int,
    embedder: Embedder,
    win: float = 1.5,
    hop: float = 0.25,
    threshold: float = 0.55,
    min_gap: float | None = None,
    min_voiced: float = 0.5,
) -> list[tuple[float, float]]:
    """Return (time_in_clip_seconds, similarity) for detected speaker changes.

    Windows that are mostly silence are skipped: an embedding of silence is noise and
    would create false changes.
    """
    min_gap = win if min_gap is None else min_gap
    dur = speech.size / sr
    if dur < 2 * win:
        return []
    steps = int(round(win / hop))
    starts = np.arange(0.0, dur - win + 1e-9, hop)
    n = int(0.02 * sr)
    all_frames = speech[: speech.size // n * n].reshape(-1, n)
    ref_energy = float(np.percentile((all_frames**2).mean(axis=1), 95)) if all_frames.size else 0.0
    windows = [speech[int(s * sr) : int((s + win) * sr)] for s in starts]
    usable = [voiced_fraction(w, sr, ref_energy) >= min_voiced for w in windows]
    embs = [embedder.embed(w, sr) if ok else None for w, ok in zip(windows, usable, strict=True)]
    # Boundary at starts[i] compares window [starts[i]-win, starts[i]] with [starts[i], +win].
    sims = {
        i: float(embs[i - steps] @ embs[i])
        for i in range(steps, len(embs))
        if embs[i - steps] is not None and embs[i] is not None
    }
    candidates = [
        i
        for i, s in sims.items()
        if s < threshold and s <= sims.get(i - 1, 1.0) and s <= sims.get(i + 1, 1.0)
    ]
    chosen: list[int] = []
    for i in sorted(candidates, key=lambda k: sims[k]):  # strongest changes first
        t = starts[i]
        if t < min_gap or dur - t < min_gap:
            continue
        if all(abs(t - starts[j]) >= min_gap for j in chosen):
            chosen.append(i)
    return sorted((float(starts[i]), sims[i]) for i in chosen)


def split_words(words: list[Word], changes: list[float], max_snap: float = 0.6) -> list[list[Word]]:
    """Split the word list at the word gap closest to (and longest near) each change time."""
    if not changes or len(words) < 2:
        return [words] if words else []
    cut_after: set[int] = set()
    for c in changes:
        best, best_score = None, None
        for k in range(len(words) - 1):
            gap_start, gap_end = words[k].last, words[k + 1].start
            mid = 0.5 * (gap_start + gap_end)
            if abs(mid - c) > max_snap:
                continue
            score = (gap_end - gap_start) - 0.3 * abs(mid - c)  # prefer long pauses near c
            if best_score is None or score > best_score:
                best, best_score = k, score
        if best is not None:
            cut_after.add(best)
    groups, current = [], []
    for k, w in enumerate(words):
        current.append(w)
        if k in cut_after:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def speaker_pass(
    embedder: Embedder | None,
    speech: np.ndarray,
    sr: int,
    clip_start: float,
    words: list[Word],
    seg_start: float,
    seg_end: float,
    min_embed_seconds: float = 1.0,
    detect_changes: bool = True,
    change_threshold: float = 0.55,
    change_hop: float = 0.25,
    lead: float = 0.10,
    tail: float = 0.45,
) -> list[SubSegment]:
    """Split a closed segment by speaker and embed each part. `speech` starts at `clip_start`."""
    changes: list[tuple[float, float]] = []
    if embedder is not None and detect_changes and words:
        changes = change_points(speech, sr, embedder, hop=change_hop, threshold=change_threshold)
    groups = split_words(words, [clip_start + t for t, _ in changes]) if words else []
    if not groups:
        groups = [words]

    subs: list[SubSegment] = []
    for gi, g in enumerate(groups):
        if g:
            start = max(seg_start, g[0].start - lead)
            end = min(seg_end, g[-1].last + tail)
            if gi + 1 < len(groups) and groups[gi + 1]:
                end = min(end, groups[gi + 1][0].start - 0.02)
        else:
            start, end = seg_start, seg_end
        sub = SubSegment(g, round(start, 3), round(max(end, start), 3))
        a = max(0, int(round((start - clip_start) * sr)))
        b = max(a, int(round((end - clip_start) * sr)))
        clip = speech[a:b]
        if embedder is not None and clip.size / sr >= min_embed_seconds:
            sub.embedding = embedder.embed(clip, sr)
        subs.append(sub)
    if changes:
        subs[0].change_scores = [round(s, 3) for _, s in changes]
    return subs
