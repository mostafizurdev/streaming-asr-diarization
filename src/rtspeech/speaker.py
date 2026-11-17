"""Speaker embeddings (NVIDIA NeMo TitaNet via ONNX) and online speaker clustering."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray: ...


class TitaNetEmbedder:
    """192-dim TitaNet-small speaker embeddings, L2-normalised."""

    def __init__(self, model: Path, num_threads: int = 1):
        import sherpa_onnx

        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(model), num_threads=num_threads)
        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
        self.dim = self._extractor.dim

    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        stream = self._extractor.create_stream()
        stream.accept_waveform(sample_rate, samples)
        stream.input_finished()
        vec = np.asarray(self._extractor.compute(stream), dtype=np.float32)
        return vec / (np.linalg.norm(vec) + 1e-9)


@dataclass
class SpeakerAssignment:
    label: str
    score: float | None  # cosine similarity to the chosen centroid (None if inherited)
    is_new: bool


@dataclass
class _Speaker:
    label: str
    centroid: np.ndarray | None  # None until the speaker has a usable embedding
    weight: float = 0.0


@dataclass
class OnlineSpeakerClustering:
    """Leader-follower clustering of segment embeddings.

    A segment joins the most similar known speaker if cosine similarity >= threshold,
    otherwise it starts a new speaker (until `max_speakers` is reached). Centroids are
    duration-weighted running means, re-normalised after each update. Segments too short
    for a reliable embedding (embedding=None) inherit the previous speaker, which is the
    most likely label in turn-taking conversation.
    """

    threshold: float = 0.65
    max_speakers: int = 8
    speakers: list[_Speaker] = field(default_factory=list)
    _last: str | None = None

    def assign(self, embedding: np.ndarray | None, weight: float = 1.0) -> SpeakerAssignment:
        if embedding is None:
            if self._last is None:
                return self._new(None, weight)
            return SpeakerAssignment(self._last, None, False)

        known = [s for s in self.speakers if s.centroid is not None]
        if not known:
            placeholder = next((s for s in self.speakers if s.centroid is None), None)
            if placeholder is not None:
                # The conversation so far only had short segments: give that speaker
                # this first reliable voiceprint.
                placeholder.centroid = embedding.astype(np.float32)
                placeholder.weight = weight
                self._last = placeholder.label
                return SpeakerAssignment(placeholder.label, None, False)
            return self._new(embedding, weight)

        sims = np.array([float(s.centroid @ embedding) for s in known])
        best = int(sims.argmax())
        if sims[best] >= self.threshold or len(self.speakers) >= self.max_speakers:
            spk = known[best]
            total = spk.weight + weight
            c = (spk.centroid * spk.weight + embedding * weight) / total
            spk.centroid = c / (np.linalg.norm(c) + 1e-9)
            spk.weight = total
            self._last = spk.label
            return SpeakerAssignment(spk.label, float(sims[best]), False)
        return self._new(embedding, weight)

    def _new(self, embedding: np.ndarray | None, weight: float) -> SpeakerAssignment:
        label = f"S{len(self.speakers) + 1}"
        centroid = None if embedding is None else embedding.astype(np.float32)
        self.speakers.append(_Speaker(label, centroid, weight if embedding is not None else 0.0))
        self._last = label
        return SpeakerAssignment(label, None, True)
