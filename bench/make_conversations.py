"""Generate synthetic multi-speaker conversations with known speakers, timings and text.

Real conversational corpora with speaker labels (AMI, VoxConverse) are large and licence-bound,
so the benchmark builds reproducible test conversations with a multi-speaker TTS voice
(Piper, en_US libritts_r medium: 904 voices). Every turn's speaker, start, end and text is
known exactly, which gives ground truth for WER and diarization error rate.

Synthetic speech is cleaner than real speech, so treat these numbers as an upper bound on
accuracy; the benchmark also reports WER on real LibriSpeech utterances.

    python bench/make_conversations.py --n 12 --out bench/data
"""

from __future__ import annotations

import argparse
import json
import random
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

# Short, plain sentences written for this benchmark (no digits or acronyms, so the
# reference text matches what an English ASR model is expected to output).
SENTENCES = [
    "Let's move the release to Thursday so we can finish the load tests.",
    "The new model handles noisy phone calls much better than the old one.",
    "We should check the latency numbers before the demo tomorrow.",
    "I think the database migration can wait until next week.",
    "Can you share the dashboard link with the whole team?",
    "The customer asked for a summary of every meeting in one place.",
    "Our error rate went down after we cleaned the training data.",
    "Please send me the meeting notes when you have a moment.",
    "The servers in the second region are still running the old version.",
    "We need a better way to separate speakers on short calls.",
    "I will review the pull request after lunch.",
    "The recording sounds clear, but the room has a strong echo.",
    "Should we cache the results or compute them every time?",
    "The marketing team wants the feature ready before the conference.",
    "Let me check the logs and get back to you.",
    "That sounds reasonable, but we should measure it first.",
    "The battery drains quickly when the microphone stays open.",
    "Our users mostly speak in short sentences with long pauses.",
    "We can reduce the cost by running the small model on the edge.",
    "I agree with the plan, as long as we keep the old path as a fallback.",
    "The transcript looks good, but the names are often wrong.",
    "Could we add a feedback button next to each sentence?",
    "The deadline is tight, so let's keep the scope small.",
    "Most of the delay comes from waiting for the end of the sentence.",
    "I tried the new settings and the results were much more stable.",
    "Let's write down the decision so nobody forgets it.",
    "The support team gets a lot of calls on Monday morning.",
    "We should test with older phones as well as new ones.",
    "Who is responsible for the release notes this time?",
    "The model sometimes confuses similar voices in the same family.",
    "Maybe we can split the work between the two teams.",
    "I will update the documentation before the end of the day.",
    "The quality is fine for search, but not for legal records.",
    "Let's run the experiment again with a different random seed.",
    "The audio was recorded at a lower sample rate than we expected.",
    "We could stream the partial results to make the screen feel faster.",
    "Thanks, that answers my question about the pricing.",
    "The meeting room microphone picks up the air conditioner.",
    "Our next goal is to support more languages without slowing down.",
    "Please make sure the test set does not overlap with the training set.",
    "The graph shows a small spike every night around midnight.",
    "I would rather fix the root cause than add another workaround.",
    "Can we schedule a short call with the design team?",
    "The new interface is simpler, and people seem to like it.",
    "Let's keep an eye on memory usage when many users connect.",
    "The speaker labels were correct for almost the whole call.",
    "We should warn users when the connection becomes unstable.",
    "I can take that task if nobody else has time.",
]


def synth(voice, text: str, speaker_id: int, sr_out: int = 16000) -> np.ndarray:
    from piper.config import SynthesisConfig

    chunks = voice.synthesize(text, syn_config=SynthesisConfig(speaker_id=speaker_id))
    audio = np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32)
    sr_in = voice.config.sample_rate
    g = gcd(sr_in, sr_out)
    return resample_poly(audio, sr_out // g, sr_in // g).astype(np.float32)


def trim(audio: np.ndarray, thresh: float = 0.01) -> np.ndarray:
    idx = np.where(np.abs(audio) > thresh)[0]
    return audio if idx.size == 0 else audio[max(0, idx[0] - 160) : idx[-1] + 160]


def make_conversation(voice, rng: random.Random, speakers: list[int], n_turns: int, sr: int = 16000):
    pieces: list[np.ndarray] = [np.zeros(int(0.5 * sr), np.float32)]
    t = 0.5
    turns = []
    gains = {s: rng.uniform(0.55, 0.95) for s in speakers}
    current = rng.choice(speakers)
    pool = SENTENCES.copy()
    rng.shuffle(pool)
    for i in range(n_turns):
        if i and rng.random() < 0.85:
            current = rng.choice([s for s in speakers if s != current])
        text = " ".join(pool.pop() for _ in range(1 if rng.random() < 0.6 else 2))
        audio = trim(synth(voice, text, current, sr))
        audio = audio / (np.abs(audio).max() + 1e-9) * gains[current]
        turns.append({"speaker": f"spk{speakers.index(current) + 1}", "voice_id": current,
                      "start": round(t, 3), "end": round(t + audio.size / sr, 3), "text": text})
        pieces.append(audio)
        t += audio.size / sr
        gap = rng.uniform(0.3, 1.5)  # some gaps are shorter than the endpointing silence
        pieces.append(np.zeros(int(gap * sr), np.float32))
        t += gap
    return np.concatenate(pieces), turns


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="models/vits-piper-en_US-libritts_r-medium/en_US-libritts_r-medium.onnx")
    ap.add_argument("--out", type=Path, default=Path("bench/data"))
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    from piper import PiperVoice

    voice = PiperVoice.load(args.voice)
    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for k in range(args.n):
        n_speakers = rng.choice([2, 2, 3, 3, 4])
        speakers = rng.sample(range(voice.config.num_speakers), n_speakers)
        audio, turns = make_conversation(voice, rng, speakers, n_turns=rng.randint(8, 12))
        name = f"conv{k:02d}"
        sf.write(args.out / f"{name}.wav", audio, 16000, subtype="PCM_16")
        meta = {"name": name, "duration": round(audio.size / 16000, 2), "num_speakers": n_speakers, "turns": turns}
        (args.out / f"{name}.json").write_text(json.dumps(meta, indent=2))
        manifest.append({"name": name, "duration": meta["duration"], "num_speakers": n_speakers})
        print(f"{name}: {meta['duration']:6.1f}s, {n_speakers} speakers, {len(turns)} turns")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
