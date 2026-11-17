"""Stream a WAV file to the server as if it were a live microphone, and print the transcript.

    python clients/stream_wav.py samples/meeting.wav
    python clients/stream_wav.py samples/meeting.wav --fast     # no real-time pacing
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import websockets

COLORS = ["\033[96m", "\033[93m", "\033[95m", "\033[92m", "\033[91m", "\033[94m"]
RESET = "\033[0m"


def load_pcm16(path: Path, target_rate: int = 16000) -> bytes:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_rate:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(sr, target_rate)
        audio = resample_poly(audio, target_rate // g, sr // g).astype(np.float32)
    return (np.clip(audio, -1, 32767 / 32768) * 32768).astype("<i2").tobytes()


async def stream_file(url: str, pcm: bytes, chunk_ms: int = 100, realtime: bool = True, quiet: bool = False) -> dict:
    """Send `pcm` (16 kHz PCM16) in chunks; return events with client-side timing."""
    bytes_per_chunk = 16000 * 2 * chunk_ms // 1000
    sent_at: dict[float, float] = {}  # audio time at end of chunk -> wall clock when sent
    events: list[dict] = []

    async with websockets.connect(f"{url}?sample_rate=16000", max_size=None) as ws:
        ready = json.loads(await ws.recv())
        assert ready["type"] == "ready", ready
        t_start = time.perf_counter()

        async def sender():
            for i, off in enumerate(range(0, len(pcm), bytes_per_chunk)):
                if realtime:  # pace against the wall clock, not by sleeping a fixed amount
                    await asyncio.sleep(max(0.0, t_start + i * chunk_ms / 1000 - time.perf_counter()))
                chunk = pcm[off : off + bytes_per_chunk]
                audio_end = round((off + len(chunk)) / 32000, 3)
                sent_at[audio_end] = time.perf_counter()
                await ws.send(chunk)
            await ws.send(json.dumps({"type": "eof"}))

        async def receiver():
            async for raw in ws:
                ev = json.loads(raw)
                ev["_recv"] = time.perf_counter()
                events.append(ev)
                if quiet:
                    continue
                if ev["type"] == "partial":
                    print(f"\r  … {ev['text'][-90:]}", end="", flush=True)
                elif ev["type"] == "final":
                    c = COLORS[(int(ev["speaker"][1:]) - 1) % len(COLORS)]
                    print(f"\r{c}[{ev['start']:7.2f}-{ev['end']:7.2f}] {ev['speaker']}{RESET}: {ev['text']}" + " " * 10)
                elif ev["type"] == "done":
                    print(f"\n{ev}")

        await asyncio.gather(sender(), receiver())

    return {"events": events, "sent_at": sent_at, "t_start": t_start}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", type=Path)
    ap.add_argument("--url", default="ws://localhost:8000/v1/stream")
    ap.add_argument("--chunk-ms", type=int, default=100)
    ap.add_argument("--fast", action="store_true", help="send as fast as possible (no real-time pacing)")
    args = ap.parse_args()
    asyncio.run(stream_file(args.url, load_pcm16(args.wav), args.chunk_ms, realtime=not args.fast))


if __name__ == "__main__":
    main()
