"""End-to-end benchmark: accuracy (WER, DER), speed (RTF) and live latency under concurrency.

    python bench/make_conversations.py --n 12 --seed 7 --out bench/data   # test set
    python bench/run_benchmark.py --data bench/data

Writes bench/results/results.json and bench/results/RESULTS.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "clients"))

from rtspeech.asr import SherpaStreamingAsr  # noqa: E402
from rtspeech.config import Settings  # noqa: E402
from rtspeech.speaker import TitaNetEmbedder  # noqa: E402

from evaluate import WordCounts, diarization_error, run_sync  # noqa: E402
from stream_wav import load_pcm16, stream_file  # noqa: E402


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def accuracy(convs: list[Path], settings: Settings, asr, emb) -> dict:
    words = WordCounts()
    der_frames = {"none": [], "0.25": []}
    count_ok, audio_s, compute_s = 0, 0.0, 0.0
    per_conv = []
    for meta_path in convs:
        meta = json.loads(meta_path.read_text())
        audio, _ = sf.read(meta_path.with_suffix(".wav"), dtype="float32")
        finals, elapsed = run_sync(audio, asr, emb, settings)
        ref = [(t["speaker"], t["start"], t["end"]) for t in meta["turns"]]
        hyp = [(f["speaker"], f["start"], f["end"]) for f in finals]
        speech = sum(e - s for _, s, e in ref)
        d0, d25 = diarization_error(ref, hyp), diarization_error(ref, hyp, collar=0.25)
        der_frames["none"].append((d0["der"], speech))
        der_frames["0.25"].append((d25["der"], speech))
        words.add(" ".join(t["text"] for t in meta["turns"]), " ".join(f["text"] for f in finals))
        count_ok += d0["hyp_speakers"] == d0["ref_speakers"]
        audio_s += audio.size / 16000
        compute_s += elapsed
        per_conv.append({"name": meta["name"], "speakers": meta["num_speakers"], "found": d0["hyp_speakers"],
                         "der": round(d0["der"], 4), "der_collar_0.25": round(d25["der"], 4)})

    def weighted(xs):
        return sum(d * w for d, w in xs) / sum(w for _, w in xs)

    return {
        "conversations": len(convs),
        "audio_minutes": round(audio_s / 60, 2),
        "wer": round(words.wer, 4),
        "der": round(weighted(der_frames["none"]), 4),
        "der_collar_0.25": round(weighted(der_frames["0.25"]), 4),
        "speaker_count_accuracy": round(count_ok / len(convs), 3),
        "rtf_single_stream": round(compute_s / audio_s, 4),
        "per_conversation": per_conv,
    }


def librispeech_wer(asr, settings: Settings) -> dict:
    d = settings.asr_model_dir / "test_wavs"
    refs = dict(line.split(" ", 1) for line in (d / "trans.txt").read_text().splitlines() if line.strip())
    words = WordCounts()
    for name, ref in refs.items():
        audio, sr = sf.read(d / name, dtype="float32")
        if sr != 16000:
            from rtspeech.audio import resample

            audio = resample(audio, sr, 16000)
        finals, _ = run_sync(audio, asr, None, replace(settings, detect_speaker_changes=False))
        words.add(ref, " ".join(f["text"] for f in finals))
    return {"utterances": len(refs), "words": words.ref_words, "wer": round(words.wer, 4)}


async def live_latency(convs: list[Path], concurrency: int, port: int, seconds: float) -> dict:
    url = f"ws://127.0.0.1:{port}/v1/stream"
    pcms = []
    for p in convs[:concurrency]:
        pcm = load_pcm16(p.with_suffix(".wav"))
        pcms.append(pcm[: int(seconds * 32000)])
    runs = await asyncio.gather(*(stream_file(url, pcm, 100, realtime=True, quiet=True) for pcm in pcms))
    partial_ms, final_ms = [], []
    for run in runs:
        keys = np.array(sorted(run["sent_at"]))
        for ev in run["events"]:
            if ev["type"] == "partial":
                sent = run["sent_at"].get(ev["audio_end"])
                if sent is not None:
                    partial_ms.append((ev["_recv"] - sent) * 1000)
        # An utterance can be split into several speaker segments; they are all labelled when
        # the utterance closes, so measure from the end of its last segment.
        last_of_utt: dict[int, dict] = {}
        for ev in run["events"]:
            if ev["type"] == "final" and ev["end"] >= last_of_utt.get(ev["utterance"], {"end": -1})["end"]:
                last_of_utt[ev["utterance"]] = ev
        for ev in last_of_utt.values():
            if ev["end"] > keys[-1]:  # closed by end-of-stream, not by the endpointer
                continue
            k = keys[np.searchsorted(keys, ev["end"])]
            final_ms.append((ev["_recv"] - run["sent_at"][float(k)]) * 1000)

    def pct(xs, q):
        return round(float(np.percentile(xs, q)), 1) if xs else None

    return {
        "concurrent_streams": concurrency,
        "partial_latency_ms": {"p50": pct(partial_ms, 50), "p95": pct(partial_ms, 95), "n": len(partial_ms)},
        "finalization_delay_ms": {"p50": pct(final_ms, 50), "p95": pct(final_ms, 95), "n": len(final_ms)},
    }


def start_server(port: int) -> subprocess.Popen:
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    proc = subprocess.Popen(
        [sys.executable, "-m", "rtspeech.server", "--port", str(port)],
        cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    import urllib.request

    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1)
            return proc
        except OSError:
            time.sleep(0.5)
    proc.kill()
    raise RuntimeError("server did not start")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("bench/data"))
    ap.add_argument("--out", type=Path, default=Path("bench/results"))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--live-seconds", type=float, default=30.0)
    ap.add_argument("--concurrency", default="1,2,4,6,8")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    settings = Settings.from_env()
    asr = SherpaStreamingAsr(settings)
    emb = TitaNetEmbedder(settings.speaker_model, settings.num_threads)
    convs = sorted(args.data.glob("conv*.json"))

    results = {
        "hardware": {"cpu": cpu_name(), "logical_cores": os.cpu_count(), "gpu": None},
        "settings": {k: str(v) for k, v in settings.__dict__.items()},
    }
    print("accuracy (full system)…", flush=True)
    results["synthetic_conversations"] = accuracy(convs, settings, asr, emb)
    print("ablation: no in-segment speaker-change detection…", flush=True)
    abl = accuracy(convs, replace(settings, detect_speaker_changes=False), asr, emb)
    results["ablation_no_change_detection"] = {k: v for k, v in abl.items() if k != "per_conversation"}
    print("LibriSpeech sample WER…", flush=True)
    results["librispeech_samples"] = librispeech_wer(asr, settings)

    print("live latency over WebSocket…", flush=True)
    server = start_server(args.port)
    try:
        results["live"] = []
        for n in [int(x) for x in args.concurrency.split(",")]:
            r = asyncio.run(live_latency(convs, n, args.port, args.live_seconds))
            print(r, flush=True)
            results["live"].append(r)
        import urllib.request

        results["server_metrics"] = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{args.port}/metrics").read())
    finally:
        server.terminate()

    (args.out / "results.json").write_text(json.dumps(results, indent=2))
    (args.out / "RESULTS.md").write_text(render(results))
    print(render(results))


def render(r: dict) -> str:
    s, a = r["synthetic_conversations"], r["ablation_no_change_detection"]
    lines = [
        "# Benchmark results",
        "",
        f"Hardware: {r['hardware']['cpu']}, {r['hardware']['logical_cores']} logical cores, no GPU. "
        "Streaming Zipformer transducer (int8), TitaNet-small speaker embeddings, ONNX Runtime on CPU.",
        "",
        f"Test set: {s['conversations']} synthetic multi-speaker conversations ({s['audio_minutes']} min, "
        "2-4 speakers, Piper libritts_r voices), generated with seed 7; thresholds were tuned on a separate "
        "dev set (seed 1).",
        "",
        "| Metric | Full system | Without in-segment change detection |",
        "|---|---|---|",
        f"| WER | {s['wer']:.1%} | {a['wer']:.1%} |",
        f"| DER (no collar) | {s['der']:.1%} | {a['der']:.1%} |",
        f"| DER (0.25 s collar) | {s['der_collar_0.25']:.1%} | {a['der_collar_0.25']:.1%} |",
        f"| Correct number of speakers | {s['speaker_count_accuracy']:.0%} | {a['speaker_count_accuracy']:.0%} |",
        f"| Real-time factor, 1 stream | {s['rtf_single_stream']:.3f} | {a['rtf_single_stream']:.3f} |",
        "",
        f"Real speech: WER {r['librispeech_samples']['wer']:.1%} on the {r['librispeech_samples']['utterances']} "
        f"LibriSpeech test utterances shipped with the model ({r['librispeech_samples']['words']} words; "
        "a smoke test, not a benchmark).",
        "",
        "Live WebSocket latency (100 ms chunks sent in real time, 30 s per stream):",
        "",
        "| Concurrent streams | Partial latency p50 / p95 | Utterance finalization delay p50 / p95 |",
        "|---|---|---|",
    ]
    for x in r["live"]:
        p, f = x["partial_latency_ms"], x["finalization_delay_ms"]
        lines.append(f"| {x['concurrent_streams']} | {p['p50']} / {p['p95']} ms | {f['p50']} / {f['p95']} ms |")
    lines += [
        "",
        "Partial latency: time from sending a chunk to receiving the partial transcript that includes it. "
        "Utterance finalization delay: time from the end of an utterance's speech to its speaker-labelled "
        "final result; it includes the 0.8 s of silence the endpointer waits for. When two people talk "
        "without a pause, both turns are labelled together when that utterance closes.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
