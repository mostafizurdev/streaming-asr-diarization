"""Grid-search the two speaker thresholds on a separate *dev* set (never on the test set).

    python bench/make_conversations.py --n 4 --seed 1 --out bench/dev
    python bench/tune.py --data bench/dev
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import replace
from pathlib import Path

import soundfile as sf

from rtspeech.asr import SherpaStreamingAsr
from rtspeech.config import Settings
from rtspeech.speaker import TitaNetEmbedder

from evaluate import diarization_error, run_sync


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("bench/dev"))
    args = ap.parse_args()

    base = Settings()
    asr = SherpaStreamingAsr(base)
    emb = TitaNetEmbedder(base.speaker_model, base.num_threads)
    convs = sorted(args.data.glob("conv*.json"))
    rows = []
    for spk_t, chg_t in itertools.product([0.55, 0.6, 0.65, 0.7], [0.35, 0.45, 0.55]):
        s = replace(base, speaker_threshold=spk_t, change_threshold=chg_t)
        ders = []
        for meta_path in convs:
            meta = json.loads(meta_path.read_text())
            audio, _ = sf.read(meta_path.with_suffix(".wav"), dtype="float32")
            finals, _ = run_sync(audio, asr, emb, s)
            ref = [(t["speaker"], t["start"], t["end"]) for t in meta["turns"]]
            hyp = [(f["speaker"], f["start"], f["end"]) for f in finals]
            ders.append(diarization_error(ref, hyp, collar=0.25)["der"])
        mean = sum(ders) / len(ders)
        rows.append({"speaker_threshold": spk_t, "change_threshold": chg_t, "der_collar_0.25": round(mean, 4)})
        print(rows[-1], flush=True)
    best = min(rows, key=lambda r: r["der_collar_0.25"])
    print("best:", best)
    out = Path("bench/results/tuning_dev.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"grid": rows, "best": best}, indent=2))


if __name__ == "__main__":
    main()
