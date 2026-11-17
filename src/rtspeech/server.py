"""FastAPI service: WebSocket streaming endpoint, health and metrics, and a browser demo."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from . import __version__
from .asr import SherpaStreamingAsr, StreamingAsr
from .audio import SUPPORTED_SAMPLE_RATES, pcm16_to_float32
from .config import Settings
from .scheduler import DecodeScheduler
from .session import StreamingSession
from .speaker import Embedder, OnlineSpeakerClustering, TitaNetEmbedder

log = logging.getLogger("rtspeech")
def _web_dir() -> Path:
    """The demo page lives in ./web (repo checkout or the Docker image's working dir)."""
    for candidate in (os.environ.get("WEB_DIR"), Path(__file__).resolve().parents[2] / "web", Path.cwd() / "web"):
        if candidate and (Path(candidate) / "index.html").exists():
            return Path(candidate)
    return Path.cwd() / "web"


WEB_DIR = _web_dir()


class Metrics:
    """In-process counters; enough for a demo, swap for Prometheus in production."""

    def __init__(self) -> None:
        self.active_sessions = 0
        self.total_sessions = 0
        self.audio_seconds = 0.0
        self.compute_seconds = 0.0
        self.chunk_ms: deque[float] = deque(maxlen=20000)

    def snapshot(self, scheduler: DecodeScheduler) -> dict:
        lat = np.array(self.chunk_ms) if self.chunk_ms else np.zeros(1)
        return {
            "active_sessions": self.active_sessions,
            "total_sessions": self.total_sessions,
            "audio_seconds": round(self.audio_seconds, 2),
            # wall-clock processing per second of audio, including queueing under load
            "processing_to_audio_ratio": (
                round(self.compute_seconds / self.audio_seconds, 4) if self.audio_seconds else None
            ),
            "chunk_processing_ms": {
                "p50": round(float(np.percentile(lat, 50)), 2),
                "p95": round(float(np.percentile(lat, 95)), 2),
            },
            "decode_batches": scheduler.stats.batches,
            "mean_decode_batch": round(scheduler.stats.mean_batch, 2),
        }


def create_app(
    settings: Settings | None = None,
    asr: StreamingAsr | None = None,
    embedder: Embedder | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    state: dict = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state["asr"] = asr or SherpaStreamingAsr(settings)
        state["embedder"] = embedder if embedder is not None else TitaNetEmbedder(settings.speaker_model, 1)
        state["embed_pool"] = ThreadPoolExecutor(max_workers=1, thread_name_prefix="spk-embed")
        state["scheduler"] = DecodeScheduler(state["asr"], settings.max_batch, settings.max_wait_ms)
        state["metrics"] = Metrics()
        await state["scheduler"].start()
        log.info("models loaded; ready")
        yield
        await state["scheduler"].stop()
        state["embed_pool"].shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="Streaming ASR + online diarization", version=__version__, lifespan=lifespan)

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok", "version": __version__}

    @app.get("/metrics")
    async def metrics():
        return JSONResponse(state["metrics"].snapshot(state["scheduler"]))

    @app.get("/")
    async def index():
        return FileResponse(WEB_DIR / "index.html")

    @app.websocket("/v1/stream")
    async def stream(ws: WebSocket, sample_rate: int = 16000):
        await ws.accept()
        if sample_rate not in SUPPORTED_SAMPLE_RATES:
            await ws.send_json({"type": "error", "message": f"unsupported sample_rate {sample_rate}"})
            await ws.close(code=1003)
            return

        loop = asyncio.get_running_loop()
        pool = state["embed_pool"]

        async def run_blocking(fn, *args):
            return await loop.run_in_executor(pool, fn, *args)

        send_lock = asyncio.Lock()  # partials (handler) and finals (labelling task) share the socket

        async def send(events: list[dict]) -> None:
            async with send_lock:
                for event in events:
                    await ws.send_json(event)

        session = StreamingSession(
            asr=state["asr"],
            scheduler=state["scheduler"],
            clustering=OnlineSpeakerClustering(settings.speaker_threshold, settings.max_speakers),
            embedder=state["embedder"],
            run_blocking=run_blocking,
            input_sample_rate=sample_rate,
            min_embed_seconds=settings.min_embed_seconds,
            detect_changes=settings.detect_speaker_changes,
            change_threshold=settings.change_threshold,
            change_hop=settings.change_hop,
            on_final=send,
        )
        m: Metrics = state["metrics"]
        m.active_sessions += 1
        m.total_sessions += 1
        max_bytes = int(settings.max_message_seconds * sample_rate) * 2
        compute = 0.0
        await send([{"type": "ready", "sample_rate": sample_rate, "format": "pcm_s16le_mono"}])
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                data = msg.get("bytes")
                if data is not None:
                    if len(data) > max_bytes:
                        await send([{"type": "error", "message": "audio chunk too large"}])
                        await ws.close(code=1009)
                        break
                    t0 = time.perf_counter()
                    events = await session.feed(pcm16_to_float32(data))
                    dt = time.perf_counter() - t0
                    compute += dt
                    m.chunk_ms.append(dt * 1000)
                    await send(events)
                    continue
                text = msg.get("text")
                if text and json.loads(text).get("type") == "eof":
                    t0 = time.perf_counter()
                    await send(await session.finish())
                    compute += time.perf_counter() - t0
                    speakers = sorted({s["speaker"] for s in session.segments})
                    await send(
                        [{
                            "type": "done",
                            "audio_seconds": round(session.audio_seconds, 2),
                            "segments": len(session.segments),
                            "speakers": speakers,
                            "real_time_factor": (
                                round(compute / session.audio_seconds, 4) if session.audio_seconds else None
                            ),
                        }]
                    )
                    await ws.close()
                    break
        except WebSocketDisconnect:
            pass
        finally:
            m.active_sessions -= 1
            m.audio_seconds += session.audio_seconds
            m.compute_seconds += compute

    return app


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Run the streaming ASR + diarization server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(create_app(), host=args.host, port=args.port, ws_max_size=4 * 1024 * 1024)


if __name__ == "__main__":  # pragma: no cover
    main()
