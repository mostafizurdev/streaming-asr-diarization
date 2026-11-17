"""Dynamic micro-batching of decode calls across concurrent sessions.

Each WebSocket session pushes its stream here after appending audio. A single worker
collects whatever streams are waiting (up to `max_batch`, or until `max_wait_ms` passes)
and decodes them in one ONNX call on a dedicated thread, so the event loop never blocks
and the model runs on larger batches when many users are connected.
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from .asr import StreamingAsr


@dataclass
class SchedulerStats:
    batches: int = 0
    streams: int = 0
    busy_seconds: float = 0.0

    @property
    def mean_batch(self) -> float:
        return self.streams / self.batches if self.batches else 0.0


class DecodeScheduler:
    def __init__(self, asr: StreamingAsr, max_batch: int = 16, max_wait_ms: float = 4.0):
        self.asr = asr
        self.max_batch = max_batch
        self.max_wait = max_wait_ms / 1000.0
        self.stats = SchedulerStats()
        self._queue: asyncio.Queue[tuple[Any, asyncio.Future]] | None = None
        self._task: asyncio.Task | None = None
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="asr-decode")

    async def start(self) -> None:
        if self._task is None:
            self._queue = asyncio.Queue()
            self._task = asyncio.create_task(self._run(), name="decode-scheduler")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self._pool.shutdown(wait=False, cancel_futures=True)

    async def decode(self, stream: Any) -> None:
        """Decode every frame that is ready in `stream`; returns when done."""
        if self._queue is None:
            raise RuntimeError("scheduler not started")
        fut = asyncio.get_running_loop().create_future()
        await self._queue.put((stream, fut))
        await fut

    async def _run(self) -> None:
        assert self._queue is not None
        loop = asyncio.get_running_loop()
        while True:
            batch = [await self._queue.get()]
            deadline = loop.time() + self.max_wait
            while len(batch) < self.max_batch:
                timeout = deadline - loop.time()
                if timeout <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self._queue.get(), timeout))
                except asyncio.TimeoutError:
                    break
            streams = [s for s, _ in batch]
            t0 = time.perf_counter()
            try:
                await loop.run_in_executor(self._pool, self.asr.decode_batch, streams)
            except Exception as exc:  # surface model errors to every waiting session
                for _, fut in batch:
                    if not fut.done():
                        fut.set_exception(exc)
                continue
            self.stats.busy_seconds += time.perf_counter() - t0
            self.stats.batches += 1
            self.stats.streams += len(streams)
            for _, fut in batch:
                if not fut.done():
                    fut.set_result(None)
