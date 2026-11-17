import asyncio

import numpy as np
import pytest

from rtspeech.scheduler import DecodeScheduler
from rtspeech.session import StreamingSession, display_text
from rtspeech.speaker import OnlineSpeakerClustering

from fakes import FakeAsr, FakeEmbedder, silence, tone


async def run(audio: np.ndarray, chunk: int = 1600, **kw):
    asr = FakeAsr()
    sched = DecodeScheduler(asr, max_wait_ms=0)
    await sched.start()
    s = StreamingSession(asr, sched, OnlineSpeakerClustering(threshold=0.6), embedder=FakeEmbedder(), **kw)
    events = []
    for i in range(0, audio.size, chunk):
        events += await s.feed(audio[i : i + chunk])
    events += await s.finish()
    await sched.stop()
    return events, s


def finals(events):
    return [e for e in events if e["type"] == "final"]


@pytest.mark.asyncio
async def test_two_speakers_separated_by_pauses():
    audio = np.concatenate(
        [silence(0.5), tone(220, 2.0), silence(1.2), tone(880, 2.0), silence(1.2), tone(220, 1.5), silence(0.5)]
    )
    events, s = await run(audio)
    f = finals(events)
    assert [e["speaker"] for e in f] == ["S1", "S2", "S1"]
    assert f[0]["start"] < f[0]["end"] <= f[1]["start"] < f[1]["end"] <= f[2]["start"]
    assert abs(f[1]["start"] - 3.7) < 0.3  # second voice starts at 0.5 + 2.0 + 1.2 s
    assert any(e["type"] == "partial" for e in events)


@pytest.mark.asyncio
async def test_speech_starting_inside_the_endpoint_window_is_not_lost():
    # The second utterance starts 0.5 s after the first: inside the fake 0.8 s endpoint
    # window measured from the (lagged) last token. Carry-over must keep its words.
    audio = np.concatenate([tone(220, 2.0), silence(0.5), tone(220, 2.0), silence(1.5)])
    events, _ = await run(audio)
    words = " ".join(e["text"] for e in finals(events)).split()
    assert len(words) >= 12  # ~6 tokens per 2 s utterance


@pytest.mark.asyncio
async def test_speaker_change_without_pause_is_split():
    audio = np.concatenate([tone(220, 2.5), silence(0.3), tone(880, 2.5), silence(1.2)])
    events, _ = await run(audio)
    assert [e["speaker"] for e in finals(events)] == ["S1", "S2"]


@pytest.mark.asyncio
async def test_finish_is_idempotent_and_feed_after_finish_fails():
    events, s = await run(tone(300, 1.5))
    assert await s.finish() == []
    with pytest.raises(RuntimeError):
        await s.feed(tone(300, 0.1))


@pytest.mark.asyncio
async def test_silence_only_produces_no_finals():
    events, _ = await run(silence(5.0))
    assert finals(events) == []


@pytest.mark.asyncio
async def test_scheduler_batches_concurrent_streams():
    asr = FakeAsr()
    sched = DecodeScheduler(asr, max_batch=16, max_wait_ms=20)
    await sched.start()
    streams = [asr.create_stream() for _ in range(8)]
    await asyncio.gather(*(sched.decode(st) for st in streams))
    await sched.stop()
    assert max(asr.batch_sizes) > 1
    assert sched.stats.streams == 8


def test_display_text():
    assert display_text("I THINK I'LL TRY IT") == "I think I'll try it"


@pytest.mark.asyncio
async def test_48khz_input_is_resampled_inside_the_session():
    from rtspeech.audio import resample

    audio16 = np.concatenate([tone(220, 2.0), silence(1.2), tone(880, 2.0), silence(1.0)])
    audio48 = resample(audio16, 16000, 48000)
    events, s = await run(audio48, chunk=4800, input_sample_rate=48000)
    assert [e["speaker"] for e in finals(events)] == ["S1", "S2"]
    assert abs(s.audio_seconds - audio16.size / 16000) < 0.05
