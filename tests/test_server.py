import json

import numpy as np
from fastapi.testclient import TestClient

from rtspeech.audio import float32_to_pcm16, pcm16_to_float32
from rtspeech.config import Settings
from rtspeech.server import create_app

from fakes import FakeAsr, FakeEmbedder, silence, tone


def make_client():
    app = create_app(Settings(max_wait_ms=0), asr=FakeAsr(), embedder=FakeEmbedder())
    return TestClient(app)


def test_health_and_metrics():
    with make_client() as client:
        assert client.get("/healthz").json()["status"] == "ok"
        assert client.get("/metrics").json()["total_sessions"] == 0


def test_websocket_stream_end_to_end():
    audio = np.concatenate([tone(220, 2.0), silence(1.2), tone(880, 2.0), silence(0.5)])
    pcm = float32_to_pcm16(audio)
    with make_client() as client:
        with client.websocket_connect("/v1/stream?sample_rate=16000") as ws:
            assert ws.receive_json()["type"] == "ready"
            for i in range(0, len(pcm), 3200):
                ws.send_bytes(pcm[i : i + 3200])
            ws.send_text(json.dumps({"type": "eof"}))
            events = []
            while True:
                ev = ws.receive_json()
                events.append(ev)
                if ev["type"] == "done":
                    break
        finals = [e for e in events if e["type"] == "final"]
        assert [e["speaker"] for e in finals] == ["S1", "S2"]
        assert events[-1]["speakers"] == ["S1", "S2"]
        m = client.get("/metrics").json()
        assert m["total_sessions"] == 1 and m["active_sessions"] == 0


def test_rejects_unsupported_sample_rate():
    with make_client() as client:
        with client.websocket_connect("/v1/stream?sample_rate=12345") as ws:
            assert ws.receive_json()["type"] == "error"


def test_rejects_oversized_chunk():
    with make_client() as client:
        with client.websocket_connect("/v1/stream?sample_rate=16000") as ws:
            ws.receive_json()
            ws.send_bytes(b"\x00\x00" * 16000 * 3)  # 3 s in one message
            assert ws.receive_json()["message"] == "audio chunk too large"


def test_pcm16_roundtrip():
    x = np.array([0.0, 0.5, -0.5, 0.99], np.float32)
    assert np.allclose(pcm16_to_float32(float32_to_pcm16(x)), x, atol=1e-4)
