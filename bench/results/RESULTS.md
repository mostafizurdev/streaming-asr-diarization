# Benchmark results

Hardware: Intel(R) Xeon(R) Processor @ 2.10GHz, 2 logical cores, no GPU. Streaming Zipformer transducer (int8), TitaNet-small speaker embeddings, ONNX Runtime on CPU.

Test set: 12 synthetic multi-speaker conversations (8.29 min, 2-4 speakers, Piper libritts_r voices), generated with seed 7; thresholds were tuned on a separate dev set (seed 1).

| Metric | Full system | Without in-segment change detection |
|---|---|---|
| WER | 8.8% | 8.8% |
| DER (no collar) | 22.1% | 55.3% |
| DER (0.25 s collar) | 11.9% | 44.6% |
| Correct number of speakers | 67% | 25% |
| Real-time factor, 1 stream | 0.216 | 0.170 |

Real speech: WER 0.0% on the 3 LibriSpeech test utterances shipped with the model (77 words; a smoke test, not a benchmark).

Live WebSocket latency (100 ms chunks sent in real time, 30 s per stream):

| Concurrent streams | Partial latency p50 / p95 | Utterance finalization delay p50 / p95 |
|---|---|---|
| 1 | 42.5 / 70.8 ms | 1519.7 / 3071.4 ms |
| 2 | 50.8 / 99.5 ms | 1774.3 / 3348.6 ms |
| 4 | 61.5 / 158.9 ms | 2636.8 / 4646.5 ms |
| 6 | 392.8 / 724.2 ms | 4654.7 / 6034.6 ms |
| 8 | 963.5 / 3699.9 ms | 7889.8 / 12256.8 ms |

Partial latency: time from sending a chunk to receiving the partial transcript that includes it. Utterance finalization delay: time from the end of an utterance's speech to its speaker-labelled final result; it includes the 0.8 s of silence the endpointer waits for. When two people talk without a pause, both turns are labelled together when that utterance closes.
