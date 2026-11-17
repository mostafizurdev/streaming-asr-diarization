import numpy as np

from rtspeech.speaker import OnlineSpeakerClustering


def unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


A, B, C = unit([1, 0, 0]), unit([0, 1, 0]), unit([0, 0, 1])


def test_two_speakers_alternating():
    c = OnlineSpeakerClustering(threshold=0.6)
    labels = [c.assign(e).label for e in (A, B, A, unit([0.9, 0.1, 0]), B)]
    assert labels == ["S1", "S2", "S1", "S1", "S2"]


def test_new_speaker_flag_and_score():
    c = OnlineSpeakerClustering(threshold=0.6)
    first = c.assign(A)
    again = c.assign(A)
    assert first.is_new and first.score is None
    assert not again.is_new and again.score > 0.99


def test_short_segment_inherits_previous_speaker():
    c = OnlineSpeakerClustering()
    c.assign(A)
    c.assign(B)
    assert c.assign(None).label == "S2"


def test_short_first_segment_becomes_placeholder_that_adopts_first_voiceprint():
    c = OnlineSpeakerClustering(threshold=0.6)
    assert c.assign(None).label == "S1"
    assert c.assign(A).label == "S1"  # first reliable embedding fills the placeholder
    assert c.assign(B).label == "S2"
    assert c.assign(A).label == "S1"


def test_max_speakers_caps_new_clusters():
    c = OnlineSpeakerClustering(threshold=0.99, max_speakers=2)
    labels = [c.assign(e).label for e in (A, B, C)]
    assert labels[:2] == ["S1", "S2"]
    assert labels[2] in {"S1", "S2"}
    assert len(c.speakers) == 2


def test_centroid_stays_normalised():
    c = OnlineSpeakerClustering(threshold=0.1)
    for e in (A, unit([1, 1, 0]), unit([1, 0, 1])):
        c.assign(e, weight=2.0)
    assert abs(np.linalg.norm(c.speakers[0].centroid) - 1.0) < 1e-5
