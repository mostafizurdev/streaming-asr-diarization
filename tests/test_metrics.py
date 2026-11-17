from evaluate import WordCounts, diarization_error, normalize


def test_perfect_diarization_with_different_labels():
    ref = [("alice", 0.0, 2.0), ("bob", 2.5, 4.0)]
    hyp = [("S2", 0.0, 2.0), ("S1", 2.5, 4.0)]
    assert diarization_error(ref, hyp)["der"] == 0.0


def test_confusion_and_missed_speech():
    ref = [("a", 0.0, 2.0), ("b", 2.0, 4.0)]
    hyp = [("S1", 0.0, 3.0)]  # b's first second labelled as a, last second missed
    out = diarization_error(ref, hyp)
    assert abs(out["confusion"] - 0.25) < 0.01
    assert abs(out["missed"] - 0.25) < 0.01


def test_collar_forgives_boundary_errors():
    ref = [("a", 0.0, 2.0), ("b", 2.0, 4.0)]
    hyp = [("S1", 0.0, 2.2), ("S2", 2.2, 4.0)]
    assert diarization_error(ref, hyp)["der"] > 0
    assert diarization_error(ref, hyp, collar=0.25)["der"] == 0.0


def test_wer_counts():
    w = WordCounts()
    w.add("The new model works.", "the new model work")
    assert w.ref_words == 4 and w.substitutions == 1
    assert normalize("Fall-back, OK?") == "FALL BACK OK"
