import numpy as np

from rtspeech.segmenter import Word, change_points, speaker_pass, split_words, words_from_tokens

from fakes import FakeEmbedder, silence, tone


def test_words_from_bpe_tokens_and_lag_correction():
    words = words_from_tokens([" HEL", "LO", " WOR", "LD"], [1.0, 1.1, 1.5, 1.6], lag=0.2)
    assert [w.text for w in words] == ["HELLO", "WORLD"]
    assert words[0].start == 0.8 and abs(words[0].last - 0.9) < 1e-9


def test_split_prefers_the_long_pause_near_the_change():
    words = [Word("A", 0.0, 0.2), Word("B", 0.35, 0.5), Word("C", 1.2, 1.3), Word("D", 1.4, 1.5)]
    groups = split_words(words, changes=[0.6])  # nearest gap is A|B, but B|C is the real pause
    assert [[w.text for w in g] for g in groups] == [["A", "B"], ["C", "D"]]


def test_no_change_keeps_one_group():
    words = [Word("A", 0.0, 0.1), Word("B", 0.3, 0.4)]
    assert split_words(words, []) == [words]


def test_change_point_found_between_two_voices():
    audio = np.concatenate([tone(220, 2.5), tone(880, 2.5)])
    found = change_points(audio, 16000, FakeEmbedder(), threshold=0.5)
    assert len(found) == 1
    assert abs(found[0][0] - 2.5) <= 0.25


def test_same_voice_has_no_change_point():
    assert change_points(tone(300, 5.0), 16000, FakeEmbedder()) == []


def test_speaker_pass_splits_text_by_voice():
    audio = np.concatenate([tone(220, 2.0), silence(0.3), tone(880, 2.0)])
    # one word per 0.5 s, the second voice starts at 2.3 s
    words = [Word(f"w{i}", t, t + 0.1) for i, t in enumerate([0.1, 0.6, 1.1, 1.6, 2.4, 2.9, 3.4, 3.9])]
    subs = speaker_pass(FakeEmbedder(), audio, 16000, 0.0, words, 0.0, 4.3, min_embed_seconds=0.5)
    assert [s.text for s in subs] == ["w0 w1 w2 w3", "w4 w5 w6 w7"]
    assert all(s.embedding is not None for s in subs)
    assert float(subs[0].embedding @ subs[1].embedding) < 0.5
