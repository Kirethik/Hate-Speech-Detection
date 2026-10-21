import numpy as np

from voice.splice import fit, match_loud, snap, soft_beep, splice, trim_silence, xjoin

SR = 16000


def tone(dur, f=220, a=0.3):
    t = np.arange(int(dur * SR)) / SR
    return (a * np.sin(2 * np.pi * f * t)).astype(np.float32)


def test_snap_moves_cut_into_silence():
    y = np.concatenate([tone(1.0), np.zeros(int(0.1 * SR), np.float32), tone(1.0)])
    t = snap(y, SR, 1.02)
    assert 1.0 <= t <= 1.1


def test_fit_stretches_mild_ratio_to_exact_slot():
    w = tone(1.1)
    out = fit(w, SR)
    assert len(out) == SR


def test_fit_leaves_big_mismatch_alone():
    w = tone(2.0)
    assert len(fit(w, SR)) == len(w)


def test_xjoin_crossfade_length():
    a, b = tone(1.0), tone(1.0)
    out = xjoin([a, b], SR, ms=25)
    assert len(out) == len(a) + len(b) - int(SR * 0.025)


def test_match_loud_equalises_level():
    quiet, loud = tone(1.0, a=0.02), tone(1.0, a=0.4)
    out = match_loud(quiet, loud, SR)
    assert abs(np.sqrt(np.mean(out ** 2)) - np.sqrt(np.mean(loud ** 2))) < 0.05


def test_splice_shifts_timeline_and_keeps_rest():
    y = np.concatenate([tone(1.0), np.zeros(SR // 5, np.float32), tone(1.0, f=330),
                        np.zeros(SR // 5, np.float32), tone(1.0)])
    w = tone(2.0, f=500)  # 2 s replacement for a 1 s slot -> timeline grows ~1 s
    out, log = splice(y, SR, [{"start": 1.2, "end": 2.2, "wav": w, "x": 1}])
    assert log[0]["x"] == 1 and "wav" not in log[0]
    assert 0.8 < (len(out) - len(y)) / SR < 1.1  # 2 s into a ~1.1 s slot, minus crossfades
    assert log[0]["new_end"] - log[0]["new_start"] > 1.9
    assert np.max(np.abs(out)) <= 1.0


def test_trim_and_beep():
    w = np.concatenate([np.zeros(SR), tone(0.5), np.zeros(SR)])
    assert len(trim_silence(w, SR)) < len(w) * 0.5
    b = soft_beep(1.0, SR)
    assert len(b) == SR and np.max(np.abs(b)) <= 0.051
