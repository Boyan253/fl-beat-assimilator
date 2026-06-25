"""verify_transcription.py — measure how close the transcribed MIDI is to the real stems.

For each stem it compares the MIDI (re-synthesized to audio) against the original audio:
  - CHROMA similarity : do the same pitches/notes sound at the same times? (0-100%)
  - ONSET F-measure   : do hits/notes start at the same moments? (50 ms tolerance)
  - PITCH accuracy     : (melodic only) fraction of voiced frames where MIDI pitch
                         matches the audio's detected pitch within 1 semitone.

  python verify_transcription.py <stems_dir> <midi_dir>
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, librosa, pretty_midi

STEMS = sys.argv[1]
MDIR  = sys.argv[2]
SR = 22050

PAIRS = [  # (audio stem, midi, is_melodic)
    ("Melody.wav", "Melody.mid", True),
    ("808.wav",    "808.mid",    True),
    ("Drums.wav",  "Drums.mid",  False),
]


def onset_f(a_on, m_on, tol=0.05):
    a = sorted(a_on); m = sorted(m_on); used = [False] * len(a); hit = 0
    for t in m:
        for i, at in enumerate(a):
            if not used[i] and abs(at - t) <= tol:
                used[i] = True; hit += 1; break
    P = hit / len(m) if m else 0.0
    R = hit / len(a) if a else 0.0
    F = 2 * P * R / (P + R) if (P + R) else 0.0
    return P, R, F


def chroma_sim(y_audio, y_midi):
    n = min(len(y_audio), len(y_midi)); y_audio, y_midi = y_audio[:n], y_midi[:n]
    ca = librosa.feature.chroma_cqt(y=y_audio, sr=SR)
    cm = librosa.feature.chroma_cqt(y=y_midi,  sr=SR)
    f = min(ca.shape[1], cm.shape[1]); ca, cm = ca[:, :f], cm[:, :f]
    ca = ca / (np.linalg.norm(ca, axis=0, keepdims=True) + 1e-9)
    cm = cm / (np.linalg.norm(cm, axis=0, keepdims=True) + 1e-9)
    return float((ca * cm).sum(axis=0).mean())


def pitch_acc(y_audio, midi):
    f0, voiced, _ = librosa.pyin(y_audio, fmin=55, fmax=2000, sr=SR)
    times = librosa.times_like(f0, sr=SR)
    # piano-roll of the MIDI sampled at the same frame times
    ok = tot = 0
    for t, f, v in zip(times, f0, voiced):
        if not v or not np.isfinite(f):
            continue
        tot += 1
        a_midi_pitch = librosa.hz_to_midi(f)
        # active MIDI notes at time t -> nearest pitch
        active = [n.pitch for inst in midi.instruments for n in inst.notes
                  if n.start <= t <= n.end]
        if active and min(abs(p - a_midi_pitch) for p in active) <= 1.0:
            ok += 1
    return ok / tot if tot else 0.0


print(f"{'stem':<12}{'chroma':>9}{'onsetF':>9}{'  (P/R)':>14}{'pitch':>9}", flush=True)
print("-" * 55, flush=True)
for stem, mid, melodic in PAIRS:
    ap = os.path.join(STEMS, stem); mp = os.path.join(MDIR, mid)
    if not (os.path.exists(ap) and os.path.exists(mp)):
        print(f"{stem:<12} (missing)", flush=True); continue
    y, _ = librosa.load(ap, sr=SR, mono=True)
    midi = pretty_midi.PrettyMIDI(mp)
    ym = midi.synthesize(fs=SR)  # sine render of the MIDI

    def clean(x):
        x = np.nan_to_num(np.asarray(x, dtype="float32"), nan=0.0, posinf=0.0, neginf=0.0)
        p = np.abs(x).max()
        return (x / p) if p > 0 else x
    y, ym = clean(y), clean(ym)
    cs = chroma_sim(y, ym)
    a_on = librosa.onset.onset_detect(y=y, sr=SR, units="time", backtrack=True)
    m_on = [n.start for inst in midi.instruments for n in inst.notes]
    P, R, F = onset_f(a_on, m_on)
    pa = pitch_acc(y, midi) if melodic else None
    pstr = f"{pa*100:7.1f}%" if pa is not None else "     - "
    print(f"{stem:<12}{cs*100:8.1f}%{F*100:8.1f}%   {P*100:4.0f}/{R*100:<4.0f}{pstr}", flush=True)
print("-" * 55, flush=True)
print("chroma=note/pitch content match | onsetF=timing match | pitch=melodic pitch within 1 semitone", flush=True)
