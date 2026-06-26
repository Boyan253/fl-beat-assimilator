"""extract_oneshots.py — pull ONE clean playable sample of each instrument from MIRAGE's stems.
These one-shots have MIRAGE's real timbre but are played by the EDITABLE MIDI, so you get the
song's sound AND full remix freedom. Tonal samples are named with their root note.

  python extract_oneshots.py <out_dir>
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, librosa, soundfile as sf, pretty_midi

OUT = sys.argv[1] if len(sys.argv) > 1 else "/mnt/d/flbeat/MIRAGE_FL/4_Instruments (remixable)"
os.makedirs(OUT, exist_ok=True)
SR = 44100
M = "/mnt/d/flbeat/data/MIRAGE"

# (source wav, base name, max length s, tonal?, fmin, fmax)
JOBS = [
    (f"{M}/Stems/Melody.wav",       "Cowbell", 0.55, True,  150, 2000),
    (f"{M}/Stems/808.wav",          "808",     0.75, True,  28,  300),
    (f"{M}/DrumSep/Drums_(kick)_MDX23C-DrumSep-aufr33-jarredou.flac",  "Kick",  0.45, False, 0, 0),
    (f"{M}/DrumSep/Drums_(snare)_MDX23C-DrumSep-aufr33-jarredou.flac", "Snare", 0.35, False, 0, 0),
    (f"{M}/DrumSep/Drums_(hh)_MDX23C-DrumSep-aufr33-jarredou.flac",    "Hat",   0.18, False, 0, 0),
]


def pick_oneshot(y, maxlen):
    """Choose the cleanest isolated hit: strong peak with room before the next onset."""
    on = librosa.onset.onset_detect(y=y, sr=SR, units="time", backtrack=True)
    if len(on) == 0:
        return 0, min(len(y), int(maxlen * SR))
    best, best_score = None, -1
    for i, t in enumerate(on):
        nxt = on[i + 1] if i + 1 < len(on) else (len(y) / SR)
        gap = nxt - t
        s0 = int(t * SR); seg = y[s0:s0 + int(0.05 * SR)]
        peak = float(np.max(np.abs(seg))) if len(seg) else 0
        score = peak * min(gap, maxlen)        # loud AND isolated
        if score > best_score:
            best_score, best = score, (s0, gap)
    s0, gap = best
    length = int(min(maxlen, max(0.1, gap * 0.95)) * SR)
    return s0, length


def zero_snap(y, s):
    zc = np.where(np.diff(np.signbit(y)))[0]
    return int(zc[np.argmin(np.abs(zc - s))]) if len(zc) else s


results = []
for src, name, maxlen, tonal, fmin, fmax in JOBS:
    if not os.path.exists(src):
        results.append(f"{name}: MISSING"); continue
    y, _ = librosa.load(src, sr=SR, mono=True)
    s0, length = pick_oneshot(y, maxlen)
    s0 = zero_snap(y, s0)
    clip = y[s0:s0 + length].astype("float32")
    # fade in 3ms / out 20ms -> no clicks
    fi = min(int(0.003 * SR), len(clip) // 4); fo = min(int(0.02 * SR), len(clip) // 4)
    if fi: clip[:fi] *= np.linspace(0, 1, fi)
    if fo: clip[-fo:] *= np.linspace(1, 0, fo)
    pk = np.max(np.abs(clip))
    if pk > 0: clip = clip * (0.97 / pk)        # normalise
    root = ""
    if tonal:
        f0 = librosa.yin(clip, fmin=fmin, fmax=fmax, sr=SR)
        f0 = f0[np.isfinite(f0)]
        if len(f0):
            note = int(round(librosa.hz_to_midi(np.median(f0))))
            root = "_" + pretty_midi.note_number_to_name(note)
    fn = os.path.join(OUT, f"{name}{root}.wav")
    sf.write(fn, clip, SR)
    results.append(f"{name}{root}  ({length/SR:.2f}s)")

print("one-shots ->", OUT)
for r in results:
    print("  " + r, flush=True)
