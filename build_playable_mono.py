"""PLAYABLE (monophonic) — for single-note leads/bass, build a clean PLAYABLE instrument plus
a companion MIDI that replicates the part with HIGH accuracy. Uses librosa.pyin (a dedicated
monophonic pitch tracker) instead of the polyphonic Basic Pitch, which is far more accurate
on one-note-at-a-time parts -> the clean-instrument replica can exceed 90%.
  python build_playable_mono.py <stem.wav> <out_dir> <prefix> [fmin=80] [fmax=1200]
"""
import os, sys, warnings
from collections import defaultdict
warnings.filterwarnings("ignore")
import numpy as np, soundfile as sf, librosa, pretty_midi

STEM = sys.argv[1]; OUTDIR = sys.argv[2]; PREFIX = sys.argv[3]
FMIN = float(sys.argv[4]) if len(sys.argv) > 4 else 80.0
FMAX = float(sys.argv[5]) if len(sys.argv) > 5 else 1200.0
SAMPDIR = os.path.join(OUTDIR, PREFIX + "_samples"); os.makedirs(SAMPDIR, exist_ok=True)
MAXLEN = 2.0

y, SR = sf.read(STEM, always_2d=False)
if getattr(y, "ndim", 1) > 1:
    y = y.mean(axis=1)
y = y.astype("float32"); DUR = len(y) / SR

print("onsets + pyin monophonic pitch tracking...", flush=True)
on = [float(t) for t in librosa.onset.onset_detect(y=y, sr=SR, backtrack=True, units="time") if t < DUR - 0.02]
if not on or on[0] > 0.05:
    on = [0.0] + on
f0, _, _ = librosa.pyin(y, fmin=FMIN, fmax=FMAX, sr=SR)
ftimes = librosa.times_like(f0, sr=SR)

zc = np.where(np.diff(np.signbit(y)))[0]


def snap(s):
    return int(zc[np.argmin(np.abs(zc - s))]) if len(zc) else int(s)


def pitch_for(a, b):
    seg = f0[(ftimes >= a) & (ftimes < b)]
    seg = seg[np.isfinite(seg)]
    return int(round(librosa.hz_to_midi(np.median(seg)))) if len(seg) >= 2 else None


notes = []
for i, t in enumerate(on):
    b = on[i + 1] if i + 1 < len(on) else DUR
    p = pitch_for(t, b)
    if p is None or not (0 <= p <= 127):
        continue
    seg = y[int(t * SR):int(b * SR)]
    amp = float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0
    notes.append((t, b, p, amp))

bypitch = defaultdict(list)
for t, b, p, amp in notes:
    bypitch[p].append((t, b))
samples = {}
for p, occs in bypitch.items():
    t, b = max(occs, key=lambda o: o[1] - o[0])
    a0 = snap(int(t * SR)); a1 = snap(int(min(b + 0.15, t + MAXLEN, DUR) * SR))
    seg = y[a0:a1].copy()
    if len(seg) < int(0.05 * SR):
        continue
    fi = min(int(0.004 * SR), len(seg) // 4); fo = min(int(0.04 * SR), len(seg) // 4)
    seg[:fi] *= np.linspace(0, 1, fi); seg[-fo:] *= np.linspace(1, 0, fo)
    fn = os.path.join(SAMPDIR, "p%03d.wav" % p); sf.write(fn, seg, SR); samples[p] = fn.replace("\\", "/")

pitches = sorted(samples)
lines = ["// PLAYABLE (pyin mono) — velocity = loudness, full keyboard filled",
         "<global> ampeg_attack=0.002 ampeg_release=0.2 loop_mode=no_loop"]
for i, p in enumerate(pitches):
    lo = 0 if i == 0 else (pitches[i - 1] + p) // 2 + 1
    hi = 127 if i == len(pitches) - 1 else (pitches[i + 1] + p) // 2
    lines.append("<region> sample=%s lokey=%d hikey=%d pitch_keycenter=%d" % (samples[p], lo, hi, p))
open(os.path.join(OUTDIR, PREFIX + "_playable.sfz"), "w").write("\n".join(lines))

amax = max([amp for t, b, p, amp in notes] or [1.0])
pm = pretty_midi.PrettyMIDI(); inst = pretty_midi.Instrument(program=0)
inst.notes.append(pretty_midi.Note(velocity=1, pitch=0, start=0.0, end=0.02))
for t, b, p, amp in notes:
    if p not in samples:
        continue
    vel = max(14, min(127, int(round(amp / amax * 127))))
    inst.notes.append(pretty_midi.Note(velocity=vel, pitch=p, start=t, end=max(t + 0.06, b)))
pm.instruments.append(inst); pm.write(os.path.join(OUTDIR, PREFIX + "_playable.mid"))
print("PLAYABLE-mono: notes=%d  pitches=%d  -> %s_playable.sfz/.mid" % (len(notes), len(pitches), PREFIX))
