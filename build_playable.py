"""PLAYABLE multisample — turn a stem into a real performable instrument for CREATING beats.

The per-occurrence engine reconstructs the original from its own MIDI (not meant for hand
playing). This instead picks ONE clean sample per detected pitch, extracts it as its own
short WAV, and maps it chromatically WITH FILL so EVERY key plays (nearest sample
pitch-shifted) and velocity = LOUDNESS (normal). Load it in Sforzando and perform melodies
/ basslines by hand to build your own beats.
  python build_playable.py <stem.wav> <out_dir> <prefix>
"""
import os, sys, warnings
from collections import defaultdict
warnings.filterwarnings("ignore")
import numpy as np, soundfile as sf, pretty_midi
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH

STEM = sys.argv[1]
OUTDIR = sys.argv[2]
PREFIX = sys.argv[3]
SAMPDIR = os.path.join(OUTDIR, PREFIX + "_samples")
os.makedirs(SAMPDIR, exist_ok=True)
MAXLEN = 2.0  # cap each sample length (s)

y, SR = sf.read(STEM, always_2d=False)
if getattr(y, "ndim", 1) > 1:
    y = y.mean(axis=1)
y = y.astype("float32")
DUR = len(y) / SR

print("transcribing (to find one clean sample per pitch)...", flush=True)
_, _, notes = predict(STEM, ICASSP_2022_MODEL_PATH, onset_threshold=0.5, frame_threshold=0.3,
                      minimum_note_length=100, minimum_frequency=60, maximum_frequency=3000)
ev = [(float(n[0]), float(n[1]), int(n[2]), float(n[3])) for n in notes]

bypitch = defaultdict(list)
for s, e, p, a in ev:
    bypitch[p].append((s, e, a))

zc = np.where(np.diff(np.signbit(y)))[0]


def snap(samp):
    return int(zc[np.argmin(np.abs(zc - samp))]) if len(zc) else int(samp)


samples = {}
for p, occs in bypitch.items():
    s, e, a = max(occs, key=lambda o: o[1] - o[0])            # longest = cleanest sustained
    a0 = snap(int(s * SR))
    a1 = snap(int(min(e + 0.15, s + MAXLEN, DUR) * SR))       # small tail, capped
    seg = y[a0:a1].copy()
    if len(seg) < int(0.05 * SR):
        continue
    fi = min(int(0.004 * SR), len(seg) // 4)
    fo = min(int(0.04 * SR), len(seg) // 4)
    seg[:fi] *= np.linspace(0, 1, fi)
    seg[-fo:] *= np.linspace(1, 0, fo)
    fn = os.path.join(SAMPDIR, "p%03d.wav" % p)
    sf.write(fn, seg, SR)
    samples[p] = fn.replace("\\", "/")

pitches = sorted(samples)
lines = ["// PLAYABLE multisample — velocity = loudness; every key plays (nearest sample shifted)",
         "<global> ampeg_attack=0.002 ampeg_release=0.25 loop_mode=no_loop"]
for i, p in enumerate(pitches):
    lo = 0 if i == 0 else (pitches[i - 1] + p) // 2 + 1
    hi = 127 if i == len(pitches) - 1 else (pitches[i + 1] + p) // 2
    lines.append("<region> sample=%s lokey=%d hikey=%d pitch_keycenter=%d" % (samples[p], lo, hi, p))
sfz = os.path.join(OUTDIR, PREFIX + "_playable.sfz")
open(sfz, "w").write("\n".join(lines))
print("PLAYABLE: %d sampled pitches, full keyboard filled -> %s" % (len(pitches), sfz))
print("samples in:", SAMPDIR)

# companion MIDI: a TRUE musical transcription (real pitch / timing / loudness) that
# replicates the original beat when played THROUGH this playable instrument, and is fully
# editable as real notes. velocity = loudness (amplitude normalised to the velocity range).
pm = pretty_midi.PrettyMIDI(); inst = pretty_midi.Instrument(program=0)
inst.notes.append(pretty_midi.Note(velocity=1, pitch=0, start=0.0, end=0.02))   # silent t=0 anchor
amax = max([a for s, e, p, a in ev if p in samples] or [1.0])
for s, e, p, a in sorted(ev, key=lambda x: x[0]):
    if p not in samples:
        continue
    vel = max(14, min(127, int(round(a / amax * 127))))
    inst.notes.append(pretty_midi.Note(velocity=vel, pitch=p, start=s, end=max(s + 0.06, e)))
pm.instruments.append(inst); pm.write(os.path.join(OUTDIR, PREFIX + "_playable.mid"))
print("companion MIDI (replicates the beat through this instrument):",
      os.path.join(OUTDIR, PREFIX + "_playable.mid"))
