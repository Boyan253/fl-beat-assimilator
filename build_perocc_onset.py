"""PER-OCCURRENCE via ENERGY-ONSET detection — for bass / 808 / percussive stems where the
melodic transcriber (build_perocc.py / Basic Pitch) can't find sub-bass onsets and fires
chunks at the wrong times. Uses librosa energy onsets for TIMING (robust on sub-bass),
estimates a pitch per chunk only for the MIDI label, then the SAME real-audio engine:
velocity-layer selection (restart-safe) + silent t=0 anchor (keeps layers time-locked).
  python build_perocc_onset.py <stem.wav> <out_dir> <prefix> [default_pitch=36]
"""
import os, sys, warnings
from collections import defaultdict
warnings.filterwarnings("ignore")
import numpy as np, soundfile as sf, librosa, pretty_midi

STEM = sys.argv[1]
OUTDIR = sys.argv[2]
PREFIX = sys.argv[3]
DEFP = int(sys.argv[4]) if len(sys.argv) > 4 else 36

info = sf.info(STEM)
SR = info.samplerate
DUR = info.frames / SR

y, _ = librosa.load(STEM, sr=SR, mono=True)
print("detecting energy onsets (sub-bass safe)...", flush=True)
oframes = librosa.onset.onset_detect(y=y, sr=SR, backtrack=True, units="frames")
otimes = [float(t) for t in librosa.frames_to_time(oframes, sr=SR) if t < DUR - 0.02]
if not otimes or otimes[0] > 0.05:
    otimes = [0.0] + otimes


def est_pitch(a, b):
    seg = y[int(a * SR):int(b * SR)]
    if len(seg) < 512:
        return DEFP
    f0 = librosa.yin(seg, fmin=30, fmax=400, sr=SR)
    f0 = f0[np.isfinite(f0)]
    return int(round(librosa.hz_to_midi(np.median(f0)))) if len(f0) else DEFP


zc = np.where(np.diff(np.signbit(y)))[0]


def snap(samp):
    return int(zc[np.argmin(np.abs(zc - samp))]) if len(zc) else int(samp)


onset_samp = [snap(int(t * SR)) for t in otimes]
end_samp = onset_samp[1:] + [int(DUR * SR)]
occ = []
for i in range(len(otimes)):
    b = otimes[i + 1] if i + 1 < len(otimes) else DUR
    p = max(0, min(127, est_pitch(otimes[i], b)))
    occ.append((onset_samp[i], end_samp[i], p, otimes[i]))

stem_abs = os.path.abspath(STEM).replace("\\", "/")
lines = ["// per-occurrence (energy-onset) — velocity-layer selection, restart-safe",
         "<global> ampeg_attack=0.004 ampeg_release=0.15 amp_veltrack=0"]
used = defaultdict(set)
slots = []
for off, end, p, t in occ:
    key = max(0, min(127, p))
    if len(used[key]) >= 127:
        for d in range(1, 128):
            cand = [k for k in (key - d, key + d) if 0 <= k <= 127 and len(used[k]) < 127]
            if cand:
                key = cand[0]; break
    v = 1
    while v in used[key]:
        v += 1
    used[key].add(v); slots.append((key, v))
    lines.append("<region> sample=%s lokey=%d hikey=%d pitch_keycenter=%d lovel=%d hivel=%d offset=%d"
                 % (stem_abs, key, key, key, v, v, off))
sfz = os.path.join(OUTDIR, PREFIX + ".sfz")
open(sfz, "w").write("\n".join(lines))

pm = pretty_midi.PrettyMIDI(); inst = pretty_midi.Instrument(program=0)
inst.notes.append(pretty_midi.Note(velocity=1, pitch=0, start=0.0, end=0.02))  # silent t=0 anchor
for i, (off, end, p, t) in enumerate(occ):
    nxt = occ[i + 1][3] if i + 1 < len(occ) else DUR
    key, v = slots[i]
    inst.notes.append(pretty_midi.Note(velocity=v, pitch=key, start=t, end=max(t + 0.05, nxt)))
pm.instruments.append(inst); pm.write(os.path.join(OUTDIR, PREFIX + ".mid"))

maxper = max((len(s) for s in used.values()), default=0)
print("occurrences: %d  (energy onsets)" % len(occ))
print("velocity-layer map: %d occurrences, max %d on one key" % (len(occ), maxper))
print("SFZ:", sfz)
