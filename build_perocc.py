"""PER-OCCURRENCE real-audio engine (always-faithful method).
Instead of one re-synthesized slice per pitch, every note plays the ACTUAL recorded
audio chunk from that moment (contiguous slices of the stem) -> faithful on ANY source
(distorted guitar included), because it never depends on transcription being *right*,
only on the onset *timing*. Pitch labels come from the transcription (best-effort, for
editing); round-robin handles repeated pitches. Output = SFZ (load in sfizz) + MIDI.

  python build_perocc.py <stem.wav> <out_dir> <prefix>
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
MERGE_GAP = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0  # merge held same-pitch slices within this gap (s)

info = sf.info(STEM)
SR = info.samplerate
DUR = info.frames / SR

print("transcribing (for onset timing + pitch labels)...", flush=True)
_, _, notes = predict(STEM, ICASSP_2022_MODEL_PATH, onset_threshold=0.5, frame_threshold=0.3,
                      minimum_note_length=80, minimum_frequency=80, maximum_frequency=2000)
ev = sorted([(float(n[0]), float(n[1]), int(n[2]), float(n[3])) for n in notes], key=lambda x: x[0])

# thin near-simultaneous detections (harmonic + fundamental at the same instant = doubling).
thinned = []
for s, e, p, a in ev:
    if thinned and s - thinned[-1][0] < 0.04:
        if a > thinned[-1][3]:
            thinned[-1] = (s, e, p, a)
    else:
        thinned.append((s, e, p, a))

# merge consecutive SAME-pitch slices within MERGE_GAP of each other (a HELD note the transcriber
# re-triggered) into ONE long chunk -> sustained notes keep their body instead of being chopped
# into dipping slivers (which makes them sound thinner/quieter when exposed without the kick).
clean = []
prev_s = prev_p = -1
for s, e, p, a in thinned:
    if clean and p == prev_p and (s - prev_s) < MERGE_GAP:
        clean[-1] = (clean[-1][0], max(clean[-1][1], e), p, max(clean[-1][3], a))
    else:
        clean.append((s, e, p, a))
    prev_s, prev_p = s, p

# snap each chunk boundary to a zero-crossing so chunks start/end at amplitude 0 -> no pops.
# boundary[i] is shared as end of chunk i-1 and start of chunk i, keeping them contiguous.
ya, _sr = sf.read(STEM, always_2d=False)
mono = ya.mean(axis=1) if getattr(ya, "ndim", 1) > 1 else ya
zc = np.where(np.diff(np.signbit(mono)))[0]


def snap(samp):
    return int(zc[np.argmin(np.abs(zc - samp))]) if len(zc) else int(samp)


onset_samp = [snap(int(s * SR)) for (s, e, p, a) in clean]
end_samp = onset_samp[1:] + [int(DUR * SR)]
occ = [(onset_samp[i], end_samp[i], clean[i][2]) for i in range(len(clean))]

stem_abs = os.path.abspath(STEM).replace("\\", "/")
# VELOCITY-LAYER selection (RESTART-SAFE). Round-robin (seq_position) keeps a global cycle
# counter that does NOT reset on transport stop -> after a pause+restart the chunks play out
# of order ("sounds bad and different"). Instead, give every occurrence a UNIQUE velocity on
# its key, so note-P-at-velocity-N ALWAYS plays the same chunk no matter where playback starts.
# amp_veltrack=0 -> velocity only SELECTS the chunk, it never changes the volume.
lines = ["// per-occurrence real-audio engine — velocity-layer selection (deterministic / restart-safe)",
         "<global> ampeg_attack=0.006 ampeg_release=0.12 amp_veltrack=0"]
used = defaultdict(set)   # key -> velocities already taken
slots = []                # (key, vel) per occurrence, parallel to occ / clean
for off, end, p in occ:
    key = max(0, min(127, p))
    if len(used[key]) >= 127:                      # pitch overflow (>127 hits): spill to nearest free key
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
# silent t=0 anchor: key 0 has no region -> makes no sound, but forces the clip's origin to 0
# so a late-entering layer (e.g. bass first hits at ~5.4s) stays time-locked to the melody
# instead of FL stripping its leading silence and sliding it out of alignment.
inst.notes.append(pretty_midi.Note(velocity=1, pitch=0, start=0.0, end=0.02))
for i, (s, e, p, a) in enumerate(clean):
    nxt = clean[i + 1][0] if i + 1 < len(clean) else DUR
    key, v = slots[i]
    inst.notes.append(pretty_midi.Note(velocity=v, pitch=key, start=s, end=max(s + 0.05, nxt)))
pm.instruments.append(inst); pm.write(os.path.join(OUTDIR, PREFIX + ".mid"))

spilled = sum(1 for i, (k, v) in enumerate(slots) if k != max(0, min(127, occ[i][2])))
maxper = max((len(s) for s in used.values()), default=0)
print("velocity-layer map: %d occurrences, max %d on one key, %d spilled to neighbour keys" % (len(slots), maxper, spilled))

print("occurrences: %d  (raw %d -> thinned %d)" % (len(occ), len(ev), len(clean)))
print("SFZ:", sfz)
print("MID:", os.path.join(OUTDIR, PREFIX + ".mid"))
print("NOTE: load the .sfz in sfizz (free SFZ VST) — references the stem:", stem_abs)
