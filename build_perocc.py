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

info = sf.info(STEM)
SR = info.samplerate
DUR = info.frames / SR

print("transcribing (for onset timing + pitch labels)...", flush=True)
_, _, notes = predict(STEM, ICASSP_2022_MODEL_PATH, onset_threshold=0.5, frame_threshold=0.3,
                      minimum_note_length=80, minimum_frequency=80, maximum_frequency=2000)
ev = sorted([(float(n[0]), float(n[1]), int(n[2]), float(n[3])) for n in notes], key=lambda x: x[0])

# thin near-simultaneous detections (a harmonic + fundamental at the same instant would
# play two copies of the same audio chunk = doubling). Keep the louder.
clean = []
for s, e, p, a in ev:
    if clean and s - clean[-1][0] < 0.04:
        if a > clean[-1][3]:
            clean[-1] = (s, e, p, a)
    else:
        clean.append((s, e, p, a))

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

totals = defaultdict(int)
for off, end, p in occ:
    totals[p] += 1
seqpos = defaultdict(int)

stem_abs = os.path.abspath(STEM).replace("\\", "/")
lines = ["// per-occurrence real-audio engine — each region plays a real chunk of the stem",
         # no one_shot: note plays for its (contiguous) duration, then the release fade
         # crossfades into the next note's attack -> seamless transitions, no cut-off pop
         "<global> ampeg_attack=0.006 ampeg_release=0.09"]
for off, end, p in occ:
    seqpos[p] += 1
    lines.append("<region> sample=%s lokey=%d hikey=%d pitch_keycenter=%d offset=%d end=%d seq_length=%d seq_position=%d"
                 % (stem_abs, p, p, p, off, end, totals[p], seqpos[p]))
sfz = os.path.join(OUTDIR, PREFIX + ".sfz")
open(sfz, "w").write("\n".join(lines))

pm = pretty_midi.PrettyMIDI(); inst = pretty_midi.Instrument(program=0)
for i, (s, e, p, a) in enumerate(clean):
    nxt = clean[i + 1][0] if i + 1 < len(clean) else DUR
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=max(0, min(127, p)), start=s, end=max(s + 0.05, nxt)))
pm.instruments.append(inst); pm.write(os.path.join(OUTDIR, PREFIX + ".mid"))

print("occurrences: %d  (raw %d -> thinned %d)" % (len(occ), len(ev), len(clean)))
print("SFZ:", sfz)
print("MID:", os.path.join(OUTDIR, PREFIX + ".mid"))
print("NOTE: load the .sfz in sfizz (free SFZ VST) — references the stem:", stem_abs)
