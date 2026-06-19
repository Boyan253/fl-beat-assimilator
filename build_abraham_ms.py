"""Build a REAL-PITCH multisample from abraham's melody stem: each detected pitch
gets a real recorded slice mapped to that key (not a staircase). Load the .sfz into
DirectWave/sfizz (one channel) + the .mid -> the piano roll shows abraham's actual
melody as movable notes, playing the real sound."""
import os, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np, librosa, soundfile as sf, pretty_midi

STEM = r"D:\flmidi\out\abraham\stems\other.wav"
OUTDIR = r"D:\flmidi\out\abraham\slices\melody"
MID = r"D:\flmidi\out\abraham\melody_real.mid"
SR = 44100
os.makedirs(OUTDIR, exist_ok=True)
for old in glob.glob(os.path.join(OUTDIR, "p*.wav")):
    os.remove(old)

y22, _ = librosa.load(STEM, sr=22050, mono=True)
yhi, _ = librosa.load(STEM, sr=SR, mono=True)
f0, vf, vp = librosa.pyin(y22, fmin=110, fmax=1200, sr=22050, frame_length=4096)
t = librosa.times_like(f0, sr=22050)

notes, cur, start, probs = [], None, 0.0, []
for i in range(len(f0)):
    m = int(round(librosa.hz_to_midi(f0[i]))) if (vf[i] and not np.isnan(f0[i])) else None
    if m != cur:
        if cur is not None and (t[i] - start) > 0.08:
            notes.append((start, t[i], cur, float(np.mean(probs)) if probs else 0.0))
        cur, start, probs = m, t[i], []
    if vf[i] and not np.isnan(vp[i]):
        probs.append(float(vp[i]))

# clearest occurrence per pitch -> the sampled slice
best = {}
for (s, e, p, prob) in notes:
    score = (prob + 0.1) * min(e - s + 0.05, 0.4)
    if p not in best or score > best[p][0]:
        best[p] = (score, s)

pitches = sorted(best)
regions = ["// abraham melody — each region = real slice at its TRUE pitch (movable melody, not a staircase)"]
for idx, p in enumerate(pitches):
    score, s = best[p]
    seg = yhi[int(s * SR):int((s + 1.1) * SR)]
    if len(seg) < 256:
        continue
    seg = seg / (np.max(np.abs(seg)) or 1) * 0.97
    fo = int(0.25 * SR)
    if len(seg) > fo:
        seg[-fo:] *= np.linspace(1, 0, fo)
    sf.write(os.path.join(OUTDIR, "p%03d.wav" % p), seg.astype("float32"), SR)
    lo = 0 if idx == 0 else (pitches[idx - 1] + p) // 2 + 1
    hi = 127 if idx == len(pitches) - 1 else (p + pitches[idx + 1]) // 2
    regions.append("<region> sample=p%03d.wav pitch_keycenter=%d lokey=%d hikey=%d loop_mode=no_loop ampeg_release=0.25"
                   % (p, p, lo, hi))

with open(os.path.join(OUTDIR, "abraham_melody.sfz"), "w") as f:
    f.write("\n".join(regions))

pm = pretty_midi.PrettyMIDI(); inst = pretty_midi.Instrument(program=0, name="abraham melody")
for (s, e, p, prob) in notes:
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=p, start=s, end=e))
pm.instruments.append(inst); pm.write(MID)
print("melody: %d notes, %d real pitches" % (len(notes), len(pitches)))
print("SFZ:", os.path.join(OUTDIR, "abraham_melody.sfz"))
print("MID:", MID)
