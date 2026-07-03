"""build_multisample.py — turn a melody stem into a REAL multisampled instrument: one genuine
recorded note per pitch, mapped to its own key zone (like a Kontakt/SFZ library). Played by the
melody MIDI, every note is a different real recording with near-zero pitch-shift -> no "one sample
pitched around" tell, but keeps the source's exact timbre. Editable (plays by pitch).

  python build_multisample.py <melody_stem.wav> <out_dir> <NAME>
Outputs: <NAME>.sfz + per-note <NAME>_<note>.wav + <NAME>.mid (aligned) + <NAME>_preview.wav
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
import numpy as np, librosa, soundfile as sf, pretty_midi
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH

STEM = sys.argv[1]; OUT = sys.argv[2]; NAME = sys.argv[3]
os.makedirs(OUT, exist_ok=True)
SR = 44100
y, _ = librosa.load(STEM, sr=SR, mono=True)

# 1) transcribe -> note events (start, end, pitch, amplitude)
_, midi, notes = predict(STEM, ICASSP_2022_MODEL_PATH, onset_threshold=0.5, frame_threshold=0.3,
                         minimum_note_length=90, minimum_frequency=80, maximum_frequency=3000)
ev = [(float(n[0]), float(n[1]), int(n[2]), float(n[3])) for n in notes]
if not ev:
    print("no notes detected"); sys.exit(1)

# 2) for each unique pitch, sample the CLEANEST occurrence (long + loud + isolated)
by_pitch = {}
for s, e, p, a in ev:
    by_pitch.setdefault(p, []).append((s, e, a))

def zsnap(samp):
    zc = np.where(np.diff(np.signbit(y)))[0]
    return int(zc[np.argmin(np.abs(zc - samp))]) if len(zc) else int(samp)

bank = {}   # pitch -> wav filename
for p, occs in by_pitch.items():
    s, e, a = max(occs, key=lambda o: (o[1] - o[0]) * o[2])   # longest*loudest
    s0 = zsnap(int(s * SR))
    length = int(min(0.9, max(0.25, (e - s) + 0.15)) * SR)
    clip = y[s0:s0 + length].astype("float32")
    if len(clip) < 1000:
        continue
    fo = min(int(0.04 * SR), len(clip) // 4)
    clip[-fo:] *= np.linspace(1, 0, fo)
    pk = np.max(np.abs(clip))
    if pk > 0:
        clip = clip / pk * 0.95
    fn = f"{NAME}_{pretty_midi.note_number_to_name(p)}.wav"
    sf.write(os.path.join(OUT, fn), clip, SR)
    bank[p] = fn
pitches = sorted(bank)
print(f"sampled {len(pitches)} real notes: {[pretty_midi.note_number_to_name(p) for p in pitches]}", flush=True)

# 3) SFZ: each sample covers a key zone up to the midpoint with its neighbours
lines = [f"// {NAME} multisample — one real recorded note per pitch (no pitch-stretch tell)",
         "<global> ampeg_attack=0.005 ampeg_release=0.25 loop_mode=one_shot"]
for i, p in enumerate(pitches):
    lo = 0 if i == 0 else (pitches[i - 1] + p) // 2 + 1
    hi = 127 if i == len(pitches) - 1 else (p + pitches[i + 1]) // 2
    lines.append(f"<region> sample={bank[p]} lokey={lo} hikey={hi} pitch_keycenter={p}")
open(os.path.join(OUT, f"{NAME}.sfz"), "w").write("\n".join(lines))
midi.write(os.path.join(OUT, f"{NAME}.mid"))

# 4) preview render (proves the result): nearest sample per note, tiny pitch-shift
allnotes = [n for ins in midi.instruments for n in ins.notes]
total = max(n.end for n in allnotes) + 1.0
buf = np.zeros(int(total * SR) + SR, dtype="float32")
cache = {}
parr = np.array(pitches)
for n in allnotes:
    near = int(parr[np.argmin(np.abs(parr - n.pitch))])
    semis = n.pitch - near
    key = (near, semis)
    if key not in cache:
        s, _ = librosa.load(os.path.join(OUT, bank[near]), sr=SR, mono=True)
        cache[key] = librosa.effects.pitch_shift(s, sr=SR, n_steps=semis) if semis else s
    seg = cache[key]
    dur = max(int((n.end - n.start) * SR), int(0.12 * SR))
    seg = (seg[:dur] if len(seg) >= dur else np.pad(seg, (0, dur - len(seg)))) * (n.velocity / 110.0)
    i0 = int(n.start * SR)
    buf[i0:i0 + len(seg)] += seg
pk = np.max(np.abs(buf))
buf = buf / pk * 0.97 if pk else buf
sf.write(os.path.join(OUT, f"{NAME}_preview.wav"), buf, SR)
print(f"built multisample ({len(pitches)} zones) + preview -> {OUT}", flush=True)
