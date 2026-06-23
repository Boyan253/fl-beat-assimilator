"""Re-transcribe abraham's melody with Basic Pitch (polyphonic, accurate) and
re-cut one real slice per distinct pitch, so the MIDI actually matches the song
and the SF2 covers exactly those pitches (no shifting)."""
import os, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np, librosa, soundfile as sf, pretty_midi
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH

STEM = r"D:\flmidi\out\abraham\stems\other.wav"
SLICEDIR = r"D:\flmidi\out\abraham\slices\melody"
MID = r"D:\flmidi\out\abraham\melody_bp.mid"
SR = 44100
os.makedirs(SLICEDIR, exist_ok=True)

print("transcribing with Basic Pitch...")
_, midi_data, notes = predict(STEM, ICASSP_2022_MODEL_PATH,
                              onset_threshold=0.6, frame_threshold=0.4,
                              minimum_note_length=120,
                              minimum_frequency=110, maximum_frequency=1300)
# the strict 284-note set was "almost perfect"; only gently bridge the small gaps:
# extend each note to the next onset, but cap so a note before a long rest doesn't sprawl.
evs = sorted([(float(n[0]), float(n[1]), int(n[2])) for n in notes], key=lambda x: x[0])
for i in range(len(evs) - 1):
    s, e, p = evs[i]
    gap_end = evs[i + 1][0]
    evs[i] = (s, min(max(e, gap_end), s + 1.3), p)   # legato cap 1.3s (fill the longer rests)
pm = pretty_midi.PrettyMIDI(); inst = pretty_midi.Instrument(program=0, name="abraham melody")
for s, e, p in evs:
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=p, start=s, end=max(e, s + 0.05)))
pm.instruments.append(inst); pm.write(MID)
print("notes:", len(evs))

# note_events: (start, end, pitch, amplitude, bends)
best = {}
for n in notes:
    s, e, p, amp = float(n[0]), float(n[1]), int(n[2]), float(n[3])
    score = (e - s) * (amp + 0.1)
    if p not in best or score > best[p][0]:
        best[p] = (score, s)

yhi, _ = librosa.load(STEM, sr=SR, mono=True)
for old in glob.glob(os.path.join(SLICEDIR, "p*.wav")):
    os.remove(old)
for p, (score, s) in best.items():
    seg = yhi[int(s * SR):int((s + 1.5) * SR)]
    if len(seg) < 256:
        continue
    seg = seg / (np.max(np.abs(seg)) or 1) * 0.97
    fo = int(0.40 * SR)
    if len(seg) > fo:
        seg[-fo:] *= np.linspace(1, 0, fo)
    sf.write(os.path.join(SLICEDIR, "p%03d.wav" % p), seg.astype("float32"), SR)
print("distinct pitches sampled:", len(best))
