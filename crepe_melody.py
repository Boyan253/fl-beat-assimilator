"""crepe_melody.py — transcribe GRIM's melody with CREPE (monophonic pitch tracker).
basic_pitch is polyphonic and mangles a single melodic line; CREPE follows the actual
fundamental frequency, which is the right tool for a one-note-at-a-time cowbell melody.
Outputs a clean monophonic melody MIDI that should actually match the song.
  python crepe_melody.py [in.wav] [out.mid]
"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch, torchcrepe, librosa, pretty_midi
import scipy.signal as ss

IN  = sys.argv[1] if len(sys.argv) > 1 else "/mnt/d/flbeat/data/generated/GRIM_melody_6s_CLEAN.wav"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/mnt/d/flbeat/data/generated/GRIM_melody_crepe.mid"

y, _ = librosa.load(IN, sr=16000, mono=True)
audio = torch.tensor(y, dtype=torch.float32)[None]
HOP = 160                                                  # 10 ms frames
pitch, period = torchcrepe.predict(audio, 16000, hop_length=HOP, fmin=110, fmax=1500,
                                   model="full", decoder=torchcrepe.decode.viterbi,
                                   return_periodicity=True, batch_size=512, device="cpu", pad=True)
pitch = pitch[0].numpy()
period = period[0].numpy()

f0 = ss.medfilt(pitch, 5)                                  # de-jitter
voiced = (period > 0.5) & (f0 > 110) & (f0 < 1500)
midi = 69 + 12 * np.log2(np.maximum(f0, 1e-6) / 440.0)
mr = np.round(midi)
TP = HOP / 16000.0

notes, i, N = [], 0, len(mr)
while i < N:
    if not voiced[i]:
        i += 1; continue
    p = mr[i]; j = i
    while j < N and voiced[j] and abs(mr[j] - p) < 0.6:    # same note while pitch stable
        j += 1
    s, e = i * TP, j * TP
    if e - s >= 0.08 and 24 <= p <= 100:                   # min length, sane range
        notes.append((int(p), s, e))
    i = j

merged = []                                                # join same-pitch notes split by tiny gaps
for n in notes:
    if merged and merged[-1][0] == n[0] and n[1] - merged[-1][2] < 0.04:
        merged[-1] = (n[0], merged[-1][1], n[2])
    else:
        merged.append(n)

pm = pretty_midi.PrettyMIDI(initial_tempo=122.0)
inst = pretty_midi.Instrument(program=0, name="GRIM Melody")
for (p, s, e) in merged:
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=p, start=s, end=e))
pm.instruments.append(inst)
pm.write(OUT)
from collections import Counter
print(f"CREPE melody: {len(merged)} notes -> {OUT}", flush=True)
print("pitches:", sorted(Counter(pretty_midi.note_number_to_name(p) for p, _, _ in merged).items(),
                         key=lambda x: -x[1])[:14], flush=True)
