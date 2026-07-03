"""slice_melody.py — faithful slice playback that LOOKS like a real melody.
Slice GRIM's real melody audio at onsets (sounds 1-to-1). Label each slice with its real pitch
(CREPE) and place the MIDI notes at those pitches -> the piano roll looks like a real melody, not a
flat line or staircase. The instrument uses round-robin (each note = next slice in order) with
pitchKeyTrack=0 (the MIDI pitch does NOT transpose the slice) -> pitch is just for the look, sound
stays the real recording.
"""
import os, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np, soundfile as sf, librosa, pretty_midi, torch, torchcrepe
import scipy.signal as ss

STEM = "/mnt/d/flbeat/data/generated/GRIM_melody_6s_CLEAN.wav"
OUT  = "/mnt/d/flbeat/data/SEND/GRIMPack"
SAMP = os.path.join(OUT, "Samples")
MIDIOUT = "/mnt/d/flbeat/data/generated/GRIM_melody_sliced.mid"
os.makedirs(SAMP, exist_ok=True)

y, sr = librosa.load(STEM, sr=44100, mono=True)
on = list(librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=True))
if not on or on[0] > 0.05:
    on = [0.0] + on
on.append(len(y) / sr)

# CREPE pitch track -> label each slice with its real pitch (for the melody LOOK only)
y16 = librosa.resample(y, orig_sr=sr, target_sr=16000)
pit, per = torchcrepe.predict(torch.tensor(y16, dtype=torch.float32)[None], 16000, hop_length=160,
                              fmin=110, fmax=1500, model="full", decoder=torchcrepe.decode.viterbi,
                              return_periodicity=True, batch_size=512, device="cpu", pad=True)
f0 = ss.medfilt(pit[0].numpy(), 5)
per = per[0].numpy()
tc = np.arange(len(f0)) * 160 / 16000.0


def slice_pitch(s, e, fallback):
    m = (tc >= s) & (tc < e) & (per > 0.5) & (f0 > 110) & (f0 < 1500)
    return int(round(69 + 12 * np.log2(np.median(f0[m]) / 440.0))) if m.sum() >= 2 else fallback


for f in glob.glob(os.path.join(SAMP, "GRIMslice_*.wav")):
    os.remove(f)

rows = []
pm = pretty_midi.PrettyMIDI(initial_tempo=122.0)
inst = pretty_midi.Instrument(program=0, name="GRIM Melody")
k, last = 0, 60
for i in range(len(on) - 1):
    s, e = on[i], on[i + 1]
    if e - s < 0.04:
        continue
    p = slice_pitch(s, e, last); last = p
    a = int(s * sr); b = min(len(y), int((e + 0.02) * sr))
    seg = y[a:b].copy()
    fo = min(int(0.01 * sr), len(seg) // 4)
    if fo:
        seg[-fo:] *= np.linspace(1, 0, fo)
    seg = seg / (np.max(np.abs(seg)) + 1e-9) * 0.97
    nm = f"GRIMslice_{k:03d}.wav"
    sf.write(os.path.join(SAMP, nm), seg.astype(np.float32), sr)
    rows.append(nm)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=p, start=s, end=e))   # real pitch = melody look
    k += 1

pm.instruments.append(inst)
pm.write(MIDIOUT)

# round-robin advances to the next slice on each note; pitchKeyTrack=0 -> the note's pitch does NOT
# transpose the slice (it's just for the look). LOOKS like a real melody, PLAYS the real slices 1-to-1.
L = ['<?xml version="1.0" encoding="UTF-8"?>', '<DecentSampler minVersion="1.0.0">',
     '  <groups seqMode="round_robin" attack="0.0" decay="0.0" sustain="1.0" release="0.05">']
for nm in rows:
    L.append(f'    <group><sample path="Samples/{nm}" loNote="0" hiNote="127" rootNote="60" pitchKeyTrack="0" /></group>')
L += ['  </groups>', '</DecentSampler>']
open(os.path.join(OUT, "GRIMMelodySliced.dspreset"), "w", encoding="utf-8").write("\n".join(L))
print(f"sliced melody-look: {len(rows)} slices at real pitches -> {MIDIOUT}", flush=True)
