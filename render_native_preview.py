"""Render GRIM melody MIDI through the 16 REAL per-pitch samples at NATIVE pitch
(keytrack OFF, one-shot) -- exactly how a DirectWave-with-keytracking-off would sound.
This is the 1:1 test we can listen to BEFORE touching FL."""
import os, glob, re
import numpy as np
import soundfile as sf

SR = 44100
MS_DIR = "/mnt/d/flbeat/data/generated/GRIM_multisample"
MIDI   = "/mnt/d/flbeat/data/generated/GRIM_multisample/GRIM_mel.mid"
OUT    = "/mnt/d/flbeat/data/generated/GRIM_PREVIEW_native.wav"

NAMES = {'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}
def name_to_midi(s):
    m = re.match(r'^([A-G])(#?)(-?\d+)$', s)
    if not m: return None
    base = NAMES[m.group(1)] + (1 if m.group(2)=='#' else 0)
    octv = int(m.group(3))
    return base + (octv+1)*12

# --- load the per-pitch real samples, key = midi number ---
samples = {}
for wav in glob.glob(os.path.join(MS_DIR, "GRIM_mel_*.wav")):
    base = os.path.splitext(os.path.basename(wav))[0]      # GRIM_mel_A#5
    if base.endswith("_preview"): continue
    note = base.replace("GRIM_mel_", "")
    midi = name_to_midi(note)
    if midi is None: continue
    audio, sr = sf.read(wav, always_2d=True)
    if sr != SR:
        # simple linear resample fallback (shouldn't trigger; samples are 44.1k)
        idx = np.linspace(0, len(audio)-1, int(len(audio)*SR/sr))
        audio = np.stack([np.interp(idx, np.arange(len(audio)), audio[:,c]) for c in range(audio.shape[1])], axis=1)
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)               # mono -> stereo
    samples[midi] = audio.astype(np.float32)
print(f"loaded {len(samples)} samples: {sorted(samples.keys())}")

# --- read MIDI notes ---
import pretty_midi
pm = pretty_midi.PrettyMIDI(MIDI)
notes = []
for inst in pm.instruments:
    for n in inst.notes:
        notes.append((n.pitch, n.start, n.velocity))
notes.sort(key=lambda x: x[1])
print(f"{len(notes)} notes, span {notes[0][1]:.2f}..{notes[-1][1]:.2f}s")

# --- mix: place each note's REAL sample at its onset, velocity-scaled (native pitch) ---
tail = 1.0
total = int((max(n[1] for n in notes) + 3.0 + tail) * SR)
buf = np.zeros((total, 2), dtype=np.float32)
missing = set()
for pitch, start, vel in notes:
    s = samples.get(pitch)
    if s is None:
        missing.add(pitch); continue
    pos = int(start * SR)
    end = min(pos + len(s), total)
    gain = (vel / 127.0)
    buf[pos:end] += s[:end-pos] * gain
if missing:
    print(f"!! MIDI pitches with NO sample (skipped): {sorted(missing)}")

# normalize to -1 dBFS
peak = np.max(np.abs(buf)) or 1.0
buf = buf / peak * 0.89
sf.write(OUT, buf, SR)
print(f"WROTE {OUT}  ({total/SR:.1f}s)")
