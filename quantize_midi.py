"""quantize_midi.py — snap transcribed MIDI to a tempo grid + embed BPM (so it syncs in FL).

  python quantize_midi.py <midi_dir> <reference_audio_for_bpm> [grid=16]
Quantizes every .mid in midi_dir to the nearest 1/grid note at the detected BPM and writes
the BPM into the file, so dropping them into FL (at that BPM) lines everything up.
"""
import os, sys, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np, librosa, pretty_midi

MDIR = sys.argv[1]
REF  = sys.argv[2]
GRID = int(sys.argv[3]) if len(sys.argv) > 3 else 16   # 16 = 16th notes

y, sr = librosa.load(REF, sr=22050, mono=True, duration=90)
bpm = float(np.atleast_1d(librosa.beat.beat_track(y=y, sr=sr)[0])[0])
bpm = round(bpm)
step = (60.0 / bpm) * (4.0 / GRID)     # seconds per 1/GRID note
print(f"detected BPM: {bpm}  | grid: 1/{GRID} ({step*1000:.0f} ms)", flush=True)

for mf in glob.glob(os.path.join(MDIR, "*.mid")):
    src = pretty_midi.PrettyMIDI(mf)
    out = pretty_midi.PrettyMIDI(initial_tempo=float(bpm))
    for inst in src.instruments:
        ni = pretty_midi.Instrument(program=inst.program, is_drum=inst.is_drum, name=inst.name)
        for n in inst.notes:
            qs = round(n.start / step) * step
            qe = round(n.end / step) * step
            if qe <= qs:
                qe = qs + step
            ni.notes.append(pretty_midi.Note(n.velocity, n.pitch, qs, qe))
        out.instruments.append(ni)
    out.write(mf)   # overwrite, now quantized + tempo-tagged
    print(f"  quantized {os.path.basename(mf)} ({sum(len(i.notes) for i in out.instruments)} notes)", flush=True)
print(f"DONE — set FL project tempo to {bpm} BPM", flush=True)
