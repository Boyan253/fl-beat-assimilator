"""transcribe_drumsep.py — turn DrumSep'd stems (kick/snare/hh/toms/ride/crash) into clean MIDI.
Each stem is already ISOLATED, so onset detection is accurate (no band-split guessing).

  python transcribe_drumsep.py <drumsep_dir> <out_midi_dir> [bpm=144]
Writes Kick/Snare/Hat/Toms/Ride/Crash.mid (single-note C5 for samplers) + Drums.mid (GM kit).
"""
import os, sys, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np, librosa, pretty_midi

DIR = sys.argv[1]; OUT = sys.argv[2]
BPM = float(sys.argv[3]) if len(sys.argv) > 3 else 144.0
os.makedirs(OUT, exist_ok=True)
SR = 44100
# stem tag -> (nice name, General-MIDI drum note)
MAP = [("kick", "Kick", 36), ("snare", "Snare", 38), ("hh", "Hat", 42),
       ("toms", "Toms", 45), ("ride", "Ride", 51), ("crash", "Crash", 49)]

kit = pretty_midi.PrettyMIDI(initial_tempo=BPM)
drum = pretty_midi.Instrument(program=0, is_drum=True, name="drums")
summary = []
for tag, nice, gm in MAP:
    fs = glob.glob(os.path.join(DIR, f"*({tag})*"))
    if not fs:
        continue
    y, _ = librosa.load(fs[0], sr=SR, mono=True)
    if np.max(np.abs(y)) < 1e-4:
        summary.append(f"{nice}=0(silent)"); continue
    # onset detect on the isolated stem, then gate weak hits by local peak energy
    on = librosa.onset.onset_detect(y=y, sr=SR, units="time", backtrack=True)
    kept = []
    peak = np.max(np.abs(y))
    for t in on:
        i = int(t * SR); w = y[i:i + int(0.03 * SR)]
        if len(w) and np.max(np.abs(w)) > 0.06 * peak:   # ignore bleed/noise hits
            kept.append(float(t))
    for t in kept:
        drum.notes.append(pretty_midi.Note(100, gm, t, t + 0.12))
    p = pretty_midi.PrettyMIDI(initial_tempo=BPM); ins = pretty_midi.Instrument(program=0)
    for t in kept:
        ins.notes.append(pretty_midi.Note(110, 60, t, t + 0.12))
    p.instruments.append(ins); p.write(os.path.join(OUT, f"{nice}.mid"))
    summary.append(f"{nice}={len(kept)}")
kit.instruments.append(drum); kit.write(os.path.join(OUT, "Drums.mid"))
print("drumsep MIDI:", "  ".join(summary), "->", OUT, flush=True)
