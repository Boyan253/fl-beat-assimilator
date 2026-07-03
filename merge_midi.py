"""merge_midi.py — merge a pack's per-instrument MIDI into ONE multi-track MIDI.
Import this into FL (File > Import > MIDI) and it auto-builds a channel + the full arrangement
per instrument. Then assign each channel its one-shot -> a complete FL project that plays the song.

  python merge_midi.py <pack_midi_dir> <out.mid> <bpm>
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
import pretty_midi

MDIR = sys.argv[1]; OUT = sys.argv[2]; BPM = float(sys.argv[3]) if len(sys.argv) > 3 else 120.0
# (track name, source file, GM program)  — order = channel order in FL
ORDER = [("Melody", "Piano.mid", 0), ("Bass", "808.mid", 38),
         ("Kick", "Kick.mid", 0), ("Snare", "Snare.mid", 0),
         ("Hat", "Hat.mid", 0), ("Ride", "Ride.mid", 0)]

out = pretty_midi.PrettyMIDI(initial_tempo=BPM)
made = []
for name, f, prog in ORDER:
    p = os.path.join(MDIR, f)
    if not os.path.exists(p):
        continue
    src = pretty_midi.PrettyMIDI(p)
    inst = pretty_midi.Instrument(program=prog, is_drum=False, name=name)
    for i in src.instruments:
        inst.notes.extend(i.notes)
    if inst.notes:
        out.instruments.append(inst)
        made.append(f"{name}({len(inst.notes)})")
out.write(OUT)
print("merged tracks:", "  ".join(made), "->", OUT, flush=True)
