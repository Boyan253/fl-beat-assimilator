"""clean_melody.py — turn the noisy basic_pitch transcription into a clean, MONOPHONIC GRIM
melody line ("melody only"): drop ghost/short notes, then keep one note at a time (the loudest),
trimming overlaps. Output a real .mid file for the GRIM instrument.
  python clean_melody.py [in.mid] [out.mid]
"""
import sys, warnings
warnings.filterwarnings("ignore")
import pretty_midi

IN  = sys.argv[1] if len(sys.argv) > 1 else "/mnt/d/flbeat/data/generated/GRIM_melody_aligned.mid"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/mnt/d/flbeat/data/generated/GRIM_melody_clean.mid"

pm = pretty_midi.PrettyMIDI(IN)
notes = [n for ins in pm.instruments for n in ins.notes]
raw = len(notes)
notes = [n for n in notes if (n.end - n.start) >= 0.08 and n.velocity >= 28]   # kill ghosts/blips
notes.sort(key=lambda n: (n.start, -n.velocity))

mono = []
for n in notes:
    if mono and n.start < mono[-1].end - 0.02:          # overlaps the note we're holding
        if n.velocity > mono[-1].velocity:              # louder -> it takes over
            mono[-1].end = max(mono[-1].start + 0.05, n.start)
            mono.append(n)
        # quieter overlap -> drop it (keeps it monophonic = "melody only")
    else:
        if mono and mono[-1].end > n.start:
            mono[-1].end = n.start                      # trim tail so notes don't overlap
        mono.append(n)

out = pretty_midi.PrettyMIDI(initial_tempo=122.0)
inst = pretty_midi.Instrument(program=0, name="GRIM Melody")
inst.notes = mono
out.instruments.append(inst)
out.write(OUT)
print(f"clean melody: {len(mono)} notes (from {raw}) -> {OUT}", flush=True)
