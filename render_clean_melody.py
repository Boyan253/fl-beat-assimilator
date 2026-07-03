"""render_clean_melody.py — render a MIDI through a clean GM piano (no separation artifacts).
  python render_clean_melody.py <in.mid> <out.wav> [program=0]
"""
import sys, os, subprocess, warnings
warnings.filterwarnings("ignore")
import pretty_midi

MID = sys.argv[1]; OUT = sys.argv[2]; PROG = int(sys.argv[3]) if len(sys.argv) > 3 else 0
pm = pretty_midi.PrettyMIDI(MID)
for ins in pm.instruments:
    ins.program = PROG; ins.is_drum = False
tmp = OUT + ".tmp.mid"; pm.write(tmp)
subprocess.run(["fluidsynth", "-ni", "-g", "1.4", "-F", OUT, "-r", "44100",
                "/usr/share/sounds/sf2/FluidR3_GM.sf2", tmp], check=True, capture_output=True)
os.remove(tmp)
n = sum(len(i.notes) for i in pm.instruments)
print(f"rendered {n} notes -> {OUT}", flush=True)
