"""CREATE-from-reference — generate an ORIGINAL melody in the STYLE of a real reference riff.
Keeps the reference's EXACT rhythm/groove (what makes phonk coherent) but reassigns the pitches
via an order-2 Markov chain learned from the reference -> new notes, same feel & key. This is
'AI creates in your style' (vs generic rules, which come out chaotic). Pair with the cowbell.
  python create_from_ref.py <ref.mid> <name> [seed]
"""
import os, sys, random
from collections import defaultdict
import pretty_midi

REF = sys.argv[1]; NAME = sys.argv[2]
random.seed(int(sys.argv[3]) if len(sys.argv) > 3 else 0)
OUT = os.path.join("D:/flmidi/out/created", NAME); os.makedirs(OUT, exist_ok=True)

pm = pretty_midi.PrettyMIDI(REF)
notes = sorted([n for n in pm.instruments[0].notes if n.pitch > 1], key=lambda n: n.start)
pitches = [n.pitch for n in notes]
pset = sorted(set(pitches))

t2 = defaultdict(list)
for i in range(len(pitches) - 2):
    t2[(pitches[i], pitches[i + 1])].append(pitches[i + 2])
t1 = defaultdict(list)
for i in range(len(pitches) - 1):
    t1[pitches[i]].append(pitches[i + 1])

gen = [pitches[0], pitches[1]] if len(pitches) >= 2 else list(pitches)
for i in range(2, len(notes)):
    nxt = t2.get((gen[-2], gen[-1])) or t1.get(gen[-1]) or pset
    gen.append(random.choice(nxt))

out = pretty_midi.PrettyMIDI(initial_tempo=145)
inst = pretty_midi.Instrument(program=0)
inst.notes.append(pretty_midi.Note(velocity=1, pitch=0, start=0.0, end=0.02))   # t=0 anchor
for n, p in zip(notes, gen):
    inst.notes.append(pretty_midi.Note(velocity=n.velocity, pitch=int(p), start=n.start, end=n.end))
out.instruments.append(inst); out.write(os.path.join(OUT, NAME + "_melody.mid"))
print("generated %d notes in reference style (pitch set %d..%d) -> %s_melody.mid"
      % (len(notes), min(pset), max(pset), NAME))
