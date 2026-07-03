"""Split a beat MIDI into clearly-named single-purpose tracks for easy FL import:
Melody, 808, Kick, Snare, Hat — each loads ONE instrument/sample on its own channel.
  python make_fl_friendly.py <in.mid> <out.mid> <bpm>
"""
import sys
import pretty_midi

IN, OUT, BPM = sys.argv[1], sys.argv[2], float(sys.argv[3])
src = pretty_midi.PrettyMIDI(IN)
out = pretty_midi.PrettyMIDI(initial_tempo=BPM)

mel = pretty_midi.Instrument(program=11, name="Melody")
b808 = pretty_midi.Instrument(program=38, name="808")
kick = pretty_midi.Instrument(program=0, name="Kick")
snare = pretty_midi.Instrument(program=0, name="Snare")
hat = pretty_midi.Instrument(program=0, name="Hat")
DRUM = {36: kick, 38: snare, 42: hat}

for inst in src.instruments:
    nm = (inst.name or "").lower()
    if inst.is_drum or "drum" in nm:
        for n in inst.notes:
            tgt = DRUM.get(n.pitch)
            if tgt is not None:                       # one fixed trigger note (C5) per drum channel
                tgt.notes.append(pretty_midi.Note(n.velocity, 60, n.start, n.start + 0.12))
    elif "808" in nm or "bass" in nm:
        for n in inst.notes:
            b808.notes.append(n)
    else:
        for n in inst.notes:
            mel.notes.append(n)

for t in (mel, b808, kick, snare, hat):
    if t.notes:
        out.instruments.append(t)
out.write(OUT)
print("FL-friendly:", ", ".join(f"{t.name}({len(t.notes)})" for t in out.instruments), "->", OUT, flush=True)
