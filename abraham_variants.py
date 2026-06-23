"""Three spacing variants of abraham's melody so the user can A/B/C and pick the
exact one. Same notes/pitches; only the note LENGTHS differ -> different spacing.
Load abraham_GOOD.sf2 once, then drag each .mid to compare."""
import os, warnings
warnings.filterwarnings("ignore")
import pretty_midi
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH

STEM = r"D:\flmidi\out\abraham\stems\other.wav"
OUT = r"D:\flmidi\out\abraham"

_, _, notes = predict(STEM, ICASSP_2022_MODEL_PATH, onset_threshold=0.6, frame_threshold=0.4,
                      minimum_note_length=120, minimum_frequency=110, maximum_frequency=1300)
evs = sorted([(float(n[0]), float(n[1]), int(n[2])) for n in notes], key=lambda x: x[0])


def write(name, end_fn):
    pm = pretty_midi.PrettyMIDI(); inst = pretty_midi.Instrument(program=0)
    for i, (s, e, p) in enumerate(evs):
        nxt = evs[i + 1][0] if i + 1 < len(evs) else e + 0.5
        end = end_fn(s, e, nxt)
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=p, start=s, end=max(end, s + 0.05)))
    pm.instruments.append(inst)
    fn = os.path.join(OUT, name)
    pm.write(fn)
    print("wrote", name)


# A: SPACED  — notes as detected (natural gaps, most space)
write("abraham_A_spaced.mid", lambda s, e, nxt: e)
# B: MEDIUM  — gentle legato cap 1.0 (this is the old "GOOD"/bp3 setting)
write("abraham_B_medium.mid", lambda s, e, nxt: min(max(e, nxt), s + 1.0))
# C: CONNECTED — full legato, notes touch the next one (least/no space)
write("abraham_C_connected.mid", lambda s, e, nxt: nxt)
print("DONE")
