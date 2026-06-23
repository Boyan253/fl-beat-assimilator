"""Finer spacing variants BETWEEN B(medium) and A(spaced). Lighter legato caps =
more space than B, less than A. Load abraham_GOOD.sf2, A/B/C these three."""
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


def write(name, cap):
    pm = pretty_midi.PrettyMIDI(); inst = pretty_midi.Instrument(program=0)
    for i, (s, e, p) in enumerate(evs):
        nxt = evs[i + 1][0] if i + 1 < len(evs) else e + 0.5
        end = min(max(e, nxt), s + cap)        # extend toward next note but only up to `cap`
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=p, start=s, end=max(end, s + 0.05)))
    pm.instruments.append(inst); pm.write(os.path.join(OUT, name)); print("wrote", name, "cap=%.2f" % cap)


write("abraham_D_cap04.mid", 0.40)   # more space (closest to A)
write("abraham_E_cap06.mid", 0.60)   # middle
write("abraham_F_cap08.mid", 0.80)   # less space (closest to B)
print("DONE")
