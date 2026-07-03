"""make_fl_pyscript.py — turn a beat MIDI into FL Studio piano-roll scripts (.pyscript) that BAKE
the notes and apply them via flpianoroll. This is the TRIAL-COMPATIBLE "notes into FL, no mouse"
path: FL 2025's free trial supports piano-roll scripting. One script per instrument — open that
channel's piano roll, run the script, click Apply, notes appear. (Same mechanism as the MCP bridge.)

Matches FL 2025's own factory-script convention exactly (createDialog + apply(form), Note in ticks,
velocity 0..1). Times are baked as BEATS and multiplied by score.PPQ at runtime = PPQ-independent.

  python make_fl_pyscript.py <in.mid> <outdir>
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
import pretty_midi

IN, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
pm = pretty_midi.PrettyMIDI(IN)
tempi = pm.get_tempo_changes()[1]
BPM = float(tempi[0]) if len(tempi) else 140.0
if len(sys.argv) > 3:                                  # optional BPM override (keep all parts consistent)
    BPM = float(sys.argv[3])
BPS = BPM / 60.0


def classify(inst):
    nm = (inst.name or "").lower()
    if inst.is_drum or "drum" in nm: return "Drums"
    if "808" in nm or "bass" in nm: return "808"
    return "Cowbell"


buckets = {"Cowbell": [], "808": [], "Drums": []}
for inst in pm.instruments:
    cat = classify(inst)
    for n in inst.notes:
        sb = round(float(n.start) * BPS, 5)                        # float() = strip numpy types
        lb = round(max(0.0625, float(n.end - n.start) * BPS), 5)   # >= 1/16 beat
        vel = round(min(1.0, max(0.05, float(n.velocity) / 127.0)), 4)
        buckets[cat].append((int(n.pitch), float(sb), float(lb), float(vel)))

TEMPLATE = '''import flpianoroll as flp
# {title} - auto-generated phonk part  ({bpm:.0f} BPM, {count} notes)
# USE: open the {title} channel's Piano Roll -> Scripting menu -> run "{title}" -> Apply.
# (midi_note, start_beats, length_beats, velocity_0_1)
BAKED = [
{rows}
]


def createDialog():
    form = flp.ScriptDialog("{title} (phonk)",
                            "Drops the generated {title} notes into THIS channel's piano roll.\\r\\n"
                            "Open it on the {title} channel.")
    form.AddInputCombo("Clear existing notes first", "Yes,No", 0)
    return form


def apply(form):
    if form.GetInputValue("Clear existing notes first") == 0:
        flp.score.clearNotes(True)
    ppq = flp.score.PPQ
    for num, tb, lb, vel in BAKED:
        n = flp.Note()
        n.number = int(num)
        n.time = int(round(tb * ppq))
        n.length = max(1, int(round(lb * ppq)))
        n.velocity = float(vel)
        flp.score.addNote(n)
'''

made = []
for cat, notes in buckets.items():
    if not notes:
        continue
    notes.sort(key=lambda x: (x[1], x[0]))
    rows = "\n".join(f"    {t!r}," for t in notes)
    __import__("ast").literal_eval("[" + rows + "]")               # guarantee PURE literals (no numpy/np.)
    src = TEMPLATE.format(title=cat, bpm=BPM, count=len(notes), rows=rows)
    compile(src, cat + ".pyscript", "exec")                        # syntax-validate before writing
    with open(os.path.join(OUT, cat + ".pyscript"), "w", encoding="utf-8") as fh:
        fh.write(src)
    made.append((cat, len(notes)))

print(f"[pyscript] {BPM:.0f}bpm  " + " ".join(f"{c}={n}" for c, n in made) + f"  -> {OUT}", flush=True)
