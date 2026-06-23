"""FL trial can't re-open saved .flp, so read untitled.flp directly with PyFLP and
recover its contents: dump each pattern's notes to a .mid (the trial CAN open MIDI),
and list any sample paths the project referenced so we know what instruments it used."""
import sys, os, re, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"D:\flmidi")
import pyflp_patch  # noqa: F401  (py3.11 fix)
import pyflp, pretty_midi

FLP = r"D:\flmidi\out\excalibur\untitled.flp"
OUTDIR = r"D:\flmidi\out\excalibur\recovered"
os.makedirs(OUTDIR, exist_ok=True)
TEMPO = 140.0

p = pyflp.parse(FLP)
ppq = getattr(p, "ppq", 96) or 96
t2s = lambda ticks: ticks / ppq * (60.0 / TEMPO)

print("ppq=%d" % ppq)
for idx, pat in enumerate(p.patterns):
    name = getattr(pat, "name", None)
    notes = list(pat.notes)
    if not notes:
        print("pattern %d %r: empty" % (idx + 1, name)); continue
    pm = pretty_midi.PrettyMIDI(); inst = pretty_midi.Instrument(program=0)
    for n in notes:
        try:
            key = int(n["key"]); pos = int(n["position"]); ln = int(n["length"])
            vel = int(n["velocity"]) if "velocity" in dir(n) or True else 100
        except Exception:
            continue
        s = t2s(pos)
        inst.notes.append(pretty_midi.Note(velocity=max(1, min(127, vel)), pitch=max(0, min(127, key)),
                                            start=s, end=s + max(t2s(ln), 0.05)))
    pm.instruments.append(inst)
    safe = re.sub(r"[^A-Za-z0-9]+", "_", str(name or "pattern%d" % (idx + 1)))
    fn = os.path.join(OUTDIR, "%02d_%s.mid" % (idx + 1, safe))
    pm.write(fn)
    print("pattern %d %r: %d notes -> %s" % (idx + 1, name, len(inst.notes), fn))

# scan raw bytes for sample paths (event id 196 = 0xC4: varint len + UTF-16-LE path)
print("\n--- sample paths referenced ---")
data = open(FLP, "rb").read()
seen = set()
for m in re.finditer(rb"[A-Za-z]:\x00(?:[ -~]\x00){4,}", data):  # UTF-16-LE drive paths
    try:
        s = m.group().decode("utf-16-le")
        if s.lower().endswith((".wav", ".sf2", ".mp3", ".flac", ".ogg")) and s not in seen:
            seen.add(s); print(" ", s)
    except Exception:
        pass
print("DONE — load the .mid files from", OUTDIR)
