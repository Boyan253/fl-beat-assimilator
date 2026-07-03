"""Read-only probe of a local .flp: can pyflp see channels+their plugins+notes, and are notes
writable? If yes -> a TEMPLATE .flp (instruments loaded once in FL) can be cloned and have our
generated notes injected = instruments + notes in ONE openable file.  usage: python this.py <flp>"""
import sys, warnings
warnings.filterwarnings("ignore")
import pyflp
print("pyflp", getattr(pyflp, "__version__", "?"))
p = pyflp.parse(sys.argv[1])
print("tempo", getattr(p, "tempo", None), "ppq", getattr(p, "ppq", None))
print("== CHANNELS ==")
for c in p.channels:
    plug = getattr(c, "plugin", None)
    print(f"  [{type(c).__name__}] name={getattr(c,'name',None)!r} plugin={type(plug).__name__ if plug else None}")
print("== PATTERNS ==")
for pat in p.patterns:
    notes = list(getattr(pat, "notes", []) or [])
    print(f"  name={getattr(pat,'name',None)!r} notes={len(notes)}")
    for n in notes[:2]:
        print(f"     pos={getattr(n,'position',None)} key={getattr(n,'key',None)} len={getattr(n,'length',None)} rack={getattr(n,'rack_channel',None)}")
try:
    import pyflp.pattern as pat
    Note = pat.Note
    print("Note constructible:", callable(Note), "fields:", [a for a in dir(Note) if not a.startswith('_')][:18])
except Exception as e:
    print("Note introspect err:", e)
print("project savable:", hasattr(p, "save"))
