"""Probe the newest autosave .flp with PyFLP: dump channels (+ sample paths) and patterns.
Read-only — never writes the source. Tells us exactly what we can author into."""
import sys, glob, os, warnings
warnings.filterwarnings("ignore")
import pyflp

backup = r"C:\Users\User\Documents\Image-Line\FL Studio\Projects\Backup"
flps = sorted(glob.glob(os.path.join(backup, "*.flp")), key=os.path.getmtime)
if not flps:
    print("no flp found"); sys.exit(1)
src = flps[-1]
print("NEWEST:", os.path.basename(src))
proj = pyflp.parse(src)

for attr in ("tempo", "ppq"):
    try:
        print(attr, "=", getattr(proj, attr))
    except Exception as e:
        print(attr, "ERR", e)

print("--- channels ---")
try:
    for ch in proj.channels:
        t = type(ch).__name__
        name = getattr(ch, "name", None)
        iid = getattr(ch, "iid", None)
        sp = None
        try:
            sp = getattr(ch, "sample_path", None)
        except Exception:
            sp = "<n/a>"
        print(f"  iid={iid} {t:14} name={name!r} sample_path={sp!r}")
except Exception as e:
    print("channels ERR", type(e).__name__, e)

print("--- patterns ---")
try:
    for p in proj.patterns:
        n = 0
        try:
            n = len(list(p.notes))
        except Exception as e:
            n = f"<{e}>"
        print(f"  name={p.name!r} notes={n}")
except Exception as e:
    print("patterns ERR", type(e).__name__, e)

# what does a Note look like / can we construct one?
print("--- note model ---")
try:
    import pyflp.pattern as pat
    print("  pattern module attrs:", [a for a in dir(pat) if "Note" in a])
except Exception as e:
    print("  ERR", e)
