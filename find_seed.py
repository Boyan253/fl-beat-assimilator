"""Find a .flp that PyFLP 2.2.1 can actually parse (older format) to use as a
write seed. FL 2025 opens older projects, so an editable old seed is all we need."""
import os, glob, warnings
warnings.filterwarnings("ignore")
import pyflp

roots = [r"D:\fl_studio"]
found_ok = []
checked = 0
errs = {}
allflp = []
for root in roots:
    if not os.path.isdir(root):
        continue
    allflp += [p for p in glob.glob(os.path.join(root, "**", "*.flp"), recursive=True) if "Backup" not in p]
# prefer templates / simplest first
allflp.sort(key=lambda p: (("Template" not in p), os.path.getsize(p)))
for p in allflp:
    checked += 1
    try:
        proj = pyflp.parse(p)
        nch = len(list(proj.channels))
        npat = len(list(proj.patterns))
        found_ok.append((p, nch))
        print("OK  ch=%-3d pat=%-3d %s" % (nch, npat, os.path.relpath(p, roots[0])))
        if len(found_ok) >= 8:
            break
    except Exception as e:
        errs[type(e).__name__] = errs.get(type(e).__name__, 0) + 1
print("\nchecked %d flp, %d parse OK, errors=%s" % (checked, len(found_ok), errs))
