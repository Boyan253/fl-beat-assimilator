"""Monkeypatch PyFLP to work on Python 3.11 (its empty-base EventEnum can't be
called directly anymore; _missing_ still works), then parse:
  1) our hand-written excalibur_skeleton.flp  -> proves the .flp is valid
  2) the newest FL 2025 autosave               -> proves the cleaner bypass too
"""
import sys, glob, os, warnings
warnings.filterwarnings("ignore")
import pyflp
import pyflp._events as _ev

_real = _ev.EventEnum
_meta = type(_real)              # _EventEnumMeta (subclass of enum.EnumMeta)
_orig_call = _meta.__call__

def _patched_call(cls, *args, **kwargs):
    # Python 3.11 refuses to call an empty enum; route the empty base (no members)
    # through its own _missing_, which dispatches to subclasses / makes pseudo members.
    if len(args) == 1 and not kwargs and not cls._member_map_:
        r = cls._missing_(args[0])
        if r is not None:
            return r
    return _orig_call(cls, *args, **kwargs)

_meta.__call__ = _patched_call


def dump(path, label):
    print("\n==================", label, "==================")
    print(os.path.basename(path))
    try:
        proj = pyflp.parse(path)
    except Exception as e:
        import traceback
        print("PARSE FAILED:", type(e).__name__, e)
        traceback.print_exc()
        return False
    try:
        print("tempo:", proj.tempo)
    except Exception as e:
        print("  tempo err:", e)
    nch = 0
    try:
        for ch in proj.channels:
            nch += 1
            sp = None
            try:
                sp = ch.sample_path
            except Exception:
                pass
            print("  ch:", type(ch).__name__, repr(getattr(ch, "name", None)), "| sample:", sp)
    except Exception as e:
        print("  channels iter err:", type(e).__name__, e)
    npat = 0
    try:
        for p in proj.patterns:
            npat += 1
            try:
                cnt = len(list(p.notes))
            except Exception as e:
                cnt = "err:%s" % e
            print("  pattern:", repr(p.name), "notes=", cnt)
    except Exception as e:
        print("  patterns iter err:", type(e).__name__, e)
    print("  TOTAL channels=%d patterns=%d" % (nch, npat))
    return True


# 1) our hand-written file
mine = r"D:\flmidi\out\excalibur\excalibur_skeleton.flp"
ok1 = dump(mine, "HAND-WRITTEN SKELETON")

# 2) the newest FL 2025 autosave (the real project)
backup = r"C:\Users\User\Documents\Image-Line\FL Studio\Projects\Backup"
auto = sorted(glob.glob(os.path.join(backup, "*.flp")), key=os.path.getmtime)
ok2 = dump(auto[-1], "FL 2025 AUTOSAVE") if auto else False

print("\n=== RESULT: skeleton_parses=%s  fl2025_parses=%s ===" % (ok1, ok2))
