"""Proper recovery of untitled.flp: read its channels (name + sample/SF2 path) with a
raw event walk, and split each pattern's notes BY channel into separate MIDIs, so the
full multi-instrument arrangement can be rebuilt (not flattened into one track)."""
import sys, os, re, struct, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"D:\flmidi")
import pyflp_patch  # noqa: F401
import pyflp, pretty_midi

FLP = r"D:\flmidi\out\excalibur\untitled.flp"
OUTDIR = r"D:\flmidi\out\excalibur\recovered"
os.makedirs(OUTDIR, exist_ok=True)
TEMPO = 140.0


def raw_channels(path):
    data = open(path, "rb").read()
    hlen = struct.unpack("<I", data[4:8])[0]
    pos = 8 + hlen + 8                       # skip FLhd + FLdt headers
    end = len(data)
    chans, cur = {}, -1
    while pos < end:
        eid = data[pos]; pos += 1
        if eid < 64:
            pos += 1
        elif eid < 128:
            pos += 2
        elif eid < 192:
            pos += 4
        else:
            length = 0; shift = 0
            while True:
                b = data[pos]; pos += 1
                length |= (b & 0x7F) << shift
                if not (b & 0x80):
                    break
                shift += 7
            val = data[pos:pos + length]; pos += length
            if eid == 192:
                try: chans.setdefault(cur, {})["name"] = val.decode("utf-16-le").rstrip("\x00")
                except Exception: pass
            elif eid == 196:
                try: chans.setdefault(cur, {})["sample"] = val.decode("utf-16-le").rstrip("\x00")
                except Exception: pass
            continue
        if eid == 64:                        # New channel (WORD) — value already skipped above
            cur = struct.unpack("<H", data[pos - 2:pos])[0]
            chans.setdefault(cur, {})
    return chans


chans = raw_channels(FLP)
print("=== channels in untitled.flp ===")
for idx in sorted(chans):
    c = chans[idx]
    print("  ch %-3d name=%-22r sample=%s" % (idx, c.get("name"), c.get("sample", "")))

p = pyflp.parse(FLP)
ppq = getattr(p, "ppq", 96) or 96
t2s = lambda t: t / ppq * (60.0 / TEMPO)
print("\n=== notes per pattern, split by channel ===")
for pidx, pat in enumerate(p.patterns):
    bych = {}
    for n in list(pat.notes):
        try:
            rc = int(n["rack_channel"])
            bych.setdefault(rc, []).append((int(n["key"]), int(n["position"]), int(n["length"]),
                                            int(n["velocity"])))
        except Exception:
            pass
    if not bych:
        continue
    for rc, ns in sorted(bych.items()):
        pm = pretty_midi.PrettyMIDI(); inst = pretty_midi.Instrument(program=0)
        for key, pos, ln, vel in ns:
            s = t2s(pos)
            inst.notes.append(pretty_midi.Note(velocity=max(1, min(127, vel)),
                                               pitch=max(0, min(127, key)), start=s,
                                               end=s + max(t2s(ln), 0.05)))
        pm.instruments.append(inst)
        cname = re.sub(r"[^A-Za-z0-9]+", "_", str(chans.get(rc, {}).get("name") or "ch%d" % rc))[:24]
        fn = os.path.join(OUTDIR, "pat%d_ch%d_%s.mid" % (pidx + 1, rc, cname))
        pm.write(fn)
        print("  pat%d ch%-3d (%s): %d notes -> %s" % (pidx + 1, rc, chans.get(rc, {}).get("name"),
                                                        len(ns), os.path.basename(fn)))
