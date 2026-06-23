"""Bypass FL's missing 'load instrument' API by AUTHORING the .flp directly.
Emits an old-format (v20) project FL 2025 opens via backward-compat: Sampler
channels pointing at our auto-sampled WAVs + the transcribed notes baked in.
Verifies by re-parsing with PyFLP before handoff (no FL, no mouse)."""
import os, json, struct, pathlib, warnings, sys
warnings.filterwarnings("ignore")
import pretty_midi

OUT = r"D:\flmidi\out\excalibur"
INSTR = os.path.join(OUT, "instruments")
TEMPO = 140.0
PPQ = 96
VERSION = "20.9.2.2963"

BYTE, WORD, DWORD, TEXT, DATA = 0, 64, 128, 192, 208


def varlen(n):
    out = bytearray()
    while True:
        b = n & 0x7F; n >>= 7
        if n:
            b |= 0x80
        out.append(b)
        if not n:
            break
    return bytes(out)


def e_byte(i, v):  return struct.pack("<BB", i, v & 0xFF)
def e_word(i, v):  return struct.pack("<BH", i, v & 0xFFFF)
def e_dword(i, v): return struct.pack("<BI", i, v & 0xFFFFFFFF)
def e_t16(i, s):
    d = s.encode("utf-16-le") + b"\x00\x00"
    return bytes([i]) + varlen(len(d)) + d
def e_ascii(i, s):
    d = s.encode("ascii") + b"\x00"
    return bytes([i]) + varlen(len(d)) + d
def e_data(i, d):  return bytes([i]) + varlen(len(d)) + d


def sec2tick(s):
    return max(0, int(round(s * TEMPO / 60.0 * PPQ)))


def note_struct(pos, key, length, rack, vel=100):
    key = max(0, min(131, key))
    return struct.pack("<IHHIHHBBBBBBBB",
                       pos, 0, rack, max(1, length), key, 0,
                       120, 0, 64, 0, 64, max(1, min(127, vel)), 128, 128)


summary = json.load(open(os.path.join(OUT, "summary.json")))
kc_mel = summary["stems"]["melody"]["instrument_keycenter"]
kc_bass = summary["stems"]["bass"]["instrument_keycenter"]

channels = [
    # mono=True collapses to a single line (kills the over-detected note cloud);
    # minlen drops fragment notes; vel balances the mix.
    dict(name="Melody (excalibur)",   wav="melody_lead.wav", mid="melody.mid", kc=kc_mel,  drum=None, mono=True,  minlen=0.10, vel=66),
    dict(name="808 Bass (excalibur)", wav="bass_808.wav",    mid="bass.mid",   kc=kc_bass, drum=None, mono=True,  minlen=0.07, vel=100),
    dict(name="Kick",  wav="kick.wav",  mid="drums.mid", kc=None, drum=36, mono=False, minlen=0.0, vel=110),
    dict(name="Snare", wav="snare.wav", mid="drums.mid", kc=None, drum=38, mono=False, minlen=0.0, vel=92),
    dict(name="Hat",   wav="hat.wav",   mid="drums.mid", kc=None, drum=42, mono=False, minlen=0.0, vel=64),
]

midcache = {}
all_notes = bytearray()
counts = []
for idx, ch in enumerate(channels):
    mp = os.path.join(OUT, ch["mid"])
    if mp not in midcache:
        midcache[mp] = pretty_midi.PrettyMIDI(mp)
    raw = []
    for inst in midcache[mp].instruments:
        for n in inst.notes:
            if ch["drum"] is not None:
                if n.pitch != ch["drum"]:
                    continue
                key = 60                                  # one-shot at natural pitch
            else:
                key = 60 + (n.pitch - ch["kc"])           # map so sample sounds at real pitch
            raw.append([n.start, n.end, key])
    raw.sort(key=lambda r: r[0])
    raw = [r for r in raw if (r[1] - r[0]) >= ch["minlen"]]   # drop fragments
    if ch["mono"]:                                            # one note at a time
        mono = []
        for r in raw:
            if mono and r[0] < mono[-1][1]:
                mono[-1][1] = r[0]                            # truncate previous at this onset
            if not mono or (r[1] - r[0]) >= 0.03:
                mono.append(r)
        raw = [r for r in mono if (r[1] - r[0]) >= 0.03]
    for s, e, key in raw:
        all_notes += note_struct(sec2tick(s), key, sec2tick(e - s), idx, ch["vel"])
    counts.append((ch["name"], len(raw)))

events = bytearray()
events += e_ascii(199, VERSION)                # FLVersion (ASCII)
events += e_dword(156, int(TEMPO * 1000))      # Tempo (x1000)
for idx, ch in enumerate(channels):
    events += e_word(64, idx)                  # New channel (selects current)
    events += e_byte(0, 1)                     # ChannelIsEnabled = 1
    events += e_byte(21, 0)                    # Type = Sampler
    events += e_dword(145, 0)                  # GroupNum = 0 (unfiltered group)
    events += e_t16(192, ch["name"])           # Name (UTF-16-LE)
    events += e_t16(196, os.path.join(INSTR, ch["wav"]))  # SamplePath
events += e_word(65, 1)                        # New pattern 1
events += e_data(224, bytes(all_notes))        # Notes

hdr = b"FLhd" + struct.pack("<IHHH", 6, 0, len(channels), PPQ)
dat = b"FLdt" + struct.pack("<I", len(events)) + bytes(events)
outflp = os.path.join(OUT, "excalibur_skeleton.flp")
pathlib.Path(outflp).write_bytes(hdr + dat)
print("WROTE", outflp)
print("  size=%d bytes, %d notes total" % (os.path.getsize(outflp), len(all_notes) // 24))
for nm, c in counts:
    print("   ", nm.ljust(22), c, "notes")

# ---- verify by re-parsing with patched PyFLP (proves structural validity) ----
print("\n--- PyFLP re-parse verification ---")
import pyflp_patch  # noqa: F401  (Python 3.11 fix)
import pyflp
proj = pyflp.parse(outflp)
print("tempo:", getattr(proj, "tempo", "?"))
for p in proj.patterns:
    try:
        print("  pattern:", repr(p.name), "notes=", len(list(p.notes)))
    except Exception as e:
        print("  pattern err:", e)
