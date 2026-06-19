"""Author a .flp that loads excalibur's REAL separated stems (Demucs output) as
Sampler channels, each triggered once at bar 1. Opening it sounds EXACTLY like
excalibur (it IS excalibur's audio), split into drums/bass/melody/vocals layers
to chop and build on. Same no-FL, no-mouse authoring bypass we proved works."""
import os, struct, pathlib, warnings
warnings.filterwarnings("ignore")

OUT = r"D:\flmidi\out\excalibur"
STEMS = os.path.join(OUT, "demucs", "htdemucs", "excalibur")
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
def e_t16(i, s):   d = s.encode("utf-16-le") + b"\x00\x00"; return bytes([i]) + varlen(len(d)) + d
def e_ascii(i, s): d = s.encode("ascii") + b"\x00"; return bytes([i]) + varlen(len(d)) + d
def e_data(i, d):  return bytes([i]) + varlen(len(d)) + d


def note_struct(pos, key, length, rack, vel=100):
    return struct.pack("<IHHIHHBBBBBBBB", pos, 0, rack, length, key, 0,
                       120, 0, 64, 0, 64, vel, 128, 128)


channels = [
    ("Drums (excalibur)",      "drums.wav"),
    ("Bass (excalibur)",       "bass.wav"),
    ("Melody/Lead (excalibur)", "other.wav"),
    ("Vocals (excalibur)",     "vocals.wav"),
]

SONGLEN = int(TEMPO / 60.0 * PPQ * 140)   # ~ long enough to span the whole track

events = bytearray()
events += e_ascii(199, VERSION)
events += e_dword(156, int(TEMPO * 1000))
notes = bytearray()
for idx, (name, wav) in enumerate(channels):
    events += e_word(64, idx)            # New channel
    events += e_byte(0, 1)               # enabled
    events += e_byte(21, 0)              # Sampler
    events += e_dword(145, 0)            # GroupNum
    events += e_t16(192, name)           # Name
    events += e_t16(196, os.path.join(STEMS, wav))  # SamplePath -> the real stem
    notes += note_struct(0, 60, SONGLEN, idx, 100)  # one trigger at bar 1, natural pitch

events += e_word(65, 1)
events += e_data(224, bytes(notes))

hdr = b"FLhd" + struct.pack("<IHHH", 6, 0, len(channels), PPQ)
dat = b"FLdt" + struct.pack("<I", len(events)) + bytes(events)
out = os.path.join(OUT, "excalibur_stems.flp")
pathlib.Path(out).write_bytes(hdr + dat)
print("WROTE", out, os.path.getsize(out), "bytes")

import pyflp_patch  # noqa: F401
import pyflp
proj = pyflp.parse(out)
print("verify tempo:", proj.tempo)
for p in proj.patterns:
    print("verify notes:", len(list(p.notes)))
for nm, wav in channels:
    pth = os.path.join(STEMS, wav)
    print("  stem exists:", os.path.exists(pth), wav)
