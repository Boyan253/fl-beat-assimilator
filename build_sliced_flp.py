"""Author excalibur_sliced.flp: one native Sampler channel per real per-pitch
slice, with every transcribed note routed to its slice (played at natural pitch,
key 60). Opens with ZERO manual loading; every note triggers real excalibur audio."""
import os, glob, re, struct, pathlib, warnings
warnings.filterwarnings("ignore")
import pretty_midi

OUT = r"D:\flmidi\out\excalibur"
SLICEDIR = os.path.join(OUT, "slices")
INSTR = os.path.join(OUT, "instruments")
TEMPO, PPQ, VERSION = 140.0, 96, "20.9.2.2963"
NOTE = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def varlen(n):
    o = bytearray()
    while True:
        b = n & 0x7F; n >>= 7
        if n: b |= 0x80
        o.append(b)
        if not n: break
    return bytes(o)


def e_byte(i, v):  return struct.pack("<BB", i, v & 0xFF)
def e_word(i, v):  return struct.pack("<BH", i, v & 0xFFFF)
def e_dword(i, v): return struct.pack("<BI", i, v & 0xFFFFFFFF)
def e_t16(i, s):   d = s.encode("utf-16-le") + b"\x00\x00"; return bytes([i]) + varlen(len(d)) + d
def e_ascii(i, s): d = s.encode("ascii") + b"\x00"; return bytes([i]) + varlen(len(d)) + d
def e_data(i, d):  return bytes([i]) + varlen(len(d)) + d
def e_cutgroup(cself, cby): return bytes([132]) + struct.pack("<HH", cself & 0xFFFF, cby & 0xFFFF)
def sec2tick(s):   return max(0, int(round(s * TEMPO / 60.0 * PPQ)))
def nm(p):         return NOTE[p % 12] + str(p // 12 - 1)
def note_struct(pos, length, rack, vel):
    return struct.pack("<IHHIHHBBBBBBBB", pos, 0, rack, max(1, length), 60, 0,
                       120, 0, 64, 0, 64, max(1, min(127, vel)), 128, 128)

chans = []          # (name, wav_abspath)
evs = []            # (rack, pos, length, vel)


def add_pitched(stem_dir, midfile, prefix, vel, cut):
    pitch_to_ch = {}
    for w in sorted(glob.glob(os.path.join(SLICEDIR, stem_dir, "p*.wav"))):
        p = int(re.search(r"p(\d+)\.wav", os.path.basename(w)).group(1))
        pitch_to_ch[p] = len(chans)
        chans.append(("%s %s" % (prefix, nm(p)), w, cut))
    pm = pretty_midi.PrettyMIDI(os.path.join(OUT, midfile))
    raw = []
    for inst in pm.instruments:
        for n in inst.notes:
            raw.append([n.start, n.end, n.pitch])
    raw.sort(key=lambda r: r[0])
    for i in range(len(raw) - 1):              # extend each note to the next onset (no gaps)
        raw[i][1] = max(raw[i][1], raw[i + 1][0])
    for s, e, pit in raw:
        ch = pitch_to_ch.get(pit)
        if ch is None:
            ch = pitch_to_ch[min(pitch_to_ch, key=lambda x: abs(x - pit))]
        evs.append((ch, sec2tick(s), sec2tick(e - s), vel))


add_pitched("melody", "melody_sliced.mid", "Mel", 80, cut=None)  # polyphonic guitar = "really close"
add_pitched("bass", "bass_sliced.mid", "Bass", 100, cut=2)       # cut group 2 = monophonic 808

# drums: real one-shots on GM keys
drum = {36: ("kick.wav", 110), 38: ("snare.wav", 92), 42: ("hat.wav", 64)}
drum_ch = {}
for p, (wav, _) in drum.items():
    wp = os.path.join(INSTR, wav)
    if os.path.exists(wp):
        drum_ch[p] = len(chans)
        chans.append((wav.split(".")[0].capitalize(), wp, None))   # drums stay polyphonic
pmd = pretty_midi.PrettyMIDI(os.path.join(OUT, "drums.mid"))
for inst in pmd.instruments:
    for n in inst.notes:
        if n.pitch in drum_ch:
            evs.append((drum_ch[n.pitch], sec2tick(n.start), sec2tick(n.end - n.start), drum[n.pitch][1]))

events = bytearray()
events += e_ascii(199, VERSION)
events += e_dword(156, int(TEMPO * 1000))
for idx, (name, wav, cut) in enumerate(chans):
    events += e_word(64, idx)
    events += e_byte(0, 1)
    events += e_byte(21, 0)
    events += e_dword(145, 0)
    if cut is not None:
        events += e_cutgroup(cut, cut)        # monophonic: retrigger cuts the previous note
    events += e_t16(192, name)
    events += e_t16(196, wav)
events += e_word(65, 1)
notes = bytearray()
for (ch, pos, length, vel) in sorted(evs, key=lambda x: x[1]):
    notes += note_struct(pos, length, ch, vel)
events += e_data(224, bytes(notes))

hdr = b"FLhd" + struct.pack("<IHHH", 6, 0, len(chans), PPQ)
dat = b"FLdt" + struct.pack("<I", len(events)) + bytes(events)
out = os.path.join(OUT, "excalibur_build.flp")
pathlib.Path(out).write_bytes(hdr + dat)
print("WROTE", out, "| %d channels, %d notes" % (len(chans), len(evs)))

import pyflp_patch  # noqa: F401
import pyflp
proj = pyflp.parse(out)
print("verify tempo:", proj.tempo)
for p in proj.patterns:
    print("verify notes:", len(list(p.notes)))
