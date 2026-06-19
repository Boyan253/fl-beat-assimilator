"""Fix C#6 glitch + give 'closer than B' spacing options.
- SF2: exclude bugged pitch 73 (C#6) so a clean neighbor covers it; fade-in on every
  slice to kill onset clicks; 1.6s slices so notes can ring longer (=closer together).
- MIDIs: legato caps ABOVE B(1.0) -> notes closer together than medium."""
import os, glob, re, struct, warnings
warnings.filterwarnings("ignore")
import numpy as np, librosa, soundfile as sf, pretty_midi
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH

STEM = r"D:\flmidi\out\abraham\stems\other.wav"
OUT = r"D:\flmidi\out\abraham"
SLICEDIR = os.path.join(OUT, "slices", "melody_fixed")
os.makedirs(SLICEDIR, exist_ok=True)
SR = 44100
SLICE_LEN = 1.6
EXCLUDE = {73}                      # C#6 — corrupted slice; remap to clean neighbor


def build_sf2(slicedir, out_path, name="Abraham"):
    wavs = sorted(glob.glob(os.path.join(slicedir, "p*.wav")), key=lambda w: int(re.search(r"p(\d+)", w).group(1)))
    pitches = [int(re.search(r"p(\d+)", w).group(1)) for w in wavs]
    N = len(wavs)
    pad = lambda b: b + (b"\x00" if len(b) % 2 else b"")
    sub = lambda cid, d: cid + struct.pack("<I", len(d)) + pad(d)
    n20 = lambda s: s.encode("ascii")[:20].ljust(20, b"\x00")
    smpl = bytearray(); hdrs = []
    for w, p in zip(wavs, pitches):
        y, sr = sf.read(w, dtype="int16", always_2d=False)
        if y.ndim > 1: y = y[:, 0]
        st = len(smpl) // 2; smpl += y.tobytes(); en = len(smpl) // 2; smpl += b"\x00\x00" * 46
        hdrs.append(("p%03d" % p, st, en, sr, p))
    INFO = sub(b"LIST", b"INFO" + sub(b"ifil", struct.pack("<HH", 2, 1)) + sub(b"isng", pad(b"EMU8000\x00")) + sub(b"INAM", pad(n20(name))))
    sdta = sub(b"LIST", b"sdta" + sub(b"smpl", bytes(smpl)))
    phdr = n20(name) + struct.pack("<HHHIII", 0, 0, 0, 0, 0, 0) + n20("EOP") + struct.pack("<HHHIII", 0, 0, 1, 0, 0, 0)
    pbag = struct.pack("<HH", 0, 0) + struct.pack("<HH", 1, 0)
    pmod = struct.pack("<HHHHHHHHHH", *([0] * 10))
    pgen = struct.pack("<HH", 41, 0) + struct.pack("<HH", 0, 0)
    inst = n20(name) + struct.pack("<H", 0) + n20("EOI") + struct.pack("<H", N)
    ibag = b"".join(struct.pack("<HH", i * 3, 0) for i in range(N)) + struct.pack("<HH", N * 3, 0)
    imod = struct.pack("<HHHHHHHHHH", *([0] * 10))
    igen = b""
    for i, p in enumerate(pitches):
        lo = 0 if i == 0 else (pitches[i - 1] + p) // 2 + 1
        hi = 127 if i == N - 1 else (p + pitches[i + 1]) // 2
        igen += struct.pack("<HH", 43, (lo & 0xFF) | ((hi & 0xFF) << 8)) + struct.pack("<HH", 58, p & 0xFF) + struct.pack("<HH", 53, i)
    igen += struct.pack("<HH", 0, 0)
    shdr = b""
    for (nm, st, en, sr, root) in hdrs:
        shdr += n20(nm) + struct.pack("<IIII", st, en, st, en) + struct.pack("<I", sr) + struct.pack("<BbHH", root, 0, 0, 1)
    shdr += n20("EOS") + struct.pack("<IIII", 0, 0, 0, 0) + struct.pack("<I", 0) + struct.pack("<BbHH", 0, 0, 0, 0)
    pdta = sub(b"phdr", phdr) + sub(b"pbag", pbag) + sub(b"pmod", pmod) + sub(b"pgen", pgen) + sub(b"inst", inst) + sub(b"ibag", ibag) + sub(b"imod", imod) + sub(b"igen", igen) + sub(b"shdr", shdr)
    body = b"sfbk" + INFO + sdta + sub(b"LIST", b"pdta" + pdta)
    open(out_path, "wb").write(b"RIFF" + struct.pack("<I", len(body)) + body)
    return N


print("transcribing...", flush=True)
_, _, notes = predict(STEM, ICASSP_2022_MODEL_PATH, onset_threshold=0.6, frame_threshold=0.4,
                      minimum_note_length=120, minimum_frequency=110, maximum_frequency=1300)
evs = sorted([(float(n[0]), float(n[1]), int(n[2])) for n in notes], key=lambda x: x[0])

yhi, _ = librosa.load(STEM, sr=SR, mono=True)
best = {}
for n in notes:
    s, e, p, amp = float(n[0]), float(n[1]), int(n[2]), float(n[3])
    if p in EXCLUDE:
        continue
    sc = (e - s) * (amp + 0.1)
    if p not in best or sc > best[p][0]:
        best[p] = (sc, s)
for old in glob.glob(os.path.join(SLICEDIR, "p*.wav")):
    os.remove(old)
for p, (sc, s) in best.items():
    seg = yhi[int(s * SR):int((s + SLICE_LEN) * SR)].copy()
    if len(seg) < 256:
        continue
    seg = seg / (np.max(np.abs(seg)) or 1) * 0.97
    fi = int(0.006 * SR)
    if len(seg) > fi:
        seg[:fi] *= np.linspace(0, 1, fi)        # fade-in: kill onset click
    fo = int(0.30 * SR)
    if len(seg) > fo:
        seg[-fo:] *= np.linspace(1, 0, fo)
    sf.write(os.path.join(SLICEDIR, "p%03d.wav" % p), seg.astype("float32"), SR)

SF2 = os.path.join(OUT, "abraham_fixed.sf2")
N = build_sf2(SLICEDIR, SF2, "Abraham Fixed")


def write_mid(name, cap):
    pm = pretty_midi.PrettyMIDI(); inst = pretty_midi.Instrument(program=0)
    for i, (s, e, p) in enumerate(evs):
        nxt = evs[i + 1][0] if i + 1 < len(evs) else e + 0.5
        end = min(max(e, nxt), s + cap)
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=p, start=s, end=max(end, s + 0.05)))
    pm.instruments.append(inst); pm.write(os.path.join(OUT, name)); print("wrote", name, "cap=%.2f" % cap)


write_mid("abraham_close12.mid", 1.2)
write_mid("abraham_close15.mid", 1.5)
write_mid("abraham_close18.mid", 1.8)

from sf2utils.sf2parse import Sf2File
v = Sf2File(open(SF2, "rb"))
print("SF2 OK: %d samples (C#6 excluded), bags=%d" % (len(v.samples), len(v.instruments[0].bags)))
print("SF2:", SF2)
