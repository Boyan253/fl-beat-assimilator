"""Convert the abraham per-pitch melody slices into an SF2 SoundFont (multisample),
loadable by FL's stock 'Fruity Soundfont Player' / DirectWave. Each sample is mapped
to its real key range with its true root pitch -> a real, pitch-editable melody."""
import os, glob, re, struct, warnings
warnings.filterwarnings("ignore")
import numpy as np, soundfile as sf

SLICEDIR = r"D:\flmidi\out\abraham\slices\melody"
OUT = r"D:\flmidi\out\abraham\abraham_GOOD2.sf2"

wavs = sorted(glob.glob(os.path.join(SLICEDIR, "p*.wav")),
              key=lambda w: int(re.search(r"p(\d+)", w).group(1)))
pitches = [int(re.search(r"p(\d+)", w).group(1)) for w in wavs]
N = len(wavs)


def pad(b):
    return b + (b"\x00" if len(b) % 2 else b"")


def sub(cid, data):
    return cid + struct.pack("<I", len(data)) + pad(data)


# ---- sample data (smpl): int16 mono, 46 zero frames after each ----
smpl = bytearray()
hdrs = []          # (name, start, end, sr, root)
for w, p in zip(wavs, pitches):
    y, sr = sf.read(w, dtype="int16", always_2d=False)
    if y.ndim > 1:
        y = y[:, 0]
    start = len(smpl) // 2
    smpl += y.tobytes()
    end = len(smpl) // 2
    smpl += b"\x00\x00" * 46
    hdrs.append(("p%03d" % p, start, end, sr, p))

info = sub(b"ifil", struct.pack("<HH", 2, 1)) + sub(b"isng", pad(b"EMU8000\x00")) + sub(b"INAM", pad(b"Abraham Melody\x00"))
INFO = sub(b"LIST", b"INFO" + info)
sdta = sub(b"LIST", b"sdta" + sub(b"smpl", bytes(smpl)))

# ---- pdta ----
# phdr: 1 preset + terminal (38 bytes each)
def name20(s): return s.encode("ascii")[:20].ljust(20, b"\x00")
phdr = name20("Abraham") + struct.pack("<HHHIII", 0, 0, 0, 0, 0, 0)
phdr += name20("EOP") + struct.pack("<HHHIII", 0, 0, 1, 0, 0, 0)
# pbag: bag0 + terminal (genNdx, modNdx)
pbag = struct.pack("<HH", 0, 0) + struct.pack("<HH", 1, 0)
pmod = struct.pack("<HHHHHHHHHH", *([0] * 10))  # 10-byte terminal modulator
pgen = struct.pack("<HH", 41, 0) + struct.pack("<HH", 0, 0)  # instrument=0, terminal
# inst: 1 instrument + terminal
inst = name20("Abraham") + struct.pack("<H", 0)
inst += name20("EOI") + struct.pack("<H", N)   # terminal bagNdx indexes IBAG (bag count), not IGEN
# ibag: one per zone + terminal (genNdx steps by 3)
ibag = b""
for i in range(N):
    ibag += struct.pack("<HH", i * 3, 0)
ibag += struct.pack("<HH", N * 3, 0)
imod = struct.pack("<HHHHHHHHHH", *([0] * 10))
# igen: per zone keyRange(43), overridingRootKey(58), sampleID(53); + terminal
igen = b""
for i, p in enumerate(pitches):
    lo = 0 if i == 0 else (pitches[i - 1] + p) // 2 + 1
    hi = 127 if i == N - 1 else (p + pitches[i + 1]) // 2
    igen += struct.pack("<HH", 43, (lo & 0xFF) | ((hi & 0xFF) << 8))   # keyRange
    igen += struct.pack("<HH", 58, p & 0xFF)                           # overridingRootKey
    igen += struct.pack("<HH", 53, i)                                  # sampleID (must be last)
igen += struct.pack("<HH", 0, 0)
# shdr: 46 bytes each + terminal
shdr = b""
for (nm, start, end, sr, root) in hdrs:
    shdr += name20(nm) + struct.pack("<IIII", start, end, start, end) + struct.pack("<I", sr)
    shdr += struct.pack("<BbHH", root, 0, 0, 1)   # originalPitch, correction, link, type=1 mono
shdr += name20("EOS") + struct.pack("<IIII", 0, 0, 0, 0) + struct.pack("<I", 0) + struct.pack("<BbHH", 0, 0, 0, 0)

pdta = (sub(b"phdr", phdr) + sub(b"pbag", pbag) + sub(b"pmod", pmod) + sub(b"pgen", pgen) +
        sub(b"inst", inst) + sub(b"ibag", ibag) + sub(b"imod", imod) + sub(b"igen", igen) + sub(b"shdr", shdr))
PDTA = sub(b"LIST", b"pdta" + pdta)

body = b"sfbk" + INFO + sdta + PDTA
riff = b"RIFF" + struct.pack("<I", len(body)) + body
with open(OUT, "wb") as f:
    f.write(riff)
print("WROTE", OUT, "| %d samples, %.0f KB" % (N, len(riff) / 1024))
