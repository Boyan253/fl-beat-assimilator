"""Reusable melody-from-stem builder (the locked abraham 'sustain6' recipe):
  python build_melody_sf2.py <stem.wav> <out_dir> <prefix> [bad_pitches_csv]
Thin harmonic doubles -> merge held notes -> slice each pitch ending before the next
attack -> SF2 with looping sustain + natural release (polyphonic, no hard cut) ->
legato MIDI. Optionally rebuild mis-pitched notes from a clean neighbor."""
import os, glob, re, struct, sys, bisect, warnings
warnings.filterwarnings("ignore")
import numpy as np, librosa, soundfile as sf, pretty_midi
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH

STEM = sys.argv[1]
OUTDIR = sys.argv[2]
PREFIX = sys.argv[3]
BAD = set(int(x) for x in sys.argv[4].split(",")) if len(sys.argv) > 4 and sys.argv[4] else set()
SLICEDIR = os.path.join(OUTDIR, "slices_" + PREFIX)
os.makedirs(SLICEDIR, exist_ok=True)
SR = 44100
SLICE_LEN = 2.5
MERGE_GAP = 0.50


def build_sf2(slicedir, out_path, name):
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
        L = len(y); zc = np.where(np.diff(np.signbit(y)))[0]
        nz = lambda t: int(zc[np.argmin(np.abs(zc - t))]) if len(zc) else t
        ls = nz(int(0.40 * L)); le = nz(int(0.85 * L))
        if le - ls < 256: le = min(L - 1, ls + max(256, L // 5))
        hdrs.append(("p%03d" % p, st, en, st + ls, st + le, sr, p))
    INFO = sub(b"LIST", b"INFO" + sub(b"ifil", struct.pack("<HH", 2, 1)) + sub(b"isng", pad(b"EMU8000\x00")) + sub(b"INAM", pad(n20(name))))
    sdta = sub(b"LIST", b"sdta" + sub(b"smpl", bytes(smpl)))
    phdr = n20(name) + struct.pack("<HHHIII", 0, 0, 0, 0, 0, 0) + n20("EOP") + struct.pack("<HHHIII", 0, 0, 1, 0, 0, 0)
    pbag = struct.pack("<HH", 0, 0) + struct.pack("<HH", 1, 0)
    pmod = struct.pack("<HHHHHHHHHH", *([0] * 10))
    pgen = struct.pack("<HH", 41, 0) + struct.pack("<HH", 0, 0)
    inst = n20(name) + struct.pack("<H", 0) + n20("EOI") + struct.pack("<H", N)
    G = 5; REL = (-1500) & 0xFFFF
    ibag = b"".join(struct.pack("<HH", i * G, 0) for i in range(N)) + struct.pack("<HH", N * G, 0)
    imod = struct.pack("<HHHHHHHHHH", *([0] * 10))
    igen = b""
    for i, p in enumerate(pitches):
        lo = 0 if i == 0 else (pitches[i - 1] + p) // 2 + 1
        hi = 127 if i == N - 1 else (p + pitches[i + 1]) // 2
        igen += struct.pack("<HH", 43, (lo & 0xFF) | ((hi & 0xFF) << 8))
        igen += struct.pack("<HH", 58, p & 0xFF) + struct.pack("<HH", 38, REL) + struct.pack("<HH", 54, 1) + struct.pack("<HH", 53, i)
    igen += struct.pack("<HH", 0, 0)
    shdr = b""
    for (nm, st, en, lst, lend, sr, root) in hdrs:
        shdr += n20(nm) + struct.pack("<IIII", st, en, lst, lend) + struct.pack("<I", sr) + struct.pack("<BbHH", root, 0, 0, 1)
    shdr += n20("EOS") + struct.pack("<IIII", 0, 0, 0, 0) + struct.pack("<I", 0) + struct.pack("<BbHH", 0, 0, 0, 0)
    pdta = sub(b"phdr", phdr) + sub(b"pbag", pbag) + sub(b"pmod", pmod) + sub(b"pgen", pgen) + sub(b"inst", inst) + sub(b"ibag", ibag) + sub(b"imod", imod) + sub(b"igen", igen) + sub(b"shdr", shdr)
    body = b"sfbk" + INFO + sdta + sub(b"LIST", b"pdta" + pdta)
    open(out_path, "wb").write(b"RIFF" + struct.pack("<I", len(body)) + body)
    return N


print("transcribing", os.path.basename(STEM), flush=True)
_, _, notes = predict(STEM, ICASSP_2022_MODEL_PATH, onset_threshold=0.6, frame_threshold=0.4,
                      minimum_note_length=120, minimum_frequency=110, maximum_frequency=1300)
ev_amp = sorted([(float(n[0]), float(n[1]), int(n[2]), float(n[3])) for n in notes], key=lambda x: x[0])
clean = []
for s, e, p, a in ev_amp:                                  # thin near-simultaneous harmonic doubles
    if clean and s - clean[-1][0] < 0.05:
        if a > clean[-1][3]:
            clean[-1] = (s, e, p, a)
    else:
        clean.append((s, e, p, a))
ev0 = [(s, e, p) for (s, e, p, a) in clean]
merged = []
for s, e, p in ev0:                                        # merge held notes
    if merged and merged[-1][2] == p and s - merged[-1][1] <= MERGE_GAP:
        merged[-1] = (merged[-1][0], max(merged[-1][1], e), p)
    else:
        merged.append((s, e, p))
print("notes %d -> thin %d -> merge %d" % (len(ev_amp), len(clean), len(merged)))

yhi, _ = librosa.load(STEM, sr=SR, mono=True)
starts = sorted(x[0] for x in ev0)
gap_after = lambda t: (starts[bisect.bisect_right(starts, t + 0.02)] - t) if bisect.bisect_right(starts, t + 0.02) < len(starts) else 3.0
best = {}
for s, e, p, amp in clean:
    g = gap_after(s); sc = min(g, SLICE_LEN) * (amp + 0.1)
    if p not in best or sc > best[p][0]:
        best[p] = (sc, s, g)
for old in glob.glob(os.path.join(SLICEDIR, "p*.wav")):
    os.remove(old)
for p, (sc, s, g) in best.items():
    if p in BAD:
        continue
    slen = min(SLICE_LEN, max(0.4, g - 0.03))
    seg = yhi[int(s * SR):int((s + slen) * SR)].copy()
    if len(seg) < 256:
        continue
    seg = seg / (np.max(np.abs(seg)) or 1) * 0.97
    fi = int(0.006 * SR)
    if len(seg) > fi: seg[:fi] *= np.linspace(0, 1, fi)
    fo = min(int(0.30 * SR), len(seg) // 3)
    if fo > 0: seg[-fo:] *= np.linspace(1, 0, fo)
    sf.write(os.path.join(SLICEDIR, "p%03d.wav" % p), seg.astype("float32"), SR)
have = sorted(int(re.search(r"p(\d+)", os.path.basename(w)).group(1)) for w in glob.glob(os.path.join(SLICEDIR, "p*.wav")))
for bp in sorted(BAD):
    if bp not in best or not have:
        continue
    src = min(have, key=lambda x: abs(x - bp))
    ysrc, _sr = sf.read(os.path.join(SLICEDIR, "p%03d.wav" % src))
    if ysrc.ndim > 1: ysrc = ysrc[:, 0]
    ysh = librosa.effects.pitch_shift(ysrc.astype("float32"), sr=SR, n_steps=float(bp - src))
    sf.write(os.path.join(SLICEDIR, "p%03d.wav" % bp), (ysh / (np.max(np.abs(ysh)) or 1) * 0.97).astype("float32"), SR)
    print("rebuilt bad pitch %d from %d (%+d st)" % (bp, src, bp - src))

SF2 = os.path.join(OUTDIR, PREFIX + ".sf2")
N = build_sf2(SLICEDIR, SF2, PREFIX[:18])
pm = pretty_midi.PrettyMIDI(); inst = pretty_midi.Instrument(program=0)
for i, (s, e, p) in enumerate(merged):
    nxt = merged[i + 1][0] if i + 1 < len(merged) else e + 0.5
    end = min(max(e, nxt), s + SLICE_LEN)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=p, start=s, end=max(end, s + 0.05)))
pm.instruments.append(inst); pm.write(os.path.join(OUTDIR, PREFIX + ".mid"))
from sf2utils.sf2parse import Sf2File
v = Sf2File(open(SF2, "rb"))
print("DONE  SF2 samples=%d bags=%d" % (len(v.samples), len(v.instruments[0].bags)))
print("SF2:", SF2, "\nMID:", os.path.join(OUTDIR, PREFIX + ".mid"))
