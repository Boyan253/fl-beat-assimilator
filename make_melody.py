"""Reusable melody pipeline (the locked 'sounds perfect' recipe):
  python make_melody.py "<song.mp3>" <songname>
-> separates, transcribes melody with Basic Pitch (strict + gentle legato),
   cuts a real slice per pitch, builds a validated SF2 multisample.
Load <song>_melody.sf2 into Fruity Soundfont Player + drag <song>_melody.mid in."""
import os, sys, glob, re, struct, warnings
warnings.filterwarnings("ignore")
D = r"D:\flmidi"
os.environ.update(TMP=D + r"\tmp", TEMP=D + r"\tmp", TORCH_HOME=D + r"\torchcache",
                  HF_HOME=D + r"\hf", XDG_CACHE_HOME=D + r"\hf")
import numpy as np, librosa, soundfile as sf, pretty_midi, torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH

MP3 = sys.argv[1]
SONG = sys.argv[2]
SR = 44100
OUT = os.path.join(D, "out", SONG)
STEMDIR = os.path.join(OUT, "stems")
SLICEDIR = os.path.join(OUT, "slices", "melody")
for dd in (STEMDIR, SLICEDIR):
    os.makedirs(dd, exist_ok=True)


def build_sf2(slicedir, out_path, name="Melody"):
    wavs = sorted(glob.glob(os.path.join(slicedir, "p*.wav")), key=lambda w: int(re.search(r"p(\d+)", w).group(1)))
    pitches = [int(re.search(r"p(\d+)", w).group(1)) for w in wavs]
    N = len(wavs)
    pad = lambda b: b + (b"\x00" if len(b) % 2 else b"")
    sub = lambda cid, data: cid + struct.pack("<I", len(data)) + pad(data)
    n20 = lambda s: s.encode("ascii")[:20].ljust(20, b"\x00")
    smpl = bytearray(); hdrs = []
    for w, p in zip(wavs, pitches):
        y, sr = sf.read(w, dtype="int16", always_2d=False)
        if y.ndim > 1:
            y = y[:, 0]
        start = len(smpl) // 2; smpl += y.tobytes(); end = len(smpl) // 2; smpl += b"\x00\x00" * 46
        hdrs.append(("p%03d" % p, start, end, sr, p))
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
        igen += struct.pack("<HH", 43, (lo & 0xFF) | ((hi & 0xFF) << 8))
        igen += struct.pack("<HH", 58, p & 0xFF) + struct.pack("<HH", 53, i)
    igen += struct.pack("<HH", 0, 0)
    shdr = b""
    for (nm, start, end, sr, root) in hdrs:
        shdr += n20(nm) + struct.pack("<IIII", start, end, start, end) + struct.pack("<I", sr) + struct.pack("<BbHH", root, 0, 0, 1)
    shdr += n20("EOS") + struct.pack("<IIII", 0, 0, 0, 0) + struct.pack("<I", 0) + struct.pack("<BbHH", 0, 0, 0, 0)
    pdta = sub(b"phdr", phdr) + sub(b"pbag", pbag) + sub(b"pmod", pmod) + sub(b"pgen", pgen) + sub(b"inst", inst) + sub(b"ibag", ibag) + sub(b"imod", imod) + sub(b"igen", igen) + sub(b"shdr", shdr)
    body = b"sfbk" + INFO + sdta + sub(b"LIST", b"pdta" + pdta)
    open(out_path, "wb").write(b"RIFF" + struct.pack("<I", len(body)) + body)
    return N


# 1) separate -> other stem
MODEL = sys.argv[3] if len(sys.argv) > 3 else "htdemucs"
print("separating with", MODEL, flush=True)
model = get_model(MODEL); model.eval()
y, _ = librosa.load(MP3, sr=int(model.samplerate), mono=False)
if y.ndim == 1:
    y = np.stack([y, y])
wav = torch.tensor(y, dtype=torch.float32); ref = wav.mean(0)
wav = (wav - ref.mean()) / (ref.std() + 1e-8)
with torch.no_grad():
    srcs = apply_model(model, wav[None], device="cpu", split=True, overlap=0.25, progress=True)[0]
srcs = srcs * ref.std() + ref.mean()
stempaths = {}
for name, src in zip(model.sources, srcs):
    p = os.path.join(STEMDIR, name + ".wav")
    sf.write(p, src.detach().cpu().numpy().T.astype("float32"), int(model.samplerate))
    stempaths[name] = p

# pick the single most-melodic stem so all slices share ONE instrument's timbre
def mid_energy(path):
    yy, _ = librosa.load(path, sr=22050, mono=True, duration=90)
    Y = np.abs(np.fft.rfft(yy)); f = np.fft.rfftfreq(len(yy), 1 / 22050)
    return float(np.mean(Y[(f >= 200) & (f < 3500)]) or 0)
cands = [s for s in ("guitar", "piano", "other") if s in stempaths]
lead = max(cands, key=lambda s: mid_energy(stempaths[s]))
other = stempaths[lead]
print("melody source stem:", lead, flush=True)

# 2) transcribe (strict) + gentle legato
print("transcribing...", flush=True)
_, midi_data, notes = predict(other, ICASSP_2022_MODEL_PATH, onset_threshold=0.6,
                              frame_threshold=0.4, minimum_note_length=120,
                              minimum_frequency=110, maximum_frequency=1300)
evs = sorted([(float(n[0]), float(n[1]), int(n[2])) for n in notes], key=lambda x: x[0])
for i in range(len(evs) - 1):
    s, e, p = evs[i]
    evs[i] = (s, min(max(e, evs[i + 1][0]), s + 1.0), p)
pm = pretty_midi.PrettyMIDI(); inst = pretty_midi.Instrument(program=0, name=SONG)
for s, e, p in evs:
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=p, start=s, end=max(e, s + 0.05)))
pm.instruments.append(inst)
MID = os.path.join(OUT, SONG + "_melody.mid"); pm.write(MID)

# 3) one real slice per pitch
yhi, _ = librosa.load(other, sr=SR, mono=True)
best = {}
for n in notes:
    s, e, p, amp = float(n[0]), float(n[1]), int(n[2]), float(n[3])
    sc = (e - s) * (amp + 0.1)
    if p not in best or sc > best[p][0]:
        best[p] = (sc, s)
for old in glob.glob(os.path.join(SLICEDIR, "p*.wav")):
    os.remove(old)
for p, (sc, s) in best.items():
    seg = yhi[int(s * SR):int((s + 1.2) * SR)]
    if len(seg) < 256:
        continue
    seg = seg / (np.max(np.abs(seg)) or 1) * 0.97
    fo = int(0.25 * SR)
    if len(seg) > fo:
        seg[-fo:] *= np.linspace(1, 0, fo)
    sf.write(os.path.join(SLICEDIR, "p%03d.wav" % p), seg.astype("float32"), SR)

# 4) SF2 + validate
SF2 = os.path.join(OUT, SONG + "_melody.sf2")
N = build_sf2(SLICEDIR, SF2, name=SONG[:18])
from sf2utils.sf2parse import Sf2File
v = Sf2File(open(SF2, "rb"))
print("DONE notes=%d pitches=%d | SF2 samples=%d bags=%d" % (len(evs), N, len(v.samples), len(v.instruments[0].bags)))
print("SF2:", SF2)
print("MID:", MID)
