import numpy as np, soundfile as sf, librosa
from scipy.signal import butter, sosfilt

SR = 48000
BPM = 123.0
BAR = int(round(60.0/BPM * 4 * SR))      # samples per 4/4 bar
XF  = int(0.015 * SR)                     # 15ms equal-power crossfade at splices
OUT = "/mnt/d/flbeat/data/SEND/GRIM3_SneakerS"
NDB_PATH = "/mnt/d/flbeat/data/grim_roformer/GRIM_3_GRIT_(No Drum-Bass)_model_bs_roformer_ep_937_sdr_10.flac"

def load(p):
    y, sr = sf.read(p, always_2d=True)
    if sr != SR: y = librosa.resample(y.T, orig_sr=sr, target_sr=SR).T
    if y.shape[1] == 1: y = np.repeat(y, 2, axis=1)
    return y.astype(np.float32)

FULL = load(OUT + "/GRIM_3_TIGHTBASS.wav")   # regulated-bass full mix (DJ's bass fix)
NDB  = load(NDB_PATH)                          # melody/atmosphere only (no drums, no 808)

mono = FULL.mean(axis=1)
on = librosa.onset.onset_detect(y=mono, sr=SR, units='samples', backtrack=True)
origin = (int(on[0]) % BAR) if len(on) else 0

def bars(src, sb, n):
    a = origin + sb*BAR; b = a + n*BAR
    seg = src[a:b]
    if len(seg) < n*BAR:
        pad = n*BAR - len(seg)
        seg = np.concatenate([seg, src[origin:origin+pad]], axis=0)
    return seg.copy()

def rms(x): return float(np.sqrt(np.mean(x.mean(axis=1)**2)) + 1e-9)
NDBg = NDB * (rms(FULL) * 0.75 / rms(NDB))    # match melody level; breakdowns a touch softer

# section, source-start-bar, num-bars
plan = [("mel",0,8), ("full",8,16), ("full",24,16), ("mel",8,8),
        ("full",8,16), ("full",24,16), ("mel",24,8), ("full",32,8)]

def xfade(a, b, n):
    n = min(n, len(a), len(b))
    if n <= 0: return np.concatenate([a, b], 0)
    t = np.linspace(0, np.pi/2, n)[:, None]
    return np.concatenate([a[:-n], a[-n:]*np.cos(t) + b[:n]*np.sin(t), b[n:]], 0)

out = None
for kind, sb, n in plan:
    seg = bars(NDBg if kind == "mel" else FULL, sb, n)
    out = seg if out is None else xfade(out, seg, XF)

fi = 2*BAR; fo = 2*BAR
out[:fi]  *= np.linspace(0, 1, fi)[:, None]
out[-fo:] *= np.linspace(1, 0, fo)[:, None]

def norm(x, db):
    p = np.max(np.abs(x)); return x * (10**(db/20) / (p + 1e-9))

ext = norm(out, -1.0)
sf.write(OUT + "/GRIM_3_EXTENDED.wav", ext, SR, subtype='PCM_24')

# ---- mix/master chain ----
def hp(x, f):
    sos = butter(2, f/(SR/2), btype='high', output='sos')
    return np.stack([sosfilt(sos, x[:, c]) for c in range(x.shape[1])], 1)
def lowshelf(x, f, gain_db):
    sos = butter(2, f/(SR/2), btype='low', output='sos')
    low = np.stack([sosfilt(sos, x[:, c]) for c in range(x.shape[1])], 1)
    return x + low * (10**(gain_db/20) - 1)

m = hp(ext, 30.0)              # remove subsonic rumble
m = lowshelf(m, 100.0, -1.5)   # tame the hot low-end a touch more

voc = norm(m, -6.0)            # vocal-ready: 6 dB headroom, no limiting
sf.write(OUT + "/GRIM_3_EXTENDED_VOCALREADY.wav", voc, SR, subtype='PCM_24')

mono2 = m.mean(axis=1); r = np.sqrt(np.mean(mono2**2))
g = min((10**(-8.5/20)) / (r + 1e-9), 6.0)
x = m * g; drive = 1.15
lim = np.tanh(x*drive) / np.tanh(drive)
os4 = librosa.resample(lim.T, orig_sr=SR, target_sr=SR*4).T
tp = np.max(np.abs(os4))
mst = lim * (10**(-1/20) / max(tp, 1e-9))
sf.write(OUT + "/GRIM_3_EXTENDED_MASTERED.wav", mst, SR, subtype='PCM_24')

for nm, a in (("EXTENDED", ext), ("VOCALREADY", voc), ("MASTERED", mst)):
    mm = a.mean(axis=1)
    print("%-11s %5.1fs  peak %.3f  rms %6.1f dB" % (nm, len(a)/SR, np.max(np.abs(a)), 20*np.log10(np.sqrt(np.mean(mm**2))+1e-9)))
print("BAR=%d samples  origin=%d  sections=%d bars" % (BAR, origin, sum(n for _,_,n in plan)))
