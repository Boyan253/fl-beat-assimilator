"""Cut a clean 4-bar GRIM melody loop, MONO, ~7.9s (fits Samplab's free 10s/mono limit).
Picks the loudest 4-bar window = the main melodic hook.  python cut_loop.py [stem] [out] [bpm] [bars]"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, soundfile as sf, librosa

STEM = sys.argv[1] if len(sys.argv) > 1 else "/mnt/d/flbeat/data/generated/GRIM_melody_6s_CLEAN.wav"
OUT  = sys.argv[2] if len(sys.argv) > 2 else "/mnt/d/flbeat/data/SEND/GRIMPack/GRIM_melody_loop4.wav"
BPM  = float(sys.argv[3]) if len(sys.argv) > 3 else 122.0
BARS = int(sys.argv[4]) if len(sys.argv) > 4 else 4

y, sr = librosa.load(STEM, sr=44100, mono=True)
loop = BARS * 4 * 60.0 / BPM                      # seconds for BARS bars
n = int(loop * sr)
hop = int(0.1 * sr)

best_i, best_rms = 0, -1.0                         # slide to the loudest BARS-bar window (the hook)
for i in range(0, max(1, len(y) - n), hop):
    r = float(np.sqrt(np.mean(y[i:i + n] ** 2)))
    if r > best_rms:
        best_rms, best_i = r, i

seg = y[best_i:best_i + n].copy()
f = int(0.005 * sr)
seg[:f] *= np.linspace(0, 1, f)
seg[-f:] *= np.linspace(1, 0, f)
sf.write(OUT, seg.astype(np.float32), sr)
print(f"loop: {BARS} bars, start {best_i/sr:.2f}s, len {len(seg)/sr:.2f}s, MONO {sr}Hz -> {OUT}", flush=True)
