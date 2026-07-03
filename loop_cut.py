"""loop_cut.py — cut a tight bar-aligned loop from a stem (for Slicex / sampling).
  python loop_cut.py <in.wav> <out.wav> <bpm> [bars=8]
Finds the first musically-active bar and cuts <bars> whole bars from a downbeat, with tiny fades.
"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, librosa, soundfile as sf

IN = sys.argv[1]; OUT = sys.argv[2]; BPM = float(sys.argv[3])
BARS = int(sys.argv[4]) if len(sys.argv) > 4 else 8
y, sr = librosa.load(IN, sr=None, mono=False)
mono = y.mean(0) if y.ndim > 1 else y
bar_s = int(round((4 * 60.0 / BPM) * sr))
nbars = len(mono) // bar_s
rms = np.array([np.sqrt(np.mean(mono[i * bar_s:(i + 1) * bar_s] ** 2)) for i in range(nbars)])
start = next((i for i, r in enumerate(rms) if r > 0.25 * rms.max()), 0)
s0 = start * bar_s
end = s0 + BARS * bar_s
seg = (y[:, s0:end] if y.ndim > 1 else y[s0:end]).astype("float32")
n = seg.shape[-1]
fade = min(int(0.01 * sr), n // 8)
env = np.ones(n); env[:fade] = np.linspace(0, 1, fade); env[-fade:] = np.linspace(1, 0, fade)
seg = seg * env if seg.ndim == 1 else seg * env[None, :]
sf.write(OUT, seg.T if seg.ndim > 1 else seg, sr)
print(f"loop: {BARS} bars from bar {start} ({n/sr:.1f}s @ {BPM:.0f}bpm) -> {OUT}", flush=True)
