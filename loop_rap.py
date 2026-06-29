"""loop_rap.py — extend a beat into a longer rap version by beat-aligned looping.
Keeps the original intro + outro, repeats the main body (whole bars) with tiny crossfades
so the groove continues seamlessly. Same beat, just long enough to rap over.

  python loop_rap.py <in.wav> <out.wav> <bpm> [target_seconds=135]
"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, librosa, soundfile as sf

IN = sys.argv[1]; OUT = sys.argv[2]; BPM = float(sys.argv[3])
TARGET = float(sys.argv[4]) if len(sys.argv) > 4 else 135.0

y, sr = librosa.load(IN, sr=None, mono=False)
if y.ndim == 1:
    y = y[None, :]
bar = int(round((4 * 60.0 / BPM) * sr))     # samples per bar
total_bars = y.shape[1] // bar
intro_bars, outro_bars = 4, 4
intro = y[:, :intro_bars * bar]
body = y[:, intro_bars * bar:(total_bars - outro_bars) * bar]
outro = y[:, (total_bars - outro_bars) * bar:]

cf = int(0.04 * sr)
def xfade(a, b):
    n = min(cf, a.shape[1], b.shape[1])
    fo = np.linspace(1, 0, n); fi = np.linspace(0, 1, n)
    a = a.copy(); b = b.copy()
    a[:, -n:] *= fo; b[:, :n] *= fi
    return np.concatenate([a[:, :-n], a[:, -n:] + b[:, :n], b[:, n:]], axis=1)

out = intro
target = int(TARGET * sr)
while out.shape[1] < target - body.shape[1]:
    out = xfade(out, body)
out = xfade(out, outro)
sf.write(OUT, out.T, sr)
print(f"extended {y.shape[1]/sr:.0f}s -> {out.shape[1]/sr:.0f}s @ {BPM:.0f}bpm -> {OUT}", flush=True)
