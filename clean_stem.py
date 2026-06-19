"""Cheap targeted stem cleanup for the 'muddy bass + dull' separation artifacts:
high-pass out the sub-100Hz mud (doesn't belong in a lead) + a gentle high-shelf
brighten to restore clarity lost in separation.
  python clean_stem.py <in.wav> <out.wav> [hp_hz=100]"""
import sys, numpy as np, soundfile as sf
from scipy.signal import butter, sosfilt

inp, outp = sys.argv[1], sys.argv[2]
hp = float(sys.argv[3]) if len(sys.argv) > 3 else 100.0
y, sr = sf.read(inp, always_2d=True)
ny = sr / 2.0
# high-pass: kill low-end separation mud
y = sosfilt(butter(4, hp / ny, btype="high", output="sos"), y, axis=0)
# brighten: add a touch of >3kHz content back (counters the 'dull/480p' feel)
hi = sosfilt(butter(2, 3000 / ny, btype="high", output="sos"), y, axis=0)
y = y + 0.45 * hi
y = y / (np.max(np.abs(y)) or 1) * 0.97
sf.write(outp, y.astype("float32"), sr)
print("cleaned -> %s (hp=%.0fHz + brighten)" % (outp, hp))
