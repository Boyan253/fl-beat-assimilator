"""Stem cleanup: tame low-end mud + restore clarity WITHOUT making it thin/distant.
  python clean_stem.py <in.wav> <out.wav> [hp=75] [bright=0.2] [body=0.35]
hp   = high-pass (removes sub mud; lower = keeps more warmth/body)
body = low-mid (150-450Hz) presence boost -> pulls the sound 'forward'/closer
bright = >3.5kHz boost -> clarity (keep modest so it doesn't get thin)"""
import sys, numpy as np, soundfile as sf
from scipy.signal import butter, sosfilt

inp, outp = sys.argv[1], sys.argv[2]
hp = float(sys.argv[3]) if len(sys.argv) > 3 else 75.0
bright = float(sys.argv[4]) if len(sys.argv) > 4 else 0.2
body = float(sys.argv[5]) if len(sys.argv) > 5 else 0.35
y, sr = sf.read(inp, always_2d=True); ny = sr / 2.0

y = sosfilt(butter(4, hp / ny, btype="high", output="sos"), y, axis=0)          # gentle mud cut
band = sosfilt(butter(2, [150 / ny, 450 / ny], btype="band", output="sos"), y, axis=0)
y = y + body * band                                                             # proximity / body
hi = sosfilt(butter(2, 3500 / ny, btype="high", output="sos"), y, axis=0)
y = y + bright * hi                                                             # clarity
y = y / (np.max(np.abs(y)) or 1) * 0.97
sf.write(outp, y.astype("float32"), sr)
print("cleaned -> %s (hp=%.0f body=%.2f bright=%.2f)" % (outp, hp, body, bright))
