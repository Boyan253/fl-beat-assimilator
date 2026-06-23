"""Low-pass a bass stem so it lives in the sub/low-end only and stops masking the
melodic layer's mids.  python clean_bass.py <in.wav> <out.wav> [lp=260]"""
import sys, numpy as np, soundfile as sf
from scipy.signal import butter, sosfilt
inp, outp = sys.argv[1], sys.argv[2]
lp = float(sys.argv[3]) if len(sys.argv) > 3 else 260.0
y, sr = sf.read(inp, always_2d=True); ny = sr / 2.0
y = sosfilt(butter(4, lp / ny, btype="low", output="sos"), y, axis=0)
y = y / (np.max(np.abs(y)) or 1) * 0.97
sf.write(outp, y.astype("float32"), sr)
print("bass low-passed @%.0fHz -> %s" % (lp, outp))
