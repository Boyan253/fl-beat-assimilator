"""phonkify.py — add the LO-FI GRIT that separates phonk from clean trap.

Research: 'pristine/radio-ready KILLS phonk; embrace grime.' ACE-Step output is too clean.
This applies the phonk texture chain: drive/saturation + 808 low-end grit + bitcrush +
cassette wobble + vinyl noise + dark master.

  python phonkify.py <in.wav> <out.wav> [intensity=1.0]
"""
import os, sys, subprocess, warnings
warnings.filterwarnings("ignore")
import numpy as np, soundfile as sf, scipy.signal as ss

IN  = sys.argv[1]
OUT = sys.argv[2]
AMT = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0

y, sr = sf.read(IN)
if y.ndim == 1:
    y = np.stack([y, y], axis=1)
y = y.astype(np.float32)


def sat(x, drive):           # analog-style saturation (adds harmonic grit)
    return np.tanh(x * drive) / np.tanh(drive)


# 1) 808/low-end grit: distort the lows separately and blend back (heavy phonk bass)
sos_low = ss.butter(4, 200, btype="low", fs=sr, output="sos")
low = ss.sosfilt(sos_low, y, axis=0)
low_driven = sat(low, 3.5 * AMT)
y = y + (low_driven - low) * (0.6 * AMT)

# 2) overall tape saturation
y = sat(y, 1.6 + 0.8 * AMT)

# 3) bitcrush (digital grit): reduce bit depth
bits = 10 - int(2 * AMT)          # ~8-10 bits
q = 2 ** (bits - 1)
y = np.round(y * q) / q
# sample-rate reduction (sample-and-hold) for extra crunch
hold = 1 + int(2 * AMT)
if hold > 1:
    for c in range(2):
        held = y[::hold, c]
        y[:, c] = np.repeat(held, hold)[:len(y)]

# 4) cassette wobble: slow pitch/time modulation
t = np.arange(len(y))
lfo = 0.0006 * AMT * np.sin(2 * np.pi * 0.7 * t / sr) + 0.0006 * AMT * np.sin(2 * np.pi * 3.1 * t / sr)
idx = np.clip(t + lfo * sr, 0, len(y) - 1)
y = np.stack([np.interp(idx, t, y[:, c]) for c in range(2)], axis=1).astype(np.float32)

# 5) dark master (phonk is dark/lo-fi): low-pass the trap crispness
sos = ss.butter(4, 6000, btype="low", fs=sr, output="sos")
y = ss.sosfilt(sos, y, axis=0).astype(np.float32)

# 6) subtle vinyl hiss/crackle
noise = np.random.randn(*y.shape).astype(np.float32) * 0.0015 * AMT
crackle = (np.random.rand(*y.shape) > 0.9995).astype(np.float32) * np.random.randn(*y.shape) * 0.05 * AMT
y = y + noise + crackle

# normalize + gentle limit
y = sat(y, 1.1)
y = y / (np.max(np.abs(y)) + 1e-9) * 0.97

sf.write(OUT, y, sr)
mp3 = os.path.splitext(OUT)[0] + ".mp3"
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", OUT, "-b:a", "192k", mp3], capture_output=True)
print(f"phonkified -> {mp3} (intensity {AMT})", flush=True)
