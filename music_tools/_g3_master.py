import numpy as np, soundfile as sf, librosa
SRC = "/mnt/d/flbeat/data/generated/GRIM_3_TIGHTBASS.wav"
VOC = "/mnt/d/flbeat/data/SEND/GRIM3_SneakerS/GRIM_3_VOCALREADY.wav"
MST = "/mnt/d/flbeat/data/SEND/GRIM3_SneakerS/GRIM_3_MASTERED.wav"
y, sr = sf.read(SRC)          # keep stereo, native SR
if y.ndim == 1: y = y[:, None]

# --- VOCAL READY: just set peak to -6 dBFS, no processing (headroom for the vocal mix) ---
peak = np.max(np.abs(y))
voc = y * (10 ** (-6/20) / peak)
sf.write(VOC, voc, sr, subtype="PCM_24")

# --- MASTER: gain toward target RMS, soft-knee tanh limiter, true-peak-ish -1 dB ---
mono = y.mean(axis=1)
rms = np.sqrt(np.mean(mono**2))
target_rms = 10 ** (-8.5/20)            # phonk-loud but not destroyed
g = min(target_rms / (rms + 1e-9), 6.0) # cap the push
x = y * g
drive = 1.15
m = np.tanh(x * drive) / np.tanh(drive) # soft limiter, gentle knee
# 4x oversampled peak check for inter-sample peaks
os4 = librosa.resample(m.T, orig_sr=sr, target_sr=sr*4).T
tp = np.max(np.abs(os4))
m = m * (10 ** (-1/20) / max(tp, 1e-9))
sf.write(MST, m, sr, subtype="PCM_24")

for name, a in (("VOCALREADY", voc), ("MASTERED", m)):
    mm = a.mean(axis=1); r = np.sqrt(np.mean(mm**2))
    print("%-11s peak %.3f  rms %6.1f dB" % (name, np.max(np.abs(a)), 20*np.log10(r + 1e-9)))
print("sr =", sr)
