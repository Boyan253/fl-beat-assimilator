import numpy as np, soundfile as sf, librosa
SR = 48000
p = "/mnt/d/flbeat/data/SEND/GRIM3_SneakerS/GRIM_3_EXTENDED_MASTERED.wav"
y, sr = sf.read(p, always_2d=True)
mono = y.mean(axis=1)
peak = float(np.max(np.abs(y)))
clip = float(np.mean(np.abs(y) > 0.999) * 100)
secs = [("intro mel",0,15.6),("verse1",15.6,46.8),("verse2",46.8,78.0),
        ("BREAK1",78.0,93.7),("verse3",93.7,124.9),("verse4",124.9,156.1),
        ("BREAK2",156.1,171.7),("outro",171.7,187.3)]
def band(sig):
    S = np.abs(librosa.stft(sig, n_fft=4096))**2; fr = librosa.fft_frequencies(sr=SR, n_fft=4096)
    return float(S[fr < 120].sum() / (S.sum() + 1e-9))
print("dur %.1fs  peak %.3f  clipped %.3f%%" % (len(mono)/SR, peak, clip))
for lbl, a, b in secs:
    seg = mono[int(a*SR):int(b*SR)]
    r = np.sqrt(np.mean(seg**2)) + 1e-9
    print("%-9s %5.1f-%5.1fs  rms %6.1f dB  bass<120Hz %4.1f%%" % (lbl, a, b, 20*np.log10(r), band(seg)*100))
