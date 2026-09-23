import librosa, numpy as np, sys
SR = 44100
for p in sys.argv[1:]:
    y, _ = librosa.load(p, sr=SR, mono=True)
    peak = float(np.max(np.abs(y))); rms = float(np.sqrt(np.mean(y**2)))
    S = np.abs(librosa.stft(y, n_fft=4096))**2; fr = librosa.fft_frequencies(sr=SR, n_fft=4096)
    bass = float(S[fr < 120].sum() / (S.sum() or 1))
    clip = float(np.mean(np.abs(y) > 0.985) * 100)
    print("%-28s peak %.3f  rms %.3f (%5.1f dB)  bass<120Hz %4.1f%%  clipped-samples %.3f%%" % (
        p.split("/")[-1], peak, rms, 20*np.log10(rms + 1e-9), bass * 100, clip))
