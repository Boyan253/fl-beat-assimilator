import librosa, numpy as np, glob
SR = 44100
for p in sorted(glob.glob("/mnt/d/flbeat/data/generated/STAY_DH_*_raw.wav")):
    y, _ = librosa.load(p, sr=SR, mono=True)
    peak = float(np.max(np.abs(y)) or 1e-9); rms = float(np.sqrt(np.mean(y**2)))
    S = np.abs(librosa.stft(y, n_fft=4096))**2; fr = librosa.fft_frequencies(sr=SR, n_fft=4096)
    bass = float(S[fr < 150].sum() / (S.sum() or 1))
    tempo, _ = librosa.beat.beat_track(y=y, sr=SR); tempo = float(np.atleast_1d(tempo)[0])
    print("%-22s dur %2.0fs  peak %.2f  rms %.3f (%.1f dB)  bass<150Hz %2.0f%%  ~%.0f BPM" % (
        p.split("/")[-1], len(y)/SR, peak, rms, 20*np.log10(rms+1e-9), bass*100, tempo))
