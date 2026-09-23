import librosa, numpy as np, glob
from scipy.signal import butter, filtfilt
SR = 44100
beat = 60.0 / 126.0  # 0.476s per beat @126BPM
b, a = butter(4, [40/(SR/2), 120/(SR/2)], btype='band')
for p in sorted(glob.glob("/mnt/d/flbeat/data/generated/STAY_DHK_*_raw.wav")):
    y, _ = librosa.load(p, sr=SR, mono=True)
    peak = float(np.max(np.abs(y))); rms = float(np.sqrt(np.mean(y**2)))
    nan = bool(np.isnan(y).any())
    S = np.abs(librosa.stft(y, n_fft=4096))**2; fr = librosa.fft_frequencies(sr=SR, n_fft=4096)
    bass = float(S[fr < 150].sum() / (S.sum() or 1))
    tempo, _ = librosa.beat.beat_track(y=y, sr=SR); tempo = float(np.atleast_1d(tempo)[0])
    low = filtfilt(b, a, y).astype(np.float32)
    on = librosa.onset.onset_detect(y=low, sr=SR, backtrack=False, delta=0.02, wait=4, units='time')
    ioi = np.diff(on) if len(on) > 1 else np.array([0.0])
    medgap = float(np.median(ioi)) if len(ioi) else 0.0
    kick = "4-ON-FLOOR YES" if (abs(medgap - beat) < 0.09 and len(on) > 60) else "weak/none"
    print("%-20s peak %.2f rms %.3f(%3.0fdB) nan=%s bass %2.0f%% ~%3.0fBPM | lowOnsets=%3d medGap=%.3fs (beat=%.3f) -> %s" % (
        p.split('/')[-1], peak, rms, 20*np.log10(rms+1e-9), nan, bass*100, tempo, len(on), medgap, beat, kick))
