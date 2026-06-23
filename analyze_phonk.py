"""analyze_phonk.py — measure what separates real phonk from my (trap-leaning) output.

Compares audio features across reference phonk tracks vs generated tracks:
  tempo, brightness, 808/low energy, cowbell/high-mid energy, onset density,
  spectral contrast, and — key — STRUCTURE VARIATION (does the track have different
  parts, or is it one loop?).

  python analyze_phonk.py
"""
import os, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np, librosa

REAL = "/mnt/d/flbeat/data/train"          # real phonk songs
GEN  = "/mnt/d/flbeat/data/generated"      # my outputs
SR = 22050


def band_energy(S, freqs, lo, hi):
    m = (freqs >= lo) & (freqs < hi)
    return float(S[m].mean())


def features(path):
    try:
        y, sr = librosa.load(path, sr=SR, mono=True, duration=90)
    except Exception:
        return None
    if len(y) < sr * 5:
        return None
    tempo = float(np.atleast_1d(librosa.beat.beat_track(y=y, sr=sr)[0])[0])
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    Smean = S.mean(axis=1)
    low  = band_energy(Smean[:, None], freqs, 20, 120)     # 808/sub
    mid  = band_energy(Smean[:, None], freqs, 300, 2000)
    high = band_energy(Smean[:, None], freqs, 3000, 9000)  # cowbell/hat region
    cent = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
    contrast = float(librosa.feature.spectral_contrast(y=y, sr=sr).mean())
    onset_rate = len(librosa.onset.onset_detect(y=y, sr=sr, units="time")) / (len(y) / sr)

    # STRUCTURE VARIATION: split into 12 windows, feature vector per window, measure spread
    win = len(y) // 12
    vecs = []
    for i in range(12):
        seg = y[i * win:(i + 1) * win]
        if len(seg) < sr:
            continue
        v = np.concatenate([
            [librosa.feature.rms(y=seg).mean()],
            [librosa.feature.spectral_centroid(y=seg, sr=sr).mean() / 5000],
            librosa.feature.chroma_cqt(y=seg, sr=sr).mean(axis=1),
        ])
        vecs.append(v)
    vecs = np.array(vecs)
    # normalized variation = mean std across windows (higher = more distinct parts)
    struct_var = float(np.mean(np.std(vecs, axis=0)))
    rms_dyn = float(np.std(librosa.feature.rms(y=y)[0]))  # loudness movement over time

    return dict(tempo=tempo, cent=cent, contrast=contrast, onset=onset_rate,
                low=low, high=high, hi_lo=high / (low + 1e-9),
                struct=struct_var, dyn=rms_dyn)


def summarize(paths, label):
    rows = [features(p) for p in paths]
    rows = [r for r in rows if r]
    if not rows:
        print(f"{label}: no tracks"); return None
    keys = rows[0].keys()
    avg = {k: np.mean([r[k] for r in rows]) for k in keys}
    print(f"\n=== {label} (n={len(rows)}) ===")
    print(f"  tempo {avg['tempo']:.0f} | brightness {avg['cent']:.0f}Hz | contrast {avg['contrast']:.1f}")
    print(f"  onset/s {avg['onset']:.1f} | high/low ratio {avg['hi_lo']:.2f}")
    print(f"  STRUCTURE-variation {avg['struct']:.4f} | dynamics {avg['dyn']:.4f}")
    return avg


real = sorted(glob.glob(os.path.join(REAL, "*.mp3")))[:10]
mine = [os.path.join(GEN, f) for f in
        ["PHONK_full_v2.wav", "PHONK_full_looped.wav", "PHONK_finetuned_01.wav", "XL_phonk_test.wav"]
        if os.path.exists(os.path.join(GEN, f))]

a = summarize(real, "REAL PHONK (training songs)")
b = summarize(mine, "MY GENERATED")
if a and b:
    print("\n=== KEY GAPS (mine vs real) ===")
    for k, nm in [("struct", "structure/different-parts"), ("hi_lo", "cowbell-vs-808 balance"),
                  ("contrast", "spectral punch"), ("dyn", "dynamics over time"), ("onset", "rhythm density")]:
        print(f"  {nm}: mine {b[k]:.3f} vs real {a[k]:.3f}  ({'LOW' if b[k] < a[k] else 'high'})")
