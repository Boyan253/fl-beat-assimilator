"""prep_hard_dataset.py — curate a HARD/aggressive phonk dataset (exclude slowed + cute
melodic 'anime phonk'), pick the loudest/darkest tracks by analysis, annotate for LoRA.

  python prep_hard_dataset.py [n=40]
Output -> /mnt/d/flbeat/ACE-Step-1.5/lora_phonk_hard  (inside repo = passes path-safety)
"""
import os, sys, glob, json, shutil, warnings
warnings.filterwarnings("ignore")
import numpy as np, librosa

N   = int(sys.argv[1]) if len(sys.argv) > 1 else 40
SRC = "/mnt/d/flbeat/data/train"
OUT = "/mnt/d/flbeat/ACE-Step-1.5/lora_phonk_hard"
os.makedirs(OUT, exist_ok=True)

# cute/melodic 'isekai anime phonk' names to exclude (these made it light)
CUTE = {"pink cloud", "colors", "fire emblem", "rough days", "isekai", "kawaii desu ne",
        "nuggets", "sazon", "lychee", "memories", "adventures", "suspicious", "pops",
        "heroes", "final battle"}
CAPTION = "hard aggressive phonk, heavy distorted 808 bass, loud cowbell melody, banging memphis drums, dark, brutal, drift phonk"


def hardness(path):
    """Score = loudness * darkness * low-end (hard phonk = loud, dark, bass-heavy)."""
    try:
        y, sr = librosa.load(path, sr=22050, mono=True, duration=60)
    except Exception:
        return None
    if len(y) < sr * 5:
        return None
    rms = float(np.sqrt(np.mean(y ** 2)))
    cent = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
    S = np.abs(librosa.stft(y)).mean(axis=1)
    freqs = librosa.fft_frequencies(sr=sr)
    low = float(S[(freqs >= 20) & (freqs < 150)].mean())
    return rms * (1.0 / (cent + 1e-6)) * (low + 1e-6) * 1e6


cands = []
for mp3 in sorted(glob.glob(os.path.join(SRC, "*.mp3"))):
    base = os.path.splitext(os.path.basename(mp3))[0].lower()
    if "slow" in base or base in CUTE:
        continue
    s = hardness(mp3)
    if s is not None:
        cands.append((s, mp3))
cands.sort(reverse=True)
picked = cands[:N]
print(f"candidates {len(cands)}, picking hardest {len(picked)}", flush=True)

for i, (score, mp3) in enumerate(picked):
    base = f"hard_{i+1:02d}"
    try:
        y, sr = librosa.load(mp3, sr=22050, mono=True, duration=60)
        tempo = float(np.atleast_1d(librosa.beat.beat_track(y=y, sr=sr)[0])[0])
        bpm = int(round(tempo)) if 60 < tempo < 220 else 150
    except Exception:
        bpm = 150
    shutil.copy(mp3, os.path.join(OUT, base + ".mp3"))
    open(os.path.join(OUT, base + ".lyrics.txt"), "w").write("[Instrumental]")
    json.dump({"caption": CAPTION, "bpm": bpm, "language": "unknown", "timesignature": "4"},
              open(os.path.join(OUT, base + ".json"), "w"))
    print(f"  {base}: {os.path.basename(mp3)[:40]} (score {score:.1f}, bpm {bpm})", flush=True)
print(f"done -> {OUT}", flush=True)
