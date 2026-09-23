"""prep_memphis.py — curate Memphis Cult tracks into an ACE-Step LoRA dataset (in repo)."""
import os, glob, json, shutil, warnings, random
warnings.filterwarnings("ignore")
import librosa, numpy as np

N   = 80
OUT = "/mnt/d/flbeat/ACE-Step-1.5/memphis_cult"
os.makedirs(OUT, exist_ok=True)
SRCS = glob.glob("/mnt/d/flbeat/data/refs/*/") + ["/mnt/d/flbeat/data/groove_dealers/"]
CAPTION = "aggressive memphis phonk, distorted 808 bass, dark menacing cowbell, gritty lo-fi, memphis rap vocal chops, hard hitting, hypnotic"

# round-robin across albums for variety
pools = [sorted(glob.glob(os.path.join(s, "*.mp3"))) for s in SRCS]
picked = []
i = 0
while len(picked) < N and any(pools):
    p = pools[i % len(pools)]
    if p:
        picked.append(p.pop(0))
    i += 1
    if i > 10000:
        break
print(f"picked {len(picked)} tracks from {len(SRCS)} albums", flush=True)

ok = 0
for j, mp3 in enumerate(picked):
    base = f"mc_{j+1:02d}"
    try:
        y, sr = librosa.load(mp3, sr=22050, mono=True, duration=60)
        tempo = float(np.atleast_1d(librosa.beat.beat_track(y=y, sr=sr)[0])[0])
        bpm = int(round(tempo)) if 60 < tempo < 220 else 130
    except Exception:
        bpm = 130
    shutil.copy(mp3, os.path.join(OUT, base + ".mp3"))
    open(os.path.join(OUT, base + ".lyrics.txt"), "w").write("[Instrumental]")
    json.dump({"caption": CAPTION, "bpm": bpm, "language": "unknown", "timesignature": "4"},
              open(os.path.join(OUT, base + ".json"), "w"))
    ok += 1
print(f"done: {ok} tracks -> {OUT}", flush=True)
