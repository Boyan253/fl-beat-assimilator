"""prep_lora_dataset.py — build an ACE-Step LoRA dataset from phonk tracks.

For N phonk mp3s: copy to dataset dir + write .lyrics.txt ([Instrumental]) + .json
annotation (caption, bpm via librosa, language). EDUCATIONAL test data (not for selling).

  python prep_lora_dataset.py [n=16] [src_dir] [out_dir]
"""
import os, sys, glob, json, shutil, warnings
warnings.filterwarnings("ignore")
import librosa, numpy as np

N   = int(sys.argv[1]) if len(sys.argv) > 1 else 16
SRC = sys.argv[2] if len(sys.argv) > 2 else "/mnt/d/flbeat/data/train"
OUT = sys.argv[3] if len(sys.argv) > 3 else "/mnt/d/flbeat/data/lora_phonk"
os.makedirs(OUT, exist_ok=True)

CAPTION = "dark phonk, distorted 808 bass, cowbell melody, memphis trap drums, lo-fi, aggressive, hypnotic"

mp3s = sorted(glob.glob(os.path.join(SRC, "*.mp3")))[:N]
print(f"preparing {len(mp3s)} tracks -> {OUT}", flush=True)
ok = 0
for i, mp3 in enumerate(mp3s):
    base = f"phonk_{i+1:02d}"
    try:
        y, sr = librosa.load(mp3, sr=22050, mono=True, duration=90)
        tempo = float(librosa.beat.beat_track(y=y, sr=sr)[0])
        bpm = int(round(tempo)) if 60 < tempo < 220 else 150
    except Exception:
        bpm = 150
    shutil.copy(mp3, os.path.join(OUT, base + ".mp3"))
    open(os.path.join(OUT, base + ".lyrics.txt"), "w").write("[Instrumental]")
    json.dump({"caption": CAPTION, "bpm": bpm, "language": "unknown", "timesignature": "4"},
              open(os.path.join(OUT, base + ".json"), "w"))
    print(f"  {base}: bpm~{bpm}", flush=True)
    ok += 1
print(f"done: {ok} tracks annotated in {OUT}", flush=True)
