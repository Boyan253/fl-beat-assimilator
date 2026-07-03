"""Prep GRIM melody for RAVE training: 44.1k mono, trimmed, normalized, looped to ~8 min so RAVE
has enough data to learn the timbre."""
import warnings; warnings.filterwarnings("ignore")
import os, numpy as np, soundfile as sf, librosa
IN = "/mnt/d/flbeat/data/generated/GRIM_melody_6s_CLEAN.wav"
OUTDIR = "/opt/rave_data/GRIM/audio"
os.makedirs(OUTDIR, exist_ok=True)
y, sr = librosa.load(IN, sr=44100, mono=True)
y, _ = librosa.effects.trim(y, top_db=30)
y = y / (np.max(np.abs(y)) + 1e-9) * 0.95
tgt = 8 * 60 * 44100
if len(y) < tgt:
    y = np.tile(y, int(np.ceil(tgt / len(y))))[:tgt]
sf.write(os.path.join(OUTDIR, "GRIM_melody.wav"), y.astype(np.float32), 44100)
print(f"rave audio: {len(y)/44100:.0f}s 44.1k mono -> {OUTDIR}", flush=True)
