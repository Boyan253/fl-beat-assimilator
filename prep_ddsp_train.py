"""Prep GRIM's melody as DDSP training audio: mono, 16kHz, silence-trimmed, normalized, padded to
~4 min (DDSP learns the TIMBRE, so the looped melody is fine training material)."""
import warnings; warnings.filterwarnings("ignore")
import os, numpy as np, soundfile as sf, librosa
IN  = "/mnt/d/flbeat/data/generated/GRIM_melody_6s_CLEAN.wav"
OUT = "/mnt/d/flbeat/data/SEND/GRIM_DDSP_train/GRIM_melody_train.wav"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
y, sr = librosa.load(IN, sr=16000, mono=True)
y, _ = librosa.effects.trim(y, top_db=30)
y = y / (np.max(np.abs(y)) + 1e-9) * 0.95
tgt = 4 * 60 * 16000
if len(y) < tgt:
    y = np.tile(y, int(np.ceil(tgt / len(y))))[:tgt]
sf.write(OUT, y.astype(np.float32), 16000)
print(f"DDSP train audio: {len(y)/16000:.0f}s mono 16k -> {OUT}", flush=True)
