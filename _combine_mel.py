import numpy as np, soundfile as sf, librosa, warnings
warnings.filterwarnings("ignore")
d = "/mnt/d/flbeat/data/grim_6s/htdemucs_6s/GRIM_3_raw"
o, sr = librosa.load(f"{d}/other.wav", sr=44100, mono=False)
p, _ = librosa.load(f"{d}/piano.wav", sr=44100, mono=False)
mix = o + p
pk = np.max(np.abs(mix))
if pk > 0:
    mix = mix / pk * 0.97
out = "/mnt/d/flbeat/data/generated/GRIM_melody_6s_CLEAN.wav"
sf.write(out, mix.T if mix.ndim > 1 else mix, sr)
print("wrote", out)
