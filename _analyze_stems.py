import numpy as np, librosa, scipy.signal as ss, warnings, os
warnings.filterwarnings("ignore")

def stats(p, sr=44100):
    y, _ = librosa.load(p, sr=sr, mono=True)
    low = ss.sosfilt(ss.butter(4, 150, btype="low", fs=sr, output="sos"), y)
    mid = ss.sosfilt(ss.butter(4, [300, 3000], btype="band", fs=sr, output="sos"), y)
    return (float(np.sqrt(np.mean(low**2))), float(np.sqrt(np.mean(mid**2))),
            float(np.sqrt(np.mean(y**2))), float(librosa.onset.onset_strength(y=y, sr=sr).mean()))

d6 = "/mnt/d/flbeat/data/grim_6s/htdemucs_6s/GRIM_3_raw"
files = {f"6s/{k}": f"{d6}/{k}.wav" for k in ("piano", "other", "guitar", "bass", "drums", "vocals")}
files["ft/other (old)"] = "/mnt/d/flbeat/data/FL_PACKS/GRIM_1to1/Melody.wav"

print("%-18s %9s %9s %9s %9s" % ("stem", "bass", "MID(mel)", "total", "onset"))
print("-" * 58)
for k, p in files.items():
    if not os.path.exists(p):
        print("%-18s  (missing)" % k); continue
    b, m, t, o = stats(p)
    print("%-18s %9.4f %9.4f %9.4f %9.3f" % (k, b, m, t, o))
