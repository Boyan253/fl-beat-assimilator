import librosa, numpy as np, soundfile as sf, glob, os
SR = 44100
files = [
 "/mnt/d/flbeat/data/generated/GRIM_3_GRIT.wav",
 "/mnt/d/flbeat/data/generated/GRIM_3_TIGHTBASS.wav",
 "/mnt/d/flbeat/data/generated/GRIM_3_raw.wav",
]
files += sorted(glob.glob("/mnt/d/flbeat/data/grim_roformer/GRIM_3*"))
for p in files:
    if not os.path.exists(p):
        print(f"MISSING {p}"); continue
    info = sf.info(p)
    y, _ = librosa.load(p, sr=SR, mono=True)
    tempo, beats = librosa.beat.beat_track(y=y, sr=SR)
    tempo = float(np.atleast_1d(tempo)[0])
    bar = 4 * 60.0 / tempo
    bars = info.duration / bar
    print(f"{os.path.basename(p):58s} {info.duration:6.1f}s {info.samplerate}Hz {info.channels}ch  tempo~{tempo:5.1f}  bar={bar:.3f}s  ~{bars:.1f} bars")
