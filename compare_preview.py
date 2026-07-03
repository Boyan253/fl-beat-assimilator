"""Objectively compare the native-pitch render vs the original clean melody stem:
timbre (mel), pitch content (chroma), and timing (onset envelope)."""
import numpy as np, librosa

SR = 44100
ORIG = "/mnt/d/flbeat/data/grim_roformer/GRIM_melody_clean.wav"
REND = "/mnt/d/flbeat/data/generated/GRIM_PREVIEW_native.wav"

def load(p):
    y, _ = librosa.load(p, sr=SR, mono=True)
    return y

a = load(ORIG); b = load(REND)
n = min(len(a), len(b)); a = a[:n]; b = b[:n]

def corr(x, y):
    x = x.flatten(); y = y.flatten()
    x = (x - x.mean()); y = (y - y.mean())
    d = (np.linalg.norm(x) * np.linalg.norm(y)) or 1.0
    return float(np.dot(x, y) / d)

# timbre: log-mel
ma = librosa.power_to_db(librosa.feature.melspectrogram(y=a, sr=SR, n_mels=64))
mb = librosa.power_to_db(librosa.feature.melspectrogram(y=b, sr=SR, n_mels=64))
m = min(ma.shape[1], mb.shape[1])
mel_c = corr(ma[:,:m], mb[:,:m])

# pitch content: chroma
ca = librosa.feature.chroma_cqt(y=a, sr=SR)
cb = librosa.feature.chroma_cqt(y=b, sr=SR)
m = min(ca.shape[1], cb.shape[1])
chroma_c = corr(ca[:,:m], cb[:,:m])

# timing: onset envelope
oa = librosa.onset.onset_strength(y=a, sr=SR)
ob = librosa.onset.onset_strength(y=b, sr=SR)
m = min(len(oa), len(ob))
onset_c = corr(oa[:m], ob[:m])

print(f"timbre  (log-mel)  : {mel_c*100:5.1f}%")
print(f"pitch   (chroma)   : {chroma_c*100:5.1f}%")
print(f"timing  (onsets)   : {onset_c*100:5.1f}%")
print(f"OVERALL (avg)      : {(mel_c+chroma_c+onset_c)/3*100:5.1f}%")
