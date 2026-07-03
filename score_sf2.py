import numpy as np, librosa
SR=44100
ORIG="/mnt/d/flbeat/data/grim_roformer/GRIM_melody_clean.wav"
REND="/mnt/d/flbeat/data/SEND/GRIM_NATIVE/GRIM_SF2_render.wav"
a=librosa.load(ORIG,sr=SR,mono=True)[0]; b=librosa.load(REND,sr=SR,mono=True)[0]
print(f"render dur={len(b)/SR:.1f}s peak={np.max(np.abs(b)):.3f} rms={np.sqrt(np.mean(b**2)):.4f}")
n=min(len(a),len(b)); a=a[:n]; b=b[:n]
def corr(x,y):
    x=x-x.mean(); y=y-y.mean(); d=(np.linalg.norm(x)*np.linalg.norm(y)) or 1; return float((x*y).sum()/d)
ma=librosa.power_to_db(librosa.feature.melspectrogram(y=a,sr=SR,n_mels=64))
mb=librosa.power_to_db(librosa.feature.melspectrogram(y=b,sr=SR,n_mels=64))
m=min(ma.shape[1],mb.shape[1])
print(f"timbre(mel) vs original = {corr(ma[:,:m].flatten(),mb[:,:m].flatten())*100:.1f}%")
