import numpy as np, librosa
SR=44100
REF="/mnt/d/flbeat/data/SEND/GRIM_NATIVE/GRIM_melody_REFERENCE.wav"
T="/mnt/d/flbeat/data/SEND/GRIM_NATIVE/GRIM_velkey_test.wav"
def L(p): return librosa.load(p,sr=SR,mono=True)[0]
r=L(REF); t=L(T)
# onset counts (notes present?)
for nm,y in [("REFERENCE",r),("velkey",t)]:
    on=librosa.onset.onset_detect(y=y,sr=SR,backtrack=False,delta=0.03,wait=2)
    print(f"{nm}: onsets={len(on)} peak={np.max(np.abs(y)):.3f} rms={np.sqrt(np.mean(y**2)):.4f}")
# normalize both and re-score mel + silence
def norm(y): return y/(np.max(np.abs(y)) or 1)*0.9
def cor(x,y):
    n=min(len(x),len(y)); x=x[:n]-x[:n].mean(); y=y[:n]-y[:n].mean()
    d=(np.linalg.norm(x)*np.linalg.norm(y)) or 1; return float((x*y).sum()/d)
rn=norm(r); tn=norm(t)
ma=librosa.power_to_db(librosa.feature.melspectrogram(y=rn,sr=SR,n_mels=64))
mb=librosa.power_to_db(librosa.feature.melspectrogram(y=tn,sr=SR,n_mels=64))
m=min(ma.shape[1],mb.shape[1])
print(f"normalized velkey vs REFERENCE mel = {cor(ma[:,:m].flatten(),mb[:,:m].flatten())*100:.1f}%")
fr=4410;n=len(tn)//fr
sil=np.mean(np.array([np.sqrt(np.mean(tn[i*fr:(i+1)*fr]**2)) for i in range(n)])<10**(-50/20))*100
print(f"normalized velkey silent = {sil:.1f}%")
