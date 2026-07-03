import numpy as np, librosa
SR=44100
ORIG="/mnt/d/flbeat/data/grim_roformer/GRIM_melody_clean.wav"
REF="/mnt/d/flbeat/data/SEND/GRIM_NATIVE/GRIM_melody_REFERENCE.wav"
T="/mnt/d/flbeat/data/SEND/GRIM_NATIVE/GRIM_velkey_test.wav"
def L(p): return librosa.load(p,sr=SR,mono=True)[0]
def cor(x,y):
    n=min(len(x),len(y)); x=x[:n]-x[:n].mean(); y=y[:n]-y[:n].mean()
    d=(np.linalg.norm(x)*np.linalg.norm(y)) or 1; return float((x*y).sum()/d)
def mel(p):
    y=L(p); return librosa.power_to_db(librosa.feature.melspectrogram(y=y,sr=SR,n_mels=64))
def ms(a,b):
    A=mel(a);B=mel(b);m=min(A.shape[1],B.shape[1]);return cor(A[:,:m].flatten(),B[:,:m].flatten())*100
t=L(T); fr=4410; n=len(t)//fr
sil=np.mean(np.array([np.sqrt(np.mean(t[i*fr:(i+1)*fr]**2)) for i in range(n)])<10**(-50/20))*100
print(f"velkey: dur={len(t)/SR:.1f}s peak={np.max(np.abs(t)):.3f} silent={sil:.1f}%")
print(f"velkey vs REFERENCE(99.8) = {ms(T,REF):.1f}%")
print(f"velkey vs ORIGINAL        = {ms(T,ORIG):.1f}%")
