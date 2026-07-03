import numpy as np, librosa
SR=44100
ORIG="/mnt/d/flbeat/data/grim_roformer/GRIM_melody_clean.wav"
REF ="/mnt/d/flbeat/data/SEND/GRIM_NATIVE/GRIM_melody_REFERENCE.wav"
SFZ ="/mnt/d/flbeat/data/SEND/GRIM_NATIVE/GRIM_sfizz_test.wav"
def load(p): return librosa.load(p,sr=SR,mono=True)[0]
def corr(x,y):
    n=min(len(x),len(y)); x=x[:n]-np.mean(x[:n]); y=y[:n]-np.mean(y[:n])
    d=(np.linalg.norm(x)*np.linalg.norm(y)) or 1; return float((x*y).sum()/d)
def mel(p):
    y=load(p); return librosa.power_to_db(librosa.feature.melspectrogram(y=y,sr=SR,n_mels=64))
def melscore(a,b):
    A=mel(a);B=mel(b);m=min(A.shape[1],B.shape[1]);return corr(A[:,:m].flatten(),B[:,:m].flatten())*100
s=load(SFZ)
print(f"sfizz render: dur={len(s)/SR:.1f}s peak={np.max(np.abs(s)):.3f} rms={np.sqrt(np.mean(s**2)):.4f}")
# silence check: fraction of time below -50dB
frame=4410; n=len(s)//frame
rms_frames=np.array([np.sqrt(np.mean(s[i*frame:(i+1)*frame]**2)) for i in range(n)])
silent=np.mean(rms_frames < 10**(-50/20))
print(f"silent fraction (<-50dB): {silent*100:.1f}%")
print(f"sfizz vs ORIGINAL  (mel): {melscore(SFZ,ORIG):.1f}%")
print(f"sfizz vs REFERENCE (mel): {melscore(SFZ,REF):.1f}%   <- should be ~100 if round-robin reconstructs")
print(f"REFERENCE vs ORIGINAL(mel): {melscore(REF,ORIG):.1f}%")
