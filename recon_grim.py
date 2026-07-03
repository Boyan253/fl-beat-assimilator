"""TRUE 1:1 path: slice the stem into ALL real note-hits, detect each slice's pitch
(torchcrepe), reconstruct by playing the real slices in song order (=1:1 audio), and
emit a MIDI mapped to true pitches so the piano roll looks like a real melody.
Outputs: reconstruction WAV, MIDI, piano-roll PNG, per-slice samples, and a score."""
import os, numpy as np, soundfile as sf, librosa
SR=44100
STEM="/mnt/d/flbeat/data/grim_roformer/GRIM_melody_clean.wav"
OUTDIR="/mnt/d/flbeat/data/generated/GRIM_slices"; os.makedirs(OUTDIR,exist_ok=True)
RECON="/mnt/d/flbeat/data/generated/GRIM_RECON_slices.wav"
MIDOUT="/mnt/d/flbeat/data/generated/GRIM_mel_recon.mid"
PNG="/mnt/d/flbeat/data/generated/GRIM_pianoroll.png"
def nm(p):
    n=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];return f"{n[p%12]}{p//12-1}"

# ---- load ----
y,_=sf.read(STEM,always_2d=True)
if y.shape[1]==1: y=np.repeat(y,2,axis=1)
y=y.astype(np.float32)
mono=y.mean(axis=1)

# ---- onsets (percussive, backtracked to attack) ----
yh,yp=librosa.effects.hpss(mono)
on_f=librosa.onset.onset_detect(y=yp,sr=SR,backtrack=True,units='frames',
                                hop_length=512,delta=0.05,wait=2)
on=librosa.frames_to_samples(on_f,hop_length=512)
on=np.unique(np.concatenate([[0],on,[len(mono)]]))
bounds=[(on[i],on[i+1]) for i in range(len(on)-1) if on[i+1]-on[i]>int(0.03*SR)]
print(f"slices={len(bounds)}",flush=True)

# ---- pitch per slice via librosa.pyin on a HIGH-PASSED copy (kills sub-bass bleed -> fewer octave errors) ----
print("pyin...",flush=True)
from scipy.signal import butter, filtfilt
_b,_a=butter(4,150.0/(SR/2),btype='high'); mono_hp=filtfilt(_b,_a,mono).astype(np.float32)
f0,vflag,vprob=librosa.pyin(mono_hp,fmin=165,fmax=1600,sr=SR,frame_length=2048,hop_length=256)
fr_t=librosa.times_like(f0,sr=SR,hop_length=256)
def pitch_of(st,en):
    i0=np.searchsorted(fr_t,st); i1=max(i0+1,np.searchsorted(fr_t,en))
    seg=f0[i0:i1]; seg=seg[~np.isnan(seg)]
    if len(seg)<1: return None
    hz=float(np.median(seg))
    if hz<=0: return None
    return int(round(69+12*np.log2(hz/440.0)))

# ---- reconstruct (real slices in order) + collect notes + per-pitch slices ----
recon=np.zeros((len(mono)+SR,2),dtype=np.float32)
notes=[]; xf=int(0.004*SR)
for (a,b) in bounds:
    seg=y[a:b].copy()
    if xf>0 and len(seg)>2*xf:
        seg[:xf]*=np.linspace(0,1,xf)[:,None]; seg[-xf:]*=np.linspace(1,0,xf)[:,None]
    recon[a:a+len(seg)]+=seg
    p=pitch_of(a/SR,b/SR)
    if p is None or p<24 or p>108: continue
    notes.append((p,a/SR,b/SR,float(np.sqrt(np.mean(seg**2)))))
peak=np.max(np.abs(recon)) or 1.0
sf.write(RECON,recon/peak*0.92,SR)
print(f"WROTE {RECON}",flush=True)

# save per-distinct-pitch one clean sample (loudest) for the FL instrument later
byp={}
for (a,b) in bounds:
    p=pitch_of(a/SR,b/SR)
    if p is None: continue
    byp.setdefault(p,[]).append((a,b))
for p,segs in byp.items():
    a,b=max(segs,key=lambda s:np.sqrt(np.mean(y[s[0]:s[1]]**2)))
    s=y[a:min(b,a+int(1.2*SR))].copy()
    fo=min(2048,len(s)//4)
    if fo>0: s[-fo:]*=np.linspace(1,0,fo)[:,None]
    sf.write(os.path.join(OUTDIR,f"GRIM_{nm(p).replace('#','s')}.wav"),s,SR)
pitches=sorted(byp); print(f"distinct pitches={len(pitches)}: {[nm(p) for p in pitches]}",flush=True)
seq=[n[0] for n in notes]
print(f"notes={len(notes)} STAIRCASE={all(seq[i]<=seq[i+1] for i in range(len(seq)-1))}",flush=True)

# ---- MIDI ----
import pretty_midi
pm=pretty_midi.PrettyMIDI(); inst=pretty_midi.Instrument(program=0)
amps=[n[3] for n in notes]; lo,hi=min(amps),max(amps) if amps else (0,1)
for (p,st,en,a) in notes:
    v=int(np.clip(np.interp(a,[lo,hi],[50,127]),1,127))
    inst.notes.append(pretty_midi.Note(velocity=v,pitch=p,start=st,end=max(en,st+0.05)))
pm.instruments.append(inst); pm.write(MIDOUT); print(f"WROTE {MIDOUT}",flush=True)

# ---- piano-roll PNG ----
try:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(16,5))
    for (p,st,en,a) in notes:
        ax.add_patch(plt.Rectangle((st,p-0.4),max(en-st,0.05),0.8,
                     color=plt.cm.viridis((a-lo)/(hi-lo+1e-9))))
    ax.set_xlim(0,min(32,notes[-1][2])); ax.set_ylim(min(seq)-2,max(seq)+2)
    ax.set_xlabel("time (s)"); ax.set_ylabel("MIDI note")
    yt=sorted(set(seq)); ax.set_yticks(yt); ax.set_yticklabels([nm(p) for p in yt],fontsize=7)
    ax.set_title("GRIM melody — reconstructed from REAL slices, mapped to true pitches (first 32s)")
    ax.grid(True,alpha=0.2); plt.tight_layout(); plt.savefig(PNG,dpi=90)
    print(f"WROTE {PNG}",flush=True)
except Exception as e:
    print(f"PNG skip: {e}",flush=True)

# ---- score (reconstruction should be ~original) ----
def corr(x,yv):
    x=x.flatten()-x.mean(); yv=yv.flatten()-yv.mean()
    d=(np.linalg.norm(x)*np.linalg.norm(yv)) or 1.0; return float(np.dot(x,yv)/d)
A=librosa.load(STEM,sr=SR,mono=True)[0]; B=librosa.load(RECON,sr=SR,mono=True)[0]
n=min(len(A),len(B)); A=A[:n]; B=B[:n]
ma=librosa.power_to_db(librosa.feature.melspectrogram(y=A,sr=SR,n_mels=64))
mb=librosa.power_to_db(librosa.feature.melspectrogram(y=B,sr=SR,n_mels=64))
m=min(ma.shape[1],mb.shape[1])
print(f"=== RECON score vs original: timbre(mel)={corr(ma[:,:m],mb[:,:m])*100:.1f}% (should be high = real audio) ===",flush=True)
