"""Build the full native-FL GRIM melody pack from the clean stem:
 - slice into real note-hits, detect pitch (high-passed pyin)
 - save ALL slices (round-robin, exact 1:1) + one clean sample per pitch (clean edit)
 - write TWO pitch_keytrack=0 SFZs (round-robin / one-per-pitch) for DirectWave
 - write the musical MIDI + a reference reconstruction WAV
All keytracking OFF -> every key plays the REAL recorded note at native pitch."""
import os, shutil, numpy as np, soundfile as sf, librosa
from scipy.signal import butter, filtfilt
SR=44100
STEM="/mnt/d/flbeat/data/grim_roformer/GRIM_melody_clean.wav"
ROOT="/mnt/d/flbeat/data/SEND/GRIM_NATIVE"
RR=os.path.join(ROOT,"slices_RR"); KEYS=os.path.join(ROOT,"samples_keys")
for d in (ROOT,RR,KEYS):
    shutil.rmtree(d,ignore_errors=True); os.makedirs(d)
def nm(p):
    n=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];return f"{n[p%12]}{p//12-1}"
def safe(p): return nm(p).replace('#','s')

# ---- load + slice ----
y,_=sf.read(STEM,always_2d=True)
if y.shape[1]==1: y=np.repeat(y,2,axis=1)
y=y.astype(np.float32); mono=y.mean(axis=1)
yh,yp=librosa.effects.hpss(mono)
on=librosa.frames_to_samples(librosa.onset.onset_detect(y=yp,sr=SR,backtrack=True,units='frames',hop_length=512,delta=0.05,wait=2),hop_length=512)
on=np.unique(np.concatenate([[0],on,[len(mono)]]))
bounds=[(on[i],on[i+1]) for i in range(len(on)-1) if on[i+1]-on[i]>int(0.03*SR)]

# ---- pitch (high-passed pyin) ----
_b,_a=butter(4,150.0/(SR/2),btype='high'); mhp=filtfilt(_b,_a,mono).astype(np.float32)
f0,_,_=librosa.pyin(mhp,fmin=165,fmax=1600,sr=SR,frame_length=2048,hop_length=256)
ft=librosa.times_like(f0,sr=SR,hop_length=256)
def pitch_of(a,b):
    i0=np.searchsorted(ft,a/SR); i1=max(i0+1,np.searchsorted(ft,b/SR))
    seg=f0[i0:i1]; seg=seg[~np.isnan(seg)]
    if len(seg)<1: return None
    return int(round(69+12*np.log2(float(np.median(seg))/440.0)))

# ---- collect notes (pitch, bounds, order) ----
notes=[]
for (a,b) in bounds:
    p=pitch_of(a,b)
    if p is None or p<28 or p>100: continue
    notes.append({'p':p,'a':a,'b':b,'rms':float(np.sqrt(np.mean(y[a:b]**2)))})
notes.sort(key=lambda n:n['a'])
pitches=sorted(set(n['p'] for n in notes))

# ---- save ALL slices (round-robin) with per-pitch seq order ----
seqcount={p:0 for p in pitches}
for i,n in enumerate(notes):
    p=n['p']; seqcount[p]+=1; n['seq']=seqcount[p]
    s=y[n['a']:min(n['b'],n['a']+int(1.2*SR))].copy()
    fo=min(1536,len(s)//4)
    if fo>0: s[-fo:]*=np.linspace(1,0,fo)[:,None]
    fn=f"rr_{i:04d}_{safe(p)}_{n['seq']:03d}.wav"; n['file']=fn
    sf.write(os.path.join(RR,fn),s,SR)

# ---- save ONE clean sample per pitch (loudest, full) ----
keyfile={}
for p in pitches:
    best=max([n for n in notes if n['p']==p],key=lambda n:n['rms'])
    s=y[best['a']:min(best['b'],best['a']+int(1.2*SR))].copy()
    fo=min(2048,len(s)//4)
    if fo>0: s[-fo:]*=np.linspace(1,0,fo)[:,None]
    fn=f"GRIM_{safe(p)}.wav"; keyfile[p]=fn
    sf.write(os.path.join(KEYS,fn),s,SR)

# ---- SFZ #1: round-robin (EXACT 1:1) ----
with open(os.path.join(ROOT,"GRIM_melody_EXACT.sfz"),"w") as f:
    f.write("// GRIM melody - EXACT 1:1. Every real slice, native pitch (keytrack OFF),\n")
    f.write("// round-robin per key in song order. Drag into FL DirectWave (set KTRK=0 if needed).\n")
    f.write("<global> loop_mode=one_shot ampeg_release=0.08 pitch_keytrack=0 amp_veltrack=70\n")
    for p in pitches:
        grp=[n for n in notes if n['p']==p]; L=min(len(grp),100)
        f.write(f"\n<group> key={p} seq_length={L} // {nm(p)} x{len(grp)}\n")
        for n in grp[:100]:
            f.write(f"<region> sample=slices_RR/{n['file']} seq_position={n['seq']}\n")

# ---- SFZ #2: one-per-pitch (clean edit) ----
with open(os.path.join(ROOT,"GRIM_melody_CLEAN.sfz"),"w") as f:
    f.write("// GRIM melody - CLEAN (one real note per key, native pitch, keytrack OFF).\n")
    f.write("<global> loop_mode=one_shot ampeg_release=0.12 pitch_keytrack=0 amp_veltrack=60\n")
    for p in pitches:
        f.write(f"<region> sample=samples_keys/{keyfile[p]} key={p} // {nm(p)}\n")

# ---- MIDI (musical) ----
import pretty_midi
pm=pretty_midi.PrettyMIDI(); inst=pretty_midi.Instrument(program=0)
amps=[n['rms'] for n in notes]; lo,hi=min(amps),max(amps)
for n in notes:
    v=int(np.clip(np.interp(n['rms'],[lo,hi],[50,127]),1,127))
    st=n['a']/SR; en=n['b']/SR
    inst.notes.append(pretty_midi.Note(velocity=v,pitch=n['p'],start=st,end=max(en,st+0.05)))
pm.instruments.append(inst); pm.write(os.path.join(ROOT,"GRIM_melody.mid"))

# ---- reference reconstruction WAV (what EXACT should sound like) ----
recon=np.zeros((len(mono)+SR,2),dtype=np.float32); xf=int(0.004*SR)
for (a,b) in bounds:
    s=y[a:b].copy()
    if len(s)>2*xf: s[:xf]*=np.linspace(0,1,xf)[:,None]; s[-xf:]*=np.linspace(1,0,xf)[:,None]
    recon[a:a+len(s)]+=s
sf.write(os.path.join(ROOT,"GRIM_melody_REFERENCE.wav"),recon/(np.max(np.abs(recon))or 1)*0.92,SR)

# ---- README ----
with open(os.path.join(ROOT,"README.txt"),"w") as f:
    f.write("GRIM melody - native FL instrument pack\n"
            "=======================================\n"
            f"{len(notes)} notes, {len(pitches)} pitches ({nm(pitches[0])}..{nm(pitches[-1])}).\n\n"
            "WHICH VST? -> DirectWave (stock FL plugin). No third-party.\n\n"
            "TO LOAD (FL Studio):\n"
            " 1. Add a DirectWave instrument (Channel Rack -> +).\n"
            " 2. Drag GRIM_melody_EXACT.sfz onto DirectWave (full DirectWave imports SFZ).\n"
            "    -> verify each zone Keytrack (KTRK) = 0 (native pitch).\n"
            " 3. Drag GRIM_melody.mid into the piano roll of that channel.\n"
            " 4. Play. = GRIM's real sound, real melody, fully editable.\n\n"
            "FILES:\n"
            " GRIM_melody_EXACT.sfz  - exact 1:1 (all real slices, round-robin)\n"
            " GRIM_melody_CLEAN.sfz  - one note per key (cleaner for editing)\n"
            " GRIM_melody.mid        - the melody (musical piano roll)\n"
            " GRIM_melody_REFERENCE.wav - how it should sound (1:1 with GRIM)\n"
            " slices_RR/ samples_keys/ - the real audio\n")

print(f"notes={len(notes)} pitches={len(pitches)} ({nm(pitches[0])}..{nm(pitches[-1])})")
print(f"RR slices={len(notes)}  key samples={len(pitches)}")
print(f"PACK -> {ROOT}")
for r,_,fs in os.walk(ROOT):
    for x in sorted(fs)[:6]: print("   ", os.path.relpath(os.path.join(r,x),ROOT))
