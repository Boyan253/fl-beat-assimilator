"""Proper rebuild: re-transcribe the clean melody stem with basic-pitch (accurate
pitch+timing), extract ONE real sample per detected pitch, render monophonically
(native pitch, each note chokes the previous), and score vs the original."""
import os, glob, re, sys
import numpy as np
import soundfile as sf
import librosa

SR = 44100
STEM = "/mnt/d/flbeat/data/grim_roformer/GRIM_melody_clean.wav"
OUTDIR = "/mnt/d/flbeat/data/generated/GRIM_multisample2"
PREVIEW = "/mnt/d/flbeat/data/generated/GRIM_PREVIEW_native2.wav"
MIDOUT = "/mnt/d/flbeat/data/generated/GRIM_mel2.mid"
os.makedirs(OUTDIR, exist_ok=True)

def nm(p):
    names=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    return f"{names[p%12]}{p//12-1}"

# ---------- 1. transcribe with basic-pitch ----------
print("=== transcribing with basic-pitch ===", flush=True)
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
_, _, note_events = predict(
    STEM,
    model_or_model_path=ICASSP_2022_MODEL_PATH,
    onset_threshold=0.5,
    frame_threshold=0.3,
    minimum_note_length=58,      # ms, short staccato cowbell
    minimum_frequency=110.0,     # kill sub-bass bleed
    maximum_frequency=2200.0,
)
# note_events: (start_s, end_s, pitch_midi, amplitude, [bends])
notes = [(int(p), float(s), float(e), float(a)) for (s, e, p, a, *_ ) in note_events]
notes.sort(key=lambda x: x[1])
pitches = sorted(set(n[0] for n in notes))
print(f"notes={len(notes)} distinct={len(pitches)} pitches={[nm(p) for p in pitches]}", flush=True)
seq=[n[0] for n in notes]
print(f"STAIRCASE(monotonic)={all(seq[i]<=seq[i+1] for i in range(len(seq)-1))}", flush=True)

# ---------- 2. extract ONE real sample per distinct pitch ----------
y, _ = sf.read(STEM, always_2d=True)              # stereo source
if y.shape[1]==1: y=np.repeat(y,2,axis=1)
yL = y.astype(np.float32)
def seg_rms(st, en):
    a=int(st*SR); b=int(en*SR); a=max(0,a); b=min(len(yL),b)
    if b<=a: return 0.0
    return float(np.sqrt(np.mean(yL[a:b]**2)))

samples={}
for p in pitches:
    cands=[(n, seg_rms(n[1], n[2])) for n in notes if n[0]==p]
    # prefer loud AND reasonably long (full note, not a ghost)
    cands.sort(key=lambda c: (c[1]*min(1.0,(c[0][2]-c[0][1])/0.25)), reverse=True)
    best=cands[0][0]
    st=best[1]; dur=min(best[2]-best[1], 1.2)
    a=int(st*SR); b=min(int((st+dur)*SR), len(yL))
    samp=yL[a:b].copy()
    # tiny fades to kill clicks
    fi=min(64,len(samp)//8); fo=min(2048,len(samp)//4)
    if fi>0: samp[:fi]*=np.linspace(0,1,fi)[:,None]
    if fo>0: samp[-fo:]*=np.linspace(1,0,fo)[:,None]
    samples[p]=samp
    sf.write(os.path.join(OUTDIR, f"GRIM_mel_{nm(p).replace('#','s')}.wav"), samp, SR)
print(f"extracted {len(samples)} real per-pitch samples -> {OUTDIR}", flush=True)

# ---------- 3. render MONOPHONIC (choke at next onset), native pitch ----------
tail=0.20
total=int((notes[-1][2]+tail+0.5)*SR)
buf=np.zeros((total,2),dtype=np.float32)
for i,(p,st,en,amp) in enumerate(notes):
    s=samples[p]
    nxt=notes[i+1][1] if i+1<len(notes) else st+1.2
    play=min(len(s), int((nxt-st+0.09)*SR))     # choke at next note + 90ms release overlap
    play=max(play, int(0.04*SR))
    seg=s[:play].copy()
    rel=min(int(0.07*SR), len(seg)//3)
    if rel>0: seg[-rel:]*=np.linspace(1,0,rel)[:,None]
    g=float(np.clip(amp,0.05,1.0))
    pos=int(st*SR); b=min(pos+len(seg),total)
    buf[pos:b]+=seg[:b-pos]*g
peak=np.max(np.abs(buf)) or 1.0
buf=buf/peak*0.89
sf.write(PREVIEW, buf, SR)
print(f"WROTE preview {PREVIEW} ({total/SR:.1f}s)", flush=True)

# write MIDI (detected pitches)
import pretty_midi
pm=pretty_midi.PrettyMIDI(); inst=pretty_midi.Instrument(program=0)
for (p,st,en,amp) in notes:
    v=int(np.clip(np.interp(amp,[0.05,1.0],[45,127]),1,127))
    inst.notes.append(pretty_midi.Note(velocity=v,pitch=p,start=st,end=max(en,st+0.05)))
pm.instruments.append(inst); pm.write(MIDOUT)
print(f"WROTE midi {MIDOUT}", flush=True)

# ---------- 4. score vs original ----------
def corr(x,y):
    x=x.flatten()-x.mean(); y=y.flatten()-y.mean()
    d=(np.linalg.norm(x)*np.linalg.norm(y)) or 1.0
    return float(np.dot(x,y)/d)
a=librosa.load(STEM,sr=SR,mono=True)[0]
b=librosa.load(PREVIEW,sr=SR,mono=True)[0]
n=min(len(a),len(b)); a=a[:n]; b=b[:n]
ma=librosa.power_to_db(librosa.feature.melspectrogram(y=a,sr=SR,n_mels=64))
mb=librosa.power_to_db(librosa.feature.melspectrogram(y=b,sr=SR,n_mels=64))
m=min(ma.shape[1],mb.shape[1]); mel_c=corr(ma[:,:m],mb[:,:m])
ca=librosa.feature.chroma_cqt(y=a,sr=SR); cb=librosa.feature.chroma_cqt(y=b,sr=SR)
m=min(ca.shape[1],cb.shape[1]); chroma_c=corr(ca[:,:m],cb[:,:m])
oa=librosa.onset.onset_strength(y=a,sr=SR); ob=librosa.onset.onset_strength(y=b,sr=SR)
m=min(len(oa),len(ob)); onset_c=corr(oa[:m],ob[:m])
print("=== SCORE vs original ===")
print(f"timbre(mel)={mel_c*100:.1f}%  pitch(chroma)={chroma_c*100:.1f}%  timing(onset)={onset_c*100:.1f}%  OVERALL={(mel_c+chroma_c+onset_c)/3*100:.1f}%")
