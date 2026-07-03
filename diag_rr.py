import re, numpy as np, librosa, pretty_midi
SR=44100
MID="/mnt/d/flbeat/data/SEND/GRIM_NATIVE/GRIM_melody.mid"
SFZ="/mnt/d/flbeat/data/SEND/GRIM_NATIVE/GRIM_melody_EXACT.sfz"
REF="/mnt/d/flbeat/data/SEND/GRIM_NATIVE/GRIM_melody_REFERENCE.wav"

# MIDI notes per pitch + velocity stats
pm=pretty_midi.PrettyMIDI(MID); notes=[n for inst in pm.instruments for n in inst.notes]
from collections import Counter
midc=Counter(n.pitch for n in notes)
vels=[n.velocity for n in notes]
print(f"MIDI: {len(notes)} notes, vel min/med/max = {min(vels)}/{int(np.median(vels))}/{max(vels)}")

# SFZ seq_length per group key + region count per key
txt=open(SFZ).read()
groups=re.findall(r'key=(\d+)\s+seq_length=(\d+)',txt)
regc=Counter()
curkey=None
for line in txt.splitlines():
    g=re.search(r'<group>\s+key=(\d+)',line)
    if g: curkey=int(g.group(1))
    if line.strip().startswith('<region>') and curkey is not None:
        regc[curkey]+=1
print("pitch | MIDI notes | SFZ regions | seq_length")
mism=0
for k,sl in sorted((int(k),int(s)) for k,s in groups):
    mn=midc.get(k,0); rc=regc.get(k,0)
    flag="" if (mn==rc==sl) else "  <-- MISMATCH"
    if flag: mism+=1
    print(f"  {k:3d} | {mn:3d} | {rc:3d} | {sl:3d}{flag}")
print(f"mismatched pitches: {mism}")

# silent fraction comparison
def silent_frac(p):
    y=librosa.load(p,sr=SR,mono=True)[0]; fr=4410; n=len(y)//fr
    r=np.array([np.sqrt(np.mean(y[i*fr:(i+1)*fr]**2)) for i in range(n)])
    return np.mean(r<10**(-50/20))*100
print(f"REFERENCE silent fraction: {silent_frac(REF):.1f}%  (vs sfizz render 53.7%)")
