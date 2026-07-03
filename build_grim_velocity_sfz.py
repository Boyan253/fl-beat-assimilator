"""Robust EXACT reconstruction: instead of stateful round-robin (which sfizz drops),
key each real slice to a UNIQUE VELOCITY per pitch. The MIDI carries that velocity ->
deterministic exact-slice selection, no dropped notes, works in any SFZ engine.
amp_veltrack=0 so velocity only SELECTS (doesn't change volume). Look stays musical."""
import os, re, glob, numpy as np, pretty_midi
ROOT="/mnt/d/flbeat/data/SEND/GRIM_NATIVE"
RR=os.path.join(ROOT,"slices_RR")
MID_IN=os.path.join(ROOT,"GRIM_melody.mid")
SFZ_OUT=os.path.join(ROOT,"GRIM_melody_VELKEY.sfz")
MID_OUT=os.path.join(ROOT,"GRIM_melody_VEL.mid")
NAMES={'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}
def to_midi(tok):
    m=re.match(r'^([A-G])(s?)(-?\d+)$',tok); base=NAMES[m.group(1)]+(1 if m.group(2)=='s' else 0)
    return base+(int(m.group(3))+1)*12

# map (pitch, seq) -> relative sample path
bypitchseq={}
for w in glob.glob(os.path.join(RR,"rr_*.wav")):
    b=os.path.basename(w)                       # rr_0000_E4_001.wav
    m=re.match(r'rr_(\d+)_([A-Gs0-9#]+)_(\d+)\.wav$',b)
    if not m: continue
    p=to_midi(m.group(2)); seq=int(m.group(3))
    bypitchseq[(p,seq)]=f"slices_RR/{b}"
print(f"indexed {len(bypitchseq)} slices")

# walk MIDI in time order; assign per-pitch occurrence -> velocity -> picks the slice
pm=pretty_midi.PrettyMIDI(MID_IN)
notes=sorted((n for inst in pm.instruments for n in inst.notes),key=lambda n:n.start)
occ={}
out=pretty_midi.PrettyMIDI(); inst=pretty_midi.Instrument(program=0)
regions=[]; missing=0
for n in notes:
    occ[n.pitch]=occ.get(n.pitch,0)+1; i=occ[n.pitch]
    if i>127:   # velocity cap; pitches max 66 so never hit, but guard
        i=127
    fn=bypitchseq.get((n.pitch,i))
    if fn is None: missing+=1; continue
    inst.notes.append(pretty_midi.Note(velocity=i,pitch=n.pitch,start=n.start,end=max(n.end,n.start+0.05)))
    regions.append((n.pitch,i,fn))
out.instruments.append(inst); out.write(MID_OUT)
print(f"notes={len(inst.notes)} missing={missing}")

# write velocity-keyed SFZ (one region per slice: key=pitch, lovel=hivel=occurrence)
with open(SFZ_OUT,"w") as f:
    f.write("// GRIM melody - EXACT 1:1 via velocity-keyed slices (deterministic, no round-robin).\n")
    f.write("// Use with GRIM_melody_VEL.mid. Load in sforzando/sfizz.\n")
    f.write("<global> loop_mode=one_shot ampeg_release=0.06 pitch_keytrack=0 amp_veltrack=0\n")
    for (p,i,fn) in regions:
        f.write(f"<region> sample={fn} key={p} lovel={i} hivel={i}\n")
print(f"WROTE {SFZ_OUT} ({len(regions)} regions) and {MID_OUT}")
