"""Rewrite GRIMMelodySliced.dspreset with the CORRECT pitchKeyTrack="0.0" (float) on both the
group and the sample, so slices play at their native pitch (no transpose). Round-robin order =
file order = slice order. No re-slicing / no CREPE needed."""
import os, glob
SAMP = "/mnt/d/flbeat/data/SEND/GRIMPack/Samples"
OUT  = "/mnt/d/flbeat/data/SEND/GRIMPack/GRIMMelodySliced.dspreset"
slices = sorted(glob.glob(os.path.join(SAMP, "GRIMslice_*.wav")))
L = ['<?xml version="1.0" encoding="UTF-8"?>', '<DecentSampler minVersion="1.0.0">',
     '  <groups seqMode="round_robin" attack="0.0" decay="0.0" sustain="1.0" release="0.05">']
for s in slices:
    nm = os.path.basename(s)
    L.append(f'    <group pitchKeyTrack="0.0"><sample path="Samples/{nm}" '
             f'loNote="0" hiNote="127" rootNote="60" pitchKeyTrack="0.0" /></group>')
L += ['  </groups>', '</DecentSampler>']
open(OUT, "w", encoding="utf-8").write("\n".join(L))
print(f"rewrote dspreset: {len(slices)} slices, pitchKeyTrack=0.0 (native pitch, no transpose)", flush=True)
