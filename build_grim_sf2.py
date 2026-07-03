"""GRIM melody -> SF2 SoundFont (multisample), loadable by FL's stock Fruity Soundfont
Player. One real sample per pitch; at each real pitch playback is native (key==root ->
0 semitone shift = 1:1 timbre). Reuses the proven SF2 RIFF structure (abraham build)."""
import os, glob, re, struct, warnings
warnings.filterwarnings("ignore")
import numpy as np, soundfile as sf

KEYS = "/mnt/d/flbeat/data/SEND/GRIM_NATIVE/samples_keys"
OUT  = "/mnt/d/flbeat/data/SEND/GRIM_NATIVE/GRIM_melody.sf2"
NAMES = {'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}
def to_midi(tok):
    m = re.match(r'^([A-G])(s?)(-?\d+)$', tok)
    base = NAMES[m.group(1)] + (1 if m.group(2)=='s' else 0)
    return base + (int(m.group(3))+1)*12

pairs = []
for w in glob.glob(os.path.join(KEYS, "GRIM_*.wav")):
    tok = os.path.splitext(os.path.basename(w))[0].replace("GRIM_","")
    pairs.append((to_midi(tok), w))
pairs.sort()
wavs    = [w for _,w in pairs]
pitches = [p for p,_ in pairs]
N = len(wavs)
print(f"{N} samples, pitches {pitches}")

def pad(b): return b + (b"\x00" if len(b)%2 else b"")
def sub(cid,data): return cid + struct.pack("<I",len(data)) + pad(data)
def name20(s): return s.encode("ascii","ignore")[:20].ljust(20,b"\x00")

# ---- smpl: int16 mono, 46 zero guard frames after each ----
smpl = bytearray(); hdrs = []
for w,p in zip(wavs,pitches):
    y,sr = sf.read(w, dtype="int16", always_2d=False)
    if y.ndim>1: y=y[:,0]
    start=len(smpl)//2; smpl+=y.tobytes(); end=len(smpl)//2; smpl+=b"\x00\x00"*46
    hdrs.append(("GRIM_%d"%p,start,end,sr,p))

INFO=sub(b"LIST",b"INFO"+sub(b"ifil",struct.pack("<HH",2,1))+sub(b"isng",pad(b"EMU8000\x00"))+sub(b"INAM",pad(b"GRIM Melody\x00")))
sdta=sub(b"LIST",b"sdta"+sub(b"smpl",bytes(smpl)))

phdr=name20("GRIM Melody")+struct.pack("<HHHIII",0,0,0,0,0,0)+name20("EOP")+struct.pack("<HHHIII",0,0,1,0,0,0)
pbag=struct.pack("<HH",0,0)+struct.pack("<HH",1,0)
pmod=struct.pack("<HHHHHHHHHH",*([0]*10))
pgen=struct.pack("<HH",41,0)+struct.pack("<HH",0,0)
inst=name20("GRIM")+struct.pack("<H",0)+name20("EOI")+struct.pack("<H",N)
ibag=b"".join(struct.pack("<HH",i*3,0) for i in range(N))+struct.pack("<HH",N*3,0)
imod=struct.pack("<HHHHHHHHHH",*([0]*10))
igen=b""
for i,p in enumerate(pitches):
    lo=0   if i==0   else (pitches[i-1]+p)//2+1
    hi=127 if i==N-1 else (p+pitches[i+1])//2
    igen+=struct.pack("<HH",43,(lo&0xFF)|((hi&0xFF)<<8))   # keyRange
    igen+=struct.pack("<HH",58,p&0xFF)                     # overridingRootKey
    igen+=struct.pack("<HH",53,i)                          # sampleID (last)
igen+=struct.pack("<HH",0,0)
shdr=b""
for (nm,start,end,sr,root) in hdrs:
    shdr+=name20(nm)+struct.pack("<IIII",start,end,start,end)+struct.pack("<I",sr)+struct.pack("<BbHH",root,0,0,1)
shdr+=name20("EOS")+struct.pack("<IIII",0,0,0,0)+struct.pack("<I",0)+struct.pack("<BbHH",0,0,0,0)

pdta=(sub(b"phdr",phdr)+sub(b"pbag",pbag)+sub(b"pmod",pmod)+sub(b"pgen",pgen)+
      sub(b"inst",inst)+sub(b"ibag",ibag)+sub(b"imod",imod)+sub(b"igen",igen)+sub(b"shdr",shdr))
PDTA=sub(b"LIST",b"pdta"+pdta)
riff=b"RIFF"+struct.pack("<I",len(b"sfbk"+INFO+sdta+PDTA))+b"sfbk"+INFO+sdta+PDTA
with open(OUT,"wb") as f: f.write(riff)
print("WROTE %s | %d samples, %.0f KB"%(OUT,N,len(riff)/1024))
