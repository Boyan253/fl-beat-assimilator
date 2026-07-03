"""Build a GRIM loop pack (zip-ready) to send the friend: 8-bar, 122 BPM, loop-ready WAVs of the
full beat + the melody (real audio = sounds amazing), plus the MIDI for editing. Same start/length
so everything is in sync and loops cleanly."""
import os, shutil, warnings
warnings.filterwarnings("ignore")
import numpy as np, soundfile as sf, librosa

BPM, BARS, SR = 122.0, 8, 44100
LOOP_SEC = BARS * 4 * 60.0 / BPM                      # 8 bars @122 = 15.74s, seamless
OUT = "/mnt/d/flbeat/data/SEND/GRIM_LoopPack"
LOOPS, MIDI = os.path.join(OUT, "loops"), os.path.join(OUT, "midi")
FULL = "/mnt/d/flbeat/data/generated/GRIM_3_GRIT.wav"
MEL  = "/mnt/d/flbeat/data/generated/GRIM_melody_6s_CLEAN.wav"

if os.path.isdir(OUT):
    shutil.rmtree(OUT)
os.makedirs(LOOPS); os.makedirs(MIDI)

# find the loudest 8-bar window aligned to a beat (= the main loop / hook)
mono, _ = librosa.load(FULL, sr=SR, mono=True)
_, beats = librosa.beat.beat_track(y=mono, sr=SR, units="time")
n = int(LOOP_SEC * SR)


def rms_at(t):
    i = int(t * SR); seg = mono[i:i + n]
    return float(np.sqrt(np.mean(seg ** 2))) if len(seg) >= n * 0.9 else -1.0


cands = [(rms_at(b), b) for b in beats]
cands = [(r, b) for r, b in cands if r > 0]
start = max(cands)[1] if cands else 13.3


def cut(path, out):
    y, _ = librosa.load(path, sr=SR, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])
    a = int(start * SR); b = a + n
    seg = y[:, a:b].copy()
    f = int(0.003 * SR)
    seg[:, :f] *= np.linspace(0, 1, f); seg[:, -f:] *= np.linspace(1, 0, f)
    sf.write(out, seg.T.astype(np.float32), SR)


cut(FULL, os.path.join(LOOPS, "GRIM_FullBeat_122bpm.wav"))
cut(MEL,  os.path.join(LOOPS, "GRIM_Melody_122bpm.wav"))

G2 = "/mnt/d/flbeat/data/FL_PACKS/GRIM/2_MIDI (editable)"
for src, dst in [("Piano.mid", "GRIM_Melody.mid"), ("808.mid", "GRIM_808.mid"), ("Drums.mid", "GRIM_Drums.mid")]:
    p = os.path.join(G2, src)
    if os.path.exists(p):
        shutil.copy(p, os.path.join(MIDI, dst))

open(os.path.join(OUT, "README.txt"), "w", encoding="utf-8").write(
    "GRIM LOOP PACK  -  122 BPM  -  8-bar loops\n"
    "==========================================\n"
    "loops/  GRIM_FullBeat = the whole beat (drop in, set to loop, rework/rap over it).\n"
    "        GRIM_Melody   = just the melody (real audio, sounds 1-to-1).\n"
    "midi/   editable notes (melody/808/drums) - for rebuilding/adding parts.\n"
    "All loops are the SAME 8 bars, in sync, loop-ready.\n")

print(f"loop pack: start {start:.2f}s, {LOOP_SEC:.2f}s loops -> {OUT}", flush=True)
