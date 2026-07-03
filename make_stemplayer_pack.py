"""make_stemplayer_pack.py — TRUE 1:1 Sforzando pack: each SFZ plays its WHOLE stem exactly
as in the song (no per-occurrence chunking). Load all SFZs + their one-note MIDIs at bar 1 ->
the song reconstructed 1:1 per instrument, fully faithful (bass included).

  python make_stemplayer_pack.py <song.wav> <NAME> <bpm>
Output -> /mnt/d/flbeat/data/FL_PACKS/<NAME>_1to1/
"""
import os, sys, shutil, subprocess, warnings
warnings.filterwarnings("ignore")
import numpy as np, soundfile as sf, pretty_midi

SONG = sys.argv[1]; NAME = sys.argv[2]; BPM = float(sys.argv[3]) if len(sys.argv) > 3 else 120.0
ROOT = f"/mnt/d/flbeat/data/FL_PACKS/{NAME}_1to1"
WORK = f"{ROOT}/_work"
for d in (ROOT, WORK):
    os.makedirs(d, exist_ok=True)
PY = sys.executable

print("separating stems (htdemucs_ft)...", flush=True)
subprocess.run([PY, "-m", "demucs", "-n", "htdemucs_ft", "-d", "cuda", "-o", WORK, SONG],
               check=True, capture_output=True)
stemdir = os.path.join(WORK, "htdemucs_ft", os.path.splitext(os.path.basename(SONG))[0])

NAMEMAP = [("other", "Melody"), ("bass", "Bass"), ("drums", "Drums"), ("vocals", "Vocals")]
made = []
for k, nice in NAMEMAP:
    src = os.path.join(stemdir, k + ".wav")
    if not os.path.exists(src):
        continue
    y, sr = sf.read(src)
    if np.max(np.abs(y)) < 0.005:          # skip silent stems (e.g. vocals on an instrumental)
        continue
    dur = len(y) / sr
    shutil.copy(src, os.path.join(ROOT, nice + ".wav"))
    # whole-file one-shot player — plays the entire stem, no pitch shift on note C5(60)
    sfz = (f"// 1:1 stem player — plays the entire {nice} stem exactly as in the song\n"
           f"<region> sample={nice}.wav pitch_keycenter=60 lokey=60 hikey=60 "
           f"loop_mode=one_shot ampeg_attack=0 ampeg_release=0.05\n")
    open(os.path.join(ROOT, nice + ".sfz"), "w").write(sfz)
    pm = pretty_midi.PrettyMIDI(initial_tempo=BPM)
    ins = pretty_midi.Instrument(program=0)
    ins.notes.append(pretty_midi.Note(110, 60, 0.0, dur))   # one note, holds the whole song
    pm.instruments.append(ins)
    pm.write(os.path.join(ROOT, nice + ".mid"))
    made.append(f"{nice} ({dur:.0f}s)")
    print("stem-player:", nice, f"{dur:.0f}s", flush=True)

open(os.path.join(ROOT, "README.txt"), "w").write(
    f"{NAME} - TRUE 1:1 stem players (set FL tempo to {int(BPM)} BPM)\n\n"
    "Each .sfz plays its WHOLE stem exactly as in the song (bass included - no chunking).\n"
    "For each instrument:\n"
    "  1. Load the .sfz in Sforzando.\n"
    "  2. Drop the same-name .mid on that channel (it's one note at bar 1 holding the song).\n"
    "  3. Make sure the clip starts at bar 1 so all stems line up.\n"
    "Load all of them together -> the full song, 1:1, but now per-instrument (mute/solo/mix).\n"
    "Keep each .sfz next to its .wav.\n\n"
    "Tip: you can also just drag the .wav files straight onto audio tracks - same 1:1 result.\n")
shutil.rmtree(WORK, ignore_errors=True)
print(f"DONE ({', '.join(made)}) -> {ROOT}", flush=True)
