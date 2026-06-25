"""Assemble a clean, FL-ready MIRAGE pack in one easy-to-find folder."""
import os, shutil, glob, re, subprocess

ROOT = "/mnt/d/flbeat/MIRAGE_FL"
SRC = "/mnt/d/flbeat/data/MIRAGE"
inst = os.path.join(ROOT, "1_Instruments (SFZ) - sounds like MIRAGE")
midi = os.path.join(ROOT, "2_MIDI (editable notes)")
drum = os.path.join(ROOT, "3_Drum Samples (separated)")
for d in (inst, midi, drum):
    os.makedirs(d, exist_ok=True)

# 1) SFZ instruments: fix sample path to relative basename + copy matched MIDI + the stem wav
for pre, stem in [("MIRAGE_melody", "Melody.wav"), ("MIRAGE_808", "808.wav"), ("MIRAGE_drums", "Drums.wav")]:
    txt = open(os.path.join(SRC, "SFZ", pre + ".sfz")).read()
    txt = txt.replace("/mnt/d/flbeat/data/MIRAGE/Stems/", "")   # -> sample=Melody.wav (relative to .sfz)
    open(os.path.join(inst, pre + ".sfz"), "w").write(txt)
    shutil.copy(os.path.join(SRC, "SFZ", pre + ".mid"), os.path.join(inst, pre + ".mid"))
    shutil.copy(os.path.join(SRC, "Stems", stem), os.path.join(inst, stem))

# 2) editable MIDI: YourMT3 melody (best) + 808 + per-drum + GM kit
shutil.copy(os.path.join(SRC, "MIDI", "Melody_yourmt3.mid"), os.path.join(midi, "Melody.mid"))
for f in ["808.mid", "Kick.mid", "Snare.mid", "Hat.mid", "Ride.mid", "Crash.mid", "Toms.mid", "Drums.mid"]:
    p = os.path.join(SRC, "MIDI", f)
    if os.path.exists(p):
        shutil.copy(p, os.path.join(midi, f))

# 3) drum samples: DrumSep flac -> wav
NAME = {"kick": "Kick", "snare": "Snare", "hh": "HiHat", "toms": "Toms", "ride": "Ride", "crash": "Crash"}
for flac in glob.glob(os.path.join(SRC, "DrumSep", "*.flac")):
    m = re.search(r"\((\w+)\)", os.path.basename(flac))
    name = NAME.get(m.group(1), m.group(1)) if m else "drum"
    subprocess.run(["ffmpeg", "-y", "-i", flac, os.path.join(drum, name + ".wav")], capture_output=True)

README = """MIRAGE - FL Studio pack
=======================
FIRST: set your FL project tempo to 144 BPM.

1_Instruments (SFZ) - sounds EXACTLY like MIRAGE
  Load each .sfz in Sforzando (the free SFZ plugin included with FL Studio).
  Then drop the .mid with the SAME name onto that channel.
  These trigger the REAL audio from the song, so it sounds like the original.
     MIRAGE_melody.sfz + MIRAGE_melody.mid   (cowbell / lead)
     MIRAGE_808.sfz    + MIRAGE_808.mid      (808 bass)
     MIRAGE_drums.sfz  + MIRAGE_drums.mid    (drums)
  Keep each .sfz next to its .wav (Melody.wav / 808.wav / Drums.wav) - it needs it.

2_MIDI (editable notes) - for editing / your own sounds
  Clean note transcriptions (best models). Load onto ANY synth or sampler.
     Melody.mid  - best AI transcription (YourMT3, 2025 SOTA)
     808.mid     - bass line
     Kick / Snare / Hat / Ride / Crash / Toms .mid - one drum each (trigger note C5)
     Drums.mid   - all drums in one General-MIDI kit (load on FPC)

3_Drum Samples (separated) - the REAL isolated MIRAGE drums
  Kick / Snare / HiHat / Ride / Crash / Toms .wav - each drum pulled out of the
  song with an AI separator (MDX23C DrumSep). Use as samples or chop them.

Tip:  SFZ folder = reproduce the song.   MIDI folder = remix it with your own sounds.
"""
open(os.path.join(ROOT, "README.txt"), "w").write(README)
print("ASSEMBLED ->", ROOT)
for d in (inst, midi, drum):
    print(" ", os.path.basename(d), ":", len(os.listdir(d)), "files")
