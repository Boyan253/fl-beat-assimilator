"""Assemble GRIM_READY — the minimal turnkey kit: melody as an audio loop (exact sound, no tells),
drums+808 as one importable multi-track MIDI (human-like notes), + the one-shot samples."""
import os, glob, shutil, warnings
warnings.filterwarnings("ignore")
import pretty_midi

PMID = "/mnt/d/flbeat/data/FL_PACKS/GRIM/2_MIDI (editable)"
PONE = "/mnt/d/flbeat/data/FL_PACKS/GRIM/4_One-shots"
OUT = "/mnt/d/flbeat/data/GRIM_READY"
os.makedirs(OUT, exist_ok=True)
BPM = 122.0

# 1) melody = clean audio loop (drag into playlist; exact GRIM sound, no artifacts)
shutil.copy("/mnt/d/flbeat/data/generated/GRIM_melody_LOOP_8bar.wav", os.path.join(OUT, "MELODY_loop.wav"))

# 2) drums + 808 as ONE multi-track MIDI (human-like note patterns)
out = pretty_midi.PrettyMIDI(initial_tempo=BPM)
for name, f, prog in [("Bass(808)", "808.mid", 38), ("Kick", "Kick.mid", 0),
                      ("Snare", "Snare.mid", 0), ("Hat", "Hat.mid", 0)]:
    p = os.path.join(PMID, f)
    if not os.path.exists(p):
        continue
    src = pretty_midi.PrettyMIDI(p)
    ins = pretty_midi.Instrument(program=prog, name=name)
    for i in src.instruments:
        ins.notes.extend(i.notes)
    out.instruments.append(ins)
out.write(os.path.join(OUT, "DRUMS_808.mid"))

# 3) one-shot samples (keep original names so root notes are visible)
copied = []
for pat, dst in [("808_*.wav", None), ("Kick.wav", "Kick.wav"),
                 ("Snare.wav", "Snare.wav"), ("HiHat.wav", "HiHat.wav")]:
    for src in glob.glob(os.path.join(PONE, pat)):
        d = dst or os.path.basename(src)
        shutil.copy(src, os.path.join(OUT, d)); copied.append(d)

README = """GRIM_READY  —  set up ONCE in FL, then Save As GRIM.flp = turnkey forever.
Set project tempo to 122 BPM.

STEP 1 — MELODY (audio, exact GRIM sound, zero setup):
   Drag MELODY_loop.wav into the PLAYLIST as an audio clip (loop it across the track).
   It's already 122 BPM and bar-aligned, so it sits in time. Done — no instrument, no tells.

STEP 2 — DRUMS + 808 (human-like MIDI notes):
   File > Import > MIDI file > DRUMS_808.mid
   -> FL creates 4 channels (Bass(808), Kick, Snare, Hat) with the note patterns already in place.
   Load the matching sample onto each channel:
     Bass(808) <- 808_*.wav   (set the Sampler ROOT to the note in the filename; turn ON glide)
     Kick      <- Kick.wav
     Snare     <- Snare.wav
     Hat       <- HiHat.wav

STEP 3 — File > Save As > GRIM.flp.
   From now on you just OPEN GRIM.flp and everything loads + plays. To share it:
   File > Export > Zipped loose files (bundles the .flp WITH all samples in one zip).

(Optional glue: a light Fruity Soft Clipper / saturator on the Master.)
"""
open(os.path.join(OUT, "HOW TO LOAD.txt"), "w").write(README)
print("GRIM_READY:", "MELODY_loop.wav, DRUMS_808.mid (", len(out.instruments), "tracks ),",
      ", ".join(sorted(set(copied))), flush=True)
