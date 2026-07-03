"""Diagnose the misalignment and re-transcribe GRIM's melody DIRECTLY from the clean stem,
so the MIDI starts where the melody actually starts (0s) and matches the audio we sample from."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, soundfile as sf, pretty_midi
from collections import Counter

STEM = "/mnt/d/flbeat/data/generated/GRIM_melody_6s_CLEAN.wav"
OUTMID = "/mnt/d/flbeat/data/generated/GRIM_melody_aligned.mid"

# --- diagnose the OLD midi ---
old = pretty_midi.PrettyMIDI("/mnt/d/flbeat/data/FL_PACKS/GRIM/GRIM_FULL_import.mid")
omel = sorted([n for ins in old.instruments
               if not ins.is_drum and "808" not in (ins.name or "").lower() and "bass" not in (ins.name or "").lower()
               for n in ins.notes], key=lambda n: n.start)
print("OLD melody MIDI: first onset %.2fs, last %.2fs, %d notes" % (omel[0].start, omel[-1].start, len(omel)))

# --- where does the stem actually have melody? ---
y, sr = sf.read(STEM)
if y.ndim > 1: y = y.mean(1)
y = y.astype(np.float32)
print("stem duration %.1fs" % (len(y) / sr))
win = sr // 2
head = [float(np.sqrt(np.mean(y[i:i + win] ** 2))) for i in range(0, min(len(y), sr * 20), win)]
print("stem RMS per 0.5s (first 20s):", [round(r, 3) for r in head])

# --- re-transcribe the stem itself ---
from basic_pitch.inference import predict
print("transcribing stem with basic_pitch ...", flush=True)
_, midi, _ = predict(STEM, minimum_note_length=70, onset_threshold=0.5, frame_threshold=0.3)
midi.write(OUTMID)
nmel = sorted([n for ins in midi.instruments for n in ins.notes], key=lambda n: n.start)
print("NEW transcription: first onset %.2fs, %d notes -> %s" % (nmel[0].start if nmel else -1, len(nmel), OUTMID))
print("NEW pitch histogram:", sorted(Counter(pretty_midi.note_number_to_name(n.pitch) for n in nmel).items(),
                                     key=lambda x: -x[1])[:16])
