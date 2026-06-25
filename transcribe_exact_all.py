"""transcribe_exact_all.py — EXACT (one-to-one) MIDI transcription from MIRAGE stems.

Melodic stems -> basic_pitch neural transcription (true pitch + true timing, NO quantize).
Drum stem    -> per-band onset detection (true hit timing).
All files tempo-tagged so they play at the song's BPM in FL, but notes keep exact timing.

  python transcribe_exact_all.py <stems_dir> <out_dir> [bpm=144]
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, librosa, pretty_midi, scipy.signal as ss
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH

STEMS = sys.argv[1]
OUT   = sys.argv[2]
BPM   = float(sys.argv[3]) if len(sys.argv) > 3 else 144.0
os.makedirs(OUT, exist_ok=True)
SR = 44100


def retag(midi, path):
    """Rewrite with the song tempo, keeping exact note times (no quantize)."""
    out = pretty_midi.PrettyMIDI(initial_tempo=BPM)
    for inst in midi.instruments:
        out.instruments.append(inst)
    out.write(path)
    return sum(len(i.notes) for i in midi.instruments)


def melodic(stem, out_name, minf, maxf, onset=0.4):
    path = os.path.join(STEMS, stem)
    if not os.path.exists(path):
        print(f"  skip {stem} (missing)", flush=True); return
    _, midi, _ = predict(
        path, model_or_model_path=ICASSP_2022_MODEL_PATH,
        onset_threshold=onset, frame_threshold=0.25, minimum_note_length=58,
        minimum_frequency=minf, maximum_frequency=maxf,
        multiple_pitch_bends=False, melodia_trick=True)
    n = retag(midi, os.path.join(OUT, out_name))
    print(f"  {out_name}: {n} notes (exact pitch+timing)", flush=True)


# ---- melodic stems: real neural transcription ----
print("transcribing melodic stems (basic_pitch)...", flush=True)
melodic("Melody.wav", "Melody.mid", 110, 3000, onset=0.4)   # cowbell / lead
melodic("808.wav",    "808.mid",     28,  350, onset=0.35)   # 808 bass

# ---- drum stem: exact onset detection per band ----
print("transcribing drums (band onsets)...", flush=True)
dpath = os.path.join(STEMS, "Drums.wav")
y, _ = librosa.load(dpath, sr=SR, mono=True)


def band_onsets(lo=None, hi=None):
    if lo and hi:   sos = ss.butter(4, [lo, hi], btype="band", fs=SR, output="sos")
    elif lo:        sos = ss.butter(4, lo, btype="high", fs=SR, output="sos")
    else:           sos = ss.butter(4, hi, btype="low",  fs=SR, output="sos")
    yb = ss.sosfilt(sos, y).astype("float32")
    return list(librosa.onset.onset_detect(y=yb, sr=SR, units="time", backtrack=True))


kick  = band_onsets(hi=150)
snare = band_onsets(lo=250, hi=2500)
hat   = band_onsets(lo=6000)
ks = sorted(kick)
snare = [t for t in snare if not any(abs(t - k) < 0.04 for k in ks)]

# combined GM kit
pm = pretty_midi.PrettyMIDI(initial_tempo=BPM)
drum = pretty_midi.Instrument(program=0, is_drum=True, name="drums")
for t in kick:  drum.notes.append(pretty_midi.Note(100, 36, t, t + 0.12))
for t in snare: drum.notes.append(pretty_midi.Note(100, 38, t, t + 0.12))
for t in hat:   drum.notes.append(pretty_midi.Note(95, 42, t, t + 0.10))
pm.instruments.append(drum); pm.write(os.path.join(OUT, "Drums.mid"))

def save_one(times, path):
    p = pretty_midi.PrettyMIDI(initial_tempo=BPM); ins = pretty_midi.Instrument(program=0)
    for t in times: ins.notes.append(pretty_midi.Note(110, 60, t, t + 0.12))
    p.instruments.append(ins); p.write(path)

save_one(kick,  os.path.join(OUT, "Kick.mid"))
save_one(snare, os.path.join(OUT, "Snare.mid"))
save_one(hat,   os.path.join(OUT, "Hat.mid"))
print(f"  drums: kick={len(kick)} snare={len(snare)} hat={len(hat)}", flush=True)
print(f"DONE -> {OUT}  (set FL tempo to {int(BPM)} BPM; notes are EXACT/un-quantized)", flush=True)
