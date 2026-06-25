"""transcribe_drums.py — turn a drum stem into MIDI patterns (kick/snare/hat).

Outputs into <out_dir>:
  Drums.mid  - one GM drum track (36=kick, 38=snare, 42=hat) for an FPC/drum kit
  Kick.mid  Snare.mid  Hat.mid  - separate single-note patterns (one per sampler instrument)

  python transcribe_drums.py <drums.wav> <out_dir>
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, librosa, pretty_midi, scipy.signal as ss

DRUMS = sys.argv[1]
OUT = sys.argv[2]
os.makedirs(OUT, exist_ok=True)
SR = 44100

y, _ = librosa.load(DRUMS, sr=SR, mono=True)


def band_onsets(lo=None, hi=None):
    if lo and hi:
        sos = ss.butter(4, [lo, hi], btype="band", fs=SR, output="sos")
    elif lo:
        sos = ss.butter(4, lo, btype="high", fs=SR, output="sos")
    else:
        sos = ss.butter(4, hi, btype="low", fs=SR, output="sos")
    yb = ss.sosfilt(sos, y).astype("float32")
    return list(librosa.onset.onset_detect(y=yb, sr=SR, units="time", backtrack=True))


# per-band onset detection (reliable kick/snare/hat separation)
kick  = band_onsets(hi=150)            # sub/kick thump
snare = band_onsets(lo=250, hi=2500)   # snare body/crack
hat   = band_onsets(lo=6000)           # hi-hats/cymbals
# de-dup snare hits that coincide with a kick (kick body leaks into mids)
kick_set = sorted(kick)
snare = [t for t in snare if not any(abs(t - k) < 0.04 for k in kick_set)]

# combined GM drum MIDI
pm = pretty_midi.PrettyMIDI(); drum = pretty_midi.Instrument(program=0, is_drum=True, name="drums")
for t in kick:  drum.notes.append(pretty_midi.Note(100, 36, t, t + 0.12))
for t in snare: drum.notes.append(pretty_midi.Note(100, 38, t, t + 0.12))
for t in hat:   drum.notes.append(pretty_midi.Note(95, 42, t, t + 0.10))
pm.instruments.append(drum); pm.write(os.path.join(OUT, "Drums.mid"))

# separate single-note patterns (trigger note C5=60) for the individual samplers
def save_one(times, path):
    p = pretty_midi.PrettyMIDI(); ins = pretty_midi.Instrument(program=0)
    for t in times:
        ins.notes.append(pretty_midi.Note(110, 60, t, t + 0.12))
    p.instruments.append(ins); p.write(path)

save_one(kick,  os.path.join(OUT, "Kick.mid"))
save_one(snare, os.path.join(OUT, "Snare.mid"))
save_one(hat,   os.path.join(OUT, "Hat.mid"))
print(f"drums: kick={len(kick)} snare={len(snare)} hat={len(hat)} -> {OUT}", flush=True)
