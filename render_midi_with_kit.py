"""render_midi_with_kit.py — render a multi-track MIDI through a REAL sample kit (GRIM's sounds)
instead of a cheap GM soundfont, so the preview sounds like GRIM *before* any DAW.
  Drums -> GRIM kick/snare/hat | Bass(808) -> GRIM 808 pitched | Melody -> GRIM 16-note multisample.

  python render_midi_with_kit.py <in.mid> <out.wav>
"""
import sys, os, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np, librosa, soundfile as sf, pretty_midi

MIDI = sys.argv[1]; OUT = sys.argv[2]; SR = 44100
G = "/mnt/d/flbeat/data/FL_PACKS/GRIM/4_One-shots"
MS = "/mnt/d/flbeat/data/generated/GRIM_multisample"

def load(p):
    y, _ = librosa.load(p, sr=SR, mono=True); return y.astype("float32")

kick, snare, hat = load(f"{G}/Kick.wav"), load(f"{G}/Snare.wav"), load(f"{G}/HiHat.wav")
o808 = glob.glob(f"{G}/808_*.wav")[0]
b808 = load(o808)
root808 = pretty_midi.note_name_to_number(os.path.basename(o808).split("_")[1].split(".")[0])
mel_bank = {}
for f in glob.glob(f"{MS}/GRIM_mel_*.wav"):
    nm = os.path.basename(f).replace("GRIM_mel_", "").replace(".wav", "")
    try:
        mel_bank[pretty_midi.note_name_to_number(nm)] = load(f)
    except Exception:
        pass
mel_pitches = sorted(mel_bank)

pm = pretty_midi.PrettyMIDI(MIDI)
buf = np.zeros(int((pm.get_end_time() + 1) * SR) + SR, dtype="float32")
def place(samp, t, gain):
    i = int(t * SR); n = min(len(samp), len(buf) - i)
    if n > 0: buf[i:i + n] += samp[:n] * gain

pcache = {}
def pitched(samp, key, semis):
    k = (key, semis)
    if k not in pcache:
        pcache[k] = librosa.effects.pitch_shift(samp, sr=SR, n_steps=semis) if semis else samp
    return pcache[k]

for inst in pm.instruments:
    nm = (inst.name or "").lower()
    for n in inst.notes:
        if inst.is_drum or "drum" in nm:
            s = {36: kick, 38: snare, 42: hat}.get(n.pitch)
            if s is not None: place(s, n.start, 0.9)
        elif "808" in nm or "bass" in nm:
            seg = pitched(b808, "808", n.pitch - root808)
            d = int((n.end - n.start) * SR)
            seg = seg[:d] if len(seg) >= d else np.pad(seg, (0, max(0, d - len(seg))))
            place(seg, n.start, 0.95)
        else:
            near = min(mel_pitches, key=lambda p: abs(p - n.pitch))
            seg = pitched(mel_bank[near], near, n.pitch - near)
            place(seg, n.start, 0.7 * (n.velocity / 110.0))
pk = np.max(np.abs(buf)); buf = buf / pk * 0.97 if pk else buf
sf.write(OUT, buf, SR)
print("rendered with GRIM kit ->", OUT, flush=True)
