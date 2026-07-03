"""render_melody_with_oneshot.py — play a transcribed melody MIDI using ONE clean note sampled
from the song's own melody stem -> the song's real timbre, but clean (no bass/drums/fade) + editable.

  python render_melody_with_oneshot.py <melody_stem.wav> <melody.mid> <out.wav>
Picks the clearest isolated note (max mid-band / bass-band ratio), then pitch-shifts it per MIDI note.
Also saves <out>_oneshot.wav (the chosen one-shot) for loading into an FL sampler.
"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, librosa, soundfile as sf, scipy.signal as ss, pretty_midi

STEM = sys.argv[1]; MID = sys.argv[2]; OUT = sys.argv[3]
SR = 44100
y, _ = librosa.load(STEM, sr=SR, mono=True)

# pick the CLEAREST one-shot: loud, isolated, and high mid/bass energy ratio (least bleed)
sos_mid = ss.butter(4, [200, 4000], btype="band", fs=SR, output="sos")
sos_low = ss.butter(4, 200, btype="low", fs=SR, output="sos")
on = librosa.onset.onset_detect(y=y, sr=SR, units="time", backtrack=True)
best, bs = None, -1
for i, t in enumerate(on):
    nxt = on[i + 1] if i + 1 < len(on) else len(y) / SR
    gap = nxt - t
    if gap < 0.2:
        continue
    s0 = int(t * SR); w = y[s0:s0 + int(0.25 * SR)]
    if len(w) < 1000:
        continue
    mid = np.sqrt(np.mean(ss.sosfilt(sos_mid, w) ** 2))
    low = np.sqrt(np.mean(ss.sosfilt(sos_low, w) ** 2))
    clarity = mid / (low + 1e-6)               # high = melody dominates, little bass bleed
    loud = float(np.max(np.abs(w)))
    score = clarity * loud * min(gap, 0.6)
    if score > bs:
        bs, best = score, (s0, gap)
s0, gap = best
clip = y[s0:s0 + int(min(0.6, gap * 0.95) * SR)].astype("float32")
fo = min(int(0.03 * SR), len(clip) // 4)
clip[-fo:] *= np.linspace(1, 0, fo)
pk = np.max(np.abs(clip))
clip = clip / pk * 0.97 if pk else clip
f0 = librosa.yin(clip, fmin=80, fmax=2000, sr=SR); f0 = f0[np.isfinite(f0)]
root = int(round(librosa.hz_to_midi(np.median(f0)))) if len(f0) else 60
sf.write(OUT.replace(".wav", "_oneshot.wav"), clip, SR)
print(f"one-shot: root {pretty_midi.note_number_to_name(root)} ({gap:.2f}s gap)", flush=True)

# render the MIDI with that one-shot (pitch-shift per note)
pm = pretty_midi.PrettyMIDI(MID)
notes = [n for ins in pm.instruments for n in ins.notes]
total = max(n.end for n in notes) + 1.0
buf = np.zeros(int(total * SR) + len(clip) + SR, dtype="float32")
cache = {}
for n in notes:
    semis = n.pitch - root
    if semis not in cache:
        cache[semis] = librosa.effects.pitch_shift(clip, sr=SR, n_steps=semis)
    s = cache[semis]
    dur = max(int((n.end - n.start) * SR), int(0.1 * SR))
    seg = s[:dur] if len(s) >= dur else np.pad(s, (0, dur - len(s)))
    seg = seg * (n.velocity / 110.0)
    i0 = int(n.start * SR)
    buf[i0:i0 + len(seg)] += seg
pk = np.max(np.abs(buf))
buf = buf / pk * 0.97 if pk else buf
sf.write(OUT, buf, SR)
print(f"rendered {len(notes)} notes with the song's own piano -> {OUT}", flush=True)
