"""build_grim_melody_real.py — PLAYABLE GRIM-melody instrument from GRIM's OWN audio.
For each pitch GRIM's melody uses, cut the CLEANEST single occurrence straight out of the
separated melody stem (real GRIM audio, native pitch = no stretch), map it to that key.
Skips silent/ghost occurrences. Driven by the aligned GRIM melody MIDI it reconstructs the
melody from GRIM's own sound.   usage: python build_grim_melody_real.py [melody.mid] [stem.wav]
"""
import os, sys, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np, soundfile as sf, pretty_midi

MIDI = sys.argv[1] if len(sys.argv) > 1 else "/mnt/d/flbeat/data/generated/GRIM_melody_aligned.mid"
STEM = sys.argv[2] if len(sys.argv) > 2 else "/mnt/d/flbeat/data/generated/GRIM_melody_6s_CLEAN.wav"
OUT  = "/mnt/d/flbeat/data/SEND/GRIMPack"
SAMP = os.path.join(OUT, "Samples")
os.makedirs(SAMP, exist_ok=True)
for f in glob.glob(os.path.join(SAMP, "GRIMmelR_*.wav")):       # clear old (misaligned) samples
    os.remove(f)

y, sr = sf.read(STEM)
if y.ndim > 1:
    y = y.mean(axis=1)
y = y.astype(np.float32)

pm = pretty_midi.PrettyMIDI(MIDI)
mel = [n for ins in pm.instruments
       if not ins.is_drum and "808" not in (ins.name or "").lower() and "bass" not in (ins.name or "").lower()
       for n in ins.notes]
mel.sort(key=lambda n: n.start)

def onset_energy(s):                                            # loudness right at the hit
    a = int(s * sr); b = min(len(y), a + int(0.15 * sr))
    seg = y[a:b]
    return float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0


best = {}                                                       # LOUDEST occurrence per pitch = the real hit
for n in mel:
    en = onset_energy(n.start)
    if n.pitch not in best or en > best[n.pitch][0]:
        best[n.pitch] = (en, n.start, n.end)


def cut(s, e):
    dur = min(max(e - s, 0.18), 0.35)                           # short hit window (cowbell) = minimal bleed
    a = max(0, int((s - 0.005) * sr)); b = min(len(y), int((s + dur + 0.06) * sr))
    seg = y[a:b].copy()
    fi = min(int(0.005 * sr), len(seg) // 4)
    fo = min(int(0.05 * sr), len(seg) // 3)
    if fi: seg[:fi] *= np.linspace(0, 1, fi)
    if fo: seg[-fo:] *= np.linspace(1, 0, fo)
    return (seg / (np.max(np.abs(seg)) + 1e-9) * 0.97).astype(np.float32)


rows, skipped = [], []
for p in sorted(best):
    en, s, e = best[p]
    if en < 0.015:                                              # genuinely silent everywhere = ghost pitch
        skipped.append(pretty_midi.note_number_to_name(p)); continue
    nm = f"GRIMmelR_{pretty_midi.note_number_to_name(p)}.wav"
    sf.write(os.path.join(SAMP, nm), cut(s, e), sr)
    rows.append((p, nm))

# zone-tile: EVERY key plays the nearest GRIM sample (pitched) = full playable instrument
ps = [p for p, _ in rows]
bnd = [(ps[i] + ps[i + 1]) // 2 for i in range(len(ps) - 1)]
zns = [(0 if i == 0 else bnd[i - 1] + 1, 127 if i == len(ps) - 1 else bnd[i]) for i in range(len(ps))]
L = ['<?xml version="1.0" encoding="UTF-8"?>', '<DecentSampler minVersion="1.0.0">',
     '  <groups attack="0.0" decay="0.0" sustain="1.0" release="0.12">', '    <group>']
for (p, nm), (lo, hi) in zip(rows, zns):
    L.append(f'      <sample path="Samples/{nm}" rootNote="{p}" loNote="{lo}" hiNote="{hi}" />')
L += ['    </group>', '  </groups>', '</DecentSampler>']
open(os.path.join(OUT, "GRIMMelodyReal.dspreset"), "w", encoding="utf-8").write("\n".join(L))

print(f"[grim-mel-real] {len(rows)} notes mapped: " +
      " ".join(pretty_midi.note_number_to_name(p) for p, _ in rows), flush=True)
if skipped:
    print("  skipped (silent/ghost): " + " ".join(skipped), flush=True)
