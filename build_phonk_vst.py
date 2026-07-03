"""build_phonk_vst.py — build OUR OWN sendable phonk instrument as a DecentSampler library
AND an SFZ, filled with copyright-free, from-scratch synthesized sounds (808 cowbell + 808 +
drums). No AI audio, no GRIM, nothing extracted. Each key is a clean single tone (multisampled
every few semitones) so there is NO "one note sounds like several" artifact.

Your friend installs the free DecentSampler (or Sforzando) VST once, loads this library, done.
Both load in FL Studio as a normal VST. Honest answer to "what is it": "my own cowbell
instrument in DecentSampler."

  python build_phonk_vst.py [outdir=/mnt/d/flbeat/data/SEND/PhonkPack]
"""
import os, sys, shutil, warnings
warnings.filterwarnings("ignore")
import numpy as np, soundfile as sf, scipy.signal as ss, pretty_midi

SR  = 44100
OUT = sys.argv[1] if len(sys.argv) > 1 else "/mnt/d/flbeat/data/SEND/PhonkPack"
SAMP = os.path.join(OUT, "Samples")
if os.path.isdir(OUT): shutil.rmtree(OUT)
os.makedirs(SAMP, exist_ok=True)


def f_of(p): return 440.0 * 2.0 ** ((p - 69) / 12.0)


# ---- copyright-free synth voices (identical character to the preview) -----------------
def cowbell(f, dur=0.6, vel=110):          # authentic 808 cowbell: 2 squares a ~fifth apart
    fb = f * 2.0
    n = int(dur * SR); t = np.arange(n) / SR
    sq = lambda fr: np.sign(np.sin(2 * np.pi * fr * t))
    sig = 0.5 * sq(fb) + 0.5 * sq(fb * 1.48)
    lo, hi = max(fb * 0.6, 320), min(fb * 2.2, 7000)
    sig = ss.sosfilt(ss.butter(2, [lo, hi], btype="band", fs=SR, output="sos"), sig)
    env = np.exp(-t * 9.0); a = int(0.002 * SR); env[:a] *= np.linspace(0, 1, a)
    return (np.tanh(sig * env * 2.2) / np.tanh(2.2) * (vel / 110.0)).astype(np.float32)


def bass808(f, dur=1.0, vel=120):
    n = int(dur * SR); t = np.arange(n) / SR
    gl = int(0.03 * SR); fc = np.full(n, f, dtype=np.float32)
    if gl: fc[:gl] = np.linspace(f * 1.5, f, gl)
    env = np.ones(n, dtype=np.float32)
    a = int(0.004 * SR); r = int(0.25 * SR); env[:a] = np.linspace(0, 1, a); env[-r:] = np.linspace(1, 0, r)
    sig = np.sin(2 * np.pi * np.cumsum(fc) / SR) * env
    return (np.tanh(sig * 3.2) / np.tanh(3.2) * (vel / 120.0)).astype(np.float32)


def kick():
    n = int(0.32 * SR); t = np.arange(n) / SR
    fc = 120 * np.exp(-t * 38) + 45
    sig = np.sin(2 * np.pi * np.cumsum(fc) / SR) * np.exp(-t * 9); sig[:80] += np.linspace(1, 0, 80) * 0.6
    return np.tanh(sig * 2.0).astype(np.float32)


def snare():
    n = int(0.22 * SR); t = np.arange(n) / SR
    tone = np.sin(2 * np.pi * 185 * t) * np.exp(-t * 22) * 0.5
    nz = ss.sosfilt(ss.butter(4, [1400, 3600], btype="band", fs=SR, output="sos"), np.random.randn(n)) * np.exp(-t * 16)
    return (tone + nz).astype(np.float32)


def hat():
    n = int(0.06 * SR); t = np.arange(n) / SR
    return (ss.sosfilt(ss.butter(4, [3500, 5800], btype="band", fs=SR, output="sos"), np.random.randn(n)) * np.exp(-t * 60)).astype(np.float32)


def wwrite(name, y):
    y = y / (np.max(np.abs(y)) + 1e-9) * 0.97
    sf.write(os.path.join(SAMP, name), y.astype(np.float32), SR)


# ---- render multisamples (one clean tone every few semitones = no stretch artifact) ----
cow = []
for r in range(48, 85, 3):                 # C3..C6
    nm = f"Cowbell_{pretty_midi.note_number_to_name(r)}.wav"; wwrite(nm, cowbell(f_of(r))); cow.append((r, nm))
bs = []
for r in range(16, 53, 4):                 # low 808 range
    nm = f"808_{pretty_midi.note_number_to_name(r)}.wav"; wwrite(nm, bass808(f_of(r))); bs.append((r, nm))
wwrite("Kick.wav", kick()); wwrite("Snare.wav", snare()); wwrite("Hat.wav", hat())


def zones(roots):                          # tile key ranges around each sampled root
    b = [(roots[i] + roots[i + 1]) // 2 for i in range(len(roots) - 1)]
    return [(0 if i == 0 else b[i - 1] + 1, 127 if i == len(roots) - 1 else b[i]) for i in range(len(roots))]


cow_full = [(r, nm, lo, hi) for (r, nm), (lo, hi) in zip(cow, zones([r for r, _ in cow]))]
bas_full = [(r, nm, lo, hi) for (r, nm), (lo, hi) in zip(bs, zones([r for r, _ in bs]))]
kit_full = [(36, "Kick.wav", 36, 36), (38, "Snare.wav", 38, 38), (42, "Hat.wav", 42, 42)]


# ---- DecentSampler (.dspreset) ---------------------------------------------------------
def dspreset(fname, rows, amp, lowpass=6000):
    L = ['<?xml version="1.0" encoding="UTF-8"?>', '<DecentSampler minVersion="1.0.0">',
         f'  <groups attack="{amp[0]}" decay="{amp[1]}" sustain="{amp[2]}" release="{amp[3]}">', '    <group>']
    for (r, nm, lo, hi) in rows:
        L.append(f'      <sample path="Samples/{nm}" rootNote="{r}" loNote="{lo}" hiNote="{hi}" />')
    L += ['    </group>', '  </groups>', '  <effects>',
          f'    <effect type="lowpass" frequency="{lowpass}.0" resonance="0.5" />',
          '    <effect type="reverb" wetLevel="0.12" roomSize="0.4" />', '  </effects>', '</DecentSampler>']
    open(os.path.join(OUT, fname), "w", encoding="utf-8").write("\n".join(L))


# ---- SFZ (.sfz, for Sforzando) ---------------------------------------------------------
def sfz(fname, rows, amp, lowpass=6000):
    L = ['<global>', f'ampeg_attack={amp[0]} ampeg_decay={amp[1]} ampeg_sustain={int(amp[2] * 100)} ampeg_release={amp[3]}',
         f'fil_type=lpf_2p cutoff={lowpass}', '']
    for (r, nm, lo, hi) in rows:
        L.append(f'<region> sample=Samples/{nm} pitch_keycenter={r} lokey={lo} hikey={hi}')
    open(os.path.join(OUT, fname), "w", encoding="utf-8").write("\n".join(L))


COW_AMP = (0.001, 0.45, 0.0, 0.20)         # percussive pluck
BAS_AMP = (0.003, 0.20, 1.0, 0.30)         # sustained bass
KIT_AMP = (0.000, 0.30, 0.0, 0.15)
for fn, rows, amp in [("PhonkCowbell.dspreset", cow_full, COW_AMP),
                      ("Phonk808.dspreset", bas_full, BAS_AMP),
                      ("PhonkDrumKit.dspreset", kit_full, KIT_AMP)]:
    dspreset(fn, rows, amp)
for fn, rows, amp in [("PhonkCowbell.sfz", cow_full, COW_AMP),
                      ("Phonk808.sfz", bas_full, BAS_AMP),
                      ("PhonkDrumKit.sfz", kit_full, KIT_AMP)]:
    sfz(fn, rows, amp)

open(os.path.join(OUT, "SEND_README.txt"), "w", encoding="utf-8").write(
    "PHONK PACK -- a custom instrument you can send.\n"
    "================================================\n"
    "These sounds are synthesized from scratch (an 808 cowbell, an 808 bass, drums).\n"
    "Copyright-free, not AI-generated audio, nothing extracted from any song. They are YOURS.\n\n"
    "HOW YOUR FRIEND USES IT (free, ~2 min):\n"
    "  Option A - DecentSampler (recommended): install the free DecentSampler VST\n"
    "             (decentsampler.com), in FL add it as a VST, click its folder icon and open\n"
    "             PhonkCowbell.dspreset (or Phonk808 / PhonkDrumKit). Keep the Samples\n"
    "             folder next to the .dspreset files.\n"
    "  Option B - Sforzando: install the free Sforzando VST (plogue.com), load the matching\n"
    "             .sfz file. Same sounds.\n\n"
    "WHAT TO SAY IT IS:  'my own 808-cowbell instrument, loaded in DecentSampler.' True.\n"
    "  - PhonkCowbell  = the melody (the phonk signature sound), playable across the keyboard\n"
    "  - Phonk808      = the bass\n"
    "  - PhonkDrumKit  = kick (C2) / snare (D2) / hat (F#2)\n")

print(f"[vst] cowbell={len(cow)} 808={len(bs)} +3 drums -> {OUT}", flush=True)
print("  presets: PhonkCowbell / Phonk808 / PhonkDrumKit  (.dspreset + .sfz)", flush=True)
