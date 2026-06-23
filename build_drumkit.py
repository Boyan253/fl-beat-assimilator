"""Build a playable DRUM KIT from a drums stem: cluster hits into kick / snare / hat by
spectral centroid (low/mid/high), pick one clean representative each, map to General-MIDI
keys (kick=36, snare=38, hat=42) as one-shots with velocity = loudness. Drive it with
create_beat.py's _drums.mid (also GM-mapped) to program your own drum patterns.
  python build_drumkit.py <drums.wav> <out_dir> <prefix>
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, soundfile as sf, librosa

STEM = sys.argv[1]; OUTDIR = sys.argv[2]; PREFIX = sys.argv[3]
KITDIR = os.path.join(OUTDIR, PREFIX + "_kit"); os.makedirs(KITDIR, exist_ok=True)

y, SR = sf.read(STEM, always_2d=False)
if getattr(y, "ndim", 1) > 1:
    y = y.mean(axis=1)
y = y.astype("float32")

ot = librosa.onset.onset_detect(y=y, sr=SR, backtrack=True, units="time")
hits = []
for t in ot:
    a = int(t * SR); b = min(int((t + 0.35) * SR), len(y))
    seg = y[a:b]
    if len(seg) < int(0.03 * SR):
        continue
    cen = float(librosa.feature.spectral_centroid(y=seg, sr=SR).mean())
    eng = float(np.sqrt(np.mean(seg ** 2)))
    hits.append((a, cen, eng))
if not hits:
    print("no hits found"); sys.exit(1)

cens = sorted(h[1] for h in hits)
lo, hi = cens[len(cens) // 3], cens[2 * len(cens) // 3]
groups = {"kick": [], "snare": [], "hat": []}
for h in hits:
    groups["kick" if h[1] <= lo else "snare" if h[1] <= hi else "hat"].append(h)

GM = {"kick": 36, "snare": 38, "hat": 42}
zc = np.where(np.diff(np.signbit(y)))[0]


def snap(s):
    return int(zc[np.argmin(np.abs(zc - s))]) if len(zc) else int(s)


lines = ["// playable DRUM KIT — kick=36 snare=38 hat=42, velocity=loudness, one-shot",
         "<global> ampeg_attack=0.001 ampeg_release=0.05 loop_mode=one_shot"]
made = []
for name, gr in groups.items():
    if not gr:
        continue
    a0 = snap(max(gr, key=lambda x: x[2])[0])              # loudest = cleanest
    seg = y[a0:min(a0 + int(0.35 * SR), len(y))].copy()
    fo = min(int(0.02 * SR), len(seg) // 4)
    seg[-fo:] *= np.linspace(1, 0, fo)
    fn = os.path.join(KITDIR, name + ".wav"); sf.write(fn, seg, SR)
    k = GM[name]
    lines.append("<region> sample=%s lokey=%d hikey=%d pitch_keycenter=%d" % (fn.replace("\\", "/"), k, k, k))
    made.append("%s=%d" % (name, k))
open(os.path.join(OUTDIR, PREFIX + "_kit.sfz"), "w").write("\n".join(lines))
print("DRUM KIT:", " ".join(made), "->", PREFIX + "_kit.sfz")
