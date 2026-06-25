"""ASSIMILATE — one command: any song -> a folder of FL-ready editable instruments.

Separates the song (htdemucs_ft), routes each stem to the right per-occurrence engine
(melodic transcriber for leads/vocals, energy-onset for bass/drums), builds SFZ + MIDI,
renders and VERIFIES each layer against its source stem (chroma + rms correlation).
Every layer is restart-safe (velocity-layer) and time-locked (silent t=0 anchor), so you
load each .sfz in its own Sforzando channel, drop the .mid, line clips up at bar 1 -> a
multi-pattern beat that sounds like the original and is fully editable.

  python assimilate.py <song.mp3> <name>
Output -> D:\\flmidi\\out\\<name>\\ : <name>_melody.sfz/.mid, <name>_bass.sfz/.mid, ...
"""
import os, sys, subprocess, warnings
warnings.filterwarnings("ignore")
import numpy as np, librosa

D = r"D:\flmidi"
PY = sys.executable
SFIZZ = os.path.join(D, r"sfizz\x\bin\Release\sfizz_render.exe")

MP3 = sys.argv[1]
NAME = sys.argv[2]
OUTDIR = os.path.join(D, "out", NAME)
STEMS = os.path.join(OUTDIR, "stems_ft")
os.makedirs(OUTDIR, exist_ok=True)

# stem -> (layer label, engine script, default pitch for the engine)
ROUTING = [
    ("other.wav",  "melody", "build_perocc.py",       "0.2"),
    ("vocals.wav", "vocals", "build_perocc.py",       "0.2"),
    ("bass.wav",   "bass",   "build_perocc_onset.py", "36"),
    ("drums.wav",  "drums",  "build_perocc_onset.py", "38"),
]
SILENCE_RMS = 0.008  # skip near-silent stems (e.g. an instrumental track's empty vocals)


def rms(p):
    y, sr = librosa.load(p, sr=22050, mono=True)
    return float(np.sqrt(np.mean(y ** 2)))


def feats(p):
    y, sr = librosa.load(p, sr=22050, mono=True); y = y[:sr * 60]
    return librosa.feature.chroma_cqt(y=y, sr=sr), librosa.feature.rms(y=y)[0]


def corr(a, b):
    m = min(a.shape[-1], b.shape[-1])
    return float(np.corrcoef(a[..., :m].flatten(), b[..., :m].flatten())[0, 1])


# 1) separate (skip if already done)
if os.path.exists(os.path.join(STEMS, "other.wav")):
    print("=== stems already present -> skipping separation ===", flush=True)
else:
    print("=== separating with htdemucs_ft (slow on CPU) ===", flush=True)
    subprocess.run([PY, os.path.join(D, "sep_ft.py"), MP3, STEMS, "htdemucs_ft"], check=True)

# 2) build + render + verify each present layer
results = []
for stem, layer, engine, defp in ROUTING:
    src = os.path.join(STEMS, stem)
    if not os.path.exists(src):
        continue
    if rms(src) < SILENCE_RMS:
        print("=== %s near-silent -> skipped ===" % layer, flush=True)
        continue
    prefix = "%s_%s" % (NAME, layer)
    print("=== %s : building via %s ===" % (layer, engine), flush=True)
    cmd = [PY, os.path.join(D, engine), src, OUTDIR, prefix]
    if defp:
        cmd.append(defp)
    subprocess.run(cmd, check=True)
    sfz = os.path.join(OUTDIR, prefix + ".sfz")
    mid = os.path.join(OUTDIR, prefix + ".mid")
    wav = os.path.join(OUTDIR, prefix + ".wav")
    subprocess.run([SFIZZ, "--sfz", sfz, "--midi", mid, "--wav", wav, "-s", "44100", "--use-eot"],
                   capture_output=True)
    if not os.path.exists(wav):
        print("!! render failed for %s" % layer, flush=True)
        continue
    # loudness-match: per-occurrence renders are quieter than the stem (envelope dips on each
    # chunk). Bake a gain into the SFZ: RMS-match for sustained content, capped by peak so
    # transient bass/drums don't clip.
    sy, _ = librosa.load(src, sr=None, mono=True); ry, _ = librosa.load(wav, sr=None, mono=True)
    s_rms = float(np.sqrt(np.mean(sy ** 2))); r_rms = float(np.sqrt(np.mean(ry ** 2))); r_pk = float(np.max(np.abs(ry)))
    if r_rms > 0 and r_pk > 0:
        gdb = 20.0 * np.log10(min(s_rms / r_rms, 0.99 / r_pk))
        _t = open(sfz).read().split("\n"); _t[1] += " volume=%.1f" % gdb; open(sfz, "w").write("\n".join(_t))
        print("   %s loudness-matched: %+.1f dB" % (layer, gdb))
    cs, rs = feats(src); cr, rr = feats(wav)
    results.append((layer, corr(cs, cr), corr(rs, rr), prefix))

# 3) report
print("\n================ ASSIMILATION COMPLETE ================")
print("song : %s" % os.path.basename(MP3))
print("out  : %s" % OUTDIR)
print("layer    chroma   rms    files  (chroma=pitch match, rms=timing match)")
for layer, cc, rc, prefix in results:
    print("  %-7s %5.2f   %5.2f   %s.sfz + .mid" % (layer, cc, rc, prefix))
print("\nLoad each .sfz in its own Sforzando channel, drop the matching .mid, place every clip")
print("at bar 1. (chroma is meaningless for drums/sub-bass — judge those by rms/ear.)")
