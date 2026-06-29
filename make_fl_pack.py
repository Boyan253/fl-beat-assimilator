"""make_fl_pack.py — one track -> full FL-ready pack (stems + 1:1 MIDI + SFZ instruments + one-shots).
Runs entirely in flbeat-venv. Best models: htdemucs_ft (stems), MDX23C-DrumSep (drums),
YourMT3 (melody/piano), basic_pitch (bass), build_perocc (SFZ from the real audio).

  python make_fl_pack.py <track.wav> <NAME> <bpm>
Output -> /mnt/d/flbeat/data/FL_PACKS/<NAME>/
"""
import os, sys, glob, re, shutil, subprocess, types, warnings
from math import ceil
warnings.filterwarnings("ignore")
import numpy as np, librosa, soundfile as sf, pretty_midi

SONG = sys.argv[1]; NAME = sys.argv[2]; BPM = float(sys.argv[3]) if len(sys.argv) > 3 else 120.0
ROOT = f"/mnt/d/flbeat/data/FL_PACKS/{NAME}"
WORK = f"{ROOT}/_work"
REPO = "/mnt/d/flbeat/fl-beat-assimilator"
PY = sys.executable
for d in (ROOT, WORK):
    os.makedirs(d, exist_ok=True)
INST = f"{ROOT}/1_Instruments (SFZ)"; MIDI = f"{ROOT}/2_MIDI (editable)"
DRS = f"{ROOT}/3_Drum Samples"; ONE = f"{ROOT}/4_One-shots"
for d in (INST, MIDI, DRS, ONE):
    os.makedirs(d, exist_ok=True)


os.chdir("/mnt/d/flbeat")   # so YourMT3 finds .mt3_checkpoints


def run(cmd, env=None):
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        sys.stderr.write((r.stderr or "")[-1500:])
        raise SystemExit(f"FAILED: {' '.join(str(c) for c in cmd[:3])} ...")


# 1) stems (htdemucs_ft, GPU)
print("[1/7] separating stems (htdemucs_ft)...", flush=True)
run([PY, "-m", "demucs", "-n", "htdemucs_ft", "-d", "cuda", "-o", WORK, SONG])
stemdir = os.path.join(WORK, "htdemucs_ft", os.path.splitext(os.path.basename(SONG))[0])
S = {k: os.path.join(stemdir, f"{k}.wav") for k in ("drums", "bass", "other", "vocals")}

# 2) DrumSep the drums (kick/snare/hh/toms/ride/crash)
print("[2/7] DrumSep drums...", flush=True)
dsdir = os.path.join(WORK, "drumsep"); os.makedirs(dsdir, exist_ok=True)
run(["audio-separator", S["drums"], "--model_filename", "MDX23C-DrumSep-aufr33-jarredou.ckpt",
     "--output_dir", dsdir])

# 3) melody/piano -> YourMT3
print("[3/7] transcribe piano/melody (YourMT3)...", flush=True)
run([PY, os.path.join(REPO, "run_mt3.py"), S["other"], os.path.join(MIDI, "Piano.mid"), "yourmt3"])

# 4) bass -> basic_pitch
print("[4/7] transcribe bass (basic_pitch)...", flush=True)
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
_, bmid, _ = predict(S["bass"], ICASSP_2022_MODEL_PATH, onset_threshold=0.5, frame_threshold=0.3,
                     minimum_note_length=90, minimum_frequency=28, maximum_frequency=350)
out = pretty_midi.PrettyMIDI(initial_tempo=BPM)
for ins in bmid.instruments:
    out.instruments.append(ins)
out.write(os.path.join(MIDI, "808.mid"))

# 5) drum MIDI from the DrumSep stems
print("[5/7] drum MIDI...", flush=True)
run([PY, os.path.join(REPO, "transcribe_drumsep.py"), dsdir, MIDI, str(int(BPM))])

# 6) SFZ instruments (play the real audio chunks)
print("[6/7] SFZ instruments...", flush=True)
run([PY, os.path.join(REPO, "build_perocc.py"), S["other"], INST, f"{NAME}_piano", "0.2"])
run([PY, os.path.join(REPO, "build_perocc_onset.py"), S["bass"], INST, f"{NAME}_bass", "36"])
run([PY, os.path.join(REPO, "build_perocc_onset.py"), S["drums"], INST, f"{NAME}_drums", "38"])
# fix sample paths -> relative basename + copy the stem next to its .sfz
for sfz, stem, nm in [(f"{NAME}_piano", S["other"], "piano.wav"),
                      (f"{NAME}_bass", S["bass"], "bass.wav"),
                      (f"{NAME}_drums", S["drums"], "drums.wav")]:
    p = os.path.join(INST, sfz + ".sfz")
    txt = open(p).read()
    txt = re.sub(r"sample=[^ ]+/", "sample=", txt)
    open(p, "w").write(txt)
    shutil.copy(stem, os.path.join(INST, nm))

# 7) drum samples (flac->wav) + one-shots
print("[7/7] drum samples + one-shots...", flush=True)
NMAP = {"kick": "Kick", "snare": "Snare", "hh": "HiHat", "toms": "Toms", "ride": "Ride", "crash": "Crash"}
for flac in glob.glob(os.path.join(dsdir, "*.flac")):
    m = re.search(r"\((\w+)\)", os.path.basename(flac))
    nm = NMAP.get(m.group(1), m.group(1)) if m else "drum"
    run(["ffmpeg", "-y", "-i", flac, os.path.join(DRS, nm + ".wav")])


def oneshot(src, name, maxlen, tonal, fmin=0, fmax=0):
    y, sr = librosa.load(src, sr=44100, mono=True)
    on = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=True)
    if len(on) == 0:
        return
    best, bs = (0, maxlen), -1
    for i, t in enumerate(on):
        nxt = on[i + 1] if i + 1 < len(on) else len(y) / sr
        s0 = int(t * sr); seg = y[s0:s0 + int(0.05 * sr)]
        pk = float(np.max(np.abs(seg))) if len(seg) else 0
        sc = pk * min(nxt - t, maxlen)
        if sc > bs:
            bs, best = sc, (s0, nxt - t)
    s0, gap = best
    clip = y[s0:s0 + int(min(maxlen, max(0.1, gap * 0.95)) * sr)].astype("float32")
    fo = min(int(0.02 * sr), len(clip) // 4)
    if fo:
        clip[-fo:] *= np.linspace(1, 0, fo)
    pk = np.max(np.abs(clip));  clip = clip * (0.97 / pk) if pk > 0 else clip
    root = ""
    if tonal:
        f0 = librosa.yin(clip, fmin=fmin, fmax=fmax, sr=sr); f0 = f0[np.isfinite(f0)]
        if len(f0):
            root = "_" + pretty_midi.note_number_to_name(int(round(librosa.hz_to_midi(np.median(f0)))))
    sf.write(os.path.join(ONE, f"{name}{root}.wav"), clip, sr)


oneshot(S["other"], "Piano", 0.8, True, 80, 2000)
oneshot(S["bass"], "808", 0.75, True, 28, 300)
for tag in ("kick", "snare", "hh"):
    fl = glob.glob(os.path.join(dsdir, f"*({tag})*"))
    if fl:
        oneshot(fl[0], NMAP[tag], 0.4 if tag != "hh" else 0.18, False)

open(os.path.join(ROOT, "README.txt"), "w").write(
    f"{NAME} - FL pack  (set project tempo to {int(BPM)} BPM)\n"
    "1_Instruments (SFZ): load .sfz in Sforzando + its same-name .mid = sounds like the beat.\n"
    "2_MIDI (editable): Piano.mid (YourMT3), 808.mid, per-drum + Drums.mid (GM). Use with your own sounds.\n"
    "3_Drum Samples: real separated drums (Kick/Snare/HiHat/...).\n"
    "4_One-shots: one real hit each (Piano/808 named with root note) - play with the MIDI in folder 2.\n")
shutil.rmtree(WORK, ignore_errors=True)
print(f"DONE -> {ROOT}", flush=True)
