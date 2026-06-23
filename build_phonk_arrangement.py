"""build_phonk_arrangement.py — turn flat phonk stems into a structured phonk TRACK.

Fixes the data-measured gaps vs real phonk:
  - DYNAMICS/STRUCTURE (mine was 2.6x too flat): build intro/build/drop/break/drop/outro
    sections where stems drop in and out.
  - WEAK BASS: saturate + boost the 808.
  - TOO BRIGHT (trap): darken the master (high-shelf cut).

  python build_phonk_arrangement.py <stems_dir> <out.wav>
stems_dir must hold 808.wav, Melody.wav, Drums.wav (+ optional Hats.wav) from build_trackout.
"""
import os, sys, subprocess, warnings
warnings.filterwarnings("ignore")
import numpy as np, soundfile as sf, librosa, scipy.signal as ss

STEMS = sys.argv[1]
OUT   = sys.argv[2]
SR = 44100


def load(name):
    p = os.path.join(STEMS, name)
    if not os.path.exists(p):
        return None
    y, sr = sf.read(p)
    if y.ndim == 1:
        y = np.stack([y, y], axis=1)
    if sr != SR:
        y = np.stack([librosa.resample(y[:, c], orig_sr=sr, target_sr=SR) for c in range(2)], axis=1)
    return y.astype(np.float32)


stems = {n: load(n + ".wav") for n in ["808", "Melody", "Drums", "Hats"]}
stems = {k: v for k, v in stems.items() if v is not None}
ref = stems["Drums"] if "Drums" in stems else next(iter(stems.values()))

# loop unit = 4 bars (16 beats) from the drum tempo, beat-aligned
mono = ref.mean(axis=1)
tempo, beats = librosa.beat.beat_track(y=mono, sr=SR)
tempo = float(np.atleast_1d(tempo)[0])
if len(beats) >= 16:
    bs = librosa.frames_to_samples(beats)
    unit = int(bs[16] - bs[0])
else:
    unit = len(mono)
print(f"tempo~{tempo:.0f}, loop unit {unit/SR:.1f}s", flush=True)


def tile(y, n):
    need = unit * n
    reps = int(np.ceil(need / len(y)))
    return np.tile(y, (reps, 1))[:need]


def saturate(y, drive=2.2):
    return np.tanh(y * drive) / np.tanh(drive)


# section plan: (n_units, {stem: gain}, section_master_gain)  -- big gain swings = dynamics
PLAN = [
    ("intro", 1, {"Hats": 0.6, "Melody": 0.5}, 0.42),
    ("build", 1, {"Drums": 0.9, "Hats": 0.6, "Melody": 0.8}, 0.65),
    ("drop1", 2, {"Drums": 1.0, "808": 1.6, "Melody": 0.9, "Hats": 0.55}, 1.0),
    ("break", 1, {"Melody": 0.85, "808": 0.6}, 0.48),
    ("drop2", 2, {"Drums": 1.0, "808": 1.7, "Melody": 0.9, "Hats": 0.6}, 1.0),
    ("outro", 1, {"Drums": 0.6, "808": 0.9}, 0.5),
]

sections = []
for name, n, mix, mg in PLAN:
    seg = np.zeros((unit * n, 2), dtype=np.float32)
    for stem, g in mix.items():
        if stem in stems:
            s = tile(stems[stem], n)
            if stem == "808":
                s = saturate(s, 2.0) * g          # heavy 808 (gentler drive = less brightness)
            else:
                s = s * g
            seg += s
    seg *= mg                                      # section master gain -> dynamics
    sections.append(seg)
    print(f"  {name}: {n} units x{mg}, stems {list(mix)}", flush=True)

# concat with short crossfades
xf = int(0.03 * SR)
fi = np.linspace(0, 1, xf)[:, None]; fo = np.linspace(1, 0, xf)[:, None]
track = sections[0]
for seg in sections[1:]:
    track = np.concatenate([track[:-xf], track[-xf:] * fo + seg[:xf] * fi, seg[xf:]], axis=0)

# DARKEN master hard (phonk is dark/lo-fi): low-pass ~5.2kHz to kill trap crispness
sos = ss.butter(4, 5200, btype="low", fs=SR, output="sos")
track = ss.sosfilt(sos, track, axis=0).astype(np.float32)
# normalize WITHOUT master saturation (saturation was adding brightness)
track = track / (np.max(np.abs(track)) + 1e-9) * 0.97
# 2s fade out
fo2 = int(2 * SR); track[-fo2:] *= np.linspace(1, 0, fo2)[:, None]

sf.write(OUT, track, SR)
mp3 = os.path.splitext(OUT)[0] + ".mp3"
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", OUT, "-b:a", "192k", mp3], capture_output=True)
print(f"wrote {mp3}  ({len(track)/SR:.0f}s, {len(PLAN)} sections)", flush=True)
