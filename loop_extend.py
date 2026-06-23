"""loop_extend.py — extend a short (phonk) loop into a full track by beat-aligned
crossfade tiling. Phonk is loop-based, so looping a strong short loop keeps the style
that long-form generation drifts away from.

  python loop_extend.py <in.wav> <out.wav> [target_seconds=150]
"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, soundfile as sf, librosa

IN  = sys.argv[1]
OUT = sys.argv[2]
TGT = float(sys.argv[3]) if len(sys.argv) > 3 else 150.0

y, sr = sf.read(IN)
if y.ndim == 1:
    y = np.stack([y, y], axis=1)

# find a beat-aligned loop length near the full clip (so repeats line up rhythmically)
mono = y.mean(axis=1)
tempo, beats = librosa.beat.beat_track(y=mono, sr=sr)
if len(beats) >= 8:
    beat_samps = librosa.frames_to_samples(beats)
    # use a loop that is a whole number of bars (4 beats), as long as possible within the clip
    bar_beats = 4
    n_bars = (len(beats) - 1) // bar_beats
    end_beat = n_bars * bar_beats
    loop = y[beat_samps[0]:beat_samps[end_beat]] if end_beat < len(beat_samps) else y
else:
    loop = y
loop_len = len(loop)

xf = int(0.04 * sr)  # 40ms crossfade between repeats
fade_in  = np.linspace(0, 1, xf)[:, None]
fade_out = np.linspace(1, 0, xf)[:, None]

target_samps = int(TGT * sr)
out = np.copy(loop)
while len(out) < target_samps:
    a = out[-xf:] * fade_out + loop[:xf] * fade_in   # crossfade seam
    out = np.concatenate([out[:-xf], a, loop[xf:]], axis=0)
out = out[:target_samps]

# final fade out
fo = int(2.0 * sr)
out[-fo:] *= np.linspace(1, 0, fo)[:, None]
sf.write(OUT, out, sr)
print(f"wrote {OUT}  ({len(out)/sr:.0f}s, loop {loop_len/sr:.1f}s, tempo~{float(tempo):.0f})", flush=True)
