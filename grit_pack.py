"""grit_pack.py — bake the phonkify grit into a pack's playable audio (one-shots, drum samples,
SFZ stems), in place. MIDI is left untouched (notes don't get gritted). So you load them in FL
and they're already crunchy.

  python grit_pack.py <pack_dir> [intensity=0.9]
"""
import sys, os, glob, subprocess, warnings
warnings.filterwarnings("ignore")

PACK = sys.argv[1]
INT = sys.argv[2] if len(sys.argv) > 2 else "0.9"
PHONK = "/mnt/d/flbeat/fl-beat-assimilator/phonkify.py"
PY = sys.executable
DIRS = ["1_Instruments (SFZ)", "3_Drum Samples", "4_One-shots"]

n = 0
for d in DIRS:
    for wav in sorted(glob.glob(os.path.join(PACK, d, "*.wav"))):
        tmp = wav + ".grit.wav"
        subprocess.run([PY, PHONK, wav, tmp, INT], capture_output=True, text=True)
        if not os.path.exists(tmp):
            print("FAIL:", os.path.basename(wav)); continue
        os.replace(tmp, wav)                       # overwrite original with gritted
        stray = os.path.splitext(tmp)[0] + ".mp3"  # phonkify also drops an .mp3
        if os.path.exists(stray):
            os.remove(stray)
        n += 1
print(f"gritted {n} files in {os.path.basename(PACK)} (intensity {INT})", flush=True)
