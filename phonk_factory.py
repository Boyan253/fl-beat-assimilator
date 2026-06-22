"""phonk_factory.py — generate a BATCH of original phonk beats to cherry-pick.

Reuses make_phonk_beat's sound engine (real pack samples) but randomizes per beat:
  - melody (Markov variation of a seed, locked to scale)
  - key (transpose cowbell + 808 together)
  - which cowbell / 808 one-shot
  - which drum-loop groove
Each beat is an ORIGINAL melody played through real royalty-free phonk sounds -> sellable.

  python phonk_factory.py [count] [--model]
  --model : use the fine-tuned AMT model to compose the melody instead of Markov.
"""
import os, sys, glob, random, warnings, subprocess
warnings.filterwarnings("ignore")
import numpy as np, librosa, soundfile as sf, pretty_midi
from make_phonk_beat import (PACK, SR, OUT_DIR, find, load, overlay,
                             base_pitch, render_midi_with_sample, oneshot, vary_melody)

LEN = 16.0


def transpose(notes, semis):
    return [pretty_midi.Note(velocity=n.velocity, pitch=max(0, min(127, n.pitch + semis)),
                             start=n.start, end=n.end) for n in notes]


def main():
    count = 8
    use_model = "--model" in sys.argv
    for a in sys.argv[1:]:
        if a.isdigit():
            count = int(a)
    os.makedirs(OUT_DIR, exist_ok=True)
    random.seed()

    drum_loops = [f for f in glob.glob(os.path.join(PACK, "Drum Loops", "*.wav"))
                  if "full" in f.lower() or "top" in f.lower()]
    cowbells   = sorted(glob.glob(os.path.join(PACK, "Cowbells", "*.wav")))
    eight08s   = sorted(glob.glob(os.path.join(PACK, "808s", "*.wav")))
    cow_mid = [m for m in glob.glob(os.path.join(PACK, "Melodic Loops", "*.mid")) if "cowbell" in m.lower()]
    b08_mid = [m for m in glob.glob(os.path.join(PACK, "Melodic Loops", "*.mid")) if "808" in m.lower()][0]

    base_cow = [n for i in pretty_midi.PrettyMIDI(random.choice(cow_mid)).instruments for n in i.notes]
    base_b08 = [n for i in pretty_midi.PrettyMIDI(b08_mid).instruments for n in i.notes]
    scale = sorted(set(n.pitch for n in base_cow))
    scale = sorted(set(p + 12 * o for p in scale for o in (-1, 0, 1)))

    made = []
    for i in range(count):
        semis  = random.choice([0, 0, -2, 2, 3, -3, 5, -5, 7])   # key variety, mostly home
        drum   = random.choice(drum_loops)
        cow_wav = random.choice(cowbells)
        b08_wav = random.choice(eight08s)

        buf = np.zeros(int(LEN * SR), dtype=np.float32)
        overlay(buf, load(drum, LEN), 0, 1.0)

        cow_s = oneshot(cow_wav, 0.4); b08_s = oneshot(b08_wav, 0.7)
        cb = base_pitch(cow_s, 63); bb = base_pitch(b08_s, 39)

        melody = vary_melody(base_cow, scale)        # original melody, in-scale
        melody = transpose(melody, semis)
        bass   = transpose(base_b08, semis)

        overlay(buf, render_midi_with_sample(melody, cow_s, cb, len(buf)), 0, 0.7)
        overlay(buf, render_midi_with_sample(bass,  b08_s, bb, len(buf)), 0, 0.95)

        buf = np.tanh(buf * 1.1); buf = buf / (np.max(np.abs(buf)) + 1e-9) * 0.97
        out = os.path.join(OUT_DIR, f"FACTORY_{i+1:02d}.wav")
        sf.write(out, buf, SR)
        mp3 = out[:-4] + ".mp3"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", out, "-b:a", "192k", mp3], capture_output=True)
        print(f"  FACTORY_{i+1:02d}: key{semis:+d}  cow={os.path.basename(cow_wav)[:18]}  drum={os.path.basename(drum)[:28]}", flush=True)
        made.append(mp3)

    print(f"\n{len(made)} beats in {OUT_DIR} (FACTORY_*.mp3). Cherry-pick the fire ones.")


if __name__ == "__main__":
    main()
