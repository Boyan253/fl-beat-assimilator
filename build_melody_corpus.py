"""build_melody_corpus.py — turn our 248 transcribed phonk songs into a CLEAN
monophonic phonk-melody training corpus (the fuel for a real melody model).

For each multi-track MIDI we made from a phonk song, take the melody track and:
  - monophonic-ize (drop the Basic Pitch harmonic/ghost notes -> one note at a time)
  - thin very short notes
  - snap every note to the song's best-fit minor scale (phonk = minor/Phrygian)
  - transpose-augment to 12 keys (multiplies the data)
Output -> /mnt/d/flbeat/data/melody_corpus/*.mid  (single-track cowbell-style melodies)

  python build_melody_corpus.py
"""
import os, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np, pretty_midi

SRC = "/mnt/d/flbeat/data/midi_mt"
PACK = "/mnt/d/flbeat/data/phonk_packs"
OUT = "/mnt/d/flbeat/data/melody_corpus"

# natural-minor scale template (semitone classes), used to pick the key
MINOR = np.array([1, 0, .4, 1, 0, 1, 0, 1, 1, 0, .6, 0], dtype=float)  # weights for minor degrees


def best_minor_root(pitches):
    hist = np.zeros(12)
    for p in pitches:
        hist[p % 12] += 1
    if hist.sum() == 0:
        return 0
    hist /= hist.sum()
    scores = [np.dot(hist, np.roll(MINOR, r)) for r in range(12)]
    return int(np.argmax(scores))


def scale_tones(root):
    return set((root + d) % 12 for d in (0, 2, 3, 5, 7, 8, 10))  # natural minor


def monophonic(notes):
    notes = sorted(notes, key=lambda n: (n.start, -n.pitch))
    out = []
    for n in notes:
        if out and n.start < out[-1].end - 0.01:        # overlaps the held note
            if n.start - out[-1].start < 0.06 and n.pitch > out[-1].pitch:
                out[-1] = n                               # near-simultaneous: keep the higher
            continue                                      # else drop (it's harmony/ghost)
        out.append(n)
    return [n for n in out if (n.end - n.start) >= 0.05]


def snap_to_scale(notes, tones):
    snapped = []
    for n in notes:
        pc = n.pitch % 12
        if pc in tones:
            p = n.pitch
        else:
            # move to nearest scale tone
            d = min(range(-2, 3), key=lambda k: (abs(k), 0) if (n.pitch + k) % 12 in tones else (9, 0))
            p = n.pitch + d
        snapped.append(pretty_midi.Note(velocity=n.velocity, pitch=int(p), start=n.start, end=n.end))
    return snapped


def save(notes, path):
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=30, name="cowbell")
    inst.notes = notes
    pm.instruments.append(inst)
    pm.write(path)


def main():
    os.makedirs(OUT, exist_ok=True)
    files = glob.glob(os.path.join(SRC, "*.mid"))
    n_in = n_out = 0
    for f in files:
        try:
            pm = pretty_midi.PrettyMIDI(f)
        except Exception:
            continue
        mel = [n for i in pm.instruments if not i.is_drum and i.program == 30 for n in i.notes]
        if len(mel) < 12:
            continue
        n_in += 1
        clean = monophonic(mel)
        if len(clean) < 8:
            continue
        root = best_minor_root([n.pitch for n in clean])
        tones = scale_tones(root)
        clean = snap_to_scale(clean, tones)
        base = os.path.splitext(os.path.basename(f))[0][:40]
        # transpose-augment: a few keys (not all 12, to keep it phonk-register and bounded)
        for t in (-4, -2, 0, 2, 4, 5):
            notes = [pretty_midi.Note(velocity=n.velocity, pitch=max(24, min(96, n.pitch + t)),
                                      start=n.start, end=n.end) for n in clean]
            save(notes, os.path.join(OUT, f"{base}_k{t:+d}.mid"))
            n_out += 1

    # add clean pack melodies (already monophonic & in-scale) + augment
    for f in glob.glob(os.path.join(PACK, "**", "*.mid"), recursive=True):
        if "cowbell" not in os.path.basename(f).lower():
            continue
        try:
            mel = [n for i in pretty_midi.PrettyMIDI(f).instruments for n in i.notes]
        except Exception:
            continue
        if len(mel) < 6:
            continue
        base = "PACK_" + os.path.splitext(os.path.basename(f))[0][:30]
        for t in range(-5, 6):
            notes = [pretty_midi.Note(velocity=n.velocity, pitch=max(24, min(96, n.pitch + t)),
                                      start=n.start, end=n.end) for n in mel]
            save(notes, os.path.join(OUT, f"{base}_k{t:+d}.mid"))
            n_out += 1

    total = len(glob.glob(os.path.join(OUT, "*.mid")))
    print(f"input songs with melody: {n_in}")
    print(f"clean melody MIDIs written (incl. transpositions): {total}")


if __name__ == "__main__":
    main()
