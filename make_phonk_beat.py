"""make_phonk_beat.py — assemble a phonk beat from a BVKER-style sample pack.

Two modes:
  --mix   : mix the pack's pre-rendered WAV loops (cowbell + 808 + drums) -> proves the sound.
  --build : render a (possibly NEW/varied) cowbell-melody MIDI + 808 MIDI through the pack's
            cowbell/808 one-shot samples, over a real drum loop -> flexible, can be original.

  python make_phonk_beat.py [--mix|--build] [--vary] [out.wav]

Everything here is royalty-free pack content (BVKER) -> sellable.
"""
import os, sys, glob, random, warnings, subprocess
warnings.filterwarnings("ignore")
import numpy as np, librosa, soundfile as sf, pretty_midi

SR   = 44100
PACK = "/mnt/d/flbeat/data/phonk_packs/BVKER - Lunatic Phonk"
OUT_DIR = "/mnt/d/flbeat/data/generated"


def find(folder, *kw):
    """First wav in PACK/folder whose name contains all keywords (case-insensitive)."""
    for f in sorted(glob.glob(os.path.join(PACK, folder, "*.wav"))):
        low = os.path.basename(f).lower()
        if all(k.lower() in low for k in kw):
            return f
    # fallback: any wav in folder
    g = sorted(glob.glob(os.path.join(PACK, folder, "*.wav")))
    return g[0] if g else None


def load(p, dur=None):
    y, _ = librosa.load(p, sr=SR, mono=True)
    if dur:
        n = int(dur * SR)
        y = np.tile(y, int(np.ceil(n / len(y))))[:n] if len(y) < n else y[:n]
    return y


def overlay(buf, clip, at, gain):
    e = min(at + len(clip), len(buf))
    if at < 0 or e <= at:
        return
    buf[at:e] += clip[:e - at] * gain


def base_pitch(clip, fallback):
    try:
        f0, _, _ = librosa.pyin(clip, fmin=30, fmax=1200, sr=SR)
        f0 = f0[~np.isnan(f0)]
        if len(f0) >= 3:
            return float(librosa.hz_to_midi(np.median(f0)))
    except Exception:
        pass
    return float(fallback)


def render_midi_with_sample(notes, sample, base, length_samp):
    buf = np.zeros(length_samp, dtype=np.float32)
    cache = {}
    for n in notes:
        semis = max(-24, min(24, int(round(n.pitch - base))))
        if semis not in cache:
            cache[semis] = librosa.effects.pitch_shift(sample, sr=SR, n_steps=semis)
        s = cache[semis]
        overlay(buf, s, int(n.start * SR), 0.4 + 0.6 * n.velocity / 127)
    return buf


def oneshot(p, dur):
    y = load(p)[: int(dur * SR)]
    fo = min(int(0.03 * SR), len(y) // 3)
    if fo > 0:
        y[-fo:] *= np.linspace(1, 0, fo)
    return y / (np.max(np.abs(y)) + 1e-9)


def vary_melody(notes, scale_pitches):
    """Generate a NEW melody: keep the rhythm, resample pitches from the song's scale
    via a simple order-1 Markov over the original pitch sequence (stays in D# Phrygian)."""
    if len(notes) < 4:
        return notes
    seq = [n.pitch for n in sorted(notes, key=lambda x: x.start)]
    trans = {}
    for a, b in zip(seq, seq[1:]):
        trans.setdefault(a, []).append(b)
    out = []
    cur = random.choice(seq)
    for n in sorted(notes, key=lambda x: x.start):
        nxt = random.choice(trans.get(cur, seq))
        # snap to scale
        nxt = min(scale_pitches, key=lambda p: abs(p - nxt))
        out.append(pretty_midi.Note(velocity=n.velocity, pitch=nxt, start=n.start, end=n.end))
        cur = nxt
    return out


def main():
    mode = "--mix"
    vary = "--vary" in sys.argv
    for a in sys.argv[1:]:
        if a in ("--mix", "--build"):
            mode = a
    outs = [a for a in sys.argv[1:] if a.endswith(".wav")]
    out_wav = outs[0] if outs else os.path.join(OUT_DIR, "PHONK_BEAT.wav")
    os.makedirs(OUT_DIR, exist_ok=True)

    drum = find("Drum Loops", "full drum loop")
    LEN = 16.0  # one 175 BPM loop cycle
    drums = load(drum, LEN)
    buf = np.zeros(int(LEN * SR), dtype=np.float32)
    overlay(buf, drums, 0, 1.0)
    print(f"drums: {os.path.basename(drum)}", flush=True)

    if mode == "--mix":
        cow = find("Melodic Loops", "cowbell melody", "175")
        b08 = find("Melodic Loops", "808 loop", "175")
        overlay(buf, load(cow, LEN), 0, 0.7); print(f"cowbell loop: {os.path.basename(cow)}", flush=True)
        overlay(buf, load(b08, LEN), 0, 0.95); print(f"808 loop: {os.path.basename(b08)}", flush=True)
    else:  # --build : render MIDI through one-shot samples
        cow_mid = [m for m in glob.glob(os.path.join(PACK, "Melodic Loops", "*.mid")) if "cowbell" in m.lower() and "175" in m][0]
        b08_mid = [m for m in glob.glob(os.path.join(PACK, "Melodic Loops", "*.mid")) if "808" in m.lower()][0]
        cow_s = oneshot(sorted(glob.glob(os.path.join(PACK, "Cowbells", "*.wav")))[0], 0.4)
        b08_s = oneshot(sorted(glob.glob(os.path.join(PACK, "808s", "*.wav")))[0], 0.7)
        cb = base_pitch(cow_s, 63); bb = base_pitch(b08_s, 39)
        cow_notes = [n for i in pretty_midi.PrettyMIDI(cow_mid).instruments for n in i.notes]
        b08_notes = [n for i in pretty_midi.PrettyMIDI(b08_mid).instruments for n in i.notes]
        if vary:
            scale = sorted(set(n.pitch for n in cow_notes))
            # extend scale across 2 octaves for variety
            scale = sorted(set([p + 12 * o for p in scale for o in (-1, 0, 1)]))
            cow_notes = vary_melody(cow_notes, scale)
            print("melody: VARIED (original, in-scale)", flush=True)
        else:
            print("melody: pack MIDI (verbatim)", flush=True)
        overlay(buf, render_midi_with_sample(cow_notes, cow_s, cb, len(buf)), 0, 0.7)
        overlay(buf, render_midi_with_sample(b08_notes, b08_s, bb, len(buf)), 0, 0.95)

    buf = np.tanh(buf * 1.1)
    buf = buf / (np.max(np.abs(buf)) + 1e-9) * 0.97
    sf.write(out_wav, buf, SR)
    mp3 = os.path.splitext(out_wav)[0] + ".mp3"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", out_wav, "-b:a", "192k", mp3], capture_output=True)
    print(f"wrote {mp3}  ({LEN:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
