"""Per-note slicing for EVERY layer: cut each stem into real-audio slices and map
them across the keyboard as an SFZ multisample. Playing the MIDI then triggers the
ORIGINAL recording at each pitch -> sounds like excalibur AND notes stay editable.

Output per instrument (melody/bass): slices/<name>/p###.wav + <name>.sfz + <name>_sliced.mid
Drums: instruments/drums.sfz mapping the kick/snare/hat one-shots to GM keys.
Load each .sfz into a DirectWave/sfizz channel; the MIDI drives it.
"""
import os, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np, librosa, soundfile as sf, pretty_midi

OUT = r"D:\flmidi\out\excalibur"
STEMS = os.path.join(OUT, "demucs", "htdemucs", "excalibur")
SLICEDIR = os.path.join(OUT, "slices")
INSTR = os.path.join(OUT, "instruments")
SR = 44100


def transcribe(y22, sr, fmin, fmax):
    """Monophonic dominant-line note events: list of (start, end, midi)."""
    f0, vf, vp = librosa.pyin(y22, fmin=fmin, fmax=fmax, sr=sr, frame_length=4096)
    t = librosa.times_like(f0, sr=sr)
    notes, cur, start, probs = [], None, 0.0, []
    for i in range(len(f0)):
        m = int(round(librosa.hz_to_midi(f0[i]))) if (vf[i] and not np.isnan(f0[i])) else None
        if m != cur:
            if cur is not None and (t[i] - start) > 0.08:
                notes.append((start, t[i], cur, float(np.mean(probs)) if probs else 0.0))
            cur, start, probs = m, t[i], []
        if vf[i] and not np.isnan(vp[i]):
            probs.append(float(vp[i]))
    return notes


def build_multisample(stem_path, name, fmin, fmax, slice_len=0.5, release=0.15):
    y22, _ = librosa.load(stem_path, sr=22050, mono=True)
    yhi, _ = librosa.load(stem_path, sr=SR, mono=True)
    notes = transcribe(y22, 22050, fmin, fmax)

    # per pitch, keep the CLEAREST single occurrence (NOT the longest — long slices
    # replay seconds of the song and stack into a "doubled" wash).
    best = {}
    for (s, e, p, prob) in notes:
        score = (prob + 0.1) * min(e - s + 0.05, 0.30)
        if p not in best or score > best[p][0]:
            best[p] = (score, s)
    insdir = os.path.join(SLICEDIR, name); os.makedirs(insdir, exist_ok=True)
    for old in glob.glob(os.path.join(insdir, "p*.wav")):   # clear stale slices from prior runs
        os.remove(old)
    pitches = sorted(best)
    regions = []
    for idx, p in enumerate(pitches):
        score, s = best[p]
        seg = yhi[int(s * SR):int((s + slice_len) * SR)]   # short, controlled one-note slice
        if len(seg) < 256:
            continue
        seg = seg / (np.max(np.abs(seg)) or 1) * 0.97
        fo = int(release * SR)                             # release tail: no hard cut, no doubling
        if len(seg) > fo:
            seg[-fo:] *= np.linspace(1, 0, fo)
        wav = "p%03d.wav" % p
        sf.write(os.path.join(insdir, wav), seg.astype("float32"), SR)
        # range-map so the whole keyboard is covered (gaps filled by nearest, mild shift)
        lo = 0 if idx == 0 else (pitches[idx - 1] + p) // 2 + 1
        hi = 127 if idx == len(pitches) - 1 else (p + pitches[idx + 1]) // 2
        regions.append("<region> sample=%s pitch_keycenter=%d lokey=%d hikey=%d loop_mode=no_loop"
                        % (wav, p, lo, hi))
    sfz = os.path.join(insdir, name + ".sfz")
    with open(sfz, "w") as f:
        f.write("// %s — each region is a REAL recorded slice of excalibur at that pitch\n" % name)
        f.write("\n".join(regions))
    # MIDI to drive it (the editable notes)
    pm = pretty_midi.PrettyMIDI(); inst = pretty_midi.Instrument(program=0)
    for (s, e, p, prob) in notes:
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=p, start=s, end=e))
    pm.instruments.append(inst); pm.write(os.path.join(OUT, name + "_sliced.mid"))
    print("  %s: %d notes, %d sampled pitches -> %s" % (name, len(notes), len(regions), sfz))
    return len(regions)


def extract_drums():
    # re-extract kick/snare/hat with FULL decay tails (the old ones were clipped short
    # -> sounded abrupt/chopped). Longer windows + gentle 30ms release.
    y, sr = librosa.load(os.path.join(STEMS, "drums.wav"), sr=SR, mono=True)
    S = np.abs(librosa.stft(y, n_fft=2048)); fr = librosa.fft_frequencies(sr=sr, n_fft=2048)
    for lo, hi, wav, dur in [(20, 120, "kick.wav", 0.50), (150, 500, "snare.wav", 0.28),
                             (6000, 14000, "hat.wav", 0.14)]:
        band = (fr >= lo) & (fr < hi)
        oenv = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S[band, :] + 1e-9), sr=sr)
        on = librosa.onset.onset_detect(onset_envelope=oenv, sr=sr, units="samples", backtrack=True)
        if len(on) == 0:
            continue
        w = int(0.1 * sr)
        best = int(on[int(np.argmax([np.sum(y[int(s):int(s) + w] ** 2) for s in on]))])
        seg = y[best:best + int(dur * sr)]
        seg = seg / (np.max(np.abs(seg)) or 1) * 0.97
        fo = int(0.03 * sr)
        if len(seg) > fo:
            seg[-fo:] *= np.linspace(1, 0, fo)
        sf.write(os.path.join(INSTR, wav), seg.astype("float32"), sr)
        print("  re-extracted", wav, "%.2fs" % dur)


def regenerate_drums_midi():
    # rebuild the drum PATTERN with stricter onset detection -> fewer false hits,
    # less chaotic groove. GM pitches: kick 36, snare 38, hat 42.
    y, sr = librosa.load(os.path.join(STEMS, "drums.wav"), sr=22050, mono=True)
    S = np.abs(librosa.stft(y, n_fft=2048)); fr = librosa.fft_frequencies(sr=sr, n_fft=2048)
    pm = pretty_midi.PrettyMIDI(); drum = pretty_midi.Instrument(program=0)
    for lo, hi, pitch, gap, delta in [(20, 120, 36, 0.11, 0.32), (150, 500, 38, 0.13, 0.34),
                                      (6000, 14000, 42, 0.06, 0.26)]:
        band = (fr >= lo) & (fr < hi)
        oenv = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S[band, :] + 1e-9), sr=sr)
        if oenv.max() > 0:
            oenv = oenv / oenv.max()
        wait = max(1, int(librosa.time_to_frames(gap, sr=sr)))
        on = librosa.util.peak_pick(oenv, pre_max=4, post_max=4, pre_avg=8, post_avg=8,
                                    delta=delta, wait=wait)
        for t in librosa.frames_to_time(on, sr=sr):
            drum.notes.append(pretty_midi.Note(velocity=110, pitch=pitch, start=float(t), end=float(t) + 0.1))
    pm.instruments.append(drum); pm.write(os.path.join(OUT, "drums.mid"))
    c = {}
    for n in drum.notes:
        c[n.pitch] = c.get(n.pitch, 0) + 1
    print("  drums.mid regenerated (stricter):", c)


def drums_sfz():
    # map the already-extracted real one-shots to GM keys
    path = os.path.join(INSTR, "drums.sfz")
    rows = [(36, "kick.wav"), (38, "snare.wav"), (42, "hat.wav")]
    with open(path, "w") as f:
        f.write("// excalibur drums — real sliced hits on GM keys (kick36/snare38/hat42)\n")
        for key, wav in rows:
            if os.path.exists(os.path.join(INSTR, wav)):
                f.write("<region> sample=%s pitch_keycenter=%d lokey=%d hikey=%d loop_mode=no_loop\n"
                        % (wav, key, key, key))
    print("  drums.sfz ->", path)


if __name__ == "__main__":
    print("Building per-note multisamples from real excalibur stems...")
    # per-instrument: long guitar (continuous = "really close"), short/low bass (no spurious highs)
    build_multisample(os.path.join(STEMS, "other.wav"), "melody", 110, 1200, slice_len=2.2, release=0.6)
    build_multisample(os.path.join(STEMS, "bass.wav"), "bass", 30, 150, slice_len=0.45, release=0.12)
    extract_drums()
    drums_sfz()
    regenerate_drums_midi()
    print("DONE")
