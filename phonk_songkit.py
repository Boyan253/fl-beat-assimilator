"""phonk_songkit.py — generate a SONG KIT of MIDI patterns for building a full phonk
song in FL Studio, plus an audio preview.

Output (one key + tempo so everything fits together) -> /mnt/d/flbeat/data/songkit_<n>/ :
  melody_01..04.mid   - cowbell-melody variations (original, in-scale)
  bass_01..02.mid     - 808 basslines (follow the melody roots, glide-friendly)
  drums.mid           - drum pattern (GM kick=36/snare=38/hat=42), from a real loop
  drumloop_*.wav      - the real drum-loop WAVs (drop straight onto a track)
  preview.mp3         - a quick audio preview (melody_01 + bass_01 + drums)
  README.txt          - what to load where in FL

  python phonk_songkit.py [kit_number] [--key N]

Load the .mid files in FL, put a cowbell on melody / an 808 (with glide) on bass /
your drum kit on drums, arrange the patterns into intro/verse/drop = full song.
"""
import os, sys, glob, random, shutil, warnings, subprocess
warnings.filterwarnings("ignore")
import numpy as np, librosa, soundfile as sf, pretty_midi
from make_phonk_beat import PACK, SR, OUT_DIR, load, oneshot, base_pitch, vary_melody

TEMPO = 175.0
BEAT  = 60.0 / TEMPO
MELODY_MODEL = "/mnt/d/flbeat/data/models/phonk_amt_melody"
MELODY_CORPUS = "/mnt/d/flbeat/data/melody_corpus"

_MODEL = None
def model_melody(seed_notes):
    """Compose a NEW melody with the fine-tuned melody model: seed it with a clean
    corpus melody, generate a continuation, strip the prompt -> original phonk line."""
    global _MODEL
    import torch
    from transformers import AutoModelForCausalLM
    from anticipation.sample import generate as amt_generate
    from anticipation.convert import midi_to_events, events_to_midi
    from anticipation import ops
    if _MODEL is None:
        _MODEL = AutoModelForCausalLM.from_pretrained(MELODY_MODEL, attn_implementation="eager").eval().cuda()
    seed_mid = random.choice(glob.glob(os.path.join(MELODY_CORPUS, "*.mid")))
    inp = ops.clip(midi_to_events(seed_mid), 0, 6, clip_duration=False, seconds=True)
    full = amt_generate(_MODEL, 6, 22, inputs=inp, top_p=0.94)
    gen = ops.clip(full, 6, 22, clip_duration=False, seconds=True)
    if not gen:
        return seed_notes
    gen = ops.translate(gen, -6, seconds=True)
    tmp = "/tmp/_mel.mid"; events_to_midi(gen).save(tmp)
    return [n for i in pretty_midi.PrettyMIDI(tmp).instruments for n in i.notes]


def save_midi(notes, path, program=0, is_drum=False, name=""):
    pm = pretty_midi.PrettyMIDI(initial_tempo=TEMPO)
    inst = pretty_midi.Instrument(program=program, is_drum=is_drum, name=name)
    inst.notes = sorted(notes, key=lambda n: n.start)
    pm.instruments.append(inst)
    pm.write(path)


def bassline_from_melody(melody, octave_root=39):
    """One sustained 808 per bar that follows the lowest melody note in that bar,
    held until the next change (gives the glide-friendly long 808, not pop-pop)."""
    if not melody:
        return []
    end = max(n.end for n in melody)
    bar = 4 * BEAT
    notes = []
    t = 0.0
    while t < end:
        seg = [n for n in melody if t <= n.start < t + bar]
        if seg:
            root = min(n.pitch for n in seg) % 12 + octave_root  # pull to 808 register
            notes.append(pretty_midi.Note(velocity=110, pitch=root, start=t, end=t + bar))
        t += bar
    return notes


def transcribe_drumloop(wav):
    """Onset + spectral-centroid -> GM drum MIDI (kick 36 / snare 38 / hat 42)."""
    y, _ = librosa.load(wav, sr=SR, mono=True)
    onsets = librosa.onset.onset_detect(y=y, sr=SR, units="time", backtrack=True)
    notes = []
    for t in onsets:
        i0 = int(t * SR); w = y[i0:i0 + int(0.05 * SR)]
        if len(w) < 128:
            continue
        cent = float(librosa.feature.spectral_centroid(y=w, sr=SR).mean())
        pitch = 36 if cent < 800 else (38 if cent < 4000 else 42)
        notes.append(pretty_midi.Note(velocity=100, pitch=pitch, start=float(t), end=float(t + 0.12)))
    return notes


def transpose(notes, s):
    return [pretty_midi.Note(velocity=n.velocity, pitch=max(0, min(127, n.pitch + s)),
                             start=n.start, end=n.end) for n in notes]


def render_bass_smooth(notes, sample, base, length):
    """Resample-based pitch (no phase-vocoder lag), full sustain, attack/release fades."""
    buf = np.zeros(length, dtype=np.float32)
    for n in notes:
        semis = max(-24, min(12, int(round(n.pitch - base))))
        factor = 2 ** (semis / 12.0)
        idx = np.arange(0, len(sample), factor)
        s = np.interp(idx, np.arange(len(sample)), sample).astype(np.float32)
        dur = int((n.end - n.start) * SR)
        if dur > 0:
            if len(s) < dur:
                s = np.pad(s, (0, dur - len(s)))
            else:
                s = s[:dur]
        a = min(256, len(s) // 8); r = min(int(0.05 * SR), len(s) // 3)
        if a > 0: s[:a] *= np.linspace(0, 1, a)
        if r > 0: s[-r:] *= np.linspace(1, 0, r)
        at = int(n.start * SR); e = min(at + len(s), length)
        buf[at:e] += s[:e - at] * (0.5 + 0.5 * n.velocity / 127)
    return buf


def render_cowbell(notes, sample, base, length):
    buf = np.zeros(length, dtype=np.float32)
    cache = {}
    for n in notes:
        semis = max(-24, min(24, int(round(n.pitch - base))))
        if semis not in cache:
            cache[semis] = librosa.effects.pitch_shift(sample, sr=SR, n_steps=semis)
        s = cache[semis]
        at = int(n.start * SR); e = min(at + len(s), length)
        if e > at:
            buf[at:e] += s[:e - at] * (0.4 + 0.6 * n.velocity / 127)
    return buf


def main():
    kit_n = next((a for a in sys.argv[1:] if a.isdigit()), "1")
    key = 0
    if "--key" in sys.argv:
        key = int(sys.argv[sys.argv.index("--key") + 1])

    kit = os.path.join("/mnt/d/flbeat/data", f"songkit_{kit_n}")
    os.makedirs(kit, exist_ok=True)
    random.seed()

    cow_mid = [m for m in glob.glob(os.path.join(PACK, "Melodic Loops", "*.mid")) if "cowbell" in m.lower()]
    seed = [n for i in pretty_midi.PrettyMIDI(random.choice(cow_mid)).instruments for n in i.notes]
    scale = sorted(set(n.pitch for n in seed))
    scale = sorted(set(p + 12 * o for p in scale for o in (-1, 0, 1)))

    use_model = "--model" in sys.argv and os.path.exists(os.path.join(MELODY_MODEL, "config.json"))
    print("melody source:", "MODEL" if use_model else "markov-variation", flush=True)

    # 4 melody variations (transposed to chosen key)
    melodies = []
    for i in range(4):
        raw = model_melody(seed) if use_model else vary_melody(seed, scale)
        m = transpose(raw, key)
        melodies.append(m)
        save_midi(m, os.path.join(kit, f"melody_{i+1:02d}.mid"), program=30, name="cowbell")

    # 2 basslines following melody roots
    basses = []
    for i in range(2):
        b = bassline_from_melody(melodies[i])
        basses.append(b)
        save_midi(b, os.path.join(kit, f"bass_{i+1:02d}.mid"), program=38, name="808")

    # drums from a real full loop + copy a few loop WAVs
    full_loops = [f for f in glob.glob(os.path.join(PACK, "Drum Loops", "*.wav")) if "full" in f.lower()]
    drum_notes = transcribe_drumloop(full_loops[0])
    save_midi(drum_notes, os.path.join(kit, "drums.mid"), is_drum=True, name="drums")
    for f in full_loops[:3] + [g for g in glob.glob(os.path.join(PACK, "Drum Loops", "*.wav")) if "hat" in g.lower()][:1]:
        shutil.copy(f, os.path.join(kit, "drumloop_" + os.path.basename(f).split(" - ")[-1]))

    # combined multitrack MIDI (for quick drag-in)
    pm = pretty_midi.PrettyMIDI(initial_tempo=TEMPO)
    for notes, prog, drum, nm in [(melodies[0], 30, False, "cowbell"), (basses[0], 38, False, "808"), (drum_notes, 0, True, "drums")]:
        inst = pretty_midi.Instrument(program=prog, is_drum=drum, name=nm); inst.notes = notes
        pm.instruments.append(inst)
    pm.write(os.path.join(kit, "song_all_tracks.mid"))

    # audio preview (smooth 808 + cowbell + real drum loop)
    LEN = max(max(n.end for n in melodies[0]), 16.0)
    L = int((LEN + 0.5) * SR)
    cow_s = oneshot(sorted(glob.glob(os.path.join(PACK, "Cowbells", "*.wav")))[0], 0.4)
    b08_s = load(sorted(glob.glob(os.path.join(PACK, "808s", "*.wav")))[0])
    cb = base_pitch(cow_s, 63); bb = base_pitch(b08_s[:int(0.5*SR)], 39)
    buf = np.zeros(L, dtype=np.float32)
    dl = load(full_loops[0]); buf[:min(len(dl), L)] += dl[:min(len(dl), L)] * 1.0
    buf += render_cowbell(melodies[0], cow_s, cb, L) * 0.7
    buf += render_bass_smooth(basses[0], b08_s, bb, L) * 0.95
    buf = np.tanh(buf * 1.05); buf = buf / (np.max(np.abs(buf)) + 1e-9) * 0.97
    pv = os.path.join(kit, "preview.wav"); sf.write(pv, buf, SR)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", pv, "-b:a", "192k", pv[:-4] + ".mp3"], capture_output=True)

    with open(os.path.join(kit, "README.txt"), "w") as f:
        f.write(f"PHONK SONG KIT {kit_n}  —  {TEMPO:.0f} BPM, key offset {key:+d} (base D# Phrygian)\n\n"
                "In FL Studio:\n"
                "  melody_01..04.mid -> a COWBELL instrument (try each, pick favourites)\n"
                "  bass_01..02.mid   -> an 808 with GLIDE/portamento (this kills the 'pop pop')\n"
                "  drums.mid         -> your drum kit (36=kick 38=snare 42=hat)\n"
                "  drumloop_*.wav    -> or just drop these real loops on an audio track\n"
                "  song_all_tracks.mid -> all three at once for a quick start\n\n"
                "Arrange: intro (drums+1 melody) -> drop (full) -> variation (swap melody) -> outro.\n")

    print(f"KIT -> {kit}", flush=True)
    for f in sorted(os.listdir(kit)):
        print("  ", f, flush=True)


if __name__ == "__main__":
    main()
