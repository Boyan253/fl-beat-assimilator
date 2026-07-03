"""make_phonk_midi.py — generate a structured multi-track PHONK MIDI from scratch (no audio, no
transcription). Clean, editable, human-like notes. Designed to be played by FL's free VSTs:
  Melody -> FLEX (Phonk pack bell/lead) | Bass -> FLEX (Hard 808s, glide on) | Drums -> FPC.

  python make_phonk_midi.py <out.mid> [bpm=140] [bars=32] [seed=int]
Outputs a 3-track MIDI: 'Melody', 'Bass(808)', 'Drums'(GM kit). Dark natural-minor, phonk patterns.
"""
import sys, random
import pretty_midi

OUT = sys.argv[1]
BPM = int(sys.argv[2]) if len(sys.argv) > 2 else 140
BARS = int(sys.argv[3]) if len(sys.argv) > 3 else 32
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 7
random.seed(SEED)

SPB = 60.0 / BPM            # sec per beat
BAR = 4 * SPB               # 4/4
STEP = BAR / 16             # 16th-note grid
ROOT = random.choice([45, 47, 48, 50, 52])   # A2..E3 — dark register
MINOR = [0, 2, 3, 5, 7, 8, 10]               # natural minor scale degrees
GM_KICK, GM_SNARE, GM_HAT, GM_OHAT = 36, 38, 42, 46

mel = pretty_midi.Instrument(program=11, name="Melody")    # FLEX phonk bell/lead in FL
bass = pretty_midi.Instrument(program=38, name="Bass(808)")  # FLEX Hard-808 in FL
drum = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")


def t(bar, step):
    return bar * BAR + step * STEP


def scale_note(degree, octave=0):
    return ROOT + 12 * octave + MINOR[degree % 7] + 12 * (degree // 7)


# --- melodic motif: a short dark minor riff, repeated with variation (musical, not random) ---
def make_motif():
    # 8 sixteenth-steps; -1 = rest. minor pentatonic-ish movement, lands on root/5th
    shapes = [
        [0, -1, 2, -1, 0, -1, -1, 4],
        [0, 0, -1, 3, 2, -1, 0, -1],
        [4, -1, 3, -1, 2, -1, 0, -1],
        [0, -1, -1, 5, 4, -1, 2, -1],
    ]
    return random.choice(shapes)


motif = make_motif()


def melody_bar(bar, octave=1, density=1.0):
    for s in range(16):
        deg = motif[s % 8]
        if deg < 0 or random.random() > density:
            continue
        p = scale_note(deg, octave)
        dur = STEP * random.choice([1, 1, 2])
        mel.notes.append(pretty_midi.Note(velocity=random.randint(96, 118),
                                           pitch=p, start=t(bar, s), end=t(bar, s) + dur))


# --- 808 bass: follows the bar's root, sparse + gliding feel (long notes) ---
def bass_bar(bar, root_deg=0):
    pattern = [(0, 6), (6, 4), (11, 5)] if random.random() < 0.5 else [(0, 8), (10, 6)]
    for st, length in pattern:
        p = scale_note(root_deg, -1)
        bass.notes.append(pretty_midi.Note(velocity=120, pitch=p,
                                            start=t(bar, st), end=t(bar, st) + STEP * length))


# --- drums: phonk groove (kick + backbeat snare + fast hats with rolls) ---
def drum_bar(bar, fill=False):
    # kick
    for st in (0, 6, 10) if random.random() < 0.5 else (0, 3, 8, 11):
        drum.notes.append(pretty_midi.Note(120, GM_KICK, t(bar, st), t(bar, st) + STEP))
    # snare on beats 2 and 4 (steps 4 and 12) — the phonk backbeat
    for st in (4, 12):
        drum.notes.append(pretty_midi.Note(118, GM_SNARE, t(bar, st), t(bar, st) + STEP))
    # hats: 16ths, occasional double-time rolls
    for st in range(16):
        if random.random() < 0.85:
            v = random.randint(60, 95)
            drum.notes.append(pretty_midi.Note(v, GM_HAT, t(bar, st), t(bar, st) + STEP * 0.5))
        if fill and st >= 12 and random.random() < 0.6:   # end-of-phrase roll
            drum.notes.append(pretty_midi.Note(90, GM_HAT, t(bar, st) + STEP * 0.5, t(bar, st) + STEP))


# --- arrangement: intro (melody) -> build -> full -> ... with fills every 4 bars ---
prog = [0, 0, 5, 3]   # i - i - VI - iv (dark minor)
for bar in range(BARS):
    sec = bar // 8
    fill = (bar % 4 == 3)
    root_deg = prog[(bar // 2) % len(prog)]
    if sec == 0:            # intro: melody + light 808, no drums
        melody_bar(bar, octave=1, density=0.9)
        if bar >= 2:
            bass_bar(bar, root_deg)
    else:                  # full sections
        melody_bar(bar, octave=1, density=0.85 if sec == 1 else 1.0)
        bass_bar(bar, root_deg)
        drum_bar(bar, fill=fill)

pm = pretty_midi.PrettyMIDI(initial_tempo=float(BPM))
pm.instruments += [mel, bass, drum]
pm.write(OUT)
key = pretty_midi.note_number_to_name(ROOT)
print(f"phonk MIDI: key={key} minor, {BPM}bpm, {BARS} bars, seed={SEED} -> {OUT}", flush=True)
print(f"  notes: melody={len(mel.notes)} bass={len(bass.notes)} drums={len(drum.notes)}", flush=True)
