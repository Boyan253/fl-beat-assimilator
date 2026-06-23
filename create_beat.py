"""CREATE engine — generate a 100% ORIGINAL phonk/trap beat as editable MIDI patterns
(melody + bass + drums) in a chosen key & tempo, via music-theory generation (no copying),
so it is legally sellable. Load each MIDI onto any instrument (your assimilated playable
sounds / an 808 / a drum kit) and arrange into a beat.
  python create_beat.py <name> [key=A] [bpm=145] [bars=8] [seed=0]
"""
import os, sys, random
import pretty_midi

NAME = sys.argv[1]
KEY = sys.argv[2] if len(sys.argv) > 2 else "A"
BPM = float(sys.argv[3]) if len(sys.argv) > 3 else 145.0
BARS = int(sys.argv[4]) if len(sys.argv) > 4 else 8
SEED = int(sys.argv[5]) if len(sys.argv) > 5 else 0
random.seed(SEED)
OUT = os.path.join("D:/flmidi/out/created", NAME)
os.makedirs(OUT, exist_ok=True)

N2M = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}
SCALE = [0, 2, 3, 5, 7, 8, 10]                 # natural minor (dark / phonk)
PROG = [0, 5, 2, 6]                            # i - VI - III - VII (looped each 4 bars)
ROOT_MEL = 60 + N2M.get(KEY, 9)               # lead register
ROOT_BASS = ROOT_MEL - 36                      # 808 register (3 octaves down)

spb = 60.0 / BPM                               # seconds per beat
step = spb / 4.0                               # 16th-note grid
barlen = spb * 4.0
KICK, SNARE, CHH, OHH = 36, 38, 42, 46         # General-MIDI drum map


def sdeg(d):                                   # scale degree -> semitones (octave-wrapping)
    return SCALE[d % 7] + 12 * (d // 7)


PENT = [0, 3, 5, 7, 10, 12]                    # minor pentatonic (semitones) — the phonk sound
# one catchy 2-bar cowbell riff (step, len, semitone-from-root) — a real HOOK that repeats
RIFF = [(0, 2, 0), (2, 2, 0), (4, 2, 12), (6, 2, 10), (8, 2, 7), (10, 2, 7), (12, 2, 3), (14, 2, 0),
        (16, 2, 0), (18, 2, 3), (20, 2, 5), (22, 2, 7), (24, 2, 10), (26, 2, 12), (28, 4, 7)]


def melody():
    out = []
    for blk in range(max(1, BARS // 2)):           # repeat the riff every 2 bars, transposed by the prog
        off = sdeg(PROG[blk % 4])
        t0 = blk * 2 * barlen
        for s, ln, semi in RIFF:
            p = max(48, min(88, ROOT_MEL + off + semi))
            st = t0 + s * step
            out.append((st, st + ln * step * 0.9, p, 112 if semi == 0 else 95))   # accent the root
    return out


def bass():
    out = []
    for blk in range(max(1, BARS // 2)):
        p = ROOT_BASS + sdeg(PROG[blk % 4])
        t0 = blk * 2 * barlen
        for bar in range(2):
            tb = t0 + bar * barlen
            for s, ln in [(0, 6), (6, 2), (8, 4), (12, 3)]:   # 808 root holds + accents, follows the riff
                st = tb + s * step
                out.append((st, st + ln * step * 0.98, p, 116))
    return out


def drums():
    out = []
    for bar in range(BARS):
        t0 = bar * barlen
        for s in [0, 3, 6, 10, 11]:                       # syncopated trap kick
            if random.random() < 0.85:
                out.append((t0 + s * step, t0 + s * step + step * 0.5, KICK, 120))
        out.append((t0 + 8 * step, t0 + 9 * step, SNARE, 112))   # half-time snare on beat 3
        for s in range(0, 16, 2):                         # 8th-note hats
            out.append((t0 + s * step, t0 + s * step + step * 0.5, CHH, random.randint(58, 92)))
        for s in [5, 13]:                                 # a couple of 16th ghosts
            out.append((t0 + s * step, t0 + s * step + step * 0.5, CHH, random.randint(48, 74)))
        if bar % 4 == 3:                                  # phonk triplet roll, last beat
            for k in range(6):
                st = t0 + 12 * step + k * (4 * step / 6)
                out.append((st, st + (4 * step / 6) * 0.8, CHH, 66 + k * 5))
    return out


def write(notes, fn, drum=False):
    pm = pretty_midi.PrettyMIDI(initial_tempo=BPM)
    inst = pretty_midi.Instrument(program=0, is_drum=drum)
    for st, en, p, v in notes:
        inst.notes.append(pretty_midi.Note(velocity=int(v), pitch=int(p), start=st, end=en))
    pm.instruments.append(inst); pm.write(fn)


write(melody(), os.path.join(OUT, NAME + "_melody.mid"))
write(bass(),   os.path.join(OUT, NAME + "_bass.mid"))
write(drums(),  os.path.join(OUT, NAME + "_drums.mid"), drum=True)
print("CREATED original beat: key=%s minor  bpm=%.0f  bars=%d  prog=i-VI-III-VII" % (KEY, BPM, BARS))
print("  ", OUT)
