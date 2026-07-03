"""make_rap_beat.py — UK/US rap-trap beat as multi-track MIDI.
Melody  = Anticipatory Music Transformer (AI), cleaned to a dark scale + 16th grid, looped.
Drums   = rule-based trap (US) or drill (UK) pattern.
808     = gliding bassline following the melody root (legato for FLEX glide).
For FL free VSTs:  Melody -> FLEX (dark bell/lead) | 808 -> FLEX Hard-808 (glide ON) | Drums -> FPC.

  python make_rap_beat.py <out.mid> [trap|drill] [bpm] [bars=16] [seed]
"""
import sys, random, warnings
warnings.filterwarnings("ignore")
import numpy as np, pretty_midi, torch
from transformers import AutoModelForCausalLM
from anticipation.sample import generate
from anticipation.convert import events_to_midi

OUT = sys.argv[1]
STYLE = sys.argv[2] if len(sys.argv) > 2 else "trap"
BPM = int(sys.argv[3]) if len(sys.argv) > 3 else (140 if STYLE == "trap" else 142)
BARS = int(sys.argv[4]) if len(sys.argv) > 4 else 16
SEED = int(sys.argv[5]) if len(sys.argv) > 5 else 7
random.seed(SEED)

SPB = 60.0 / BPM; BAR = 4 * SPB; STEP = BAR / 16
ROOT = random.choice([45, 47, 48, 50])                 # dark register
SCALE = [0, 2, 3, 5, 7, 8, 10] if STYLE == "trap" else [0, 1, 3, 5, 7, 8, 10]  # minor / phrygian(drill)
GK, GS, GH = 36, 38, 42

def tt(bar, step): return bar * BAR + step * STEP
def snap_scale(p):
    pc = (p - ROOT) % 12
    best = min(SCALE, key=lambda s: min(abs(pc - s), 12 - abs(pc - s)))
    return p + (best - pc) if abs(best - pc) <= 6 else p + (best - pc) + (-12 if best > pc else 12)

# ---------- 1) AI melody (AMT) -> monophonic, in-scale, quantized, looped phrase ----------
def ai_melody():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("  [AMT] generating melody...", flush=True)
    model = AutoModelForCausalLM.from_pretrained("stanford-crfm/music-medium-800k").to(dev).eval()
    ev = generate(model, start_time=0, end_time=14, top_p=0.96)
    tmp = OUT + ".amt.mid"; events_to_midi(ev).save(tmp)
    pm = pretty_midi.PrettyMIDI(tmp)
    raw = sorted([n for ins in pm.instruments if not ins.is_drum for n in ins.notes], key=lambda n: n.start)
    if not raw:
        return None
    # bin to 16th grid, keep the HIGHEST note per active step (monophonic top line)
    grid = {}
    for n in raw:
        step = int(round(n.start / STEP))
        if step not in grid or n.pitch > grid[step]:
            grid[step] = n.pitch
    if len(grid) < 6:
        return None
    steps = sorted(grid)
    # normalise into a 1.5-octave lead register above root, snap to scale
    lo = ROOT + 12
    seq = []
    for s in steps:
        p = grid[s]
        while p < lo: p += 12
        while p > lo + 19: p -= 12
        seq.append((s - steps[0], snap_scale(p)))
    # take the first 8 bars (128 steps) as the loop phrase
    return [(s, p) for (s, p) in seq if s < 128]

# ---------- 2) drums ----------
def drums(drum, bar, fill):
    if STYLE == "trap":
        kicks = random.choice([(0, 6, 10), (0, 3, 7, 10), (0, 7, 8)])
        snare = (8,)                                    # snare on beat 3 (trap)
    else:  # drill
        kicks = random.choice([(0, 5, 8), (0, 6, 11), (0, 4, 7, 10)])
        snare = (8, 14) if random.random() < 0.4 else (8,)   # drill snare incl. the 'a' of 4
    for st in kicks:
        drum.notes.append(pretty_midi.Note(120, GK, tt(bar, st), tt(bar, st) + STEP))
    for st in snare:
        drum.notes.append(pretty_midi.Note(116, GS, tt(bar, st), tt(bar, st) + STEP))
    triplet = STYLE == "trap"
    for st in range(16):
        if random.random() < 0.9:
            drum.notes.append(pretty_midi.Note(random.randint(58, 92), GH, tt(bar, st), tt(bar, st) + STEP * 0.5))
        if (fill or triplet) and st >= 12 and random.random() < 0.5:   # rolls
            for k in range(3):
                o = tt(bar, st) + STEP * k / 3
                drum.notes.append(pretty_midi.Note(84, GH, o, o + STEP / 3))

# ---------- build ----------
mel_notes = ai_melody()
if not mel_notes:
    print("  [AMT] weak output, using fallback motif", flush=True)
    motif = [0, -1, 3, -1, 5, -1, 3, 2] * 2
    mel_notes = [(i, ROOT + 12 + SCALE[d % 7]) for i, d in enumerate(motif) if d >= 0]

mel = pretty_midi.Instrument(program=11, name="Melody")
bass = pretty_midi.Instrument(program=38, name="Bass(808)")
drum = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")

PHRASE = 128  # 8 bars of 16ths
for rep in range(max(1, BARS // 8)):
    base = rep * 8
    for (s, p) in mel_notes:
        st = base * 16 + s
        bar, step = divmod(st, 16)
        if bar >= BARS: break
        mel.notes.append(pretty_midi.Note(random.randint(98, 118), p,
                                           tt(bar, step), tt(bar, step) + STEP * random.choice([1, 2, 2])))
# 808 follows melody root per bar (gliding legato), drums after intro
roots_by_bar = {}
for n in mel.notes:
    b = int(n.start / BAR)
    roots_by_bar.setdefault(b, n.pitch)
for bar in range(BARS):
    if bar in roots_by_bar:
        p = roots_by_bar[bar] - 24
        bass.notes.append(pretty_midi.Note(122, p, tt(bar, 0), tt(bar, 0) + BAR * 0.5))
        bass.notes.append(pretty_midi.Note(122, snap_scale(p + 5), tt(bar, 10), tt(bar, 10) + BAR * 0.35))
    if bar >= 2:
        drums(drum, bar, fill=(bar % 4 == 3))

pm = pretty_midi.PrettyMIDI(initial_tempo=float(BPM))
pm.instruments += [mel, bass, drum]
pm.write(OUT)
print(f"{STYLE} beat: key={pretty_midi.note_number_to_name(ROOT)} | {BPM}bpm | {BARS} bars | seed={SEED}", flush=True)
print(f"  notes: melody={len(mel.notes)} 808={len(bass.notes)} drums={len(drum.notes)} -> {OUT}", flush=True)
