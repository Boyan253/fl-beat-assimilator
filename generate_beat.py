"""A brand-new phonk chop arrangement for a Slicex channel. Slicex maps slices to
ascending keys from C5 (MIDI 60), so these notes re-trigger your slices in a fresh,
syncopated groove. Drop the .mid onto the Slicex channel's piano roll.
Keys kept in 60-67 (the first 8 slices, which almost any auto-slice will have)."""
import pretty_midi

BPM = 145.0
STEP = 60.0 / BPM / 2.0          # 1/8 note in seconds
pm = pretty_midi.PrettyMIDI(initial_tempo=BPM)
inst = pretty_midi.Instrument(program=0, name="abraham chop")

# (step within a 2-bar/16-step block, slice-key, length in steps)
motif = [
    (0, 60, 2), (2, 63, 1), (3, 60, 1), (5, 65, 1), (6, 63, 1),
    (8, 60, 1), (10, 63, 1), (11, 67, 2), (13, 65, 1), (14, 62, 1),
]
# 4 blocks (8 bars), with light variation so it's not a flat loop
variations = [0, 0, +2, -1]      # transpose the block (still within slice range when clamped)
for blk, tr in enumerate(variations):
    base = blk * 16
    for step, key, ln in motif:
        k = max(60, min(67, key + tr))
        s = (base + step) * STEP
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=k, start=s, end=s + ln * STEP * 0.92))
    # a little end-of-block fill (fast triplet-ish stabs)
    if blk in (1, 3):
        for j, k in enumerate([60, 63, 65]):
            s = (base + 15) * STEP + j * STEP / 3
            inst.notes.append(pretty_midi.Note(velocity=90, pitch=k, start=s, end=s + STEP / 3 * 0.9))

pm.instruments.append(inst)
out = r"D:\flmidi\out\new_beat.mid"
pm.write(out)
print("WROTE", out, "|", len(inst.notes), "notes,", "%.0f BPM" % BPM,
      "| span %.1fs" % (max(n.end for n in inst.notes)))
