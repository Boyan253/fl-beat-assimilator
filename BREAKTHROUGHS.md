# Breakthroughs

The things that actually took time to learn while building the local AI-music
pipeline (ACE-Step on a 7900 XTX / ROCm, plus turning generated songs into
playable instruments). Written down so they don't have to be rediscovered.

## 1. Model choice and step count decide the quality
Use the full model `acestep-v15-xl-sft` at **32 inference steps**, not the turbo
model. Turbo's 8 steps produce thin, weak-sounding output. Turbo is only worth it
for durations the full model can't fit in VRAM.

## 2. bfloat16 is the stability fix (biggest single win)
In float32 the model occupies ~21.5 GB, which barely fits a 24 GB card. Any other
GPU consumer — even a browser playing video — pushes it over and generation dies
with a VRAM pre-flight failure. Setting `ACESTEP_ROCM_DTYPE=bfloat16` roughly halves
it, runs faster, and makes batches reliable. Symptom before the fix: the first take
succeeds, later takes fail, or all fail depending on what else touched the GPU.

## 3. The raw model output is too clean for phonk
A post-processing "grit" pass (saturation, bitcrush, tape wobble, distorted low end)
is what makes it read as the genre. Intensity ~1.1 is maximum dirt; ~0.7 keeps the
bass tighter and clearer. See `phonkify.py`.

## 4. Prompt wording changes the arrangement, not just the timbre
Words like *atmospheric, spacious, introspective, room for vocals* make the model
drop the drums and hand back an ambient intro section. If a driving beat is wanted,
describe the drums literally: "punchy four-on-the-floor kick on every beat, tight
drums". Conversely, for dark phonk, avoid bright/happy descriptors and the
four-on-the-floor club kick.

## 5. Pitch detection: pyin, not crepe
To map a melody to real notes, high-pass the stem (~150 Hz) and use `librosa.pyin`.
Neural pitch trackers failed badly on metallic/inharmonic material (a cowbell lead
returned a single pitch for the whole track). pyin on a high-passed signal produced
a clean, musical note map.

## 6. You cannot have exact audio and a musical MIDI instrument at once
Established by measurement, not opinion:

| Approach | Timbre match | Playable? |
|---|---|---|
| The rendered audio itself | ~99.8% | no, it's a recording |
| Real slices mapped to detected pitches | ~99.8% | yes, but the piano roll is a staircase |
| One sample per pitch (SF2) | ~52% | yes, full keyboard |
| Velocity-keyed exact slices | ~87% | drops notes in samplers |
| RAVE neural model via Neutone | ~90% | yes, any note, no dead keys |

Exact playback means playing the recording. A normal-looking MIDI performance means
re-performing it, which costs fidelity. Pick one per use case; RAVE is the best
compromise when a playable native instrument matters more than being bit-exact.

## 7. Extending a track: arrange, don't regenerate
Regenerating at a longer duration gives a *different song*. To lengthen an existing
one: separate stems, work on the bar grid (tempo -> seconds per bar), and rebuild an
arrangement — use the drums-removed stem for breakdowns so the vocal has space.
Splice on the downbeat with a short (~15 ms) equal-power crossfade; the kick masks
the seam. Verify afterwards by measuring per-section RMS — a "breakdown" that is
only ~1 dB below the verses will not be heard as a breakdown.

## 8. Mix/master chain that worked
High-pass ~30 Hz to remove subsonic energy, a gentle low-shelf cut to stop the 808
dominating, then soft-knee limiting toward roughly -8.5 dB RMS with a -1 dB
true-peak ceiling (check true peak on a 4x oversampled signal, not the raw samples).
Always ship a separate vocal-ready version with ~6 dB of headroom and no limiting —
a vocalist cannot sit a take on top of a fully mastered instrumental.

## Layout
- `ace_runners/` — shell entry points for generation / batching / rendering
- `music_tools/` — analysis, arrangement, mastering and QC scripts
- `PHONK_RECIPE.md` — the dialed-in genre recipe
- `setup_acestep_rocm.md` — standing the engine up on ROCm / WSL
- `MEASURE.md` — stems to MIDI / FL pack tooling
