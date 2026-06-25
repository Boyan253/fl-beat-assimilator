# MEASURE — benchmarking audio→MIDI note accuracy + choosing the best transcriber

## Why this exists
The per-occurrence engine reproduces a song's layers as faithful **audio** (~0.96 chroma vs
the stem) — that part is solved. What is **not** 1:1 is the **piano-roll NOTES**, which come
from Basic Pitch. This doc defines the objective way to measure how close a transcriber's
notes are to the real ones, and to pick the best model so we can swap it into the melody path.

**Key distinction:** we are NOT measuring audio fidelity (already high). We are measuring
**note-level transcription accuracy** (onset times + pitches), which is the actual gap.

## The hard part: ground truth
You can't score "closeness to the real notes" without a reference. Four options, best first:

1. **Synthetic ground truth (PRIMARY, fully objective).** Take known MIDI melodies → render
   to audio through realistic instruments → transcribe → compare the transcription to the
   *known* MIDI with `mir_eval`. Ground truth is exact, no ambiguity. This ranks transcribers
   cleanly. Use varied test MIDIs (mono leads + polyphonic, different tempos/registers) and
   render through ≥2 timbres (e.g. a cowbell and a soft synth) so results generalise.
2. **Pseudo-GT on real stems (VALIDATION).** On a clean **monophonic** lead stem, a strong f0
   tracker (CREPE) → quantised notes is a reliable reference; benchmark the *other* methods
   against it. (Don't use CREPE as both reference and candidate.)
3. **Hand-labelled bars (ANCHOR).** Manually transcribe 4–8 bars of one real song = small but
   true GT to sanity-check the synthetic ranking holds on real material.
4. **Official MIDI** of a song, if it exists — gold standard, but rare for phonk.

## Metrics (use `mir_eval.transcription`)
Report all of these, per stem and averaged:
- **Onset F1** (onset tolerance **±50 ms**) — timing accuracy.
- **Onset + Pitch F1** — the PRIMARY number ("are the right notes at the right times").
- **Onset + Pitch + Offset F1** — strict (includes note duration).
- **Chroma cosine similarity** — coarse pitch-class agreement.
- **Note-count ratio** (transcribed / reference) — over/under-segmentation.
- **Framewise pitch accuracy** (`mir_eval.melody`) — for monophonic leads.
Always finish with an **ear check**: render each transcription through a playable instrument
and confirm the metric matches what you hear.

## Models to test (ranked, with best fit)
| Model | Type | Best for | Notes |
|---|---|---|---|
| **CREPE** | deep monophonic f0 | **monophonic leads (phonk cowbell/synth)** | most accurate on clean mono; segment f0→notes. Top candidate. GPU = fast. |
| **MT3** (Google) | transformer, multi-instrument | **polyphonic / multi-layer** | SOTA multi-instrument transcription; heavy (T5, JAX/TF) — get it on ROCm. |
| **Onsets & Frames / ByteDance hi-res** | piano-specialised | **piano parts** | SOTA for piano only. |
| **Basic Pitch** (Spotify) | light polyphonic | baseline / fast fallback | current engine; decent, not SOTA. |
| **pYIN** (librosa) | classic monophonic | mono fallback | lighter than CREPE, slightly less accurate. |
| **Omnizart (drum) / onset+centroid** | drum | drum transcription | for kick/snare/hat MIDI. |

## Best-model recommendation (the hypothesis to confirm)
- **Monophonic lead (the phonk case) → CREPE.** Expected to be near-1:1 on clean mono stems.
- **Polyphonic / chordy → MT3.** Best available, still imperfect (the unsolved wall).
- **Piano → Onsets & Frames / ByteDance.**
- **Drums → onset + spectral-centroid classification (or Omnizart drum).**
- **Fallback / speed → Basic Pitch.**
Decision rule: per stem-type, pick the model with the highest **Onset+Pitch F1** on the
synthetic benchmark, then confirm on real stems + by ear.

## `bench_transcribe.py` — what to build (on the strong PC)
```
inputs : a folder of GT MIDIs + the timbres to render with (cowbell.sfz, a synth)
         + optionally real stems with a CREPE pseudo-GT
for each GT midi:
    render -> audio (sfizz/fluidsynth)
    for each model in [basic_pitch, crepe, pyin, mt3, ...]:
        audio -> midi
        score = mir_eval.transcription(ref=GT, est=transcribed)  # onset, onset+pitch, +offset
        + chroma cosine, note-count ratio
output : results.md  (table: model x metric, per stem-type + average)  + the winner
```

## Decide → integrate
Once a winner is confirmed (likely **CREPE for monophonic leads**):
- Swap the **melody NOTE source** in the assimilate path from Basic Pitch to the winner
  **for monophonic stems** (keep per-occurrence real-audio playback for the SOUND).
- Keep faithful per-occurrence (current) where the part is polyphonic and no transcriber
  clears the bar — and **report per-stem** so we route mono→best-transcriber, poly→faithful.
- Commit `bench_transcribe.py` + `results.md`; update this file with the chosen model.

## Honest ceiling
Even the best model is near-perfect only on **clean monophonic** material. Polyphonic/layered
melodies remain approximate — that's the open MP3→MIDI problem, not a tooling gap. The win
here is: get **monophonic leads to ~1:1 notes**, and know objectively where the line is.
