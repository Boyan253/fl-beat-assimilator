# results.md — transcriber benchmark (synthetic ground truth)

Rendered 4 GT MIDIs through 2 GM timbres via fluidsynth, transcribed with each model, scored with `mir_eval.transcription` (±50 ms, ±50 cents).
Headline = **Onset+Pitch F1** (right notes at right times).

## Monophonic (the phonk lead / 808 case)

| Model | Onset+Pitch F1 | Onset F1 | Onset+Pitch+Offset F1 | note-count ratio |
|---|---|---|---|---|
| CREPE | **81.6%** | 92.1% | 38.7% | 1.08 |
| Basic Pitch | **83.2%** | 83.7% | 9.3% | 1.40 |
| pYIN | **49.6%** | 82.9% | 21.9% | 0.91 |

## Polyphonic (the open-problem ceiling)

| Model | Onset+Pitch F1 | Onset F1 | note-count ratio |
|---|---|---|---|
| CREPE | 0.0% | 0.0% | 0.00 |
| Basic Pitch | 64.3% | 64.3% | 2.33 |
| pYIN | 0.0% | 0.0% | 0.00 |

## Winner (monophonic): **Basic Pitch** — 83.2% Onset+Pitch F1.

Decision: route **monophonic** stems (lead, 808) through **Basic Pitch**; keep Basic Pitch as fast fallback; polyphonic stays approximate (use faithful per-occurrence audio playback for the sound).
