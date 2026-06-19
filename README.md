# Assimilate — turn any song into editable FL Studio instruments

Take any song and rebuild its layers as **playable, editable instruments** that sound like
the original — a melody, a bassline, drums — each as its own pattern, ready to remix into a
new beat inside FL Studio.

It works on *any* source, including distorted electric guitar and sub-bass 808s that defeat
ordinary audio-to-MIDI, because every note plays the **actual recorded audio** of the
original rather than a re-synthesized guess.

## How it works

1. **Separate** the song into stems (drums / bass / other / vocals) with Demucs `htdemucs_ft`.
2. **Per-occurrence real-audio engine** — for each note it slices the real recording at the
   onset and plays *that exact chunk*. Fidelity depends only on onset *timing*, never on a
   transcription being "right", so it stays faithful on sources nothing else can transcribe.
3. **Two onset engines, auto-routed per stem:**
   - `build_perocc.py` — melodic transcriber onsets, for leads / synths / vocals.
   - `build_perocc_onset.py` — energy onsets, for bass / 808 / drums (sub-bass safe).
4. Output is an **SFZ instrument + a MIDI file** per layer. Two design details make them
   production-ready:
   - **velocity-layer selection** — each chunk is pinned to a unique velocity, so playback is
     deterministic and survives pause/restart (no round-robin drift).
   - **silent t=0 anchor** — every layer shares a clip origin, so late-entering parts (a bass
     that drops in at 5s) stay time-locked to the melody when placed at bar 1.
5. **Verify** — each rendered layer is correlated (chroma + RMS) against its source stem.

## Usage

```
python assimilate.py <song.mp3> <name>
```

Produces `out/<name>/<name>_melody.sfz/.mid`, `<name>_bass.sfz/.mid`, etc.

In FL Studio: load each `.sfz` in its own [Sforzando](https://www.plogue.com/products/sforzando.html)
channel (free SFZ player), drop the matching `.mid`, line every clip up at **bar 1**. Move,
retime, duplicate and rearrange the notes freely — just don't change note *velocity* (it
selects which audio chunk plays).

## Built on

Original engine code; standalone open-source dependencies do the heavy lifting:
[Demucs](https://github.com/facebookresearch/demucs) (stem separation),
[Basic Pitch](https://github.com/spotify/basic-pitch) (melodic onsets),
[librosa](https://librosa.org/) (energy onsets / analysis),
[sfizz](https://github.com/sfztools/sfizz) (headless SFZ rendering),
[pretty_midi](https://github.com/craffel/pretty-midi). Playback in FL uses Sforzando.
