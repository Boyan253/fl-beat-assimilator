# PHONK_RECIPE.md — the dialed-in config (keep this)

Working phonk generation recipe on the RX 7900 XTX (ROCm/WSL). Base = **ACE-Step 1.5**.
Generation env (aceenv) + grit pass + covers. This is the config that sounds right — don't lose it.

## One-command
```
bash /mnt/d/flbeat/data/run_full_track.sh "TITLE" "song prompt" "image prompt" <bpm> <dur>
# song (auto model by dur) -> grit pass -> cover -> MP4
```
Batch (beat the random-seed lottery — recommended):
```
bash /mnt/d/flbeat/data/batch_phonk.sh "NAME" "song prompt" <bpm> <dur> <count>   # rolls N takes, pick best
```

## Model by duration (auto in run_full_track / batch_phonk)
- **<= 60 s -> acestep-v15-xl-sft**, 32 steps, guidance 7 — full quality, **fuller & clearer bass**.
- **> 60 s -> acestep-v15-turbo**, 8 steps — needed for length, but **thinner/brighter** (grit thickens it).
- Boyan usually wants ~90 s -> turbo.

## Prompt presets
**DARK / evil phonk (default "real phonk" — not happy):**
`evil memphis phonk, slow and heavy menacing, demonic detuned cowbell melody, dissonant dark minor key,
deep distorted 808 bass, haunting cursed choir, horrorcore sinister creepy, slow halftime pounding drums,
grimy lo-fi, cold demonic dark`
- **BPM ~120–122** (slow = menacing).
- AVOID: "bright cowbell", four-on-the-floor / club kick, "happy/uplifting/major" -> these make it bouncy.

**Kordhell style = phonk HOUSE (aggressive/danceable):**
`aggressive drift phonk house, four on the floor club kick, hard gliding distorted 808 bass,
bright syncopated cowbell lead, busy hi-hat rolls, crisp snares, dark fast intense, kordhell style`
- **BPM ~150–152.** The four-on-the-floor kick is the signature.

**General avoid-words** (make ACE-Step sound rock/metal or too soft): brutal, gritty, metal, dreamy,
atmospheric, pads, nostalgic. ("distorted 808" alone is fine and wanted.)

## Grit pass — `phonkify.py` (THE key step; ACE-Step is too clean)
```
python phonkify.py <in.wav> <out.wav> <intensity>
```
- **1.1** = default, max phonk dirt (bass a bit muddier).
- **0.7** = clearer / tighter bass, less dirt.
- **~0.9** = the sweet spot (grit + still-readable bass).
Adds: distorted 808 low-end, tape saturation, bitcrush, cassette wobble, dark master, vinyl hiss.

## Covers
- **Cool (Kordhell / RJ Pasin look)** — central character + neon rim-glow + film grain (square 1:1):
  `python gen_cover_phonk.py "NAME::subject description"`  -> data/generated/covers/NAME_cover.png
- **Plain 16:9** (used inside the video): `gen_cover_image.py` (called by make_youtube.sh).

## ROCm/WSL env (required for every GPU run)
```
LD_LIBRARY_PATH=/opt/rocm/lib  HSA_ENABLE_SDMA=0  GPU_MAX_HW_QUEUES=1
PYTORCH_HIP_ALLOC_CONF=expandable_segments:True  ACESTEP_CHECKPOINTS_DIR=/opt/ace-models
MIOPEN_FIND_MODE=2  MIOPEN_DEBUG_CONV_IMMED_FALLBACK=1
```
Do NOT set HSA_OVERRIDE_GFX_VERSION (WSL libhsakmt abort). venvs: `/opt/aceenv` (gen), `/root/flbeat-venv` (analysis).

## Stems -> MIDI / FL pack (for editing tracks)
- Melody transcription: **YourMT3** (`run_mt3.py`) — best on polyphonic; needs `transformers==4.40.2`.
- Bass/808: **CREPE** (torchcrepe). Drums: **MDX23C-DrumSep** via `audio-separator` (kick/snare/hh/toms/ride/crash).
- `assemble_mirage_fl.py` -> 4-folder FL pack (SFZ instruments, editable MIDI, drum samples, one-shots).
- Benchmark/details: `MEASURE.md`, `results.md`.
