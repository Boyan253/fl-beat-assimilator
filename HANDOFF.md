# HANDOFF — for continuing on another machine

Read this first. Owner: **Boyan253** (git identity must be `Boyan253 <Boyan253@users.noreply.github.com>`).

## What this project is
An engine that **(1) ASSIMILATES** any song into faithful, editable per-layer instruments for
FL Studio (90%+ fidelity on any source incl. distorted guitar / 808s), and **(2) CREATES**
original phonk/trap beats as editable MIDI. See `README.md` for usage.

## ⚠️ What is and isn't in this repo
- ✅ In repo: all `*.py` scripts, `README.md`, `requirements.txt`, this file.
- ❌ NOT in repo (gitignored): `env/` (venv), `out/` (ALL data — stems, samples, SFZ/MIDI,
  demos), `sfizz/` (the headless renderer binary). **The other machine must rebuild these.**

## Setup on the new machine (AMD Radeon RX 7900 XTX, 24GB)
1. `python -m venv env` → activate → `pip install -r requirements.txt`.
2. **torch on AMD:** install the **ROCm** build (best via **WSL2 + Ubuntu**, ROCm supports
   gfx1100/7900 XTX). Native-Windows fallback: `torch-directml` (slower). Basic Pitch (ONNX)
   can use DirectML on Windows directly.
3. Playback/render needs **sfizz_render** (build/download sfizz CLI → put at
   `sfizz/x/bin/Release/sfizz_render.exe` or adjust paths) and, inside FL, **Sforzando**
   (free SFZ VST). Neither is in the repo.

## Current state (June 2026)
- ASSIMILATE: proven on Great Wall (distorted guitar ~99%), Excalibur, Full Moon (4 layers
  0.86–0.99). `assimilate.py <mp3> <name>` = one command → FL-ready instruments.
- Two per-occurrence engines: `build_perocc.py` (melodic, leads) + `build_perocc_onset.py`
  (energy onsets, bass/drums). Velocity-layer = restart-safe; silent t=0 anchor = time-locked.
- CREATE: `build_playable.py` / `build_playable_mono.py` (performable multisample instruments),
  `build_drumkit.py` (kick/snare/hat from a drums stem), `create_beat.py` (rule-based phonk —
  sounds GENERIC/chaotic), `create_from_ref.py` (Markov in the style of a reference riff —
  better, still limited). Rule/Markov generation has hit its ceiling.

## NEXT DIRECTION — train/fine-tune a symbolic (MIDI) model for phonk
Goal: AI generates original, coherent phonk parts (cherry-pick the best) → remix into beats.
- **Base model (fine-tune, don't train from zero):** recommended **Anticipatory Music
  Transformer** (modern PyTorch, Lakh = multi-genre+multi-track, AMD-friendly); **aria** for
  pure monophonic lead generation; **Magenta** only if a clean Apache license is required (it's
  dated TF, piano-centric). **VERIFY each model's COMMERCIAL license before selling output.**
  Alternative that's "fully ours": train a small model FROM SCRATCH on our assimilated corpus.
- **Dataset = built with THIS engine:** run `assimilate.py` over ~2000 phonk songs, keep only
  CLEAN MONOPHONIC leads, tokenize (miditok/REMI) → fine-tune set. Data cleanliness > quantity.
- **Speed (7900 XTX, ROCm):** dataset build ~½ day, fine-tune ~15–30 min. No Colab limits.
- **Milestones:** riffs (≈weekend) → multi-track sections (weeks) → structured full songs
  (hard; semi-auto: AI parts + template arrangement) → other genres (just needs that genre's data).
- **Reality:** no model is good "every time" — generate many, pick winners. GPU removes the
  speed limit; DATA quality is the real lever.

## Sellability rule (important)
Assimilated songs are **copyrighted** → study / remix / learn only, **never sell**. Only the
**original CREATE output** is sellable. Confirm the base model's license is commercial, and add
a similarity check (generated vs training corpus) to guarantee originals before selling.

## Workflow
Plan → build → verify (chroma/rms vs source for assimilate; A/B + ear for create) → commit +
push. Keep this file and `README.md` updated as the source of truth across machines.
