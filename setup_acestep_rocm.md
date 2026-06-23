# ACE-Step 1.5 on the RX 7900 XTX (ROCm/WSL2) — working recipe

Generates **real phonk audio** from a text prompt, locally, MIT-licensed (sellable), on AMD.
Verified 2026-06-23: `SUCCESS: True` → 20s/48kHz/stereo mp3 on the 7900 XTX.

This is the copyright-clean Suno/Udio alternative base (trained on licensed/royalty-free data).

## The setup that works (hard-won)

1. **Repo** (code on D:, small): `git clone https://github.com/ace-step/ACE-Step-1.5` → `/mnt/d/flbeat/ACE-Step-1.5`
2. **venv on FAST ext4, NOT /mnt/d** (NTFS over 9p is unusably slow for multi-GB installs):
   `python3 -m venv /opt/aceenv`
3. **torch = AMD's WSL-tested wheel, matching system ROCm 6.4.4** (official pytorch.org rocm wheels
   give CUDA=False / HSA asserts on WSL). Use repo.radeon.com, `--no-deps`, then add runtime deps:
   - `torch-2.6.0+rocm6.4.4` + `torchaudio-2.6.0+rocm6.4.4` + `pytorch_triton_rocm-3.2.0` (cp312)
   - **libhsa WSL fix**: symlink torch/lib/libhsa-runtime64.so → /opt/rocm/lib/libhsa-runtime64.so
   - torch 2.6.0 is the minimum that satisfies ACE-Step AND has the WSL-tested AMD wheel
4. **transformers MUST be <4.58** (ACE-Step model code uses the 4.5x API; 5.x → "meta tensors" error).
   `pip install "transformers>=4.51,<4.58"` (4.57.6 works). torch 2.6 clears the CVE guard.
5. **Other deps**: diffusers gradio accelerate soundfile einops scipy numba peft lightning(--no-deps)
   tensorboard typer-slim pywavelets modelscope **vector_quantize_pytorch**. Pin torch via a
   constraints file so nothing swaps in a CUDA build.
6. **LM backend = "pt"** (not vllm — vllm/nano-vllm unavailable on ROCm). LLM = `acestep-5Hz-lm-1.7B`.
7. **Runtime env**: `LD_LIBRARY_PATH=/opt/rocm/lib`, `HSA_ENABLE_SDMA=0`, `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`.
   **Do NOT set HSA_OVERRIDE_GFX_VERSION** — it makes the WSL libhsakmt topology read ASSERT/abort
   (system already detects gfx1100 correctly).
8. **Models** (9.4GB) → fast ext4: `ACESTEP_CHECKPOINTS_DIR=/opt/ace-models python -m acestep.model_downloader`
   (gets vae, Qwen text encoder, `acestep-v15-turbo` DiT, `acestep-5Hz-lm-1.7B`).

## Generate
`bash /mnt/d/flbeat/data/run_ace_gen.sh "dark drift phonk, distorted 808, cowbell melody, ..."`
→ uses `gen_phonk_audio.py`: AceStepHandler.initialize_service(config_path="acestep-v15-turbo",
use_flash_attention=False, use_mlx_dit=False) + LLMHandler(backend="pt") + generate_music(...).
inference_steps=8 (turbo), instrumental=True.

## Next: fine-tune on royalty-free phonk to specialize + make it "ours".

## LoRA/LoKr fine-tuning on ROCm/WSL — additional fixes (verified 2026-06-23, training works)
- Data: audio + <name>.lyrics.txt ("[Instrumental]") + <name>.json {caption,bpm,language}. ~15 tracks is enough.
- ACE-Step path-safety guard hardwires the safe-root to the REPO dir -> dataset, tensors, and
  output_dir MUST live under /mnt/d/flbeat/ACE-Step-1.5/ (copy data in).
- MIOpen on WSL: VAE-encode convs fail ("Invalid elapsed time"/"no suitable algorithm"). Fix env:
  MIOPEN_FIND_MODE=2  MIOPEN_DEBUG_CONV_IMMED_FALLBACK=1
- LyCORIS bug: inject_lokr_into_dit() apply_to() leaves adapter params on CPU -> "cuda:0 and cpu".
  PATCHED acestep/training/lokr_utils.py: after apply_to(), lycoris_net.to(decoder device).
- Headless trainer: train_phonk_lokr.py (DatasetBuilder.scan_directory -> preprocess_to_tensors
  -> LoKRTrainer.train_from_preprocessed). LoKr ~16.5s/epoch, loss ~0.04. Adapter -> lora_phonk_out.
