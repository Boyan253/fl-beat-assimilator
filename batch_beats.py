"""batch_beats.py — generate a curated set of phonk + trap beats (model loaded once)."""
import os, sys, shutil, subprocess, gc
import torch
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music

REPO, CKPT = "/mnt/d/flbeat/ACE-Step-1.5", "/opt/ace-models"
OUT = "/mnt/d/flbeat/data/generated/beats_batch"
os.makedirs(OUT, exist_ok=True)

# (name, prompt, bpm, grit)  grit>0 = phonk lo-fi pass, grit=0 = clean trap
BEATS = [
    ("phonk_attack",  "aggressive intense phonk, fast driving cowbell melody, pounding 808 bass, relentless memphis drums, dark battle energy, drift phonk", 150, 1.2),
    ("phonk_night",   "dark atmospheric phonk, haunting cowbell melody, deep sliding 808, memphis trap drums, neon night drive, hypnotic", 140, 1.2),
    ("phonk_brazil",  "brazilian automotivo phonk, fast catchy cowbell hook, hard 808 bass, energetic funk drums, hype", 130, 1.2),
    ("phonk_evil",    "evil dark phonk, menacing cowbell riff, distorted 808 bass, hard memphis drums, sinister horror, drift phonk", 145, 1.3),
    ("trap_dark",     "dark hard trap beat, deep booming 808 bass, crisp fast hi-hats, eerie bell melody, menacing modern trap", 140, 0),
    ("trap_melodic",  "melodic emotional trap beat, sad piano melody, deep 808, rolling hi-hats, atmospheric, modern", 130, 0),
    ("trap_rage",     "aggressive rage trap beat, distorted 808, hard hitting drums, dark synth lead, hype energy", 150, 0),
]

dit = AceStepHandler()
dit.initialize_service(project_root=REPO, config_path="acestep-v15-xl-sft", device="cuda",
                       use_flash_attention=False, use_mlx_dit=False)
llm = LLMHandler()
llm.initialize(checkpoint_dir=CKPT, lm_model_path="acestep-5Hz-lm-1.7B", backend="pt", device="cuda")
cfg = GenerationConfig(batch_size=1, audio_format="wav", use_random_seed=True)

for i, (name, prompt, bpm, grit) in enumerate(BEATS):
    print(f"--- [{i+1}/{len(BEATS)}] {name} (bpm {bpm}, grit {grit}) ---", flush=True)
    if os.path.exists(os.path.join(OUT, name + ("_GRIT.mp3" if grit > 0 else ".mp3"))):
        print("  skip(done)", flush=True); continue
    gc.collect(); torch.cuda.empty_cache()   # free VRAM before each gen (XL-SFT is large)
    try:
        p = GenerationParams(caption=prompt, instrumental=True, bpm=bpm, duration=35,
                             inference_steps=32, guidance_scale=7.0)
        r = generate_music(dit, llm, p, cfg, save_dir=OUT)
        if not r.success:
            print("  fail:", r.error, flush=True); continue
        raw = os.path.join(OUT, name + ".wav")
        shutil.copy(r.audios[0]["path"], raw)
        if grit > 0:
            subprocess.run(["python", "/mnt/d/flbeat/fl-beat-assimilator/phonkify.py",
                            raw, os.path.join(OUT, name + "_GRIT.wav"), str(grit)],
                           capture_output=True)
            print(f"  ok -> {name}_GRIT.mp3", flush=True)
        else:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw,
                            "-b:a", "192k", os.path.join(OUT, name + ".mp3")], capture_output=True)
            print(f"  ok -> {name}.mp3", flush=True)
    except Exception as e:
        print("  err:", repr(e)[:150], flush=True)

print("\n=== batch done ->", OUT, "===", flush=True)
