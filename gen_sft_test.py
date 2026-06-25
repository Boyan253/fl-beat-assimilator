"""Test the full-quality XL-SFT base (more steps + guidance) for harder/fuller phonk."""
import os, sys, shutil
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music

REPO, CKPT, OUT = "/mnt/d/flbeat/ACE-Step-1.5", "/opt/ace-models", "/mnt/d/flbeat/data/generated"
prompt = sys.argv[1] if len(sys.argv) > 1 else "brutal aggressive phonk, hard distorted 808 bass, loud cowbell melody, banging memphis drums, dark drift phonk, heavy"
name   = sys.argv[2] if len(sys.argv) > 2 else "PHONK_SFT_test"
steps  = int(sys.argv[3]) if len(sys.argv) > 3 else 32
bpm    = int(sys.argv[4]) if len(sys.argv) > 4 else 145
dur    = int(sys.argv[5]) if len(sys.argv) > 5 else 40
cfg    = sys.argv[6] if len(sys.argv) > 6 else "acestep-v15-xl-sft"

dit = AceStepHandler()
dit.initialize_service(project_root=REPO, config_path=cfg, device="cuda",
                       use_flash_attention=False, use_mlx_dit=False)
llm = LLMHandler()
llm.initialize(checkpoint_dir=CKPT, lm_model_path="acestep-5Hz-lm-1.7B", backend="pt", device="cuda")

params = GenerationParams(caption=prompt, instrumental=True, bpm=bpm, duration=dur,
                          inference_steps=steps, guidance_scale=7.0)
config = GenerationConfig(batch_size=1, audio_format="wav", use_random_seed=True)
print(f"=== generating ({steps} steps, xl-sft) ===", flush=True)
res = generate_music(dit, llm, params, config, save_dir=OUT)
print("SUCCESS:", res.success, flush=True)
if res.success:
    final = os.path.join(OUT, f"{name}.wav"); shutil.copy(res.audios[0]["path"], final)
    print("FINAL:", final, flush=True)
else:
    print("ERROR:", res.error, flush=True)
