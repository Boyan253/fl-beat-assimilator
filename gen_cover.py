"""Style-transfer: generate a beat in the STYLE of a reference track (task_type=cover)."""
import os, sys, shutil
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music

REPO, CKPT, OUT = "/mnt/d/flbeat/ACE-Step-1.5", "/opt/ace-models", "/mnt/d/flbeat/data/generated"
ref    = sys.argv[1]                                   # reference audio (style source)
name   = sys.argv[2] if len(sys.argv) > 2 else "cover_out"
prompt = sys.argv[3] if len(sys.argv) > 3 else "aggressive memphis phonk, heavy distorted 808 bass, hypnotic, dark"
bpm    = int(sys.argv[4]) if len(sys.argv) > 4 else 92

dit = AceStepHandler()
dit.initialize_service(project_root=REPO, config_path="acestep-v15-turbo", device="cuda",
                       use_flash_attention=False, use_mlx_dit=False)
llm = LLMHandler()  # cover task skips the LM, so DON'T load it (frees ~3-4GB VRAM for the cover encode)

params = GenerationParams(caption=prompt, instrumental=True, bpm=bpm, duration=30,
                          inference_steps=32, guidance_scale=7.0,
                          task_type="cover", reference_audio=ref, src_audio=ref)
config = GenerationConfig(batch_size=1, audio_format="wav", use_random_seed=True)
print(f"=== style-transfer (cover) from {os.path.basename(ref)} ===", flush=True)
res = generate_music(dit, llm, params, config, save_dir=OUT)
print("SUCCESS:", res.success, flush=True)
if res.success:
    final = os.path.join(OUT, f"{name}.wav"); shutil.copy(res.audios[0]["path"], final)
    print("FINAL:", final, flush=True)
else:
    print("ERROR:", res.error, flush=True)
