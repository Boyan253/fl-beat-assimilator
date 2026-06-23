"""ace_factory.py — batch-generate varied phonk tracks (load model once, generate many).

  python ace_factory.py [count] [config_path]
  config_path: acestep-v15-turbo (default) or acestep-v15-xl-turbo (higher quality)

Output -> /mnt/d/flbeat/data/generated/factory/*.wav  (cherry-pick the fire ones,
then run build_trackout.py on the winners to make exclusive packs).
"""
import os, sys, shutil
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music

REPO = "/mnt/d/flbeat/ACE-Step-1.5"
CKPT = "/opt/ace-models"
OUT  = "/mnt/d/flbeat/data/generated/factory"
os.makedirs(OUT, exist_ok=True)

COUNT  = int(sys.argv[1]) if len(sys.argv) > 1 else 8
CONFIG = sys.argv[2] if len(sys.argv) > 2 else "acestep-v15-turbo"

# varied phonk sub-styles + bpm; (name, caption, bpm)
STYLES = [
    ("drift",     "dark drift phonk, distorted 808 bass, cowbell melody, hard memphis drums, aggressive, lo-fi", 150),
    ("brazilian", "brazilian phonk, fast hard 808, bright cowbell melody, gritty drums, energetic", 130),
    ("memphis",   "memphis phonk, dusty vinyl, deep 808, dark eerie melody, slow menacing, cassette", 140),
    ("rage",      "aggressive rage phonk, heavy distorted 808, screaming lead synth, hard trap drums", 160),
    ("ambient",   "atmospheric ambient phonk, dreamy cowbell, deep sub bass, spacey pads, hypnotic", 145),
    ("house",     "phonk house, four on the floor kick, groovy 808, catchy cowbell hook, club energy", 132),
    ("horror",    "horror phonk, sinister bells, distorted 808, creepy atmosphere, dark cinematic", 138),
    ("g-house",   "dark g-house phonk, punchy bass, vocal chops, driving drums, underground", 128),
]

print(f"=== init DiT {CONFIG} + LM (once) ===", flush=True)
dit = AceStepHandler()
dit.initialize_service(project_root=REPO, config_path=CONFIG, device="cuda",
                       use_flash_attention=False, use_mlx_dit=False)
llm = LLMHandler()
llm.initialize(checkpoint_dir=CKPT, lm_model_path="acestep-5Hz-lm-1.7B", backend="pt", device="cuda")
config = GenerationConfig(batch_size=1, audio_format="wav", use_random_seed=True)

made = 0
for i in range(COUNT):
    name, caption, bpm = STYLES[i % len(STYLES)]
    tag = f"{name}_{i+1:02d}"
    print(f"--- [{i+1}/{COUNT}] {tag} ---", flush=True)
    try:
        params = GenerationParams(caption=caption, instrumental=True, bpm=bpm, duration=20, inference_steps=8)
        res = generate_music(dit, llm, params, config, save_dir=OUT)
        if res.success:
            dst = os.path.join(OUT, tag + ".wav")
            shutil.copy(res.audios[0]["path"], dst)
            print(f"  ok -> {tag}.wav", flush=True)
            made += 1
        else:
            print(f"  fail: {res.error}", flush=True)
    except Exception as e:
        print(f"  err: {repr(e)[:160]}", flush=True)

print(f"\n=== {made}/{COUNT} tracks in {OUT} ===", flush=True)
