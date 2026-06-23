"""ace_gen.py — flexible ACE-Step generation: instrumental OR vocals, any length, any model.

  python ace_gen.py --prompt "..." [--name out] [--bpm 150] [--dur 20]
                    [--lyrics "..."] [--lang en] [--config acestep-v15-turbo|acestep-v15-xl-turbo]

  # instrumental phonk beat:
  python ace_gen.py --prompt "dark drift phonk, 808, cowbell" --dur 20 --name beat1
  # FULL SONG with vocals:
  python ace_gen.py --prompt "melodic phonk with vocals, emotional" --dur 120 \
                    --lyrics "[Verse] ... [Chorus] ..." --lang en --name song1
"""
import os, sys, shutil, argparse
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music

REPO, CKPT, OUT = "/mnt/d/flbeat/ACE-Step-1.5", "/opt/ace-models", "/mnt/d/flbeat/data/generated"

p = argparse.ArgumentParser()
p.add_argument("--prompt", required=True)
p.add_argument("--name", default="ace_track")
p.add_argument("--bpm", type=int, default=150)
p.add_argument("--dur", type=int, default=20)
p.add_argument("--lyrics", default="")
p.add_argument("--lang", default="en")
p.add_argument("--config", default="acestep-v15-turbo")
p.add_argument("--fmt", default="wav")
a = p.parse_args()
os.makedirs(OUT, exist_ok=True)

instrumental = (a.lyrics.strip() == "")
print(f"=== init {a.config} ({'instrumental' if instrumental else 'VOCALS'}) ===", flush=True)
dit = AceStepHandler()
dit.initialize_service(project_root=REPO, config_path=a.config, device="cuda",
                       use_flash_attention=False, use_mlx_dit=False)
llm = LLMHandler()
llm.initialize(checkpoint_dir=CKPT, lm_model_path="acestep-5Hz-lm-1.7B", backend="pt", device="cuda")

params = GenerationParams(
    caption=a.prompt, bpm=a.bpm, duration=a.dur, inference_steps=8,
    instrumental=instrumental,
    lyrics=(a.lyrics if not instrumental else "[Instrumental]"),
    vocal_language=(a.lang if not instrumental else "unknown"),
)
config = GenerationConfig(batch_size=1, audio_format=a.fmt, use_random_seed=True)
print("=== generating ===", flush=True)
res = generate_music(dit, llm, params, config, save_dir=OUT)
print("SUCCESS:", res.success, flush=True)
if res.success:
    final = os.path.join(OUT, f"{a.name}.{a.fmt}")
    shutil.copy(res.audios[0]["path"], final)
    print("FINAL:", final, flush=True)
else:
    print("ERROR:", res.error, flush=True)
