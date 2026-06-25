"""gen_cover_image.py — generate a CLEAN album-style cover with SDXL (no text on it).

  python gen_cover_image.py "<title(ignored for art)>" "<image prompt>" <out.png>
Title arg kept for the make_youtube wrapper (used only for the filename), NOT drawn.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import torch

TITLE  = sys.argv[1] if len(sys.argv) > 1 else "cover"
PROMPT = sys.argv[2] if len(sys.argv) > 2 else (
    "dark moody phonk album cover, neon city at night, cinematic, atmospheric, highly detailed")
OUT    = sys.argv[3] if len(sys.argv) > 3 else "/mnt/d/flbeat/data/generated/cover.png"
NEG    = "text, words, letters, watermark, signature, logo, blurry, low quality, deformed, ugly, jpeg artifacts"

from diffusers import StableDiffusionXLPipeline
print("loading SDXL...", flush=True)
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16,
    variant="fp16", use_safetensors=True)
pipe = pipe.to("cuda")
pipe.set_progress_bar_config(disable=True)

print("generating SDXL image (1344x768, 16:9)...", flush=True)
img = pipe(prompt=PROMPT + ", masterpiece, best quality, sharp focus, 8k",
           negative_prompt=NEG, height=768, width=1344,
           num_inference_steps=30, guidance_scale=7.0).images[0]

# clean: just upscale to 1080p (cover-crop), NO text
from PIL import Image
src = img.convert("RGB")
sc = max(1920 / src.width, 1080 / src.height)
src = src.resize((int(src.width * sc), int(src.height * sc)), Image.LANCZOS)
lft, top = (src.width - 1920) // 2, (src.height - 1080) // 2
src.crop((lft, top, lft + 1920, top + 1080)).save(OUT)
print("FINAL_IMAGE:", OUT, flush=True)
