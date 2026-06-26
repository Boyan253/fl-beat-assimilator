"""gen_cover_phonk.py — drift-phonk album covers in the Kordhell / RJ Pasin style:
central edgy character + neon rim-glow + dark high-contrast bg, then a lo-fi pass
(film grain + chromatic aberration + vignette + contrast) for the VHS/phonk look.
NO text on the image.

  python gen_cover_phonk.py "NAME::subject description" ["NAME2::subject2" ...]
Outputs square 1:1 covers to /mnt/d/flbeat/data/generated/covers/<NAME>_cover.png
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch
from PIL import Image, ImageEnhance

OUTDIR = "/mnt/d/flbeat/data/generated/covers"
os.makedirs(OUTDIR, exist_ok=True)

STYLE = (", drift phonk album cover art, single central character, dramatic cinematic lighting, "
         "intense neon rim light, glowing red and purple accents, dark high-contrast background, "
         "ominous horror atmosphere, gritty, ultra detailed, sharp focus, moody, vignette")
NEG = ("text, words, letters, title, watermark, signature, logo, frame, border, blurry, low quality, "
       "deformed, disfigured, extra limbs, bad anatomy, bad hands, cropped, jpeg artifacts, washed out, flat")


def stylize(img):
    """lo-fi phonk grade: contrast+sat, chromatic aberration, film grain, vignette."""
    img = ImageEnhance.Contrast(img).enhance(1.13)
    img = ImageEnhance.Color(img).enhance(1.20)
    a = np.asarray(img).astype(np.int16)
    a[:, :, 0] = np.roll(a[:, :, 0], 2, axis=1)    # red shifted right
    a[:, :, 2] = np.roll(a[:, :, 2], -2, axis=1)   # blue shifted left  -> chromatic aberration
    noise = np.random.normal(0, 9, a.shape[:2])
    for c in range(3):
        a[:, :, c] = np.clip(a[:, :, c] + noise, 0, 255)
    out = a.astype(np.float32)
    h, w = out.shape[:2]
    Y, X = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing="ij")
    vig = np.clip(1.0 - 0.55 * np.clip(np.sqrt(X ** 2 + Y ** 2) - 0.45, 0, 1), 0.35, 1.0)
    for c in range(3):
        out[:, :, c] *= vig
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


from diffusers import StableDiffusionXLPipeline
print("loading SDXL...", flush=True)
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16,
    variant="fp16", use_safetensors=True).to("cuda")
pipe.set_progress_bar_config(disable=True)

for arg in sys.argv[1:]:
    name, _, subject = arg.partition("::")
    name = name.strip()
    prompt = subject.strip() + STYLE
    print(f"generating cover: {name}", flush=True)
    img = pipe(prompt=prompt, negative_prompt=NEG, height=1024, width=1024,
               num_inference_steps=40, guidance_scale=8.0).images[0]
    out = stylize(img.convert("RGB"))
    path = os.path.join(OUTDIR, f"{name}_cover.png")
    out.save(path)
    print("SAVED:", path, flush=True)
