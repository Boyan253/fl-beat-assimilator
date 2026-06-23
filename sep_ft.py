"""Separate with a chosen (higher-quality) Demucs model.
  python sep_ft.py <song.mp3> <out_stems_dir> [model=htdemucs_ft]
htdemucs_ft = fine-tuned 4-model ensemble: cleaner stems, much better low-end."""
import os, sys, warnings
warnings.filterwarnings("ignore")
D = r"D:\flmidi"
os.environ.update(TMP=D + r"\tmp", TEMP=D + r"\tmp", TORCH_HOME=D + r"\torchcache",
                  HF_HOME=D + r"\hf", XDG_CACHE_HOME=D + r"\hf")
import numpy as np, librosa, soundfile as sf, torch
from demucs.pretrained import get_model
from demucs.apply import apply_model

MP3 = sys.argv[1]
OUTDIR = sys.argv[2]
MODEL = sys.argv[3] if len(sys.argv) > 3 else "htdemucs_ft"
os.makedirs(OUTDIR, exist_ok=True)
print("loading", MODEL, "(first run downloads ~300MB)...", flush=True)
model = get_model(MODEL); model.eval()
sr = int(model.samplerate)
y, _ = librosa.load(MP3, sr=sr, mono=False)
if y.ndim == 1:
    y = np.stack([y, y])
wav = torch.tensor(y, dtype=torch.float32); ref = wav.mean(0)
wav = (wav - ref.mean()) / (ref.std() + 1e-8)
print("separating (ft is ~4x slower — be patient)...", flush=True)
with torch.no_grad():
    srcs = apply_model(model, wav[None], device="cpu", split=True, overlap=0.25, progress=True)[0]
srcs = srcs * ref.std() + ref.mean()
for name, src in zip(model.sources, srcs):
    p = os.path.join(OUTDIR, name + ".wav")
    sf.write(p, src.detach().cpu().numpy().T.astype("float32"), sr)
    print("wrote", name, flush=True)
print("DONE ->", OUTDIR)
