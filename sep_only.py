"""Separate a song into stems only (no transcription) so a clean single stem can be
loaded into Slicex for chopping. Load = librosa, separate = demucs apply_model,
save = soundfile (avoids torchaudio.save->torchcodec)."""
import os, sys, warnings
warnings.filterwarnings("ignore")
D = r"D:\flmidi"
os.environ.update(TMP=D + r"\tmp", TEMP=D + r"\tmp", TORCH_HOME=D + r"\torchcache",
                  HF_HOME=D + r"\hf", XDG_CACHE_HOME=D + r"\hf")
import numpy as np, librosa, soundfile as sf, torch
from demucs.pretrained import get_model
from demucs.apply import apply_model

mp3 = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\User\Downloads\abraham (feat. RJ Pasin).mp3"
base = "abraham"
out = os.path.join(D, "out", base, "stems"); os.makedirs(out, exist_ok=True)
print("separating:", mp3)
model = get_model("htdemucs"); model.eval()
sr = int(model.samplerate)
y, _ = librosa.load(mp3, sr=sr, mono=False)
if y.ndim == 1:
    y = np.stack([y, y])
wav = torch.tensor(y, dtype=torch.float32)
ref = wav.mean(0)
wav = (wav - ref.mean()) / (ref.std() + 1e-8)
with torch.no_grad():
    sources = apply_model(model, wav[None], device="cpu", split=True, overlap=0.25, progress=True)[0]
sources = sources * ref.std() + ref.mean()
for name, src in zip(model.sources, sources):
    p = os.path.join(out, name + ".wav")
    sf.write(p, src.detach().cpu().numpy().T.astype("float32"), sr)
    print("  wrote", p)
print("DONE -> load other.wav (melody/guitar) into Slicex")
