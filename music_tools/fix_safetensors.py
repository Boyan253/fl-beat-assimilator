"""Ensure stanford-crfm/music-medium-800k has a valid local model.safetensors
so from_pretrained can load it without tripping the torch<2.6 CVE guard on .bin."""
import os, glob, torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file, load_file

REPO = "stanford-crfm/music-medium-800k"

# resolve files from cache (downloads if missing)
binp = hf_hub_download(REPO, "pytorch_model.bin")
hf_hub_download(REPO, "config.json")
snap = os.path.dirname(binp)
sf_path = os.path.join(snap, "model.safetensors")
print("snapshot dir:", snap)
print("existing files:", sorted(os.listdir(snap)))

need_convert = True
if os.path.exists(sf_path):
    try:
        d = load_file(sf_path)
        print("existing safetensors OK:", len(d), "tensors")
        need_convert = False
    except Exception as e:
        print("existing safetensors invalid -> reconverting:", e)

if need_convert:
    print("converting .bin -> safetensors ...")
    sd = torch.load(binp, map_location="cpu", weights_only=True)
    # safetensors needs contiguous, non-shared tensors
    save_file({k: v.clone().contiguous() for k, v in sd.items()}, sf_path)
    print("wrote", sf_path)

# print the snapshot path so the caller can use it as the model path
print("MODELPATH=" + snap)
