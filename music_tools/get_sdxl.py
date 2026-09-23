"""Robust SDXL fp16 download via snapshot_download (resumes; like the ACE-Step models that worked)."""
from huggingface_hub import snapshot_download
p = snapshot_download(
    "stabilityai/stable-diffusion-xl-base-1.0",
    allow_patterns=["*.json", "*.txt", "**/*.json", "**/*.fp16.safetensors"],
    max_workers=2,
)
print("SDXL_AT:", p)
