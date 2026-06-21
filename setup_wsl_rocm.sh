#!/usr/bin/env bash
# Working ROCm + PyTorch + AMT setup on SOF-W-BLD-0009 (AMD RX 7900 XTX / gfx1100) in WSL2 Ubuntu 24.04.
# Verified 2026-06-20: torch.cuda.is_available()=True, AMT generates multi-track MIDI on the GPU.
# Run as root inside WSL: wsl -d Ubuntu-24.04 -u root -- bash setup_wsl_rocm.sh
set -euxo pipefail
ROCM=6.4.4
export DEBIAN_FRONTEND=noninteractive

# 1) ROCm runtime for WSL (uses the Windows driver via /dev/dxg; NO dkms kernel module)
DEB=$(curl -s https://repo.radeon.com/amdgpu-install/$ROCM/ubuntu/noble/ | grep -oE "amdgpu-install_[^\"]+\.deb" | head -1)
wget -q "https://repo.radeon.com/amdgpu-install/$ROCM/ubuntu/noble/$DEB" -O "/root/$DEB"
apt-get update -y && apt-get install -y "/root/$DEB" && apt-get update -y
amdgpu-install -y --usecase=wsl,rocm --no-dkms          # rocminfo should now show gfx1100

# 2) venv + AMD's ROCm PyTorch wheels (torch 2.4.1+rocm6.4.4). Wheels live at repo.radeon.com/rocm/manylinux.
apt-get install -y python3-venv python3-pip
python3 -m venv /root/flbeat-venv && source /root/flbeat-venv/bin/activate
pip install --upgrade pip wheel
WB=https://repo.radeon.com/rocm/manylinux/rocm-rel-$ROCM
for pkg in pytorch_triton_rocm torch torchvision; do
  f=$(curl -s $WB/ | grep -oE "${pkg}-[^\"]+\.whl" | grep -E "cp312|none-any" | head -1)
  pip install "$WB/$f"
done

# 3) CRUCIAL WSL FIX: torch ships its own libhsa-runtime64.so that does NOT talk to /dev/dxg.
#    Point it at the working /opt/rocm one, else torch.cuda.is_available()==False.
LIBDIR=/root/flbeat-venv/lib/python3.12/site-packages/torch/lib
mv "$LIBDIR/libhsa-runtime64.so" "$LIBDIR/libhsa-runtime64.so.bundled.bak" || true
ln -sf /opt/rocm/lib/libhsa-runtime64.so "$LIBDIR/libhsa-runtime64.so"
export LD_LIBRARY_PATH=/opt/rocm/lib:${LD_LIBRARY_PATH:-}    # keep this set for all runs

# 4) AMT base model deps. anticipation + mido WITHOUT deps so pip can't replace ROCm-torch.
pip install --no-deps "git+https://github.com/jthickstun/anticipation.git" mido transformers
# NOTE: new transformers blocks loading .bin with torch<2.6 (CVE-2025-32434). Convert the model
# checkpoint to safetensors once, then from_pretrained loads it with NO guard:
#   from huggingface_hub import hf_hub_download; from safetensors.torch import save_file
#   binp=hf_hub_download(repo,"pytorch_model.bin"); hf_hub_download(repo,"config.json")
#   sd=torch.load(binp,map_location="cpu",weights_only=True)
#   save_file({k:v.clone().contiguous() for k,v in sd.items()}, os.path.dirname(binp)+"/model.safetensors")
#   m=AutoModelForCausalLM.from_pretrained(os.path.dirname(binp)).cuda()

python -c "import torch;print('cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))"

# 5) TRAINING/INFERENCE STABILITY ON RDNA3 (7900 XTX) + WSL — hard-won, do not skip:
#  - attn_implementation="eager" when loading the model. The default sdpa/flash-attention
#    is experimental on Navi31: its BACKWARD pass produces NaN gradients (training diverges).
#  - Export these before any sustained GPU run, or the driver TDR-crashes after a few steps
#    (process spins at ~200% CPU on a wedged GPU; AMD bug-report tool fires):
#       export HSA_ENABLE_SDMA=0        # disable SDMA copy engine — the key WSL hang fix
#       export GPU_MAX_HW_QUEUES=1      # single HW queue, less driver contention
#       export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
#  - Keep sequences short (<=512 tokens) so kernels stay well under the driver watchdog.
#  - Use a plain fp32 + AdamW + grad-clip loop with a FLAT indexed loop (no generator/closure
#    feeding batches — that hung at step 0). HF Trainer also diverged to NaN here.
#  - Long autoregressive generation can still hit "HSA BlockAllocator::alloc failed" from VRAM
#    fragmentation; generate ONE riff per fresh process to avoid cross-run degradation.
