#!/bin/bash
# Replace official torch with AMD's WSL-tested torch 2.6.0+rocm6.4.4 (matches system, >=2.6).
set -e
source /opt/aceenv/bin/activate
pip uninstall -y torch torchaudio torchvision pytorch-triton-rocm 2>&1 | tail -1
WB=https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.4
# pick cp312 wheels: torch 2.6.0, matching torchaudio 2.6.0, and triton
ttriton=$(curl -s $WB/ | grep -oE "pytorch_triton_rocm-[^\"]+cp312[^\"]+\.whl" | head -1)
ttorch=$(curl -s $WB/ | grep -oE "torch-2\.6\.0[^\"]+cp312[^\"]+\.whl" | head -1)
taudio=$(curl -s $WB/ | grep -oE "torchaudio-2\.6\.0[^\"]+cp312[^\"]+\.whl" | head -1)
echo "triton=$ttriton"; echo "torch=$ttorch"; echo "audio=$taudio"
pip install "$WB/$ttriton" "$WB/$ttorch"
[ -n "$taudio" ] && pip install "$WB/$taudio" || echo "(no matching torchaudio 2.6.0 — will handle separately)"
# WSL libhsa fix
LIBDIR=/opt/aceenv/lib/python3.12/site-packages/torch/lib
if [ -f "$LIBDIR/libhsa-runtime64.so" ] && [ ! -L "$LIBDIR/libhsa-runtime64.so" ]; then
  mv "$LIBDIR/libhsa-runtime64.so" "$LIBDIR/libhsa-runtime64.so.bak"
fi
ln -sf /opt/rocm/lib/libhsa-runtime64.so "$LIBDIR/libhsa-runtime64.so"
export LD_LIBRARY_PATH=/opt/rocm/lib HSA_ENABLE_SDMA=0
echo "=== TORCH CHECK ==="
python -c "import torch;print('v',torch.__version__,'cuda',torch.cuda.is_available());\
import torch as t; x=t.randn(64,64,device='cuda'); t.cuda.synchronize(); print('GPU OK', float((x@x).sum()), t.cuda.get_device_name(0))" 2>&1 | grep -vE "NumPy|cpu ="
