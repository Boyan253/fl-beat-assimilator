#!/bin/bash
set -e
source /opt/aceenv/bin/activate
pip uninstall -y torch torchaudio torchvision pytorch-triton-rocm 2>&1 | tail -1
WB=https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.4
TORCH=torch-2.6.0%2Brocm6.4.4.git05ad6928-cp312-cp312-linux_x86_64.whl
AUDIO=torchaudio-2.6.0%2Brocm6.4.4.gitd8831425-cp312-cp312-linux_x86_64.whl
# matching triton for torch 2.6.0 (3.2.0), if present
TRITON=$(curl -s $WB/ | grep -oE "pytorch_triton_rocm-3\.2[^\"]+cp312[^\"]+\.whl" | head -1)
echo "triton=$TRITON"
pip install --no-deps "$WB/$TORCH" "$WB/$AUDIO"
[ -n "$TRITON" ] && pip install --no-deps "$WB/$TRITON" || echo "(no triton 3.2 — eager mode is fine)"
# torch runtime deps (since --no-deps skipped them)
pip install numpy filelock typing_extensions sympy networkx jinja2 fsspec mpmath markupsafe
# WSL libhsa fix
LIBDIR=/opt/aceenv/lib/python3.12/site-packages/torch/lib
if [ -f "$LIBDIR/libhsa-runtime64.so" ] && [ ! -L "$LIBDIR/libhsa-runtime64.so" ]; then
  mv "$LIBDIR/libhsa-runtime64.so" "$LIBDIR/libhsa-runtime64.so.bak"
fi
ln -sf /opt/rocm/lib/libhsa-runtime64.so "$LIBDIR/libhsa-runtime64.so"
export LD_LIBRARY_PATH=/opt/rocm/lib HSA_ENABLE_SDMA=0
echo "=== TORCH CHECK ==="
python -c "import torch;print('v',torch.__version__,'cuda',torch.cuda.is_available());\
x=torch.randn(64,64,device='cuda'); torch.cuda.synchronize(); print('GPU OK', float((x@x).sum()), torch.cuda.get_device_name(0))" 2>&1 | grep -vE "NumPy|cpu ="
