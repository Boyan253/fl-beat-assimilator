#!/bin/bash
# Step 1 (v2): venv on FAST ext4 (/opt), not /mnt/d NTFS. Recent ROCm torch + WSL fix.
set -e
python3 -m venv /opt/aceenv
source /opt/aceenv/bin/activate
pip install --upgrade pip wheel setuptools
pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
LIBDIR=$(python -c "import torch,os;print(os.path.join(os.path.dirname(torch.__file__),'lib'))")
if [ -f "$LIBDIR/libhsa-runtime64.so" ]; then
  mv "$LIBDIR/libhsa-runtime64.so" "$LIBDIR/libhsa-runtime64.so.bundled.bak" || true
  ln -sf /opt/rocm/lib/libhsa-runtime64.so "$LIBDIR/libhsa-runtime64.so"
fi
export LD_LIBRARY_PATH=/opt/rocm/lib
echo "=== TORCH CHECK ==="
python -c "import torch;print('version',torch.__version__);print('cuda',torch.cuda.is_available());print('dev', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
