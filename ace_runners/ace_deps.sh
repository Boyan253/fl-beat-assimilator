#!/bin/bash
source /opt/aceenv/bin/activate
# pin installed ROCm torch by exact version so no dep swaps in a CUDA build
cat > /opt/ace_constraints.txt <<'EOF'
torch==2.6.0+rocm6.4.4.git05ad6928
torchaudio==2.6.0+rocm6.4.4.gitd8831425
pytorch-triton-rocm==3.2.0+rocm6.4.4.git20943800
EOF
echo "=== installing core deps ==="
pip install -c /opt/ace_constraints.txt \
  transformers diffusers gradio matplotlib scipy soundfile loguru einops accelerate \
  fastapi diskcache "uvicorn[standard]" numba toml peft lycoris-lora tensorboard \
  typer-slim pywavelets modelscope huggingface_hub 2>&1 | tail -4
echo "=== lightning + nano-vllm (no-deps) ==="
pip install --no-deps lightning torchmetrics nano-vllm 2>&1 | tail -2
echo "=== register acestep package ==="
cd /mnt/d/flbeat/ACE-Step-1.5
pip install -c /opt/ace_constraints.txt -e . --no-deps 2>&1 | tail -3
# re-assert WSL libhsa symlink in case torch was touched
LIBDIR=/opt/aceenv/lib/python3.12/site-packages/torch/lib
[ -f "$LIBDIR/libhsa-runtime64.so" ] && [ ! -L "$LIBDIR/libhsa-runtime64.so" ] && mv "$LIBDIR/libhsa-runtime64.so" "$LIBDIR/libhsa-runtime64.so.bak"
ln -sf /opt/rocm/lib/libhsa-runtime64.so "$LIBDIR/libhsa-runtime64.so"
export LD_LIBRARY_PATH=/opt/rocm/lib HSA_ENABLE_SDMA=0
echo "=== FINAL CHECK ==="
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())" 2>&1 | grep -vE "NumPy|cpu ="
python -c "import transformers,diffusers;import acestep;print('IMPORTS OK')" 2>&1 | tail -3
