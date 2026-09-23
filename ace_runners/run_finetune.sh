#!/bin/bash
export LD_LIBRARY_PATH=/opt/rocm/lib:${LD_LIBRARY_PATH:-}
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
# ROCm-on-WSL sustained-load stability knobs (the GPU TDR-crashes mid-training without these):
export HSA_ENABLE_SDMA=0        # disable SDMA copy engine — common WSL hang fix
export GPU_MAX_HW_QUEUES=1       # single HW queue — less driver contention
source /root/flbeat-venv/bin/activate
python -u /mnt/d/flbeat/fl-beat-assimilator/finetune_amt.py
echo "PYTHON_EXIT_CODE=$?"
