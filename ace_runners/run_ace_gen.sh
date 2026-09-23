#!/bin/bash
export LD_LIBRARY_PATH=/opt/rocm/lib:${LD_LIBRARY_PATH:-}
export HSA_ENABLE_SDMA=0
export GPU_MAX_HW_QUEUES=1
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export ACESTEP_CHECKPOINTS_DIR=/opt/ace-models
source /opt/aceenv/bin/activate
cd /mnt/d/flbeat/ACE-Step-1.5
python -u gen_phonk_audio.py "$@"
