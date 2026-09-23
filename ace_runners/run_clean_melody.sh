#!/bin/bash
export LD_LIBRARY_PATH=/opt/rocm/lib HSA_ENABLE_SDMA=0 GPU_MAX_HW_QUEUES=1 \
       PYTORCH_HIP_ALLOC_CONF=expandable_segments:True ACESTEP_CHECKPOINTS_DIR=/opt/ace-models \
       MIOPEN_FIND_MODE=2 MIOPEN_DEBUG_CONV_IMMED_FALLBACK=1
source /opt/aceenv/bin/activate
cd /mnt/d/flbeat/ACE-Step-1.5
G=/mnt/d/flbeat/data/generated; OUT=$G/phonk_clean; mkdir -p "$OUT"
P="drift phonk, memphis phonk, melodic cowbell lead riff, deep booming 808 bass, hard trap hi-hats, dark hypnotic, brazilian funk phonk, vintage cassette"
for i in 1 2 3; do
  echo "### melody_$i ###"
  python -u gen_sft_test.py "$P" "clean_melody_$i" 32 140 || continue
  python -u /mnt/d/flbeat/fl-beat-assimilator/phonkify.py "$G/clean_melody_$i.wav" "$OUT/clean_melody_${i}_GRIT.wav" 1.0
done
echo "ALLDONE"
