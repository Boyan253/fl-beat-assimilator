#!/bin/bash
# Generate N takes of the same prompt (beat the seed lottery), grit each.
#   bash batch_phonk.sh "NAME" "song prompt" <bpm> <dur> <count>
NAME="$1"; PROMPT="$2"; BPM="${3:-125}"; DUR="${4:-90}"; N="${5:-3}"
export LD_LIBRARY_PATH=/opt/rocm/lib HSA_ENABLE_SDMA=0 GPU_MAX_HW_QUEUES=1 \
       PYTORCH_HIP_ALLOC_CONF=expandable_segments:True ACESTEP_CHECKPOINTS_DIR=/opt/ace-models \
       MIOPEN_FIND_MODE=2 MIOPEN_DEBUG_CONV_IMMED_FALLBACK=1
source /opt/aceenv/bin/activate
cd /mnt/d/flbeat/ACE-Step-1.5
G=/mnt/d/flbeat/data/generated
if [ "$DUR" -gt 60 ]; then STEPS=8; MODEL=acestep-v15-turbo; else STEPS=32; MODEL=acestep-v15-xl-sft; fi
for i in $(seq 1 "$N"); do
  echo "### take $i / $N ###"
  python -u gen_sft_test.py "$PROMPT" "${NAME}_${i}_raw" "$STEPS" "$BPM" "$DUR" "$MODEL" 2>&1 | grep -E "SUCCESS|FINAL:|ERROR"
  python -u /mnt/d/flbeat/fl-beat-assimilator/phonkify.py "$G/${NAME}_${i}_raw.wav" "$G/${NAME}_${i}_GRIT.wav" 1.1 2>&1 | grep -E "phonkified"
done
echo "### BATCH DONE: ${NAME}_1..${N}_GRIT.mp3 in $G ###"
