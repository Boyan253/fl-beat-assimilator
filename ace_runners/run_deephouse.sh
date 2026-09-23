#!/bin/bash
# Emotional deep house instrumental (style of "all things break - stay"), CLEAN (no phonk grit).
# usage: bash run_deephouse.sh <count> <dur>
CNT="${1:-3}"; DUR="${2:-60}"
G=/mnt/d/flbeat/data/generated
export LD_LIBRARY_PATH=/opt/rocm/lib HSA_ENABLE_SDMA=0 GPU_MAX_HW_QUEUES=1 \
       PYTORCH_HIP_ALLOC_CONF=expandable_segments:True ACESTEP_CHECKPOINTS_DIR=/opt/ace-models \
       MIOPEN_FIND_MODE=2 MIOPEN_DEBUG_CONV_IMMED_FALLBACK=1
source /opt/aceenv/bin/activate
cd /mnt/d/flbeat/ACE-Step-1.5 || exit 1
if [ "$DUR" -gt 60 ]; then STEPS=8; MODEL=acestep-v15-turbo; else STEPS=32; MODEL=acestep-v15-xl-sft; fi
PROMPT="deep house, 126 BPM, B flat minor, punchy four-on-the-floor kick drum on every beat, driving danceable club groove, tight punchy drums, rolling deep sub bass locked to the kick, off-beat open hi-hats, crisp closed hats, claps on beats 2 and 4, shaker groove, warm emotional chords, soft melodic pluck, groovy energetic main drop section, clean modern club mix"
echo "PROMPT: $PROMPT"
echo "MODEL=$MODEL STEPS=$STEPS DUR=$DUR BPM=126 KEY=Bbmin COUNT=$CNT"
for i in $(seq 1 "$CNT"); do
  echo "### take $i/$CNT ###"
  python -u gen_sft_test.py "$PROMPT" "STAY_DHK_${i}_raw" "$STEPS" 126 "$DUR" "$MODEL" || echo "!!! take $i FAILED"
done
echo "### ALL DONE ###"
ls -la "$G"/STAY_DHK_*_raw.wav 2>/dev/null
