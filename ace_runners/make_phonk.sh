#!/bin/bash
# THE WINNING PHONK RECIPE (cracked 2026-06-23). One command -> insane phonk track.
#   bash make_phonk.sh "<phonk prompt>" <name> [grit=1.2]
# Recipe: XL-SFT base (full quality, 32 steps) + phonk-forward prompt (cowbell/808/memphis,
# NO 'brutal/gritty/distorted/metal' words) + phonkify grit pass (lo-fi/cassette/distorted 808).
PROMPT="${1:-drift phonk, memphis phonk, melodic cowbell lead riff, deep booming 808 bass, hard trap hi-hats, dark hypnotic, brazilian funk phonk, vintage cassette}"
NAME="${2:-phonk_song}"
GRIT="${3:-1.2}"
export LD_LIBRARY_PATH=/opt/rocm/lib HSA_ENABLE_SDMA=0 GPU_MAX_HW_QUEUES=1 \
       PYTORCH_HIP_ALLOC_CONF=expandable_segments:True ACESTEP_CHECKPOINTS_DIR=/opt/ace-models \
       MIOPEN_FIND_MODE=2 MIOPEN_DEBUG_CONV_IMMED_FALLBACK=1
source /opt/aceenv/bin/activate
cd /mnt/d/flbeat/ACE-Step-1.5
echo "### 1/2 generate (XL-SFT, 32 steps) ###"
python -u gen_sft_test.py "$PROMPT" "$NAME" 32 || exit 1
echo "### 2/2 phonk grit pass ###"
python -u /mnt/d/flbeat/fl-beat-assimilator/phonkify.py \
  /mnt/d/flbeat/data/generated/$NAME.wav /mnt/d/flbeat/data/generated/${NAME}_GRIT.wav "$GRIT"
echo "### DONE -> /mnt/d/flbeat/data/generated/${NAME}_GRIT.mp3 ###"
