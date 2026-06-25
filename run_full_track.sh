#!/bin/bash
# Full package: generate phonk song (~65s, proper ending) + clean SDXL cover + MP4.
#   bash run_full_track.sh "TITLE" "song prompt" "image prompt" <bpm> <dur>
TITLE="$1"; SP="$2"; IP="$3"; BPM="${4:-145}"; DUR="${5:-65}"
SAFE=$(echo "$TITLE" | tr ' ' '_'); G=/mnt/d/flbeat/data/generated
export LD_LIBRARY_PATH=/opt/rocm/lib HSA_ENABLE_SDMA=0 GPU_MAX_HW_QUEUES=1 \
       PYTORCH_HIP_ALLOC_CONF=expandable_segments:True ACESTEP_CHECKPOINTS_DIR=/opt/ace-models \
       MIOPEN_FIND_MODE=2 MIOPEN_DEBUG_CONV_IMMED_FALLBACK=1
source /opt/aceenv/bin/activate
cd /mnt/d/flbeat/ACE-Step-1.5
echo "### 1/3 song (XL-SFT ${DUR}s) ###"
python -u gen_sft_test.py "$SP" "${SAFE}_raw" 32 "$BPM" "$DUR" || exit 1
echo "### 2/3 grit ###"
python -u /mnt/d/flbeat/fl-beat-assimilator/phonkify.py "$G/${SAFE}_raw.wav" "$G/${SAFE}_GRIT.wav" 1.1
echo "### 3/3 cover + video (SDXL, fresh process) ###"
bash /mnt/d/flbeat/data/make_youtube.sh "$TITLE" "$IP" "$G/${SAFE}_GRIT.mp3"
echo "### ALL DONE -> $G/${SAFE}.mp4 ###"
