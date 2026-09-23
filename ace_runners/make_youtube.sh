#!/bin/bash
# One command: title + image-vibe + song -> YouTube-ready MP4 (AI cover + title + audio).
#   bash make_youtube.sh "<TITLE>" "<image vibe prompt>" <song.(mp3|wav)>
TITLE="$1"; VIBE="$2"; SONG="$3"
SAFE=$(echo "$TITLE" | tr ' ' '_')
G=/mnt/d/flbeat/data/generated
export LD_LIBRARY_PATH=/opt/rocm/lib HSA_ENABLE_SDMA=0 GPU_MAX_HW_QUEUES=1 \
       PYTORCH_HIP_ALLOC_CONF=expandable_segments:True MIOPEN_FIND_MODE=2 MIOPEN_DEBUG_CONV_IMMED_FALLBACK=1
source /opt/aceenv/bin/activate
cd /mnt/d/flbeat/fl-beat-assimilator
echo "### 1/2 cover image ###"
python -u gen_cover_image.py "$TITLE" "$VIBE" "$G/${SAFE}_cover.png" || exit 1
echo "### 2/2 video ###"
bash make_video.sh "$G/${SAFE}_cover.png" "$SONG" "$G/${SAFE}.mp4"
echo "### DONE -> $G/${SAFE}.mp4 ###"
