#!/bin/bash
# More PHONK only (one per fresh process; XL-SFT frees VRAM only on exit).
export LD_LIBRARY_PATH=/opt/rocm/lib HSA_ENABLE_SDMA=0 GPU_MAX_HW_QUEUES=1 \
       PYTORCH_HIP_ALLOC_CONF=expandable_segments:True ACESTEP_CHECKPOINTS_DIR=/opt/ace-models \
       MIOPEN_FIND_MODE=2 MIOPEN_DEBUG_CONV_IMMED_FALLBACK=1
source /opt/aceenv/bin/activate
cd /mnt/d/flbeat/ACE-Step-1.5
G=/mnt/d/flbeat/data/generated
OUT=$G/phonk_more; mkdir -p "$OUT"

beat() {  # name prompt bpm grit
  [ -f "$OUT/${1}_GRIT.mp3" ] && { echo "skip $1"; return; }
  echo "### $1 ###"
  python -u gen_sft_test.py "$2" "$1" 32 "$3" || return
  python -u /mnt/d/flbeat/fl-beat-assimilator/phonkify.py "$G/${1}.wav" "$OUT/${1}_GRIT.wav" "$4"
}

beat phonk_chase   "fast driving drift phonk, racing cowbell melody, deep sliding 808 bass, hard memphis drums, high speed chase energy, hypnotic" 150 1.2
beat phonk_doom    "dark heavy phonk, deep ominous cowbell riff, distorted booming 808, slow heavy memphis drums, doom dread, drift phonk" 135 1.3
beat phonk_melody2 "melodic phonk, beautiful catchy cowbell melody hook, emotional minor riff, warm 808 bass, memphis drift, atmospheric" 140 1.1
beat phonk_rave    "energetic phonk house, bouncy cowbell melody, punchy 808 bass, four on the floor groove, club hype, brazilian phonk" 130 1.0
beat phonk_ghost   "haunting eerie phonk, ghostly cowbell melody, deep dark 808, sparse memphis drums, creepy atmospheric, drift phonk, night" 142 1.2
echo "ALLDONE"
