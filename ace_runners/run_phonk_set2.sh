#!/bin/bash
# Generic phonk with the WINNING recipe (XL-SFT @32 steps + grit), one per fresh process.
export LD_LIBRARY_PATH=/opt/rocm/lib HSA_ENABLE_SDMA=0 GPU_MAX_HW_QUEUES=1 \
       PYTORCH_HIP_ALLOC_CONF=expandable_segments:True ACESTEP_CHECKPOINTS_DIR=/opt/ace-models \
       MIOPEN_FIND_MODE=2 MIOPEN_DEBUG_CONV_IMMED_FALLBACK=1
source /opt/aceenv/bin/activate
cd /mnt/d/flbeat/ACE-Step-1.5
G=/mnt/d/flbeat/data/generated; OUT=$G/phonk_set2; mkdir -p "$OUT"
beat() {  # name prompt bpm
  [ -f "$OUT/${1}_GRIT.mp3" ] && { echo "skip $1"; return; }
  echo "### $1 ###"
  python -u gen_sft_test.py "$2" "$1" 32 "$3" || return
  python -u /mnt/d/flbeat/fl-beat-assimilator/phonkify.py "$G/${1}.wav" "$OUT/${1}_GRIT.wav" 1.1
}
beat set2_night    "dark drift phonk, melodic cowbell lead riff, deep booming 808 bass, hard memphis drums, neon night drive, hypnotic, vintage cassette" 140
beat set2_battle   "aggressive drift phonk, fast driving cowbell melody, pounding 808 bass, relentless memphis drums, dark battle energy, intense" 150
beat set2_melodic  "melodic phonk, beautiful catchy cowbell melody hook, emotional dark minor riff, warm 808 bass, atmospheric, brazilian phonk" 138
beat set2_brazil   "brazilian automotivo phonk, fast bouncy cowbell hook, hard distorted 808, energetic funk drums, hype, club" 130
beat set2_evil     "evil dark phonk, menacing cowbell riff, deep distorted 808 bass, hard memphis drums, sinister horror, drift phonk, night" 145
echo "ALLDONE"
