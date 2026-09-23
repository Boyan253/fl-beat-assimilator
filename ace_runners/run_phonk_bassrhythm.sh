#!/bin/bash
# Phonk with strong BASS RHYTHM (sliding/bouncing 808 groove), one per fresh process.
export LD_LIBRARY_PATH=/opt/rocm/lib HSA_ENABLE_SDMA=0 GPU_MAX_HW_QUEUES=1 \
       PYTORCH_HIP_ALLOC_CONF=expandable_segments:True ACESTEP_CHECKPOINTS_DIR=/opt/ace-models \
       MIOPEN_FIND_MODE=2 MIOPEN_DEBUG_CONV_IMMED_FALLBACK=1
source /opt/aceenv/bin/activate
cd /mnt/d/flbeat/ACE-Step-1.5
G=/mnt/d/flbeat/data/generated; OUT=$G/phonk_more; mkdir -p "$OUT"
beat() {
  [ -f "$OUT/${1}_GRIT.mp3" ] && { echo "skip $1"; return; }
  echo "### $1 ###"; python -u gen_sft_test.py "$2" "$1" 32 "$3" || return
  python -u /mnt/d/flbeat/fl-beat-assimilator/phonkify.py "$G/${1}.wav" "$OUT/${1}_GRIT.wav" 1.0
}
beat phonk_groove1 "dark phonk, sliding 808 bassline groove, bouncing rhythmic 808 bass pattern, haunting cowbell melody, memphis trap drums, neon night drive, hypnotic" 140 0
beat phonk_groove2 "drift phonk, deep sliding 808 bass riff, syncopated 808 bassline, catchy cowbell melody, hard memphis drums, dark, hypnotic groove" 145 0
beat phonk_groove3 "memphis phonk, melodic 808 bassline groove, gliding 808 bass notes, cowbell lead melody, boom bap drums, dark cassette, night" 138 0
echo "ALLDONE"
