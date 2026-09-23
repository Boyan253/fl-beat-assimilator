#!/bin/bash
# Generate beats ONE PER FRESH PROCESS (XL-SFT is 19GB; VRAM only frees on process exit).
export LD_LIBRARY_PATH=/opt/rocm/lib HSA_ENABLE_SDMA=0 GPU_MAX_HW_QUEUES=1 \
       PYTORCH_HIP_ALLOC_CONF=expandable_segments:True ACESTEP_CHECKPOINTS_DIR=/opt/ace-models \
       MIOPEN_FIND_MODE=2 MIOPEN_DEBUG_CONV_IMMED_FALLBACK=1
source /opt/aceenv/bin/activate
cd /mnt/d/flbeat/ACE-Step-1.5
G=/mnt/d/flbeat/data/generated
OUT=$G/beats_batch

# name | prompt | bpm | grit
beat() {
  local name="$1" prompt="$2" bpm="$3" grit="$4"
  [ -f "$OUT/${name}.mp3" -o -f "$OUT/${name}_GRIT.mp3" ] && { echo "skip $name"; return; }
  echo "### $name ###"
  python -u gen_sft_test.py "$prompt" "$name" 32 "$bpm" || return
  if [ "${grit%.*}" != "0" ]; then
    python -u /mnt/d/flbeat/fl-beat-assimilator/phonkify.py "$G/${name}.wav" "$OUT/${name}_GRIT.wav" "$grit"
  else
    ffmpeg -y -loglevel error -i "$G/${name}.wav" -b:a 192k "$OUT/${name}.mp3"
    echo "  -> ${name}.mp3"
  fi
}

beat phonk_brazil "brazilian automotivo phonk, fast catchy cowbell hook, hard 808 bass, energetic funk drums, hype" 130 1.2
beat phonk_evil   "evil dark phonk, menacing cowbell riff, distorted 808 bass, hard memphis drums, sinister horror, drift phonk" 145 1.3
beat trap_dark    "dark hard trap beat, deep booming 808 bass, crisp fast hi-hats, eerie bell melody, menacing modern trap" 140 0
beat trap_melodic "melodic emotional trap beat, sad piano melody, deep 808, rolling hi-hats, atmospheric, modern" 130 0
beat trap_rage    "aggressive rage trap beat, hard hitting 808, fast hi-hats, dark synth lead, hype energy" 150 0
echo "ALLDONE"
