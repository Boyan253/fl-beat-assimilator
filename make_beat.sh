#!/bin/bash
# One command: prompt -> finished phonk track -> full EXCLUSIVE trackout pack.
#   bash make_beat.sh "<prompt>" <name>
# Output: D:\flbeat\data\exclusive_pack_<name>\  (stems + MIDI + master)
PROMPT="${1:-dark drift phonk, distorted 808, cowbell melody, hard memphis phonk, lo-fi}"
NAME="${2:-beat}"
GEN=/mnt/d/flbeat/data/generated
PACK=/mnt/d/flbeat/data/exclusive_pack_${NAME}

echo "########## 1/2 GENERATE (ACE-Step) ##########"
# ACE-Step lives in /opt/aceenv; generate a WAV master
bash /mnt/d/flbeat/data/run_ace_gen.sh "$PROMPT" "$NAME" wav
TRACK=${GEN}/${NAME}.wav
if [ ! -f "$TRACK" ]; then echo "GENERATION FAILED (no $TRACK)"; exit 1; fi
echo "generated: $TRACK"

echo "########## 2/2 TRACKOUT (demucs + MIDI) ##########"
# stem-split + transcribe lives in /root/flbeat-venv
export LD_LIBRARY_PATH=/opt/rocm/lib HSA_ENABLE_SDMA=0
source /root/flbeat-venv/bin/activate
python -u /mnt/d/flbeat/fl-beat-assimilator/build_trackout.py "$TRACK" "$PACK"
echo "########## DONE -> $PACK ##########"
