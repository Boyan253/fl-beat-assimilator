#!/bin/bash
# Generate N phonk riffs, ONE per fresh python process (avoids VRAM fragmentation that
# makes later riffs come back empty), then render every riff to MP3 for previewing.
#   bash gen_batch.sh <count> [--prompt]
# Output: D:\flbeat\data\generated\*.mid + *.mp3
set -u
export LD_LIBRARY_PATH=/opt/rocm/lib:${LD_LIBRARY_PATH:-}
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export HSA_ENABLE_SDMA=0
export GPU_MAX_HW_QUEUES=1
source /root/flbeat-venv/bin/activate
cd /mnt/d/flbeat/fl-beat-assimilator

COUNT=${1:-10}
PROMPT_FLAG=""
[ "${2:-}" = "--prompt" ] && PROMPT_FLAG="--prompt"

for n in $(seq 1 "$COUNT"); do
  echo "=== riff $n/$COUNT ==="
  python -u generate_phonk.py --count 1 --seed "$n" $PROMPT_FLAG 2>&1 \
    | grep -E "saved:|seed:|skip:|failed" || true
done

echo "=== rendering all to MP3 ==="
python -u render_audio.py 2>&1 | grep -E "ok:|FAIL|Done"
