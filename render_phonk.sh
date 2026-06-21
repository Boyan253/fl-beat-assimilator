#!/bin/bash
# Render every generated multi-track riff through real phonk samples.
#   bash render_phonk.sh [source_song]
source /root/flbeat-venv/bin/activate
SRC="${1:-COWBELL WARRIOR!}"
for f in /mnt/d/flbeat/data/generated/riff_cont_*.mid; do
  base=$(basename "$f" .mid)
  out="/mnt/d/flbeat/data/generated/PHONK_${base}.wav"
  python -u /mnt/d/flbeat/fl-beat-assimilator/sample_render.py "$f" "$SRC" "$out" 2>&1 | grep -E "wrote|missing"
done
echo ALLDONE
