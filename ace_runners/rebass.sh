#!/bin/bash
source /root/flbeat-venv/bin/activate
G=/mnt/d/flbeat/data/generated
OUT=$G/phonk_more
for n in phonk_chase phonk_doom phonk_melody2 phonk_rave phonk_ghost; do
  echo "### $n ###"
  python -u /mnt/d/flbeat/fl-beat-assimilator/phonkify.py "$G/$n.wav" "$OUT/${n}_BASS.wav" 1.2 2>&1 | grep -E "phonkified|Error"
done
echo "ALLDONE"
