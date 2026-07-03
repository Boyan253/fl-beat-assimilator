#!/bin/bash
LOG=/opt/rave_data/GRIM/train.log
i=0
until grep -qE "Epoch|it/s|Sanity|Validation|Traceback|Error|FATAL|Killed|RuntimeError|HSA|hipError|assert|step=|loss" "$LOG" 2>/dev/null || [ "$i" -ge 50 ]; do
  sleep 5
  i=$((i + 1))
done
echo "=== train.log tail (after ~$((i * 5))s) ==="
tail -18 "$LOG" 2>/dev/null
echo "=== checkpoints so far ==="
find /opt/rave_data/GRIM -name "*.ckpt" 2>/dev/null | head
