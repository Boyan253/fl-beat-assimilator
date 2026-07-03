#!/bin/bash
# Wait until RAVE saves its first checkpoint (then a model can be exported), then notify.
until find /opt/rave_data/GRIM -name "*.ckpt" 2>/dev/null | grep -q .; do
  # also bail out if training died (no python process) so we don't wait forever
  if ! pgrep -f "rave-env/bin/rave" >/dev/null 2>&1; then
    echo "=== training process gone (check train.log) ==="
    tail -8 /opt/rave_data/GRIM/train.log 2>/dev/null
    exit 0
  fi
  sleep 120
done
echo "=== first checkpoint saved! ==="
find /opt/rave_data/GRIM -name "*.ckpt" -exec ls -la {} +
echo "=== latest training step ==="
grep -oE "Epoch [0-9]+:" /opt/rave_data/GRIM/train.log 2>/dev/null | tail -1
