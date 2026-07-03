#!/bin/bash
SRC=/root/flbeat-venv/lib/python3.12/site-packages
DST=/opt/rave-env/lib/python3.12/site-packages
rm -rf "$DST"/torch "$DST"/torchaudio "$DST"/torch.libs "$DST"/torchaudio.libs "$DST"/torchgen "$DST"/functorch "$DST"/torch-*.dist-info "$DST"/torchaudio-*.dist-info
cp -r "$SRC"/torch "$DST"/
cp -r "$SRC"/torchaudio "$DST"/
for d in torch.libs torchaudio.libs torchgen functorch; do [ -e "$SRC/$d" ] && cp -r "$SRC/$d" "$DST"/; done
cp -r "$SRC"/torch-*.dist-info "$DST"/ 2>/dev/null
cp -r "$SRC"/torchaudio-*.dist-info "$DST"/ 2>/dev/null
echo "=== copied flbeat torch -> rave-env, testing GPU ==="
export LD_LIBRARY_PATH=/opt/rocm/lib HSA_ENABLE_SDMA=0 PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
/opt/rave-env/bin/python /mnt/d/flbeat/fl-beat-assimilator/check_torch.py
