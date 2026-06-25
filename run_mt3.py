"""run_mt3.py — transcribe a stem with an MT3-family model (yourmt3 / mt3_pytorch / mr_mt3).
  python run_mt3.py <stem.wav> <out.mid> [model=yourmt3]
"""
import sys, types, warnings
from math import ceil
warnings.filterwarnings("ignore")

# Shim: newer transformers dropped transformers.utils.model_parallel_utils, which YourMT3
# imports. Provide the two functions (no-op for single-GPU inference).
if "transformers.utils.model_parallel_utils" not in sys.modules:
    _m = types.ModuleType("transformers.utils.model_parallel_utils")
    def assert_device_map(device_map, num_blocks):  # noqa
        return
    def get_device_map(n_layers, devices):  # noqa
        n = int(ceil(n_layers / len(devices))); ls = list(range(n_layers))
        return dict(zip(devices, [ls[i:i + n] for i in range(0, n_layers, n)]))
    _m.assert_device_map = assert_device_map; _m.get_device_map = get_device_map
    sys.modules["transformers.utils.model_parallel_utils"] = _m

import librosa, mt3_infer

stem = sys.argv[1]; out = sys.argv[2]
model = sys.argv[3] if len(sys.argv) > 3 else "yourmt3"
y, _ = librosa.load(stem, sr=16000, mono=True)
print(f"transcribing {stem} with {model} ...", flush=True)
midi = mt3_infer.transcribe(y, model=model, sr=16000, device="cuda")
midi.save(out)
n = sum(1 for tr in midi.tracks for m in tr if m.type == "note_on" and m.velocity > 0)
print(f"MODEL={model}  notes={n}  -> {out}", flush=True)
