"""train_phonk_lokr.py — headless LoKr fine-tune of ACE-Step on phonk (educational test).

scan dataset -> preprocess to tensors -> train LoKr adapter. Produces a phonk adapter
you load at generation time to bias output toward the trained style.
"""
import os, sys
from acestep.handler import AceStepHandler
from acestep.training.dataset_builder_modules.builder import DatasetBuilder
from acestep.training.configs import LoKRConfig, TrainingConfig
from acestep.training.trainer import LoKRTrainer

REPO    = "/mnt/d/flbeat/ACE-Step-1.5"   # safe-root is hardwired to the repo dir
DATASET = "/mnt/d/flbeat/ACE-Step-1.5/lora_phonk"
TENSORS = "/mnt/d/flbeat/ACE-Step-1.5/lora_phonk_tensors"
OUT     = "/mnt/d/flbeat/ACE-Step-1.5/lora_phonk_out"
EPOCHS  = int(sys.argv[1]) if len(sys.argv) > 1 else 200

print("=== init DiT (loads VAE/text-enc/DiT) ===", flush=True)
dit = AceStepHandler()
dit.initialize_service(project_root=REPO, config_path="acestep-v15-turbo", device="cuda",
                       use_flash_attention=False, use_mlx_dit=False)

print("=== scan dataset ===", flush=True)
builder = DatasetBuilder()
_, status = builder.scan_directory(DATASET)
print("scan:", status, "| samples:", builder.get_sample_count(), flush=True)

print("=== preprocess -> tensors ===", flush=True)
builder.preprocess_to_tensors(dit_handler=dit, output_dir=TENSORS, preprocess_mode="lokr")
print("preprocess done", flush=True)

print(f"=== train LoKr ({EPOCHS} epochs) ===", flush=True)
lokr = LoKRConfig()
tc = TrainingConfig(output_dir=OUT, max_epochs=EPOCHS, num_workers=0,
                    save_every_n_epochs=50, persistent_workers=False)
trainer = LoKRTrainer(dit_handler=dit, lokr_config=lokr, training_config=tc)
last = None
for step, loss, st in trainer.train_from_preprocessed(TENSORS):
    last = (step, loss, st)
    if (isinstance(step, int) and step % 25 == 0) or "epoch" in str(st).lower() or "✅" in str(st) or "❌" in str(st):
        print(f"step {step} loss {loss} | {st}", flush=True)
print("LAST:", last, flush=True)
print("TRAINING DONE ->", OUT, flush=True)
