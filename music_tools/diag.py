import os, glob, json, torch
from transformers import AutoModelForCausalLM
from anticipation.convert import midi_to_events, VOCAB_SIZE

SNAP = glob.glob('/root/.cache/huggingface/hub/models--stanford-crfm--music-medium-800k/snapshots/*')[0]
print("=== config.json ===")
cfg = json.load(open(os.path.join(SNAP, "config.json")))
for k in ("architectures", "model_type", "vocab_size", "n_positions", "n_ctx", "n_embd", "n_layer", "loss_type"):
    print(f"  {k}: {cfg.get(k)}")
print("anticipation VOCAB_SIZE:", VOCAB_SIZE)

# token range in our data
midis = sorted(glob.glob('/mnt/d/flbeat/data/midi/*.mid'))[:30]
toks = []
for m in midis:
    try:
        toks.extend(midi_to_events(m))
    except Exception:
        pass
print(f"\n=== our tokens (from {len(midis)} files) ===")
print("  count:", len(toks), " min:", min(toks), " max:", max(toks))
print("  any >= model vocab_size?", max(toks) >= cfg.get("vocab_size", 0))

# single forward pass on the base model with a 1024-chunk
print("\n=== base-model forward pass ===")
model = AutoModelForCausalLM.from_pretrained(SNAP).cuda().eval()
chunk = toks[:1024]
ids = torch.tensor([chunk], dtype=torch.long, device="cuda")
with torch.no_grad():
    out = model(input_ids=ids, labels=ids)
print("  loss (labels=input_ids):", float(out.loss))
print("  logits shape:", tuple(out.logits.shape), " (last dim should == vocab_size)")
print("  logits finite?", torch.isfinite(out.logits).all().item())
