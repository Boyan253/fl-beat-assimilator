import glob, torch
from transformers import AutoModelForCausalLM
from anticipation.convert import midi_to_events

SNAP = glob.glob('/root/.cache/huggingface/hub/models--stanford-crfm--music-medium-800k/snapshots/*')[0]

toks = []
for m in sorted(glob.glob('/mnt/d/flbeat/data/midi/*.mid'))[:10]:
    try: toks.extend(midi_to_events(m))
    except Exception: pass

def run(attn, ckpt):
    print(f"\n=== attn={attn}  grad_checkpoint={ckpt} ===")
    model = AutoModelForCausalLM.from_pretrained(SNAP, attn_implementation=attn).cuda().train()
    if ckpt:
        model.gradient_checkpointing_enable(); model.config.use_cache = False
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    ids = torch.tensor([toks[:1024]], dtype=torch.long, device="cuda")
    for step in range(5):
        opt.zero_grad()
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        print(f"  step {step}: loss={float(out.loss):.4f}  grad_norm={float(gn):.4f}")
    del model, opt; torch.cuda.empty_cache()

run("eager", False)
