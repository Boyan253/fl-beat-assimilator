import glob, torch
from transformers import AutoModelForCausalLM
from anticipation.convert import midi_to_events

SNAP = glob.glob('/root/.cache/huggingface/hub/models--stanford-crfm--music-medium-800k/snapshots/*')[0]

# build several distinct 1024-chunks so we're not overfitting one window
toks = []
for m in sorted(glob.glob('/mnt/d/flbeat/data/midi/*.mid'))[:40]:
    try: toks.extend(midi_to_events(m) + [0])
    except Exception: pass
chunks = [toks[i:i+1024] for i in range(0, len(toks)-1024, 1024)][:30]
print("chunks:", len(chunks))

model = AutoModelForCausalLM.from_pretrained(SNAP, attn_implementation="eager").cuda().train()
opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
for step, ch in enumerate(chunks):
    ids = torch.tensor([ch], dtype=torch.long, device="cuda")
    opt.zero_grad()
    out = model(input_ids=ids, labels=ids)
    loss = out.loss
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    bad = (not torch.isfinite(loss)) or (not torch.isfinite(gn))
    if step % 3 == 0 or bad:
        print(f"  step {step:2d}: loss={float(loss):.4f}  grad_norm={float(gn):.2f}{'  <<< NaN/inf!' if bad else ''}")
    if bad:
        print("DIVERGED"); break
else:
    print("STABLE for", len(chunks), "steps")
