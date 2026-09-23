import glob, time, torch
from transformers import AutoModelForCausalLM
from anticipation.convert import midi_to_events

SNAP = glob.glob('/root/.cache/huggingface/hub/models--stanford-crfm--music-medium-800k/snapshots/*')[0]

t0 = time.time()
toks = []
for m in sorted(glob.glob('/mnt/d/flbeat/data/midi/*.mid')):
    try: toks.extend(midi_to_events(m) + [0])
    except Exception: pass
print(f"[{time.time()-t0:.1f}s] tokenized all 246 files, {len(toks)} tokens", flush=True)

chunks = [toks[i:i+1024] for i in range(0, len(toks)-1024, 512)]
print(f"chunks: {len(chunks)}", flush=True)

print(f"[{time.time()-t0:.1f}s] loading model...", flush=True)
model = AutoModelForCausalLM.from_pretrained(SNAP, attn_implementation="eager").cuda().train()
print(f"[{time.time()-t0:.1f}s] model on cuda. param device: {next(model.parameters()).device}", flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=1e-5)

for step in range(4):
    ts = time.time()
    ids = torch.tensor([chunks[step]], dtype=torch.long, device="cuda")
    out = model(input_ids=ids, labels=ids)
    loss = out.loss
    print(f"  step {step}: forward done in {time.time()-ts:.1f}s, loss={float(loss):.4f}, loss.device={loss.device}", flush=True)
    tb = time.time()
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.zero_grad()
    torch.cuda.synchronize()
    print(f"  step {step}: backward+step in {time.time()-tb:.1f}s, grad_norm={float(gn):.1f}, total step {time.time()-ts:.1f}s", flush=True)
print("DONE", flush=True)
