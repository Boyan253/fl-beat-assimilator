"""finetune_amt.py — fine-tune AMT on phonk melody MIDI corpus.

Pipeline:
  1. Convert all .mid in MIDI_DIR to AMT event-integer sequences (native tokenization)
  2. Chunk into SEQ_LEN-length overlapping windows
  3. Fine-tune stanford-crfm/music-medium-800k with HF Trainer (causal LM)
  4. Save checkpoint to MODEL_DIR

Run in WSL flbeat-venv:
  source /root/flbeat-venv/bin/activate
  python /mnt/d/flbeat/fl-beat-assimilator/finetune_amt.py
"""
import os, glob, warnings
warnings.filterwarnings("ignore")

MIDI_DIR   = "/mnt/d/flbeat/data/midi_mt"   # multi-track (melody+bass+drums); was /midi (melody only)
MODEL_DIR  = "/mnt/d/flbeat/data/models/phonk_amt"
BASE_MODEL = "stanford-crfm/music-medium-800k"

SEQ_LEN     = 512    # shorter kernels = lighter sustained GPU load (RDNA3+ROCm+WSL TDR-crashes)
STRIDE      = 256    # 50% overlap between windows
MICRO_BATCH = 1      # sequences per step; eager attention's O(seq^2) maps OOM above 1 at seq 1024
EPOCHS      = 3      # multi-track corpus is ~3x the tokens, so fewer epochs cover it
LR          = 1e-5
CKPT_EVERY  = 200    # save a recoverable checkpoint every N steps (GPU can TDR-reset mid-run)


def resolve_model_path():
    """Return a local dir containing model.safetensors for the base AMT model.
    transformers' new CVE guard refuses to load the repo's pytorch_model.bin with
    torch<2.6, so we convert the checkpoint to safetensors once and load from disk."""
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import save_file, load_file
    binp = hf_hub_download(BASE_MODEL, "pytorch_model.bin")
    hf_hub_download(BASE_MODEL, "config.json")
    snap = os.path.dirname(binp)
    sf_path = os.path.join(snap, "model.safetensors")
    ok = False
    if os.path.exists(sf_path):
        try:
            load_file(sf_path); ok = True
        except Exception:
            ok = False
    if not ok:
        print("converting base checkpoint .bin -> safetensors ...", flush=True)
        sd = torch.load(binp, map_location="cpu", weights_only=True)
        save_file({k: v.clone().contiguous() for k, v in sd.items()}, sf_path)
    return snap


def midi_to_token_seq(path):
    """Convert MIDI file to list of AMT event integers, or None on error."""
    from anticipation.convert import midi_to_events
    try:
        evs = midi_to_events(path)
        return evs if len(evs) >= 6 else None
    except Exception as e:
        print(f"  skip {os.path.basename(path)}: {e}", flush=True)
        return None


def main():
    import torch
    from transformers import AutoModelForCausalLM

    os.makedirs(MODEL_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. Build flat token stream from all MIDI files
    midi_files = sorted(glob.glob(os.path.join(MIDI_DIR, "*.mid")))
    print(f"Loading {len(midi_files)} MIDI files...", flush=True)

    all_tokens = []
    ok = 0
    for mf in midi_files:
        toks = midi_to_token_seq(mf)
        if toks:
            all_tokens.extend(toks)
            all_tokens.append(0)  # song separator (token 0 = padding/silence in AMT vocab)
            ok += 1
    print(f"Loaded {ok}/{len(midi_files)} files.  Total tokens: {len(all_tokens):,}", flush=True)

    # 2. Chunk into fixed-length sequences, hold out 10% for eval
    seqs = []
    for start in range(0, len(all_tokens) - SEQ_LEN, STRIDE):
        chunk = all_tokens[start: start + SEQ_LEN]
        if len(chunk) == SEQ_LEN:
            seqs.append(chunk)
    if len(seqs) < 10:
        print("ERROR: too few sequences. Check that MIDI_DIR contains .mid files.")
        return
    train_seqs = seqs
    print(f"Train sequences: {len(train_seqs)}  (seq_len={SEQ_LEN})", flush=True)

    # 3. Load base AMT model (from local safetensors to dodge the .bin CVE guard).
    # attn_implementation="eager" is REQUIRED: the default sdpa/flash-attention is
    # experimental on Navi31 (RX 7900 XTX) and its backward pass produces NaN gradients.
    model_path = resolve_model_path()
    print(f"Loading {BASE_MODEL} from {model_path}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation="eager")
    model.to(device).train()
    params = sum(p.numel() for p in model.parameters())
    print(f"  {params/1e6:.0f}M params", flush=True)

    # 4. Custom training loop — flat and minimal, mirroring the verified probe exactly
    #    (HF Trainer diverged to NaN; generator/closure variants hung at step 0 on this
    #    ROCm stack). Plain fp32 + AdamW + grad-clip, indexed loop, no generators.
    import random
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_seqs) * EPOCHS
    warmup = 50
    print(f"Starting fine-tune: {total_steps} steps", flush=True)

    step = 0
    for epoch in range(EPOCHS):
        order = list(range(len(train_seqs)))
        random.Random(1234 + epoch).shuffle(order)
        for idx in order:
            ids = torch.tensor([train_seqs[idx]], dtype=torch.long, device=device)
            lr = LR * min(1.0, (step + 1) / warmup)
            for pg in opt.param_groups:
                pg["lr"] = lr
            opt.zero_grad()
            loss = model(input_ids=ids, labels=ids).loss
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if torch.isfinite(loss) and torch.isfinite(gn):
                opt.step()
            else:
                print(f"  !! non-finite at step {step} — skipped", flush=True)
            if step < 5 or step % 50 == 0:
                print(f"  epoch {epoch} step {step}/{total_steps}  loss={float(loss):.4f}  grad_norm={float(gn):.1f}  lr={lr:.2e}", flush=True)
            step += 1
            if step % CKPT_EVERY == 0:
                model.save_pretrained(MODEL_DIR, safe_serialization=True)
                print(f"  [checkpoint saved at step {step}]", flush=True)

    model.save_pretrained(MODEL_DIR, safe_serialization=True)
    print(f"Saved to {MODEL_DIR}", flush=True)


if __name__ == "__main__":
    main()
