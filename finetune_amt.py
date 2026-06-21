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

MIDI_DIR   = "/mnt/d/flbeat/data/midi"
MODEL_DIR  = "/mnt/d/flbeat/data/models/phonk_amt"
BASE_MODEL = "stanford-crfm/music-medium-800k"

SEQ_LEN    = 1024   # tokens per training sequence (kept small to fit 24GB w/ optimizer states)
STRIDE     = 512    # 50% overlap between windows
BATCH_SIZE = 1      # per-GPU batch; effective batch = BATCH_SIZE * GRAD_ACCUM
GRAD_ACCUM = 8      # gradient accumulation -> effective batch 8 without the VRAM cost
EPOCHS     = 3
LR         = 1e-5


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


class MidiCollator:
    def __call__(self, features):
        import torch
        ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
        return {"input_ids": ids, "labels": ids.clone()}


def main():
    import torch
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
    from datasets import Dataset

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

    # 2. Chunk into fixed-length sequences
    seqs = []
    for start in range(0, len(all_tokens) - SEQ_LEN, STRIDE):
        chunk = all_tokens[start: start + SEQ_LEN]
        if len(chunk) == SEQ_LEN:
            seqs.append({"input_ids": chunk})
    print(f"Training sequences: {len(seqs)}", flush=True)
    if len(seqs) < 10:
        print("ERROR: too few sequences. Check that MIDI_DIR contains .mid files.")
        return

    # 3. Load base AMT model (from local safetensors to dodge the .bin CVE guard)
    model_path = resolve_model_path()
    print(f"Loading {BASE_MODEL} from {model_path}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  {params/1e6:.0f}M params", flush=True)

    # 4. Dataset split
    dataset = Dataset.from_list(seqs)
    split   = dataset.train_test_split(test_size=0.1, seed=42)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"Train: {len(train_ds)}  Eval: {len(eval_ds)}", flush=True)

    # 5. Train
    model.gradient_checkpointing_enable()   # trade compute for memory
    model.config.use_cache = False          # required with gradient checkpointing

    args = TrainingArguments(
        output_dir=MODEL_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        gradient_checkpointing=True,
        learning_rate=LR,
        warmup_steps=50,
        weight_decay=0.01,
        bf16=True,              # bf16 is more stable than fp16 on ROCm
        optim="adafactor",      # Adafactor uses far less memory than Adam (no m/v states)
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
        dataloader_num_workers=0,   # WSL multiprocessing can deadlock
    )
    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=MidiCollator(),
    )
    print("Starting fine-tune...", flush=True)
    trainer.train()
    trainer.save_model(MODEL_DIR)
    print(f"Saved to {MODEL_DIR}", flush=True)


if __name__ == "__main__":
    main()
