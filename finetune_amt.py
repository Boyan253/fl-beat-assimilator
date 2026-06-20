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

SEQ_LEN    = 2048   # tokens per training sequence (AMT context = 8192; 2048 fits comfortably)
STRIDE     = 1024   # 50% overlap between windows
BATCH_SIZE = 4      # per-GPU batch (24GB VRAM is plenty for 360M params at this seq len)
EPOCHS     = 3
LR         = 1e-5


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

    # 3. Load base AMT model
    print(f"Loading {BASE_MODEL}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    model.to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  {params/1e6:.0f}M params", flush=True)

    # 4. Dataset split
    dataset = Dataset.from_list(seqs)
    split   = dataset.train_test_split(test_size=0.1, seed=42)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"Train: {len(train_ds)}  Eval: {len(eval_ds)}", flush=True)

    # 5. Train
    args = TrainingArguments(
        output_dir=MODEL_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        warmup_steps=50,
        weight_decay=0.01,
        bf16=True,              # bf16 is more stable than fp16 on ROCm
        logging_steps=20,
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
