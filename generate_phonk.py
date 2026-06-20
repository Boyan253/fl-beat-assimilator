"""generate_phonk.py — generate original phonk riffs from the fine-tuned AMT model.

Generates N MIDI riffs and saves them to GENERATED_DIR.
Cherry-pick the ones that sound good -> import into FL Studio.

Run in WSL flbeat-venv:
  source /root/flbeat-venv/bin/activate
  python /mnt/d/flbeat/fl-beat-assimilator/generate_phonk.py [--count 10] [--seed 42]

After generation, listen to each .mid in FL Studio (load any instrument).
The best ones are the ones that sound melodic and not chaotic.
"""
import os, argparse, warnings
warnings.filterwarnings("ignore")

MODEL_DIR     = "/mnt/d/flbeat/data/models/phonk_amt"
GENERATED_DIR = "/mnt/d/flbeat/data/generated"
BASE_MODEL    = "stanford-crfm/music-medium-800k"

MAX_NEW_TOKENS = 1024   # ~30-60 notes per riff
TEMPERATURE    = 0.92   # slightly below 1.0: creative but not chaotic
TOP_P          = 0.95


def events_to_midi_file(events, out_path):
    """Convert AMT event integers to a MIDI file."""
    from anticipation.convert import events_to_midi
    mid = events_to_midi(events)
    mid.save(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10, help="Number of riffs to generate")
    parser.add_argument("--seed",  type=int, default=42,  help="Random seed")
    args = parser.parse_args()

    os.makedirs(GENERATED_DIR, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load fine-tuned model (fall back to base if fine-tune not ready)
    if os.path.exists(os.path.join(MODEL_DIR, "config.json")):
        model_path = MODEL_DIR
        print(f"Loading fine-tuned model from {MODEL_DIR}...", flush=True)
    else:
        model_path = BASE_MODEL
        print(f"Fine-tuned model not found — using base AMT ({BASE_MODEL})", flush=True)

    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval().to(device)

    torch.manual_seed(args.seed)
    generated = []

    for i in range(args.count):
        print(f"Generating riff {i+1}/{args.count}...", flush=True)
        # Start from a single silence token (seed the generation)
        input_ids = torch.tensor([[0]], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
        tokens = out[0].tolist()
        # AMT events are in groups of EVENT_SIZE=3; trim to multiple of 3
        from anticipation.convert import EVENT_SIZE
        trim = (len(tokens) // EVENT_SIZE) * EVENT_SIZE
        events = tokens[:trim]

        if len(events) < 6:
            print(f"  skip: too short ({len(events)} tokens)", flush=True)
            continue

        out_path = os.path.join(GENERATED_DIR, f"riff_{args.seed}_{i+1:03d}.mid")
        try:
            events_to_midi_file(events, out_path)
            notes_approx = len(events) // EVENT_SIZE
            print(f"  saved: {os.path.basename(out_path)}  (~{notes_approx} events)", flush=True)
            generated.append(out_path)
        except Exception as e:
            print(f"  midi conversion failed: {e}", flush=True)

    print(f"\n=== Generated {len(generated)}/{args.count} riffs in {GENERATED_DIR} ===")
    print("Import the .mid files into FL Studio and listen — pick the ones that sound good.")


if __name__ == "__main__":
    main()
