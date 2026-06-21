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

SECONDS        = 15     # length of each generated riff (AMT sampler is time-based)
TOP_P          = 0.95   # nucleus sampling: creative but coherent


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

    # eager attention: sdpa/flash is experimental + unstable on Navi31 (RX 7900 XTX)
    model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation="eager")
    model.eval().to(device)

    # AMT has its own autoregressive sampler — raw HF generate() does not produce valid
    # event triples. sample.generate(model, 0, SECONDS, top_p) returns proper AMT events.
    from anticipation.sample import generate as amt_generate
    from anticipation.convert import EVENT_SIZE

    torch.manual_seed(args.seed)
    generated = []

    for i in range(args.count):
        print(f"Generating riff {i+1}/{args.count}...", flush=True)
        try:
            events = amt_generate(model, 0, SECONDS, top_p=TOP_P)
        except Exception as e:
            print(f"  generation failed: {e}", flush=True)
            continue

        if not events or len(events) < EVENT_SIZE:
            print(f"  skip: empty generation", flush=True)
            continue

        out_path = os.path.join(GENERATED_DIR, f"riff_{args.seed}_{i+1:03d}.mid")
        try:
            events_to_midi_file(events, out_path)
            print(f"  saved: {os.path.basename(out_path)}  (~{len(events)//EVENT_SIZE} notes)", flush=True)
            generated.append(out_path)
        except Exception as e:
            print(f"  midi conversion failed: {e}", flush=True)

    print(f"\n=== Generated {len(generated)}/{args.count} riffs in {GENERATED_DIR} ===")
    print("Import the .mid files into FL Studio and listen — pick the ones that sound good.")


if __name__ == "__main__":
    main()
