"""generate_phonk.py — generate phonk riffs from the fine-tuned AMT model.

Two modes:
  from-scratch (default): generate with no prompt. More "original" but drifts toward
      AMT's generic prior — can sound un-phonk.
  prompt-continue (--prompt): seed the model with the first few seconds of a REAL phonk
      MIDI from the training set, then let it CONTINUE in that style. Far more in-style.
      NOTE: the prompt portion is borrowed from a copyrighted song — use --prompt only to
      AUDITION the model's style. For sellable output use from-scratch (or strip the prompt).

Run in WSL flbeat-venv (with the GPU stability env vars — see run_finetune.sh):
  python generate_phonk.py [--count 10] [--seed 42] [--prompt] [--seconds 15]

After generation, render to MP3 with render_audio.py to preview without FL.
"""
import os, glob, random, argparse, warnings
warnings.filterwarnings("ignore")

MODEL_DIR     = "/mnt/d/flbeat/data/models/phonk_amt"
GENERATED_DIR = "/mnt/d/flbeat/data/generated"
MIDI_DIR      = "/mnt/d/flbeat/data/midi_mt"   # multi-track training MIDIs used as style prompts
BASE_MODEL    = "stanford-crfm/music-medium-800k"

# The multi-track model won't cold-start; it needs a RICH prompt (the full beat, not just
# the intro) to continue bass+drums+melody. We prompt with PROMPT_SECONDS of a real song,
# generate to PROMPT_SECONDS+SECONDS, then STRIP the prompt so the saved riff is the model's
# own original full arrangement (the seed only primes it, it's not in the output).
PROMPT_SECONDS = 24     # seconds of seed context (must cover the full beat, not just intro)
SECONDS        = 16     # seconds of ORIGINAL continuation to keep
TOP_P          = 0.95   # nucleus sampling


def events_to_midi_file(events, out_path):
    from anticipation.convert import events_to_midi
    events_to_midi(events).save(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count",   type=int,   default=10)
    parser.add_argument("--seed",    type=int,   default=42)
    parser.add_argument("--prompt",  action="store_true", help="continue a real phonk seed (in-style)")
    parser.add_argument("--seconds", type=int,   default=SECONDS)
    parser.add_argument("--top-p",   type=float, default=TOP_P)
    args = parser.parse_args()

    os.makedirs(GENERATED_DIR, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM
    from anticipation.sample import generate as amt_generate
    from anticipation.convert import midi_to_events, EVENT_SIZE
    from anticipation import ops

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model_path = MODEL_DIR if os.path.exists(os.path.join(MODEL_DIR, "config.json")) else BASE_MODEL
    print(f"Loading model from {model_path}...", flush=True)
    # eager attention: sdpa/flash is experimental + unstable on Navi31 (RX 7900 XTX)
    model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation="eager")
    model.eval().to(device)

    seeds = sorted(glob.glob(os.path.join(MIDI_DIR, "*.mid"))) if args.prompt else []
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    tag = "cont" if args.prompt else "scratch"
    generated = []

    end_time = PROMPT_SECONDS + args.seconds
    for i in range(args.count):
        print(f"Generating riff {i+1}/{args.count} ({tag})...", flush=True)
        try:
            if args.prompt and seeds:
                seed_midi = rng.choice(seeds)
                # Prompt with the full beat (PROMPT_SECONDS, not just the intro) so the model
                # has bass+drums+melody context, generate the continuation, then STRIP the
                # prompt so the saved riff is the model's OWN original arrangement.
                inputs = ops.clip(midi_to_events(seed_midi), 0, PROMPT_SECONDS,
                                  clip_duration=False, seconds=True)
                full = amt_generate(model, PROMPT_SECONDS, end_time, inputs=inputs, top_p=args.top_p)
                events = ops.clip(full, PROMPT_SECONDS, end_time, clip_duration=False, seconds=True)
                # shift the kept events back to t=0 (remove the prompt's leading silence)
                if events:
                    events = ops.translate(events, -PROMPT_SECONDS, seconds=True)
                print(f"  seed: {os.path.basename(seed_midi)[:50]}", flush=True)
            else:
                events = amt_generate(model, 0, args.seconds, top_p=args.top_p)
        except Exception as e:
            print(f"  generation failed: {e}", flush=True)
            continue

        if not events or len(events) < EVENT_SIZE:
            print(f"  skip: empty generation", flush=True)
            continue

        out_path = os.path.join(GENERATED_DIR, f"riff_{tag}_{args.seed}_{i+1:03d}.mid")
        try:
            events_to_midi_file(events, out_path)
            print(f"  saved: {os.path.basename(out_path)}  (~{len(events)//EVENT_SIZE} notes)", flush=True)
            generated.append(out_path)
        except Exception as e:
            print(f"  midi conversion failed: {e}", flush=True)

    print(f"\n=== Generated {len(generated)}/{args.count} riffs in {GENERATED_DIR} ===")
    print("Render to MP3:  python render_audio.py")


if __name__ == "__main__":
    main()
