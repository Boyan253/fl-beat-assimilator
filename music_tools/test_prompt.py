import glob, torch, pretty_midi
from transformers import AutoModelForCausalLM
from anticipation.sample import generate as amt_generate
from anticipation.convert import midi_to_events, events_to_midi, EVENT_SIZE
from anticipation import ops

M = "/mnt/d/flbeat/data/models/phonk_amt"
model = AutoModelForCausalLM.from_pretrained(M, attn_implementation="eager").eval().cuda()

def summary(events, label):
    p = "/tmp/_t.mid"
    events_to_midi(events).save(p)
    pm = pretty_midi.PrettyMIDI(p)
    parts = [("DRUMS" if i.is_drum else pretty_midi.program_to_instrument_name(i.program)) + f"({len(i.notes)})" for i in pm.instruments]
    print(f"  {label}: {pm.get_end_time():.1f}s  " + "  ".join(parts), flush=True)

seed = "/mnt/d/flbeat/data/midi_mt/DAT PHONK.mid"
full = midi_to_events(seed)
for sec in (4, 8, 12):
    summary(ops.clip(full, 0, sec, clip_duration=False, seconds=True), f"seed first {sec}s")

print("\n=== generate: prompt = seed 0-24s -> continue 24->40s ===", flush=True)
torch.manual_seed(3)
midprompt = ops.clip(full, 0, 24, clip_duration=False, seconds=True)
ev = amt_generate(model, 24, 40, inputs=midprompt, top_p=0.95)
summary(ev, "full(prompt+gen)")
gen_only = ops.clip(ev, 24, 40, clip_duration=False, seconds=True)
if gen_only and len(gen_only) >= EVENT_SIZE:
    summary(gen_only, "gen-only(>24s)")
else:
    print("  gen-only: EMPTY (model stopped at prompt)", flush=True)
