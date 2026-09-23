import glob, torch
from transformers import AutoModelForCausalLM
from anticipation.sample import generate as amt_generate
from anticipation.convert import midi_to_events, EVENT_SIZE
from anticipation import ops

M = "/mnt/d/flbeat/data/models/phonk_amt"
model = AutoModelForCausalLM.from_pretrained(M, attn_implementation="eager").eval().cuda()
torch.manual_seed(1)

print("=== from-scratch (0->10s) ===")
ev = amt_generate(model, 0, 10, top_p=0.95)
print("events:", len(ev), "notes:", len(ev)//EVENT_SIZE)
if ev: print("first 12:", ev[:12])

print("\n=== prompt-continue (4s seed -> 12s) ===")
seed = sorted(glob.glob("/mnt/d/flbeat/data/midi_mt/*.mid"))[0]
inp = ops.clip(midi_to_events(seed), 0, 4, clip_duration=False, seconds=True)
print("seed:", seed.split("/")[-1], "prompt notes:", len(inp)//EVENT_SIZE)
ev2 = amt_generate(model, 4, 12, inputs=inp, top_p=0.95)
print("events:", len(ev2), "notes:", len(ev2)//EVENT_SIZE)
if ev2: print("first 12:", ev2[:12])
