"""Validate the Anticipatory Music Transformer: load model, generate a short MIDI."""
import sys, warnings
warnings.filterwarnings("ignore")
import torch
from transformers import AutoModelForCausalLM
from anticipation.sample import generate
from anticipation.convert import events_to_midi

OUT = sys.argv[1] if len(sys.argv) > 1 else "/mnt/d/flbeat/data/phonk_midi/AMT_test.mid"
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("loading AMT (music-medium-800k)...", flush=True)
model = AutoModelForCausalLM.from_pretrained("stanford-crfm/music-medium-800k").to(dev).eval()
print("generating 10s...", flush=True)
events = generate(model, start_time=0, end_time=10, top_p=0.98)
mid = events_to_midi(events)
mid.save(OUT)   # mido MidiFile uses .save()
nnotes = sum(1 for tr in mid.tracks for m in tr if m.type == "note_on" and m.velocity > 0)
print(f"OK: {len(events)} events -> {len(mid.tracks)} tracks, {nnotes} notes -> {OUT}", flush=True)
