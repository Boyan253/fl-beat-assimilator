"""bench_melody.py — YourMT3 (SOTA) vs Basic Pitch on synthetic GT, melody-focused.
Answers objectively: is YourMT3 better than Basic Pitch for the melody/polyphonic case?

  python bench_melody.py [out_dir]
"""
import sys, os, types, subprocess, warnings
from math import ceil
warnings.filterwarnings("ignore")

# transformers compat shim for YourMT3
_m = types.ModuleType("transformers.utils.model_parallel_utils")
_m.assert_device_map = lambda *a, **k: None
def _gdm(n_layers, devices):
    n = int(ceil(n_layers / len(devices))); ls = list(range(n_layers))
    return dict(zip(devices, [ls[i:i + n] for i in range(0, n_layers, n)]))
_m.get_device_map = _gdm
sys.modules["transformers.utils.model_parallel_utils"] = _m

import numpy as np, librosa, pretty_midi, mir_eval, mt3_infer

OUT = sys.argv[1] if len(sys.argv) > 1 else "/mnt/d/flbeat/data/bench"
os.makedirs(OUT, exist_ok=True)
SF2 = "/usr/share/sounds/sf2/FluidR3_GM.sf2"; SR = 22050

def mk_midi(notes, program, path, bpm=140):
    pm = pretty_midi.PrettyMIDI(initial_tempo=float(bpm)); ins = pretty_midi.Instrument(program=program)
    for (s, e, p) in notes: ins.notes.append(pretty_midi.Note(100, p, s, e))
    pm.instruments.append(ins); pm.write(path)

def scale_riff(root, dur, bpm, n=24):
    spb = 60.0 / bpm; step = spb * dur; deg = [0, 2, 3, 5, 7, 10, 12]; out = []
    for i in range(n):
        p = root + deg[(i * 3) % len(deg)] + 12 * ((i // 7) % 2)
        s = i * step; out.append((s, s + step * 0.9, int(p)))
    return out

def chord_prog(root, bpm, n=8):
    spb = 60.0 / bpm; step = spb * 2; out = []
    for i in range(n):
        base = root + [0, 5, 3, 7][i % 4]; s = i * step
        for iv in (0, 3, 7): out.append((s, s + step * 0.95, int(base + iv)))
    return out

GTS = {  # melody-relevant cases
    "mono_lead_mid":  (scale_riff(60, 0.5, 140), 81, 140),
    "mono_lead_fast": (scale_riff(67, 0.25, 150), 81, 150),
    "poly_chords":    (chord_prog(48, 130), 4, 130),  # THE polyphonic-melody case
}
TIMBRES = [("sawlead", 81), ("epiano", 4)]

def render(midi_path, program, wav):
    pm = pretty_midi.PrettyMIDI(midi_path)
    for ins in pm.instruments: ins.program = program
    tmp = wav.replace(".wav", "_t.mid"); pm.write(tmp)
    subprocess.run(["fluidsynth", "-ni", "-g", "1.0", "-F", wav, "-r", str(SR), SF2, tmp],
                   check=True, capture_output=True); os.remove(tmp)

def notes_from_pm(pm): return [(n.start, n.end, n.pitch) for i in pm.instruments for n in i.notes]

def t_basicpitch(wav):
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH
    _, midi, _ = predict(wav, model_or_model_path=ICASSP_2022_MODEL_PATH,
                         onset_threshold=0.5, frame_threshold=0.3, minimum_note_length=70)
    return notes_from_pm(midi)

def t_yourmt3(wav):
    y, _ = librosa.load(wav, sr=16000, mono=True)
    midi = mt3_infer.transcribe(y.astype("float32"), model="yourmt3", sr=16000, device="cuda")
    tmp = wav.replace(".wav", "_ymt3.mid"); midi.save(tmp)
    notes = notes_from_pm(pretty_midi.PrettyMIDI(tmp)); os.remove(tmp); return notes

def to_arr(notes):
    if not notes: return np.zeros((0, 2)), np.zeros(0)
    iv = np.array([[s, max(e, s + 1e-3)] for s, e, _ in notes])
    pi = np.array([440.0 * 2 ** ((p - 69) / 12.0) for _, _, p in notes]); return iv, pi

def score(ref, est):
    ri, rp = to_arr(ref); ei, ep = to_arr(est)
    if len(ei) == 0: return dict(onF=0, opF=0, ncr=0)
    _, _, onF = mir_eval.transcription.onset_precision_recall_f1(ri, ei, onset_tolerance=0.05)
    _, _, opF, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ri, rp, ei, ep, onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None)
    return dict(onF=onF, opF=opF, ncr=len(ei) / len(ref))

MODELS = ["Basic Pitch", "YourMT3"]
agg = {m: {"lead": [], "poly": []} for m in MODELS}
for g, (notes, prog, bpm) in GTS.items():
    gm = os.path.join(OUT, f"{g}.mid"); mk_midi(notes, prog, gm, bpm)
    kind = "poly" if g.startswith("poly") else "lead"
    for tn, tp in TIMBRES:
        wav = os.path.join(OUT, f"{g}__{tn}.wav"); render(gm, tp, wav)
        ests = {"Basic Pitch": t_basicpitch(wav), "YourMT3": t_yourmt3(wav)}
        for m in MODELS:
            sc = score(notes, ests[m]); agg[m][kind].append(sc)
            print(f"  {g:<16}{tn:<9}{m:<13} onset+pitch F1={sc['opF']*100:5.1f}%  "
                  f"onsetF1={sc['onF']*100:5.1f}%  notes×{sc['ncr']:.2f}", flush=True)

def avg(l, k): return sum(s[k] for s in l) / len(l) if l else 0.0
L = ["# results_melody.md — YourMT3 (SOTA) vs Basic Pitch", "",
     "Synthetic GT, rendered through 2 GM timbres, scored with mir_eval (Onset+Pitch F1).", "",
     "| Case | Model | Onset+Pitch F1 | Onset F1 | note ratio |", "|---|---|---|---|---|"]
for kind in ("lead", "poly"):
    for m in MODELS:
        A = agg[m][kind]
        L.append(f"| {kind} | {m} | **{avg(A,'opF')*100:.1f}%** | {avg(A,'onF')*100:.1f}% | {avg(A,'ncr'):.2f} |")
with open(os.path.join(OUT, "results_melody.md"), "w") as f: f.write("\n".join(L))
with open("/mnt/d/flbeat/fl-beat-assimilator/results_melody.md", "w") as f: f.write("\n".join(L))
print("\n" + "\n".join(L[4:]), flush=True)
