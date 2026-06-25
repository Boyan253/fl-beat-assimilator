"""bench_transcribe.py — rank audio->MIDI transcribers on SYNTHETIC ground truth.

Method (per MEASURE.md):
  known GT MIDI  -> render to audio (fluidsynth, real GM timbres)  -> transcribe with each model
  -> score vs the KNOWN MIDI with mir_eval.transcription (Onset F1, Onset+Pitch F1, +Offset).
Writes results.md with a per-model table so we can pick the best and swap it into the melody path.

  python bench_transcribe.py [out_dir]
"""
import os, sys, subprocess, warnings, math
warnings.filterwarnings("ignore")
import numpy as np, librosa, pretty_midi, mir_eval
import torch, torchcrepe

OUT = sys.argv[1] if len(sys.argv) > 1 else "/mnt/d/flbeat/data/bench"
os.makedirs(OUT, exist_ok=True)
SF2 = "/usr/share/sounds/sf2/FluidR3_GM.sf2"
SR = 22050
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- ground-truth MIDI generators (deterministic) ----------------
def mk_midi(notes, program, path, bpm=140):
    pm = pretty_midi.PrettyMIDI(initial_tempo=float(bpm))
    ins = pretty_midi.Instrument(program=program)
    for (s, e, p) in notes:
        ins.notes.append(pretty_midi.Note(100, p, s, e))
    pm.instruments.append(ins); pm.write(path)

def scale_riff(root, steps, dur, bpm, n=24):
    spb = 60.0 / bpm; step = spb * dur; notes = []
    deg = [0, 2, 3, 5, 7, 10, 12]  # minor-ish (phonk)
    for i in range(n):
        p = root + deg[(i * 3) % len(deg)] + 12 * ((i // 7) % 2)
        s = i * step; notes.append((s, s + step * 0.9, int(p)))
    return notes

def chord_prog(root, bpm, n=8):
    spb = 60.0 / bpm; step = spb * 2; notes = []
    for i in range(n):
        base = root + [0, 5, 3, 7][i % 4]
        s = i * step
        for iv in (0, 3, 7):  # triad
            notes.append((s, s + step * 0.95, int(base + iv)))
    return notes

GTS = {  # name -> (notes, default_program, bpm, is_mono)
    "mono_lead_mid":  (scale_riff(60, 0.25, 0.5, 140), 81, 140, True),   # eighths, phonk cowbell range
    "mono_lead_fast": (scale_riff(67, 0.5, 0.25, 150), 81, 150, True),   # sixteenths, fast
    "mono_bass":      (scale_riff(36, 0.5, 0.5, 140),  38, 140, True),   # 808 low register
    "poly_chords":    (chord_prog(48, 130),            4,  130, False),  # polyphonic (the ceiling)
}
TIMBRES = [("sawlead", 81), ("epiano", 4)]  # render each GT through 2 timbres to generalise

# ---------------- rendering ----------------
def render(midi_path, program, wav_path):
    pm = pretty_midi.PrettyMIDI(midi_path)
    for ins in pm.instruments:
        ins.program = program
    tmp = wav_path.replace(".wav", "_tmp.mid"); pm.write(tmp)
    subprocess.run(["fluidsynth", "-ni", "-g", "1.0", "-F", wav_path, "-r", str(SR), SF2, tmp],
                   check=True, capture_output=True)
    os.remove(tmp)

# ---------------- transcribers -> list of (start, end, midipitch) ----------------
def onset_notes(y, times, f0, period, thr=0.4, min_dur=0.035):
    """Anchor notes to detected onsets; assign the median CREPE/pYIN pitch per segment.
    Robust on fast notes (pure pitch-change grouping is not)."""
    onsets = list(librosa.onset.onset_detect(y=y, sr=SR, units="time", backtrack=True))
    dur = len(y) / SR
    bounds = sorted(set([0.0] + onsets + [dur]))
    notes = []
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        if e - s < min_dur:
            continue
        mask = (times >= s) & (times < e) & (period > thr) & np.isfinite(f0) & (f0 > 0)
        if mask.sum() < 2:
            continue
        pitch = int(round(librosa.hz_to_midi(float(np.median(f0[mask])))))
        notes.append((s, e * 0.97, pitch))
    return notes

def t_crepe(y):
    y16 = librosa.resample(y, orig_sr=SR, target_sr=16000)
    a = torch.tensor(y16, dtype=torch.float32)[None]
    f0, pd = torchcrepe.predict(a, 16000, hop_length=80, fmin=50, fmax=2000,
                                model="full", return_periodicity=True, device=DEV, batch_size=512)
    # smooth: median-filter pitch, mean-filter periodicity (torchcrepe's recommended cleanup)
    f0 = torchcrepe.filter.median(f0, 3)
    pd = torchcrepe.filter.mean(pd, 3)
    f0 = f0[0].cpu().numpy(); pd = pd[0].cpu().numpy()
    times = np.arange(len(f0)) * 80 / 16000.0
    return onset_notes(y, times, f0, pd, thr=0.35)

def t_pyin(y):
    f0, vo, vp = librosa.pyin(y, fmin=50, fmax=2000, sr=SR)
    times = librosa.times_like(f0, sr=SR)
    period = np.nan_to_num(vp); f0 = np.nan_to_num(f0)
    return onset_notes(y, times, f0, period, thr=0.4)

_BP = {}
def t_basicpitch(wav):
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH
    _, midi, _ = predict(wav, model_or_model_path=ICASSP_2022_MODEL_PATH,
                         onset_threshold=0.5, frame_threshold=0.3, minimum_note_length=70,
                         minimum_frequency=40, maximum_frequency=2200)
    return [(n.start, n.end, n.pitch) for ins in midi.instruments for n in ins.notes]

# ---------------- scoring ----------------
def to_arrays(notes):
    if not notes: return np.zeros((0, 2)), np.zeros(0)
    iv = np.array([[s, max(e, s + 1e-3)] for s, e, _ in notes])
    pi = np.array([440.0 * 2 ** ((p - 69) / 12.0) for _, _, p in notes])
    return iv, pi

def score(ref, est):
    ri, rp = to_arrays(ref); ei, ep = to_arrays(est)
    if len(ri) == 0 or len(ei) == 0:
        return dict(onF=0, opF=0, opoF=0, ncr=(len(ei) / max(1, len(ri))))
    onP, onR, onF = mir_eval.transcription.onset_precision_recall_f1(ri, ei, onset_tolerance=0.05)
    _, _, opF, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ri, rp, ei, ep, onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None)
    _, _, opoF, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ri, rp, ei, ep, onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=0.2)
    return dict(onF=onF, opF=opF, opoF=opoF, ncr=len(ei) / len(ri))

# ---------------- run ----------------
MODELS = ["CREPE", "Basic Pitch", "pYIN"]
rows = {m: {"mono": [], "poly": []} for m in MODELS}
detail = []
print(f"device={DEV}", flush=True)
for gname, (notes, prog, bpm, is_mono) in GTS.items():
    gmid = os.path.join(OUT, f"{gname}.mid"); mk_midi(notes, prog, gmid, bpm)
    ref = [(s, e, p) for (s, e, p) in notes]
    for tname, tprog in TIMBRES:
        wav = os.path.join(OUT, f"{gname}__{tname}.wav"); render(gmid, tprog, wav)
        y, _ = librosa.load(wav, sr=SR, mono=True)
        y = np.nan_to_num(y).astype("float32")
        ests = {"CREPE": t_crepe(y), "Basic Pitch": t_basicpitch(wav), "pYIN": t_pyin(y)}
        for m in MODELS:
            sc = score(ref, ests[m]); sc["gt"] = gname; sc["timbre"] = tname
            rows[m]["mono" if is_mono else "poly"].append(sc); detail.append((m, sc))
            print(f"  {gname:<16}{tname:<9}{m:<12} onset+pitch F1={sc['opF']*100:5.1f}%  "
                  f"onsetF1={sc['onF']*100:5.1f}%  notes×{sc['ncr']:.2f}", flush=True)

def avg(lst, k): return (sum(s[k] for s in lst) / len(lst)) if lst else 0.0

# ---------------- results.md ----------------
L = ["# results.md — transcriber benchmark (synthetic ground truth)", "",
     f"Rendered {len(GTS)} GT MIDIs through {len(TIMBRES)} GM timbres via fluidsynth, "
     "transcribed with each model, scored with `mir_eval.transcription` (±50 ms, ±50 cents).",
     "Headline = **Onset+Pitch F1** (right notes at right times).", "",
     "## Monophonic (the phonk lead / 808 case)", "",
     "| Model | Onset+Pitch F1 | Onset F1 | Onset+Pitch+Offset F1 | note-count ratio |",
     "|---|---|---|---|---|"]
for m in MODELS:
    M = rows[m]["mono"]
    L.append(f"| {m} | **{avg(M,'opF')*100:.1f}%** | {avg(M,'onF')*100:.1f}% | "
             f"{avg(M,'opoF')*100:.1f}% | {avg(M,'ncr'):.2f} |")
L += ["", "## Polyphonic (the open-problem ceiling)", "",
      "| Model | Onset+Pitch F1 | Onset F1 | note-count ratio |", "|---|---|---|---|"]
for m in MODELS:
    P = rows[m]["poly"]
    L.append(f"| {m} | {avg(P,'opF')*100:.1f}% | {avg(P,'onF')*100:.1f}% | {avg(P,'ncr'):.2f} |")
winner = max(MODELS, key=lambda m: avg(rows[m]["mono"], "opF"))
L += ["", f"## Winner (monophonic): **{winner}** "
      f"— {avg(rows[winner]['mono'],'opF')*100:.1f}% Onset+Pitch F1.", "",
      "Decision: route **monophonic** stems (lead, 808) through "
      f"**{winner}**; keep Basic Pitch as fast fallback; polyphonic stays approximate "
      "(use faithful per-occurrence audio playback for the sound).", ""]
with open(os.path.join(OUT, "results.md"), "w") as f: f.write("\n".join(L))
# also drop a copy in the repo
with open("/mnt/d/flbeat/fl-beat-assimilator/results.md", "w") as f: f.write("\n".join(L))
print("\n".join(L[5:]), flush=True)
print(f"\nWROTE {OUT}/results.md + repo/results.md", flush=True)
