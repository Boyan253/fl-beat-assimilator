"""extract_multitrack.py — build MULTI-TRACK phonk MIDI (melody + bass + drums) per song.

Unlike extract_leads.py (melody only), this separates ALL stems and transcribes each into
its own instrument in one MIDI file, so a model trained on it generates a full arrangement:
  - melody : Basic Pitch on the 'other' stem        -> program 30 (distortion guitar = phonk lead)
  - bass   : onset + pyin pitch on the 'bass' stem   -> program 38 (synth bass)
  - drums  : onset + spectral-centroid classes       -> GM drum channel (kick/snare/hat)

  python extract_multitrack.py            # batch all songs in DATA_DIR
  python extract_multitrack.py <one.mp3>  # test a single song

Run in WSL flbeat-venv with the GPU stability env vars (see run_finetune.sh).
Output -> MIDI_MT_DIR (used by finetune_amt.py).
"""
import os, sys, glob, gc, warnings
warnings.filterwarnings("ignore")

DATA_DIR   = "/mnt/d/flbeat/data/train"
STEMS_DIR  = "/mnt/d/flbeat/data/stems_mt"
MIDI_MT_DIR = "/mnt/d/flbeat/data/midi_mt"

SR        = 22050
MIN_RMS   = 0.004
MELODY_PROG = 30   # distortion guitar
BASS_PROG   = 38   # synth bass 1


def transcribe_melody(stem_wav):
    """Basic Pitch -> list of pretty_midi Notes (the lead)."""
    import pretty_midi
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH
    _, midi_obj, _ = predict(stem_wav, ICASSP_2022_MODEL_PATH,
                             onset_threshold=0.5, frame_threshold=0.3,
                             minimum_note_length=90, minimum_frequency=110, maximum_frequency=2000)
    return [n for inst in midi_obj.instruments for n in inst.notes]


def transcribe_bass(stem_wav):
    """Monophonic bass: onsets + median pyin pitch per segment -> Notes."""
    import numpy as np, librosa, pretty_midi
    y, sr = librosa.load(stem_wav, sr=SR, mono=True)
    if np.sqrt(np.mean(y ** 2)) < MIN_RMS:
        return []
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=True)
    if len(onsets) < 2:
        return []
    f0, _, _ = librosa.pyin(y, fmin=30, fmax=400, sr=sr)
    times = librosa.times_like(f0, sr=sr)
    notes = []
    bounds = list(onsets) + [len(y) / sr]
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        if e - s < 0.05:
            continue
        seg = f0[(times >= s) & (times < e)]
        seg = seg[~np.isnan(seg)]
        if len(seg) < 2:
            continue
        midi = int(round(librosa.hz_to_midi(np.median(seg))))
        if 20 <= midi <= 60:  # bass register
            notes.append(pretty_midi.Note(velocity=96, pitch=midi, start=float(s), end=float(min(e, s + 1.0))))
    return notes


def transcribe_drums(stem_wav):
    """Onset detection + spectral centroid -> GM kick(36)/snare(38)/hat(42) Notes."""
    import numpy as np, librosa, pretty_midi
    y, sr = librosa.load(stem_wav, sr=SR, mono=True)
    if np.sqrt(np.mean(y ** 2)) < MIN_RMS:
        return []
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=True)
    notes = []
    for t in onsets:
        i0 = int(t * sr)
        win = y[i0: i0 + int(0.05 * sr)]
        if len(win) < 64:
            continue
        cent = float(librosa.feature.spectral_centroid(y=win, sr=sr).mean())
        if cent < 800:
            pitch = 36       # kick
        elif cent < 4000:
            pitch = 38       # snare
        else:
            pitch = 42       # closed hat
        notes.append(pretty_midi.Note(velocity=100, pitch=pitch, start=float(t), end=float(t + 0.1)))
    return notes


def main():
    import torch, numpy as np, librosa, soundfile as sf, pretty_midi
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    os.makedirs(STEMS_DIR, exist_ok=True)
    os.makedirs(MIDI_MT_DIR, exist_ok=True)

    if len(sys.argv) > 1 and sys.argv[1].endswith(".mp3"):
        mp3s = [sys.argv[1]]
    else:
        mp3s = sorted(glob.glob(os.path.join(DATA_DIR, "*.mp3")))
    print(f"{len(mp3s)} song(s) to process", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model("htdemucs_ft"); model.eval(); model.to(device)
    sr_m = int(model.samplerate)
    srcs_names = model.sources  # ['drums','bass','other','vocals']

    ok = fail = 0
    for i, mp3 in enumerate(mp3s):
        name = os.path.splitext(os.path.basename(mp3))[0]
        out_mid = os.path.join(MIDI_MT_DIR, name + ".mid")
        if os.path.exists(out_mid):
            print(f"[{i+1}/{len(mp3s)}] skip(done): {name[:50]}", flush=True); ok += 1; continue
        print(f"[{i+1}/{len(mp3s)}] {name[:55]}", flush=True)
        try:
            # separate -> save the 3 stems we need
            paths = {}
            need = ["other", "bass", "drums"]
            if all(os.path.exists(os.path.join(STEMS_DIR, f"{name}_{s}.wav")) for s in need):
                for s in need: paths[s] = os.path.join(STEMS_DIR, f"{name}_{s}.wav")
            else:
                y, _ = librosa.load(mp3, sr=sr_m, mono=False)
                if y.ndim == 1: y = np.stack([y, y])
                wav = torch.tensor(y, dtype=torch.float32); ref = wav.mean(0)
                wav_n = (wav - ref.mean()) / (ref.std() + 1e-8)
                with torch.no_grad():
                    out = apply_model(model, wav_n.unsqueeze(0).to(device), device=device,
                                      split=True, overlap=0.25, progress=False)[0]
                out = out * ref.std() + ref.mean()
                for j, s in enumerate(srcs_names):
                    if s in need:
                        p = os.path.join(STEMS_DIR, f"{name}_{s}.wav")
                        sf.write(p, out[j].detach().cpu().numpy().T.astype("float32"), sr_m)
                        paths[s] = p
                del out, wav, wav_n; gc.collect(); torch.cuda.empty_cache()

            pm = pretty_midi.PrettyMIDI()
            mel = pretty_midi.Instrument(program=MELODY_PROG, name="melody")
            mel.notes = transcribe_melody(paths["other"])
            bass = pretty_midi.Instrument(program=BASS_PROG, name="bass")
            bass.notes = transcribe_bass(paths["bass"])
            drums = pretty_midi.Instrument(program=0, is_drum=True, name="drums")
            drums.notes = transcribe_drums(paths["drums"])

            counts = (len(mel.notes), len(bass.notes), len(drums.notes))
            for inst in (mel, bass, drums):
                if inst.notes:
                    pm.instruments.append(inst)
            total = sum(counts)
            if total < 30:
                print(f"  skip: too sparse {counts}", flush=True); continue
            pm.write(out_mid)
            print(f"  ok: melody={counts[0]} bass={counts[1]} drums={counts[2]} -> {name[:40]}.mid", flush=True)
            ok += 1
        except KeyboardInterrupt:
            print("interrupted"); break
        except Exception as e:
            import traceback; print(f"  FAIL: {e}"); traceback.print_exc(); fail += 1

    n = len(glob.glob(os.path.join(MIDI_MT_DIR, "*.mid")))
    print(f"\n=== done: {ok} ok, {fail} failed. Total multi-track MIDI: {n} ===")


if __name__ == "__main__":
    main()
