"""extract_leads.py — batch extract melody MIDI from training songs.

For each MP3 in DATA_DIR:
  1. Separate with demucs htdemucs_ft (GPU) -> other.wav
  2. Transcribe with Basic Pitch -> .mid
  3. Skip near-silent or near-empty results
Output MIDI goes to MIDI_DIR (used by finetune_amt.py).

Run in WSL flbeat-venv:
  source /root/flbeat-venv/bin/activate
  python /mnt/d/flbeat/fl-beat-assimilator/extract_leads.py
"""
import os, glob, gc, warnings
warnings.filterwarnings("ignore")

DATA_DIR  = "/mnt/d/flbeat/data/train"
STEMS_DIR = "/mnt/d/flbeat/data/stems"
MIDI_DIR  = "/mnt/d/flbeat/data/midi"

MIN_RMS   = 0.005   # skip near-silent other stems
MIN_NOTES = 20      # skip sparse transcriptions


def main():
    os.makedirs(STEMS_DIR, exist_ok=True)
    os.makedirs(MIDI_DIR,  exist_ok=True)

    mp3s = sorted(glob.glob(os.path.join(DATA_DIR, "*.mp3")))
    print(f"Found {len(mp3s)} MP3s in {DATA_DIR}", flush=True)

    import torch, librosa, soundfile as sf, numpy as np
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading htdemucs_ft on {device}...", flush=True)
    model = get_model("htdemucs_ft")
    model.eval()
    model.to(device)
    sr_model = int(model.samplerate)  # 44100

    ok, skipped, failed = 0, 0, []

    for i, mp3 in enumerate(mp3s):
        song_name = os.path.splitext(os.path.basename(mp3))[0]
        midi_out  = os.path.join(MIDI_DIR,  song_name + ".mid")
        stem_out  = os.path.join(STEMS_DIR, song_name + "_other.wav")

        if os.path.exists(midi_out):
            print(f"[{i+1}/{len(mp3s)}] skip(done): {song_name[:55]}", flush=True)
            ok += 1; skipped += 1
            continue

        print(f"[{i+1}/{len(mp3s)}] {song_name[:60]}", flush=True)
        try:
            # 1. Separate with demucs (GPU)
            if not os.path.exists(stem_out):
                y, _ = librosa.load(mp3, sr=sr_model, mono=False)
                if y.ndim == 1:
                    y = np.stack([y, y])
                wav = torch.tensor(y, dtype=torch.float32)
                ref = wav.mean(0)
                wav_n = (wav - ref.mean()) / (ref.std() + 1e-8)
                with torch.no_grad():
                    srcs = apply_model(model, wav_n.unsqueeze(0).to(device),
                                       device=device, split=True,
                                       overlap=0.25, progress=False)[0]
                srcs = srcs * ref.std() + ref.mean()
                stems = {sname: srcs[j].detach().cpu().numpy()
                         for j, sname in enumerate(model.sources)}
                sf.write(stem_out, stems["other"].T.astype("float32"), sr_model)
                del srcs, stems, wav, wav_n; gc.collect(); torch.cuda.empty_cache()

            # 2. RMS check
            audio, _ = librosa.load(stem_out, sr=22050, mono=True, duration=60)
            rms = float(np.sqrt(np.mean(audio ** 2)))
            if rms < MIN_RMS:
                print(f"  skip: near-silent (rms={rms:.4f})", flush=True)
                skipped += 1; continue

            # 3. Basic Pitch transcription
            from basic_pitch.inference import predict
            from basic_pitch import ICASSP_2022_MODEL_PATH
            _, midi_obj, _ = predict(
                stem_out, ICASSP_2022_MODEL_PATH,
                onset_threshold=0.5, frame_threshold=0.3,
                minimum_note_length=80, minimum_frequency=80, maximum_frequency=2000
            )
            notes = sum(len(inst.notes) for inst in midi_obj.instruments)
            if notes < MIN_NOTES:
                print(f"  skip: only {notes} notes", flush=True)
                skipped += 1; continue

            midi_obj.write(midi_out)
            print(f"  ok: {notes} notes -> {os.path.basename(midi_out)}", flush=True)
            ok += 1

        except KeyboardInterrupt:
            print("\nInterrupted."); break
        except Exception as e:
            import traceback
            print(f"  FAIL: {e}", flush=True); traceback.print_exc()
            failed.append((song_name, str(e)))

    midi_files = glob.glob(os.path.join(MIDI_DIR, "*.mid"))
    print(f"\n=== Done: {ok - skipped} new  {skipped} already done  {len(failed)} failed ===")
    print(f"Total MIDI in {MIDI_DIR}: {len(midi_files)}")
    if failed:
        print("Failed songs:")
        for n, e in failed:
            print(f"  {n}: {e}")


if __name__ == "__main__":
    main()
