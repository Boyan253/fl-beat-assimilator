"""build_trackout.py — turn a finished track into an EXCLUSIVE trackout pack.

Separates the audio into sellable stems + transcribes the melodic parts to MIDI, so a
buyer gets everything needed to mix/edit in FL:
  808.wav  Melody.wav  Drums.wav  (Vocals.wav)   - WAV stems (trackout)
  Kick.wav  Hats.wav                              - drum split (band) from the drum stem
  Melody.mid  808.mid                             - editable MIDI for the melodic parts
  <track>.wav                                     - the full master

  python build_trackout.py <track.(wav|mp3)> <out_dir>

Run in the demucs venv (flbeat-venv: has demucs + basic_pitch).
"""
import os, sys, shutil, warnings
warnings.filterwarnings("ignore")
import numpy as np, librosa, soundfile as sf, torch
import scipy.signal as ss

TRACK = sys.argv[1]
OUT   = sys.argv[2]
os.makedirs(OUT, exist_ok=True)
LABELS = {"bass": "808", "other": "Melody", "drums": "Drums", "vocals": "Vocals"}


def bandfilter(y, sr, lo=None, hi=None):
    if lo and hi:
        sos = ss.butter(4, [lo, min(hi, sr/2 - 1)], btype="band", fs=sr, output="sos")
    elif lo:
        sos = ss.butter(4, lo, btype="high", fs=sr, output="sos")
    else:
        sos = ss.butter(4, hi, btype="low", fs=sr, output="sos")
    return ss.sosfilt(sos, y, axis=0).astype("float32")


def main():
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"separating on {device}...", flush=True)
    model = get_model("htdemucs_ft"); model.eval().to(device)
    sr = int(model.samplerate)

    y, _ = librosa.load(TRACK, sr=sr, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])
    wav = torch.tensor(y, dtype=torch.float32); ref = wav.mean(0)
    wav_n = (wav - ref.mean()) / (ref.std() + 1e-8)
    with torch.no_grad():
        srcs = apply_model(model, wav_n[None].to(device), device=device, split=True, overlap=0.25)[0]
    srcs = srcs * ref.std() + ref.mean()
    stems = {n: srcs[i].cpu().numpy() for i, n in enumerate(model.sources)}  # (chan,samples)

    saved = []
    for s, arr in stems.items():
        if float(np.sqrt(np.mean(arr ** 2))) < 0.004:
            continue
        p = os.path.join(OUT, LABELS[s] + ".wav")
        sf.write(p, arr.T.astype("float32"), sr); saved.append(LABELS[s])

    # drum split (band approximation) from the drum stem
    if "drums" in stems:
        d = stems["drums"].T  # (samples,chan)
        sf.write(os.path.join(OUT, "Kick.wav"), bandfilter(d, sr, hi=120), sr)
        sf.write(os.path.join(OUT, "Hats.wav"), bandfilter(d, sr, lo=7000), sr)
        saved += ["Kick", "Hats"]

    # MIDI for melodic stems
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH
    for label, fmin, fmax in [("Melody", 110, 2000), ("808", 30, 400)]:
        wp = os.path.join(OUT, label + ".wav")
        if os.path.exists(wp):
            try:
                _, midi, _ = predict(wp, ICASSP_2022_MODEL_PATH, onset_threshold=0.5,
                                     frame_threshold=0.3, minimum_note_length=90,
                                     minimum_frequency=fmin, maximum_frequency=fmax)
                midi.write(os.path.join(OUT, label + ".mid")); saved.append(label + ".mid")
            except Exception as e:
                print(f"  midi {label} failed: {e}", flush=True)

    shutil.copy(TRACK, os.path.join(OUT, "MASTER" + os.path.splitext(TRACK)[1]))
    with open(os.path.join(OUT, "README.txt"), "w") as f:
        f.write("EXCLUSIVE TRACKOUT PACK\n\n"
                "WAV stems: 808, Melody, Drums (+ Kick/Hats split), Vocals (if any)\n"
                "MIDI: Melody.mid, 808.mid (edit notes/key in FL)\n"
                "MASTER.* = full track\n\n"
                "Load stems on audio tracks; load .mid on your own instruments to re-voice.\n")
    print("PACK:", OUT, "->", ", ".join(saved), flush=True)


if __name__ == "__main__":
    main()
