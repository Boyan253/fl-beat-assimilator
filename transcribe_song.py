"""Free SOTA pipeline: MP3 -> Demucs stems -> per-stem MIDI, ready to drag into FL.

  python transcribe_song.py "<song.mp3>"

Writes to D:\flmidi\out\<song>\: bass.mid, melody.mid (other), drums.mid + a summary.
Melodic stems go through Basic Pitch (polyphonic SOTA); bass falls back to monophonic
pyin if Basic Pitch struggles; drums become GM hits from per-band onset detection.
"""
import os, sys, json, subprocess, glob, warnings
warnings.filterwarnings("ignore")

# pin every cache to D: so nothing touches C:
D = r"D:\flmidi"
os.environ.update(TMP=D + r"\tmp", TEMP=D + r"\tmp", TORCH_HOME=D + r"\torchcache",
                  HF_HOME=D + r"\hf", XDG_CACHE_HOME=D + r"\hf")

import numpy as np, librosa, soundfile as sf, pretty_midi

NOTE = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
PY = sys.executable


def demucs_separate(mp3, workdir):
    """Separate with the low-level demucs model (load=librosa, save=soundfile) to
    avoid demucs.api / torchaudio.save->torchcodec. Returns name -> wav path."""
    import torch
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    base = os.path.splitext(os.path.basename(mp3))[0]
    folder = os.path.join(workdir, "demucs", "htdemucs", base)
    os.makedirs(folder, exist_ok=True)
    print("  [demucs] loading model (cached)...", flush=True)
    model = get_model("htdemucs"); model.eval()
    sr = int(model.samplerate)
    y, _ = librosa.load(str(mp3), sr=sr, mono=False)     # [channels, samples] or [samples]
    if y.ndim == 1:
        y = np.stack([y, y])
    wav = torch.tensor(y, dtype=torch.float32)
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / (ref.std() + 1e-8)        # demucs normalization
    print("  [demucs] separating on CPU...", flush=True)
    with torch.no_grad():
        sources = apply_model(model, wav[None], device="cpu", split=True,
                              overlap=0.25, progress=True)[0]
    sources = sources * ref.std() + ref.mean()
    stems = {}
    for name, src in zip(model.sources, sources):
        arr = src.detach().cpu().numpy().T               # [samples, channels]
        p = os.path.join(folder, name + ".wav")
        sf.write(p, arr.astype("float32"), sr)
        stems[name] = p
        print("    wrote", name, flush=True)
    return stems


def bp_transcribe(wav, out_mid, fmin=None, fmax=None):
    """Basic Pitch -> MIDI. Returns (n_notes, pitch_classes) or None on failure."""
    try:
        from basic_pitch.inference import predict
        from basic_pitch import ICASSP_2022_MODEL_PATH
        kw = {}
        if fmin: kw["minimum_frequency"] = fmin
        if fmax: kw["maximum_frequency"] = fmax
        _, midi_data, notes = predict(wav, ICASSP_2022_MODEL_PATH, **kw)
        midi_data.write(out_mid)
        pcs = sorted({NOTE[int(round(librosa.hz_to_midi(440*2**((n[2]-69)/12)) if False else n[2])) % 12] for n in notes}) if notes else []
        # note_events tuples: (start, end, pitch, amplitude, [bends]); pitch is MIDI int
        pcs = sorted({NOTE[int(n[2]) % 12] for n in notes})
        return len(notes), pcs
    except Exception as e:
        print("  [basic-pitch] fell back (%s: %s)" % (type(e).__name__, e), flush=True)
        return None


def pyin_transcribe(wav, out_mid, fmin, fmax, program=33):
    """Monophonic fallback: pyin pitch track -> merged note events -> MIDI."""
    y, sr = librosa.load(wav, sr=22050, mono=True)
    f0, vf, vp = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr, frame_length=4096)
    t = librosa.times_like(f0, sr=sr)
    pm = pretty_midi.PrettyMIDI(); inst = pretty_midi.Instrument(program=program)
    cur, start = None, 0.0; pcs = set(); n = 0
    for i in range(len(f0)):
        m = int(round(librosa.hz_to_midi(f0[i]))) if (vf[i] and not np.isnan(f0[i])) else None
        if m != cur:
            if cur is not None and (t[i] - start) > 0.06:
                inst.notes.append(pretty_midi.Note(velocity=100, pitch=cur, start=start, end=t[i]))
                pcs.add(NOTE[cur % 12]); n += 1
            cur, start = m, t[i]
    pm.instruments.append(inst); pm.write(out_mid)
    return n, sorted(pcs)


def extract_oneshot(stem_wav, out_wav, lo, hi, min_dur=0.22, max_dur=1.1):
    """Find the cleanest single-note segment in a melodic stem -> one-shot WAV.
    Returns keycenter MIDI note. This IS the instrument (sampled from the real audio)."""
    y, sr = librosa.load(stem_wav, sr=44100, mono=True)
    oenv = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=oenv, sr=sr, backtrack=True, units="samples")
    best = None
    for i in range(len(onsets) - 1):
        s, e = int(onsets[i]), int(onsets[i + 1])
        seg = y[s:e]
        if len(seg) / sr < min_dur:
            continue
        seg = seg[:int(max_dur * sr)]
        f0, vf, vp = librosa.pyin(seg, fmin=lo, fmax=hi, sr=sr, frame_length=2048)
        f0v = f0[~np.isnan(f0)]
        if len(f0v) < 5:
            continue
        midis = librosa.hz_to_midi(f0v)
        stab = 1.0 / (float(np.std(midis)) + 0.1)
        amp = float(np.sqrt(np.mean(seg ** 2)))
        score = stab * amp
        if best is None or score > best[0]:
            best = (score, seg.copy(), int(round(np.median(midis))))
    if best is None:                       # fallback: loudest 0.6s window
        w = int(0.6 * sr)
        if len(y) <= w:
            seg, mid = y, 60
        else:
            rms = np.array([np.sqrt(np.mean(y[k:k + w] ** 2)) for k in range(0, len(y) - w, w // 2)])
            k = int(np.argmax(rms)) * (w // 2)
            seg, mid = y[k:k + w], 60
    else:
        seg, mid = best[1], best[2]
    seg = seg / (np.max(np.abs(seg)) or 1) * 0.97
    fo = int(0.01 * sr)
    if len(seg) > fo:
        seg[-fo:] *= np.linspace(1, 0, fo)
    sf.write(out_wav, seg.astype("float32"), sr)
    return mid


def write_sfz(out_sfz, wav_name, keycenter):
    with open(out_sfz, "w") as f:
        f.write("// auto-sampled from the isolated stem; drop into FL DirectWave / sfizz\n")
        f.write("<region>\nsample=%s\npitch_keycenter=%d\nlokey=0\nhikey=127\nloop_mode=no_loop\n"
                % (wav_name, keycenter))


def extract_drum_oneshots(drum_wav, outdir):
    """Pull one clean kick/snare/hat hit out of the drum stem -> WAV one-shots."""
    y, sr = librosa.load(drum_wav, sr=44100, mono=True)
    S = np.abs(librosa.stft(y, n_fft=2048)); fr = librosa.fft_frequencies(sr=sr, n_fft=2048)
    res = {}
    for (lo, hi, nm, dur) in [(20,120,"kick",0.35),(150,500,"snare",0.3),(6000,14000,"hat",0.15)]:
        band = (fr >= lo) & (fr < hi)
        oenv = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S[band,:]+1e-9), sr=sr)
        on = librosa.onset.onset_detect(onset_envelope=oenv, sr=sr, units="samples", backtrack=True)
        if len(on) == 0:
            continue
        w = int(0.1 * sr)
        best_s = int(on[int(np.argmax([np.sum(y[int(s):int(s)+w]**2) for s in on]))])
        seg = y[best_s:best_s + int(dur * sr)]
        if len(seg) < 8:
            continue
        seg = seg / (np.max(np.abs(seg)) or 1) * 0.97
        fo = int(0.005 * sr)
        if len(seg) > fo:
            seg[-fo:] *= np.linspace(1, 0, fo)
        p = os.path.join(outdir, nm + ".wav"); sf.write(p, seg.astype("float32"), sr)
        res[nm] = os.path.basename(p)
    return res


def drums_to_midi(wav, out_mid, bpm=140):
    """Per-band onsets -> GM drum MIDI (kick 36, snare 38, hat 42)."""
    y, sr = librosa.load(wav, sr=22050, mono=True)
    S = np.abs(librosa.stft(y, n_fft=2048)); fr = librosa.fft_frequencies(sr=sr, n_fft=2048)
    pm = pretty_midi.PrettyMIDI(); drum = pretty_midi.Instrument(program=0, is_drum=True)
    counts = {}
    for (lo, hi, pitch, nm) in [(20,120,36,"kick"),(150,500,38,"snare"),(6000,14000,42,"hat")]:
        band = (fr >= lo) & (fr < hi)
        oenv = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S[band,:]+1e-9), sr=sr)
        if oenv.max() > 0: oenv = oenv/oenv.max()
        on = librosa.util.peak_pick(oenv, pre_max=3, post_max=3, pre_avg=5, post_avg=5, delta=0.18, wait=4)
        ts = librosa.frames_to_time(on, sr=sr)
        for tt in ts:
            drum.notes.append(pretty_midi.Note(velocity=110, pitch=pitch, start=float(tt), end=float(tt)+0.05))
        counts[nm] = len(ts)
    pm.instruments.append(drum); pm.write(out_mid)
    return counts


if __name__ == "__main__":
    mp3 = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\User\Downloads\excalibur.mp3"
    base = os.path.splitext(os.path.basename(mp3))[0]
    workdir = os.path.join(D, "out", base); os.makedirs(workdir, exist_ok=True)
    print("INPUT:", mp3); print("OUTPUT DIR:", workdir)

    stems = demucs_separate(mp3, workdir)
    print("stems:", list(stems.keys()))
    summary = {"input": mp3, "stems": {}}

    instdir = os.path.join(workdir, "instruments"); os.makedirs(instdir, exist_ok=True)

    if "bass" in stems:
        r = bp_transcribe(stems["bass"], os.path.join(workdir, "bass.mid"), fmin=30, fmax=350)
        if not r or r[0] == 0:
            r = pyin_transcribe(stems["bass"], os.path.join(workdir, "bass.mid"), 30, 350, program=38)
        kc = extract_oneshot(stems["bass"], os.path.join(instdir, "bass_808.wav"), 30, 350)
        write_sfz(os.path.join(instdir, "bass_808.sfz"), "bass_808.wav", kc)
        summary["stems"]["bass"] = {"notes": r[0], "pitch_classes": r[1], "instrument_keycenter": kc}
    if "other" in stems:
        r = bp_transcribe(stems["other"], os.path.join(workdir, "melody.mid"))
        if not r:
            r = pyin_transcribe(stems["other"], os.path.join(workdir, "melody.mid"), 110, 1500, program=81)
        kc = extract_oneshot(stems["other"], os.path.join(instdir, "melody_lead.wav"), 110, 1500)
        write_sfz(os.path.join(instdir, "melody_lead.sfz"), "melody_lead.wav", kc)
        summary["stems"]["melody"] = {"notes": r[0], "pitch_classes": r[1], "instrument_keycenter": kc}
    if "vocals" in stems:
        r = bp_transcribe(stems["vocals"], os.path.join(workdir, "vocal_chops.mid"))
        if r: summary["stems"]["vocal_chops"] = {"notes": r[0], "pitch_classes": r[1]}
    if "drums" in stems:
        c = drums_to_midi(stems["drums"], os.path.join(workdir, "drums.mid"))
        kits = extract_drum_oneshots(stems["drums"], instdir)
        summary["stems"]["drums"] = {"hits": c, "oneshots": kits}

    json.dump(summary, open(os.path.join(workdir, "summary.json"), "w"), indent=2)
    print("\n=== RESULT ===")
    for k, v in summary["stems"].items():
        print(" ", k.ljust(12), v)
    print("\nMIDI files in:", workdir)
    for m in glob.glob(os.path.join(workdir, "*.mid")):
        print("  ", os.path.basename(m))
