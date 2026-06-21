"""sample_render.py — play a generated multi-track MIDI through REAL phonk samples.

The phonk SOUND lives in the samples, not the MIDI. This extracts one-shots from a source
song's separated stems (cowbell from 'other', 808 from 'bass', kick/snare/hat from 'drums')
and renders the generated arrangement through them — pitch-shifting the cowbell across the
melody and the 808 across the bassline, which is literally how phonk is produced.

  python sample_render.py <generated.mid> [source_song] [out.wav]

source_song = a name in stems_mt/ to lift samples from (default: COWBELL WARRIOR!).
NOTE: samples lifted from copyrighted songs => AUDITION only. Swap your own/royalty-free
one-shots (cowbell.wav/808.wav/kick.wav/...) for sellable output.
"""
import os, sys, glob, warnings, subprocess
warnings.filterwarnings("ignore")
import numpy as np, librosa, soundfile as sf, pretty_midi

SR        = 44100
STEMS_DIR = "/mnt/d/flbeat/data/stems_mt"
DEF_SRC   = "COWBELL WARRIOR!"


def strongest_onset_clip(y, sr, dur, smooth=0.0):
    """Return a one-shot window (dur seconds) around the strongest onset, faded."""
    env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=env, sr=sr, units="samples", backtrack=True)
    if len(onsets) == 0:
        i0 = int(np.argmax(np.abs(y)))
    else:
        # pick the onset with the most energy just after it
        n = int(dur * sr)
        best, bi = -1, onsets[0]
        for o in onsets:
            e = float(np.sum(y[o:o + n] ** 2))
            if e > best:
                best, bi = e, o
        i0 = bi
    clip = y[i0: i0 + int(dur * sr)].copy()
    if len(clip) < 64:
        clip = np.pad(clip, (0, 64))
    # fade in/out to kill clicks
    fi = min(64, len(clip) // 8)
    fo = min(int(0.03 * sr), len(clip) // 3)
    clip[:fi] *= np.linspace(0, 1, fi)
    clip[-fo:] *= np.linspace(1, 0, fo)
    m = np.max(np.abs(clip)) + 1e-9
    return clip / m


def base_pitch(clip, sr, fallback=48):
    """Estimate the sample's fundamental MIDI pitch (for relative pitch-shifting)."""
    try:
        f0, vf, _ = librosa.pyin(clip, fmin=40, fmax=1000, sr=sr)
        f0 = f0[~np.isnan(f0)]
        if len(f0) >= 3:
            return float(librosa.hz_to_midi(np.median(f0)))
    except Exception:
        pass
    return float(fallback)


def overlay(buf, clip, at_sample, gain):
    e = at_sample + len(clip)
    if e > len(buf):
        clip = clip[:len(buf) - at_sample]
        e = len(buf)
    if at_sample < 0 or len(clip) <= 0:
        return
    buf[at_sample:e] += clip * gain


def render_pitched(buf, notes, sample, base, sr):
    """Render a pitched track (melody/bass) by pitch-shifting the sample per note."""
    cache = {}
    for n in notes:
        semis = int(round(n.pitch - base))
        semis = max(-24, min(24, semis))   # clamp absurd shifts
        if semis not in cache:
            cache[semis] = librosa.effects.pitch_shift(sample, sr=sr, n_steps=semis)
        s = cache[semis]
        gain = 0.4 + 0.6 * (n.velocity / 127.0)
        overlay(buf, s, int(n.start * sr), gain)


def main():
    mid_path = sys.argv[1]
    src      = sys.argv[2] if len(sys.argv) > 2 else DEF_SRC
    out_wav  = sys.argv[3] if len(sys.argv) > 3 else os.path.splitext(mid_path)[0] + "_phonk.wav"

    other_p = os.path.join(STEMS_DIR, f"{src}_other.wav")
    bass_p  = os.path.join(STEMS_DIR, f"{src}_bass.wav")
    drums_p = os.path.join(STEMS_DIR, f"{src}_drums.wav")
    for p in (other_p, bass_p, drums_p):
        if not os.path.exists(p):
            print(f"missing stem: {p}"); return
    print(f"samples from: {src}", flush=True)

    yo, _ = librosa.load(other_p, sr=SR, mono=True)
    yb, _ = librosa.load(bass_p,  sr=SR, mono=True)
    yd, _ = librosa.load(drums_p, sr=SR, mono=True)

    # cowbell (melody) + 808 (bass) one-shots
    cowbell = strongest_onset_clip(yo, SR, 0.28)
    cb_base = base_pitch(cowbell, SR, fallback=60)
    eight08 = strongest_onset_clip(yb, SR, 0.5)
    b_base  = base_pitch(eight08, SR, fallback=36)
    print(f"cowbell base~{cb_base:.0f}  808 base~{b_base:.0f}", flush=True)

    # drum one-shots: cluster onsets by spectral centroid -> kick/snare/hat
    onsets = librosa.onset.onset_detect(y=yd, sr=SR, units="samples", backtrack=True)
    by_kind = {36: None, 38: None, 42: None}
    for o in onsets:
        w = yd[o:o + int(0.16 * SR)]
        if len(w) < 256:
            continue
        cent = float(librosa.feature.spectral_centroid(y=w, sr=SR).mean())
        k = 36 if cent < 800 else (38 if cent < 4000 else 42)
        if by_kind[k] is None:
            cl = w.copy()
            fo = min(int(0.03 * SR), len(cl) // 3)
            cl[-fo:] *= np.linspace(1, 0, fo)
            by_kind[k] = cl / (np.max(np.abs(cl)) + 1e-9)
    # fallbacks if a class wasn't found
    for k in by_kind:
        if by_kind[k] is None:
            by_kind[k] = strongest_onset_clip(yd, SR, 0.12)

    pm = pretty_midi.PrettyMIDI(mid_path)
    dur = pm.get_end_time() + 0.6
    buf = np.zeros(int(dur * SR), dtype=np.float32)

    GAIN = {"melody": 0.55, "bass": 0.95, "drums": 1.0}
    for inst in pm.instruments:
        if inst.is_drum:
            for n in inst.notes:
                clip = by_kind.get(n.pitch, by_kind[38])
                overlay(buf, clip, int(n.start * SR), GAIN["drums"] * (0.5 + 0.5 * n.velocity / 127))
        elif inst.program == 38 or "bass" in (inst.name or "").lower():
            tmp = np.zeros_like(buf); render_pitched(tmp, inst.notes, eight08, b_base, SR)
            buf += tmp * GAIN["bass"]
        else:  # melody -> cowbell
            tmp = np.zeros_like(buf); render_pitched(tmp, inst.notes, cowbell, cb_base, SR)
            buf += tmp * GAIN["melody"]

    # soft clip + normalize
    buf = np.tanh(buf * 1.2)
    buf = buf / (np.max(np.abs(buf)) + 1e-9) * 0.97
    sf.write(out_wav, buf, SR)
    mp3 = os.path.splitext(out_wav)[0] + ".mp3"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", out_wav, "-b:a", "192k", mp3],
                   capture_output=True)
    print(f"wrote {mp3}  ({dur:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
