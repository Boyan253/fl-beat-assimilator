"""pack_beat.py — turn an AI beat MIDI into a DRAG-AND-DROP, instruments-included FL folder.

The problem: a .mid carries notes only, never instruments; the FL trial won't reopen a .flp.
The fix: ship the instruments as the things the trial DOES allow — copyright-free one-shot
samples + a rendered preview + exact Vital/stock-plugin recipes. You load the rack once per
session (presets+samples, ~2 min), then DRAG each per-instrument .mid onto its ready channel.

Outputs  FL_READY/<NAME>/
  midi/  Melody.mid 808.mid Kick.mid Snare.mid Hat.mid   (drag each onto its channel)
  oneshots/  Kick.wav Snare.wav Hat.wav                  (drag onto sampler/FPC — copyright-free synth)
  preview.wav                                            (already GRIM-dark — proves the sound)
  LOAD_ME.txt                                            (exact Vital knobs + master grit chain)

  python pack_beat.py <in.mid> <NAME> [outroot=/mnt/d/flbeat/data/FL_READY]
"""
import os, sys, shutil, subprocess, warnings
warnings.filterwarnings("ignore")
import numpy as np, soundfile as sf, scipy.signal as ss, pretty_midi

IN   = sys.argv[1]
NAME = sys.argv[2]
ROOT = sys.argv[3] if len(sys.argv) > 3 else "/mnt/d/flbeat/data/FL_READY"
SR   = 44100

OUT = os.path.join(ROOT, NAME)
if os.path.isdir(OUT):
    shutil.rmtree(OUT)
for sub in ("midi", "oneshots"):
    os.makedirs(os.path.join(OUT, sub), exist_ok=True)

pm  = pretty_midi.PrettyMIDI(IN)
END = pm.get_end_time() + 0.5
N   = int(END * SR) + SR


def f_of(p):                       # midi pitch -> Hz
    return 440.0 * 2.0 ** ((p - 69) / 12.0)


def adsr(n, a, d, s, r):           # sample-count ADSR envelope over n samples
    a, d, r = int(a * SR), int(d * SR), int(r * SR)
    env = np.ones(n, dtype=np.float32) * s
    if a: env[:a] = np.linspace(0, 1, a, dtype=np.float32)
    if d: env[a:a + d] = np.linspace(1, s, d, dtype=np.float32)
    if r and n - r > 0: env[n - r:] = np.linspace(env[n - r], 0, r, dtype=np.float32)
    return env


# ---- copyright-free synth voices (no samples = no copyright, ever) --------------------
def cowbell_voice(f, dur, vel):    # AUTHENTIC 808 cowbell = the phonk melody signature
    fb = f * 2.0                                       # lift into the cowbell register
    n = int(min(max(dur, 0.18), 0.55) * SR); t = np.arange(n) / SR
    sq = lambda fr: np.sign(np.sin(2 * np.pi * fr * t))
    sig = 0.5 * sq(fb) + 0.5 * sq(fb * 1.48)           # 1.48 inharmonic ratio = metallic clang
    lo, hi = max(fb * 0.6, 320), min(fb * 2.2, 7000)
    sig = ss.sosfilt(ss.butter(2, [lo, hi], btype="band", fs=SR, output="sos"), sig)
    env = np.exp(-t * 9.0); a = int(0.002 * SR); env[:a] *= np.linspace(0, 1, a)
    sig = np.tanh(sig * env * 2.2) / np.tanh(2.2)      # decay + grit
    return sig.astype(np.float32) * (0.6 * vel / 110.0)


def bass_voice(f, dur, vel):       # 808: sine + short pitch-glide head + drive
    n = int(max(dur, 0.25) * SR); t = np.arange(n) / SR
    gl = int(0.03 * SR)
    fcurve = np.full(n, f, dtype=np.float32)
    if gl: fcurve[:gl] = np.linspace(f * 1.5, f, gl)
    ph = 2 * np.pi * np.cumsum(fcurve) / SR
    sig = np.sin(ph) * adsr(n, 0.004, 0.10, 0.85, 0.30)
    sig = np.tanh(sig * 3.2) / np.tanh(3.2)          # 808 grit
    return sig.astype(np.float32) * (0.95 * vel / 120.0)


def kick_voice():                  # 808-kick: pitch-drop sine + click
    n = int(0.32 * SR); t = np.arange(n) / SR
    fc = 120 * np.exp(-t * 38) + 45
    sig = np.sin(2 * np.pi * np.cumsum(fc) / SR) * np.exp(-t * 9)
    sig[:80] += np.linspace(1, 0, 80) * 0.6          # click
    return (np.tanh(sig * 2.0) * 0.9).astype(np.float32)


def snare_voice():                 # tone + noise burst
    n = int(0.22 * SR); t = np.arange(n) / SR
    tone = np.sin(2 * np.pi * 185 * t) * np.exp(-t * 22) * 0.5
    nz = np.random.randn(n).astype(np.float32)
    sos = ss.butter(4, [1400, 3600], btype="band", fs=SR, output="sos")
    nz = ss.sosfilt(sos, nz) * np.exp(-t * 16)
    return ((tone + nz) * 0.8).astype(np.float32)


def hat_voice():                   # short dark noise (stays under the 6 kHz master roll-off)
    n = int(0.06 * SR); t = np.arange(n) / SR
    nz = np.random.randn(n).astype(np.float32)
    sos = ss.butter(4, [3500, 5800], btype="band", fs=SR, output="sos")
    return (ss.sosfilt(sos, nz) * np.exp(-t * 60) * 0.35).astype(np.float32)


KICK, SNARE, HAT = kick_voice(), snare_voice(), hat_voice()
DRUM_ROLE = {35: "Kick", 36: "Kick", 38: "Snare", 40: "Snare", 42: "Hat", 44: "Hat", 46: "Hat"}
DRUM_SAMP = {"Kick": KICK, "Snare": SNARE, "Hat": HAT}

mix = np.zeros(N, dtype=np.float32)
def place(samp, t, g=1.0):
    i = int(t * SR); m = min(len(samp), N - i)
    if m > 0: mix[i:i + m] += samp[:m] * g

# ---- classify source tracks, render preview, AND collect notes per output file --------
buckets = {"Melody": [], "808": [], "Kick": [], "Snare": [], "Hat": []}
for inst in pm.instruments:
    nm = (inst.name or "").lower()
    if inst.is_drum or "drum" in nm:
        for n in inst.notes:
            role = DRUM_ROLE.get(n.pitch, "Hat")
            place(DRUM_SAMP[role], n.start, n.velocity / 120.0)
            buckets[role].append((n.start, n.velocity))     # -> trigger note C5 later
    elif "808" in nm or "bass" in nm:
        for n in inst.notes:
            place(bass_voice(f_of(n.pitch), n.end - n.start, n.velocity), n.start)
            buckets["808"].append(n)
    else:
        for n in inst.notes:
            place(cowbell_voice(f_of(n.pitch), n.end - n.start, n.velocity), n.start)
            buckets["Melody"].append(n)

# ---- GRIM grit (the same chain as phonkify.py: low-end drive, tape sat, crush, dark, limit) ----
y = np.stack([mix, mix], axis=1)
sos_low = ss.butter(4, 200, btype="low", fs=SR, output="sos")
low = ss.sosfilt(sos_low, y, axis=0)
y = y + (np.tanh(low * 3.5) / np.tanh(3.5) - low) * 0.6     # 808/low-end grit
y = np.tanh(y * 2.0) / np.tanh(2.0)                         # tape saturation
q = 2 ** (12 - 1); y = np.round(y * q) / q                  # 12-bit crush
y = ss.sosfilt(ss.butter(4, 6000, btype="low", fs=SR, output="sos"), y, axis=0)  # dark <6 kHz
y = y + np.random.randn(*y.shape).astype(np.float32) * 0.0008                    # tape hiss
y = np.tanh(y * 0.9) / np.tanh(0.9)
y = y / (np.max(np.abs(y)) + 1e-9) * 0.95
sf.write(os.path.join(OUT, "preview.wav"), y.astype(np.float32), SR)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", os.path.join(OUT, "preview.wav"),
                "-b:a", "192k", os.path.join(OUT, "preview.mp3")], capture_output=True)

# ---- per-instrument MIDI (drag each onto its ready channel) ---------------------------
def write_midi(fname, notes, program, is_drum=False, trigger=None):
    out = pretty_midi.PrettyMIDI()
    ins = pretty_midi.Instrument(program=program, is_drum=is_drum, name=fname)
    if trigger is not None:                       # drums: one fixed trigger note (C5)
        for (start, vel) in notes:
            ins.notes.append(pretty_midi.Note(int(vel), trigger, start, start + 0.12))
    else:
        for n in notes: ins.notes.append(n)
    out.instruments.append(ins)
    out.write(os.path.join(OUT, "midi", fname + ".mid"))
    return len(ins.notes)

cnt = {}
cnt["Melody"] = write_midi("Melody", buckets["Melody"], 11)
cnt["808"]    = write_midi("808",    buckets["808"], 38)
for role in ("Kick", "Snare", "Hat"):
    cnt[role] = write_midi(role, buckets[role], 0, is_drum=False, trigger=60)

# ---- copyright-free one-shots (drag straight onto a sampler / FPC pad) -----------------
for role, samp in DRUM_SAMP.items():
    s = samp / (np.max(np.abs(samp)) + 1e-9) * 0.97
    sf.write(os.path.join(OUT, "oneshots", role + ".wav"), s.astype(np.float32), SR)
# the 808 cowbell as a playable one-shot (rendered at a mid melody pitch, no decay cap = let it ring)
cb_pitch = int(np.median([n.pitch for n in buckets["Melody"]])) if buckets["Melody"] else 72
cb = cowbell_voice(f_of(cb_pitch), 0.5, 110)
cb = cb / (np.max(np.abs(cb)) + 1e-9) * 0.97
sf.write(os.path.join(OUT, "oneshots", f"Cowbell_{pretty_midi.note_number_to_name(cb_pitch)}.wav"),
         cb.astype(np.float32), SR)

# ---- LOAD_ME.txt: the one-time rack + the GRIM master grit chain ----------------------
tempi = pm.get_tempo_changes()[1]
BPM = int(round(tempi[0])) if len(tempi) else 0
root808 = ""
if buckets["808"]:
    from collections import Counter
    root808 = pretty_midi.note_number_to_name(Counter(n.pitch for n in buckets["808"]).most_common(1)[0][0])

guide = f"""{NAME} - FL READY  ({BPM} BPM)  -- instruments included, copyright-free

WHY THIS FOLDER EXISTS
  A .mid stores notes only -- never an instrument. The FL trial won't reopen a .flp.
  So the SOUND rides in on what the trial DOES allow: free presets + one-shot samples.
  Set the rack up ONCE per FL session (~2 min), then just DRAG each .mid onto its channel.

  preview.wav / preview.mp3  = exactly how this beat sounds once the rack below is set.
                               (rendered here from free synths -- play it first.)

ONE-TIME RACK  (do this once after opening FL; project tempo = {BPM} BPM)
  1) MELODY = 808 COWBELL (the phonk signature -- NOT a synth pluck). Two easy ways:
       EASY: drop  oneshots/Cowbell_*.wav  onto an FL Sampler channel (it pitches across keys),
             or load a free royalty-free phonk-cowbell pack (Sample Focus "PHONK COWBELL",
             loopsy Brazilian-Phonk free pack, Phonk Kong). "Which sound?" = "an 808 cowbell."
       VITAL: Osc 1 + Osc 2 = SQUARE, Osc 2 tuned ~ +7 semis (a fifth) = the metallic clang;
             band-pass filter, fast Decay, short release. (This is literally how an 808 makes it.)
       Then drag  midi/Melody.mid  onto the channel.
  2) 808     -> 2nd Vital instance.
       - Osc 1: Sine.  Pitch tuned to {root808 or 'the song root'}.  Mono + Portamento/Glide ON (~80 ms) = slides.
       - Add Distortion module (drive up) for grit.  Long release.
       - Drag  midi/808.mid  onto it.
  3) KICK/SNARE/HAT -> 3 sampler channels (or one FPC). Drag the matching file from
       oneshots/  onto each pad, then drag  midi/Kick.mid / Snare.mid / Hat.mid  on top.
       (each drum MIDI triggers on C5 -- one fixed note per channel.)

THE GRIM MASTER CHAIN  (put on the Master mixer track -- this is ~70% of "the GRIM sound";
  these are phonkify's grit settings translated to FL STOCK plugins, matched to GRIM's
  measured curve: sub-heavy, ~nothing above 6 kHz)
  1) Fruity Parametric EQ 2 -> HIGH-CUT (low-pass) at ~6 kHz, steep.  <- biggest darkness lever
       (optional: small low-shelf boost under 90 Hz for 808 weight)
  2) Fruity Blood Overdrive -> Pre-band low, Post-gain ~ to taste = tape saturation/grit.
  3) Fruity Squeeze         -> drop bit depth a little = lo-fi 12-bit crush.
  4) (optional) a quiet vinyl-crackle/hiss sample on its own channel = tape texture.
  5) Fruity Soft Clipper (or Limiter) -> glue + a slightly-crushed master.
  Keep the 808 LOUD so it blurs into the kick -- that fat low blur is the phonk signature.

TRIAL NOTE
  The trial can't reopen a saved project, so the rack doesn't persist between sessions --
  but loading these presets/samples again is the 2-minute routine above. Everything here
  (Vital, the synth one-shots, the stock master chain) is free and copyright-clean.

notes: """ + " ".join(f"{k}={v}" for k, v in cnt.items()) + "\n"

with open(os.path.join(OUT, "LOAD_ME.txt"), "w", encoding="utf-8") as fh:
    fh.write(guide)

print(f"[pack] {NAME}: " + " ".join(f"{k}={v}" for k, v in cnt.items()) + f" -> {OUT}", flush=True)
