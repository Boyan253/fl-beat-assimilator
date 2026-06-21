"""render_audio.py — render MIDI to WAV + MP3 so you can hear it without opening FL.

Uses FluidSynth (offline software synth) with a General MIDI soundfont, then ffmpeg
for MP3. The GM soundfont gives a generic-instrument preview — enough to judge whether
a generated riff is musically good before importing it into FL with proper phonk sounds.

  python render_audio.py <file.mid | directory> [out_dir]

Default: renders every .mid in /mnt/d/flbeat/data/generated to .wav + .mp3 alongside it.
Needs (WSL): apt install fluidsynth fluid-soundfont-gm ffmpeg
"""
import os, sys, glob, subprocess

SOUNDFONT = "/usr/share/sounds/sf2/FluidR3_GM.sf2"
GAIN      = "0.8"   # fluidsynth output gain


def render_one(mid, out_dir):
    base = os.path.splitext(os.path.basename(mid))[0]
    wav  = os.path.join(out_dir, base + ".wav")
    mp3  = os.path.join(out_dir, base + ".mp3")
    # 1) MIDI -> WAV (FluidSynth, headless)
    r = subprocess.run(
        ["fluidsynth", "-ni", "-g", GAIN, "-F", wav, "-r", "44100", SOUNDFONT, mid],
        capture_output=True, text=True)
    if not os.path.exists(wav) or os.path.getsize(wav) < 1000:
        print(f"  FAIL {base}: fluidsynth produced no audio\n  {r.stderr.strip()[:200]}")
        return None
    # 2) WAV -> MP3 (ffmpeg)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-b:a", "192k", mp3],
                   capture_output=True)
    secs = None
    try:
        import soundfile as sf
        secs = round(sf.info(wav).duration, 1)
    except Exception:
        pass
    print(f"  ok: {base}.wav + .mp3" + (f"  ({secs}s)" if secs else ""))
    return mp3


def main():
    target  = sys.argv[1] if len(sys.argv) > 1 else "/mnt/d/flbeat/data/generated"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else (
        target if os.path.isdir(target) else os.path.dirname(target))
    os.makedirs(out_dir, exist_ok=True)

    if os.path.isdir(target):
        mids = sorted(glob.glob(os.path.join(target, "*.mid")))
    else:
        mids = [target]
    if not mids:
        print(f"No .mid files found in {target}")
        return

    if not os.path.exists(SOUNDFONT):
        print(f"Soundfont missing: {SOUNDFONT}\n  apt install fluid-soundfont-gm")
        return

    print(f"Rendering {len(mids)} MIDI file(s) -> {out_dir}")
    done = 0
    for m in mids:
        if render_one(m, out_dir):
            done += 1
    print(f"\nDone: {done}/{len(mids)} rendered. Play the .mp3 files to preview.")


if __name__ == "__main__":
    main()
