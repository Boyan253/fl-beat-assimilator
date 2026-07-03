"""Probe the env + inspect existing GRIM melody MIDI and the clean stem, so we can
decide reuse-vs-reextract for the native-FL pitch_keytrack=0 rebuild."""
import importlib, sys

def have(mod):
    try:
        m = importlib.import_module(mod)
        return getattr(m, "__version__", "ok")
    except Exception as e:
        return f"NO ({type(e).__name__})"

print("=== env ===")
for m in ["numpy","scipy","soundfile","librosa","torch","torchaudio","torchcrepe","pretty_midi","mido","basic_pitch"]:
    print(f"  {m:12s}: {have(m)}")

# --- inspect the clean melody stem ---
print("\n=== clean melody stem ===")
stem = "/mnt/d/flbeat/data/grim_roformer/GRIM_melody_clean.wav"
try:
    import soundfile as sf
    info = sf.info(stem)
    print(f"  {stem}")
    print(f"  sr={info.samplerate} ch={info.channels} dur={info.frames/info.samplerate:.2f}s frames={info.frames}")
except Exception as e:
    print(f"  stem read FAIL: {e}")

# --- inspect existing melody MIDI: which pitches does it actually use? ---
print("\n=== existing GRIM_mel.mid pitches ===")
midp = "/mnt/d/flbeat/data/generated/GRIM_multisample/GRIM_mel.mid"
notes = []
try:
    try:
        import pretty_midi
        pm = pretty_midi.PrettyMIDI(midp)
        for inst in pm.instruments:
            for n in inst.notes:
                notes.append((n.pitch, round(n.start,3), round(n.end-n.start,3), n.velocity))
    except Exception:
        import mido
        mf = mido.MidiFile(midp)
        t = 0.0
        on = {}
        for msg in mf:
            t += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                on[msg.note] = (t, msg.velocity)
            elif msg.type in ("note_off",) or (msg.type=="note_on" and msg.velocity==0):
                if msg.note in on:
                    st, vel = on.pop(msg.note)
                    notes.append((msg.note, round(st,3), round(t-st,3), vel))
    notes.sort(key=lambda x: x[1])
    pitches = sorted(set(n[0] for n in notes))
    def nm(p):
        names=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
        return f"{names[p%12]}{p//12-1}"
    print(f"  total notes={len(notes)}  distinct pitches={len(pitches)}")
    print(f"  pitches: {[f'{p}({nm(p)})' for p in pitches]}")
    print(f"  first 24 notes (pitch,start,len,vel):")
    for n in notes[:24]:
        print(f"    {n[0]}({nm(n[0])}) t={n[1]} len={n[2]} v={n[3]}")
    # staircase test: is the pitch sequence monotonic non-decreasing?
    seq = [n[0] for n in notes]
    mono = all(seq[i] <= seq[i+1] for i in range(len(seq)-1))
    print(f"  STAIRCASE? (monotonic ascending) = {mono}")
except Exception as e:
    print(f"  midi read FAIL: {e}")
