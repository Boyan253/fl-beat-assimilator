"""Claude-ears: drive FL playback via the bridge, capture the output loopback,
and analyze what's actually sounding (signal level, pitches, density, dissonance)."""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
os.environ["FLMCP_TRANSPORT"] = "file"
sys.path.insert(0, r"C:\Users\User\Desktop\FLStudioMCP\src")
import numpy as np, soundcard as sc, librosa
from fl_studio_mcp.bridge_client import get_client

NOTE = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
SR = 44100

# pick the loopback of whatever device is the default render (where FL plays now)
spk = sc.default_speaker()
mic = sc.get_microphone(spk.name, include_loopback=True)
print("listening on:", spk.name)

c = get_client()
try:
    c.call("transport.start")
except Exception as e:
    print("bridge start note:", e)
time.sleep(0.3)
with mic.recorder(samplerate=SR, channels=2, blocksize=2048) as r:
    d = r.record(numframes=int(SR * 6))
try:
    c.call("transport.stop")
except Exception:
    pass

y = d.mean(axis=1).astype("float32")
rms = float(np.sqrt(np.mean(y ** 2)))
peak = float(np.max(np.abs(y)))
print("signal: rms=%.4f peak=%.3f  (%s)" % (rms, peak, "SILENT" if peak < 0.002 else "audio present"))
if peak < 0.002:
    print(">> nothing captured — FL may be routed elsewhere or not playing this pattern")
    sys.exit(0)

# pitches present (chroma) + density + brightness
chroma = librosa.feature.chroma_cqt(y=y, sr=SR).mean(axis=1)
order = np.argsort(chroma)[::-1]
print("strongest pitch-classes:", [NOTE[i] for i in order[:5]])
cen = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=SR)))
oenv = librosa.onset.onset_strength(y=y, sr=SR)
onsets = librosa.onset.onset_detect(onset_envelope=oenv, sr=SR)
print("brightness(centroid): %d Hz" % round(cen))
print("event density: %.1f hits/sec" % (len(onsets) / 6.0))
# rough dissonance: how spread/flat the chroma is (flat = many clashing pitches)
flat = float(np.exp(np.mean(np.log(chroma + 1e-9))) / (np.mean(chroma) + 1e-9))
print("harmonic focus: %.2f  (1.0 = muddy/all-pitches-at-once, 0.2 = clear tonal)" % flat)
