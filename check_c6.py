import warnings, os, collections
warnings.filterwarnings("ignore")
import numpy as np, soundfile as sf
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH

STEM = r"D:\flmidi\out\abraham\stems\other.wav"
NOTE = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
_, _, notes = predict(STEM, ICASSP_2022_MODEL_PATH, onset_threshold=0.6, frame_threshold=0.4,
                      minimum_note_length=120, minimum_frequency=110, maximum_frequency=1300)
cnt = collections.Counter(int(n[2]) for n in notes)
amp = collections.defaultdict(list)
for n in notes:
    amp[int(n[2])].append(float(n[3]))
print("pitch  FLname  count  avgAmp")
for p in sorted(cnt):
    nm = NOTE[p % 12] + str(p // 12)        # FL octave (60=C5)
    mark = "   <== C#6 (the bugged one)" if p == 73 else ""
    print("  %3d  %-4s  %3d   %.2f%s" % (p, nm, cnt[p], sum(amp[p]) / len(amp[p]), mark))
sl = r"D:\flmidi\out\abraham\slices\melody_sus\p073.wav"
if os.path.exists(sl):
    y, sr = sf.read(sl)
    if y.ndim > 1:
        y = y[:, 0]
    print("\np073 slice: %.2fs  rms=%.4f  peak=%.3f" % (len(y) / sr, float(np.sqrt(np.mean(y ** 2))), float(np.max(np.abs(y)))))
