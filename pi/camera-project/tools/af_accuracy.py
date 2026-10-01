"""Does continuous AF (PDAF) land on the true sharpness peak? Let AF settle
from a wrong start, then sweep the lens manually around the result and score
the centre third of the frame (Laplacian variance; the AF's default metering
is centre-weighted).

    python3 tools/af_accuracy.py W H [--oneshot]

--oneshot uses a triggered single AF cycle (AfMode Auto) instead of continuous,
with the AF window set to the centre third (what tap-to-focus does).
"""
import os, sys, time
import cv2, numpy as np
from picamera2 import Picamera2

W, H = int(sys.argv[1]), int(sys.argv[2])
p = Picamera2(0, tuning=Picamera2.load_tuning_file(os.path.abspath("tuning_64mp_sharper.json")))
p.configure(p.create_video_configuration(main={"size": (W, H), "format": "RGB888"},
                                         raw={"size": (W, H)}, buffer_count=4))
p.start()
p.set_controls({"AfMode": 0, "LensPosition": 1.0}); time.sleep(1.0)
if "--oneshot" in sys.argv:
    from libcamera import controls
    x, y, w, h = p.camera_controls["ScalerCrop"][1]
    p.set_controls({"AfMetering": controls.AfMeteringEnum.Windows,
                    "AfWindows": [(x + w // 3, y + h // 3, w // 3, h // 3)]})
    time.sleep(0.3)
    t0 = time.time()
    p.set_controls({"AfMode": controls.AfModeEnum.Auto, "AfTrigger": controls.AfTriggerEnum.Start})
    while True:
        md = p.capture_metadata()
        if time.time() - t0 > 0.3 and md.get("AfState") in (2, 3):  # Focused / Failed
            break
        if time.time() - t0 > 8:
            break
    print(f"one-shot AF: state {md.get('AfState')} after {time.time() - t0:.2f}s")
else:
    p.set_controls({"AfMode": 2}); time.sleep(4.0)
af_lens = p.capture_metadata()["LensPosition"]

def score():
    g = cv2.cvtColor(p.capture_array("main"), cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    return cv2.Laplacian(g[h // 3: 2 * h // 3, w // 3: 2 * w // 3], cv2.CV_64F).var()

p.set_controls({"AfMode": 0})
results = []
for lp in np.round(np.arange(af_lens - 0.4, af_lens + 0.41, 0.05), 3):
    p.set_controls({"LensPosition": float(lp)}); time.sleep(0.35)
    s = np.mean([score() for _ in range(3)])
    results.append((lp, s))
p.stop(); p.close()
best = max(results, key=lambda r: r[1])
af_score = min(results, key=lambda r: abs(r[0] - af_lens))[1]
print(f"{W}x{H}: AF settled at {af_lens:.2f}; sweep peak at {best[0]:.2f} (score {best[1]:.0f}); "
      f"AF position scores {af_score:.0f} = {100 * af_score / best[1]:.0f}% of peak")
print("  " + "  ".join(f"{lp:.2f}:{s:.0f}" for lp, s in results))
