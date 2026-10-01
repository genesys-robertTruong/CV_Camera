"""One line-length trial of a 1:1 crop mode: start the camera in that mode
(at whatever llp_* the driver param currently holds), measure fps and check the
raw still matches a known-good reference of the same mode.

    python3 tools/llp_sweep_one.py W H LABEL [--ref]

--ref saves this run's raw as the reference (work/ref_WxH.npy) -- do that at
the full mode's safe line length (46770) first. Each run also saves the ISP
output as work/llp_WxH_LABEL.png. Exposure/gain are fixed so auto-exposure
can't stretch frames or change the image between runs.
"""
import os, sys, time, numpy as np
from picamera2 import Picamera2

W, H, LABEL = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
MAKE_REF = "--ref" in sys.argv
REF = f"work/ref_{W}x{H}.npy"

def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).mean() / (a.std() * b.std() + 1e-9))

p = Picamera2(0, tuning=Picamera2.load_tuning_file(os.path.abspath("tuning_64mp_sharper.json")))
p.configure(p.create_video_configuration(main={"size": (W, H), "format": "RGB888"},
                                         raw={"size": (W, H), "format": "SRGGB16"}, buffer_count=4,
                                         controls={"FrameDurationLimits": (1000, 1000000)}))
p.start()
p.set_controls({"AfMode": 0, "LensPosition": 6.65, "AeEnable": False, "ExposureTime": 5000, "AnalogueGain": 6.0})
time.sleep(1.5)
ts = []
last = None
t0 = time.time()
while time.time() - t0 < 3:
    req = p.capture_request(); ts.append(req.get_metadata()["SensorTimestamp"])
    if last is not None:
        last.release()
    last = req
import cv2
cv2.imwrite(f"work/llp_{W}x{H}_{LABEL}.png", last.make_array("main"))
raw = last.make_array("raw").view(np.uint16)[:, :W].astype(np.float32); last.release()
md = p.capture_metadata(); p.stop(); p.close()
if MAKE_REF:
    np.save(REF, raw)
c = corr(raw, np.load(REF).astype(np.float32)) if os.path.exists(REF) else float("nan")
dt = np.diff(ts) / 1e6
print(f"fps {len(ts)/3:.1f}  frame-interval ms {np.median(dt):.2f} (max {dt.max():.1f})  "
      f"FrameDuration {md['FrameDuration']/1000:.2f}ms  mean {raw.mean():.0f}  corr-vs-ref {c:.4f}  "
      f"zero-rows {int((raw.max(axis=1) == 0).sum())}")
