"""One crop_llp trial: start camera in the 1280x768 1:1 mode, measure fps and
check the raw still matches the known-good reference (work/raw1280.npy)."""
import os, sys, time, numpy as np
from picamera2 import Picamera2

ref = np.load("work/raw1280.npy").astype(np.float64)
def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).mean() / (a.std() * b.std() + 1e-9))

p = Picamera2(0, tuning=Picamera2.load_tuning_file(os.path.abspath("tuning_64mp_sharper.json")))
p.configure(p.create_video_configuration(main={"size": (1280, 768), "format": "RGB888"},
                                         raw={"size": (1280, 768), "format": "SRGGB16"}, buffer_count=4,
                                         controls={"FrameDurationLimits": (1000, 1000000)}))
p.start(); p.set_controls({"AfMode": 0, "LensPosition": 6.65, "AeEnable": False, "ExposureTime": 5000, "AnalogueGain": 6.0}); time.sleep(1.5)
ts = []
last = None
t0 = time.time()
while time.time() - t0 < 3:
    req = p.capture_request(); ts.append(req.get_metadata()["SensorTimestamp"])
    if last is not None:
        last.release()
    last = req
import cv2; cv2.imwrite(f"work/llp_{sys.argv[1]}.png", last.make_array("main")[24:744])
raw = last.make_array("raw").view(np.uint16)[:, :1280].astype(np.float64); last.release()
md = p.capture_metadata(); p.stop(); p.close()
dt = np.diff(ts) / 1e6
print(f"fps {len(ts)/3:.1f}  frame-interval ms {np.median(dt):.2f} (max {dt.max():.1f})  "
      f"FrameDuration {md['FrameDuration']/1000:.2f}ms  mean {raw.mean():.0f}  corr-vs-ref {corr(raw, ref):.4f}  "
      f"zero-rows {int((raw.max(axis=1) == 0).sum())}")
