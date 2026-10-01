"""Soak test of a 1:1 crop mode at the driver's current llp_*/crop_min_vblank:
check captured frames for dropped (zero) rows and for corruption (correlation
vs the first frame; scene and exposure are static).

    python3 tools/soak.py [SECS] [W H]      (default 30 s, 1280x768)
"""
import os, sys, time, numpy as np
from picamera2 import Picamera2

SECS = float(sys.argv[1]) if len(sys.argv) > 1 else 30
W, H = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (1280, 768)
CHECK_EVERY = 2 if W * H <= 1920 * 1104 else 4  # keep up with the frame rate
p = Picamera2(0, tuning=Picamera2.load_tuning_file(os.path.abspath("tuning_64mp_sharper.json")))
p.configure(p.create_video_configuration(main={"size": (W, H), "format": "RGB888"},
                                         raw={"size": (W, H), "format": "SRGGB16"}, buffer_count=6 if W <= 1920 else 4,
                                         controls={"FrameDurationLimits": (1000, 1000000)}))
p.start()
p.set_controls({"AfMode": 0, "LensPosition": 6.65, "AeEnable": False, "ExposureTime": 5000, "AnalogueGain": 6.0})
time.sleep(1.5)
ref = None; n = checked = bad_rows = bad_corr = 0; worst = 1.0; ts = []
t0 = time.time()
while time.time() - t0 < SECS:
    req = p.capture_request(); n += 1
    ts.append(req.get_metadata()["SensorTimestamp"])
    if n % CHECK_EVERY == 0:
        r = req.make_array("raw").view(np.uint16)[:, :W].astype(np.float32)
        if ref is None:
            ref = r - r.mean(); ref_sd = ref.std()
        else:
            if (r.max(axis=1) == 0).any(): bad_rows += 1
            d = r - r.mean(); c = float((d * ref).mean() / (d.std() * ref_sd + 1e-9))
            worst = min(worst, c)
            if c < 0.6: bad_corr += 1
        checked += 1
    req.release()
p.stop(); p.close()
dt = np.diff(ts) / 1e6
print(f"{W}x{H}: {n} frames in {SECS:.0f}s = {n/SECS:.1f} fps; interval median {np.median(dt):.2f}ms max {dt.max():.2f}ms "
      f"(gaps >1.5x: {int((dt > 1.5*np.median(dt)).sum())}); checked {checked}: "
      f"frames with zero rows {bad_rows}, corrupt (corr<0.6) {bad_corr}, worst corr {worst:.3f}")
