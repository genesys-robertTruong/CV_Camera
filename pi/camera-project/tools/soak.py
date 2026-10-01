"""Soak test of the 1:1 crop mode at the current crop_llp/crop_min_vblank:
check every captured frame for dropped (zero) rows and for corruption
(correlation vs the first frame; scene and exposure are static)."""
import os, sys, time, numpy as np
from picamera2 import Picamera2

SECS = float(sys.argv[1]) if len(sys.argv) > 1 else 30
p = Picamera2(0, tuning=Picamera2.load_tuning_file(os.path.abspath("tuning_64mp_sharper.json")))
p.configure(p.create_video_configuration(main={"size": (1280, 768), "format": "RGB888"},
                                         raw={"size": (1280, 768), "format": "SRGGB16"}, buffer_count=6,
                                         controls={"FrameDurationLimits": (1000, 1000000)}))
p.start()
p.set_controls({"AfMode": 0, "LensPosition": 6.65, "AeEnable": False, "ExposureTime": 5000, "AnalogueGain": 6.0})
time.sleep(1.5)
ref = None; n = checked = bad_rows = bad_corr = 0; worst = 1.0; ts = []
t0 = time.time()
while time.time() - t0 < SECS:
    req = p.capture_request(); n += 1
    ts.append(req.get_metadata()["SensorTimestamp"])
    if n % 2 == 0:  # check every other frame to keep up
        r = req.make_array("raw").view(np.uint16)[:, :1280].astype(np.float32)
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
print(f"{n} frames in {SECS:.0f}s = {n/SECS:.1f} fps; interval median {np.median(dt):.2f}ms max {dt.max():.2f}ms "
      f"(gaps >1.5x: {int((dt > 1.5*np.median(dt)).sum())}); checked {checked}: "
      f"frames with zero rows {bad_rows}, corrupt (corr<0.6) {bad_corr}, worst corr {worst:.3f}")
