"""Trace continuous AF in a given sensor mode, from a deliberately wrong start
position, with libcamera's AF debug log (phase/confidence per frame).

    LIBCAMERA_LOG_LEVELS=RPiAf:0 python3 tools/af_trace.py W H [START_LENS] [SECS]
"""
import os, sys, time
from picamera2 import Picamera2

W, H = int(sys.argv[1]), int(sys.argv[2])
start = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
secs = float(sys.argv[4]) if len(sys.argv) > 4 else 4.0
p = Picamera2(0, tuning=Picamera2.load_tuning_file(os.path.abspath("tuning_64mp_sharper.json")))
p.configure(p.create_video_configuration(main={"size": (min(W, 1920), min(H, 1104)), "format": "RGB888"},
                                         raw={"size": (W, H)}, buffer_count=4))
p.start()
p.set_controls({"AfMode": 0, "LensPosition": start}); time.sleep(1.0)
t0 = time.time()
p.set_controls({"AfMode": 2})
last = None
while time.time() - t0 < secs:
    md = p.capture_metadata()
    cur = (round(md.get("LensPosition", -1), 2), md.get("AfState"))
    if cur != last:
        print(f"t={time.time() - t0:5.2f}s lens={cur[0]} afstate={cur[1]}", file=sys.stderr)
        last = cur
p.stop(); p.close()
