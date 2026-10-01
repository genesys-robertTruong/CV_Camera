#!/usr/bin/env python3
"""Is switching the ISP outputs at runtime (same raw sensor mode) quick and
reliable? Alternates main 3072x1728 RGB888 (MJPEG) and main 1920x1080
YUV420 (H.264) with stop/configure/start, N times, and checks frames flow
after each switch.

    python3 tools/reconfig_test.py [MODE] [N]   (default 3k 10)
"""
import os
import sys
import time

from picamera2 import Picamera2

MODES = {"720": ((1280, 768), (1280, 720)), "1080": ((1920, 1104), (1920, 1080)),
         "3k": ((3072, 1728), (3072, 1728)), "4k": ((3840, 2208), (3840, 2160))}
mode = sys.argv[1] if len(sys.argv) > 1 else "3k"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
raw, out = MODES[mode]
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
picam2 = Picamera2(camera_num=0, tuning=Picamera2.load_tuning_file(os.path.join(here, "tuning_64mp_sharper.json")))
cfgs = {
    "mjpeg": picam2.create_video_configuration(main={"size": out, "format": "RGB888"}, raw={"size": raw},
                                              buffer_count=4, controls={"FrameDurationLimits": (40000, 40000)}),
    "h264": picam2.create_video_configuration(main={"size": (min(1920, out[0]), min(1080, out[1])), "format": "YUV420"},
                                             raw={"size": raw}, buffer_count=6,
                                             controls={"FrameDurationLimits": (40000, 40000)}),
}
picam2.configure(cfgs["mjpeg"])
picam2.start()
times = []
for i in range(n):
    kind = "h264" if i % 2 == 0 else "mjpeg"
    t0 = time.monotonic()
    picam2.stop()
    picam2.configure(cfgs[kind])
    picam2.start()
    req = picam2.capture_request()
    shape = req.make_array("main").shape
    req.release()
    t1 = time.monotonic()
    t2 = time.monotonic()
    for _ in range(25):
        picam2.capture_request().release()
    fps = 25 / (time.monotonic() - t2)
    times.append(t1 - t0)
    print(f"{i}: -> {kind} {shape}, first frame after {t1 - t0:.2f}s, then {fps:.1f} fps")
picam2.stop()
print(f"switch time min {min(times):.2f}s max {max(times):.2f}s")
