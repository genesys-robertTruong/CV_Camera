#!/usr/bin/env python3
"""Does the lens (VCM) still respond after a runtime reconfigure? Moves the
lens, reconfigures (stop/configure/start, or a full close/reopen with
REOPEN=1), and moves it again, printing metadata LensPosition each time.
PRESET=1 sets AfMode/LensPosition before start() (as mjpeg_crop.py does),
NUDGE=1 runs continuous AF for 0.3s after start, CTL=ScalerCrop,Sharpness sets those
before start().

    python3 tools/reconfig_lens_test.py [MODE]   (default 3k)
"""
import os
import sys
import time

from picamera2 import Picamera2

MODES = {"720": ((1280, 768), (1280, 720)), "1080": ((1920, 1104), (1920, 1080)),
         "3k": ((3072, 1728), (3072, 1728)), "4k": ((3840, 2208), (3840, 2160))}
mode = sys.argv[1] if len(sys.argv) > 1 else "3k"
raw, out = MODES[mode]
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tuning = Picamera2.load_tuning_file(os.path.join(here, "tuning_64mp_sharper.json"))


def config(picam2, kind):
    main = {"size": out, "format": "RGB888"} if kind == "mjpeg" else {"size": (1920, 1080), "format": "RGB888"}
    extra = {} if kind == "mjpeg" else {"lores": {"size": (1920, 1080), "format": "YUV420"}}
    return picam2.create_video_configuration(main=main, **extra, raw={"size": raw}, buffer_count=4,
                                             controls={"FrameDurationLimits": (40000, 40000)})


def lens_at(picam2, pos):
    picam2.set_controls({"AfMode": 0, "LensPosition": pos})
    time.sleep(1.0)
    return picam2.capture_metadata().get("LensPosition")


picam2 = Picamera2(camera_num=0, tuning=tuning)
picam2.configure(config(picam2, "mjpeg"))
picam2.start()
print("initial:", lens_at(picam2, 3.0), lens_at(picam2, 9.0))
for kind in ("h264", "mjpeg", "h264", "mjpeg"):
    t0 = time.monotonic()
    if os.environ.get("REOPEN"):
        picam2.stop()
        picam2.close()
        picam2 = Picamera2(camera_num=0, tuning=tuning)
    else:
        picam2.stop()
    picam2.configure(config(picam2, kind))
    if os.environ.get("PRESET"):  # as mjpeg_crop.py restores focus: set before start()
        picam2.set_controls({"AfMode": 0, "LensPosition": 9.0})
    if os.environ.get("CTL"):  # the other controls mjpeg_crop.py restores before start()
        x, y, w, h = picam2.camera_controls["ScalerCrop"][1]
        ctl = {"ScalerCrop": (x + (w - out[0]) // 2, y + (h - out[1]) // 2, *out), "Sharpness": 1.0}
        picam2.set_controls({k: ctl[k] for k in os.environ["CTL"].split(",")})
    picam2.start()
    if os.environ.get("NUDGE"):  # brief continuous AF, then back to manual
        picam2.set_controls({"AfMode": 2})
        time.sleep(0.3)
        picam2.set_controls({"AfMode": 0})
    picam2.capture_metadata()
    print(f"-> {kind} in {time.monotonic() - t0:.2f}s; LensPosition control:",
          picam2.camera_controls.get("LensPosition"), "moves:", lens_at(picam2, 3.0), lens_at(picam2, 9.0))
picam2.stop()
