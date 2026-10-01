#!/usr/bin/env python3
"""Hardware H.264 feasibility: a sensor crop mode with main at full size (as
mjpeg_crop.py needs) plus an ISP-scaled 1080p YUV420 lores stream into the
VC4 hardware encoder, while a capture loop pulls main frames like the server.

    python3 tools/h264_test.py MODE FPS [SECS] [BITRATE]   e.g. 3k 25 10 10000000

Env: NOCOPY=1 skips the main copy, ZOOM=N applies an N x ScalerCrop zoom,
BUFS=N sets the buffer count (default 4), MAIN=WxH overrides the main size,
SINGLE=1 uses only a YUV420 main at the H.264 size (as mjpeg_crop.py does),
FMT=RGB888 etc. changes that main's format, SHARP=N sets Sharpness.

Reports encoded fps, bitrate, keyframes and process CPU; writes the stream
to work/h264_MODE.h264 for inspection (ffprobe).
"""
import os
import sys
import threading
import time

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import Output

MODES = {
    "720": ((1280, 768), (1280, 720)),
    "1080": ((1920, 1104), (1920, 1080)),
    "3k": ((3072, 1728), (3072, 1728)),
    "4k": ((3840, 2208), (3840, 2160)),
}
mode, fps = sys.argv[1], float(sys.argv[2])
secs = float(sys.argv[3]) if len(sys.argv) > 3 else 10
bitrate = int(sys.argv[4]) if len(sys.argv) > 4 else 10_000_000
raw, out = MODES[mode]
if os.environ.get("MAIN"):  # e.g. MAIN=1920x1080: main no bigger than lores
    out = tuple(int(v) for v in os.environ["MAIN"].split("x"))
lores = (min(1920, out[0]), min(1080, out[1]))
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(here, "work"), exist_ok=True)
path = os.path.join(here, "work", f"h264_{mode}.h264")


class Counter(Output):
    def __init__(self, f):
        super().__init__()
        self.f, self.frames, self.bytes, self.keys, self.ts = f, 0, 0, 0, []

    def outputframe(self, frame, keyframe=True, timestamp=None, packet=None, audio=False):
        self.f.write(frame)
        self.frames += 1
        self.bytes += len(frame)
        self.keys += bool(keyframe)
        self.ts.append(time.monotonic())


picam2 = Picamera2(camera_num=0, tuning=Picamera2.load_tuning_file(os.path.join(here, "tuning_64mp_sharper.json")))
single = bool(os.environ.get("SINGLE"))  # mjpeg_crop.py's H.264 configuration: one YUV420 main, no lores
streams = ({"main": {"size": lores, "format": os.environ.get("FMT", "YUV420")}} if single else
           {"main": {"size": out, "format": "RGB888"}, "lores": {"size": lores, "format": "YUV420"}})
picam2.configure(picam2.create_video_configuration(
    **streams,
    raw={"size": raw}, buffer_count=int(os.environ.get("BUFS", 4)),
    controls={"FrameDurationLimits": (int(1e6 / fps),) * 2}))
picam2.start()
x, y, w, h = picam2.camera_controls["ScalerCrop"][1]
zoom = float(os.environ.get("ZOOM", 1))  # >1: ISP upscales a centred sub-crop, as the server's zoom does
if os.environ.get("SHARP"):
    picam2.set_controls({"Sharpness": float(os.environ["SHARP"])})
cw, ch = int(out[0] / zoom) & ~1, int(out[1] / zoom) & ~1
picam2.set_controls({"ScalerCrop": (x + (w - cw) // 2, y + (h - ch) // 2, cw, ch)})

stop = False
captured = [0]


def capture():
    while not stop:
        req = picam2.capture_request()
        if os.environ.get("NOCOPY"):  # as the server does with no MJPEG viewer
            req.release()
            captured[0] += 1
            continue
        try:
            req.make_array("main")
        finally:
            req.release()
        captured[0] += 1


threading.Thread(target=capture, daemon=True).start()
time.sleep(2)
f = open(path, "wb")
cnt = Counter(f)
enc = H264Encoder(bitrate=bitrate, iperiod=int(fps), framerate=fps)
cpu0, t0 = os.times(), time.monotonic()
cap0 = captured[0]
picam2.start_encoder(enc, cnt, name="main" if single else "lores")
time.sleep(secs)
picam2.stop_encoder()
cpu1, t1 = os.times(), time.monotonic()
stop = True
picam2.stop()
f.close()
el = t1 - t0
gaps = [b - a for a, b in zip(cnt.ts, cnt.ts[1:])]
cpu = (cpu1.user + cpu1.system - cpu0.user - cpu0.system) / el * 100
print(f"{mode} {out[0]}x{out[1]} main + {lores[0]}x{lores[1]} lores -> H.264 @ {fps:g}fps, {bitrate / 1e6:g} Mbit/s")
print(f"  encoded {cnt.frames / el:.1f} fps ({cnt.frames} frames, {cnt.keys} keyframes), "
      f"{cnt.bytes * 8 / el / 1e6:.1f} Mbit/s, main captured {(captured[0] - cap0) / el:.1f} fps")
if gaps:
    print(f"  frame gap max {max(gaps) * 1000:.0f}ms, process CPU {cpu:.0f}%  -> {path}")
